# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torch.distributed.checkpoint as dcp
import torchstore as ts

from forge.actors._torchstore_utils import (
    DcpHandle,
    get_dcp_whole_state_dict_key,
    get_param_key,
    rdma_available,
)

from forge.controller import ForgeActor
from forge.data.utils import batch_to_device
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

from monarch.actor import endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Job,
    LRScheduler,
    MemoryEstimation,
    Model,
    Optimizer,
    Parallelism,
    Quantize,
    Training,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class TitanTrainer(ForgeActor):
    """A generic trainer actor implementation built on top of TorchTitan.

    Built on top of TorchTitan's training engine, this actor provides a complete training
    loop for reinforcement learning. It performs forward and backward passes with gradient
    computation, optimization steps, and checkpoint management. Unlike the ReferenceModel
    actor which only runs forward passes, RLTrainer actively updates the policy model
    parameters through gradient descent.

    The trainer supports the same distributed training strategies that TorchTitan does,
    including but not limited to, tensor parallelism, data parallelism, and FSDP
    (Fully Sharded Data Parallel). It is typically used in conjunction with ReferenceModel
    for policy optimization algorithms like GRPO (Group Relative Policy Optimization),
    where it optimizes the policy against a loss that includes KL divergence penalties
    from the reference model.

    The trainer handles:
    - Forward and backward propagation with automatic mixed precision (AMP)
    - Optimizer steps with learning rate scheduling

    Provides reusable components for SFT and RL workloads:
    - rank_should_record_loss: Only last PP stage records loss
    - record_batch_metrics(): Generic batch metric recording
    - setup_metric_logger(): Metric logger initialization
    - forward_backward(): Forward/backward with PP+CP support
    - train_step_sft(): SFT-style training step
    """

    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    # Non JobConfig-related fields
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"
    use_dcp: bool = not rdma_available()
    dcp_path: str = "forge_dcp_tmp"

    def __post_init__(self):
        super().__init__()
        if self.use_dcp:
            torch.serialization.set_crc32_options(False)

        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 1  # fragile contract.
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Compiling loss")
        self.loss = torch.compile(self.loss)
        # ADD THIS LINE: Initialize ForgeEngine here (during __init__) to ensure
        # init_distributed is called before Monarch initializes its process group
        self._setup_engine()

    @endpoint
    async def setup(self):
        # Already initialized in post_init__
        pass

    def _setup_engine(self):
        """Initialize the ForgeEngine (non-endpoint helper for subclasses)."""
        # TODO: update ForgeEngine to not use ForgeJobConfig
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in {
            "loss",
            "state_dict_key",
            "use_dcp",
            "dcp_path",
        }:
            engine_config.pop(key)  # Not part of job config
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))

        # all ranks should record loss, except when PP=True. Then, only the last stage should record loss.
        self.rank_should_record_loss = True
        if (
            hasattr(self.engine, "pp_has_last_stage")
            and not self.engine.pp_has_last_stage
        ):
            self.rank_should_record_loss = False

        self.engine.checkpointer.load(step=self.step)
        self.engine.optimizers.zero_grad()

    async def setup_metric_logger(self):
        """Initialization happens in the main process. Here we just retrieve it."""
        mlogger = await get_or_create_metric_logger()
        return mlogger

    def record_batch_metrics(self, data_metrics: list) -> None:
        """Record metrics from batch.

        Since the dataloader creates new processes, we don't call `record_metric` in the dataset.
        Instead, pop the metrics from the batch and record them here.
        """
        for metric in data_metrics:
            record_metric(metric.key, metric.value, metric.reduction)

    def forward_backward(
        self,
        input_dict: dict[str, Tensor],
        labels: Tensor,
        skip_backward: bool = False,
    ) -> Tensor:
        """Forward and backward pass with PP and CP support.

        Args:
            input_dict: Dict containing input tensors (e.g., {"tokens": tensor})
            labels: Target labels tensor
            skip_backward: If True, skip backward pass (useful for eval)

        Returns:
            Loss tensor
        """
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["tokens"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.engine.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.engine.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.engine.pp_has_last_stage else (None, None)
                )
                if self.engine.pp_has_first_stage:
                    self.engine.pp_schedule.step(inputs, target=targets, losses=losses)
                else:
                    self.engine.pp_schedule.step(target=targets, losses=losses)

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.sum(torch.stack(losses)).to(self.engine.device)
                if self.engine.pp_has_last_stage
                else torch.tensor(-1.0, device=self.engine.device)
            )

            # TODO: PP requires gradients enabled and cant deactive with no_grad
            if skip_backward:
                loss = loss.detach()

        else:
            # Non-PP forward / backward
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.engine.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred

                # Only run backward if requested. Useful for eval.
                if not skip_backward:
                    loss.backward()

        return loss

    def forward_backward_rl(
        self, inputs: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> Tensor:
        """Forward and backward for RL (uses custom self.loss instead of engine.loss_fn)."""
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        optional_context_parallel_ctx = None
        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet for RL")
        else:
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    logits = model_parts[0](**inputs)
                    loss = self.loss(logits, **targets)
                del logits  # Free to before bwd to avoid peaking memory
                loss.backward()
        return loss

    def train_step_sft(self, batch: dict[str, Tensor]) -> None:
        """Execute a single SFT training step.

        This method is designed for supervised fine-tuning where the batch
        contains "tokens" and "labels". It performs forward/backward pass,
        and optimizer step.

        Args:
            batch: Batch dict containing "tokens" and "labels"
        """
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)

        if self.rank_should_record_loss:
            loss_val = loss.item()
            record_metric("TitanTrainer/train_step_sft/loss", loss_val, Reduce.MEAN)
            logger.info(
                f"step {self.step} / {self.num_training_steps} | Loss: {loss_val}"
            )

        self.engine.optimizers.step()
        self.engine.lr_schedulers.step()
        self.step += 1

    @endpoint
    async def train_step(
        self, inputs: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
    ) -> float:
        """Training step for RL workloads (endpoint version)."""
        t = Tracer("rl_trainer_perf/step", timer="gpu", track_memory=True)
        t.start()

        self.engine.gc_handler.run(self.step)
        local_inputs = inputs[self.engine.dp_rank]
        local_targets = targets[self.engine.dp_rank]
        batch_to_device(local_inputs, self.engine.device)
        batch_to_device(local_targets, self.engine.device)

        loss = self.forward_backward_rl(local_inputs, local_targets)
        torch.distributed.all_reduce(loss)

        t.step("forward_backward")

        current_lr = self.engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        record_metric("rl_trainer/learning_rate", current_lr, Reduce.MIN)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()
        t.step("optimizer_step")

        # TODO: delete item() to avoid cpu-gpu sync
        loss = loss.detach().item()
        record_metric("rl_trainer/loss", loss, Reduce.MEAN)

        self.step += 1
        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )
        t.step("save_checkpoint")
        t.stop()
        return loss

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to torchstore in HF format."""
        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()
        if "model" not in self.engine.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")

        sd = self.engine.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        if self.engine.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
        hf_state_dict = self.engine.checkpointer.sd_adapter.to_hf(flattened_state_dict)
        if self.use_dcp:
            key = get_dcp_whole_state_dict_key(policy_version)
            dcp_id = f"{self.dcp_path}/{key}"
            storage_writer = torch.distributed.checkpoint.FileSystemWriter(
                dcp_id, single_file_per_rank=False, thread_count=8
            )
            metadata = dcp.save(storage_writer=storage_writer, state_dict=hf_state_dict)
            dcp_handle = DcpHandle(
                checkpoint_id=dcp_id,
                metadata=metadata,
                param_names=hf_state_dict.keys(),
            )
            await ts.put(key, dcp_handle)
        else:
            for name, param in hf_state_dict.items():
                key = get_param_key(policy_version, name)
                await ts.put(key, param)
        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
