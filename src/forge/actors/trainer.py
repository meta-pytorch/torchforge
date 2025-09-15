# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
from monarch.actor import current_rank, current_size, endpoint
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Float8,
    LRScheduler,
    Model,
    Optimizer,
    Parallelism,
    Training,
)

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.controller import ForgeActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RLTrainer(ForgeActor):
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
    float8: Float8 = field(default_factory=Float8)
    comm: Comm = field(default_factory=Comm)

    def __post_init__(self):
        """Initializes config types and env variables.

        torchrun normally hands env variables, but we need to do it ourselves
        in monarch for now.

        """
        # Instantiate dict fields
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.current_step = 0
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        self.rank = current_rank().rank
        self.size = math.prod(current_size().values())

        env = {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.rank),
            "LOCAL_WORLD_SIZE": str(self.size),
            "GROUP_RANK": str(self.size),
            "GROUP_WORLD_SIZE": str(self.size),
            "ROLE_RANK": str(self.rank),
            "ROLE_WORLD_SIZE": str(self.size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self.size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)

    @endpoint
    async def setup(self):
        # TODO: update ForgeEngine to not use ForgeJobConfig
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.current_step)
        self.engine.optimizers.zero_grad()

    def forward_backward(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["tokens"]

        if getattr(self.engine.model_args, "use_flex_attn", False):
            cp_mesh = (
                parallel_dims.world_mesh["cp"] if parallel_dims.cp_enabled else None
            )
            init_attention_mask(
                inputs, self.engine.tokenizer.base_tokenizer.eos_id, cp_mesh
            )

        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
            # TODO implement PP
            # # Pipeline Parallel forward / backward inside step() call
            # with self.train_context(optional_context_parallel_ctx):
            #     targets, losses = (
            #         (labels, []) if self.pp_has_last_stage else (None, None)
            #     )
            #     if self.pp_has_first_stage:
            #         self.pp_schedule.step(
            #             inputs, target=targets, losses=losses, input_batch=inputs
            #         )
            #     else:
            #         self.pp_schedule.step(
            #             target=targets, losses=losses, input_batch=inputs
            #         )
            #
            # # accumulate losses across pipeline microbatches
            # # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            # loss = (
            #     torch.mean(torch.stack(losses)).to(self.device)
            #     if self.pp_has_last_stage
            #     else torch.tensor([-1.0], device=self.device)
            # )
        else:
            # Non-PP forward / backward
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.engine.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    @endpoint
    def train_step(self, batch) -> None:
        # Move tensors to the appropriate device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")  # TODO: hardcoded for now

        # TODO implement gradient accumulation
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        # TODO: convert to GRPO Loss
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()

        self.current_step += 1
        self.engine.checkpointer.save(
            curr_step=self.current_step,
            last_step=self.current_step == self.num_training_steps,
        )

    # TODO: integrate the grpo app step with the above step
    # def train_step(self, self, batch: list(Episode)):
    #     total_loss = 0.0
    #     num_groups_processed = 0
    #
    #     for episode in batch:
    #         groups = episode.groups
    #
    #         # Collect all response texts and corresponding data
    #         response_texts = []
    #         ref_logprobs_list = []
    #         advantages_list = []
    #
    #         for group in groups:
    #             response_texts.append(group.response)
    #             ref_logprobs_list.append(group.ref_logprobs)
    #             advantages_list.append(group.advantage)
    #
    #         # Tokenize all responses in batch
    #         tokenized = self.tokenizer(
    #             response_texts,
    #             padding=True,
    #             truncation=True,
    #             return_tensors="pt",
    #             max_length=512,  # Adjust based on your needs
    #         )
    #
    #         input_ids = tokenized["input_ids"].to(self.device)
    #         attention_mask = tokenized["attention_mask"].to(self.device)
    #
    #         # Compute current policy log probabilities using the model
    #         current_logprobs = compute_sequence_logprobs(
    #             self.model, input_ids, attention_mask, requires_grad=True
    #         )
    #
    #         # Convert ref_logprobs and advantages to tensors
    #         ref_logprobs_tensor = torch.stack(ref_logprobs_list).to(self.device)
    #         advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32).to(
    #             self.device
    #         )
    #
    #         # Compute GRPO loss components
    #         # Ratio between current policy and reference policy
    #         ratio = torch.exp(current_logprobs - ref_logprobs_tensor)
    #
    #         # Policy gradient loss weighted by advantages
    #         pg_loss = -torch.mean(ratio * advantages_tensor)
    #
    #         # KL penalty to prevent policy from deviating too far from reference
    #         kl_penalty = self.beta * torch.mean(
    #             (current_logprobs - ref_logprobs_tensor) ** 2
    #         )
    #
    #         # Total GRPO loss
    #         loss = pg_loss + kl_penalty
    #         total_loss += loss.item()
    #         num_groups_processed += len(groups)
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #
    #         # Gradient clipping (optional but recommended for stability)
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #
    #         self.optimizer.step()
    #
    #     avg_loss = total_loss / len(batch) if batch else 0.0
    #
    #     return {"loss": avg_loss, "groups_processed": num_groups_processed}

    @endpoint
    def push_weights(self) -> None:
        pass

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()


def _qwen3_hf_to_vllm(
    sd: dict[str, torch.Tensor], num_layers: int
) -> dict[str, torch.Tensor]:
    """Convert transformers state dict to vLLM format. Specifically, this fuses
    QKV projection and MLP gate_up_proj layers.

    Args:
        sd (dict): State dict from HF model.
        num_layers (int): Number of layers in the model.

    Returns:
        dict: State dict in vLLM format.
    """
    load_sd = {}

    # Copy over directly mapped keys
    for k in sd:
        if any(
            x in k
            for x in [
                "down_proj",
                "input_layernorm",
                "post_attention_layernorm",
                "o_proj",
                "norm.weight",
                "embed_tokens.weight",
                "lm_head.weight",
            ]
        ):
            load_sd[k] = sd[k]

    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        # QKV fusion
        q = sd[prefix + "self_attn.q_proj.weight"]
        k = sd[prefix + "self_attn.k_proj.weight"]
        v = sd[prefix + "self_attn.v_proj.weight"]
        load_sd[prefix + "self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)
        # MLP gate_up_proj fusion
        gate = sd[prefix + "mlp.gate_proj.weight"]
        up = sd[prefix + "mlp.up_proj.weight"]
        load_sd[prefix + "mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)

    return load_sd
