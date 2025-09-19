# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torchstore as ts

from monarch.actor import current_rank, current_size, endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchstore.state_dict_utils import DELIM
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
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.controller import ForgeActor
from forge.data.utils import batch_to_device


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
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"

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

        self.current_step = 1  # fragile contract.
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
        for key in {"loss", "state_dict_key"}:
            engine_config.pop(key)  # Not part of job config
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.current_step)
        self.engine.optimizers.zero_grad()

    def forward_backward(
        self, inputs: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> Tensor:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        # if getattr(self.engine.model_args, "use_flex_attn", False):
        #     cp_mesh = (
        #         parallel_dims.world_mesh["cp"] if parallel_dims.cp_enabled else None
        #     )
        #     init_attention_mask(
        #         inputs, self.engine.tokenizer.base_tokenizer.eos_id, cp_mesh
        #     )

        # optional_context_parallel_ctx = (
        #     dist_utils.create_context_parallel_ctx(
        #         cp_mesh=parallel_dims.world_mesh["cp"],
        #         cp_buffers=[inputs, targets] + [m.freqs_cis for m in model_parts],
        #         cp_seq_dims=[1, 1] + [0 for _ in model_parts],
        #         cp_no_restore_buffers={inputs, targets},
        #         cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
        #     )
        #     if parallel_dims.cp_enabled
        #     else None
        # )
        optional_context_parallel_ctx = None

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
                    logits = model_parts[0](**inputs)
                    loss = self.loss(logits, **targets)
                # need to free to before bwd to avoid peaking memory
                del logits
                loss.backward()

        return loss

    @endpoint
    def train_step(
        self, inputs: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
    ) -> float:
        local_inputs = inputs[self.engine.dp_rank]
        local_targets = targets[self.engine.dp_rank]
        batch_to_device(local_inputs, self.engine.device)
        batch_to_device(local_targets, self.engine.device)

        # compute policy logprobs
        # TODO implement gradient accumulation
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        loss = self.forward_backward(local_inputs, local_targets)
        torch.distributed.all_reduce(loss)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()

        self.current_step += 1
        self.engine.checkpointer.save(
            curr_step=self.current_step,
            last_step=self.current_step == self.num_training_steps,
        )

        return loss.item()

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        # Save to torchstore. Hacking in to the Checkpointer's prepped state-dict for now.
        # TODO:
        # 1. Checkpoint invokes state-dict flattening during dcp_save for [MODEL].
        #    May need to replicate the same in this code path.
        # 2. Unify CheckpointManager and TorchStore weights save control path.
        if "model" not in self.engine.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")
        sd = self.engine.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        if self.engine.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
        hf_state_dict = self.engine.checkpointer.sd_adapter.to_hf(flattened_state_dict)
        # TODO: Figure out how to gracefully handle which model to-vLLM conversion is needed
        vllm_ready_hf_sd = _qwen3_hf_to_vllm(sd=hf_state_dict)
        key = f"{self.state_dict_key}{DELIM}{policy_version}"
        start_time = time.time()
        await ts.put_state_dict(state_dict=vllm_ready_hf_sd, key=key)
        end_time = time.time()
        self.logger.debug(
            f"Pushed weights to {key} in {end_time - start_time:.2f} seconds"
        )

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()


def _qwen3_hf_to_vllm(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert transformers state dict to vLLM format. Specifically, this fuses
    QKV projection and MLP gate_up_proj layers.

    Args:
        sd (dict): State dict from HF model.
        num_layers (int): Number of layers in the model.

    Returns:
        dict: State dict in vLLM format.
    """
    load_sd = {}

    # Infer num_layers from the state dict by finding the highest layer index
    layer_indices = []
    pattern = re.compile(r"model\.layers\.(\d+)\.")
    for k in sd.keys():
        m = pattern.search(k)
        if m:
            layer_indices.append(int(m.group(1)))
    if not layer_indices:
        raise ValueError("Could not infer num_layers from state dict keys.")
    num_layers = max(layer_indices) + 1

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

        # QKV fusion - handle bias if present
        q_bias_key = prefix + "self_attn.q_proj.bias"
        k_bias_key = prefix + "self_attn.k_proj.bias"
        v_bias_key = prefix + "self_attn.v_proj.bias"

        if all(key in sd for key in [q_bias_key, k_bias_key, v_bias_key]):
            q_bias = sd[q_bias_key]
            k_bias = sd[k_bias_key]
            v_bias = sd[v_bias_key]
            load_sd[prefix + "self_attn.qkv_proj.bias"] = torch.cat(
                [q_bias, k_bias, v_bias], dim=0
            )

        # MLP gate_up_proj fusion
        gate = sd[prefix + "mlp.gate_proj.weight"]
        up = sd[prefix + "mlp.up_proj.weight"]
        load_sd[prefix + "mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)

        # MLP gate_up_proj fusion - handle bias if present
        gate_bias_key = prefix + "mlp.gate_proj.bias"
        up_bias_key = prefix + "mlp.up_proj.bias"

        if all(key in sd for key in [gate_bias_key, up_bias_key]):
            gate_bias = sd[gate_bias_key]
            up_bias = sd[up_bias_key]
            load_sd[prefix + "mlp.gate_up_proj.bias"] = torch.cat(
                [gate_bias, up_bias], dim=0
            )

    return load_sd
