# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
from functools import partial

import torch
from forge.data_models.api import Trainer

# from monarch._src.tensor_engine.rdma import RDMABuffer
from forge.data_models.loss import LossInput, LossOutput
from forge.data_models.minibatch import Minibatch
from forge.losses.reinforce import ReinforceLoss
from torch import distributed as dist, nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceTrainer(Trainer):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_name = model_path
        self.model = None
        self.learning_rate: float = 1e-05  # TODO: needs to come from config
        self.lossfn = ReinforceLoss()
        self._loss_weights = []

        # torch.dist
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.backend = "nccl"
        self.setup_distributed()

        # make model
        self.model, self.tokenizer = self.load_model()
        self.model.train()

        # FSDP??
        self.model = self.fsdp_wrap_model(self.model, self.local_rank)
        self.optimizer = self.build_optimizer()
        self.device = next(self.model.parameters()).device

        # for snapshotting weights
        # TODO: RDMABuffer needs integration
        self._weights_handle: dict[
            str, tuple[torch.Tensor, torch.dtype, torch.Size]
        ] = {}

        # TODO: post-init steps like load checkpoint etc.

    def accummulate_gradients(self, minibatch: Minibatch) -> LossOutput:
        minibatch = self.host_to_device(minibatch, self.device)
        trainer_logits = self.forward(
            input_ids=minibatch.input_ids,
            segment_ids=minibatch.segment_ids,
            pad_index=-1,
        ).logits.to(self.device, non_blocking=True)
        loss_fn_output = self.lossfn.loss(
            LossInput(minibatch=minibatch, trainer_logits=trainer_logits)
        )

        # Suppose we have 3 raw losses L1, L2, L3 and 3 sum of masks M1, M2, M3.
        # The losses returned by loss_fn are L1/M1, L2/M2, L3/M3.
        # At the end we want the total loss to be (L1+L2+L3)/(M1+M2+M3).
        # The problem is that when we get L1, we don't know M2 and M3.
        # So we need to keep track of Ms and rescale the gradients after each
        # microbatch.
        #         Init: L = 0
        # Microbatch 1: L = L * 0/M1 + L1/M1 * M1/M1
        # Microbatch 2: L = L * M1/(M1+M2) + L2/M2 * M2/(M1+M2)
        #                 = L1/(M1+M2) + L2/(M1+M2)
        # Microbatch 3: L = L * (M1+M2)/(M1+M2+M3) + L3/M3 * M3/(M1+M2+M3)
        #                 = L1/(M1+M2+M3) + L2/(M1+M2+M3) + L3/(M1+M2+M3)
        numerator = sum(self._loss_weights)
        weight = loss_fn_output.loss.denominator.reduce()
        self._loss_weights.append(weight)
        denominator = sum(self._loss_weights)
        self._scale_gradients(numerator / denominator)

        # Since only FSDP is used as parallelism (no TP),
        # currently num_dp == get_role_world_size().
        # In the future we might put DP / TP / CP etc. size in Trainer State, or
        # use mpu like internal trainer.
        num_dp = self.world_size

        # If we have multiple DP(FSDP) groups and 3 microbatches:
        # - DP A: has loss LA = (LA1 + LA2 + LA3) / (MA1 + MA2 + MA3)
        # - DP B: has loss LB = (LB1 + LB2 + LB3) / (MB1 + MB2 + MB3)
        # We need (LA1 + LA2 + LA3 + LB1 + LB2 + LB3) / (MA1 + MA2 + MA3 + MB1 +
        # MB2 + MB3).
        # We need to multiply by num_dp because when we do the backwards pass,
        # we'll end up taking the mean across dps: we will all reduce the final
        # loss, divided by num_dp; and so the num_dp cancels out.
        ((loss_fn_output.loss.numerator.local() * num_dp) / denominator).backward()

        # TODO: remove this return after local testing
        return loss_fn_output

    def apply_gradients(self) -> None:
        max_grad_norm = 1.0  # TODO: needs to come from config
        if isinstance(self.model, FSDP):
            self.model.clip_grad_norm_(max_norm=max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_grad_norm
            )
        self._loss_weights = []
        self.optimizer.step()
        self.optimizer.zero_grad()

    def snapshot_weights(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.dtype, torch.Size]]:
        # EPIC TODO: this assumes the model is NOT sharded.  We need to handle sharded models.
        # TODO 1: assumes the model uses FSDP
        # TODO 2: what about named_buffers()?
        with FSDP.summon_full_params(self.model):
            for k, v in self.model.named_parameters():
                uint8_tensor = v.cpu().view(torch.uint8).flatten()

                self._weights_handle[k] = (
                    # RDMABuffer(uint8_tensor),  # TODO: RDMABuffer
                    uint8_tensor,  # RDMABuffer
                    v.dtype,  # torch.dtype
                    v.shape,  # torch.Size
                )

            for k, v in self.model.named_buffers():
                uint8_tensor = v.cpu().view(torch.uint8).flatten()

                self._weights_handle[k] = (
                    # RDMABuffer(uint8_tensor),  # TODO: RDMABuffer
                    uint8_tensor,  # RDMABuffer
                    v.dtype,  # torch.dtype
                    v.shape,  # torch.Size
                )

        return self._weights_handle

    def setup_distributed(self):
        if not dist.is_initialized():
            torch.cuda.set_device(f"cuda:{self.local_rank}")
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
            )

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,  # Don't auto-assign devices
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def fsdp_wrap_model(self, model: nn.Module, local_rank: int) -> nn.Module:
        print(f"Rank {self.local_rank}: Loading model")

        # Move to GPU before wrapping with FSDP
        model = model.to(f"cuda:{self.local_rank}")

        # Wrap model with FSDP
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        # Get transformer layer classes from the model dynamically
        transformer_layer_cls = set()

        # Add standard PyTorch transformer layers
        transformer_layer_cls.update(
            {
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
            }
        )

        # Add HuggingFace transformer layer classes dynamically
        for module in model.modules():
            module_class = type(module)
            # Check if this looks like a transformer decoder layer
            if hasattr(module, "self_attn") and hasattr(module, "mlp"):
                transformer_layer_cls.add(module_class)
            # Also check for attention layers that might need wrapping
            elif "layer" in module_class.__name__.lower() and (
                "decoder" in module_class.__name__.lower()
                or "transformer" in module_class.__name__.lower()
            ):
                transformer_layer_cls.add(module_class)

        print(f"FSDP transformer layer classes: {transformer_layer_cls}")

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=self.local_rank,
            sync_module_states=True,
            param_init_fn=None,
        )
        return model

    def build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def host_to_device(self, minibatch: Minibatch, device: str) -> Minibatch:
        updated_dict = {}
        for field in dataclasses.fields(minibatch):
            tensor = getattr(minibatch, field.name)
            if tensor is not None:
                tensor = tensor.to(device=device, non_blocking=True)
            updated_dict[field.name] = tensor
        return dataclasses.replace(minibatch, **updated_dict)

    def forward(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor, pad_index: int
    ) -> torch.nn.Module:
        assert input_ids.shape == segment_ids.shape
        attention_mask = self.segment_ids_to_attention_mask(segment_ids, pad_index)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

    def segment_ids_to_attention_mask(
        self,
        segment_ids: torch.Tensor,
        pad_index: int,
    ) -> torch.Tensor:
        """
        Convert segment_ids [B, L] to position_ids [B, L] and attention_mask [B, L, L].

        Args:
            segment_ids: torch.LongTensor [B, L], where values indicate segment index
                        (tokens with the same id belong to the same contiguous segment).
            pad_index: the value of the padded indices.

        Returns:
            attention_mask: Mask allowing attention only within the same segment.
        """
        assert segment_ids.ndim == 2, "segment_ids must be [B, L]"
        seg = segment_ids.to(torch.long)
        _, L = seg.shape

        # Intra-segment attention.
        attention_mask = seg.unsqueeze(1) == seg.unsqueeze(2)
        valid = seg != pad_index
        attention_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)

        # Use causal mask.
        i = torch.arange(L, device=segment_ids.device)
        # [1, L, L] lower-triu.
        causal = (i[None, :] <= i[:, None])[None, :, :]
        # [B, L, L]
        attention_mask = attention_mask & causal

        # [B, 1, L, L]: this is to prevent the attention mask from being broadcasted.
        attention_mask = attention_mask[:, None, :, :]
        return attention_mask

    def _scale_gradients(self, scale: torch.Tensor) -> None:
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
