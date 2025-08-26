# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.
#
# """Postprocessor for trajectories.
#
# Currently returns reference logits only through Titan.
#
# """
#
#
# import logging
# import math
# import os
# from typing import Any
#
# import torch
#
# import torchtitan.experiments.forge.train_spec as forge_train_spec
#
# from monarch.actor import current_rank, current_size, endpoint
# from torch import nn
# from torchtitan.components.checkpoint import ModelWrapper
# from torchtitan.components.loss import LossFunction
# from torchtitan.components.lr_scheduler import LRSchedulersContainer
# from torchtitan.components.optimizer import OptimizersContainer
# from torchtitan.distributed import ParallelDims, utils as dist_utils
# from torchtitan.experiments.forge.engine import ForgeEngine
# from torchtitan.experiments.forge.job_config import ForgeJobConfig
#
# from forge.controller import ForgeActor
#
# # stubs for now
# Checkpointer = Any
# Dataloader = Any
# MetricLogger = Any
# Profiler = Any
# Tokenizer = Any
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
#
# class PostProcessor(ForgeActor, ForgeEngine):
#     # TODO - trim this down
#     job_config: ForgeJobConfig
#     train_spec: forge_train_spec.ForgeTrainSpec
#     parallel_dims: ParallelDims
#     model: list[nn.Module]
#     loss_fn: LossFunction
#     optimizer: OptimizersContainer
#     lr_scheduler: LRSchedulersContainer
#     checkpointer: Checkpointer
#     tokenizer: Tokenizer
#     train_dataloader: Dataloader
#     # val_dataloader: Dataloader
#     metric_logger: MetricLogger
#     profiler: Profiler
#     device: torch.device
#     step: int
#
#     def __init__(self, job_config: ForgeJobConfig):
#         self._rank = current_rank().rank
#         self._size = math.prod(current_size().values())
#         self._init_dist()
#         parallel_dims = self.parallel_dims
#         assert not parallel_dims.pp_enabled, "PostProcessor does not support PP"
#
#         super().__init__(job_config)
#
#     def _init_dist(self):
#         """Initializes torch distributed.
#
#         torchrun normally hands this, but we need to do it ourselves
#         in monarch for now.
#
#         We should consider putting this into ForgeActor, but having this
#         be explicit for now.
#
#         """
#         env = {
#             "MASTER_ADDR": "localhost",
#             "MASTER_PORT": "12345",
#             "RANK": str(self._rank),
#             "LOCAL_RANK": str(self._rank),
#             "LOCAL_WORLD_SIZE": str(self._size),
#             "GROUP_RANK": str(self._size),
#             "GROUP_WORLD_SIZE": str(self._size),
#             "ROLE_RANK": str(self._rank),
#             "ROLE_WORLD_SIZE": str(self._size),
#             "ROLE_NAME": "rank",
#             "WORLD_SIZE": str(self._size),
#             "CUDA_VISIBLE_DEVICES": str(self._rank),
#             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
#         }
#         os.environ.update(env)
#         logger.info("env: {}".format(env))
#
#     @endpoint
#     async def setup(self):
#         # self.tokenizer = HuggingFaceModelTokenizer(
#         #     tokenizer_json_path=os.path.join(
#         #         self.job_config.model.tokenizer_path, "tokenizer.json"
#         #     ),
#         #     tokenizer_config_json_path=os.path.join(
#         #         self.job_config.model.tokenizer_path, "tokenizer_config.json"
#         #     ),
#         #     generation_config_path=os.path.join(
#         #         self.job_config.model.tokenizer_path, "generation_config.json"
#         #     ),
#         # )
#
#         # Should also use load, not dcp_load
#         self.checkpointer.dcp_load(
#             state_dict=ModelWrapper(self.model_parts).state_dict(),
#             checkpoint_id=self.job_config.checkpoint.folder,
#             from_hf=True,
#         )
#
#     def forward(
#         self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
#     ) -> torch.Tensor | None:
#         # TODO - get rid of the losses completely
#         # TODO - figure out what parallelisms we really need
#         # TODO - this should really just be a
#         model_parts = self.model_parts
#         parallel_dims = self.parallel_dims
#
#         # apply context parallelism if cp is enabled
#         # ensure CP handles the separate freqs_cis buffer for each pp stage
#         inputs = input_dict["tokens"]
#         optional_context_parallel_ctx = (
#             dist_utils.create_context_parallel_ctx(
#                 cp_mesh=parallel_dims.world_mesh["cp"],
#                 cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
#                 cp_seq_dims=[1, 1] + [0 for _ in model_parts],
#                 cp_no_restore_buffers={inputs, labels},
#                 cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
#             )
#             if parallel_dims.cp_enabled
#             else None
#         )
#
#         with self.train_context(optional_context_parallel_ctx):
#             assert len(model_parts) == 1
#             with self.maybe_enable_amp:
#                 targets = model_parts[0](inputs)
#
#         return targets
#
#     @endpoint
#     async def step(self, batch) -> torch.Tensor | None:
#         labels = batch.pop("labels")
#         logits = self.forward(batch, labels)
#         return logits
#
#     @endpoint
#     async def cleanup(self) -> None:
#         if self.checkpointer:
#             self.checkpointer.close()
#         if self.metric_logger:
#             self.metric_logger.close()
#
#     def __repr__(self) -> str:
#         return "PostProcessor"
