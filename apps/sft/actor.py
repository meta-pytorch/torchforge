# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract Actor class for training/inference actors in Forge.

This provides a base class that can be extended for different types of actors
(e.g., Trainer, Evaluator, Inferencer, etc.)
"""

import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from forge.controller import ForgeActor
from monarch.actor import current_rank, current_size
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseForgeActor(ForgeActor, ForgeEngine, ABC):
    """
    Abstract base class for Forge actors.

    This class handles common initialization, distributed setup, and provides
    abstract methods that must be implemented by concrete actor classes.
    """

    job_config: ForgeJobConfig
    parallel_dims: ParallelDims
    model: list[nn.Module]
    loss_fn: Optional[LossFunction]
    optimizer: Optional[OptimizersContainer]
    lr_scheduler: Optional[LRSchedulersContainer]
    checkpointer: Optional[Checkpointer]
    tokenizer: Optional[Tokenizer]
    metric_logger: Optional[MetricLogger]
    profiler: Optional[Profiler]
    device: torch.device

    def __init__(self, config: DictConfig):
        """
        Initialize the base actor with configuration.

        Args:
            config: Configuration dictionary containing job settings
        """
        job_config = ForgeJobConfig().to_dict()
        job_config = OmegaConf.merge(job_config, config)

        self.current_step = 0
        self.metric_logger = None
        self.gradient_accumulation_steps = 1
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())

        self._init_dist()
        super().__init__(job_config)

    def _init_dist(self):
        """
        Initialize torch distributed environment.

        Sets up environment variables required for distributed training
        in the Monarch actor framework.
        """
        env = {
            "RANK": str(self._rank),
            "LOCAL_RANK": str(self._rank),
            "LOCAL_WORLD_SIZE": str(self._size),
            "GROUP_RANK": str(self._size),
            "GROUP_WORLD_SIZE": str(self._size),
            "ROLE_RANK": str(self._rank),
            "ROLE_WORLD_SIZE": str(self._size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self._size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)
        logger.info(f"Initialized distributed environment: {env}")

    @abstractmethod
    async def setup(self):
        """
        Setup the actor (load data, checkpoint, etc.).

        This method must be implemented by concrete actor classes.
        """
        pass

    @abstractmethod
    async def run(self):
        """
        Main execution logic for the actor.

        This method must be implemented by concrete actor classes.
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """
        Cleanup resources (close checkpointer, logger, etc.).

        This method must be implemented by concrete actor classes.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the actor."""
        pass
