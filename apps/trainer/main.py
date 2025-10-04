# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.trainer.main --config apps/trainer/trainer_config.yaml

import asyncio

import torch
import torch.nn.functional as F

from forge.actors.trainer import RLTrainer
from forge.cli.config import parse
from forge.controller.launcher import JOB_NAME_KEY, LAUNCHER_KEY
from forge.controller.provisioner import init_provisioner, shutdown
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.types import (
    Launcher,
    LauncherConfig,
    ProcessConfig,
    ProvisionerConfig,
    ServiceConfig,
)
from omegaconf import DictConfig


def placeholder_loss_function(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


async def main(cfg: DictConfig):
    """Main function that only initializes the trainer."""

    # Initialize provisioner
    await init_provisioner(
        ProvisionerConfig(
            launcher_config=LauncherConfig(
                launcher=Launcher(cfg.get(LAUNCHER_KEY, Launcher.SLURM.value)),
                job_name=cfg.get(JOB_NAME_KEY, None),
                services={k: ServiceConfig(**v) for k, v in cfg.services.items()},
                actors={k: ProcessConfig(**v) for k, v in cfg.actors.items()},
            )
        )
    )

    # Initialize metric logging
    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # Initialize trainer only
    print("Initializing trainer...")
    trainer = await RLTrainer.options(**cfg.actors.trainer).as_actor(
        **cfg.trainer, loss=placeholder_loss_function
    )

    print("Trainer initialized successfully!")
    print(f"Trainer configuration: {cfg.trainer}")

    # Keep the trainer running for demonstration
    # In a real scenario, you might want to expose endpoints or do other work here
    try:
        print("Trainer is running. Press Ctrl+C to shutdown...")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down trainer...")
    finally:
        await RLTrainer.shutdown(trainer)
        await mlogger.shutdown.call_one()
        await shutdown()
        print("Trainer shutdown complete.")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
