# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import sys
import time

from monarch.actor import current_rank, endpoint, get_or_spawn_controller

from forge.controller.actor import ForgeActor
from forge.controller.v3.metric_actors import GlobalLoggingActor
from forge.controller.v3.metrics import push_metrics, ReductionType


class TrainActor(ForgeActor):
    @endpoint
    async def train_step(self, step: int):
        rank = current_rank().rank
        value = rank * 1000 + 100 * step
        print(f"🔧 Train rank {rank}: Step {step}, loss={value}")
        await push_metrics("train/loss", value)


class GeneratorActor(ForgeActor):
    @endpoint
    async def generate_step(self, step: int, substep: int):
        rank = current_rank().rank
        value = rank * 1000 + step * 100 + substep * 10
        print(f"🎯 Gen rank {rank}: Step {step}.{substep}, tokens={value}")
        await push_metrics("generate/tokens", value, ReductionType.SUM)


# Main
async def main(mode: str = "wandb_all_log_all"):
    group = f"experiment_group_{int(time.time())}"
    if mode == "wandb_all_log_all":
        backends = [
            {"class": "console", "log_per_rank": True},
            {
                "class": "wandb",
                "project": "my_project",
                "group": group,
                "mode": "wandb_all_log_all",
                "log_per_rank": True,
            },
        ]
    elif mode == "wandb_rank_0_reduce_all":
        backends = [
            {
                "class": "wandb",
                "project": "my_project",
                "group": group,
                "mode": "wandb_rank_0_reduce_all",
                "log_per_rank": False,
            },
        ]
    else:  # wandb_rank_0_log_all
        backends = [
            {
                "class": "wandb",
                "project": "my_project",
                "group": group,
                "mode": "wandb_rank_0_log_all",
                "log_per_rank": True,
            },
        ]

    logging_config = {
        "backends": backends,
    }
    service_config = {"procs": 2, "num_replicas": 2, "with_gpus": False}

    # Spawn services first (triggers registrations via provisioner hook)
    trainer = await TrainActor.options(**service_config).as_service()
    generator = await GeneratorActor.options(**service_config).as_service()

    # Now init config on global (inits backends eagerly across fetchers)
    global_logger = await get_or_spawn_controller("global_logger", GlobalLoggingActor)
    await global_logger.init_config.call_one(logging_config)

    for i in range(3):
        print(f"\n=== Global Step {i} ===")
        await trainer.train_step.call(i)
        for sub in range(3):
            await generator.generate_step.call(i, sub)
        await global_logger.flush_global.call_one(i)

    await global_logger.shutdown.call_one()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "wandb_all_log_all"
    valid_modes = [
        "wandb_all_log_all",
        "wandb_rank_0_log_all",
        "wandb_rank_0_reduce_all",
    ]
    if mode not in valid_modes:
        print(f"Invalid mode: {mode}. Use {valid_modes}")
        sys.exit(1)
    asyncio.run(main(mode))
