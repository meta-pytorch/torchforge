# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import logging
import time

from forge.controller.actor import ForgeActor
from forge.controller.provisioner import shutdown
from forge.observability.metric_actors import setup_metric_logger
from forge.observability.metrics import record_metric, ReductionType
from forge.observability.perf_tracker import (
    record_perf_metrics,
    record_perf_metrics_ctx,
    Timer,
)

from monarch.actor import current_rank, endpoint

logging.basicConfig(level=logging.DEBUG)


class TrainActor(ForgeActor):
    """Example training actor that records loss metrics."""

    @endpoint
    @record_perf_metrics(
        "trainer_perf", track_time=False, track_memory=True, use_cuda=False
    )
    async def train_step(self, step: int):
        rank = current_rank().rank

        # Phase 2: Use Timer for detailed step timing
        timer = Timer("trainer_perf/step", use_cuda=False)
        timer.start()

        # Simulate forward pass
        timer.step("forward")

        # Simulate backward pass
        timer.step("backward")

        value = rank * 1000 + 100 * step

        # Record training metrics
        record_metric("trainer/avg_grpo_loss", value, ReductionType.MEAN)
        record_metric("trainer/std_grpo_loss", value, ReductionType.STD)
        record_metric("trainer/count_training_steps", 1, ReductionType.SUM)
        record_metric("trainer/learning_rate", 0.001, ReductionType.MEAN)

        print(f"🔧 Train rank {rank}: Step {step}, loss={value}")

        timer.end()
        return value


class GeneratorActor(ForgeActor):
    """Example generation actor that records token count metrics."""

    @endpoint
    async def generate_step(self, step: int, substep: int):
        rank = current_rank().rank

        with record_perf_metrics_ctx(
            "policy_perf", track_time=True, track_memory=False, use_cuda=False
        ):

            value = rank * 1000 + step * 100 + substep * 10

            # Record generation metrics following the plan
            record_metric("policy/count_requests", 1, ReductionType.SUM)
            record_metric(
                "policy/sum_tokens_requested", 50, ReductionType.SUM
            )  # Simulated max_tokens
            record_metric("policy/sum_tokens_generated", value, ReductionType.SUM)
            record_metric("policy/count_sequences_completed", 1, ReductionType.SUM)
            record_metric("policy/avg_tokens_per_sample", value, ReductionType.MEAN)

            print(f"🎯 Gen rank {rank}: Step {step}.{substep}, tokens={value}")

        return value


# Main
async def main():
    """Example demonstrating distributed metric logging with different backends."""
    group = f"grpo_exp_{int(time.time())}"

    # Config format: {backend_name: backend_config_dict}
    # Each backend can specify reduce_across_ranks to control distributed logging behavior
    config = {
        "console": {"reduce_across_ranks": True},
        "wandb": {
            "project": "my_project",
            "group": group,
            "reduce_across_ranks": True,
            # Only useful if NOT reduce_across_ranks.
            "share_run_id": False,  # Share run ID across ranks -- Not recommended.
        },
    }

    service_config = {"procs": 2, "num_replicas": 2, "with_gpus": False}
    mlogger = await setup_metric_logger()

    # Spawn services first (triggers registrations via provisioner hook)
    trainer = await TrainActor.options(**service_config).as_service()
    generator = await GeneratorActor.options(**service_config).as_service()

    # Now init config on global (inits backends eagerly across fetchers)
    await mlogger.init_backends.call_one(config)

    for i in range(3):
        print(f"\n=== Global Step {i} ===")
        await trainer.train_step.fanout(i)
        for sub in range(3):
            await generator.generate_step.fanout(i, sub)
        await mlogger.flush.call_one(i)

    # shutdown
    await mlogger.shutdown.call_one()

    await asyncio.gather(
        trainer.shutdown(),
        generator.shutdown(),
    )

    await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
