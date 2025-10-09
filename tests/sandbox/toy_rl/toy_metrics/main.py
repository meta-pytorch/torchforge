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
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import trace, Tracer

from monarch.actor import current_rank, endpoint

logging.basicConfig(level=logging.DEBUG)


class TrainActor(ForgeActor):
    """Example training actor that records loss metrics."""

    @endpoint
    async def train_step(self, step: int):
        rank = current_rank().rank

        # Phase 2: Use Tracer for detailed step timing
        tracer = Tracer("trainer_perf/step", track_memory=True, timer="gpu")
        tracer.start()

        # Simulate forward pass
        tracer.step("forward")

        # Simulate backward pass
        tracer.step("backward")

        value = rank * 1000 + 100 * step

        # Record training metrics
        record_metric("trainer/avg_grpo_loss", value, Reduce.MEAN)
        record_metric("trainer/std_grpo_loss", value, Reduce.STD)
        record_metric("trainer/count_training_steps", 1, Reduce.SUM)
        record_metric("trainer/learning_rate", 0.001, Reduce.MEAN)

        print(f"🔧 Train rank {rank}: Step {step}, loss={value}")

        tracer.stop()
        return value


class GeneratorActor(ForgeActor):
    """Example generation actor that records token count metrics."""

    @endpoint
    async def generate_step(self, step: int, substep: int):
        rank = current_rank().rank

        with trace("policy_perf", track_memory=False, timer="gpu") as tracer:

            value = rank * 1000 + step * 100 + substep * 10
            tracer.step("time_to_value")
            # Record generation metrics following the plan
            record_metric("policy/count_requests", 1, Reduce.SUM)
            record_metric(
                "policy/sum_tokens_requested", 50, Reduce.SUM
            )  # Simulated max_tokens
            record_metric("policy/sum_tokens_generated", value, Reduce.SUM)
            record_metric("policy/count_sequences_completed", 1, Reduce.SUM)
            record_metric("policy/avg_tokens_per_sample", value, Reduce.MEAN)

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
            "reduce_across_ranks": False,
            # Only useful if NOT reduce_across_ranks.
            "share_run_id": False,  # Share run ID across ranks -- Not recommended.
        },
    }

    service_config = {"procs": 2, "num_replicas": 2, "with_gpus": False}
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(config)

    # Spawn services first (triggers registrations via provisioner hook)
    trainer = await TrainActor.options(**service_config).as_service()
    generator = await GeneratorActor.options(**service_config).as_service()

    for i in range(3):
        print(f"\n=== Global Step {i} ===")
        await trainer.train_step.fanout(i)
        for sub in range(3):
            await generator.generate_step.fanout(i, sub)
        await mlogger.flush.call_one(i)

    # shutdown
    await mlogger.shutdown.call_one()
    await asyncio.sleep(2)
    await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
