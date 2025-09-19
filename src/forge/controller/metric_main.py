# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from dataclasses import dataclass
from typing import Any

from monarch.actor import context, endpoint, get_or_spawn_controller

from forge.controller.actor import ForgeActor

from forge.controller.metric_actors import debug_context, GlobalLoggingActor


def push_metrics(key: str, value: Any) -> None:
    """
    Push metrics to LocalLoggingActor

    Args:
        key: Metric name
        value: Metric value
    """
    try:
        # Just use the regular monarch context
        ctx = context()

        # Try to get LocalLoggingActor from context
        local_logging_actor = None

        if hasattr(ctx, "actor_instance") and hasattr(ctx.actor_instance, "_proc_mesh"):
            proc_mesh = ctx.actor_instance._proc_mesh
            if proc_mesh is not None and hasattr(proc_mesh, "_local_logger"):
                local_logging_actor = proc_mesh._local_logger
                print(f"✅ Found LocalLoggingActor via context for {key}")

        if local_logging_actor:
            local_logging_actor.push_metrics.broadcast(key, value)
        else:
            print(
                f"❌ No LocalLoggingActor found in context, dropping metric {key}={value}"
            )
            debug_context(ctx, f"CONTEXT DEBUG for {key}={value}")

    except Exception as e:
        print(f"❌ push_metrics failed for {key}={value}: {e}")
        import traceback

        traceback.print_exc()


async def flush(step: int) -> None:
    """Flush all metrics globally."""
    try:
        g = await get_or_spawn_controller("global_logger", GlobalLoggingActor)
        await g.flush.call_one(step)
    except Exception as e:
        print(f"❌ flush failed: {e}")
        import traceback

        traceback.print_exc()


@dataclass
class Trainer(ForgeActor):
    """Trainer that uses global_logger.push_metrics."""

    def __post_init__(self):
        self.step_counter = 0

    @endpoint
    async def train_step(self) -> int:
        """Simulate one training step."""
        self.step_counter += 1
        push_metrics("step_counter", self.step_counter)
        print(f"Trainer: Completed step {self.step_counter}")
        return self.step_counter

    @endpoint
    async def debug_context(self) -> None:
        """Debug what the service actor can see."""
        ctx = context()
        debug_context(ctx)


# ============================================================================
# Main Training Loop
# ============================================================================


async def continuous_training(trainer: Trainer, num_steps: int = 5):
    """Run training loop with periodic flushing."""
    print(f"\ Starting training for {num_steps} steps...")

    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")

        # Run training step
        await trainer.train_step.choose()

        if (step + 1) % 2 == 0:  # Flush every 2 steps
            print(f"🔄 Flushing metrics at step {step + 1}")
            await flush(step + 1)

        await asyncio.sleep(0.1)

    print("✅ Training completed!")


async def main():
    """Main function demonstrating the REAL issue (following your architecture)."""

    print("1. Spawning trainer service...")
    service_config = {"procs_per_replica": 1, "num_replicas": 1, "with_gpus": False}

    # This should internally:
    # - Call get_proc_mesh()
    # - Spawn LocalLoggingActor in that process (via provisioner.py changes)
    # - Register LocalLoggingActor with GlobalLoggingActor
    # - Make LocalLoggingActor accessible to the Trainer via context
    trainer = await Trainer.options(**service_config).as_service()

    # Debug what the service actor can see
    print("\n2. Debugging service actor context...")
    await trainer.debug_context.choose()

    # Test the full training loop with metrics + flushing
    print("\n3. Running training loop with metrics + flushing...")
    await continuous_training(trainer, num_steps=2)

    # Shutdown
    print("\n4. Shutting down...")
    await trainer.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n Failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise
