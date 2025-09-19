# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from collections import defaultdict
from typing import Any, Dict, List

from monarch.actor import Actor, endpoint


# ============================================================================
# LocalLoggingActor
# ============================================================================
class LocalLoggingActor(Actor):
    """Local logging actor that accumulates metrics within a process."""

    def __init__(self):
        self._metrics = defaultdict(list)

    @endpoint
    async def get_metrics(self) -> Dict[str, List[Any]]:
        """Get all accumulated metrics (called by GlobalLoggingActor)."""
        # Return copy and reset for next collection
        result = dict(self._metrics)
        self._metrics.clear()
        print(f"LocalLoggingActor: Returning {len(result)} metric keys")
        return result

    @endpoint
    def push_metrics(
        self, key: str, value: Any
    ) -> None:  # Note: not async for broadcast
        """Store a metric value (called by service actors)."""
        self._metrics[key].append(value)
        print(f"LocalLoggingActor: Stored {key}={value}")


# ============================================================================
# Backend System
# ============================================================================
class Backend:
    """Base class for logging backends."""

    def push(self, metrics: Dict[str, Any], step: int) -> None:
        pass


class ConsoleBackend(Backend):
    """Simple console backend for testing."""

    def push(self, metrics: Dict[str, Any], step: int) -> None:
        print(f"\n=== METRICS STEP {step} ===")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("========================\n")


# ============================================================================
# GlobalLoggingActor
# ============================================================================


class GlobalLoggingActor(Actor):
    """Global logger that coordinates across all processes."""

    def __init__(self):
        self._loggers: Dict[str, LocalLoggingActor] = {}
        self._backends: List[Backend] = [ConsoleBackend()]  # Default console backend

    @endpoint
    async def register(self, local_actor: LocalLoggingActor, name: str) -> None:
        """Register a LocalLoggingActor from a process."""
        self._loggers[name] = local_actor
        print(f"GlobalLoggingActor: Registered {name}")

    @endpoint
    async def deregister(self, name: str) -> None:
        """Deregister a LocalLoggingActor."""
        if name in self._loggers:
            del self._loggers[name]
            print(f"GlobalLoggingActor: Deregistered {name}")

    @endpoint
    async def flush(self, step: int) -> None:
        """Collect metrics from all processes and send to backends."""
        if not self._loggers:
            print("GlobalLoggingActor: No loggers registered")
            return

        print(f"GlobalLoggingActor: Flushing metrics for step {step}")

        # Collect from all local loggers
        metrics_list = await asyncio.gather(
            *[actor.get_metrics.call_one() for actor in self._loggers.values()]
        )

        # Simple aggregation - just combine all metrics
        print("metrics_list", metrics_list)
        all_metrics = {}
        for metrics in metrics_list:
            for key, values in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].extend(values)

        # Send to all backends
        for backend in self._backends:
            backend.push(all_metrics, step)


# ===========================================================================


def debug_context(ctx, label: str = "DEBUG") -> None:
    """
    Utility function to fully debug a context object and its nested attributes.

    Args:
        ctx: The context object to debug
        label: Label for this debug session
    """
    print(f"\n=== {label} ===")
    print(f"Context type: {type(ctx)}")
    print(f"Context dir: {[attr for attr in dir(ctx) if not attr.startswith('__')]}")

    # Print all attributes
    for attr in dir(ctx):
        if not attr.startswith("__"):
            try:
                val = getattr(ctx, attr)
                print(f"  ctx.{attr}: {val} (type: {type(val)})")

                # f this is actor_instance, explore it
                if attr == "actor_instance":
                    print(f"    🔍 DEEP DIVE INTO ACTOR_INSTANCE:")
                    print(f"    Type: {type(val)}")

                    # Get all attributes
                    all_attrs = [a for a in dir(val) if not a.startswith("__")]
                    print(f"    All attributes: {all_attrs}")

                    # Check for anything related to proc, mesh, logger, etc.
                    interesting_attrs = [
                        a
                        for a in all_attrs
                        if any(
                            keyword in a.lower()
                            for keyword in [
                                "proc",
                                "mesh",
                                "log",
                                "local",
                                "spawn",
                                "process",
                                "actor",
                            ]
                        )
                    ]
                    print(f"    Interesting attributes: {interesting_attrs}")

                    # Print ALL attributes with their values
                    for sub_attr in all_attrs:
                        try:
                            sub_val = getattr(val, sub_attr)
                            print(
                                f"      └─ {sub_attr}: {sub_val} (type: {type(sub_val)})"
                            )

                            # If it's an object, go one level deeper
                            if hasattr(sub_val, "__dict__") or hasattr(
                                sub_val, "__slots__"
                            ):
                                try:
                                    deep_attrs = [
                                        a
                                        for a in dir(sub_val)
                                        if not a.startswith("__")
                                    ][
                                        :3
                                    ]  # Just first 3
                                    if deep_attrs:
                                        print(
                                            f"        └─ {sub_attr} has: {deep_attrs}"
                                        )
                                        for deep_attr in deep_attrs:
                                            try:
                                                deep_val = getattr(sub_val, deep_attr)
                                                print(
                                                    f"          └─ {deep_attr}: {deep_val}"
                                                )
                                            except:
                                                print(
                                                    f"          └─ {deep_attr}: <unable to access>"
                                                )
                                except:
                                    pass
                        except Exception as e:
                            print(f"      └─ {sub_attr}: <error accessing: {e}>")

                # For other attributes, shorter exploration
                elif hasattr(val, "__dict__") or hasattr(val, "__slots__"):
                    sub_attrs = [a for a in dir(val) if not a.startswith("__")]
                    if sub_attrs:
                        print(f"    └─ {attr} attributes: {sub_attrs}")
            except:
                print(f"  ctx.{attr}: <unable to access>")

    print(f"=== END {label} ===\n")
