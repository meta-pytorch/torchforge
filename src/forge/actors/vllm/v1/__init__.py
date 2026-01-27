# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator v1 package - AsyncLLM-based integration (vLLM >= 0.13.0).

Note: Imports are lazy to avoid circular import deadlocks when vLLM's
EngineCore subprocess imports monarch_executor.py.
"""


def __getattr__(name):
    """Lazy import to avoid circular import deadlocks."""
    if name == "Generator":
        from forge.actors.vllm.v1.generator import Generator

        return Generator
    if name == "MonarchExecutor":
        from forge.actors.vllm.v1.monarch_executor import MonarchExecutor

        return MonarchExecutor
    if name == "WorkerWrapper":
        from forge.actors.vllm.v1.monarch_executor import WorkerWrapper

        return WorkerWrapper
    if name == "ForgeMonarchExecutor":
        from forge.actors.vllm.v1.forge_executor import ForgeMonarchExecutor

        return ForgeMonarchExecutor
    if name == "ForgeWorkerWrapper":
        from forge.actors.vllm.v1.forge_executor import ForgeWorkerWrapper

        return ForgeWorkerWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Generator",
    "MonarchExecutor",
    "WorkerWrapper",
    "ForgeMonarchExecutor",
    "ForgeWorkerWrapper",
]
