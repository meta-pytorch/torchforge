# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from typing import Any, Dict, Optional

from monarch.actor import Actor, current_rank, endpoint


class LocalFetcherActor(Actor):
    @endpoint
    async def log_and_reset(
        self, step: int, return_state: bool = False
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Log to local backends (if any), optionally return states, and reset."""
        from forge.controller.v3.metrics import MetricCollector

        collector = MetricCollector()
        result = await collector.log_and_reset(step, return_state=return_state)
        print(
            f"🎯 Fetcher rank {current_rank().rank}: Flushed step {step}, returned state: {return_state}"
        )
        return result

    @endpoint
    async def init_collector(self, primary_backend_states: Dict[str, Dict[str, Any]]):
        from forge.controller.v3.metrics import MetricCollector

        collector = MetricCollector()
        await collector._init(primary_backend_states)
        print(f"🎯 Fetcher rank {current_rank().rank}: Initialized collector")

    @endpoint
    async def shutdown(self):
        from forge.controller.v3.metrics import MetricCollector

        collector = MetricCollector()
        await collector.shutdown()
        print(f"🎯 Fetcher rank {current_rank().rank}: Finished all backends")


# GlobalLoggingActor (coordinator)
class GlobalLoggingActor(Actor):
    def __init__(self):
        self.fetchers: Dict[str, LocalFetcherActor] = {}
        self.config: Optional[Dict[str, Any]] = None
        self.global_backends: Dict[str, "Backend"] = {}
        self.primary_backend_states: Dict[str, Dict[str, Any]] = {}

    @endpoint
    async def init_config(self, config: Dict[str, Any]):
        """Main calls this to set config and init global backends if needed."""
        self.config = config

        # Validate unique classes
        classes = [b["class"] for b in config["backends"]]
        if len(set(classes)) != len(classes):
            raise ValueError("Duplicate backend classes in config")

        # Init global backends and states where needed
        from forge.controller.v3.metrics import create_backend

        for backend_config in config["backends"]:
            cls_name = backend_config["class"]
            backend = create_backend(
                backend_config
            )  # Factory: returns instance based on type

            await backend.setup(self.config, role="global")
            primary_state = backend.get_primary_state() or {}
            log_per_rank = backend_config.get("log_per_rank", True)
            if log_per_rank:
                self.primary_backend_states[cls_name] = primary_state
            if not log_per_rank:
                self.global_backends[cls_name] = backend
            print(
                f"🌍 Global: Processed backend {cls_name} (log_per_rank: {log_per_rank})"
            )

        # Eager init collectors on all registered fetchers in parallel, passing primary states
        if self.fetchers:
            tasks = [
                fetcher.init_collector.call(self.primary_backend_states)
                for fetcher in self.fetchers.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"🌍 Global: Initialized {len(tasks)} collectors in parallel")

        print("🌍 Global: Config set")

    @endpoint
    def get_metric_logger_cfg(self) -> Optional[Dict[str, Any]]:
        return self.config

    @endpoint
    async def register_fetcher(self, fetcher: LocalFetcherActor, name: str):
        self.fetchers[name] = fetcher
        print(f"🌍 Global: Registered {name} (total: {len(self.fetchers)})")

    @endpoint
    async def flush_global(self, step: int):
        if not self.fetchers:
            print("🌍 Global: No fetchers")
            return

        print(f"🌍 Global: Flushing step {step} across {len(self.fetchers)}")

        config = self.config
        has_reduce = any(not b.get("log_per_rank", True) for b in config["backends"])
        return_state = has_reduce  # Flag for reduce

        # Broadcast log_and_reset to all fetchers
        results = await asyncio.gather(
            *[
                f.log_and_reset.call(step, return_state=return_state)
                for f in self.fetchers.values()
            ],
            return_exceptions=True,
        )

        if has_reduce:
            # Flatten: Handle both single-process (dict/None) and multi-process (list of dicts/None)
            all_local_results = []
            for res in results:
                res = (
                    res._values
                )  # TODO: avoid using internal state. Could use items() instead, but has to parse metadata.
                if isinstance(res, list):
                    all_local_results.extend(res)
                elif res is not None:
                    all_local_results.append(res)

            # Filter states from results (None if not returned)
            all_local_states = [r for r in all_local_results if isinstance(r, dict)]
            if not all_local_states:
                print("🌍 Global: No local states gathered")
                return

            # Reduce
            from forge.controller.v3.metrics import reduce_across_ranks

            reduced_metrics = reduce_across_ranks(all_local_states)

            # Log to each global backend
            for backend_name, backend in self.global_backends.items():
                await backend.log(reduced_metrics, step)
            print(f"🌍 Global: Logged reduced metrics {reduced_metrics} at step {step}")

    @endpoint
    async def shutdown(self):
        # Finish per-rank backends via fetchers
        if self.fetchers:
            tasks = [fetcher.shutdown.call() for fetcher in self.fetchers.values()]
            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"🌍 Global: Finished {len(self.fetchers)} fetchers' backends")

        # Finish global backends
        for backend_name, backend in self.global_backends.items():
            await backend.finish()
            print(f"🌍 Global: Finished global backend {backend_name}")

        print("🌍 Global: Shutdown complete")
