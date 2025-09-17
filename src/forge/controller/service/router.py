# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from .interface import Router
from .replica import Replica


class RoundRobinRouter(Router):
    """Round-robin router for stateless requests."""

    def __init__(self):
        self._next_idx = 0

    def get_replica(
        self, replicas: List[Replica], sess_id: str | None = None
    ) -> Replica:
        healthy_replicas = [r for r in replicas if r.healthy]
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available for load balancing")

        self._next_idx = (self._next_idx + 1) % len(healthy_replicas)
        replica = healthy_replicas[self._next_idx]

        return replica
