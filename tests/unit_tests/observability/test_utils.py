# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for observability utility functions."""

import pytest

from forge.observability.utils import get_proc_name_with_rank
from monarch.actor import Actor, endpoint, this_host


class UtilActor(Actor):
    """Actor for testing get_proc_name_with_rank in spawned context."""

    @endpoint
    async def get_name(self) -> str:
        return get_proc_name_with_rank()

    @endpoint
    async def get_name_with_override(self, name: str) -> str:
        return get_proc_name_with_rank(proc_name=name)


class TestGetProcNameWithRank:
    """Tests for get_proc_name_with_rank utility."""

    def test_direct_proc(self):
        """Direct proc (test process) should return client_DPROC_r0."""
        result = get_proc_name_with_rank()
        assert result == "client_DPROC_r0"

    def test_direct_proc_with_override(self):
        """Direct proc with override should use provided name."""
        result = get_proc_name_with_rank(proc_name="MyProcess")
        assert result == "MyProcess_DPROC_r0"

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_spawned_actor(self):
        """Spawned actor should return ActorName_replica_rank format."""
        p = this_host().spawn_procs(per_host={"cpus": 2})
        actor = p.spawn("UtilActor", UtilActor)

        # no override
        results = await actor.get_name.call()

        assert len(results) == 2
        for i, (rank_info, result) in enumerate(results):
            replica_id = result.split("_")[1]
            assert result == f"UtilActor_{replica_id}_r{i}"

        # override name
        results = await actor.get_name_with_override.call("CustomName")

        for i, (rank_info, result) in enumerate(results):
            replica_id = result.split("_")[1]
            assert result == f"CustomName_{replica_id}_r{i}"
