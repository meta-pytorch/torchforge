# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for observability utility functions."""

from forge.controller.actor import ForgeActor

from forge.observability.utils import get_proc_name_with_rank
from monarch.actor import endpoint


class UtilActor(ForgeActor):
    """Actor for testing get_proc_name_with_rank in spawned context."""

    @endpoint
    async def get_name(self) -> str:
        return get_proc_name_with_rank()


class TestGetProcNameWithRank:
    """Tests for get_proc_name_with_rank utility."""

    def test_direct_proc(self):
        """Direct proc should return 'client_r0'."""
        assert get_proc_name_with_rank() == "client_r0"

    def test_direct_proc_with_override(self):
        """Direct proc with override should use provided name."""
        result = get_proc_name_with_rank(proc_name="MyProcess")
        assert result == "MyProcess_r0"

    # TODO (felipemello): currently not working with CI wheel, but passes locally
    # reactive once wheel is updated with new monarch version
    # @pytest.mark.timeout(10)
    # @pytest.mark.asyncio
    # async def test_replicas(self):
    #     """Test service with replicas returns unique names and hashes per replica."""
    #     actor = await UtilActor.options(
    #         procs=1, num_replicas=2, with_gpus=False
    #     ).as_service()
    #     results = await actor.get_name.fanout()

    #     assert len(results) == 2
    #     assert len(set(results)) == 2  # All names are unique
    #     for name in results:
    #         assert name.startswith("UtilActor")
    #         assert name.endswith("_r0")

    #     # Extract hashes from names (format: ActorName_replicaIdx_hash_r0)
    #     hashes = [name.split("_")[-2] for name in results]
    #     assert hashes[0] != hashes[1]  # Hashes are different between replicas
