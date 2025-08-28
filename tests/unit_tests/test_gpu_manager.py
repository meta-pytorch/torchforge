# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for GPU manager functionality."""

import pytest
from forge.controller.custom_actors.gpu_manager import GpuManager
from monarch.actor import ActorError, proc_mesh


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_initialization():
    """Test GPU manager initialization."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)
    available_gpus = await manager.get_available_gpus.call_one()

    # Should have 8 GPUs available by default
    assert available_gpus == [str(i) for i in range(8)]
    assert len(available_gpus) == 8


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_get_gpus_basic():
    """Test basic GPU allocation."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    # Request 2 GPUs
    result = await manager.get_gpus.call_one(2)

    # Should return 2 GPU IDs as strings
    assert len(result) == 2
    assert all(isinstance(gpu_id, str) for gpu_id in result)

    # Should be valid GPU IDs (0-7)
    gpu_ints = [int(gpu_id) for gpu_id in result]
    assert all(0 <= gpu_id <= 7 for gpu_id in gpu_ints)

    # Check remaining available GPUs
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 6

    # Allocated GPUs should no longer be available
    for gpu_id in gpu_ints:
        assert str(gpu_id) not in available_gpus


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_get_gpus_all():
    """Test allocating all available GPUs."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    # Request all 8 GPUs
    result = await manager.get_gpus.call_one(8)

    assert len(result) == 8

    # Check no GPUs are available
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 0

    # All original GPUs should be allocated
    allocated_ints = {int(gpu_id) for gpu_id in result}
    assert allocated_ints == set(range(8))


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_get_gpus_insufficient():
    """Test error when requesting more GPUs than available."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    # Request more than 8 GPUs should raise an error
    with pytest.raises(ActorError, match="Not enough GPUs available"):
        await manager.get_gpus.call_one(9)

    # Available GPUs should remain unchanged
    available_gpus = await manager.get_available_gpus.call_one()
    assert available_gpus == [str(i) for i in range(8)]


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_get_gpus_zero():
    """Test requesting zero GPUs."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    result = await manager.get_gpus.call_one(0)

    assert result == []

    # Available GPUs should remain unchanged
    available_gpus = await manager.get_available_gpus.call_one()
    assert available_gpus == [str(i) for i in range(8)]


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_release_gpus_basic():
    """Test basic GPU release functionality."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    # Allocate some GPUs
    allocated = await manager.get_gpus.call_one(3)

    # Check they're no longer available
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 5

    # Release them back
    await manager.release_gpus.call_one(allocated)

    # Should have all 8 GPUs available again
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 8
    assert set(available_gpus) == {str(i) for i in range(8)}


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_release_gpus_partial():
    """Test releasing only some of the allocated GPUs."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    # Allocate 4 GPUs
    allocated = await manager.get_gpus.call_one(4)

    # Check available count
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 4

    # Release only 2 of them
    to_release = allocated[:2]
    await manager.release_gpus.call_one(to_release)

    # Should have 6 GPUs available now
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 6

    # The released GPUs should be back in available set
    available_ints = {int(gpu_id) for gpu_id in available_gpus}
    for gpu_id in to_release:
        assert int(gpu_id) in available_ints


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_release_gpus_empty():
    """Test releasing empty list of GPUs."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    await manager.release_gpus.call_one([])

    # Should remain unchanged
    available_gpus = await manager.get_available_gpus.call_one()
    assert available_gpus == [str(i) for i in range(8)]


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_allocation_release_cycle():
    """Test multiple allocation and release cycles."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    # Cycle 1: Allocate 3, release 3
    batch1 = await manager.get_gpus.call_one(3)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 5

    await manager.release_gpus.call_one(batch1)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 8

    # Cycle 2: Allocate 5, release 5
    batch2 = await manager.get_gpus.call_one(5)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 3

    await manager.release_gpus.call_one(batch2)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 8

    # Should be back to original state
    assert set(available_gpus) == {str(i) for i in range(8)}


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_incremental_allocation():
    """Test incremental allocation until exhaustion."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)
    all_allocated = []

    # Allocate in chunks
    batch1 = await manager.get_gpus.call_one(2)  # 6 remaining
    all_allocated.extend(batch1)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 6

    batch2 = await manager.get_gpus.call_one(3)  # 3 remaining
    all_allocated.extend(batch2)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 3

    batch3 = await manager.get_gpus.call_one(3)  # 0 remaining
    all_allocated.extend(batch3)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 0

    # Should have allocated all 8 GPUs
    assert len(all_allocated) == 8

    # Should fail to allocate more
    with pytest.raises(ActorError):
        await manager.get_gpus.call_one(1)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_concurrent_operations_simulation():
    """Test simulated concurrent operations."""
    p = proc_mesh(gpus=1)
    manager = p.spawn("GpuManager", GpuManager)

    # Simulate multiple "jobs" allocating and releasing
    job1_gpus = await manager.get_gpus.call_one(2)
    job2_gpus = await manager.get_gpus.call_one(3)

    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 3

    # Job1 releases its GPUs
    await manager.release_gpus.call_one(job1_gpus)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 5

    # Job3 allocates some GPUs
    job3_gpus = await manager.get_gpus.call_one(4)
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 1

    # Job2 and Job3 release
    await manager.release_gpus.call_one(job2_gpus)
    await manager.release_gpus.call_one(job3_gpus)

    # Should be back to full capacity
    available_gpus = await manager.get_available_gpus.call_one()
    assert len(available_gpus) == 8
    assert set(available_gpus) == {str(i) for i in range(8)}
