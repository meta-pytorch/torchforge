# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Provisioner device visibility env var functionality."""

import os
from unittest import mock

import pytest
from forge.controller.provisioner import GpuManager, Provisioner


def mock_patch_visible_devices_var(value: str, *, clear: bool = True):
    return mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}, clear=clear)


@pytest.fixture(autouse=True)
def mock_accelerator_available():
    """Auto-mock torch.accelerator for all tests in this module."""
    mock_accelerator = mock.MagicMock()
    mock_accelerator.type = "cuda"

    with (
        mock.patch("torch.accelerator.is_available", return_value=True),
        mock.patch("torch.accelerator.device_count", return_value=8),
        mock.patch(
            "torch.accelerator.current_accelerator", return_value=mock_accelerator
        ),
    ):
        yield


class TestGpuManagerVisibleDevices:
    """Test GpuManager with different configurations."""

    def test_gpu_manager_default_initialization(self):
        """Test GpuManager initializes with default 8 GPUs when no specific devices provided."""
        manager = GpuManager()
        available = manager.get_available_gpus()
        assert available == [str(i) for i in range(8)]
        assert len(available) == 8

    def test_gpu_manager_custom_devices(self):
        """Test GpuManager with specific available devices."""
        custom_devices = {0, 2, 4, 6}
        manager = GpuManager(available_devices=custom_devices)
        available = manager.get_available_gpus()
        expected = ["0", "2", "4", "6"]
        assert sorted(available) == sorted(expected)
        assert len(available) == 4

    def test_gpu_manager_empty_devices(self):
        """Test GpuManager with no available devices."""
        empty_devices = set()
        manager = GpuManager(available_devices=empty_devices)
        available = manager.get_available_gpus()
        assert available == []
        assert len(available) == 0

    def test_gpu_manager_invalid_device_range(self):
        """Test GpuManager validation of device ranges."""
        with pytest.raises(AssertionError):
            GpuManager(available_devices={-1})  # Negative device

        with pytest.raises(AssertionError):
            GpuManager(available_devices={"0"})  # String instead of int

    def test_gpu_allocation_with_custom_devices(self):
        """Test GPU allocation with custom device set."""
        custom_devices = {1, 3, 5}
        manager = GpuManager(available_devices=custom_devices)

        # Get 2 GPUs
        allocated = manager.get_gpus(2)
        assert len(allocated) == 2
        assert all(gpu in ["1", "3", "5"] for gpu in allocated)

        # Check remaining
        remaining = manager.get_available_gpus()
        assert len(remaining) == 1

        # Total allocated + remaining should equal original
        all_gpus = set(allocated + remaining)
        assert all_gpus == {"1", "3", "5"}

    def test_gpu_release_with_custom_devices(self):
        """Test GPU release with custom device set."""
        custom_devices = {2, 4, 7}
        manager = GpuManager(available_devices=custom_devices)

        # Allocate all
        allocated = manager.get_gpus(3)
        assert len(allocated) == 3
        assert manager.get_available_gpus() == []

        # Release some
        manager.release_gpus([allocated[0]])
        remaining = manager.get_available_gpus()
        assert len(remaining) == 1
        assert remaining[0] == allocated[0]


class TestProvisionerVisibleDevices:
    """Test Provisioner's handling of device visibility env var."""

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_provisioner_no_visible_devices_var(self):
        """Test Provisioner when device visibility env var is not set."""
        provisioner = Provisioner()

        # Should have default GpuManager for local host
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert available == [str(i) for i in range(8)]

    @mock_patch_visible_devices_var("0,1,2,3", clear=True)
    def test_provisioner_with_visible_devices_var(self):
        """Test Provisioner with device visibility env var set."""
        provisioner = Provisioner()

        # Should have GpuManager configured with specified devices
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        expected = ["0", "1", "2", "3"]
        assert sorted(available) == sorted(expected)
        assert len(available) == 4

    @mock_patch_visible_devices_var("0,2,5,7", clear=True)
    def test_provisioner_non_contiguous_gpus(self):
        """Test Provisioner with non-contiguous GPU IDs."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        expected = ["0", "2", "5", "7"]
        assert sorted(available) == sorted(expected)
        assert len(available) == 4

    @mock_patch_visible_devices_var("3,1,4,1", clear=True)
    def test_provisioner_duplicate_gpu_ids(self):
        """Test Provisioner handles duplicate GPU IDs in device visibility env var."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        # Should deduplicate: {3, 1, 4}
        expected = ["1", "3", "4"]
        assert sorted(available) == sorted(expected)
        assert len(available) == 3

    @mock_patch_visible_devices_var("", clear=True)
    def test_provisioner_empty_visible_devices_var(self):
        """Test Provisioner with empty device visibility env var."""
        provisioner = Provisioner()

        # Empty string should result in default behavior (no devices specified)
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert available == [str(i) for i in range(8)]

    @mock_patch_visible_devices_var("0,1,2", clear=False)
    @pytest.mark.asyncio
    async def test_get_proc_mesh_respects_visible_devices_var(self):
        """Test that get_proc_mesh uses device visibility env var for local allocation."""
        provisioner = Provisioner()

        # Verify initial state
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        initial_available = local_gpu_manager.get_available_gpus()
        assert sorted(initial_available) == ["0", "1", "2"]

        # Note - this can run even on CPU because with_gpus just sets environment
        # variables.
        _ = await provisioner.get_proc_mesh(
            num_procs=2,
            mesh_name="test",
            with_gpus=True,
            num_hosts=None,
            port="12345",
            addr="localhost",
        )
        # Verify GPUs were allocated from available set
        remaining_available = local_gpu_manager.get_available_gpus()
        assert len(remaining_available) == 1  # Started with 3, allocated 2


class TestProvisionerEnvironmentIsolation:
    """Test that device visibility env var only affects local host, not remote hosts."""

    @mock_patch_visible_devices_var("0,1", clear=True)
    def test_remote_host_ignores_visible_devices_var(self):
        """Test that remote hosts get default GPU configuration."""
        provisioner = Provisioner()

        # Local host should respect device visibility env var
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        local_available = local_gpu_manager.get_available_gpus()
        assert sorted(local_available) == ["0", "1"]

        # When creating remote allocations, they should get default GPU sets
        # This is verified by checking that remote allocations create new GpuManager
        # instances without the available_devices parameter (line 154 in provisioner.py)
        assert len(provisioner._host_gpu_map) == 1  # Only local host initially

        # Remote host creation in get_proc_mesh creates GpuManager()
        # without available_devices parameter, so it gets default 8 GPUs


class TestIntegrationScenarios:
    """Integration test scenarios for device visibility env var functionality."""

    @mock_patch_visible_devices_var("1,3", clear=True)
    def test_full_allocation_cycle(self):
        """Test complete allocation and release cycle with device visibility env var."""
        provisioner = Provisioner()
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]

        # Initial state
        assert sorted(local_gpu_manager.get_available_gpus()) == ["1", "3"]

        # Allocate all available GPUs
        allocated = local_gpu_manager.get_gpus(2)
        assert len(allocated) == 2
        assert sorted(allocated) == ["1", "3"]
        assert local_gpu_manager.get_available_gpus() == []

        # Try to allocate more - should fail
        with pytest.raises(RuntimeError, match="Not enough GPUs available"):
            local_gpu_manager.get_gpus(1)

        # Release some GPUs
        local_gpu_manager.release_gpus([allocated[0]])
        remaining = local_gpu_manager.get_available_gpus()
        assert len(remaining) == 1
        assert remaining[0] == allocated[0]

        # Release all GPUs
        local_gpu_manager.release_gpus([allocated[1]])
        final_available = local_gpu_manager.get_available_gpus()
        assert sorted(final_available) == ["1", "3"]

    @mock_patch_visible_devices_var("0", clear=True)
    def test_single_gpu_scenario(self):
        """Test scenario with only one GPU available."""
        provisioner = Provisioner()
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]

        # Should have only GPU 0
        assert local_gpu_manager.get_available_gpus() == ["0"]

        # Allocate the single GPU
        allocated = local_gpu_manager.get_gpus(1)
        assert allocated == ["0"]
        assert local_gpu_manager.get_available_gpus() == []

        # Should fail to allocate any more
        with pytest.raises(RuntimeError):
            local_gpu_manager.get_gpus(1)

        # Release and verify
        local_gpu_manager.release_gpus(allocated)
        assert local_gpu_manager.get_available_gpus() == ["0"]


class TestDynamicGpuDetection:
    """Test dynamic GPU detection using torch.accelerator.device_count()."""

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch("torch.accelerator.device_count", return_value=4)  # Override fixture
    def test_provisioner_with_4_gpus(self, mock_device_count):
        """Test Provisioner detects 4 GPUs when torch.accelerator.device_count() returns 4."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert sorted(available) == ["0", "1", "2", "3"]
        assert len(available) == 4
        assert local_gpu_manager.max_device_count == 4

    @mock_patch_visible_devices_var("0,2,4", clear=True)
    def test_visible_devices_var_with_detected_gpus(self):
        """Test that device visibility env var works correctly with detected GPU count."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        # Should use device visibility env var, not all 8 detected GPUs
        assert sorted(available) == ["0", "2", "4"]
        assert len(available) == 3
        # max_device_count should still be 8 from detection
        assert local_gpu_manager.max_device_count == 8

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch("torch.accelerator.device_count", return_value=0)  # Override fixture
    @mock.patch(
        "torch.accelerator.is_available", return_value=False
    )  # Override fixture
    def test_provisioner_when_accelerator_unavailable(
        self, mock_available, mock_device_count
    ):
        """Test Provisioner defaults to 0 GPUs when no accelerator is available."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert available == []
        assert len(available) == 0
        assert local_gpu_manager.max_device_count == 0
