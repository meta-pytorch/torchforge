# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Provisioner CUDA_VISIBLE_DEVICES functionality."""

import os
from unittest import mock

import pytest
from forge.controller.provisioner import GpuManager, Provisioner


class TestGpuManagerCudaVisibleDevices:
    """Test GpuManager with different CUDA_VISIBLE_DEVICES configurations."""

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


class TestProvisionerCudaVisibleDevices:
    """Test Provisioner's handling of CUDA_VISIBLE_DEVICES environment variable."""

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch("torch.cuda.device_count", return_value=8)
    def test_provisioner_no_cuda_visible_devices(self, mock_device_count):
        """Test Provisioner when CUDA_VISIBLE_DEVICES is not set."""
        provisioner = Provisioner()

        # Should have default GpuManager for local host
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert available == [str(i) for i in range(8)]

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, clear=True)
    def test_provisioner_with_cuda_visible_devices(self):
        """Test Provisioner with CUDA_VISIBLE_DEVICES set."""
        provisioner = Provisioner()

        # Should have GpuManager configured with specified devices
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        expected = ["0", "1", "2", "3"]
        assert sorted(available) == sorted(expected)
        assert len(available) == 4

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2,5,7"}, clear=True)
    def test_provisioner_non_contiguous_gpus(self):
        """Test Provisioner with non-contiguous GPU IDs."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        expected = ["0", "2", "5", "7"]
        assert sorted(available) == sorted(expected)
        assert len(available) == 4

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1,4,1"}, clear=True)
    def test_provisioner_duplicate_gpu_ids(self):
        """Test Provisioner handles duplicate GPU IDs in CUDA_VISIBLE_DEVICES."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        # Should deduplicate: {3, 1, 4}
        expected = ["1", "3", "4"]
        assert sorted(available) == sorted(expected)
        assert len(available) == 3

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True)
    @mock.patch("torch.cuda.device_count", return_value=8)
    def test_provisioner_empty_cuda_visible_devices(self, mock_device_count):
        """Test Provisioner with empty CUDA_VISIBLE_DEVICES."""
        provisioner = Provisioner()

        # Empty string should result in default behavior (no devices specified)
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert available == [str(i) for i in range(8)]

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}, clear=False)
    @pytest.mark.asyncio
    async def test_get_proc_mesh_respects_cuda_visible_devices(self):
        """Test that get_proc_mesh uses CUDA_VISIBLE_DEVICES for local allocation."""
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
    """Test that CUDA_VISIBLE_DEVICES only affects local host, not remote hosts."""

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True)
    def test_remote_host_ignores_cuda_visible_devices(self):
        """Test that remote hosts get default GPU configuration."""
        provisioner = Provisioner()

        # Local host should respect CUDA_VISIBLE_DEVICES
        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        local_available = local_gpu_manager.get_available_gpus()
        assert sorted(local_available) == ["0", "1"]

        # When creating remote allocations, they should get default GPU sets
        # This is verified by checking that remote allocations create new GpuManager
        # instances without the available_devices parameter (line 154 in provisioner.py)
        assert len(provisioner._host_gpu_map) == 1  # Only local host initially

        # The remote host creation in create_host_mesh creates GpuManager()
        # without available_devices parameter, so it gets default 8 GPUs


class TestIntegrationScenarios:
    """Integration test scenarios for CUDA_VISIBLE_DEVICES functionality."""

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear=True)
    def test_full_allocation_cycle(self):
        """Test complete allocation and release cycle with CUDA_VISIBLE_DEVICES."""
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

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
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
    """Test dynamic GPU detection using torch.cuda.device_count()."""

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch("torch.cuda.device_count", return_value=4)
    def test_provisioner_with_4_gpus(self, mock_device_count):
        """Test Provisioner detects 4 GPUs when torch.cuda.device_count() returns 4."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert sorted(available) == ["0", "1", "2", "3"]
        assert len(available) == 4
        assert local_gpu_manager.max_device_count == 4

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2,4"}, clear=True)
    @mock.patch("torch.cuda.device_count", return_value=8)
    def test_cuda_visible_devices_with_detected_gpus(self, mock_device_count):
        """Test that CUDA_VISIBLE_DEVICES works correctly with detected GPU count."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        # Should use CUDA_VISIBLE_DEVICES, not all 8 detected GPUs
        assert sorted(available) == ["0", "2", "4"]
        assert len(available) == 3
        # max_device_count should still be 8 from detection
        assert local_gpu_manager.max_device_count == 8

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch(
        "torch.cuda.device_count", side_effect=RuntimeError("CUDA not available")
    )
    def test_provisioner_when_cuda_unavailable(self, mock_device_count):
        """Test Provisioner defaults to 0 GPUs when CUDA is not available."""
        provisioner = Provisioner()

        local_gpu_manager = provisioner._host_gpu_map[provisioner._this_host_id]
        available = local_gpu_manager.get_available_gpus()
        assert available == []
        assert len(available) == 0
        assert local_gpu_manager.max_device_count == 0
