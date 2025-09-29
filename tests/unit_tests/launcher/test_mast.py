# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import MagicMock, patch

import pytest
from forge.controller.launcher.mast import _get_port, MastProvisioner, MastSetupActor
from omegaconf import DictConfig


class TestMast:
    @pytest.mark.timeout(10)
    def test_get_port_returns_string(self):
        """Test _get_port returns a port number as string."""
        port = _get_port()
        assert isinstance(port, str)
        assert port.isdigit()
        assert 1024 <= int(port) <= 65535

    @pytest.mark.timeout(10)
    def test_job_name_from_config(self):
        """Test job name is taken from config when provided."""
        cfg = DictConfig({"job_name": "my-test-job", "services": {}})
        provisioner = MastProvisioner(cfg)
        assert provisioner.job_name == "my-test-job"

    @pytest.mark.timeout(10)
    def test_server_handle_format(self):
        """Test server handle format."""
        cfg = DictConfig({"job_name": "test-job", "services": {}})
        provisioner = MastProvisioner(cfg)
        handle = provisioner.create_server_handle()
        assert handle == "mast_conda:///test-job"

    @pytest.mark.timeout(10)
    def test_build_appdef_meshes(self):
        """Test appdef builds correct mesh list."""
        cfg = DictConfig(
            {
                "services": {
                    "service1": {"num_replicas": 2, "with_gpus": True, "hosts": 1},
                    "service2": {"num_replicas": 1, "with_gpus": False, "hosts": 0},
                }
            }
        )
        provisioner = MastProvisioner(cfg)
        with patch(
            "monarch.tools.components.meta.hyperactor.host_mesh_conda"
        ) as mock_conda:
            with patch("torchx.specs.fb.component_helpers.Packages"):
                mock_appdef = MagicMock()
                mock_appdef.roles = []
                mock_conda.return_value = mock_appdef

                provisioner.build_appdef()

                # Check that mesh list was created correctly
                call_args = mock_conda.call_args[1]
                meshes = call_args["meshes"]
                assert len(meshes) == 2  # Only GPU services with hosts > 0
                assert all("service1" in mesh for mesh in meshes)

    @pytest.mark.timeout(10)
    def test_mount_directory_exists_skip(self):
        """Test mounting skips when directory already exists."""
        actor = MastSetupActor()

        with patch("os.path.exists", return_value=True):
            with patch("builtins.print") as mock_print:
                actor.mount_mnt_directory("/test/mount")
                mock_print.assert_called_once()
                assert "skip mounting" in mock_print.call_args[0][0]
