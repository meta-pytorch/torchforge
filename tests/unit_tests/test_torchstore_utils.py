# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

from pathlib import Path

import pytest

import torch
import torch.distributed.checkpoint as dcp
from forge.util._torchstore import DcpHandle

ignore_torch_distributed_unitialized_warning = pytest.mark.filterwarnings(
    r"ignore:.*torch.distributed"
)


class TestDcpHandle(unittest.TestCase):
    def _prepare_dcp_handle(self, test_dir: str) -> tuple[str, DcpHandle]:
        """Returns path to checkpoint and DcpHandle."""
        checkpoint_id = str(Path(test_dir) / "test_checkpoint_id")
        state_dict = {"a": torch.rand(1, 1), "b": torch.rand(1, 1)}
        metadata = dcp.save(checkpoint_id=checkpoint_id, state_dict=state_dict)
        assert os.path.exists(checkpoint_id), "failed to set up test checkpoint"
        return checkpoint_id, DcpHandle(
            checkpoint_id=checkpoint_id,
            metadata=metadata,
            param_names=list(state_dict.keys()),
        )

    @ignore_torch_distributed_unitialized_warning
    def test_dcp_handle_drop_deletes(self):
        with tempfile.TemporaryDirectory() as test_dir:
            ckpt_path, handle = self._prepare_dcp_handle(test_dir)
            handle.drop()
            self.assertFalse(os.path.exists(ckpt_path))

    @ignore_torch_distributed_unitialized_warning
    def test_dcp_handle_drop_sets_none(self):
        with tempfile.TemporaryDirectory() as test_dir:
            _, handle = self._prepare_dcp_handle(test_dir)
            handle.drop()
            self.assertEqual(handle.checkpoint_id, None)
            self.assertEqual(handle.metadata, None)
            self.assertEqual(handle.param_names, None)

    @ignore_torch_distributed_unitialized_warning
    def test_dcp_handle_drop_sets_none_for_manifold(self):
        with tempfile.TemporaryDirectory() as test_dir:
            _, handle = self._prepare_dcp_handle(test_dir)
            handle.checkpoint_id = "manifold://test_bucket/tree/test_path"
            handle.drop()
            self.assertEqual(handle.checkpoint_id, None)
            self.assertEqual(handle.metadata, None)
            self.assertEqual(handle.param_names, None)
