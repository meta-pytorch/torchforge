# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import tempfile
import unittest

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.distributed.checkpoint as dcp
from forge.util._torchstore import DcpHandle, WeightCleaner

ignore_torch_distributed_unitialized_warning = pytest.mark.filterwarnings(
    r"ignore:.*torch.distributed"
)

ignore_coroutine_not_awaited = pytest.mark.filterwarnings(
    "ignore:.*coroutine.*was never awaited.*"
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


class TestWeightCleaner(unittest.IsolatedAsyncioTestCase):
    """Test suite for WeightCleaner class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cleaner = WeightCleaner()

    @ignore_coroutine_not_awaited
    def test_remove_done_tasks_with_completed_tasks(self):
        """Test _remove_done_tasks removes completed tasks."""
        # Create mock tasks - some done, some not
        done_task1 = MagicMock()
        done_task1.done.return_value = True

        done_task2 = MagicMock()
        done_task2.done.return_value = True

        pending_task = MagicMock()
        pending_task.done.return_value = False

        self.cleaner._tasks = [done_task1, pending_task, done_task2]
        self.cleaner._remove_done_tasks()

        # Only the pending task should remain
        self.assertEqual(len(self.cleaner._tasks), 1)
        self.assertEqual(self.cleaner._tasks[0], pending_task)

    @ignore_coroutine_not_awaited
    def test_remove_done_tasks_with_all_pending(self):
        """Test _remove_done_tasks with all tasks pending."""
        pending_task1 = MagicMock()
        pending_task1.done.return_value = False

        pending_task2 = MagicMock()
        pending_task2.done.return_value = False

        self.cleaner._tasks = [pending_task1, pending_task2]
        self.cleaner._remove_done_tasks()

        # All tasks should remain
        self.assertEqual(len(self.cleaner._tasks), 2)
        self.assertEqual(self.cleaner._tasks, [pending_task1, pending_task2])

    @ignore_coroutine_not_awaited
    @pytest.mark.asyncio
    @patch("forge.util._torchstore.drop_weights", new_callable=AsyncMock)
    @patch("asyncio.create_task")
    async def test_step_no_cleanup_needed_equal_version(
        self, mock_create_task, mock_drop_weights
    ):
        """Test step method when delete_up_to_version equals last_deleted_version."""
        future = asyncio.Future()
        future.set_result(None)
        mock_create_task.return_value = future

        # Step it to 5 first
        self.cleaner.step(delete_up_to_version=5)
        mock_drop_weights.assert_called()
        mock_create_task.assert_called()

        # Reset mock state to clear call history
        mock_drop_weights.reset_mock()
        mock_create_task.reset_mock()

        # Request deletion up to version 5 (already deleted)
        self.cleaner.step(delete_up_to_version=5)

        # No tasks should be created
        mock_create_task.assert_not_called()
        mock_drop_weights.assert_not_called()

    @ignore_coroutine_not_awaited
    @pytest.mark.asyncio
    @patch("forge.util._torchstore.drop_weights", new_callable=AsyncMock)
    @patch("asyncio.create_task")
    async def test_step_no_cleanup_needed_lower_version(
        self, mock_create_task, mock_drop_weights
    ):
        """Test step method when delete_up_to_version is lower than last_deleted_version."""
        future = asyncio.Future()
        future.set_result(None)
        mock_create_task.return_value = future

        # Step it to 10 first
        self.cleaner.step(delete_up_to_version=10)

        # Reset mock state to clear call history
        mock_drop_weights.reset_mock()
        mock_create_task.reset_mock()

        # Request deletion up to version 5 (lower than already deleted)
        self.cleaner.step(delete_up_to_version=5)

        # No tasks should be created
        mock_create_task.assert_not_called()
        mock_drop_weights.assert_not_called()

    @ignore_coroutine_not_awaited
    @pytest.mark.asyncio
    @patch("forge.util._torchstore.drop_weights", new_callable=AsyncMock)
    @patch("asyncio.create_task")
    async def test_step_creates_tasks_initial_call(
        self, mock_create_task, mock_drop_weights
    ):
        """Test step method creates tasks for entire version range."""

        future = asyncio.Future()
        future.set_result(None)
        mock_create_task.return_value = future

        # Request deletion up to version 5 from initial state
        self.cleaner.step(delete_up_to_version=5)

        # Should create 6 tasks (versions 0 through 5)
        self.assertEqual(mock_create_task.call_count, 6)

    @ignore_coroutine_not_awaited
    @pytest.mark.asyncio
    @patch("forge.util._torchstore.drop_weights", new_callable=AsyncMock)
    @patch("asyncio.create_task")
    async def test_step_creates_only_new_version_tasks(
        self, mock_create_task, mock_drop_weights
    ):
        """Test step method only creates tasks for versions not yet deleted."""

        future = asyncio.Future()
        future.set_result(None)
        mock_create_task.return_value = future

        # First deletion up to version 3
        self.cleaner.step(delete_up_to_version=3)

        # Reset mock to track only new calls
        mock_create_task.reset_mock()

        # Second deletion up to version 7
        self.cleaner.step(delete_up_to_version=7)

        # Should only create tasks for versions 4, 5, 6, 7
        self.assertEqual(mock_create_task.call_count, 4)
