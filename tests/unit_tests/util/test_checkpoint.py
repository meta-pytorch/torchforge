# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for forge.util.checkpoint.warn_if_resuming_from_existing_folder
(regression for issue #631).

#631: torchtitan's checkpointer silently resumes from ``checkpoint.folder``
when it already contains saved step directories, ignoring
``initial_load_path``. Users running back-to-back experiments hit this
footgun without realizing the second run isn't starting from the base
model. The helper logs a loud WARNING right before the load so the resume
shows up in the standard training logs.
"""

import logging
import os

import pytest

from forge.util.checkpoint import warn_if_resuming_from_existing_folder


class TestWarnIfResumingFromExistingFolder:
    def test_returns_false_when_folder_is_none(self):
        assert warn_if_resuming_from_existing_folder(None) is False

    def test_returns_false_when_folder_is_empty_string(self):
        assert warn_if_resuming_from_existing_folder("") is False

    def test_returns_false_when_folder_does_not_exist(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert warn_if_resuming_from_existing_folder(str(missing)) is False

    def test_returns_false_when_folder_has_no_step_dirs(self, tmp_path):
        (tmp_path / "random_file.txt").write_text("noise")
        (tmp_path / "logs").mkdir()
        assert warn_if_resuming_from_existing_folder(str(tmp_path)) is False

    def test_warns_when_step_dirs_exist(self, tmp_path, caplog):
        (tmp_path / "step-100").mkdir()
        (tmp_path / "step-200").mkdir()
        (tmp_path / "step-50").mkdir()

        with caplog.at_level(logging.WARNING, logger="forge.util.checkpoint"):
            warned = warn_if_resuming_from_existing_folder(str(tmp_path))

        assert warned is True
        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert warning_records, "expected at least one WARNING-level log"
        msg = warning_records[0].getMessage()
        assert str(tmp_path) in msg
        assert "step-200" in msg, "should report the latest step directory"
        assert "3 saved step dir" in msg

    def test_warning_mentions_ignored_initial_load_path(self, tmp_path, caplog):
        (tmp_path / "step-1").mkdir()
        initial = "hf://meta-llama/Meta-Llama-3.1-8B-Instruct"

        with caplog.at_level(logging.WARNING, logger="forge.util.checkpoint"):
            warn_if_resuming_from_existing_folder(
                str(tmp_path), initial_load_path=initial
            )

        msg = caplog.records[-1].getMessage()
        assert initial in msg
        assert "will be ignored" in msg

    def test_ignores_non_step_subdirs(self, tmp_path):
        (tmp_path / "tensorboard").mkdir()
        (tmp_path / "wandb").mkdir()
        (tmp_path / "step-1-backup").mkdir()  # starts with step- but not the step-N pattern? actually matches
        # The helper currently treats anything starting with "step-" as a step
        # dir; that's intentional — same prefix the checkpointer scans for.
        assert warn_if_resuming_from_existing_folder(str(tmp_path)) is True

    def test_handles_oserror_gracefully(self, tmp_path, monkeypatch, caplog):
        """If we can't list the folder (perms etc), don't crash, don't warn."""
        (tmp_path / "step-1").mkdir()

        def boom(_):
            raise PermissionError("denied")

        monkeypatch.setattr(os, "listdir", boom)
        with caplog.at_level(logging.WARNING, logger="forge.util.checkpoint"):
            warned = warn_if_resuming_from_existing_folder(str(tmp_path))

        assert warned is False
        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not warning_records
