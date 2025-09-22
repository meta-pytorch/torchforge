# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

from forge.actors.trainer import cleanup_old_weight_versions


class TestTrainerUtilities(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def test_cleanup_old_weight_versions_basic(self):
        """Test basic cleanup functionality - keeps current and N-1 versions."""
        # Create test directory structure
        state_dict_key = os.path.join(self.test_dir, "model")
        delim = "__"

        # Create some mock weight directories
        old_version_1 = f"{state_dict_key}{delim}1"
        previous_version = f"{state_dict_key}{delim}2"  # N-1 version
        current_version = f"{state_dict_key}{delim}3"  # Current version
        unrelated_dir = os.path.join(self.test_dir, "other_model__1")

        for dir_path in [
            old_version_1,
            previous_version,
            current_version,
            unrelated_dir,
        ]:
            os.makedirs(dir_path)

        # Run cleanup for version 3
        cleanup_old_weight_versions(
            state_dict_key=state_dict_key,
            delim=delim,
            current_policy_version=3,
        )

        # Check that only very old versions were deleted (version 1)
        self.assertFalse(os.path.exists(old_version_1))

        # Check that current and previous versions still exist
        self.assertTrue(os.path.exists(previous_version))  # N-1 version should remain
        self.assertTrue(
            os.path.exists(current_version)
        )  # Current version should remain
        self.assertTrue(os.path.exists(unrelated_dir))  # Unrelated dirs should remain

    def test_cleanup_old_weight_versions_no_cleanup_version_1(self):
        """Test that no cleanup happens when current_policy_version <= 1."""
        # Create test directory structure
        state_dict_key = os.path.join(self.test_dir, "model")
        delim = "__"

        version_1 = f"{state_dict_key}{delim}1"
        os.makedirs(version_1)

        # Run cleanup for version 1 - should do nothing
        cleanup_old_weight_versions(
            state_dict_key=state_dict_key,
            delim=delim,
            current_policy_version=1,
        )

        # Version 1 should still exist
        self.assertTrue(os.path.exists(version_1))

    def test_cleanup_old_weight_versions_version_2(self):
        """Test cleanup with version 2 as current - should keep versions 1 and 2."""
        # Create test directory structure
        state_dict_key = os.path.join(self.test_dir, "model")
        delim = "__"

        version_1 = f"{state_dict_key}{delim}1"  # N-1 version
        version_2 = f"{state_dict_key}{delim}2"  # Current version

        for dir_path in [version_1, version_2]:
            os.makedirs(dir_path)

        # Run cleanup for version 2
        cleanup_old_weight_versions(
            state_dict_key=state_dict_key,
            delim=delim,
            current_policy_version=2,
        )

        # Both versions should still exist (no deletion for version 2)
        self.assertTrue(os.path.exists(version_1))
        self.assertTrue(os.path.exists(version_2))


if __name__ == "__main__":
    unittest.main()
