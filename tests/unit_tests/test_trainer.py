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
        """Test basic cleanup functionality."""
        # Create test directory structure
        state_dict_key = os.path.join(self.test_dir, "model")
        delim = "__"

        # Create some mock weight directories
        old_version_1 = f"{state_dict_key}{delim}1"
        old_version_2 = f"{state_dict_key}{delim}2"
        current_version = f"{state_dict_key}{delim}3"
        unrelated_dir = os.path.join(self.test_dir, "other_model__1")

        for dir_path in [old_version_1, old_version_2, current_version, unrelated_dir]:
            os.makedirs(dir_path)

        # Run cleanup for version 3
        cleanup_old_weight_versions(
            state_dict_key=state_dict_key,
            delim=delim,
            current_policy_version=3,
        )

        # Check that old versions were deleted
        self.assertFalse(os.path.exists(old_version_1))
        self.assertFalse(os.path.exists(old_version_2))

        # Check that current version and unrelated directories still exist
        self.assertTrue(os.path.exists(current_version))
        self.assertTrue(os.path.exists(unrelated_dir))

    def test_cleanup_old_weight_versions_os_error(self):
        """Test error handling when deletion fails."""
        # Create test directory structure
        state_dict_key = os.path.join(self.test_dir, "model")
        delim = "__"

        old_version = f"{state_dict_key}{delim}1"
        current_version = f"{state_dict_key}{delim}2"

        os.makedirs(old_version)
        os.makedirs(current_version)

        # Make the old version directory read-only to simulate deletion failure
        os.chmod(old_version, 0o444)

        # Run cleanup
        cleanup_old_weight_versions(
            state_dict_key=state_dict_key,
            delim=delim,
            current_policy_version=2,
        )
        # Clean up by restoring permissions
        if os.path.exists(old_version):
            os.chmod(old_version, 0o755)


if __name__ == "__main__":
    unittest.main()
