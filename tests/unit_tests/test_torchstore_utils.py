# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from forge.actors._torchstore_utils import (
    extract_param_name,
    get_param_key,
    get_param_prefix,
)


class TestTorchStoreUtils(unittest.TestCase):
    def test_get_param_prefix(self) -> None:
        self.assertEqual(get_param_prefix(1), "policy_ver_0000000001")

    def test_get_param_key(self) -> None:
        self.assertEqual(get_param_key(1, "test"), "policy_ver_0000000001.test")

    def test_extract_param_name(self) -> None:
        self.assertEqual(extract_param_name("policy_ver_0000000001.test"), "test")
