# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest


class TestGeneratorVersion(unittest.TestCase):
    """Test suite for Generator version detection."""

    def test_generator_version_detection(self):
        """Test that the correct Generator version is loaded based on vLLM version."""
        import vllm
        from forge.actors.generator import Generator

        module_name = Generator.__module__

        if vllm.__version__ >= "0.13.0":
            expected = "v1"
        else:
            expected = "v0"

        print(f"\nvLLM version: {vllm.__version__}")
        print(f"Expected Generator: {expected}")
        print(f"Actual Generator module: {module_name}")

        if vllm.__version__ >= "0.13.0":
            self.assertTrue(
                ".v1." in module_name or module_name.endswith(".v1"),
                f"Expected v1 for vLLM {vllm.__version__}, got {module_name}",
            )
        else:
            self.assertTrue(
                ".v0." in module_name or module_name.endswith(".v0"),
                f"Expected v0 for vLLM {vllm.__version__}, got {module_name}",
            )

    def test_generator_has_required_methods(self):
        """Test that Generator has all required methods."""
        from forge.actors.generator import Generator

        required_methods = ["launch", "setup", "generate", "shutdown"]

        for method in required_methods:
            self.assertTrue(
                hasattr(Generator, method), f"Generator missing method: {method}"
            )


if __name__ == "__main__":
    unittest.main()
