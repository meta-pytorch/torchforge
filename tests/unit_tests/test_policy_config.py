# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import pytest
import yaml


def _import_error():
    """Check if there are import errors that would cause CI failures."""
    try:
        import forge.actors.policy  # noqa: F401

        return False
    except ImportError:
        return True


class TestPolicyConfig(unittest.TestCase):
    """Test suite for Policy configuration handling after PolicyConfig removal."""

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_policy_default_initialization(self):
        """Policy initializes with default values."""
        from forge.actors.policy import Policy
        from vllm.engine.arg_utils import EngineArgs
        from vllm.sampling_params import SamplingParams

        policy = Policy()

        # Default factories
        self.assertIsInstance(policy.engine_args, EngineArgs)
        self.assertIsInstance(policy.sampling_params, SamplingParams)
        self.assertIsNone(policy.available_devices)

        # Worker defaults
        self.assertEqual(policy.engine_args.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(policy.engine_args.tensor_parallel_size, 1)
        self.assertEqual(policy.engine_args.pipeline_parallel_size, 1)
        self.assertFalse(policy.engine_args.enforce_eager)
        self.assertTrue(policy.engine_args._is_v1_supported_oracle())

        # Sampling defaults
        self.assertEqual(policy.sampling_params.n, 1)
        self.assertFalse(policy.sampling_params.guided_decoding)
        self.assertEqual(policy.sampling_params.max_tokens, 512)

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_policy_with_dict_configs(self):
        """Policy accepts dicts for engine_args and sampling_params, including nested dicts."""
        from forge.actors.policy import Policy
        from vllm.engine.arg_utils import EngineArgs
        from vllm.sampling_params import SamplingParams

        # Test with nested dict structure
        engine_dict = {
            "model": "test-model-6789",
            "tensor_parallel_size": 7777,
            "pipeline_parallel_size": 8888,
            "enforce_eager": True,
            "nested_config": {
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
                "custom_settings": {"temperature": 0.7, "top_p": 0.9},
            },
        }

        sampling_dict = {
            "n": 1357,
            "max_tokens": 2468,
        }

        policy = Policy(
            engine_args=engine_dict,
            sampling_params=sampling_dict,
            available_devices="test-gpu-device-abcd",
        )

        self.assertIsInstance(policy.engine_args, EngineArgs)
        self.assertIsInstance(policy.sampling_params, SamplingParams)

        # Test basic fields
        self.assertEqual(policy.engine_args.model, "test-model-6789")
        self.assertEqual(policy.engine_args.tensor_parallel_size, 7777)
        self.assertEqual(policy.engine_args.pipeline_parallel_size, 8888)
        self.assertTrue(policy.engine_args.enforce_eager)
        self.assertTrue(policy.engine_args._is_v1_supported_oracle())

        self.assertEqual(policy.sampling_params.n, 1357)
        self.assertEqual(policy.sampling_params.max_tokens, 2468)

        # Test that engine_dict accepts and preserves nested dict structure
        # The original engine_dict should remain unchanged and accessible
        self.assertIn("nested_config", engine_dict)
        self.assertEqual(engine_dict["nested_config"]["gpu_memory_utilization"], 0.9)
        self.assertEqual(engine_dict["nested_config"]["max_model_len"], 4096)
        self.assertEqual(
            engine_dict["nested_config"]["custom_settings"]["temperature"], 0.7
        )
        self.assertEqual(engine_dict["nested_config"]["custom_settings"]["top_p"], 0.9)

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_policy_yaml_config_loading(self):
        """Policy can be constructed from a YAML config file."""
        from forge.actors.policy import Policy

        yaml_content = """
        engine_args:
          model: "yaml-test-model-9876"
          tensor_parallel_size: 1234
          pipeline_parallel_size: 5678
          enforce_eager: true

        sampling_params:
          n: 2468
          max_tokens: 1357

        available_devices: "yaml-test-device-xyz"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with open(f.name, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            policy = Policy(**config)

            self.assertEqual(policy.engine_args.model, "yaml-test-model-9876")
            self.assertEqual(policy.engine_args.tensor_parallel_size, 1234)
            self.assertEqual(policy.engine_args.pipeline_parallel_size, 5678)
            self.assertTrue(policy.engine_args.enforce_eager)
            self.assertTrue(policy.engine_args._is_v1_supported_oracle())

            self.assertEqual(policy.sampling_params.n, 2468)
            self.assertEqual(policy.sampling_params.max_tokens, 1357)

            self.assertEqual(policy.available_devices, "yaml-test-device-xyz")


if __name__ == "__main__":
    unittest.main()
