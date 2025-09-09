# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import yaml

from forge.actors.policy import EngineConfig, Policy, SamplingConfig


class TestPolicyConfig(unittest.TestCase):
    """Test suite for Policy configuration handling after PolicyConfig removal."""

    def test_policy_default_initialization(self):
        """Policy initializes with default values."""
        policy = Policy()

        # Default factories
        self.assertIsInstance(policy.engine_params, EngineConfig)
        self.assertIsInstance(policy.sampling_overrides, SamplingConfig)
        self.assertIsNone(policy.available_devices)

        # Worker defaults
        self.assertEqual(policy.engine_params.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(policy.engine_params.tensor_parallel_size, 1)
        self.assertEqual(policy.engine_params.pipeline_parallel_size, 1)
        self.assertFalse(policy.engine_params.enforce_eager)

        # Sampling defaults
        self.assertEqual(policy.sampling_overrides.n, 1)
        self.assertFalse(policy.sampling_overrides.guided_decoding)
        self.assertEqual(policy.sampling_overrides.max_tokens, 512)

    def test_policy_with_dict_configs(self):
        """Policy accepts dicts for engine_params and sampling_overrides, including nested dicts."""
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
            "guided_decoding": True,
            "max_tokens": 2468,
        }

        policy = Policy(
            engine_params=engine_dict,
            sampling_overrides=sampling_dict,
            available_devices="test-gpu-device-abcd",
        )

        self.assertIsInstance(policy.engine_params, EngineConfig)
        self.assertIsInstance(policy.sampling_overrides, SamplingConfig)

        # Test basic fields
        self.assertEqual(policy.engine_params.model, "test-model-6789")
        self.assertEqual(policy.engine_params.tensor_parallel_size, 7777)
        self.assertEqual(policy.engine_params.pipeline_parallel_size, 8888)
        self.assertTrue(policy.engine_params.enforce_eager)

        self.assertEqual(policy.sampling_overrides.n, 1357)
        # After __post_init__, guided_decoding becomes GuidedDecodingParams object when True
        self.assertIsNotNone(policy.sampling_overrides.guided_decoding)
        self.assertEqual(policy.sampling_overrides.max_tokens, 2468)

        # Test that engine_dict accepts and preserves nested dict structure
        # The original engine_dict should remain unchanged and accessible
        self.assertIn("nested_config", engine_dict)
        self.assertEqual(engine_dict["nested_config"]["gpu_memory_utilization"], 0.9)
        self.assertEqual(engine_dict["nested_config"]["max_model_len"], 4096)
        self.assertEqual(
            engine_dict["nested_config"]["custom_settings"]["temperature"], 0.7
        )
        self.assertEqual(engine_dict["nested_config"]["custom_settings"]["top_p"], 0.9)

    def test_policy_yaml_config_loading(self):
        """Policy can be constructed from a YAML config file."""
        yaml_content = """
        engine_params:
          model: "yaml-test-model-9876"
          tensor_parallel_size: 1234
          pipeline_parallel_size: 5678
          enforce_eager: true

        sampling_overrides:
          n: 2468
          guided_decoding: true
          max_tokens: 1357

        available_devices: "yaml-test-device-xyz"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with open(f.name, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            policy = Policy(**config)

            self.assertEqual(policy.engine_params.model, "yaml-test-model-9876")
            self.assertEqual(policy.engine_params.tensor_parallel_size, 1234)
            self.assertEqual(policy.engine_params.pipeline_parallel_size, 5678)
            self.assertTrue(policy.engine_params.enforce_eager)

            self.assertEqual(policy.sampling_overrides.n, 2468)
            # After __post_init__, guided_decoding becomes GuidedDecodingParams object when True
            self.assertIsNotNone(policy.sampling_overrides.guided_decoding)
            self.assertEqual(policy.sampling_overrides.max_tokens, 1357)

            self.assertEqual(policy.available_devices, "yaml-test-device-xyz")

    def test_engineconfig_ignores_invalid_keys(self):
        """EngineConfig.from_dict ignores unexpected keys."""
        engine_params = {
            "model": "custom-model",
            "tensor_parallel_size": 2,
            "invalid_key_123": "should be ignored",
        }

        config = EngineConfig.from_dict(engine_params)

        self.assertEqual(config.model, "custom-model")
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertFalse(hasattr(config, "invalid_key_123"))


if __name__ == "__main__":
    unittest.main()
