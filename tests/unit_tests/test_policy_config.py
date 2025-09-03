# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import tempfile
import unittest
from dataclasses import asdict

import yaml

from forge.actors.policy import Policy, SamplingOverrides, WorkerConfig
from vllm.engine.arg_utils import EngineArgs


class TestPolicyConfig(unittest.TestCase):
    """Test suite for Policy configuration handling after PolicyConfig removal."""

    def test_policy_default_initialization(self):
        """Test that Policy can be initialized with default values."""
        policy = Policy()

        # Check that default factories work
        self.assertIsInstance(policy.worker_params, WorkerConfig)
        self.assertIsInstance(policy.sampling_overrides, SamplingOverrides)
        self.assertIsNone(policy.available_devices)

        # Check default values
        self.assertEqual(policy.worker_params.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(policy.worker_params.tensor_parallel_size, 1)
        self.assertEqual(policy.worker_params.pipeline_parallel_size, 1)
        self.assertFalse(policy.worker_params.enforce_eager)

        self.assertEqual(policy.sampling_overrides.num_samples, 1)
        self.assertFalse(policy.sampling_overrides.guided_decoding)
        self.assertEqual(policy.sampling_overrides.max_tokens, 512)

    def test_policy_with_dict_configs(self):
        """Test Policy initialization with dictionary configs."""
        worker_dict = {
            "model": "test-model-6789",
            "tensor_parallel_size": 7777,
            "pipeline_parallel_size": 8888,
            "enforce_eager": True,
            "vllm_args": {"max_model_len": 9999, "gpu_memory_utilization": 0.1234},
        }

        sampling_dict = {
            "num_samples": 1357,
            "guided_decoding": True,
            "max_tokens": 2468,
        }

        policy = Policy(
            worker_params=worker_dict,
            sampling_overrides=sampling_dict,
            available_devices="test-gpu-device-abcd",
        )

        # Check that dictionaries were converted to proper objects
        self.assertIsInstance(policy.worker_params, WorkerConfig)
        self.assertIsInstance(policy.sampling_overrides, SamplingOverrides)

        self.assertEqual(policy.worker_params.model, "test-model-6789")
        self.assertEqual(policy.worker_params.tensor_parallel_size, 7777)
        self.assertEqual(policy.worker_params.pipeline_parallel_size, 8888)
        self.assertTrue(policy.worker_params.enforce_eager)

        self.assertEqual(policy.sampling_overrides.num_samples, 1357)
        self.assertTrue(policy.sampling_overrides.guided_decoding)
        self.assertEqual(policy.sampling_overrides.max_tokens, 2468)

    def test_policy_yaml_config_loading(self):
        """Test loading Policy configuration from YAML file."""
        yaml_content = """
        worker_params:
          model: "yaml-test-model-9876"
          tensor_parallel_size: 1234
          pipeline_parallel_size: 5678
          enforce_eager: true
          vllm_args:
            max_model_len: 9876
            gpu_memory_utilization: 0.1357

        sampling_overrides:
          num_samples: 2468
          guided_decoding: true
          max_tokens: 1357

        available_devices: "yaml-test-device-xyz"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            # Load YAML and create Policy
            with open(f.name, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            policy = Policy(**config)

            self.assertEqual(policy.worker_params.model, "yaml-test-model-9876")
            self.assertEqual(policy.worker_params.tensor_parallel_size, 1234)
            self.assertEqual(policy.worker_params.pipeline_parallel_size, 5678)
            self.assertTrue(policy.worker_params.enforce_eager)

            self.assertEqual(policy.sampling_overrides.num_samples, 2468)
            self.assertTrue(policy.sampling_overrides.guided_decoding)
            self.assertEqual(policy.sampling_overrides.max_tokens, 1357)

            self.assertEqual(policy.available_devices, "yaml-test-device-xyz")

    def test_invalid_worker_config_from_dict(self):
        """Test that WorkerConfig.from_dict handles invalid vllm_args gracefully."""
        config_dict = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "vllm_args": "invalid_string_instead_of_dict",  # This will be passed through as-is
        }

        worker_config = WorkerConfig.from_dict(config_dict)

        # The invalid vllm_args gets removed and default EngineArgs is used
        self.assertEqual(worker_config.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertIsInstance(worker_config.vllm_args, EngineArgs)


if __name__ == "__main__":
    unittest.main()
