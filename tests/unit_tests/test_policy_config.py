import tempfile
import unittest

import yaml

from forge.actors.policy import Policy, SamplingOverrides, WorkerConfig


class TestPolicyConfig(unittest.TestCase):
    """Test suite for Policy configuration handling after PolicyConfig removal."""

    def test_policy_default_initialization(self):
        """Policy initializes with default values."""
        policy = Policy()

        # Default factories
        self.assertIsInstance(policy.worker_params, WorkerConfig)
        self.assertIsInstance(policy.sampling_overrides, SamplingOverrides)
        self.assertIsNone(policy.available_devices)

        # Worker defaults
        self.assertEqual(policy.worker_params.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(policy.worker_params.tensor_parallel_size, 1)
        self.assertEqual(policy.worker_params.pipeline_parallel_size, 1)
        self.assertFalse(policy.worker_params.enforce_eager)

        # Sampling defaults
        self.assertEqual(policy.sampling_overrides.num_samples, 1)
        self.assertFalse(policy.sampling_overrides.guided_decoding)
        self.assertEqual(policy.sampling_overrides.max_tokens, 512)

    def test_policy_with_dict_configs(self):
        """Policy accepts dicts for worker_params and sampling_overrides."""
        worker_dict = {
            "model": "test-model-6789",
            "tensor_parallel_size": 7777,
            "pipeline_parallel_size": 8888,
            "enforce_eager": True,
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
        """Policy can be constructed from a YAML config file."""
        yaml_content = """
        worker_params:
          model: "yaml-test-model-9876"
          tensor_parallel_size: 1234
          pipeline_parallel_size: 5678
          enforce_eager: true

        sampling_overrides:
          num_samples: 2468
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

            self.assertEqual(policy.worker_params.model, "yaml-test-model-9876")
            self.assertEqual(policy.worker_params.tensor_parallel_size, 1234)
            self.assertEqual(policy.worker_params.pipeline_parallel_size, 5678)
            self.assertTrue(policy.worker_params.enforce_eager)

            self.assertEqual(policy.sampling_overrides.num_samples, 2468)
            self.assertTrue(policy.sampling_overrides.guided_decoding)
            self.assertEqual(policy.sampling_overrides.max_tokens, 1357)

            self.assertEqual(policy.available_devices, "yaml-test-device-xyz")

    def test_workerconfig_ignores_invalid_keys(self):
        """WorkerConfig.from_dict ignores unexpected keys."""
        worker_dict = {
            "model": "custom-model",
            "tensor_parallel_size": 2,
            "invalid_key_123": "should be ignored",
        }

        config = WorkerConfig.from_dict(worker_dict)

        self.assertEqual(config.model, "custom-model")
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertFalse(hasattr(config, "invalid_key_123"))


if __name__ == "__main__":
    unittest.main()
