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
        import forge.actors.generator  # noqa: F401

        return False
    except ImportError:
        return True


class TestGeneratorConfig(unittest.TestCase):
    """Test suite for Generator configuration handling after PolicyConfig removal."""

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_generator_default_initialization(self):
        """Generator initializes with default values."""
        from forge.actors.generator import EngineConfig, Generator, SamplingConfig

        generator = Generator()

        # Default factories
        self.assertIsInstance(generator.engine_config, EngineConfig)
        self.assertIsInstance(generator.sampling_config, SamplingConfig)
        self.assertIsNone(generator.available_devices)

        # Worker defaults
        self.assertEqual(
            generator.engine_config.model, "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.assertEqual(generator.engine_config.tensor_parallel_size, 1)
        self.assertEqual(generator.engine_config.pipeline_parallel_size, 1)
        self.assertFalse(generator.engine_config.enforce_eager)
        self.assertTrue(generator.engine_config._is_v1_supported_oracle())

        # Sampling defaults
        self.assertEqual(generator.sampling_config.n, 1)
        self.assertFalse(generator.sampling_config.guided_decoding)
        self.assertEqual(generator.sampling_config.max_tokens, 512)

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_generator_with_dict_configs(self):
        """Generator accepts dicts for engine_config and sampling_config, including nested dicts."""
        from forge.actors.generator import EngineConfig, Generator, SamplingConfig

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

        generator = Generator(
            engine_config=engine_dict,
            sampling_config=sampling_dict,
            available_devices="test-gpu-device-abcd",
        )

        self.assertIsInstance(generator.engine_config, EngineConfig)
        self.assertIsInstance(generator.sampling_config, SamplingConfig)

        # Test basic fields
        self.assertEqual(generator.engine_config.model, "test-model-6789")
        self.assertEqual(generator.engine_config.tensor_parallel_size, 7777)
        self.assertEqual(generator.engine_config.pipeline_parallel_size, 8888)
        self.assertTrue(generator.engine_config.enforce_eager)
        self.assertTrue(generator.engine_config._is_v1_supported_oracle())

        self.assertEqual(generator.sampling_config.n, 1357)
        # After __post_init__, guided_decoding becomes GuidedDecodingParams object when True
        self.assertIsNotNone(generator.sampling_config.guided_decoding)
        self.assertEqual(generator.sampling_config.max_tokens, 2468)

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
    def test_generator_yaml_config_loading(self):
        """Generator can be constructed from a YAML config file."""
        from forge.actors.generator import Generator

        yaml_content = """
        engine_config:
          model: "yaml-test-model-9876"
          tensor_parallel_size: 1234
          pipeline_parallel_size: 5678
          enforce_eager: true

        sampling_config:
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

            generator = Generator(**config)

            self.assertEqual(generator.engine_config.model, "yaml-test-model-9876")
            self.assertEqual(generator.engine_config.tensor_parallel_size, 1234)
            self.assertEqual(generator.engine_config.pipeline_parallel_size, 5678)
            self.assertTrue(generator.engine_config.enforce_eager)
            self.assertTrue(generator.engine_config._is_v1_supported_oracle())

            self.assertEqual(generator.sampling_config.n, 2468)
            # After __post_init__, guided_decoding becomes GuidedDecodingParams object when True
            self.assertIsNotNone(generator.sampling_config.guided_decoding)
            self.assertEqual(generator.sampling_config.max_tokens, 1357)

            self.assertEqual(generator.available_devices, "yaml-test-device-xyz")

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_engineconfig_ignores_invalid_keys(self):
        """EngineConfig.from_dict ignores unexpected keys."""
        from forge.actors.generator import EngineConfig

        engine_config = {
            "model": "custom-model",
            "tensor_parallel_size": 2,
            "invalid_key_123": "should be ignored",
        }

        config = EngineConfig.from_dict(engine_config)

        self.assertEqual(config.model, "custom-model")
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertFalse(hasattr(config, "invalid_key_123"))


if __name__ == "__main__":
    unittest.main()
