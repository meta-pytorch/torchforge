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
        from forge.actors.generator import Generator
        from vllm.engine.arg_utils import EngineArgs
        from vllm.sampling_params import SamplingParams

        generator = Generator()

        # Default factories
        self.assertIsInstance(generator.engine_args, EngineArgs)
        self.assertIsInstance(generator.sampling_params, SamplingParams)

        # Worker defaults
        self.assertEqual(generator.engine_args.model, "Qwen/Qwen3-0.6B")
        self.assertEqual(generator.engine_args.tensor_parallel_size, 1)
        self.assertEqual(generator.engine_args.pipeline_parallel_size, 1)
        self.assertFalse(generator.engine_args.enforce_eager)
        self.assertTrue(generator.engine_args._is_v1_supported_oracle())

        # Sampling defaults
        self.assertEqual(generator.sampling_params.n, 1)
        self.assertFalse(generator.sampling_params.guided_decoding)
        self.assertEqual(generator.sampling_params.max_tokens, 16)

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_generator_with_dict_configs(self):
        """Generator accepts dicts for engine_config and sampling_config, including nested dicts."""
        from forge.actors.generator import Generator
        from vllm.engine.arg_utils import EngineArgs
        from vllm.sampling_params import SamplingParams

        # Test with nested dict structure
        engine_dict = {
            "model": "test-model-6789",
            "tensor_parallel_size": 7777,
            "pipeline_parallel_size": 8888,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        }

        sampling_dict = {
            "n": 1357,
            "max_tokens": 2468,
        }

        generator = Generator(
            engine_args=engine_dict,
            sampling_params=sampling_dict,
        )

        self.assertIsInstance(generator.engine_args, EngineArgs)
        self.assertIsInstance(generator.sampling_params, SamplingParams)

        # Test basic fields
        self.assertEqual(generator.engine_args.model, "test-model-6789")
        self.assertEqual(generator.engine_args.tensor_parallel_size, 7777)
        self.assertEqual(generator.engine_args.pipeline_parallel_size, 8888)
        self.assertEqual(generator.engine_args.gpu_memory_utilization, 0.9)
        self.assertEqual(generator.engine_args.max_model_len, 4096)
        self.assertTrue(generator.engine_args.enforce_eager)
        self.assertTrue(generator.engine_args._is_v1_supported_oracle())

        self.assertEqual(generator.sampling_params.n, 1357)
        self.assertEqual(generator.sampling_params.max_tokens, 2468)

    @pytest.mark.skipif(
        _import_error(),
        reason="Import error, likely due to missing dependencies on CI.",
    )
    def test_generator_yaml_config_loading(self):
        """Generator can be constructed from a YAML config file."""
        from forge.actors.generator import Generator

        yaml_content = """
        engine_args:
          model: "yaml-test-model-9876"
          tensor_parallel_size: 1234
          pipeline_parallel_size: 5678
          enforce_eager: true

        sampling_params:
          n: 2468
          max_tokens: 1357
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with open(f.name, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            generator = Generator(**config)
            self.assertEqual(generator.engine_args.model, "yaml-test-model-9876")
            self.assertEqual(generator.engine_args.tensor_parallel_size, 1234)
            self.assertEqual(generator.engine_args.pipeline_parallel_size, 5678)
            self.assertTrue(generator.engine_args.enforce_eager)
            self.assertTrue(generator.engine_args._is_v1_supported_oracle())

            self.assertEqual(generator.sampling_params.n, 2468)
            self.assertEqual(generator.sampling_params.max_tokens, 1357)


if __name__ == "__main__":
    unittest.main()
