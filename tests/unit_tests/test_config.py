# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest

from forge.util.config import resolve_hf_hub_paths
from omegaconf import DictConfig, OmegaConf


# Core functionality tests
@pytest.mark.parametrize(
    "config_data,expected_calls",
    [
        # Simple hf:// path
        ({"model": "hf://meta-llama/Llama-2-7b-hf"}, [("meta-llama/Llama-2-7b-hf",)]),
        # Nested hf:// paths
        (
            {
                "model": {"pretrained": "hf://meta-llama/Llama-2-7b-hf"},
                "tokenizer": "hf://microsoft/DialoGPT-medium",
                "training": {"epochs": 10},
            },
            [("meta-llama/Llama-2-7b-hf",), ("microsoft/DialoGPT-medium",)],
        ),
        # hf:// in lists and tuples
        (
            {
                "models": ["hf://model1", "local/path", "hf://model2"],
                "tuple_data": ("hf://model3", "another/local/path"),
            },
            [("model1",), ("model2",), ("model3",)],
        ),
        # Deeply nested structure
        ({"level1": {"level2": {"model": "hf://deep/model"}}}, [("deep/model",)]),
    ],
)
@patch("forge.util.config.snapshot_download")
def test_hf_path_resolution(mock_download, config_data, expected_calls):
    """Test hf:// path resolution in various data structures."""
    mock_download.return_value = "/fake/cache/model"

    config = OmegaConf.create(config_data)
    result = resolve_hf_hub_paths(config)

    # Verify correct number of calls
    assert mock_download.call_count == len(expected_calls)

    # Verify each call was made with correct parameters
    for (repo_name,) in expected_calls:
        mock_download.assert_any_call(repo_name, revision="main", local_files_only=True)

    # Verify result is DictConfig
    assert isinstance(result, DictConfig)


@pytest.mark.parametrize(
    "config_data",
    [
        {"model": "local/path/to/model"},
        {"model": "/absolute/path", "tokenizer": "relative/path"},
        {"model": "https://example.com/model"},
        {"models": ["local1", "local2"], "other": "value"},
        {},  # Empty config
    ],
)
def test_non_hf_paths_unchanged(config_data):
    """Test that non-hf:// paths are left unchanged."""
    config = OmegaConf.create(config_data)
    result = resolve_hf_hub_paths(config)

    # Result should be identical to input for non-hf paths
    assert OmegaConf.to_container(result) == config_data


# Cache behavior tests
@patch("forge.util.config.snapshot_download")
def test_cache_hit_scenario(mock_download):
    """Test behavior when model is already cached."""
    mock_download.return_value = "/fake/cache/model"

    config = OmegaConf.create({"model": "hf://test/model"})
    result = resolve_hf_hub_paths(config)

    # Should call with local_files_only=True and succeed
    mock_download.assert_called_once_with(
        "test/model", revision="main", local_files_only=True
    )
    assert result.model == "/fake/cache/model"


@patch("forge.util.config.snapshot_download")
def test_cache_miss_scenario(mock_download):
    """Test behavior when model is not cached."""
    from huggingface_hub.utils import LocalEntryNotFoundError

    # First call fails (cache miss), second succeeds (download)
    mock_download.side_effect = [
        LocalEntryNotFoundError("Not in cache"),
        "/fake/cache/model",
    ]

    config = OmegaConf.create({"model": "hf://test/model"})
    result = resolve_hf_hub_paths(config)

    # Should call twice: first with local_files_only=True, then False
    assert mock_download.call_count == 2
    mock_download.assert_any_call("test/model", revision="main", local_files_only=True)
    mock_download.assert_any_call("test/model", revision="main", local_files_only=False)
    assert result.model == "/fake/cache/model"


# Error handling tests
@pytest.mark.parametrize(
    "invalid_input,expected_error",
    [
        (None, "Configuration cannot be None"),
        ({"model": "hf://test"}, "Input must be an OmegaConf config object"),
    ],
)
def test_input_validation(invalid_input, expected_error):
    """Test input validation with various invalid inputs."""
    with pytest.raises(ValueError) as exc_info:
        resolve_hf_hub_paths(invalid_input)
    assert expected_error in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_hf_url,expected_error",
    [
        ("hf://", "Empty repository name"),  # Empty repo name
        ("hf:///invalid", "Failed to resolve HuggingFace model"),  # Invalid repo format
    ],
)
def test_invalid_hf_urls(invalid_hf_url, expected_error):
    """Test handling of invalid hf:// URLs."""
    config = OmegaConf.create({"model": invalid_hf_url})

    with pytest.raises((ValueError, Exception)) as exc_info:
        resolve_hf_hub_paths(config)
    assert expected_error in str(exc_info.value)


@patch("forge.util.config.snapshot_download")
def test_download_failure_handling(mock_download):
    """Test error handling when download fails."""
    mock_download.side_effect = Exception("Network error: Repository not found")

    config = OmegaConf.create({"model": "hf://invalid/repo"})

    with pytest.raises(Exception) as exc_info:
        resolve_hf_hub_paths(config)
    assert "Failed to resolve HuggingFace model 'invalid/repo'" in str(exc_info.value)
    assert "Network error" in str(exc_info.value)


# Integration test with mixed data types
@patch("forge.util.config.snapshot_download")
def test_complex_real_world_config(mock_download):
    """Test with a realistic complex configuration."""
    mock_download.return_value = "/fake/cache/model"

    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model": "hf://meta-llama/Llama-2-7b-hf",
                "lora_rank": 64,
                "use_cache": True,
            },
            "tokenizer": "hf://meta-llama/Llama-2-7b-hf",  # Same repo
            "training": {"batch_size": 32, "learning_rate": 0.0001, "epochs": 10},
            "output_dir": "/local/output",
            "resume_from": None,
        }
    )

    result = resolve_hf_hub_paths(config)

    # Should call download twice (same repo referenced twice)
    assert mock_download.call_count == 2
    mock_download.assert_any_call(
        "meta-llama/Llama-2-7b-hf", revision="main", local_files_only=True
    )

    # Verify hf:// paths were replaced
    assert result.model.pretrained_model == "/fake/cache/model"
    assert result.tokenizer == "/fake/cache/model"

    # Verify non-hf values unchanged
    assert result.model.lora_rank == 64
    assert result.training.batch_size == 32
    assert result.output_dir == "/local/output"
    assert result.resume_from is None
