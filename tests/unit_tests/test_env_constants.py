# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for env_constants module."""

import os

import pytest

from forge.env_constants import (
    all_constants,
    all_env_vars,
    DISABLE_PERF_METRICS,
    EnvVar,
    FORGE_DISABLE_METRICS,
    get_value,
)


class TestGetValue:
    """Test the get_value function."""

    def test_get_value_returns_default_when_unset(self):
        """Test get_value returns default when env var is not set."""
        value = get_value("DISABLE_PERF_METRICS")
        assert value is False

    def test_get_value_returns_env_value_when_set(self):
        """Test get_value returns env var value when set."""
        os.environ["MONARCH_STDERR_LOG"] = "debug"

        try:
            value = get_value("MONARCH_STDERR_LOG")
            assert value == "debug"
        finally:
            del os.environ["MONARCH_STDERR_LOG"]

    def test_get_value_bool_auto_cast_with_true(self):
        """Test get_value auto-casts 'true' to bool."""
        os.environ["DISABLE_PERF_METRICS"] = "true"
        try:
            assert get_value("DISABLE_PERF_METRICS") is True
        finally:
            del os.environ["DISABLE_PERF_METRICS"]

    def test_get_value_bool_auto_cast_with_one(self):
        """Test get_value auto-casts '1' to bool."""
        os.environ["DISABLE_PERF_METRICS"] = "1"
        try:
            assert get_value("DISABLE_PERF_METRICS") is True
        finally:
            del os.environ["DISABLE_PERF_METRICS"]

    def test_get_value_bool_auto_cast_with_false(self):
        """Test get_value auto-casts 'false' to bool."""
        os.environ["DISABLE_PERF_METRICS"] = "false"
        try:
            assert get_value("DISABLE_PERF_METRICS") is False
        finally:
            del os.environ["DISABLE_PERF_METRICS"]

    def test_get_value_raises_key_error_for_unregistered(self):
        """Test get_value raises KeyError for unregistered env var."""
        with pytest.raises(KeyError, match="not registered"):
            get_value("NONEXISTENT_VAR")


class TestPredefinedConstants:
    """Test the predefined environment variable constants."""

    def test_predefined_constants_structure(self):
        """Test that predefined constants are properly defined."""
        assert isinstance(DISABLE_PERF_METRICS, EnvVar)
        assert DISABLE_PERF_METRICS.name == "DISABLE_PERF_METRICS"
        assert DISABLE_PERF_METRICS.default is False

        assert isinstance(FORGE_DISABLE_METRICS, EnvVar)
        assert FORGE_DISABLE_METRICS.name == "FORGE_DISABLE_METRICS"
        assert FORGE_DISABLE_METRICS.default is False

    def test_predefined_constants_work_with_get_value(self):
        """Test that predefined constants work with get_value."""
        if DISABLE_PERF_METRICS.name in os.environ:
            del os.environ[DISABLE_PERF_METRICS.name]

        assert get_value(DISABLE_PERF_METRICS.name) is False

        os.environ[DISABLE_PERF_METRICS.name] = "true"
        try:
            assert get_value(DISABLE_PERF_METRICS.name) is True
        finally:
            del os.environ[DISABLE_PERF_METRICS.name]


class TestRegistry:
    """Test the automatic registry functionality."""

    def test_all_constants_returns_list_of_names(self):
        """Test all_constants returns a list of env var names."""
        constants = all_constants()
        assert isinstance(constants, list)
        assert "DISABLE_PERF_METRICS" in constants
        assert "FORGE_DISABLE_METRICS" in constants

    def test_all_env_vars_returns_dict_of_env_vars(self):
        """Test all_env_vars returns a dictionary of EnvVar objects."""
        env_vars = all_env_vars()
        assert isinstance(env_vars, dict)
        assert "DISABLE_PERF_METRICS" in env_vars
        assert isinstance(env_vars["DISABLE_PERF_METRICS"], EnvVar)
        assert env_vars["DISABLE_PERF_METRICS"] == DISABLE_PERF_METRICS

    def test_registry_is_automatically_populated(self):
        """Test that the registry is automatically populated."""
        env_vars = all_env_vars()
        constants = all_constants()

        assert len(env_vars) == len(constants)
        assert len(env_vars) >= 6  # At least the 6 predefined constants
