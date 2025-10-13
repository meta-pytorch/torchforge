# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for env_constants module."""

import os

from forge.env import all_env_vars, DISABLE_PERF_METRICS, EnvVar, FORGE_DISABLE_METRICS


class TestEnvVarGetValue:
    """Test the EnvVar.get_value() method."""

    def test_get_value_returns_default_when_unset(self):
        """Test get_value returns default when env var is not set."""
        if "DISABLE_PERF_METRICS" in os.environ:
            del os.environ["DISABLE_PERF_METRICS"]

        value = DISABLE_PERF_METRICS.get_value()
        assert value is False

    def test_get_value_returns_env_value_when_set(self):
        """Test get_value returns env var value when set."""
        from forge.env import MONARCH_STDERR_LEVEL

        os.environ["MONARCH_STDERR_LOG"] = "debug"

        try:
            value = MONARCH_STDERR_LEVEL.get_value()
            assert value == "debug"
        finally:
            del os.environ["MONARCH_STDERR_LOG"]

    def test_get_value_bool_auto_cast_with_true(self):
        """Test get_value auto-casts 'true' to bool."""
        os.environ["DISABLE_PERF_METRICS"] = "true"
        try:
            assert DISABLE_PERF_METRICS.get_value() is True
        finally:
            del os.environ["DISABLE_PERF_METRICS"]

    def test_get_value_bool_auto_cast_with_one(self):
        """Test get_value auto-casts '1' to bool."""
        os.environ["DISABLE_PERF_METRICS"] = "1"
        try:
            assert DISABLE_PERF_METRICS.get_value() is True
        finally:
            del os.environ["DISABLE_PERF_METRICS"]

    def test_get_value_bool_auto_cast_with_false(self):
        """Test get_value auto-casts 'false' to bool."""
        os.environ["DISABLE_PERF_METRICS"] = "false"
        try:
            assert DISABLE_PERF_METRICS.get_value() is False
        finally:
            del os.environ["DISABLE_PERF_METRICS"]


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
        """Test that predefined constants work with get_value method."""
        if DISABLE_PERF_METRICS.name in os.environ:
            del os.environ[DISABLE_PERF_METRICS.name]

        assert DISABLE_PERF_METRICS.get_value() is False

        os.environ[DISABLE_PERF_METRICS.name] = "true"
        try:
            assert DISABLE_PERF_METRICS.get_value() is True
        finally:
            del os.environ[DISABLE_PERF_METRICS.name]


class TestAllEnvVars:
    """Test the all_env_vars() function."""

    def test_all_env_vars_returns_list(self):
        """Test that all_env_vars returns a list."""
        env_vars = all_env_vars()
        assert isinstance(env_vars, list)

    def test_all_env_vars_contains_only_env_var_instances(self):
        """Test that all_env_vars returns only EnvVar instances."""
        env_vars = all_env_vars()
        assert len(env_vars) > 0
        assert all(isinstance(var, EnvVar) for var in env_vars)

    def test_all_env_vars_contains_expected_constants(self):
        """Test that all_env_vars includes known constants."""
        env_vars = all_env_vars()
        env_var_names = {var.name for var in env_vars}

        assert "DISABLE_PERF_METRICS" in env_var_names
        assert "FORGE_DISABLE_METRICS" in env_var_names
        assert "MONARCH_STDERR_LOG" in env_var_names

    def test_all_env_vars_can_iterate_and_get_values(self):
        """Test that all_env_vars can be used to iterate and get values."""
        for env_var in all_env_vars():
            value = env_var.get_value()
            assert value is not None or env_var.default is None
