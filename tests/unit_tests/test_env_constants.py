# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for env_constants module."""

import os

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
        test_var = EnvVar(
            name="UNSET_TEST_VAR", default="default_value", description="Test"
        )
        if "UNSET_TEST_VAR" in os.environ:
            del os.environ["UNSET_TEST_VAR"]

        value = get_value(test_var)
        print(f"Got {value}")
        assert value == "default_value"

    def test_get_value_returns_env_value_when_set(self):
        """Test get_value returns env var value when set."""
        test_var = EnvVar(
            name="SET_TEST_VAR", default="default_value", description="Test"
        )
        os.environ["SET_TEST_VAR"] = "custom_value"

        try:
            value = get_value(test_var)
            assert value == "custom_value"
        finally:
            del os.environ["SET_TEST_VAR"]

    def test_get_value_bool_auto_cast(self):
        """Test get_value auto-casts to bool when default is bool."""
        test_var = EnvVar(name="BOOL_TEST_VAR", default=False, description="Test")

        os.environ["BOOL_TEST_VAR"] = "true"
        try:
            assert get_value(test_var) is True
        finally:
            del os.environ["BOOL_TEST_VAR"]

        os.environ["BOOL_TEST_VAR"] = "false"
        try:
            assert get_value(test_var) is False
        finally:
            del os.environ["BOOL_TEST_VAR"]

    def test_get_value_with_explicit_cast(self):
        """Test get_value with explicit cast_type."""
        test_var = EnvVar(name="INT_VAR", default=0, description="Test")
        os.environ["INT_VAR"] = "42"

        try:
            value = get_value(test_var, cast_type=int)
            assert value == 42
            assert isinstance(value, int)
        finally:
            del os.environ["INT_VAR"]


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

        assert get_value(DISABLE_PERF_METRICS) is False

        os.environ[DISABLE_PERF_METRICS.name] = "true"
        try:
            assert get_value(DISABLE_PERF_METRICS) is True
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
