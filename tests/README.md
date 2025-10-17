# Tests

This directory contains tests for the forge project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components
- `integration_tests/`: Contains integration tests that test multiple components together
- `sandbox/`: Contains experimental adhoc scripts used for development and debugging
- `assets/`: Contains test assets and fixtures used by the tests

## Running Tests

### Prerequisites

Ensure you have all development dependencies installed (run from forge root):

```bash
pip install .[dev]
```

### Running Integration Tests

To run all integration tests:

```bash
pytest -s tests/integration_tests/
```

To run a specific integration test file:

```bash
pytest -s tests/integration_tests/test_vllm_policy_correctness.py
```

To run a specific integration test function:

```bash
pytest -s tests/integration_tests/test_vllm_policy_correctness.py::test_same_output
```

Integration tests support custom options defined in `conftest.py`:
- `--config`: Path to YAML config file for sanity check tests
- `--use_dcp`: Override the YAML config `trainer.use_dcp` field (true/false)

Example with options:
```bash
pytest -s tests/integration_tests/ --config ./path/to/config.yaml --use_dcp true
```

### Running Unit Tests

To run all unit tests:

```bash
pytest -s tests/unit_tests/
```

To run a specific unit test file:

```bash
pytest -s tests/unit_tests/test_config.py
```

To run a specific unit test function:

```bash
pytest -s tests/unit_tests/test_config.py::test_cache_hit_scenario
```
