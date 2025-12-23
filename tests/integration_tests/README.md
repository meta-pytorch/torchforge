# Integration Tests

This directory contains end-to-end integration tests for Forge components.

## Running Tests

### Important: Monarch Cleanup Issues

Monarch has seen issues in the past with proper cleanup between tests, so **integration tests should NOT be run all together in a single pytest invocation**. Running multiple integration tests in the same process can cause state leakage and test failures.

### Recommended Approach

**Run individual test files:**
```bash
PYTHONPATH=. pytest tests/integration_tests/test_grpo_e2e.py -vv
PYTHONPATH=. pytest tests/integration_tests/test_policy_update.py -vv
```

**Run all integration tests (each in separate process):**
```bash
for f in tests/integration_tests/test_*.py; do
    PYTHONPATH=. pytest "$f" -vv
done
```

**Run a specific test:**
```bash
PYTHONPATH=. pytest -s tests/integration_tests/test_grpo_e2e.py::TestGRPOEndToEnd::test_grpo_smoke_test
```

## CI Integration

The CI pipeline (`.github/workflows/gpu_test.yaml`) automatically runs all integration tests, executing each test file in a separate process for isolation.
