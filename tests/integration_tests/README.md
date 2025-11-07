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

## Available Tests

### `test_grpo_e2e.py`
End-to-end smoke test for the GRPO training loop. Runs the full pipeline (`apps/grpo/main.py`) for 3 training steps with a minimal configuration.

**What it tests:**
- Actor/service initialization (policy, trainer, reference model, replay buffer, etc.)
- Rollout loop (data loading → generation → reward evaluation)
- Reference model logprob calculation
- Replay buffer batching
- Training step execution
- Weight synchronization (trainer → policy)
- Torchstore operations
- Clean shutdown

**Config:** `fixtures/grpo_smoke_test.yaml` (minimal model, 1 step, small dataset)

**Run:**
```bash
PYTHONPATH=. pytest tests/integration_tests/test_grpo_e2e.py -vv
```

### `test_policy_update.py`
Tests weight synchronization and sharding between RLTrainer and Policy services.

**Run with default config:**
```bash
PYTHONPATH=. pytest tests/integration_tests/test_policy_update.py -vv
```

**Run with custom config:**
```bash
PYTHONPATH=. pytest -s tests/integration_tests/test_policy_update.py::TestWeightSync::test_sanity_check \
    --config tests/integration_tests/fixtures/qwen3_1_7b_tp.yaml --use_dcp=false
```

**Default config:** `fixtures/qwen3_1_7b_no_tp.yaml`

### `test_vllm_policy_correctness.py`
Validates vLLM policy correctness.

**Run:**
```bash
PYTHONPATH=. pytest tests/integration_tests/test_vllm_policy_correctness.py -vv
```

### `test_titan_fwd_vs_hf_fwd.py`
Compares Titan forward pass with HuggingFace forward pass.

**Run:**
```bash
PYTHONPATH=. pytest tests/integration_tests/test_titan_fwd_vs_hf_fwd.py -vv
```

## Writing New Integration Tests

When adding new integration tests:

1. **Keep tests isolated** - Each test file should be runnable independently
2. **Use minimal configs** - Create small configs in `fixtures/` for fast testing
3. **Clean up resources** - Use pytest fixtures to ensure proper cleanup
4. **Skip when needed** - Use `@pytest.mark.skipif` for GPU/resource requirements
5. **Document run commands** - Add docstrings showing how to run the test

### Example Test Structure

```python
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

@pytest.mark.asyncio
@requires_cuda
async def test_my_integration():
    """
    Run with:
    PYTHONPATH=. pytest tests/integration_tests/test_my_integration.py -vv
    """
    # Your test code here
    pass
```

## CI Integration

The CI pipeline (`.github/workflows/gpu_test.yaml`) automatically runs all integration tests, executing each test file in a separate process for isolation.
