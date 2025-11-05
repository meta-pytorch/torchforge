# Weight Sync Sandbox

A minimal test environment focused exclusively on testing the weight synchronization mechanism between `RLTrainer` and `Generator`.

## Purpose

This sandbox tests the complete weight sync pipeline in isolation, without the complexity of a full RL training loop. It's designed for:

- **Debugging weight sync issues**: Isolate and test push/pull mechanisms
- **Performance profiling**: Measure push_weights() and update_weights() latency
- **Mode verification**: Test both DCP (filesystem) and Direct (RDMA) sync modes
- **Quick validation**: Verify weight sync works before running full training

## What It Tests

### Push Weights (`RLTrainer.push_weights()`)
- Converts TorchTitan state dict to HuggingFace format
- Saves to torchstore via DCP (no RDMA) or direct (with RDMA)
- Measures push performance

### Update Weights (`Generator.update_weights()`)
- Stops accepting new requests (lock mechanism)
- Waits for pending requests to complete
- Fetches weights from torchstore
- Loads weights into vLLM model
- Resumes accepting requests
- Measures update performance

### Verification
- Runs forward pass with updated weights to confirm success

## Usage

```bash
python -m tests.sandbox.weight_sync.main --config tests/sandbox/weight_sync/qwen3_1_7b.yaml
```

## Test Flow

1. **Initialize** trainer and generator with same model (Qwen3-1.7B)
2. **Run ONE training step** to create a weight delta
3. **Push weights** from trainer to torchstore (version 1)
4. **Update weights** in generator from torchstore (version 1)
5. **Verify** with forward pass to confirm weights loaded correctly

## Output

The sandbox prints:
- Sync mode used (DCP vs Direct/RDMA)
- Time for push_weights()
- Time for update_weights()
- Total sync time
- Sample generation output for verification

Example output:
```
================================================================================
WEIGHT SYNC SANDBOX
================================================================================
Model: Qwen/Qwen3-1.7B
RDMA available: False
Sync mode: DCP (Filesystem)
================================================================================

✓ Initialization complete (12.34s)

[1/4] Running single training step to create weight delta...
✓ Training step complete (0.56s)

[2/4] Testing push_weights() to torchstore...
✓ Pushed weights to torchstore (2.13s)

[3/4] Testing update_weights() from torchstore...
✓ Updated weights in generator (1.87s)

[4/4] Verification: Running forward pass with updated weights...
✓ Forward pass successful (0.23s)

================================================================================
WEIGHT SYNC TEST COMPLETE
================================================================================
Push time:         2.13s
Update time:       1.87s
Total sync time:   4.00s
Sync mode used:    DCP (Filesystem)
================================================================================
```

## Configuration

Uses Qwen3-1.7B for fast testing with minimal resource requirements:
- **Model**: Qwen/Qwen3-1.7B (small, fast to load)
- **Batch size**: 4 (minimal overhead)
- **Sequence length**: 128 tokens (64 request + 64 response)
- **Generator**: Single process actor (not service)
- **Trainer**: Single process, no FSDP
- **Data parallel**: 1 (single GPU)

## Key Differences from Other Sandboxes

- **vs vllm sandbox**: Adds trainer and tests weight updates
- **vs rl_trainer sandbox**: Adds generator and tests weight loading
- **vs toy_rl sandbox**: Focuses purely on weight sync, not full RL loop

This is the **only sandbox that tests the complete weight sync mechanism** in isolation.
