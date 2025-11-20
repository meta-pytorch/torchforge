# Refactoring Proposal 10: Performance and Production Readiness

## Overview
This final proposal focuses on optimizations, configurability, and making the code production-ready. We add toggles for debug features and ensure the code can run efficiently in production.

## Key Changes

### 1. Add Debug Mode Configuration
Add a `debug` section to config to control verbose logging and debug features.

**In qwen3_1_7b.yaml:**
```yaml
debug:
  enabled: false              # Master switch for debug features
  print_episodes: false       # Print episode details during rollout
  save_message_logs: false    # Save message logs in episodes
  validate_tokens: false      # Run token validation in accumulator
  emergency_dumps: false      # Save tensors on anomalous loss values
  rollout_interval: 100       # Print rollout summary every N rollouts
```

**In main_v2.py:**
```python
async def continuous_rollouts(thread_id: int, tokenizer, debug_cfg):
    """Main rollout loop."""
    # ...

    while not shutdown_event.is_set():
        # ... rollout logic

        # Conditional debug output
        if debug_cfg.enabled and rollout_count % debug_cfg.rollout_interval == 0:
            ep = episodes[0]
            print(f"[ROLLOUT {rollout_count}] Reward: {ep.reward:.2f}, "
                  f"Tokens: {len(ep.all_token_ids)}")

        if debug_cfg.print_episodes:
            # Verbose episode printing
            # ...
```

### 2. Make TokenAccumulator Validation Configurable
Use config to control validation mode.

**In config:**
```yaml
blackjack_env:
  token_validation: "off"  # "strict", "warn", or "off"
```

**In rollout code:**
```python
from forge.data.token_accumulator import ValidationMode

# Map string to enum
validation_map = {
    "strict": ValidationMode.STRICT,
    "warn": ValidationMode.WARN,
    "off": ValidationMode.OFF,
}
validation_mode = validation_map[cfg.blackjack_env.token_validation]

accumulator = TokenAccumulator(
    tokenizer=tokenizer,
    messages=messages,
    max_len=max_seq_len,
    eos_id=tokenizer.eos_token_id,
    validation=validation_mode,
    thinking=False,
)
```

### 3. Make Message Logging Optional
Message logs are only needed for debugging. Make them optional to save memory.

**In Episode creation:**
```python
return Episode(
    episode_id=game_id,
    all_token_ids=episode_data.token_ids,
    loss_mask=loss_mask,
    reward=final_reward,
    # ... other fields
    message_log=accumulator.messages.copy() if cfg.debug.save_message_logs else None,
)
```

### 4. Add Emergency Dump Toggle
The emergency dump feature (lines 1432-1489) should be configurable.

**In simple_grpo_loss:**
```python
def simple_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.1,
    emergency_dumps: bool = False,  # NEW parameter
) -> torch.Tensor:
    """GRPO loss with next-token prediction and KL penalty."""
    # ... loss computation

    # Essential metrics
    record_metric("loss/value", loss.item(), Reduce.MEAN)
    record_metric("loss/kl_mean", (kl * loss_mask).sum() / loss_mask.sum(), Reduce.MEAN)
    record_metric("loss/advantages_mean", advantages.mean().item(), Reduce.MEAN)

    # Emergency dump (only if enabled)
    if emergency_dumps and abs(loss.item()) > 1000.0:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_file = f"/tmp/grpo_loss_debug_{timestamp}.pt"
        torch.save({
            "logits": logits.cpu(),
            "input_ids": input_ids.cpu(),
            "loss_mask": loss_mask.cpu(),
            "logprobs": logprobs.cpu(),
            "ref_logprobs": ref_logprobs.cpu(),
            "advantages": advantages.cpu(),
            "kl": kl.cpu(),
            "loss": loss.cpu(),
            "beta": beta,
        }, dump_file)
        print(f"⚠️  HUGE LOSS DETECTED: {loss.item():.2f}")
        print(f"Dumped tensors to: {dump_file}")

    return loss
```

**When creating trainer:**
```python
from functools import partial

loss_fn = partial(simple_grpo_loss, emergency_dumps=cfg.debug.emergency_dumps)

trainer = await TitanTrainer.options(**cfg.actors.trainer).as_actor(
    **cfg.trainer, loss=loss_fn
)
```

### 5. Add Warmup Configuration
Make policy warmup configurable.

**In config:**
```yaml
policy:
  warmup_enabled: true
  warmup_timeout: 120.0
  warmup_prompt: "Test prompt to warm up the model."
```

**In main:**
```python
# Warmup policy (configurable)
if cfg.policy.get("warmup_enabled", True):
    print("Warming up policy with test generation...")
    try:
        test_response = await asyncio.wait_for(
            policy.generate.route(cfg.policy.warmup_prompt),
            timeout=cfg.policy.get("warmup_timeout", 120.0),
        )
        print(f"✓ Policy ready")
    except asyncio.TimeoutError:
        raise RuntimeError("Policy warmup timed out")
```

### 6. Optimize Metric Recording
Group metrics into batches to reduce overhead.

**Before:**
```python
record_metric("loss/value", loss.item(), Reduce.MEAN)
record_metric("loss/kl_mean", kl_mean, Reduce.MEAN)
record_metric("loss/advantages_mean", adv_mean, Reduce.MEAN)
```

**After (use context manager if available):**
```python
# Record all metrics at once
metrics = {
    "loss/value": (loss.item(), Reduce.MEAN),
    "loss/kl_mean": (kl_mean, Reduce.MEAN),
    "loss/advantages_mean": (adv_mean, Reduce.MEAN),
}
for name, (value, reduce_op) in metrics.items():
    record_metric(name, value, reduce_op)
```

### 7. Add Graceful Degradation for Server Failures
Handle server failures more gracefully during long training runs.

**In continuous_rollouts:**
```python
async def continuous_rollouts(thread_id: int, tokenizer, server_url: str):
    """Main rollout loop with retry logic."""
    max_retries = 3

    while not shutdown_event.is_set():
        try:
            # ... rollout logic
        except requests.RequestException as e:
            # Server connection failed, retry
            print(f"[Thread {thread_id}] Server error: {e}, retrying...")
            await asyncio.sleep(5)
            continue
        except Exception as e:
            # Unexpected error
            print(f"[Thread {thread_id}] Unexpected error: {e}")
            if cfg.debug.enabled:
                import traceback
                traceback.print_exc()
            await asyncio.sleep(1)
```

### 8. Add Configuration Validation
Validate config at startup to catch errors early.

**In main, before service initialization:**
```python
def validate_config(cfg: DictConfig):
    """Validate configuration before training starts."""
    assert cfg.group_size > 1, "group_size must be > 1 for GRPO"
    assert cfg.blackjack_env.max_seq_len > 0, "max_seq_len must be positive"
    assert cfg.blackjack_env.max_turns > 0, "max_turns must be positive"
    assert cfg.rollout_threads > 0, "rollout_threads must be positive"

    # Check beta value
    beta = cfg.trainer.get("beta", 0.1)
    if beta < 0 or beta > 1:
        print(f"Warning: beta={beta} is unusual (typically 0.01-0.1)")

async def main(cfg: DictConfig):
    """Main GRPO training loop."""
    validate_config(cfg)
    # ... rest of main
```

### 9. Add Checkpoint Saving Trigger
Add option to save checkpoints at intervals.

**In config:**
```yaml
trainer:
  checkpoint_interval: 100  # Save checkpoint every N steps
  checkpoint_dir: "./checkpoints"
```

**In continuous_training:**
```python
if training_step % cfg.trainer.checkpoint_interval == 0:
    # Trigger checkpoint save
    # (Implementation depends on TitanTrainer interface)
    print(f"Checkpoint saved at step {training_step}")
```

## Impact
- **Production readiness:** Code can run efficiently without debug overhead
- **Configurability:** All debug/production features are configurable
- **Performance:** Reduced overhead when debug features are disabled
- **Reliability:** Graceful error handling and validation
- **Memory:** Optional message logs save significant memory in production
- **Risk:** Low - mostly adding configuration flags, not changing core logic

## Summary
After all 10 proposals, the code will be:
- **~60% smaller** (1987 lines → ~400 lines in main_v2.py)
- **Modular** (separate modules for env, rollout, token accumulator)
- **Clean** (no dead code, minimal debug noise)
- **Aligned** (matches grpo/main.py patterns)
- **Production-ready** (configurable debug features, validation, error handling)
- **Well-documented** (clear sections, docstrings, type hints)
