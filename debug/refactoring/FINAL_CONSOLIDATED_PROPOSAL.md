# FINAL REFACTORING PROPOSAL: Consolidated Best Practices

## Executive Summary
This document consolidates the best ideas from 10 iterative refactoring proposals for `apps/blackjack/main_v2.py`. The goal is to transform a 1987-line monolithic script into a clean, modular, production-ready codebase aligned with `apps/grpo/main.py` patterns.

**Expected Outcomes:**
- Main file reduced from ~1987 lines to ~400 lines (80% reduction)
- Modular architecture with separate modules for environment, rollout, and token accumulation
- Configurable debug features for production use
- Clean, well-documented code matching grpo/main.py patterns

## Phase 1: Critical Simplifications (Immediate Impact)

### 1.1 Remove EnvironmentActor
**Problem:** Lines 1136-1156 implement an actor just to provide tokenizer access.
**Solution:** Get tokenizer directly and pass to rollout functions.

```python
# In main():
tokenizer = get_tokenizer(cfg.blackjack_env.model)
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

# Pass to rollouts:
async def continuous_rollouts(thread_id: int):
    # Use tokenizer directly
```

**Impact:** Removes 20+ lines, eliminates unnecessary abstraction.

### 1.2 Drastically Simplify simple_grpo_loss
**Problem:** 280 lines of debug metrics (lines 1214-1491), emergency dumps, excessive logging.
**Solution:** Keep only essential metrics and core loss computation.

```python
def simple_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """GRPO loss with next-token prediction and KL penalty."""
    # Create targets
    targets = create_shifted_targets(input_ids, loss_mask)
    logprobs = compute_logprobs(logits, targets, ignore_index=CROSS_ENTROPY_IGNORE_IDX)

    # KL divergence with stability clipping
    logprob_diff = torch.clamp(ref_logprobs - logprobs, min=-20.0, max=20.0)
    kl = torch.clamp(torch.exp(logprob_diff) - logprob_diff - 1, min=-10.0, max=10.0)

    # Policy loss
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)

    # Per-sequence normalization
    loss = ((per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)).mean()

    # Essential metrics only
    record_metric("loss/value", loss.item(), Reduce.MEAN)
    record_metric("loss/kl_mean", (kl * loss_mask).sum() / loss_mask.sum(), Reduce.MEAN)
    record_metric("loss/advantages_mean", advantages.mean().item(), Reduce.MEAN)

    return loss
```

**Impact:** 280 lines → 40 lines (85% reduction). Keep emergency dumps as optional config.

### 1.3 Simplify Server Management
**Problem:** 150+ lines of over-engineered health checks, verbose logging (lines 1518-1680).
**Solution:** Simple startup with basic health check.

```python
def start_servers(num_servers: int, base_port: int, game_name: str) -> list:
    """Start OpenSpiel servers for rollout workers."""
    processes = []

    for i in range(num_servers):
        port = base_port + i
        subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, stdout=subprocess.DEVNULL)

        proc = multiprocessing.Process(target=start_openspiel_server, args=(game_name, port))
        proc.start()
        processes.append(proc)

    # Health check with timeout
    time.sleep(2)
    for i in range(num_servers):
        port = base_port + i
        for attempt in range(10):
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                if attempt == 9:
                    raise RuntimeError(f"Server on port {port} failed to start")
                time.sleep(1)

    return processes
```

**Impact:** 150 lines → 30 lines (80% reduction).

## Phase 2: Modular Architecture (Code Organization)

### 2.1 Extract TokenAccumulator to Module
**Create:** `src/forge/data/token_accumulator.py`
**Move:** Lines 129-745 (TokenAccumulator class, ValidationMode, TruncationReason, EpisodeData)

```python
# src/forge/data/token_accumulator.py
"""Token accumulation for multi-turn RL episodes using delta tokenization."""

from dataclasses import dataclass
from enum import Enum
import torch

class ValidationMode(Enum):
    STRICT = "strict"
    WARN = "warn"
    OFF = "off"

class TruncationReason(Enum):
    USER_TOO_LONG = "user_too_long"
    ASSISTANT_TOO_LONG = "assistant_too_long"

@dataclass
class EpisodeData:
    token_ids: torch.Tensor
    response_mask: torch.Tensor
    logprobs: torch.Tensor
    is_truncated: bool
    truncation_reason: str | None = None

class TokenAccumulator:
    # ... (full implementation, simplified docstrings)
```

**Impact:** 600+ lines moved to dedicated module, main file much cleaner.

### 2.2 Extract BlackjackEnv to Module
**Create:** `envs/blackjack_env/blackjack_env.py`
**Move:** Lines 752-914 (BlackjackEnv, EnvStepResult)

```python
# envs/blackjack_env/blackjack_env.py
"""Blackjack environment for RL training."""

import re
from dataclasses import dataclass
from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
from forge.observability.metrics import record_metric, Reduce

@dataclass
class EnvStepResult:
    observation: dict[str, str]
    reward: float
    done: bool

class BlackjackEnv:
    """Minimal Blackjack environment wrapper."""
    # ... (full implementation, simplified)
```

**Impact:** 160+ lines moved, cleaner separation of concerns.

### 2.3 Extract Rollout Functions to Module
**Create:** `apps/blackjack/rollout.py`
**Move:** Lines 922-1113 (do_single_rollout, do_group_rollout)

```python
# apps/blackjack/rollout.py
"""Rollout utilities for Blackjack GRPO training."""

import uuid
import torch
from envs.blackjack_env import BlackjackEnv
from forge.data.token_accumulator import TokenAccumulator, ValidationMode
# ... imports

async def do_single_rollout(env, policy, tokenizer, max_seq_len, max_turns, messages, game_id=None):
    """Play one game and return one Episode."""
    # ... (full implementation)
```

**Impact:** 190+ lines moved, rollout logic is reusable.

## Phase 3: Data Model Simplification

### 3.1 Simplify Episode Dataclass
**Current:** Two episode models (Episode, EpisodeData), 20 fields with complex defaults.
**Proposed:** Single, clean Episode model.

```python
@dataclass
class Episode:
    """Single episode for GRPO training."""
    episode_id: str
    all_token_ids: torch.Tensor  # [seq_len]
    loss_mask: torch.Tensor      # [seq_len], float
    reward: float

    # Computed during rollout pipeline
    ref_logprobs: torch.Tensor | None = None
    advantage: float | None = None

    # Metadata
    policy_version: int = 0
    is_truncated: bool = False

# Type aliases (like grpo/main.py)
Group = list[Episode]
Policy = Generator
```

**Impact:** Clearer data model, aligned with grpo/main.py.

### 3.2 Simplify BlackjackEnv Methods
**Changes:**
- Remove error_type distinction in `_parse_action` (return only HIT/STAND/INVALID)
- Consolidate reward computation into single method
- Remove metadata from EnvStepResult

```python
def _parse_action(self, text: str) -> str:
    """Extract action from <answer> tags. Returns HIT, STAND, or INVALID."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip().upper()
        return answer if answer in ["HIT", "STAND"] else "INVALID"
    return "INVALID"

def _compute_reward(self, env_reward: float, has_invalid: bool) -> float:
    """Compute final reward with invalid action penalty."""
    base = 3.0 if env_reward > 0 else -1.0
    penalty = -10.0 if has_invalid else 0.0
    return base + penalty
```

**Impact:** Simpler, more maintainable environment code.

## Phase 4: Clean Up Rollout and Training Loops

### 4.1 Remove Excessive Debug Printing
**Problem:** Lines 1751-1781 print full episode details every rollout.
**Solution:** Conditional, minimal logging.

```python
# In continuous_rollouts():
if rollout_count % 100 == 0:  # Only every 100 rollouts
    ep = episodes[0]
    print(f"[ROLLOUT {rollout_count}] Reward: {ep.reward:.2f}, Tokens: {len(ep.all_token_ids)}")
```

**Impact:** 95% reduction in console noise.

### 4.2 Simplify Training Loop
**Changes:**
- Remove restart_tracer flag complexity
- Cleaner control flow with early continue
- Remove conditional logging

```python
async def continuous_training():
    training_step = 0

    while max_steps == -1 or training_step < max_steps:
        t = Tracer("main_perf/continuous_training")
        t.start()

        batch = await replay_buffer.sample.call_one(curr_policy_version=training_step)
        if batch is None:
            await asyncio.sleep(0.5)
            t.stop()
            continue
        t.step("waiting_for_buffer")

        # Train
        inputs, targets = batch
        await trainer.train_step.call(inputs, targets)
        training_step += 1
        t.step("train_step")

        # Update policy
        await trainer.push_weights.call(training_step)
        await policy.update_weights.fanout(training_step)
        t.step("update_weights")

        # Clean up old weights
        if training_step >= 2:
            await drop_weights(training_step - 1)

        t.stop()
        await mlogger.flush.call_one(training_step)
```

**Impact:** More readable, simpler control flow.

### 4.3 Simplify Collate Function

```python
def collate(batches: list[Group], pad_id: int) -> tuple[list[dict], list[dict]]:
    """Collate episode batches into model inputs and targets."""
    inputs, targets = [], []

    for batch in batches:
        tokens = torch.nn.utils.rnn.pad_sequence(
            [e.all_token_ids for e in batch], batch_first=True, padding_value=pad_id
        )
        loss_mask = torch.nn.utils.rnn.pad_sequence(
            [e.loss_mask for e in batch], batch_first=True, padding_value=0.0
        )
        ref_logprobs = torch.nn.utils.rnn.pad_sequence(
            [e.ref_logprobs for e in batch], batch_first=True, padding_value=0.0
        )
        advantages = torch.tensor([e.advantage for e in batch]).unsqueeze(-1)

        inputs.append({"tokens": tokens})
        targets.append({
            "input_ids": tokens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
        })

    return inputs, targets
```

**Impact:** More concise, cleaner.

## Phase 5: Polish and Production Readiness

### 5.1 Add Configuration for Debug Features
**Add to config:**
```yaml
debug:
  enabled: false
  print_episodes: false
  save_message_logs: false
  validate_tokens: false
  rollout_interval: 100
```

**Use in code:**
```python
# Message logs (optional, saves memory)
message_log=accumulator.messages.copy() if cfg.debug.save_message_logs else None

# Validation mode
validation_mode = ValidationMode.OFF if not cfg.debug.validate_tokens else ValidationMode.STRICT
```

### 5.2 Improve Documentation
**Add clear section headers:**
```python
# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Episode:
    # ...

# ============================================================================
# Helper Actors
# ============================================================================

@dataclass
class ComputeAdvantages(ForgeActor):
    # ...

# ============================================================================
# Training Functions
# ============================================================================

def collate(...):
    # ...
```

**Add comprehensive docstrings:**
```python
def simple_grpo_loss(...) -> torch.Tensor:
    """GRPO loss with next-token prediction and KL penalty.

    Implements Group Relative Policy Optimization (GRPO) loss:
    L = -E[(π/π_old) * A - β * KL(π || π_ref)]

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        loss_mask: Loss mask [batch_size, seq_len], 1.0 for trainable
        ref_logprobs: Reference model log probs [batch_size, seq_len]
        advantages: Advantages [batch_size, 1]
        beta: KL penalty coefficient

    Returns:
        Scalar loss value
    """
```

### 5.3 Clean Up Imports
Organize imports by category:
```python
# Standard library
import asyncio
import multiprocessing
# ...

# Third-party
import torch
import torch.nn.functional as F
# ...

# Forge imports
from forge.actors.generator import Generator
# ...

# Local imports
from apps.blackjack.rollout import do_single_rollout
from envs.blackjack_env import BlackjackEnv
```

## Final File Structure

After refactoring:
```
apps/blackjack/
├── main_v2.py              (~400 lines - main training loop)
├── rollout.py              (~200 lines - rollout functions)
└── qwen3_1_7b.yaml         (config with debug section)

envs/blackjack_env/
├── __init__.py
└── blackjack_env.py        (~150 lines - environment)

src/forge/data/
├── token_accumulator.py    (~600 lines - token accumulation)
└── common.py               (existing)
```

## Implementation Phases

**Phase 1 (Immediate - 2 hours):**
1. Remove EnvironmentActor
2. Simplify simple_grpo_loss (remove debug metrics)
3. Simplify server management
4. Remove excessive debug printing

**Phase 2 (Modularization - 3 hours):**
1. Extract TokenAccumulator to module
2. Extract BlackjackEnv to module
3. Extract rollout functions to module
4. Update imports

**Phase 3 (Polish - 2 hours):**
1. Simplify Episode dataclass
2. Add configuration for debug features
3. Improve documentation and docstrings
4. Clean up imports and formatting

## Metrics

**Before:**
- Main file: 1987 lines
- Monolithic structure
- Excessive debug output
- No modularity

**After:**
- Main file: ~400 lines (80% reduction)
- 4 focused modules (main, rollout, env, token_accumulator)
- Configurable debug features
- Production-ready
- Well-documented
- Aligned with grpo/main.py patterns

## Risk Assessment

**Low Risk:**
- Code movement to modules (no logic changes)
- Removing debug prints
- Documentation improvements

**Medium Risk:**
- Simplifying simple_grpo_loss (removing metrics)
  - Mitigation: Keep metrics configurable via debug.enabled flag
- Server management simplification
  - Mitigation: Test thoroughly on target infrastructure

**High Risk:**
- None (no core algorithm changes)
