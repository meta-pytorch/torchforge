# Refactoring Proposal 09: Polish and Documentation

## Overview
Building on Proposals 01-08, this iteration focuses on polishing the code with better comments, consistent formatting, and removing any remaining cruft. This is the "final touches" pass.

## Key Changes

### 1. Add Clear Section Headers
Like grpo/main.py, use clear section separators.

**Example:**
```python
# main_v2.py after refactoring

# Copyright header...

# Usage: python -m apps.blackjack.main_v2 --config apps/blackjack/qwen3_1_7b.yaml

import asyncio
# ... imports

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Episode:
    """Single episode for GRPO training."""
    # ...

# Type aliases
Group = list[Episode]
Policy = Generator

# ============================================================================
# Helper Actors
# ============================================================================

@dataclass
class ComputeAdvantages(ForgeActor):
    # ...

# ============================================================================
# Training Functions
# ============================================================================

def collate(batches: list[Group], pad_id: int) -> tuple[...]:
    """Collate episode batches into model inputs and targets."""
    # ...

def simple_grpo_loss(...) -> torch.Tensor:
    """GRPO loss with next-token prediction and KL penalty."""
    # ...

async def drop_weights(version: int):
    """Drop old model weights from torchstore."""
    # ...

# ============================================================================
# Main Training Loop
# ============================================================================

async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    # ...

if __name__ == "__main__":
    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()
```

### 2. Improve Function Docstrings
Follow NumPy/Google docstring style consistently.

**Before:**
```python
def collate(
    batches: list[list[Episode]],
    pad_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collates a list of batches (groups) into inputs and targets."""
```

**After:**
```python
def collate(
    batches: list[Group],
    pad_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collate episode batches into model inputs and targets.

    Args:
        batches: List of groups, where each group is a list of Episodes
        pad_id: Padding token ID from tokenizer

    Returns:
        Tuple of (inputs, targets) for training where:
        - inputs: List of dicts with 'tokens' key [batch_size, seq_len]
        - targets: List of dicts with 'input_ids', 'loss_mask', 'ref_logprobs', 'advantages'
    """
```

### 3. Add Inline Comments for Complex Logic
Clarify non-obvious operations.

**Example in simple_grpo_loss:**
```python
def simple_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """GRPO loss with next-token prediction and KL penalty.

    Implements Group Relative Policy Optimization (GRPO) loss:
    L = -E[(π/π_old) * A - β * KL(π || π_ref)]

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        loss_mask: Loss mask [batch_size, seq_len], 1.0 for trainable positions
        ref_logprobs: Reference model log probabilities [batch_size, seq_len]
        advantages: Advantages [batch_size, 1]
        beta: KL penalty coefficient (default: 0.1)

    Returns:
        Scalar loss value
    """
    # Create targets by shifting input_ids for next-token prediction
    targets = create_shifted_targets(input_ids, loss_mask)

    # Compute policy log probabilities (masked positions are 0.0)
    logprobs = compute_logprobs(logits, targets, ignore_index=CROSS_ENTROPY_IGNORE_IDX)

    # KL divergence with numerical stability clipping (following VERL implementation)
    logprob_diff = torch.clamp(ref_logprobs - logprobs, min=-20.0, max=20.0)
    kl = torch.clamp(torch.exp(logprob_diff) - logprob_diff - 1, min=-10.0, max=10.0)

    # Policy gradient term
    policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages

    # Combined loss (negative because we want to maximize)
    per_token_loss = -(policy_loss - beta * kl)

    # Per-sequence normalization: average by each sequence's trainable token count
    loss = (
        (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)
    ).mean()

    # Essential metrics
    record_metric("loss/value", loss.item(), Reduce.MEAN)
    record_metric("loss/kl_mean", (kl * loss_mask).sum() / loss_mask.sum(), Reduce.MEAN)
    record_metric("loss/advantages_mean", advantages.mean().item(), Reduce.MEAN)

    return loss
```

### 4. Clean Up Imports
Remove unused imports, organize by category.

**Before:**
```python
import asyncio
import multiprocessing
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, partial
from typing import Any, Optional

import requests

import torch
import torch.nn.functional as F
import torchstore as ts
# ... many more
```

**After:**
```python
# Standard library
import asyncio
import multiprocessing
import subprocess
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any

# Third-party
import requests
import torch
import torch.nn.functional as F
import torchstore as ts
from omegaconf import DictConfig
from vllm import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

# Forge imports
from forge.actors._torchstore_utils import get_dcp_whole_state_dict_key, get_param_prefix
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import TitanTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data.common import CROSS_ENTROPY_IGNORE_IDX
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from forge.util.ops import compute_logprobs, create_shifted_targets

# Local imports
from apps.blackjack.rollout import do_single_rollout
from envs.blackjack_env import BlackjackEnv
from envs.openspiel_env.server_utils import start_servers
```

### 5. Standardize Metric Names
Use consistent naming convention for all metrics.

**Prefix conventions:**
- `loss/*` - Loss-related metrics
- `episode/*` - Episode-level metrics
- `buffer/*` - Replay buffer metrics
- `game/*` - Game environment metrics
- `main/*` - Main loop performance metrics

**Example:**
```python
# Instead of inconsistent naming:
record_metric("groups/rate_dropped", ...)
record_metric("buffer/episodes_accepted", ...)
record_metric("main/continuous_rollouts/count_rollout_iterations", ...)

# Use consistent naming:
record_metric("rollout/groups_dropped", ..., Reduce.SUM)
record_metric("buffer/episodes_accepted", ..., Reduce.SUM)
record_metric("rollout/iterations", ..., Reduce.SUM)
```

### 6. Add Type Hints Throughout
Ensure all functions have complete type hints.

**Example:**
```python
def start_servers(
    num_servers: int,
    base_port: int,
    game_name: str,
) -> list[multiprocessing.Process]:
    """Start OpenSpiel servers for rollout workers."""
    # ...
```

### 7. Remove Redundant Comments
Remove obvious comments, keep insightful ones.

**Before:**
```python
# Initialize TokenAccumulator with BASE anchor pattern
accumulator = TokenAccumulator(...)

# Reset environment
initial_obs = env.reset()

# Multi-turn loop
final_reward = 0.0
```

**After:**
```python
accumulator = TokenAccumulator(...)
initial_obs = env.reset()
final_reward = 0.0
```

## Impact
- **Readability:** Much improved with clear sections and good documentation
- **Maintainability:** Easier to understand and modify
- **Professionalism:** Code looks polished and production-ready
- **Onboarding:** New developers can understand the code faster
- **Risk:** Zero - only documentation and formatting changes
