# Episode Class Design for Multi-Turn Tool Calling in Forge

## Executive Summary

After analyzing VERL, Prime-RL, TRL, NeMo-RL, and Tinker, we propose a clean `Episode` class for multi-turn tool calling in Forge.

**Key Insight:** Forge's current `pad_id`, `request_len`, `response_len` exist as workarounds for not having response masking. All other frameworks use explicit masks instead.

**Recommendation:** Single `Episode` dataclass with concatenated tokens and explicit `response_mask`.

---

## Current Forge Episode (Problems)

```python
@dataclass
class Episode:
    episode_id: str
    pad_id: int              # ❌ Workaround for no masking
    request_len: int         # ❌ Fixed-length workaround
    response_len: int        # ❌ Fixed-length workaround
    target: Any | None = None
    completion: Completion | None = None  # ❌ Stores entire object
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None
```

**Problems:**
- Can't handle multi-turn (variable length)
- No response masking → would train on tool results (critical bug!)
- Stores entire `Completion` object (memory waste)
- Fixed lengths incompatible with variable-turn episodes

---

## Proposed Episode Class

```python
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Episode:
    """
    Episode data for GRPO training with multi-turn tool calling support.

    Stores concatenated tokens from all turns (prompts + LLM outputs + tool results)
    with a response mask indicating which tokens to train on.

    Example multi-turn episode:
        Turn 1: User: "Search Python" → Assistant: "<tool_call>search(...)"
        Turn 2: Tool: "Found 10 results..." → Assistant: "Here are the results..."

        all_token_ids: [101, 102, 345, 346, 456, 457, 458, 567, 568]
        response_mask: [ 0,   0,   1,   1,   0,   0,   0,   1,   1 ]
                       [prompt ][LLM ][  tool result  ][LLM ]
    """

    # ============ Core Identifiers ============
    episode_id: str
    task_name: str | None = None           # Environment identifier (e.g., "websearch", "coding")

    # ============ Policy & Truncation (for eviction policy) ============
    generator_version: int                  # Which policy version generated this
    is_truncated: bool                      # Hit max_turns limit

    # ============ Token Data ============
    all_token_ids: torch.Tensor            # All tokens concatenated (prompts + responses + tool results)
                                           # Shape: (seq_len,)

    logprobs: torch.Tensor                 # Log probabilities for all tokens
                                           # Shape: (seq_len,)
                                           # 0.0 for non-LLM tokens (prompts, tool results)

    response_mask: torch.Tensor            # CRITICAL: Mask for training
                                           # Shape: (seq_len,)
                                           # 1.0 = train on this token (LLM output)
                                           # 0.0 = skip this token (prompt, tool result)

    # ============ Conversation History (Optional) ============
    target: Any | None = None              # Ground truth (optional, for evaluation)
    message_log: list[dict[str, Any]] | None = None
    # OpenAI-compatible messages for debugging/analysis
    # Example: [
    #   {"role": "user", "content": "Search Python"},
    #   {"role": "assistant", "content": "...", "tool_calls": [...]},
    #   {"role": "tool", "content": "Found 10 results..."}
    # ]

    # ============ Rewards & Training ============
    reward: float | None = None
    advantage: float | None = None         # Computed by GRPO
    ref_logprobs: torch.Tensor | None = None  # Reference model logprobs (for KL penalty)
                                              # Shape: (seq_len,)

    # ============ Metadata ============
    metadata: dict[str, Any] = field(default_factory=dict)
    # Suggested fields (all optional):
    #   - num_turns: int
    #   - num_tool_calls: int
    #   - stop_reason: str


# Type alias for GRPO groups
Group = list[Episode]
```

---

## Key Design Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| **Single class vs Multi-class?** | Single `Episode` | GRPO only needs final reward (no per-step). Simpler, less memory, easier batching. VERL/Prime-RL/TRL all use single class. |
| **response_mask** | ✅ Required | **Critical** - prevents training on tool results. Without this, model learns to hallucinate tool outputs instead of calling tools. |
| **Concatenate tokens** | All in `all_token_ids` | Multi-turn requires concatenation anyway. Simpler than separate prompt/completion fields. |
| **actual_length field?** | ❌ Drop | Redundant with `len(all_token_ids)`. Avoid consistency bugs. |
| **pad_id, request_len, response_len?** | ❌ Drop | Workarounds for missing mask. Use dynamic padding in collate_fn instead. |
| **completion object?** | ❌ Drop | Just parse needed fields from Generator. Don't store entire Prompt/text/metadata. |
| **generator_version, is_truncated** | ✅ First-class fields | Critical for eviction policy - don't hide in metadata. |
| **message_log** | Optional | Useful for debugging/analysis, not required for training. |
| **metadata** | Flexible dict | For optional debugging data (num_turns, stop_reason, etc.). |

---

## Why These Choices Matter

### 1. response_mask is Critical

**Without masking (BAD):**
```
Prompt: "Search for Python"
Assistant: "<tool_call>search(...)</tool_call>"
Tool: "Found 10 results: 1. Python.org, 2. ..."   ← MODEL TRAINED ON THIS!
Assistant: "Here are the results..."

Problem: Model learns to output fake tool responses instead of calling tools!
```

**With masking (GOOD):**
```
response_mask: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
               [prompt  ][LLM  ][tool output    ][LLM  ]

Only LLM output tokens contribute to loss → Model learns correct tool calling!
```

### 2. Single Class vs Tinker's Multi-Class

Tinker uses `Transition` → `Trajectory` → `TrajectoryGroup` (3 classes).

**Why single class for Forge:**
- GRPO only needs final reward (no per-step rewards like PPO/A2C)
- Simpler implementation (1 class vs 3)
- Less memory (no per-step objects)
- Easier batching (flat structure)
- Industry standard (VERL, Prime-RL, TRL all use single class)

### 3. Eviction Policy Needs generator_version & is_truncated

Replay buffers need to evict old data:
- **generator_version**: Discard episodes from old policy (stale data)
- **is_truncated**: Don't train on incomplete episodes (noisy signal)

Too important to hide in metadata dict.

---

## TODO: Truncation Strategy Research

**Status:** TO BE RESEARCHED

When an episode hits `max_turns`, we need a clear truncation strategy.

**Open Questions:**
1. **Turn-level:** Drop whole last turn or keep partial?
2. **Within-turn:** Truncate long tool outputs? Where (start/middle/end)?
3. **Prompt vs Response:** Prioritize which? Drop early turns to fit max_seq_len?
4. **Mask alignment:** How to ensure response_mask stays aligned after truncation?
5. **Training:** Should `is_truncated=True` episodes be excluded or down-weighted?

**Follow-up:** Create `3_truncation_strategy.md` analyzing how other frameworks handle this and propose strategies for Forge.

---

**End of Document**
