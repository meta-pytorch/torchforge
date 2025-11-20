# Refactoring Proposal 03: Simplify BlackjackEnv and Episode Models

## Overview
Building on Proposals 01-02, this iteration simplifies the BlackjackEnv class and consolidates the Episode data models. We align more closely with the original GRPO main.py structure.

## Key Changes

### 1. Simplify Episode Dataclass
Currently have two Episode-related classes (Episode, EpisodeData). The main Episode class is overly complex.

**Before (lines 92-112):**
```python
@dataclass
class Episode:
    # Required fields (no defaults)
    episode_id: str
    all_token_ids: torch.Tensor
    response_mask: torch.Tensor
    loss_mask: torch.Tensor
    reward: float

    # Optional fields (with defaults)
    task_name: str = "blackjack"
    policy_version: int = 0
    is_truncated: bool = False
    advantage: float | None = None
    logprobs: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_log: list[dict[str, str]] | None = None
```

**After (aligned with grpo/main.py style):**
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

    # Debug info (optional, can be dropped in production)
    message_log: list[dict] | None = None
```

**Rationale:** We don't need `response_mask` AND `loss_mask`. The loss_mask is sufficient (it's the shifted version). Remove task_name (always blackjack). Simplify metadata.

### 2. Simplify BlackjackEnv - Remove Excessive Metrics
The environment records too many granular metrics (lines 812-848).

**Before:**
```python
if is_invalid:
    self.has_invalid_action = True
    action_name = "STAND"
    record_metric("game/invalid_action_rate", 1, Reduce.MEAN)

    if error_type == "NO_TAGS":
        print(f"[ENV] ⚠️  INVALID action: Missing <answer> tags!")
        print(f"[ENV]     Text: '{action_text}...'")
        record_metric("game/missing_answer_tags", 1, Reduce.SUM)
    elif error_type == "INVALID_CONTENT":
        print(f"[ENV] ⚠️  INVALID action: Bad content in <answer> tags!")
        print(f"[ENV]     Text: '{action_text}...'")
        record_metric("game/invalid_answer_content", 1, Reduce.SUM)
    # ... more metrics
else:
    record_metric("game/invalid_action_rate", 0, Reduce.MEAN)
```

**After:**
```python
if is_invalid:
    self.has_invalid_action = True
    action_name = "STAND"
    record_metric("game/invalid_actions", 1, Reduce.SUM)
```

**Rationale:** One metric for invalid actions is enough. Debug prints can be removed (use proper logging if needed).

### 3. Remove Penalty Logic from Environment
The -10 penalty for invalid actions (line 841) mixes reward shaping with environment logic. Move to reward computation.

**Before:**
```python
if result.done:
    reward = self._compute_reward(result.reward)
    if self.has_invalid_action:
        reward -= 10.0
        record_metric("game/invalid_action_penalty", 1, Reduce.SUM)
```

**After:**
```python
def _compute_reward(self, env_reward: float, has_invalid: bool) -> float:
    """Compute final reward with penalty for invalid actions."""
    base_reward = 3.0 if env_reward > 0 else -1.0
    penalty = -10.0 if has_invalid else 0.0
    return base_reward + penalty
```

### 4. Simplify EnvStepResult
Remove metadata field - it's barely used.

**Before:**
```python
@dataclass
class EnvStepResult:
    observation: dict[str, str]
    reward: float
    done: bool
    metadata: dict[str, Any] = field(default_factory=dict)
```

**After:**
```python
@dataclass
class EnvStepResult:
    observation: dict[str, str]
    reward: float
    done: bool
```

### 5. Clean Up Action Parsing
The regex-based parsing is fine, but simplify the return type.

**Before:**
```python
def _parse_action(self, text: str) -> tuple[str, str]:
    """Returns: (action, error_type)"""
    # ... parsing logic
    if match:
        answer = match.group(1).strip().upper()
        if answer == "HIT":
            return ("HIT", "")
        elif answer == "STAND":
            return ("STAND", "")
        else:
            return ("INVALID", "INVALID_CONTENT")
    else:
        return ("INVALID", "NO_TAGS")
```

**After:**
```python
def _parse_action(self, text: str) -> str:
    """Extract action from <answer> tags. Returns HIT, STAND, or INVALID."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip().upper()
        return answer if answer in ["HIT", "STAND"] else "INVALID"
    return "INVALID"
```

**Rationale:** We don't need to distinguish NO_TAGS vs INVALID_CONTENT for the core logic. This simplification makes the code cleaner.

## Impact
- **Episode class:** 20 lines → 15 lines
- **BlackjackEnv:** Cleaner, less coupled to metrics
- **Readability:** Much improved, less noise
- **Alignment:** Closer to grpo/main.py style
- **Risk:** Low - simplifying without breaking functionality
