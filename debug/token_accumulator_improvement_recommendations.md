# TokenAccumulator Improvement Recommendations

## Executive Summary

This document synthesizes patterns and best practices from 5 major RL libraries (RL/nemo_rl, tinker-cookbook, verl, verifiers, trl) to improve the `TokenAccumulator` class. The goal is to make it:
- **Cleaner**: Better organized with clear documentation
- **Debuggable**: Visual tools and comprehensive logging
- **Safe**: Validation functions and sanity checks
- **Well-documented**: Concise yet comprehensive docs

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Documentation Patterns](#documentation-patterns)
3. [Validation & Safety Patterns](#validation--safety-patterns)
4. [Debugging Patterns](#debugging-patterns)
5. [Code Organization Patterns](#code-organization-patterns)
6. [Specific Recommendations](#specific-recommendations)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Current State Analysis

### What Works Well ✓

1. **Clear Design Philosophy**: VERL approach using vLLM tokens directly
2. **Anchor System**: Delta tokenization avoids repeated re-tokenization
3. **Budget Management**: Proper tracking with `get_remaining_budget()`
4. **Truncation Handling**: Explicit `TruncationReason` enum
5. **Thread Safety**: Tokenizer lock for concurrent access
6. **Parallel Arrays**: `accumulated_tokens`, `response_mask`, `logprobs` tracked together

### Areas for Improvement 🔧

1. **Documentation**: Missing docstring examples, shape annotations
2. **Validation**: `_check_structure` is minimal, EOS check commented out
3. **Debugging**: No visual tools, limited introspection
4. **Error Messages**: Lack contextual information (current state, indices)
5. **Testing Helpers**: No built-in debugging utilities
6. **Type Safety**: Missing explicit type hints in some methods

---

## Documentation Patterns

### Pattern 1: Comprehensive Docstrings with Examples

**From: tinker-cookbook, TRL**

#### Current Example:
```python
def add_user_message(self, content: str) -> bool:
    """Add user message, truncating to fit budget if necessary. Returns False if truncated."""
```

#### Recommended Enhancement:
```python
def add_user_message(self, content: str) -> bool:
    """
    Add a user message to the conversation, truncating if it exceeds budget.

    The message is tokenized using the anchor-based delta tokenization approach:
    - Tokenizes [system, new_user_message] to get full tokens
    - Extracts delta by removing system prefix
    - Truncates to fit remaining budget if necessary

    Args:
        content (str): The text content of the user message

    Returns:
        bool: True if message was added without truncation, False if truncated

    Example:
        >>> acc = TokenAccumulator(tokenizer, messages=[{"role": "system", "content": "You are helpful"}], max_seq_len=100, eos_token_id=2)
        >>> success = acc.add_user_message("Hello!")
        >>> print(success)  # True
        >>> print(len(acc.accumulated_tokens))  # e.g., 15
        >>> acc.add_user_message("x" * 10000)  # Very long message
        >>> print(acc.is_truncated)  # True
        >>> print(acc.truncation_reason)  # TruncationReason.USER_TOO_LONG

    Notes:
        - If truncation occurs, `is_truncated` is set to True
        - Truncated messages are still added (up to available budget)
        - The message is appended to `self.messages` for chat template continuity
    """
```

**Key Elements:**
- One-line summary first
- Detailed explanation of the approach
- Args/Returns with types
- Concrete example showing usage
- Notes for edge cases

---

### Pattern 2: Module-Level Documentation

**From: tinker-cookbook, verifiers**

#### Recommended Addition (at top of file):
```python
"""
Token accumulation for multi-turn RL rollouts with vLLM.

This module implements the TokenAccumulator class, which handles the complexities of:
- Multi-turn conversation token concatenation
- Response mask creation for loss computation
- vLLM token integration without re-tokenization (prevents chat template bugs)
- Budget management with truncation tracking

## Key Design Principles

### Delta Tokenization
Instead of re-tokenizing the entire conversation after each turn, we use an anchor-based
approach. The anchor ([system, empty_user]) stays constant, allowing us to tokenize new
messages against it and extract only the delta tokens.

### VERL Approach
We use generation tokens from vLLM directly, avoiding re-tokenization that can introduce
misalignments. The generation prompt (e.g., "<|im_start|>assistant\n") is computed from
the anchor and added separately.

### Response Masking
- Prefix tokens (system, user, generation prompt): `response_mask=False`
- Assistant content from vLLM: `response_mask=True`
- This ensures we only train on model-generated tokens

## Notation

We use shape annotations in comments to clarify tensor dimensions:
- `_T`: Token/sequence dimension (e.g., `tokens_T` = list of length T)
- `_B`: Batch dimension (not used in this class, but relevant for downstream)

## Usage Example

```python
# Initialize with system message
acc = TokenAccumulator(
    tokenizer=tokenizer,
    messages=[{"role": "system", "content": "You are a helpful assistant."}],
    max_seq_len=2048,
    eos_token_id=tokenizer.eos_token_id,
)

# Multi-turn conversation
acc.add_user_message("What is 2+2?")
prompt = acc.format_prompt()
response = vllm_generate(prompt, max_tokens=acc.get_remaining_budget())
acc.add_assistant_response(response.text, response.token_ids, response.logprobs)

# Finalize and extract data
acc.finalize()
episode = Episode(
    token_ids=acc.accumulated_tokens,  # Shape: (T,)
    response_mask=acc.response_mask,   # Shape: (T,), bool
    logprobs=acc.logprobs,             # Shape: (T,), float
    is_truncated=acc.is_truncated,
)
```

## See Also
- `/debug/test_token_accumulator_validation.py` - Basic validation tests
- `/debug/test_token_accumulator_v2.py` - Integration tests
"""
```

---

### Pattern 3: Inline Comments Explaining "Why"

**From: tinker-cookbook, TRL**

#### Current:
```python
# Extract only user tokens (remove system prefix)
user_tokens = full[self.system_len :]
```

#### Enhanced:
```python
# Extract only user tokens (remove system prefix)
# Why: We tokenized [system, user] to leverage chat template, but we only want
# the delta tokens from the user message. System tokens were already added during
# initialization, so we slice them off using the pre-computed system_len anchor.
user_tokens = full[self.system_len :]  # Shape: (user_len,)
```

---

### Pattern 4: Type Annotations Throughout

**From: TRL, verl**

#### Current:
```python
def _accumulate(
    self, tokens: list[int], mask: list[bool], logprobs: list[float] | None = None
):
```

#### Enhanced:
```python
def _accumulate(
    self,
    tokens: list[int],
    mask: list[bool],
    logprobs: list[float] | None = None
) -> None:
    """
    Append tokens, masks, and logprobs to internal accumulators.

    All three arrays must maintain the same length after appending (verified in _check_structure).

    Args:
        tokens: Token IDs to append (shape: T_new)
        mask: Response mask values (True for trainable tokens) (shape: T_new)
        logprobs: Log probabilities from model (shape: T_new), or None for 0.0 defaults
    """
```

---

## Validation & Safety Patterns

### Pattern 1: Multi-Way Equality Assertions

**From: tinker-cookbook, verl, verifiers**

#### Current:
```python
def _check_structure(self):
    """Verify basic structural invariants."""
    assert (
        len(self.accumulated_tokens)
        == len(self.response_mask)
        == len(self.logprobs)
    )
```

#### Enhanced:
```python
def _check_structure(self) -> None:
    """
    Verify basic structural invariants.

    Raises:
        AssertionError: If parallel arrays have mismatched lengths or exceed budget
    """
    token_len = len(self.accumulated_tokens)
    mask_len = len(self.response_mask)
    logprob_len = len(self.logprobs)

    # Multi-way equality with diagnostic info
    assert token_len == mask_len == logprob_len, (
        f"Parallel array length mismatch:\n"
        f"  tokens:        {token_len}\n"
        f"  response_mask: {mask_len}\n"
        f"  logprobs:      {logprob_len}\n"
        f"All arrays must have the same length."
    )

    # Budget validation
    if token_len > self.max_seq_len:
        raise ValueError(
            f"Budget overflow: {token_len} tokens > max_seq_len={self.max_seq_len}\n"
            f"This indicates a bug in budget tracking."
        )
```

**Key Improvements:**
- Store lengths in variables for clarity
- Multi-line error message with actual values
- Explains what went wrong AND what should be true

---

### Pattern 2: Incremental Validation After Updates

**From: verl, verifiers**

#### Recommended Addition:
```python
def _accumulate(
    self,
    tokens: list[int],
    mask: list[bool],
    logprobs: list[float] | None = None
) -> None:
    """Append tokens, masks, and logprobs to internal accumulators."""
    # Validate inputs
    if not tokens:
        raise ValueError("Cannot accumulate empty token list")

    if len(tokens) != len(mask):
        raise ValueError(
            f"Token/mask length mismatch: {len(tokens)} tokens vs {len(mask)} mask values"
        )

    if logprobs is not None and len(logprobs) != len(tokens):
        raise ValueError(
            f"Token/logprob length mismatch: {len(tokens)} tokens vs {len(logprobs)} logprobs"
        )

    # Perform accumulation
    self.accumulated_tokens.extend(tokens)
    self.response_mask.extend(mask)
    self.logprobs.extend(logprobs or [0.0] * len(tokens))

    # Validate invariants after update (only in strict mode for performance)
    if self.sanity_check_mode == SanityCheckMode.STRICT:
        self._check_structure()
```

---

### Pattern 3: Prefix Consistency Validation

**From: verifiers, verl**

This is CRITICAL for the anchor-based approach. We should validate that tokenizing incrementally produces the same result as tokenizing from scratch.

#### Recommended Addition:
```python
def _validate_prefix_consistency(self) -> bool:
    """
    Validate that incremental tokenization matches full re-tokenization.

    This catches chat template bugs where adding messages doesn't extend the
    token sequence as expected.

    Returns:
        bool: True if consistent

    Raises:
        AssertionError: If tokenization is inconsistent (in STRICT mode)
    """
    if self.sanity_check_mode == SanityCheckMode.DISABLE:
        return True

    # Re-tokenize entire conversation from scratch
    with self._tokenizer_lock:
        full_tokens = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )

    # Check if accumulated tokens match
    if len(full_tokens) != len(self.accumulated_tokens):
        error_msg = (
            f"Tokenization inconsistency detected!\n"
            f"  Incremental approach: {len(self.accumulated_tokens)} tokens\n"
            f"  Full re-tokenization: {len(full_tokens)} tokens\n"
            f"This suggests a chat template bug or anchor drift."
        )
        if self.sanity_check_mode == SanityCheckMode.STRICT:
            raise AssertionError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
            return False

    # Check token-by-token equality
    for i, (acc_token, full_token) in enumerate(zip(self.accumulated_tokens, full_tokens)):
        if acc_token != full_token:
            error_msg = (
                f"Token mismatch at position {i}:\n"
                f"  Incremental: {acc_token}\n"
                f"  Full:        {full_token}\n"
                f"  Context: ...{self.accumulated_tokens[max(0,i-3):i+3]}..."
            )
            if self.sanity_check_mode == SanityCheckMode.STRICT:
                raise AssertionError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return False

    return True
```

**Usage:**
```python
def finalize(self) -> bool:
    """Validate episode. Returns True if valid."""
    self._check_structure()

    if self.sanity_check_mode != SanityCheckMode.DISABLE:
        self._validate_prefix_consistency()
        # self._check_eos_alignment()  # Re-enable after fixing

    return True
```

---

### Pattern 4: Input Validation with Actionable Errors

**From: verifiers, TRL**

#### Current:
```python
def __init__(
    self,
    tokenizer,
    messages: list[dict],
    max_seq_len: int,
    eos_token_id: int,
    ...
):
```

#### Enhanced:
```python
def __init__(
    self,
    tokenizer,
    messages: list[dict],
    max_seq_len: int,
    eos_token_id: int,
    enable_thinking: bool = True,
    sanity_check_mode: SanityCheckMode = SanityCheckMode.STRICT,
):
    """
    Initialize TokenAccumulator for multi-turn conversation.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support
        messages: Initial conversation messages (must include system message)
        max_seq_len: Maximum sequence length (hard limit)
        eos_token_id: End-of-sequence token ID
        enable_thinking: Whether to enable <think> tags (for Qwen models)
        sanity_check_mode: Validation strictness (STRICT or DISABLE)

    Raises:
        ValueError: If tokenizer is missing required attributes
        ValueError: If messages is empty or malformed
        ValueError: If max_seq_len is invalid
    """
    # Validate tokenizer
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError(
            "Tokenizer must support apply_chat_template. "
            "Please use a recent HuggingFace transformers version (>= 4.34)."
        )

    if not hasattr(tokenizer, 'eos_token_id') and eos_token_id is None:
        raise ValueError(
            "Either tokenizer.eos_token_id must be set or eos_token_id must be provided."
        )

    # Validate messages
    if not messages:
        raise ValueError("Must provide at least a system message in messages list.")

    for i, msg in enumerate(messages):
        if 'role' not in msg or 'content' not in msg:
            raise ValueError(
                f"Message at index {i} is malformed. "
                f"Expected dict with 'role' and 'content', got: {msg.keys()}"
            )

    # Validate max_seq_len
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    if max_seq_len > 100000:
        print(f"WARNING: max_seq_len={max_seq_len} is very large. Are you sure?")

    # Initialize
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.eos_token_id = eos_token_id
    self.enable_thinking = enable_thinking
    self.sanity_check_mode = sanity_check_mode

    # ... rest of init
```

---

## Debugging Patterns

### Pattern 1: Visual Debug Printing

**From: RL/nemo_rl, TRL**

#### Recommended Addition:
```python
def debug_print(self, show_tokens: bool = False, max_turns: int = 5) -> None:
    """
    Print current accumulator state for debugging.

    Args:
        show_tokens: If True, show actual token IDs (can be verbose)
        max_turns: Maximum number of turns to display (prevents spam)
    """
    print("=" * 80)
    print(f"TokenAccumulator State")
    print("=" * 80)
    print(f"Total tokens:     {len(self.accumulated_tokens)} / {self.max_seq_len}")
    print(f"Remaining budget: {self.get_remaining_budget()}")
    print(f"Is truncated:     {self.is_truncated}")
    if self.is_truncated:
        print(f"Truncation reason: {self.truncation_reason.value}")
    print(f"Num messages:     {len(self.messages)}")
    print()

    # Print messages
    print("Messages:")
    print("-" * 80)
    for i, msg in enumerate(self.messages[:max_turns]):
        role = msg['role']
        content = msg['content']
        # Truncate long content
        if len(content) > 100:
            content = content[:97] + "..."
        print(f"  [{i}] {role:10s}: {content}")

    if len(self.messages) > max_turns:
        print(f"  ... and {len(self.messages) - max_turns} more messages")
    print()

    # Print mask statistics
    num_trainable = sum(self.response_mask)
    num_total = len(self.response_mask)
    pct_trainable = 100 * num_trainable / num_total if num_total > 0 else 0
    print(f"Response mask: {num_trainable}/{num_total} trainable ({pct_trainable:.1f}%)")

    # Optionally show tokens
    if show_tokens:
        print()
        print("Accumulated tokens (first 50):")
        print(self.accumulated_tokens[:50])
        print()
        print("Response mask (first 50):")
        print(self.response_mask[:50])

    print("=" * 80)
```

**Usage:**
```python
acc = TokenAccumulator(...)
acc.add_user_message("Hello")
acc.debug_print()  # Quick sanity check during development
```

---

### Pattern 2: Colorized Token Visualization

**From: tinker-cookbook**

#### Recommended Addition (Optional, but very helpful):
```python
def visualize_tokens(
    self,
    max_tokens: int = 200,
    use_color: bool = True
) -> str:
    """
    Create a colorized visualization of tokens with mask overlay.

    Color scheme:
    - Green (or ✓): response_mask=True (trainable)
    - Gray  (or ·): response_mask=False (not trainable)

    Args:
        max_tokens: Maximum tokens to display
        use_color: Whether to use ANSI color codes

    Returns:
        str: Formatted visualization
    """
    if not self.accumulated_tokens:
        return "[Empty accumulator]"

    # Decode tokens to text
    with self._tokenizer_lock:
        decoded_tokens = [
            self.tokenizer.decode([token_id])
            for token_id in self.accumulated_tokens[:max_tokens]
        ]

    lines = []
    lines.append("Token Visualization:")
    lines.append("-" * 80)

    for i, (token_text, is_response) in enumerate(
        zip(decoded_tokens, self.response_mask[:max_tokens])
    ):
        # Escape special characters
        token_text = repr(token_text)[1:-1]  # Remove outer quotes

        if use_color:
            # ANSI color codes
            if is_response:
                color = "\033[92m"  # Green
                reset = "\033[0m"
                marker = "✓"
            else:
                color = "\033[90m"  # Gray
                reset = "\033[0m"
                marker = "·"

            lines.append(f"{i:4d} {marker} {color}{token_text}{reset}")
        else:
            marker = "✓" if is_response else "·"
            lines.append(f"{i:4d} {marker} {token_text}")

    if len(self.accumulated_tokens) > max_tokens:
        lines.append(f"... and {len(self.accumulated_tokens) - max_tokens} more tokens")

    return "\n".join(lines)
```

**Usage:**
```python
acc = TokenAccumulator(...)
# ... add messages ...
print(acc.visualize_tokens())
```

---

### Pattern 3: Turn Boundary Tracking

**From: verifiers**

This helps debug where each message starts/ends in the token sequence.

#### Recommended Addition:
```python
class TokenAccumulator:
    def __init__(self, ...):
        # ... existing fields ...
        self.turn_boundaries = []  # List of (start_idx, end_idx, role, content_preview)

    def _accumulate(
        self,
        tokens: list[int],
        mask: list[bool],
        logprobs: list[float] | None = None,
        turn_info: dict | None = None  # NEW: optional turn metadata
    ) -> None:
        """Append tokens, masks, and logprobs to internal accumulators."""
        start_idx = len(self.accumulated_tokens)

        self.accumulated_tokens.extend(tokens)
        self.response_mask.extend(mask)
        self.logprobs.extend(logprobs or [0.0] * len(tokens))

        end_idx = len(self.accumulated_tokens)

        # Track turn boundary
        if turn_info:
            self.turn_boundaries.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "role": turn_info.get("role", "unknown"),
                "content_preview": turn_info.get("content", "")[:50],
            })

    def print_turn_boundaries(self) -> None:
        """Print turn boundaries for debugging."""
        print("Turn Boundaries:")
        print("-" * 80)
        for i, turn in enumerate(self.turn_boundaries):
            start = turn["start_idx"]
            end = turn["end_idx"]
            role = turn["role"]
            preview = turn["content_preview"]
            length = end - start
            print(f"  [{i}] {role:10s} [{start:4d}:{end:4d}] ({length:3d} tokens) {preview}")
        print("-" * 80)
```

**Update methods to use it:**
```python
def add_user_message(self, content: str) -> bool:
    # ... existing logic ...
    if user_tokens:
        self.messages.append(message)
        self._accumulate(
            user_tokens,
            mask=[False] * len(user_tokens),
            turn_info={"role": "user", "content": content}  # NEW
        )
    return len(user_tokens) == original_len
```

---

### Pattern 4: Structured Logging

**From: tinker-cookbook, verifiers**

#### Recommended Addition:
```python
def get_debug_summary(self) -> dict:
    """
    Get structured debug information (useful for logging systems like wandb).

    Returns:
        dict: Summary statistics
    """
    num_trainable = sum(self.response_mask)
    num_total = len(self.accumulated_tokens)

    # Count message types
    role_counts = {}
    for msg in self.messages:
        role = msg["role"]
        role_counts[role] = role_counts.get(role, 0) + 1

    # Logprob statistics (for trainable tokens only)
    trainable_logprobs = [
        lp for lp, mask in zip(self.logprobs, self.response_mask) if mask
    ]

    return {
        "num_tokens": num_total,
        "num_trainable_tokens": num_trainable,
        "pct_trainable": 100 * num_trainable / num_total if num_total > 0 else 0,
        "num_messages": len(self.messages),
        "role_counts": role_counts,
        "is_truncated": self.is_truncated,
        "truncation_reason": self.truncation_reason.value if self.is_truncated else None,
        "budget_used": num_total,
        "budget_remaining": self.get_remaining_budget(),
        "avg_logprob": sum(trainable_logprobs) / len(trainable_logprobs) if trainable_logprobs else 0.0,
        "min_logprob": min(trainable_logprobs) if trainable_logprobs else 0.0,
        "max_logprob": max(trainable_logprobs) if trainable_logprobs else 0.0,
    }
```

**Usage:**
```python
# In training loop
acc = TokenAccumulator(...)
# ... build episode ...
summary = acc.get_debug_summary()
wandb.log({"episode": summary})
```

---

## Code Organization Patterns

### Pattern 1: Helper Functions for Complex Operations

**From: tinker-cookbook, verifiers**

Some operations in TokenAccumulator could be extracted into pure functions:

```python
def _compute_generation_prompt_tokens(
    tokenizer,
    anchor: list[dict],
    enable_thinking: bool
) -> tuple[list[int], int]:
    """
    Compute generation prompt tokens from anchor conversation.

    The generation prompt (e.g., "<|im_start|>assistant\n") is the delta between
    tokenizing with and without add_generation_prompt=True.

    Args:
        tokenizer: HuggingFace tokenizer
        anchor: Anchor messages ([system, empty_user])
        enable_thinking: Whether to enable <think> tags

    Returns:
        tuple: (generation_prompt_tokens, generation_prompt_len)
    """
    anchor_without = tokenizer.apply_chat_template(
        anchor,
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    anchor_with = tokenizer.apply_chat_template(
        anchor,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=enable_thinking,
    )

    generation_prompt_tokens = anchor_with[len(anchor_without):]
    generation_prompt_len = len(generation_prompt_tokens)

    return generation_prompt_tokens, generation_prompt_len


def _compute_system_len(
    tokenizer,
    system_msg: dict,
    enable_thinking: bool
) -> int:
    """
    Compute number of tokens in system message alone.

    Used for slicing user message delta tokens.
    """
    return len(
        tokenizer.apply_chat_template(
            [system_msg],
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
    )
```

**Benefits:**
- Easier to test in isolation
- Can be unit tested without full TokenAccumulator setup
- Clearer purpose and reusability

---

### Pattern 2: Separate Validation Class

**From: verl, TRL**

For complex validation, consider a separate validator:

```python
class TokenAccumulatorValidator:
    """Validation utilities for TokenAccumulator."""

    @staticmethod
    def check_parallel_arrays(
        accumulated_tokens: list[int],
        response_mask: list[bool],
        logprobs: list[float],
    ) -> None:
        """Check that parallel arrays have matching lengths."""
        lengths = {
            "tokens": len(accumulated_tokens),
            "response_mask": len(response_mask),
            "logprobs": len(logprobs),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Parallel array length mismatch:\n" +
                "\n".join(f"  {k}: {v}" for k, v in lengths.items())
            )

    @staticmethod
    def check_eos_alignment(
        accumulated_tokens: list[int],
        response_mask: list[bool],
        eos_token_id: int
    ) -> None:
        """Verify each response segment ends with EOS."""
        in_response = False
        last_response_idx = -1

        for i, (token, is_response) in enumerate(zip(accumulated_tokens, response_mask)):
            if is_response and not in_response:
                in_response = True
            elif is_response:
                last_response_idx = i
            elif not is_response and in_response:
                # End of response - check last token was EOS
                if last_response_idx >= 0 and accumulated_tokens[last_response_idx] != eos_token_id:
                    raise ValueError(
                        f"Response ended at position {last_response_idx} with token "
                        f"{accumulated_tokens[last_response_idx]}, expected EOS {eos_token_id}"
                    )
                in_response = False
                last_response_idx = -1

        # Check final response
        if in_response and last_response_idx >= 0:
            if accumulated_tokens[last_response_idx] != eos_token_id:
                raise ValueError(
                    f"Final response ended at position {last_response_idx} with token "
                    f"{accumulated_tokens[last_response_idx]}, expected EOS {eos_token_id}"
                )


# Usage in TokenAccumulator:
def finalize(self) -> bool:
    """Validate episode. Returns True if valid."""
    if self.sanity_check_mode == SanityCheckMode.DISABLE:
        return True

    TokenAccumulatorValidator.check_parallel_arrays(
        self.accumulated_tokens,
        self.response_mask,
        self.logprobs,
    )

    TokenAccumulatorValidator.check_eos_alignment(
        self.accumulated_tokens,
        self.response_mask,
        self.eos_token_id,
    )

    return True
```

---

## Specific Recommendations

### Priority 1: Critical for Correctness ⚠️

1. **Re-enable EOS alignment check** (currently commented out)
   - This caught real bugs in your investigation
   - Make it work properly or replace with equivalent validation

2. **Add prefix consistency validation**
   - Verify incremental tokenization matches full re-tokenization
   - Critical for anchor-based approach

3. **Enhance error messages with context**
   - Include actual values, indices, and state
   - Make debugging faster

### Priority 2: Improve Debuggability 🔍

4. **Add `debug_print()` method**
   - Quick visual inspection during development
   - Include token counts, mask stats, truncation info

5. **Add `visualize_tokens()` method**
   - Colorized token-level view
   - Helps spot mask alignment issues

6. **Track turn boundaries**
   - Record where each message starts/ends
   - Easier to debug token alignment

7. **Add `get_debug_summary()` for structured logging**
   - Integration with wandb/tensorboard
   - Track statistics over training

### Priority 3: Documentation 📚

8. **Add module-level docstring**
   - Explain design principles (delta tokenization, VERL approach)
   - Include usage example

9. **Enhance method docstrings**
   - Add concrete examples
   - Document edge cases and return values

10. **Add inline comments explaining "why"**
    - Especially for non-obvious operations
    - Shape annotations in comments

### Priority 4: Nice to Have ✨

11. **Extract helper functions**
    - `_compute_generation_prompt_tokens()`
    - `_compute_system_len()`
    - Easier to test and reuse

12. **Add type hints everywhere**
    - Especially return types
    - Consider using mypy for static checking

13. **Create TokenAccumulatorValidator class**
    - Separate validation logic
    - Easier to extend and test

---

## Implementation Roadmap

### Phase 1: Critical Fixes (1-2 hours)
- [ ] Fix EOS alignment check or replace with equivalent
- [ ] Add prefix consistency validation
- [ ] Enhance all error messages with context

### Phase 2: Debugging Tools (2-3 hours)
- [ ] Implement `debug_print()`
- [ ] Implement `get_debug_summary()`
- [ ] Add turn boundary tracking
- [ ] Implement `visualize_tokens()` (optional but helpful)

### Phase 3: Documentation (1-2 hours)
- [ ] Add module-level docstring with design explanation
- [ ] Enhance all method docstrings with examples
- [ ] Add inline "why" comments for complex sections
- [ ] Add shape annotations

### Phase 4: Refactoring (2-3 hours)
- [ ] Extract helper functions
- [ ] Add comprehensive type hints
- [ ] Create TokenAccumulatorValidator class (optional)
- [ ] Add performance optimizations if needed

**Total Estimated Time: 6-10 hours**

---

## Example: Before vs After

### Before:
```python
def add_user_message(self, content: str) -> bool:
    """Add user message, truncating to fit budget if necessary. Returns False if truncated."""
    message = {"role": "user", "content": content}
    with self._tokenizer_lock:
        full = self.tokenizer.apply_chat_template(...)
    user_tokens = full[self.system_len :]
    budget = self.get_remaining_budget()
    original_len = len(user_tokens)
    user_tokens = self._truncate_to_fit(user_tokens, budget, TruncationReason.USER_TOO_LONG)
    if user_tokens:
        self.messages.append(message)
        self._accumulate(user_tokens, mask=[False] * len(user_tokens))
    return len(user_tokens) == original_len
```

### After:
```python
def add_user_message(self, content: str) -> bool:
    """
    Add a user message to the conversation, truncating if necessary.

    Uses delta tokenization: tokenizes [system, new_user_message] and extracts
    only the user message tokens by slicing off the pre-computed system prefix.

    Args:
        content: User message text

    Returns:
        bool: True if added without truncation, False if truncated

    Example:
        >>> acc.add_user_message("Hello!")
        True
        >>> acc.add_user_message("x" * 10000)  # Too long
        False
        >>> acc.is_truncated
        True
    """
    message = {"role": "user", "content": content}

    # Tokenize [system, user] to leverage chat template
    with self._tokenizer_lock:
        full = self.tokenizer.apply_chat_template(
            [self.anchor[0], message],
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )

    # Extract delta: remove system prefix to get only user tokens
    # Why: System was already added during initialization, we only want new tokens
    user_tokens = full[self.system_len :]  # Shape: (user_len,)

    # Check budget and truncate if needed
    budget = self.get_remaining_budget()
    original_len = len(user_tokens)

    if len(user_tokens) > budget:
        user_tokens = self._truncate_to_fit(
            user_tokens, budget, TruncationReason.USER_TOO_LONG
        )

    # Add to accumulator (user tokens are not trainable)
    if user_tokens:
        self.messages.append(message)
        self._accumulate(
            user_tokens,
            mask=[False] * len(user_tokens),  # User tokens: response_mask=False
            turn_info={"role": "user", "content": content}  # For debugging
        )

    return len(user_tokens) == original_len
```

**Key Improvements:**
- ✓ Comprehensive docstring with example
- ✓ Inline comments explaining "why"
- ✓ Shape annotations
- ✓ Turn tracking for debugging
- ✓ More descriptive variable usage

---

## Conclusion

The TokenAccumulator class has a solid foundation with the anchor-based delta tokenization approach. The main improvements needed are:

1. **Better validation** to catch bugs early (prefix consistency, EOS alignment)
2. **Debugging tools** to make development faster (debug_print, visualize_tokens)
3. **Documentation** to help users understand the design (docstrings, examples, inline comments)

These improvements will make the class:
- **Safer**: Catch bugs before they cause silent failures
- **Easier to debug**: Visual tools and structured logging
- **Easier to understand**: Clear docs with examples and explanations

The patterns from these 5 libraries show consistent best practices across the RL community. Implementing these recommendations will bring TokenAccumulator up to production quality standards.
