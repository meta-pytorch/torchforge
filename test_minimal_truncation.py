#!/usr/bin/env python3
"""
Minimal test to verify v9 fix for Qwen think tags.

Tests 4 scenarios:
1. prompt -> user -> assistant (complete)
2. prompt -> user -> assistant-truncated
3. prompt -> user -> assistant -> user (complete multi-turn)
4. prompt -> user -> assistant-truncated -> user-truncated
"""

import sys
sys.path.insert(0, "/home/felipemello/forge")

from transformers import AutoTokenizer


class TokenAccumulator:
    """Minimal token accumulator using direct token extraction (v9 approach)."""

    def __init__(self, tokenizer, system_prompt: str):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

        # Pre-compute role headers/footers for assistant
        self.role_header, self.role_footer = self._compute_role_tokens()

        # Initialize with system message
        self.messages = [{"role": "system", "content": system_prompt}]
        self.all_tokens = tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=False, tokenize=True
        )

    def _compute_role_tokens(self):
        """Pre-compute assistant role header and footer tokens."""
        # Use complete think tags to avoid auto-wrapper
        base = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        with_assistant = base + [{"role": "assistant", "content": "<think>X</think>"}]

        base_tokens = self.tokenizer.apply_chat_template(base, add_generation_prompt=False, tokenize=True)
        full_tokens = self.tokenizer.apply_chat_template(with_assistant, add_generation_prompt=False, tokenize=True)

        # Extract assistant portion
        assistant_full = full_tokens[len(base_tokens):]

        # Content tokens
        content_tokens = self.tokenizer.encode("<think>X</think>", add_special_tokens=False)

        # Find content position in assistant_full
        for i in range(len(assistant_full) - len(content_tokens) + 1):
            if assistant_full[i:i+len(content_tokens)] == content_tokens:
                header = assistant_full[:i]
                footer = assistant_full[i+len(content_tokens):]
                return header, footer

        # Fallback: assume last token is footer (eos)
        return assistant_full[:-1], assistant_full[-1:]

    def add_user_message(self, content: str):
        """Add user message using prefix matching."""
        self.messages.append({"role": "user", "content": content})

        # Tokenize to get new tokens
        new_tokens = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=False, tokenize=True
        )

        # Extract delta
        delta = new_tokens[len(self.all_tokens):]
        self.all_tokens.extend(delta)

    def add_assistant_response(self, content_tokens: list[int], text: str):
        """
        Add assistant response using DIRECT tokens (v9 approach).

        Args:
            content_tokens: Raw tokens from vLLM (content only, no role headers)
            text: Decoded text (for message log)
        """
        # Check if truncated (last token != eos)
        is_truncated = len(content_tokens) > 0 and content_tokens[-1] != self.eos_token_id

        # Combine: header + content + footer
        # BUT if truncated, don't add footer (incomplete response)
        if is_truncated:
            assistant_tokens = self.role_header + content_tokens
        else:
            # Remove eos from content if present (footer already has it)
            if content_tokens and content_tokens[-1] == self.eos_token_id:
                content_tokens = content_tokens[:-1]
            assistant_tokens = self.role_header + content_tokens + self.role_footer

        # Accumulate
        self.all_tokens.extend(assistant_tokens)

        # Add to messages
        self.messages.append({"role": "assistant", "content": text})

        return is_truncated

    def validate(self):
        """Compare accumulated tokens vs ground truth."""
        ground_truth = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=False, tokenize=True
        )

        match = self.all_tokens == ground_truth

        if match:
            print(f"  ✅ MATCH - {len(self.all_tokens)} tokens")
        else:
            print(f"  ❌ MISMATCH")
            print(f"    Accumulated: {len(self.all_tokens)} tokens")
            print(f"    Ground truth: {len(ground_truth)} tokens")
            print(f"    Diff: {len(ground_truth) - len(self.all_tokens)}")

            # Find first difference
            for i in range(min(len(self.all_tokens), len(ground_truth))):
                if self.all_tokens[i] != ground_truth[i]:
                    print(f"    First diff at position {i}:")
                    print(f"      Got: {self.all_tokens[max(0,i-3):i+5]}")
                    print(f"      Exp: {ground_truth[max(0,i-3):i+5]}")
                    break

        return match


def simulate_vllm_response(tokenizer, content: str, truncate_at: int = None):
    """
    Simulate vLLM response by encoding content.

    Args:
        content: Response text
        truncate_at: If set, truncate tokens at this position
    """
    tokens = tokenizer.encode(content, add_special_tokens=False)

    if truncate_at and truncate_at < len(tokens):
        tokens = tokens[:truncate_at]

    return tokens, tokenizer.decode(tokens)


def main():
    # Load tokenizer
    model_path = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print(f"Model: {model_path}")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})\n")
    print("=" * 80)

    # Test 1: Complete single-turn
    print("\nTEST 1: prompt -> user -> assistant (COMPLETE)")
    print("-" * 80)
    acc = TokenAccumulator(tokenizer, "You are a helpful assistant.")
    acc.add_user_message("Hand: 15, Dealer: 10")

    # Simulate complete response
    content_tokens, content_text = simulate_vllm_response(
        tokenizer,
        f"<think>Let me think...</think>\n\nHIT{tokenizer.eos_token}"
    )
    print(f"  Content tokens: {len(content_tokens)}")
    print(f"  Last token == eos: {content_tokens[-1] == tokenizer.eos_token_id}")

    is_truncated = acc.add_assistant_response(content_tokens, content_text)
    print(f"  Is truncated: {is_truncated}")
    acc.validate()

    # Test 2: Truncated single-turn
    print("\nTEST 2: prompt -> user -> assistant-truncated")
    print("-" * 80)
    acc2 = TokenAccumulator(tokenizer, "You are a helpful assistant.")
    acc2.add_user_message("Hand: 15, Dealer: 10")

    # Simulate truncated response (incomplete think tag)
    content_tokens, content_text = simulate_vllm_response(
        tokenizer,
        "<think>Let me think about this carefully...",
        truncate_at=10  # Truncate after 10 tokens
    )
    print(f"  Content tokens: {len(content_tokens)}")
    print(f"  Content text: {repr(content_text)}")
    print(f"  Last token == eos: {content_tokens[-1] == tokenizer.eos_token_id}")

    is_truncated = acc2.add_assistant_response(content_tokens, content_text)
    print(f"  Is truncated: {is_truncated}")
    acc2.validate()

    # Check for duplicate think tags in decoded output
    decoded = tokenizer.decode(acc2.all_tokens)
    has_duplicates = decoded.count("<think>") > 1
    print(f"  Duplicate <think> tags: {has_duplicates}")
    if has_duplicates:
        print(f"  ❌ FOUND DUPLICATES!")
        print(f"  Decoded:\n{decoded}")

    # Test 3: Complete multi-turn
    print("\nTEST 3: prompt -> user -> assistant -> user (COMPLETE MULTI-TURN)")
    print("-" * 80)
    acc3 = TokenAccumulator(tokenizer, "You are a helpful assistant.")
    acc3.add_user_message("Hand: 15, Dealer: 10")

    content_tokens, content_text = simulate_vllm_response(
        tokenizer,
        f"<think>Thinking...</think>\n\nHIT{tokenizer.eos_token}"
    )
    acc3.add_assistant_response(content_tokens, content_text)

    # Add second user message
    acc3.add_user_message("Hand: 16, Dealer: 10")
    print(f"  After 2 turns: {len(acc3.all_tokens)} tokens")
    acc3.validate()

    # Test 4: Truncated multi-turn
    print("\nTEST 4: prompt -> user -> assistant-truncated -> user-truncated")
    print("-" * 80)
    acc4 = TokenAccumulator(tokenizer, "You are a helpful assistant.")
    acc4.add_user_message("Hand: 15, Dealer: 10")

    # First response truncated
    content_tokens, content_text = simulate_vllm_response(
        tokenizer,
        "<think>Let me",
        truncate_at=5
    )
    is_truncated = acc4.add_assistant_response(content_tokens, content_text)
    print(f"  Turn 1 truncated: {is_truncated}")

    # Try to add another user message (would be rejected in real code)
    acc4.add_user_message("Hand: 16, Dealer: 10")
    print(f"  After truncated multi-turn: {len(acc4.all_tokens)} tokens")
    acc4.validate()

    # Check for duplicates
    decoded = tokenizer.decode(acc4.all_tokens)
    has_duplicates = decoded.count("<think>") > 1
    print(f"  Duplicate <think> tags: {has_duplicates}")
    if has_duplicates:
        print(f"  ❌ FOUND DUPLICATES!")
        # Show where duplicates appear
        lines = decoded.split('\n')
        for i, line in enumerate(lines):
            if '<think>' in line or '</think>' in line:
                print(f"    Line {i}: {repr(line)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The v9 fix (direct token extraction) should:")
    print("  1. ✅ Match ground truth for complete responses")
    print("  2. ❌ May mismatch for truncated (incomplete think tags)")
    print("  3. ✅ No duplicate <think> tags if using direct tokens correctly")
    print("\nIf we DROP truncated episodes (like Tinker):")
    print("  - Only test 1 and 3 matter (complete responses)")
    print("  - Tests 2 and 4 would be discarded anyway")
    print("  - Simplifies logic: no need to handle incomplete tags!")


if __name__ == "__main__":
    main()
