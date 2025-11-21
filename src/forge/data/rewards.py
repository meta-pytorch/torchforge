# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re


class MathReward:
    """Reward class for evaluating math correctness."""

    def __init__(self, tolerance: float = 1e-6, partial_credit: float = 0.1):
        self.tolerance = tolerance
        self.partial_credit = partial_credit

    def __call__(self, prompt: str, response: str, target: str) -> float:
        """Compute math correctness reward."""
        target_number = self._to_float(target)
        if target_number is None:
            return 0.0

        # Look for answer in <answer></answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if answer_match:
            model_answer = self._to_float(answer_match.group(1).strip())
            if (
                model_answer is not None
                and abs(target_number - model_answer) < self.tolerance
            ):
                return 1.0  # Correct answer

        # Check for partial credit: target number appears elsewhere in response
        response_without_answer_tags = re.sub(
            r"<answer>.*?</answer>", "", response, flags=re.DOTALL
        )
        # Convert to int if it's a whole number to avoid "117.0" vs "117" mismatch
        target_str = (
            str(int(target_number))
            if target_number.is_integer()
            else str(target_number)
        )
        if target_str in response_without_answer_tags:
            return self.partial_credit

        return 0.0  # No match

    def _to_float(self, text: str) -> float | None:
        """Convert text to float, return None if invalid."""
        try:
            # Remove common non-numeric characters like $, commas, etc.
            cleaned_text = re.sub(r"[$,\s]", "", text.strip())
            return float(cleaned_text)
        except (ValueError, AttributeError):
            return None


class ThinkingReward:
    """Reward class for evaluating use of <think> tags in reasoning."""

    def __init__(self, partial_reward: float = 0.2, full_reward: float = 1.0):
        self.partial_reward = partial_reward
        self.full_reward = full_reward
        self._THINK_BLOCK_RE = re.compile(
            r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL
        )
        self._THINK_TAG_ATTEMPT_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)

    def __call__(self, prompt: str, response: str, target: str | None = None) -> float:
        """Compute thinking reward."""
        if not response:
            return 0.0

        matches = self._THINK_BLOCK_RE.findall(response)
        has_well_formed = any(len(re.sub(r"\s+", "", m)) >= 1 for m in matches)
        has_attempt = bool(self._THINK_TAG_ATTEMPT_RE.search(response)) or bool(matches)
        if has_well_formed:
            return self.full_reward
        elif has_attempt:
            return self.partial_reward
        return 0.0


class CleanMathReward:
    """Clean reward for GSM8K using 'Final Answer: <NUMBER>' format.

    Designed to reinforce:
    - Correct numeric answers
    - Clean, parseable format (no tags, just "Final Answer: X")
    - Concise reasoning (600-1200 chars is sweet spot)
    - No hedging or multiple answers

    Computes three sub-rewards:
    1. Accuracy: +1.0 (correct) / -0.25 (near-miss) / -0.5 (wrong) / -1.0 (unparseable)
    2. Format: +0.25 (clean) / -0.25 (multiple/hedging) / -0.5 (missing)
    3. Length: piecewise linear based on character count
       - ≤600: 0.0 (neutral)
       - 600-1200: +0.1 (sweet spot)
       - 1200-2000: -0.1 (getting long)
       - >2000: -0.3 (rambling)

    Returns weighted sum: 1.0×R_acc + 0.7×R_fmt + 0.3×R_len

    Args:
        tolerance: Absolute tolerance for exact match (default: 1e-6)
        allow_near_miss: Allow ±1 integer tolerance (default: True)
        near_miss_reward: Reward for near-miss (default: -0.25)
        correct_reward: Reward for correct answer (default: 1.0)
        wrong_reward: Reward for wrong answer (default: -0.5)
        unparseable_reward: Reward for unparseable (default: -1.0)
        format_clean_reward: Reward for clean format (default: 0.25)
        format_multiple_reward: Reward for multiple answers (default: -0.25)
        format_missing_reward: Reward for missing format (default: -0.5)
        w_acc: Weight for accuracy reward (default: 1.0)
        w_fmt: Weight for format reward (default: 0.7)
        w_len: Weight for length reward (default: 0.3)
        debug: If True, print debug output (default: False)
        debug_sample_rate: Fraction of calls to debug (default: 0.1)
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        allow_near_miss: bool = True,
        near_miss_reward: float = -0.25,
        correct_reward: float = 1.0,
        wrong_reward: float = -0.5,
        unparseable_reward: float = -1.0,
        format_clean_reward: float = 0.25,
        format_multiple_reward: float = -0.25,
        format_missing_reward: float = -0.5,
        w_acc: float = 1.0,
        w_fmt: float = 0.7,
        w_len: float = 0.3,
        debug: bool = False,
        debug_sample_rate: float = 0.1,
    ):
        self.tolerance = tolerance
        self.allow_near_miss = allow_near_miss
        self.near_miss_reward = near_miss_reward
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.unparseable_reward = unparseable_reward
        self.format_clean_reward = format_clean_reward
        self.format_multiple_reward = format_multiple_reward
        self.format_missing_reward = format_missing_reward
        self.w_acc = w_acc
        self.w_fmt = w_fmt
        self.w_len = w_len
        self.debug = debug
        self.debug_sample_rate = debug_sample_rate
        self._debug_counter = 0

        # Regex to match "Final Answer: <NUMBER>"
        self._FINAL_ANSWER_RE = re.compile(
            r"Final\s+Answer:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE
        )

    def __call__(self, prompt: str, response: str, target: str) -> float:
        """Compute combined reward."""
        self._debug_counter += 1
        should_debug = (
            self.debug
            and self.debug_sample_rate > 0
            and (self._debug_counter % int(1 / self.debug_sample_rate)) == 0
        )

        if not response:
            reward = (
                self.w_acc * self.unparseable_reward
                + self.w_fmt * self.format_missing_reward
                + self.w_len * -0.3  # Empty is treated as too short
            )
            if should_debug:
                try:
                    print(f"\n[CleanMathReward] Empty response | Total: {reward:.2f}")
                except (BrokenPipeError, OSError):
                    pass
            return reward

        # 1. Accuracy reward
        acc_reward = self._compute_accuracy_reward(response, target)

        # 2. Format reward
        fmt_reward = self._compute_format_reward(response)

        # 3. Length reward
        len_reward = self._compute_length_reward(response)

        # Weighted total
        total = (
            self.w_acc * acc_reward + self.w_fmt * fmt_reward + self.w_len * len_reward
        )

        if should_debug:
            parsed = self._parse_final_answer(response)
            model_answer_text = f"{parsed:.2f}" if parsed is not None else "[NO ANSWER]"

            sample = response.replace("\n", " ")[:100]
            try:
                print(
                    f"\n[CleanMathReward] Combined Reward"
                    f"\n  Accuracy: {acc_reward:+5.1f} × {self.w_acc:.2f} = {self.w_acc * acc_reward:+6.2f}"
                    f"\n  Format:   {fmt_reward:+5.1f} × {self.w_fmt:.2f} = {self.w_fmt * fmt_reward:+6.2f}"
                    f"\n  Length:   {len_reward:+5.1f} × {self.w_len:.2f} = {self.w_len * len_reward:+6.2f}"
                    f"\n  Model Answer: {model_answer_text} | Target Answer: {target}"
                    f"\n  Sample: {sample}..."
                    f"\n  → Total: {total:+6.2f}"
                )
            except (BrokenPipeError, OSError):
                try:
                    print(
                        f"[CleanMathReward] Model: {sample} | Target: {target} | Total: {total:+.2f}"
                    )
                except (BrokenPipeError, OSError):
                    pass

        return total

    def _parse_final_answer(self, text: str) -> float | None:
        """Parse the last 'Final Answer: <NUMBER>' in the text.

        Args:
            text: Response text to parse

        Returns:
            Parsed number or None if not found
        """
        # Find all matches and take the last one
        matches = list(self._FINAL_ANSWER_RE.finditer(text))
        if not matches:
            return None

        last_match = matches[-1]
        try:
            return float(last_match.group(1))
        except ValueError:
            return None

    def _compute_accuracy_reward(self, response: str, target: str) -> float:
        """Compute accuracy reward based on parsed answer vs target.

        Returns:
        - correct_reward: Exact match (within tolerance)
        - near_miss_reward: Off by ±1 (if enabled)
        - wrong_reward: Wrong numeric answer
        - unparseable_reward: Cannot parse answer
        """
        # Parse target
        target_number = self._to_float(target)
        if target_number is None:
            # If we can't parse the target, give neutral reward
            return 0.0

        # Parse model answer
        model_answer = self._parse_final_answer(response)
        if model_answer is None:
            return self.unparseable_reward

        # Exact match
        if abs(target_number - model_answer) < self.tolerance:
            return self.correct_reward

        # Near-miss check (±1 for integers)
        if self.allow_near_miss and abs(target_number - model_answer) == 1.0:
            return self.near_miss_reward

        # Wrong answer
        return self.wrong_reward

    def _compute_format_reward(self, response: str) -> float:
        """Check format: single clean 'Final Answer: <NUMBER>'.

        Returns:
        - format_clean_reward: Single well-formed answer
        - format_multiple_reward: Multiple answers or hedging ("or")
        - format_missing_reward: No answer or unparseable
        """
        matches = self._FINAL_ANSWER_RE.findall(response)

        # No answer found
        if len(matches) == 0:
            return self.format_missing_reward

        # Multiple answers
        if len(matches) > 1:
            return self.format_multiple_reward

        # Check for hedging words near Final Answer
        has_or = " or " in response.lower() and "Final Answer:" in response
        if has_or:
            return self.format_multiple_reward

        # Clean format
        return self.format_clean_reward

    def _compute_length_reward(self, response: str) -> float:
        """Compute length reward based on character count.

        Piecewise linear:
        - ≤600 chars: 0.0 (neutral)
        - 600-1200: +0.1 (sweet spot)
        - 1200-2000: -0.1 (getting long)
        - >2000: -0.3 (rambling)
        """
        char_count = len(response)

        if char_count <= 600:
            return 0.0
        elif char_count <= 1200:
            return 0.1
        elif char_count <= 2000:
            return -0.1
        else:
            return -0.3

    def _to_float(self, text: str) -> float | None:
        """Convert text to float, handling various formats.

        Supports:
        - Plain numbers: "42", "54.00"
        - With currency: "$42"
        - With commas: "1,234"
        """
        try:
            text = text.strip()
            # Clean and convert
            cleaned_text = re.sub(r"[$,\s]", "", text)
            return float(cleaned_text)
        except (ValueError, AttributeError):
            return None
