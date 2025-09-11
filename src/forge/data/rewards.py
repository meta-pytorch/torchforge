# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

from forge.interfaces import Reward


class MathReward(Reward):
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


class ThinkingReward(Reward):
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
