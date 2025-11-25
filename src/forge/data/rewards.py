# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
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
    """Reward class for evaluating use of thinking tags in reasoning.

    Args:
        partial_reward: Reward for partial tag usage (incomplete/malformed)
        full_reward: Reward for well-formed thinking blocks with content
        tag: Tag name to use (default "think", can use "思考" for Japanese, etc.)
    """

    def __init__(
        self, partial_reward: float = 0.2, full_reward: float = 1.0, tag: str = "think"
    ):
        self.partial_reward = partial_reward
        self.full_reward = full_reward
        self.tag = tag
        # Build regex patterns for the specified tag
        self._THINK_BLOCK_RE = re.compile(
            rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
            re.IGNORECASE | re.DOTALL,
        )
        self._THINK_TAG_ATTEMPT_RE = re.compile(
            rf"<\s*/?\s*{re.escape(tag)}\s*>", re.IGNORECASE
        )

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


class LanguageReward:
    """Reward class for evaluating the language used in responses.

    This reward uses langid to detect the language and rewards responses that use
    the target language. The detection strategy depends on the format:
    - If exactly one thinking block: detect language of the block content
    - Otherwise (no blocks or multiple blocks): detect language of whole response

    Note: Format enforcement (single vs multiple blocks) is handled by ThinkingReward.
    This reward focuses purely on language detection.

    Args:
        target_language: ISO 639-1 language code (e.g., 'en', 'ja', 'zh', 'es')
        match_reward: Reward when detected language matches target (default: 1.0)
        no_match_reward: Reward when language doesn't match (default: 0.0)
        tag: Tag name to use (default "思考" for multilingual, can use "think", etc.)
        debug: If True, print debug samples showing model outputs and detected language
        debug_sample_rate: Fraction of calls to debug (e.g., 0.1 = 10% of calls)

    Note: Requires langid to be installed. Install with: pip install langid
    """

    def __init__(
        self,
        target_language: str = "ja",
        match_reward: float = 1.0,
        no_match_reward: float = 0.0,
        tag: str = "思考",
        debug: bool = False,
        debug_sample_rate: float = 0.1,
    ):
        self.target_language = target_language
        self.match_reward = match_reward
        self.no_match_reward = no_match_reward
        self.tag = tag
        self.debug = debug
        self.debug_sample_rate = debug_sample_rate
        self._debug_counter = 0
        # Build regex pattern for the specified tag
        self._THINK_BLOCK_RE = re.compile(
            rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>", re.DOTALL
        )

        # Lazy import langid with helpful error message
        try:
            import langid

            self._langid = langid
        except ImportError:
            raise ImportError(
                "langid is required for LanguageReward but is not installed. "
                "Please install it with: pip install langid"
            ) from None

    def __call__(self, prompt: str, response: str, target: str | None = None) -> float:
        """Compute language reward based on detected language.

        Detection strategy:
        - If exactly one thinking block: detect language of block content
        - Otherwise: detect language of whole response

        Args:
            prompt: The input prompt (unused but kept for signature consistency)
            response: The model response
            target: Optional target string (unused but kept for signature consistency)

        Returns:
            match_reward if detected language matches target, no_match_reward otherwise
        """

        # TODO: refactor pending https://github.com/meta-pytorch/torchforge/issues/187
        should_debug = self.debug and (random.random() < self.debug_sample_rate)

        if not response:
            if should_debug:
                print(
                    f"\n[LanguageReward] Empty response | Reward: {self.no_match_reward}"
                )
            return self.no_match_reward

        # Extract all thinking blocks
        matches = self._THINK_BLOCK_RE.findall(response)

        # Determine what text to analyze
        if len(matches) == 1:
            # Single block: detect language of block content only
            text_to_analyze = matches[0].strip()
            detection_mode = "single block"
        else:
            # No blocks or multiple blocks: detect language of whole response
            text_to_analyze = response.strip()
            detection_mode = f"{len(matches)} blocks, using whole response"

        # Remove extra whitespace
        text_to_analyze = re.sub(r"\s+", " ", text_to_analyze).strip()

        if not text_to_analyze:
            if should_debug:
                print(f"\n[LanguageReward] Empty text | Reward: {self.no_match_reward}")
            return self.no_match_reward

        # Detect language using langid
        detected_lang, confidence = self._langid.classify(text_to_analyze)

        # Check if language matches target
        reward = (
            self.match_reward
            if detected_lang == self.target_language
            else self.no_match_reward
        )

        if should_debug:
            sample = text_to_analyze[:1000].replace("\n", " ")
            match_symbol = "✓" if detected_lang == self.target_language else "✗"
            print(
                f"\n[LanguageReward] Detection mode: {detection_mode}"
                f"\n  Target: {self.target_language} | Detected: {detected_lang} | "
                f"Confidence: {confidence:.2f}"
                f"\n  Sample: {sample}..."
                f"\n  → Reward: {reward} {match_symbol}"
            )

        return reward
