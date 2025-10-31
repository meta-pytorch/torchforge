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


class LanguageReward:
    """Reward class for evaluating the language used in <think> tags.

    This reward uses langid to detect the language of text within thinking blocks
    and rewards responses that use the target language.

    Args:
        target_language: ISO 639-1 language code (e.g., 'en', 'ja', 'zh', 'es')
        full_reward: Reward when language matches and format is correct (single block)
        partial_reward: Reward when language matches but format is wrong (multiple blocks)
        fallback_reward: Reward when no valid blocks but response text is in target language
        no_match_reward: Reward when language doesn't match
        debug: If True, print debug samples showing model outputs and detected language
        debug_sample_rate: Fraction of calls to debug (e.g., 0.1 = 10% of calls)

    Note: Requires langid to be installed. Install with: pip install langid
    """

    def __init__(
        self,
        target_language: str = "en",
        full_reward: float = 1.0,
        partial_reward: float = 0.5,
        fallback_reward: float = 0.2,
        no_match_reward: float = 0.0,
        debug: bool = False,
        debug_sample_rate: float = 0.1,
    ):
        self.target_language = target_language
        self.full_reward = full_reward
        self.partial_reward = partial_reward
        self.fallback_reward = fallback_reward
        self.no_match_reward = no_match_reward
        self.debug = debug
        self.debug_sample_rate = debug_sample_rate
        self._debug_counter = 0
        self._THINK_BLOCK_RE = re.compile(
            r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL
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
        """Compute language reward based on thinking block content.

        Args:
            prompt: The input prompt (unused but kept for signature consistency)
            response: The model response containing <think> tags
            target: Optional target string (unused but kept for signature consistency)

        Returns:
            full_reward if language matches and exactly one thinking block is found,
            partial_reward if language matches but multiple thinking blocks found,
            fallback_reward if no valid blocks but response text is in target language,
            no_match_reward otherwise (wrong language)
        """
        # Increment counter for sampling
        self._debug_counter += 1
        should_debug = (
            self.debug
            and self.debug_sample_rate > 0
            and (self._debug_counter % int(1 / self.debug_sample_rate)) == 0
        )

        if not response:
            return self.no_match_reward

        # Extract all thinking blocks
        matches = self._THINK_BLOCK_RE.findall(response)

        # If no thinking blocks found, check if response text is in target language
        if len(matches) == 0:
            # Remove any partial tags that might exist
            response_text = re.sub(
                r"<\s*/?\s*think\s*>", "", response, flags=re.IGNORECASE
            ).strip()

            if not response_text:
                if should_debug:
                    print(
                        f"\n[LanguageReward] Empty response | Reward: {self.no_match_reward}"
                    )
                return self.no_match_reward

            # Detect language of general response
            detected_lang, confidence = self._langid.classify(response_text)

            if should_debug:
                sample = response[:150].replace("\n", " ")
                print(
                    f"\n[LanguageReward] No thinking blocks found (FALLBACK mode)"
                    f"\n  Target: {self.target_language} | Detected: {detected_lang} | "
                    f"Confidence: {confidence:.2f}"
                    f"\n  Sample: {sample}..."
                )

            # Give fallback reward if response is in target language
            if detected_lang == self.target_language:
                if should_debug:
                    print(
                        f"  → Reward: {self.fallback_reward} (fallback, correct language)"
                    )
                return self.fallback_reward

            if should_debug:
                print(f"  → Reward: {self.no_match_reward} (wrong language)")
            return self.no_match_reward

        # Concatenate all thinking blocks for language detection
        thinking_content = " ".join(matches)

        # Remove extra whitespace
        thinking_content = re.sub(r"\s+", " ", thinking_content).strip()

        if not thinking_content:
            if should_debug:
                print(
                    f"\n[LanguageReward] Empty thinking blocks | Reward: {self.no_match_reward}"
                )
            return self.no_match_reward

        # Detect language using langid
        detected_lang, confidence = self._langid.classify(thinking_content)

        if should_debug:
            sample = thinking_content[:150].replace("\n", " ")
            print(
                f"\n[LanguageReward] Found {len(matches)} thinking block(s)"
                f"\n  Target: {self.target_language} | Detected: {detected_lang} | "
                f"Confidence: {confidence:.2f}"
                f"\n  Thinking sample: {sample}..."
            )

        # Check if language matches target
        if detected_lang == self.target_language:
            # Full reward for correct format (single block)
            if len(matches) == 1:
                if should_debug:
                    print(
                        f"  → Reward: {self.full_reward} (single block, correct language) ✓"
                    )
                return self.full_reward
            # Partial reward for wrong format (multiple blocks) but correct language
            else:
                if should_debug:
                    print(
                        f"  → Reward: {self.partial_reward} (multiple blocks, correct language)"
                    )
                return self.partial_reward

        if should_debug:
            print(f"  → Reward: {self.no_match_reward} (wrong language) ✗")
        return self.no_match_reward
