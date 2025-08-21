import re
from typing import Optional

from forge.interfaces import Reward


class MathReward(Reward):
    """Reward class for evaluating math correctness."""

    def __init__(self, tolerance: float = 1e-6, partial_credit: float = 0.1):
        self.tolerance = tolerance
        self.partial_credit = partial_credit

    def _to_float(self, text) -> Optional[float]:
        """Safely parse a string into a float, or return None if invalid."""
        if text is None:
            return None
        try:
            return float(str(text).strip())
        except (ValueError, TypeError):
            return None

    def _extract_number(self, text: str) -> Optional[float]:
        """Try to extract a numeric answer from text."""
        number_pattern = r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
        patterns = [
            r"####\s*" + number_pattern,
            r"(?:the\s+)?answer\s+is\s*" + number_pattern,
            r"(?:answer:|result:)\s*" + number_pattern,
            r"\$" + number_pattern,  # currency
            number_pattern,  # fallback
            r"=\s*" + number_pattern + r"\s*(?:\.|$)",
            r"\b" + number_pattern + r"\s*(?:\.|$)",
        ]
        text = text.lower().strip()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return self._to_float(matches[-1])
        return None

    def __call__(self, prompt: str, response: str, target: str) -> float:
        """Compute math correctness reward."""
        # Parse expected
        expected_answer = self._to_float(target)

        # Parse response
        model_answer = self._extract_number(response)

        # Scoring
        if expected_answer is None or model_answer is None:
            return self.partial_credit  # Partial credit for attempting

        if abs(expected_answer - model_answer) < self.tolerance:
            return 1.0  # Correct answer
        return 0.0  # Incorrect answer
