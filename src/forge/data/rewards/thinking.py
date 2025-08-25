from typing import Optional

from forge.interfaces import Reward


class ThinkingReward(Reward):
    """Reward class for evaluating use of <think> tags in reasoning."""

    def __init__(self, reward_value: float = 0.5):
        self.reward_value = reward_value

    def __call__(
        self, prompt: str, response: str, target: Optional[str] = None
    ) -> float:
        """Check if response contains <think>...</think> tags."""
        resp = response.lower()
        if "<think>" in resp and "</think>" in resp:
            return self.reward_value
        return 0.0
