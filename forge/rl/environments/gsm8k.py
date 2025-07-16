# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Optional

from forge.rl.environments.base import Observation, Transform
from forge.rl.environments.chat import ChatObservation

from forge.rl.environments.dataset import ColumnMapping, DatasetChatEnvironment


def extract_gsm8k_answer(text: str, method: str = "flexible") -> Optional[float]:
    """Extract the final numerical answer from GSM8K text.

    For GSM8K, answers are typically marked with #### followed by the number.
    In strict mode, only the #### format is accepted.
    In flexible mode, falls back to finding the last valid number in the text.

    This implementation handles commas, dollar signs, and filters invalid strings.

    Args:
        text: Text to extract answer from
        method: Extraction method - 'strict' (only #### format) or 'flexible' (fallback to last number)

    Returns:
        The extracted numerical answer, or None if not found
    """
    assert method in [
        "strict",
        "flexible",
    ], f"Method must be 'strict' or 'flexible', got {method}"

    # First try to find #### format
    match = re.search(r"####\s*([-]?[\$]?[0-9\.\,]+)", text)
    if match:
        try:
            answer_str = match.group(1).replace(",", "").replace("$", "")
            return float(answer_str)
        except ValueError:
            pass

    if method == "strict":
        return None

    # Fallback: find last valid number in text
    numbers = re.findall(r"([-]?[\$]?[0-9\.\,]+)", text)
    if numbers:
        invalid_str = ["", "."]
        for number_str in reversed(numbers):
            if number_str not in invalid_str:
                try:
                    clean_number = number_str.replace(",", "").replace("$", "")
                    return float(clean_number)
                except ValueError:
                    continue

    return None


class GSM8KRewardTransform(Transform):
    """Transform that adds rewards to observations by comparing model answers to expected answers.

    This transform extracts the final numerical answer from both the model's response and the
    expected answer, then compares them to determine if the model got the problem correct.
    """

    def __init__(self, tolerance: float = 1e-6, method: str = "flexible"):
        """Initialize the reward transform.

        Args:
            tolerance: Numerical tolerance for comparing floating point answers
            method: Extraction method - 'strict' (only #### format) or 'flexible' (fallback to last number)
        """
        self.tolerance = tolerance
        self.method = method

    def __call__(self, observation: Observation) -> Observation:
        """Apply reward transform to an observation.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation with reward added
        """
        # Only work with ChatObservation that has messages
        if not isinstance(observation, ChatObservation):
            return observation

        # Only compute reward if we have an assistant response
        if (
            len(observation.messages) < 2
            or observation.messages[-1]["role"] != "assistant"
        ):
            return observation

        # Get the expected answer from metadata
        expected_answer = None
        model_response = observation.messages[-1]["content"]

        # Look for expected answer in metadata
        if hasattr(observation, "metadata") and observation.metadata:
            # Try common metadata keys for the expected answer
            for key in ["answer", "expected_answer", "solution"]:
                if key in observation.metadata:
                    expected_answer = extract_gsm8k_answer(
                        observation.metadata[key], self.method
                    )
                    break

        # Extract model's answer
        model_answer = extract_gsm8k_answer(model_response, self.method)

        # Compute reward
        if model_answer is not None and expected_answer is not None:
            if abs(model_answer - expected_answer) <= self.tolerance:
                reward = 1.0
            else:
                reward = 0.0
        else:
            reward = 0.0

        # Create new observation with reward
        observation.reward = reward
        observation.done = True  # GSM8K is typically single-turn

        return observation


class GSM8KEnvironment(DatasetChatEnvironment):
    """Complete GSM8K environment with sensible defaults.

    This class provides a ready-to-use GSM8K environment that handles all the setup
    automatically. Users just need to provide a tokenizer and optionally customize
    parameters.

    Example usage:
        env = GSM8KEnvironment(tokenizer=my_tokenizer)
        obs = env.reset()  # Loads a math problem
        action = ChatAction(role="assistant", content="The answer is 42. #### 42")
        obs = env.step(action)  # Gets reward based on correctness
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        system_prompt: Optional[str] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        tolerance: float = 1e-6,
        method: str = "flexible",
        **dataset_kwargs,
    ):
        """Initialize GSM8K environment with sensible defaults.

        Args:
            tokenizer: Tokenizer for processing messages (required)
            split: Dataset split to use ("train", "test"). Defaults to "train"
            system_prompt: System prompt. Defaults to helpful math assistant prompt
            shuffle: Whether to shuffle dataset. Defaults to True
            seed: Random seed for reproducibility. Defaults to None (random)
            tolerance: Numerical tolerance for answer comparison. Defaults to 1e-6
            method: Answer extraction method ("strict" or "flexible"). Defaults to "flexible"
            **dataset_kwargs: Additional arguments passed to load_dataset

        Example:
            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            >>> env = GSM8KEnvironment(tokenizer=tokenizer)
            >>> obs = env.reset()  # Loads a math problem
            >>> action = ChatAction(role="assistant", content="The answer is 42. #### 42")
            >>> obs = env.step(action)  # Gets reward based on correctness
            >>> print(f"Reward: {obs.reward}")
        """

        # Set default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that solves math problems step by step."
            )

        # Define column mappings for GSM8K dataset
        column_mappings = [
            ColumnMapping(column="question", role="user"),
            ColumnMapping(column="answer", metadata_key="expected_answer"),
        ]

        # Initialize the parent DatasetChatEnvironment
        super().__init__(
            tokenizer=tokenizer,
            dataset_name="openai/gsm8k",
            column_mappings=column_mappings,
            system_prompt=system_prompt,
            split=split,
            shuffle=shuffle,
            seed=seed,
            transform=GSM8KRewardTransform(tolerance=tolerance, method=method),
            **dataset_kwargs,
        )
