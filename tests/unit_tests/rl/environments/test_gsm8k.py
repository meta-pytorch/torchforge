# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset
from forge.rl.environments.chat import ChatAction, ChatObservation
from forge.rl.environments.gsm8k import (
    extract_gsm8k_answer,
    GSM8KEnvironment,
    GSM8KRewardTransform,
)


class TestExtractGSM8KAnswer:
    """Test the extract_gsm8k_answer function."""

    def test_extract_answer_with_hash_marks(self):
        """Test extracting answer with #### format."""
        text = "Let me solve this step by step.\nFirst, 2 + 2 = 4.\nTherefore, the answer is #### 4"
        answer = extract_gsm8k_answer(text)
        assert answer == 4.0

        text = "The calculation gives us #### 42.5"
        answer = extract_gsm8k_answer(text)
        assert answer == 42.5

        text = "Final answer: #### -10"
        answer = extract_gsm8k_answer(text)
        assert answer == -10.0

    def test_extract_answer_fallback_to_last_number(self):
        """Test fallback to last number when #### not found (flexible mode)."""
        text = "The answer is 25 dollars."
        answer = extract_gsm8k_answer(text, method="flexible")
        assert answer == 25.0

        text = "We have 10 apples and 15 oranges, so 25 total."
        answer = extract_gsm8k_answer(text, method="flexible")
        assert answer == 25.0

    def test_extract_answer_strict_mode(self):
        """Test strict mode only accepts #### format."""
        # Should work with #### format
        text = "The answer is #### 25"
        answer = extract_gsm8k_answer(text, method="strict")
        assert answer == 25.0

        # Should NOT work without #### format in strict mode
        text = "The answer is 25 dollars."
        answer = extract_gsm8k_answer(text, method="strict")
        assert answer is None

        text = "We have 10 apples and 15 oranges, so 25 total."
        answer = extract_gsm8k_answer(text, method="strict")
        assert answer is None

    def test_extract_answer_no_number(self):
        """Test when no number is found."""
        text = "I don't know the answer."
        answer = extract_gsm8k_answer(text)
        assert answer is None

    def test_extract_answer_with_commas(self):
        """Test extracting answers with comma-separated numbers."""
        # Test with #### format
        text = "The total cost is #### 1,234.56"
        answer = extract_gsm8k_answer(text)
        assert answer == 1234.56

        text = "Final answer: #### 1,000"
        answer = extract_gsm8k_answer(text)
        assert answer == 1000.0

        # Test fallback with commas
        text = "The total is 1,234.56 dollars."
        answer = extract_gsm8k_answer(text)
        assert answer == 1234.56

    def test_extract_answer_with_dollar_signs(self):
        """Test extracting answers with dollar signs."""
        # Test with #### format and dollar signs
        text = "The cost is #### $123.45"
        answer = extract_gsm8k_answer(text)
        assert answer == 123.45

        # Test fallback with dollar signs
        text = "She makes $18 every day."
        answer = extract_gsm8k_answer(text)
        assert answer == 18.0


class TestGSM8KRewardTransform:
    """Test the GSM8K reward transform."""

    def test_transform_with_correct_answer_in_metadata(self):
        """Test full transform with correct answer stored in metadata."""
        transform = GSM8KRewardTransform()

        obs = ChatObservation(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4. #### 4"},
            ],
            tokens=[],
            metadata={"expected_answer": "The answer is #### 4"},
        )

        result = transform(obs)
        assert isinstance(result, ChatObservation)
        assert result.reward == 1.0
        assert result.done is True

    def test_transform_with_incorrect_answer_in_metadata(self):
        """Test full transform with incorrect answer stored in metadata."""
        transform = GSM8KRewardTransform()

        obs = ChatObservation(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 5. #### 5"},
            ],
            tokens=[],
            metadata={"expected_answer": "The answer is #### 4"},
        )

        result = transform(obs)
        assert isinstance(result, ChatObservation)
        assert result.reward == 0.0
        assert result.done is True

    def test_transform_no_assistant_response(self):
        """Test transform when no assistant response yet."""
        transform = GSM8KRewardTransform()

        obs = ChatObservation(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            tokens=[],
            metadata={"expected_answer": "The answer is #### 4"},
        )

        result = transform(obs)
        assert result == obs  # Should return unchanged

    def test_transform_no_metadata(self):
        """Test transform when no metadata is available."""
        transform = GSM8KRewardTransform()

        obs = ChatObservation(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4. #### 4"},
            ],
            tokens=[],
            metadata={},  # Empty metadata
        )

        result = transform(obs)
        assert result.reward == 0.0  # No expected answer to compare against

    def test_transform_different_metadata_keys(self):
        """Test transform with different metadata keys for expected answer."""
        transform = GSM8KRewardTransform()

        # Test with 'answer' key
        obs = ChatObservation(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4. #### 4"},
            ],
            tokens=[],
            metadata={"answer": "The answer is #### 4"},
        )

        result = transform(obs)
        assert result.reward == 1.0

        # Test with 'solution' key
        obs = ChatObservation(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4. #### 4"},
            ],
            tokens=[],
            metadata={"solution": "The answer is #### 4"},
        )

        result = transform(obs)
        assert result.reward == 1.0

    def test_transform_tolerance(self):
        """Test reward computation with tolerance."""
        transform = GSM8KRewardTransform(tolerance=0.1)

        obs = ChatObservation(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4.05. #### 4.05"},
            ],
            tokens=[],
            metadata={"expected_answer": "The answer is #### 4"},
        )

        result = transform(obs)
        assert result.reward == 1.0  # Within tolerance

        # Test outside tolerance
        obs = ChatObservation(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4.2. #### 4.2"},
            ],
            tokens=[],
            metadata={"expected_answer": "The answer is #### 4"},
        )

        result = transform(obs)
        assert result.reward == 0.0  # Outside tolerance


class TestGSM8KEnvironment:
    """Test the complete GSM8KEnvironment class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        import torch

        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = torch.tensor(
            [[1, 2, 3]]
        )  # Mock token tensor
        tokenizer.decode.return_value = "The answer is 3. #### 3"  # Mock decoded text
        return tokenizer

    @pytest.fixture
    def sample_gsm8k_data(self):
        """Create sample GSM8K data."""
        return [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?",
                "answer": "Janet's ducks lay 16 eggs per day.\nShe eats 3 for breakfast every morning, so she has 16 - 3 = 13 eggs left.\nShe bakes 4 into muffins for her friends every day, so she has 13 - 4 = 9 eggs left.\nShe sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * 2 = $18 every day.\n#### 18",
            },
            {
                "question": "Tom has 5 apples. He gives 2 to his friend. How many apples does Tom have left?",
                "answer": "Tom starts with 5 apples.\nHe gives 2 to his friend.\nSo he has 5 - 2 = 3 apples left.\n#### 3",
            },
        ]

    @patch("forge.rl.environments.dataset.load_dataset")
    def test_gsm8k_environment_creation(
        self, mock_load_dataset, mock_tokenizer, sample_gsm8k_data
    ):
        """Test creating GSM8KEnvironment with defaults."""
        mock_dataset = Dataset.from_list(sample_gsm8k_data)
        mock_load_dataset.return_value = mock_dataset

        env = GSM8KEnvironment(tokenizer=mock_tokenizer)

        # Check that it was created successfully
        assert env is not None
        assert env.transform is not None

        # Verify load_dataset was called with correct parameters
        mock_load_dataset.assert_called_once_with("openai/gsm8k", split="train")

    @patch("forge.rl.environments.dataset.load_dataset")
    def test_gsm8k_environment_custom_params(
        self, mock_load_dataset, mock_tokenizer, sample_gsm8k_data
    ):
        """Test creating GSM8KEnvironment with custom parameters."""
        mock_dataset = Dataset.from_list(sample_gsm8k_data)
        mock_load_dataset.return_value = mock_dataset

        env = GSM8KEnvironment(
            tokenizer=mock_tokenizer,
            split="test",
            system_prompt="Custom prompt",
            shuffle=False,
            seed=42,
            tolerance=1e-3,
            method="strict",
        )

        # Check that it was created successfully
        assert env is not None

        # Verify load_dataset was called with correct parameters
        mock_load_dataset.assert_called_once_with("openai/gsm8k", split="test")

    @patch("forge.rl.environments.dataset.load_dataset")
    def test_gsm8k_environment_reset_and_step(
        self, mock_load_dataset, mock_tokenizer, sample_gsm8k_data
    ):
        """Test reset and step functionality."""
        mock_dataset = Dataset.from_list(sample_gsm8k_data)
        mock_load_dataset.return_value = mock_dataset

        env = GSM8KEnvironment(tokenizer=mock_tokenizer)

        # Test reset
        obs = env.reset()
        assert isinstance(obs, ChatObservation)
        assert len(obs.messages) >= 1  # At least system message
        assert "expected_answer" in obs.metadata

        # Extract expected answer from metadata
        expected_answer_text = obs.metadata["expected_answer"]
        expected_answer = extract_gsm8k_answer(expected_answer_text)

        # Create a mock action with the correct answer
        import torch

        action = ChatAction(tokens=torch.tensor([[1, 2, 3]]))
        action.metadata["original_role"] = "assistant"

        # Step with the action
        obs = env.step(action)
        assert isinstance(obs, ChatObservation)
        # Since we mocked the decode to return "The answer is 3. #### 3"
        # and the expected answer is 3, the reward should be 1.0
        assert obs.reward == 1.0
        assert obs.done is True

    @patch("forge.rl.environments.dataset.load_dataset")
    def test_gsm8k_environment_reset_dataloader(
        self, mock_load_dataset, mock_tokenizer, sample_gsm8k_data
    ):
        """Test reset_dataloader functionality."""
        mock_dataset = Dataset.from_list(sample_gsm8k_data)
        mock_load_dataset.return_value = mock_dataset

        env = GSM8KEnvironment(tokenizer=mock_tokenizer)

        # Test that reset_dataloader returns self for chaining
        result = env.reset_dataloader()
        assert result is env
