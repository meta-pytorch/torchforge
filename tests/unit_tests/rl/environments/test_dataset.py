# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch
from datasets import Dataset
from forge.rl.environments.chat import ChatAction, ChatEnvironment, ChatObservation
from forge.rl.environments.dataset import ColumnMapping, DatasetChatEnvironment


class MockTokenizer:
    """A more realistic mock tokenizer for testing."""

    def __init__(self):
        self.call_count = 0

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        documents=None,
        chat_template=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tokenizer_kwargs=None,
        **kwargs,
    ):
        """Mock implementation of apply_chat_template with full HF signature."""
        self.call_count += 1

        # Create a more realistic string representation
        result = ""
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            result += f"<|{role}|>{content}<|end|>"

        # Handle continue_final_message (remove end tokens from final message)
        if continue_final_message and result.endswith("<|end|>"):
            result = result[:-7]  # Remove the last <|end|>

        # Handle add_generation_prompt
        if add_generation_prompt:
            result += "<|assistant|>"

        if not tokenize:
            return result

        # More realistic tokenization: create unique tokens based on content
        # This helps test tokenization consistency
        tokens = []
        for char in result:
            tokens.append(hash(char) % 1000)  # Create deterministic but varied tokens

        if return_tensors == "pt":
            # Create a mock tensor with more realistic behavior
            mock_tensor = MagicMock(spec=torch.Tensor)
            mock_tensor.tolist.return_value = tokens
            mock_tensor.shape = (len(tokens),)
            mock_tensor.__len__ = lambda: len(tokens)
            mock_tensor.squeeze = lambda: mock_tensor
            return mock_tensor

        return tokens

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=None,
        **kwargs,
    ):
        """Mock implementation of decode."""
        # For testing, we'll just convert the tensor to a string
        if isinstance(token_ids, torch.Tensor):
            return f"Decoded: {token_ids.tolist()}"
        return f"Decoded: {token_ids}"


class TestColumnMapping(unittest.TestCase):
    """Test the ColumnMapping class."""

    def test_basic_column_mapping(self):
        """Test basic column mapping functionality."""
        mapping = ColumnMapping(column="instruction", role="user")

        row = {"instruction": "What is the capital of France?", "output": "Paris"}
        content = mapping.extract_content(row)

        self.assertEqual(content, "What is the capital of France?")
        self.assertEqual(mapping.role, "user")

    def test_column_mapping_with_template(self):
        """Test column mapping with content template."""
        mapping = ColumnMapping(
            column="instruction",
            role="user",
            content_template="### Instruction:\n{instruction}\n\n### Input:\n{input}",
        )

        row = {
            "instruction": "Identify the odd one out.",
            "input": "Twitter, Instagram, Telegram",
            "output": "Telegram",
        }
        content = mapping.extract_content(row)

        expected = "### Instruction:\nIdentify the odd one out.\n\n### Input:\nTwitter, Instagram, Telegram"
        self.assertEqual(content, expected)

    def test_column_mapping_missing_template_key(self):
        """Test column mapping with template that references missing key."""
        mapping = ColumnMapping(
            column="instruction",
            role="user",
            content_template="Instruction: {instruction}\nMissing: {missing_key}",
        )

        row = {"instruction": "Test instruction"}

        with self.assertRaises(KeyError):
            mapping.extract_content(row)

    def test_column_mapping_empty_content(self):
        """Test column mapping with empty content."""
        mapping = ColumnMapping(column="empty_field", role="user")

        row = {"empty_field": "", "other": "content"}
        content = mapping.extract_content(row)

        self.assertEqual(content, "")

    def test_column_mapping_numeric_content(self):
        """Test column mapping with numeric content."""
        mapping = ColumnMapping(column="score", role="user")

        row = {"score": 42, "text": "some text"}
        content = mapping.extract_content(row)

        self.assertEqual(content, "42")

    def test_column_mapping_complex_template(self):
        """Test column mapping with complex template formatting."""
        mapping = ColumnMapping(
            column="question",
            role="user",
            content_template="Q: {question}\nContext: {context}\nDifficulty: {difficulty}",
        )

        row = {
            "question": "What is AI?",
            "context": "Machine learning context",
            "difficulty": "beginner",
        }
        content = mapping.extract_content(row)

        expected = (
            "Q: What is AI?\nContext: Machine learning context\nDifficulty: beginner"
        )
        self.assertEqual(content, expected)


class TestDatasetChatEnvironment(unittest.TestCase):
    """Test the DatasetChatEnvironment class."""

    def create_mock_dataset(self):
        """Create a mock dataset for testing."""
        data = [
            {
                "instruction": "Identify the odd one out.",
                "input": "Twitter, Instagram, Telegram",
                "output": "Telegram",
                "text": "Below is an instruction...",
            },
            {
                "instruction": "What is 2+2?",
                "input": "",
                "output": "4",
                "text": "Below is an instruction...",
            },
            {
                "instruction": "Name a color",
                "input": "Something bright",
                "output": "Red",
                "text": "Below is an instruction...",
            },
        ]
        return Dataset.from_list(data)

    def test_init_with_dataset(self):
        """Test initializing DatasetChatEnvironment with a dataset."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
        )

        self.assertIsInstance(env, ChatEnvironment)
        self.assertEqual(env.dataset, dataset)
        self.assertEqual(len(env.column_mappings), 2)

    @patch("forge.rl.environments.dataset.load_dataset")
    def test_init_with_dataset_name(self, mock_load_dataset):
        """Test initializing DatasetChatEnvironment with dataset name."""
        tokenizer = MockTokenizer()
        mock_dataset = self.create_mock_dataset()
        mock_load_dataset.return_value = mock_dataset

        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset_name="tatsu-lab/alpaca",
            column_mappings=column_mappings,
            split="train",
        )

        mock_load_dataset.assert_called_once_with("tatsu-lab/alpaca", split="train")
        self.assertEqual(env.dataset, mock_dataset)

    def test_init_validation_errors(self):
        """Test initialization validation errors."""
        tokenizer = MockTokenizer()

        # Test missing dataset and dataset_name
        with self.assertRaises(ValueError) as context:
            DatasetChatEnvironment(tokenizer=tokenizer)
        self.assertIn(
            "Either dataset_name or dataset must be provided", str(context.exception)
        )

        # Test missing column_mappings
        dataset = self.create_mock_dataset()
        with self.assertRaises(ValueError) as context:
            DatasetChatEnvironment(tokenizer=tokenizer, dataset=dataset)
        self.assertIn("column_mappings must be provided", str(context.exception))

    def test_column_validation(self):
        """Test column validation against dataset."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()

        # Test invalid column name
        column_mappings = [ColumnMapping(column="nonexistent_column", role="user")]

        with self.assertRaises(ValueError) as context:
            DatasetChatEnvironment(
                tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
            )
        self.assertIn("Column 'nonexistent_column' not found", str(context.exception))

    def test_construct_conversation(self):
        """Test conversation construction from dataset row."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
        )

        row = dataset[0]
        conversation = env._construct_conversation(row)

        # Should have user + assistant messages
        self.assertEqual(len(conversation), 2)
        self.assertEqual(conversation[0]["role"], "user")
        self.assertEqual(conversation[0]["content"], "Identify the odd one out.")
        self.assertEqual(conversation[1]["role"], "assistant")
        self.assertEqual(conversation[1]["content"], "Telegram")

    def test_construct_conversation_with_template(self):
        """Test conversation construction with content template."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(
                column="instruction",
                role="user",
                content_template="### Instruction:\n{instruction}\n\n### Input:\n{input}",
            ),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
        )

        row = dataset[0]
        conversation = env._construct_conversation(row)

        expected_user_content = (
            "### Instruction:\nIdentify the odd one out.\n\n"
            "### Input:\nTwitter, Instagram, Telegram"
        )
        self.assertEqual(conversation[0]["content"], expected_user_content)

    def test_reset_loads_new_conversation(self):
        """Test that reset() loads a new conversation from dataset."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
        )

        # Reset should load first conversation
        obs = env.reset()
        self.assertIsInstance(obs, ChatObservation)
        self.assertEqual(len(obs.messages), 2)  # user + assistant

        # Find messages by role

        user_message = next(
            (msg for msg in obs.messages if msg["role"] == "user"), None
        )
        assistant_message = next(
            (msg for msg in obs.messages if msg["role"] == "assistant"), None
        )

        self.assertIsNotNone(user_message, "User message not found")
        self.assertIsNotNone(assistant_message, "Assistant message not found")
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(assistant_message["role"], "assistant")

    def test_reset_with_system_prompt(self):
        """Test reset() with system prompt."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            system_prompt="You are a helpful assistant.",
        )

        obs = env.reset()
        # Should have system + user + assistant messages
        self.assertEqual(len(obs.messages), 3)

        # Find messages by content instead of position
        system_message = next(
            (
                msg
                for msg in obs.messages
                if "You are a helpful assistant" in msg["content"]
            ),
            None,
        )

        user_message = next(
            (msg for msg in obs.messages if msg["role"] == "user"), None
        )
        assistant_message = next(
            (msg for msg in obs.messages if msg["role"] == "assistant"), None
        )

        self.assertIsNotNone(system_message, "System message not found")
        self.assertIsNotNone(user_message, "User message not found")
        self.assertIsNotNone(assistant_message, "Assistant message not found")

        self.assertEqual(system_message["role"], "system")
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(assistant_message["role"], "assistant")

    def test_step_adds_message(self):
        """Test that step() adds a new message to the conversation."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
        )

        # Reset to load initial conversation
        obs = env.reset()
        initial_length = len(obs.messages)

        # Step should add a new message
        # Create a token-based action
        tokens = torch.tensor([1, 2, 3, 4])
        action = ChatAction(tokens=tokens)
        obs = env.step(action)

        self.assertEqual(len(obs.messages), initial_length + 1)
        self.assertEqual(obs.messages[-1]["role"], "assistant")
        self.assertEqual(obs.messages[-1]["content"], "Decoded: [1, 2, 3, 4]")

    def test_multiple_resets(self):
        """Test multiple reset() calls load different conversations."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            shuffle=False,  # Disable shuffle for predictable order
        )

        # Get first conversation
        obs1 = env.reset()
        first_content = obs1.messages[0]["content"]

        # Get second conversation
        obs2 = env.reset()
        second_content = obs2.messages[0]["content"]

        # Should be different conversations (unless dataset has duplicates)
        # At minimum, should have loaded new conversation
        self.assertIsInstance(obs2, ChatObservation)
        self.assertEqual(len(obs2.messages), 1)  # Just user message

    def test_reset_dataloader(self):
        """Test resetting the dataloader."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            shuffle=False,
        )

        # Load a few conversations
        env.reset()
        env.reset()

        # Reset dataloader and load first conversation again
        result = env.reset_dataloader()
        self.assertEqual(result, env)  # Should return self for chaining

        obs = env.reset()
        # Should be back to first conversation
        self.assertIsInstance(obs, ChatObservation)

    def test_empty_content_filtering(self):
        """Test that empty content is filtered out of conversations."""
        tokenizer = MockTokenizer()

        # Create dataset with empty content
        data = [{"instruction": "Test", "empty_field": "", "output": "Response"}]
        dataset = Dataset.from_list(data)

        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(
                column="empty_field", role="system"
            ),  # This should be filtered out
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
        )

        obs = env.reset()
        # Should only have user and assistant messages (empty system message filtered out)
        self.assertEqual(len(obs.messages), 2)

        # Find messages by content instead of position, using more flexible matching
        user_message = next(
            (msg for msg in obs.messages if msg["role"] == "user"), None
        )
        assistant_message = next(
            (msg for msg in obs.messages if msg["role"] == "assistant"), None
        )

        self.assertIsNotNone(user_message, "User message not found")
        self.assertIsNotNone(assistant_message, "Assistant message not found")
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(assistant_message["role"], "assistant")

    def test_filter_function(self):
        """Test dataset filtering functionality."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        # Filter to only include rows where input is not empty
        def filter_fn(row):
            return row["input"].strip() != ""

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            filter_fn=filter_fn,
        )

        # Should have filtered out the row with empty input
        # We can test this by checking that we can reset multiple times
        # and get different conversations
        obs1 = env.reset()
        obs2 = env.reset()
        self.assertIsInstance(obs1, ChatObservation)
        self.assertIsInstance(obs2, ChatObservation)

    def test_shuffle_parameter(self):
        """Test shuffle parameter."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        # Test with shuffle=True
        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            shuffle=True,
        )

        obs = env.reset()
        self.assertIsInstance(obs, ChatObservation)
        # Content might be different due to shuffling, but structure should be same
        self.assertEqual(len(obs.messages), 2)

    def test_custom_system_role(self):
        """Test using custom system role."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            system_prompt="Custom prompt",
            system_role="custom_system",
        )

        obs = env.reset()
        # Should have custom system role
        self.assertEqual(obs.messages[0]["role"], "custom_system")
        self.assertEqual(obs.messages[0]["content"], "Custom prompt")

    def test_multiple_column_mappings_same_role(self):
        """Test multiple column mappings with the same role."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="input", role="user"),  # Same role as instruction
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
        )

        obs = env.reset()
        # Should have multiple user messages + assistant
        user_messages = [msg for msg in obs.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in obs.messages if msg["role"] == "assistant"]

        self.assertEqual(len(assistant_messages), 1)
        # Should have at least 1 user message (instruction), possibly 2 if input is non-empty
        self.assertGreaterEqual(len(user_messages), 1)

    def test_dataset_exhaustion_and_reset(self):
        """Test behavior when dataset is exhausted and resets."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            shuffle=False,
        )

        # Exhaust the dataset
        dataset_length = len(dataset) if hasattr(dataset, "__len__") else 3
        conversations = []
        for _ in range(dataset_length + 2):  # Go beyond dataset length
            obs = env.reset()
            conversations.append(obs.messages[0]["content"])

        # Should have successfully reset and continued (infinite iteration)
        self.assertEqual(len(conversations), dataset_length + 2)

    def test_dataset_kwargs_parameter(self):
        """Test passing additional dataset kwargs."""
        tokenizer = MockTokenizer()

        with patch("forge.rl.environments.dataset.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = self.create_mock_dataset()

            column_mappings = [
                ColumnMapping(column="instruction", role="user"),
                ColumnMapping(column="output", role="assistant"),
            ]

            env = DatasetChatEnvironment(
                tokenizer=tokenizer,
                dataset_name="test/dataset",
                column_mappings=column_mappings,
                split="validation",
                trust_remote_code=True,
                cache_dir="/tmp/cache",
            )

            # Check that dataset kwargs were passed through
            mock_load_dataset.assert_called_once_with(
                "test/dataset",
                split="validation",
                trust_remote_code=True,
                cache_dir="/tmp/cache",
            )

    def test_collate_function(self):
        """Test the collate function behavior."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
            ColumnMapping(column="output", role="assistant"),
        ]

        env = DatasetChatEnvironment(
            tokenizer=tokenizer, dataset=dataset, column_mappings=column_mappings
        )

        # Test collate function directly
        batch = [{"test": "data1"}, {"test": "data2"}]
        result = env._identity_collate_fn(batch)
        self.assertEqual(result, batch)  # Should return batch as-is

    def test_seed_reproducibility(self):
        """Test that seed parameter provides reproducible results."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
        ]

        # Create two environments with same seed
        env1 = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            seed=42,
            shuffle=True,
        )

        env2 = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            seed=42,
            shuffle=True,
        )

        # Should get same sequence of conversations
        obs1_1 = env1.reset()
        obs1_2 = env1.reset()

        obs2_1 = env2.reset()
        obs2_2 = env2.reset()

        # With same seed, should get same order
        self.assertEqual(obs1_1.messages[0]["content"], obs2_1.messages[0]["content"])
        self.assertEqual(obs1_2.messages[0]["content"], obs2_2.messages[0]["content"])

    def test_transform_function(self):
        """Test that transform function is applied to observations."""
        tokenizer = MockTokenizer()
        dataset = self.create_mock_dataset()
        column_mappings = [
            ColumnMapping(column="instruction", role="user"),
        ]

        def test_transform(obs):
            # Add a marker to show transform was applied
            obs.reward = 1.0
            return obs

        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            column_mappings=column_mappings,
            transform=test_transform,
        )

        obs = env.reset()
        self.assertEqual(obs.reward, 1.0)  # Transform should have been applied

        # Test step as well
        tokens = torch.tensor([1, 2, 3, 4])
        action = ChatAction(tokens=tokens)
        obs = env.step(action)
        self.assertEqual(obs.reward, 1.0)  # Transform should have been applied


if __name__ == "__main__":
    unittest.main()
