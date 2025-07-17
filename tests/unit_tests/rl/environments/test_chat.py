# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import torch

from forge.rl.environments.chat import (
    ChatAction,
    ChatEnvironment,
    ChatObservation,
    ChatState,
    Message,
)


class MockTokenizer:
    """Mock tokenizer implementing TokenizerProtocol for testing."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Mock implementation of apply_chat_template."""
        # For testing, we'll just return a tensor with a simple pattern based on the conversation
        # Each message contributes 10 tokens to the output
        return torch.tensor([[i for i in range(len(conversation) * 10)]])

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Mock implementation of decode."""
        # For testing, we'll just convert the tensor to a string
        if isinstance(token_ids, torch.Tensor):
            return f"Decoded: {token_ids.tolist()}"
        return f"Decoded: {token_ids}"


class TestChatAction(unittest.TestCase):
    """Test the ChatAction class."""

    def test_init(self):
        """Test initialization of ChatAction."""
        tokens = torch.tensor([1, 2, 3])
        action = ChatAction(tokens=tokens)
        self.assertTrue(torch.equal(action.tokens, tokens))

    def test_init_empty_tokens(self):
        """Test initialization with empty tokens raises ValueError."""
        with self.assertRaises(ValueError):
            ChatAction(tokens=torch.tensor([]))


class TestChatState(unittest.TestCase):
    """Test the ChatState class."""

    def test_init(self):
        """Test initialization of ChatState."""
        state = ChatState()
        self.assertEqual(state.history_messages, [])
        self.assertEqual(state.history_tokens, [])

    def test_init_with_values(self):
        """Test initialization with provided values."""
        messages: List[Message] = [{"role": "user", "content": "Hello"}]
        tokens = [torch.tensor([1, 2, 3])]
        state = ChatState(history_messages=messages, history_tokens=tokens)
        self.assertEqual(state.history_messages, messages)
        self.assertEqual(state.history_tokens, tokens)


class TestChatObservation(unittest.TestCase):
    """Test the ChatObservation class."""

    def test_init(self):
        """Test initialization of ChatObservation."""
        obs = ChatObservation()
        self.assertEqual(obs.messages, [])
        self.assertEqual(obs.tokens.numel(), 0)
        self.assertFalse(obs.done)
        self.assertIsNone(obs.reward)
        self.assertEqual(obs.metadata, {})

    def test_init_with_values(self):
        """Test initialization with provided values."""
        messages: List[Message] = [{"role": "user", "content": "Hello"}]
        tokens = torch.tensor([1, 2, 3])
        obs = ChatObservation(
            messages=messages,
            tokens=tokens,
            done=True,
            reward=1.0,
            metadata={"test": "value"},
        )
        self.assertEqual(obs.messages, messages)
        self.assertTrue(torch.equal(obs.tokens, tokens))
        self.assertTrue(obs.done)
        self.assertEqual(obs.reward, 1.0)
        self.assertEqual(obs.metadata, {"test": "value"})


class TestChatEnvironment(unittest.TestCase):
    """Test the ChatEnvironment class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = MockTokenizer()

    def test_init_no_system_prompt(self):
        """Test initialization without system prompt."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        self.assertEqual(env._state.history_messages, [])
        self.assertEqual(env._state.history_tokens, [])

    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        env = ChatEnvironment(
            tokenizer=self.tokenizer,
            system_prompt="You are a helpful assistant",
            system_role="system",
        )
        self.assertEqual(len(env._state.history_messages), 1)
        self.assertEqual(env._state.history_messages[0]["role"], "system")
        self.assertEqual(
            env._state.history_messages[0]["content"], "You are a helpful assistant"
        )
        self.assertEqual(len(env._state.history_tokens), 1)

    def test_init_invalid_tokenizer(self):
        """Test initialization with invalid tokenizer."""
        # Create a mock with no attributes by setting spec=[]
        invalid_tokenizer = MagicMock(spec=[])
        with self.assertRaises(ValueError):
            ChatEnvironment(tokenizer=invalid_tokenizer)

    def test_reset_no_system_prompt(self):
        """Test reset without system prompt."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        # Add some history first
        env._state.history_messages = [{"role": "user", "content": "Hello"}]  # type: ignore
        env._state.history_tokens = [torch.tensor([1, 2, 3])]

        # Reset should clear the history
        obs = env.reset()
        self.assertEqual(env._state.history_messages, [])
        self.assertEqual(env._state.history_tokens, [])
        self.assertEqual(obs.messages, [])
        self.assertEqual(obs.tokens.numel(), 0)

    def test_reset_with_system_prompt(self):
        """Test reset with system prompt."""
        env = ChatEnvironment(
            tokenizer=self.tokenizer,
            system_prompt="You are a helpful assistant",
            system_role="system",
        )
        # Add some history first
        env._state.history_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]  # type: ignore
        env._state.history_tokens = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

        # Reset should clear the history and add the system prompt
        obs = env.reset()
        self.assertEqual(len(env._state.history_messages), 1)
        self.assertEqual(env._state.history_messages[0]["role"], "system")
        self.assertEqual(
            env._state.history_messages[0]["content"], "You are a helpful assistant"
        )
        self.assertEqual(len(env._state.history_tokens), 1)
        self.assertEqual(len(obs.messages), 1)
        self.assertEqual(obs.messages[0]["role"], "system")
        self.assertEqual(obs.messages[0]["content"], "You are a helpful assistant")

    def test_step(self):
        """Test step method."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        action = ChatAction(tokens=torch.tensor([1, 2, 3]))

        obs = env.step(action)

        # Check that the tokens were added to history
        self.assertEqual(len(env._state.history_tokens), 1)
        self.assertTrue(
            torch.equal(env._state.history_tokens[0], torch.tensor([1, 2, 3]))
        )

        # Check that the message was added to history with decoded content
        self.assertEqual(len(env._state.history_messages), 1)
        self.assertEqual(env._state.history_messages[0]["role"], "assistant")
        self.assertEqual(
            env._state.history_messages[0]["content"], "Decoded: [1, 2, 3]"
        )

        # Check the observation
        self.assertEqual(len(obs.messages), 1)
        self.assertEqual(obs.messages[0]["role"], "assistant")
        self.assertEqual(obs.messages[0]["content"], "Decoded: [1, 2, 3]")

    def test_create_observation(self):
        """Test _create_observation method."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        env._state.history_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]  # type: ignore
        env._state.history_tokens = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5, 6]]),
        ]

        obs = env._create_observation()

        # Check the observation
        self.assertEqual(len(obs.messages), 2)
        self.assertEqual(obs.messages[0]["role"], "system")
        self.assertEqual(obs.messages[0]["content"], "You are a helpful assistant")
        self.assertEqual(obs.messages[1]["role"], "user")
        self.assertEqual(obs.messages[1]["content"], "Hello")

        # Check that the tokens were concatenated
        self.assertEqual(obs.tokens.numel(), 6)  # 2 tensors of size 3

    def test_create_observation_empty_history(self):
        """Test _create_observation method with empty history."""
        env = ChatEnvironment(tokenizer=self.tokenizer)

        obs = env._create_observation()

        # Check the observation
        self.assertEqual(obs.messages, [])
        self.assertEqual(obs.tokens.numel(), 0)

    def test_state_property(self):
        """Test state property."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        state = env.state
        self.assertIsInstance(state, ChatState)
        self.assertEqual(state.history_messages, [])
        self.assertEqual(state.history_tokens, [])

    def test_message_to_action(self):
        """Test message_to_action method."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        message: Message = {"role": "user", "content": "Hello"}

        action = env.message_to_action(message)

        self.assertIsInstance(action, ChatAction)
        self.assertEqual(
            action.tokens.numel(), 10
        )  # Mock tokenizer returns 10 tokens per message

    def test_message_to_action_missing_role(self):
        """Test message_to_action method with missing role."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        # We're intentionally creating an invalid message to test error handling
        message = {"content": "Hello"}  # type: ignore

        with self.assertRaises(ValueError):
            # Using type: ignore because we're intentionally passing an invalid message
            env.message_to_action(message)  # type: ignore

    def test_message_to_action_missing_content(self):
        """Test message_to_action method with missing content."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        # We're intentionally creating an invalid message to test error handling
        message = {"role": "user"}  # type: ignore

        with self.assertRaises(ValueError):
            # Using type: ignore because we're intentionally passing an invalid message
            env.message_to_action(message)  # type: ignore

    def test_message_to_action_none_content(self):
        """Test message_to_action method with None content."""
        env = ChatEnvironment(tokenizer=self.tokenizer)
        # We're intentionally creating an invalid message to test error handling
        message = {"role": "user", "content": None}  # type: ignore

        with self.assertRaises(ValueError):
            # Using type: ignore because we're intentionally passing an invalid message
            env.message_to_action(message)  # type: ignore

    def test_with_transform(self):
        """Test environment with a transform."""

        def transform(obs):
            obs.metadata["transformed"] = True
            obs.reward = 1.0
            return obs

        env = ChatEnvironment(tokenizer=self.tokenizer, transform=transform)
        action = ChatAction(tokens=torch.tensor([1, 2, 3]))

        obs = env.step(action)

        self.assertTrue(obs.metadata.get("transformed"))
        self.assertEqual(obs.reward, 1.0)


if __name__ == "__main__":
    unittest.main()
