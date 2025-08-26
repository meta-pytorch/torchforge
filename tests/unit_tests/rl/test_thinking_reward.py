# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from forge.data.rewards import ThinkingReward


class TestThinkingReward(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reward = ThinkingReward()
        self.custom_reward = ThinkingReward(reward_value=0.8)

    def test_init_default_values(self):
        """Test ThinkingReward initialization with default values."""
        reward = ThinkingReward()
        self.assertEqual(reward.reward_value, 0.5)

    def test_init_custom_values(self):
        """Test ThinkingReward initialization with custom values."""
        reward = ThinkingReward(reward_value=0.8)
        self.assertEqual(reward.reward_value, 0.8)

    def test_call_with_both_tags(self):
        """Test __call__ with response containing both <think> and </think> tags."""
        response = "<think>This is my reasoning</think>"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

        result = self.custom_reward("prompt", response)
        self.assertEqual(result, 0.8)

    def test_call_with_both_tags_complex_content(self):
        """Test __call__ with complex content between thinking tags."""
        response = """
        Let me solve this problem step by step.
        <think>
        First, I need to understand what the question is asking.
        Then I'll work through the calculation:
        2 + 2 = 4
        So the answer should be 4.
        </think>
        The answer is 4.
        """
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

    def test_call_with_only_opening_tag(self):
        """Test __call__ with response containing only <think> tag."""
        response = "<think>This is incomplete reasoning"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.0)

    def test_call_with_only_closing_tag(self):
        """Test __call__ with response containing only </think> tag."""
        response = "This is incomplete reasoning</think>"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.0)

    def test_call_with_no_tags(self):
        """Test __call__ with response containing no thinking tags."""
        response = "This is just a regular response without any thinking tags."
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.0)

    def test_call_case_insensitive(self):
        """Test __call__ is case insensitive for thinking tags."""
        # Mixed case tags should work
        response = "<THINK>This is my reasoning</THINK>"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

        response = "<Think>This is my reasoning</Think>"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

        response = "<think>This is my reasoning</THINK>"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

    def test_call_multiple_thinking_blocks(self):
        """Test __call__ with multiple thinking blocks."""
        response = """
        <think>First thought</think>
        Some text in between.
        <think>Second thought</think>
        """
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

    def test_call_nested_tags(self):
        """Test __call__ with nested or malformed tags."""
        # Nested tags - should still work as long as both tags exist
        response = "<think>Outer <think>inner</think> thought</think>"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

    def test_call_empty_thinking_block(self):
        """Test __call__ with empty thinking block."""
        response = "<think></think>"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.5)

    def test_call_empty_response(self):
        """Test __call__ with empty response."""
        result = self.reward("prompt", "")
        self.assertEqual(result, 0.0)

    def test_call_tags_with_extra_whitespace(self):
        """Test __call__ with thinking tags containing extra whitespace."""
        response = "< think >This has spaces< /think >"
        result = self.reward("prompt", response)
        self.assertEqual(result, 0.0)  # Should not match due to spaces in tags

    def test_call_with_target_parameter(self):
        """Test __call__ with target parameter (should be ignored)."""
        response = "<think>This is my reasoning</think>"
        result = self.reward("prompt", response, target="some target")
        self.assertEqual(result, 0.5)

        result = self.reward("prompt", "no tags", target="some target")
        self.assertEqual(result, 0.0)

    def test_call_zero_reward_value(self):
        """Test __call__ with zero reward value."""
        zero_reward = ThinkingReward(reward_value=0.0)
        response = "<think>This is my reasoning</think>"
        result = zero_reward("prompt", response)
        self.assertEqual(result, 0.0)

    def test_call_negative_reward_value(self):
        """Test __call__ with negative reward value."""
        negative_reward = ThinkingReward(reward_value=-0.5)
        response = "<think>This is my reasoning</think>"
        result = negative_reward("prompt", response)
        self.assertEqual(result, -0.5)


if __name__ == "__main__":
    unittest.main()
