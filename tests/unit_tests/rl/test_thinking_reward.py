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
        self.custom_reward = ThinkingReward(partial_reward=0.3, full_reward=0.9)

    def test_init_default_values(self):
        """Test ThinkingReward initialization with default values."""
        reward = ThinkingReward()
        self.assertEqual(reward.partial_reward, 0.2)
        self.assertEqual(reward.full_reward, 1.0)

    def test_init_custom_values(self):
        """Test ThinkingReward initialization with custom values."""
        reward = ThinkingReward(partial_reward=0.3, full_reward=0.9)
        self.assertEqual(reward.partial_reward, 0.3)
        self.assertEqual(reward.full_reward, 0.9)

    def test_regex_patterns(self):
        """Test that regex patterns are compiled correctly."""
        reward = ThinkingReward()
        self.assertIsNotNone(reward._THINK_BLOCK_RE)
        self.assertIsNotNone(reward._THINK_TAG_ATTEMPT_RE)

    def test_call_with_well_formed_thinking_block(self):
        """Test __call__ with well-formed thinking blocks."""
        result = self.reward("prompt", "<think>This is my reasoning</think>")
        self.assertEqual(result, 1.0)

        result = self.custom_reward("prompt", "<think>This is my reasoning</think>")
        self.assertEqual(result, 0.9)

    def test_call_with_well_formed_thinking_block_complex_content(self):
        """Test __call__ with complex content in thinking blocks."""
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
        self.assertEqual(result, 1.0)

    def test_call_with_minimal_content_thinking_block(self):
        """Test __call__ with minimal content that still counts as well-formed."""
        result = self.reward("prompt", "<think>x</think>")
        self.assertEqual(result, 1.0)

    def test_call_with_empty_thinking_block(self):
        """Test __call__ with empty thinking block."""
        result = self.reward("prompt", "<think></think>")
        self.assertEqual(result, 0.2)  # Should give partial reward, not full

    def test_call_with_whitespace_only_thinking_block(self):
        """Test __call__ with whitespace-only thinking block."""
        result = self.reward("prompt", "<think>   \n  \t  </think>")
        self.assertEqual(result, 0.2)  # Should give partial reward, not full

    def test_call_with_only_opening_tag(self):
        """Test __call__ with response containing only opening tag."""
        result = self.reward("prompt", "<think>This is incomplete reasoning")
        self.assertEqual(result, 0.2)  # Should give partial reward for attempt

    def test_call_with_only_closing_tag(self):
        """Test __call__ with response containing only closing tag."""
        result = self.reward("prompt", "This is incomplete reasoning</think>")
        self.assertEqual(result, 0.2)  # Should give partial reward for attempt

    def test_call_with_no_tags(self):
        """Test __call__ with response containing no thinking tags."""
        result = self.reward(
            "prompt", "This is just a regular response without any thinking tags."
        )
        self.assertEqual(result, 0.0)

    def test_call_case_insensitive(self):
        """Test __call__ is case insensitive for thinking tags."""
        result = self.reward("prompt", "<THINK>This is my reasoning</THINK>")
        self.assertEqual(result, 1.0)

        result = self.reward("prompt", "<Think>This is my reasoning</Think>")
        self.assertEqual(result, 1.0)

        result = self.reward("prompt", "<think>This is my reasoning</THINK>")
        self.assertEqual(result, 1.0)

    def test_call_with_whitespace_in_tags(self):
        """Test __call__ with whitespace in thinking tags."""
        result = self.reward("prompt", "< think >This is my reasoning</ think >")
        self.assertEqual(result, 1.0)

        result = self.reward("prompt", "<\tthink\n>Content</\tthink\n>")
        self.assertEqual(result, 1.0)

    def test_call_multiple_thinking_blocks(self):
        """Test __call__ with multiple thinking blocks."""
        response = """
        <think>First thought</think>
        Some text in between.
        <think>Second thought</think>
        """
        result = self.reward("prompt", response)
        self.assertEqual(result, 1.0)

    def test_call_nested_tags(self):
        """Test __call__ with nested or malformed tags."""
        result = self.reward(
            "prompt", "<think>Outer <think>inner</think> thought</think>"
        )
        self.assertEqual(result, 1.0)

    def test_call_multiline_thinking_block(self):
        """Test __call__ with multiline thinking blocks."""
        response = """<think>
        This is a multiline
        thinking block with
        lots of content
        </think>"""
        result = self.reward("prompt", response)
        self.assertEqual(result, 1.0)

    def test_call_empty_response(self):
        """Test __call__ with empty response."""
        result = self.reward("prompt", "")
        self.assertEqual(result, 0.0)

    def test_call_none_response(self):
        """Test __call__ with None response."""
        result = self.reward("prompt", None)
        self.assertEqual(result, 0.0)

    def test_call_with_target_parameter(self):
        """Test __call__ with target parameter (should be ignored)."""
        result = self.reward(
            "prompt", "<think>This is my reasoning</think>", target="some target"
        )
        self.assertEqual(result, 1.0)

        result = self.reward("prompt", "no tags", target="some target")
        self.assertEqual(result, 0.0)

        result = self.reward(
            "prompt", "<think>This is my reasoning</think>", target=None
        )
        self.assertEqual(result, 1.0)

    def test_call_custom_reward_values(self):
        """Test __call__ with custom reward values."""
        response_full = "<think>This is proper reasoning</think>"
        response_partial = "<think>"
        response_none = "no thinking tags"

        # Test custom partial reward
        self.assertEqual(self.custom_reward("prompt", response_full), 0.9)
        self.assertEqual(self.custom_reward("prompt", response_partial), 0.3)
        self.assertEqual(self.custom_reward("prompt", response_none), 0.0)

    def test_call_zero_custom_values(self):
        """Test __call__ with zero custom values."""
        zero_reward = ThinkingReward(partial_reward=0.0, full_reward=0.0)
        result = zero_reward("prompt", "<think>This is my reasoning</think>")
        self.assertEqual(result, 0.0)

    def test_call_negative_reward_values(self):
        """Test __call__ with negative reward values."""
        negative_reward = ThinkingReward(partial_reward=-0.1, full_reward=-0.5)

        self.assertEqual(
            negative_reward("prompt", "<think>This is proper reasoning</think>"), -0.5
        )
        self.assertEqual(negative_reward("prompt", "<think>"), -0.1)

    def test_call_edge_case_characters(self):
        """Test __call__ with edge case characters in thinking blocks."""
        result = self.reward(
            "prompt", "<think>Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?`~</think>"
        )
        self.assertEqual(result, 1.0)

    def test_call_unicode_characters(self):
        """Test __call__ with unicode characters in thinking blocks."""
        result = self.reward("prompt", "<think>Unicode: αβγδε 中文 🚀</think>")
        self.assertEqual(result, 1.0)

    def test_call_very_long_thinking_block(self):
        """Test __call__ with very long thinking blocks."""
        long_content = "A" * 10000
        result = self.reward("prompt", f"<think>{long_content}</think>")
        self.assertEqual(result, 1.0)

    def test_custom_tag(self):
        """Test that ThinkingReward uses the custom tag passed in."""
        # Create reward with custom Japanese tag
        custom_tag_reward = ThinkingReward(tag="思考")

        # Response with custom tag should get full reward
        result = custom_tag_reward("prompt", "<思考>This is my reasoning</思考>")
        self.assertEqual(result, 1.0)

        # Response with default "think" tag should get no reward
        result = custom_tag_reward("prompt", "<think>This is my reasoning</think>")
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
