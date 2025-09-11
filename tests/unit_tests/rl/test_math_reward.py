# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from forge.data.rewards import MathReward


class TestMathReward(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reward = MathReward()
        self.custom_reward = MathReward(tolerance=1e-3, partial_credit=0.2)

    def test_init_default_values(self):
        """Test MathReward initialization with default values."""
        reward = MathReward()
        self.assertEqual(reward.tolerance, 1e-6)
        self.assertEqual(reward.partial_credit, 0.1)

    def test_init_custom_values(self):
        """Test MathReward initialization with custom values."""
        reward = MathReward(tolerance=1e-3, partial_credit=0.2)
        self.assertEqual(reward.tolerance, 1e-3)
        self.assertEqual(reward.partial_credit, 0.2)

    def test_to_float_valid_numbers(self):
        """Test _to_float with valid numeric strings."""
        self.assertEqual(self.reward._to_float("42"), 42.0)
        self.assertEqual(self.reward._to_float("3.14"), 3.14)
        self.assertEqual(self.reward._to_float("-5.5"), -5.5)
        self.assertEqual(self.reward._to_float("0"), 0.0)
        self.assertEqual(self.reward._to_float("  123.45  "), 123.45)

    def test_to_float_with_currency_and_formatting(self):
        """Test _to_float with currency symbols and commas."""
        self.assertEqual(self.reward._to_float("$42"), 42.0)
        self.assertEqual(self.reward._to_float("$1,000"), 1000.0)
        self.assertEqual(self.reward._to_float("1,234.56"), 1234.56)
        self.assertEqual(self.reward._to_float("$ 42.50 "), 42.5)

    def test_to_float_invalid_inputs(self):
        """Test _to_float with invalid inputs."""
        self.assertIsNone(self.reward._to_float("abc"))
        self.assertIsNone(self.reward._to_float(""))
        self.assertIsNone(self.reward._to_float("12.34.56"))
        self.assertIsNone(self.reward._to_float("not a number"))
        self.assertIsNone(self.reward._to_float(None))

    def test_to_float_edge_cases(self):
        """Test _to_float with edge cases."""
        self.assertEqual(self.reward._to_float("1e6"), 1000000.0)
        self.assertEqual(self.reward._to_float("-1.5e-3"), -0.0015)

    def test_call_correct_answer_in_tags(self):
        """Test __call__ with correct answers in <answer></answer> tags."""
        self.assertEqual(self.reward("prompt", "<answer>42</answer>", "42"), 1.0)
        self.assertEqual(self.reward("prompt", "<answer>3.14</answer>", "3.14"), 1.0)
        self.assertEqual(self.reward("prompt", "<answer>-5.5</answer>", "-5.5"), 1.0)

    def test_call_answer_tags_with_whitespace(self):
        """Test __call__ with answer tags containing whitespace."""
        self.assertEqual(self.reward("prompt", "<answer> 42 </answer>", "42"), 1.0)
        self.assertEqual(
            self.reward("prompt", "<answer>\n3.14\n</answer>", "3.14"), 1.0
        )

    def test_call_answer_tags_with_complex_content(self):
        """Test __call__ with complex content in answer tags."""
        response = """
        Let me solve this step by step:
        First, I calculate 2 + 3 = 5
        Then, I multiply by 4: 5 * 4 = 20
        Finally, I subtract 8: 20 - 8 = 12
        <answer>12</answer>
        """
        self.assertEqual(self.reward("prompt", response, "12"), 1.0)

    def test_call_within_tolerance(self):
        """Test __call__ with answers within tolerance."""
        # Default tolerance is 1e-6
        self.assertEqual(
            self.reward("prompt", "<answer>42.0000001</answer>", "42"), 1.0
        )
        self.assertEqual(
            self.reward("prompt", "<answer>3.1400001</answer>", "3.14"), 1.0
        )

        # Custom tolerance
        self.assertEqual(
            self.custom_reward("prompt", "<answer>42.0001</answer>", "42"), 1.0
        )
        self.assertEqual(
            self.custom_reward("prompt", "<answer>3.141</answer>", "3.14"), 1.0
        )

    def test_call_outside_tolerance(self):
        """Test __call__ with answers outside tolerance."""
        self.assertEqual(self.reward("prompt", "<answer>42.1</answer>", "42"), 0.0)
        self.assertEqual(self.reward("prompt", "<answer>3.15</answer>", "3.14"), 0.0)
        self.assertEqual(
            self.custom_reward("prompt", "<answer>42.01</answer>", "42"), 0.0
        )

    def test_call_partial_credit_target_in_response(self):
        """Test __call__ with partial credit when target appears in response."""
        response = "The calculation shows 42 but I put <answer>43</answer>"
        self.assertEqual(self.reward("prompt", response, "42"), 0.1)

        response = "Let me work through this: 42 + 1 = 43. <answer>43</answer>"
        self.assertEqual(self.reward("prompt", response, "42"), 0.1)

    def test_call_partial_credit_custom_value(self):
        """Test __call__ with custom partial credit value."""
        response = "The calculation shows 42 but I put <answer>43</answer>"
        self.assertEqual(self.custom_reward("prompt", response, "42"), 0.2)

    def test_call_no_partial_credit_with_answer_tags(self):
        """Test __call__ doesn't give partial credit if target is only in answer tags."""
        response = "Let me solve this. <answer>42</answer>"
        # Target 100 is not elsewhere in response, so no partial credit
        self.assertEqual(self.reward("prompt", response, "100"), 0.0)

    def test_call_integer_target_formatting(self):
        """Test __call__ with integer targets formatted correctly."""
        # Integer targets should be formatted without decimal point
        response = "I calculated and got 117 as the answer. <answer>118</answer>"
        self.assertEqual(self.reward("prompt", response, "117"), 0.1)

        # Should work with 117.0 in target too
        self.assertEqual(self.reward("prompt", response, "117.0"), 0.1)

    def test_call_float_target_formatting(self):
        """Test __call__ with float targets."""
        response = "I calculated and got 3.14 as the answer. <answer>3.15</answer>"
        self.assertEqual(self.reward("prompt", response, "3.14"), 0.1)

    def test_call_invalid_target(self):
        """Test __call__ with invalid target values."""
        self.assertEqual(self.reward("prompt", "<answer>42</answer>", "invalid"), 0.0)
        self.assertEqual(self.reward("prompt", "<answer>42</answer>", ""), 0.0)
        self.assertEqual(
            self.reward("prompt", "<answer>42</answer>", "not a number"), 0.0
        )

    def test_call_no_answer_tags(self):
        """Test __call__ with response that has no answer tags."""
        # Should still check for partial credit
        self.assertEqual(self.reward("prompt", "The answer is 42", "42"), 0.1)
        self.assertEqual(self.reward("prompt", "No matching number", "42"), 0.0)

    def test_call_invalid_answer_in_tags(self):
        """Test __call__ with invalid answer in tags."""
        response = "<answer>not a number</answer> but 42 is correct"
        self.assertEqual(self.reward("prompt", response, "42"), 0.1)

    def test_call_zero_values(self):
        """Test __call__ with zero values."""
        self.assertEqual(self.reward("prompt", "<answer>0</answer>", "0"), 1.0)
        self.assertEqual(self.reward("prompt", "<answer>0.0</answer>", "0"), 1.0)

    def test_call_negative_values(self):
        """Test __call__ with negative values."""
        self.assertEqual(self.reward("prompt", "<answer>-42</answer>", "-42"), 1.0)
        self.assertEqual(self.reward("prompt", "<answer>-3.14</answer>", "-3.14"), 1.0)

    def test_call_large_numbers(self):
        """Test __call__ with large numbers."""
        self.assertEqual(
            self.reward("prompt", "<answer>1000000</answer>", "1000000"), 1.0
        )
        self.assertEqual(self.reward("prompt", "<answer>1e6</answer>", "1000000"), 1.0)

    def test_call_small_numbers(self):
        """Test __call__ with very small numbers."""
        self.assertEqual(
            self.reward("prompt", "<answer>0.000001</answer>", "0.000001"), 1.0
        )
        self.assertEqual(
            self.reward("prompt", "<answer>1e-6</answer>", "0.000001"), 1.0
        )

    def test_call_multiple_answer_tags(self):
        """Test __call__ with multiple answer tags (should use first one)."""
        response = "First answer: <answer>42</answer> Second: <answer>43</answer>"
        self.assertEqual(self.reward("prompt", response, "42"), 1.0)
        self.assertEqual(self.reward("prompt", response, "43"), 0.1)


if __name__ == "__main__":
    unittest.main()
