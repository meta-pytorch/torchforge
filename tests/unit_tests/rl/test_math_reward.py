# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

from forge.data.rewards.math import MathReward


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
        self.assertEqual(self.reward._to_float("inf"), float("inf"))
        self.assertEqual(self.reward._to_float("-inf"), float("-inf"))

    def test_extract_number_gsm8k_format(self):
        """Test _extract_number with GSM8K style format."""
        self.assertEqual(self.reward._extract_number("#### 42"), 42.0)
        self.assertEqual(self.reward._extract_number("#### -3.14"), -3.14)
        self.assertEqual(self.reward._extract_number("Some text #### 123.45"), 123.45)

    def test_extract_number_answer_patterns(self):
        """Test _extract_number with various answer patterns."""
        self.assertEqual(self.reward._extract_number("The answer is 42"), 42.0)
        self.assertEqual(self.reward._extract_number("answer is 3.14"), 3.14)
        self.assertEqual(self.reward._extract_number("Answer: 123"), 123.0)
        self.assertEqual(self.reward._extract_number("Result: -5.5"), -5.5)

    def test_extract_number_equals_pattern(self):
        """Test _extract_number with equals sign patterns."""
        self.assertEqual(self.reward._extract_number("x = 42."), 42.0)
        self.assertEqual(self.reward._extract_number("The result = 3.14"), 3.14)
        self.assertEqual(self.reward._extract_number("calculation = -7.5."), -7.5)

    def test_extract_number_end_of_text(self):
        """Test _extract_number with numbers at end of text."""
        self.assertEqual(self.reward._extract_number("The final result is 42."), 42.0)
        self.assertEqual(self.reward._extract_number("We get 3.14"), 3.14)
        self.assertEqual(self.reward._extract_number("Answer: -5.5."), -5.5)

    def test_extract_number_fallback_pattern(self):
        """Test _extract_number with fallback pattern (any number)."""
        self.assertEqual(self.reward._extract_number("There are 42 items"), 42.0)
        self.assertEqual(self.reward._extract_number("Cost is $3.14 per item"), 3.14)
        self.assertEqual(self.reward._extract_number("Temperature: -5.5 degrees"), -5.5)

    def test_extract_number_multiple_matches(self):
        """Test _extract_number returns the last match when multiple numbers exist."""
        # Should return the last match from the pattern
        self.assertEqual(
            self.reward._extract_number("First 10, then 20, finally 30"), 30.0
        )
        self.assertEqual(
            self.reward._extract_number("#### 5 but actually #### 10"), 10.0
        )

    def test_extract_number_no_match(self):
        """Test _extract_number when no numbers are found."""
        self.assertIsNone(self.reward._extract_number("No numbers here"))
        self.assertIsNone(self.reward._extract_number(""))
        self.assertIsNone(self.reward._extract_number("Just text"))

    def test_extract_number_case_insensitive(self):
        """Test _extract_number is case insensitive."""
        self.assertEqual(self.reward._extract_number("THE ANSWER IS 42"), 42.0)
        self.assertEqual(self.reward._extract_number("Answer: 3.14"), 3.14)
        self.assertEqual(self.reward._extract_number("RESULT: 123"), 123.0)

    def test_call_correct_answer(self):
        """Test __call__ with correct answers."""
        self.assertEqual(self.reward("prompt", "The answer is 42", "42"), 1.0)
        self.assertEqual(self.reward("prompt", "#### 3.14", "3.14"), 1.0)
        self.assertEqual(self.reward("prompt", "Result: -5.5", "-5.5"), 1.0)

    def test_call_within_tolerance(self):
        """Test __call__ with answers within tolerance."""
        # Default tolerance is 1e-6
        self.assertEqual(self.reward("prompt", "42.0000001", "42"), 1.0)
        self.assertEqual(self.reward("prompt", "3.1400001", "3.14"), 1.0)

        # Custom tolerance
        self.assertEqual(self.custom_reward("prompt", "42.0001", "42"), 1.0)
        self.assertEqual(self.custom_reward("prompt", "3.141", "3.14"), 1.0)

    def test_call_outside_tolerance(self):
        """Test __call__ with answers outside tolerance."""
        self.assertEqual(self.reward("prompt", "42.1", "42"), 0.0)
        self.assertEqual(self.reward("prompt", "3.15", "3.14"), 0.0)
        self.assertEqual(self.custom_reward("prompt", "42.01", "42"), 0.0)

    def test_call_invalid_target(self):
        """Test __call__ with invalid target values."""
        self.assertEqual(
            self.reward("prompt", "42", "invalid"), self.reward.partial_credit
        )
        self.assertEqual(self.reward("prompt", "42", ""), self.reward.partial_credit)
        self.assertEqual(
            self.reward("prompt", "42", "not a number"), self.reward.partial_credit
        )

    def test_call_invalid_response(self):
        """Test __call__ with invalid response values."""
        self.assertEqual(
            self.reward("prompt", "no number", "42"), self.reward.partial_credit
        )
        self.assertEqual(self.reward("prompt", "", "42"), self.reward.partial_credit)
        self.assertEqual(
            self.reward("prompt", "just text", "42"), self.reward.partial_credit
        )

    def test_call_both_invalid(self):
        """Test __call__ with both invalid target and response."""
        self.assertEqual(
            self.reward("prompt", "no number", "invalid"), self.reward.partial_credit
        )
        self.assertEqual(self.reward("prompt", "", ""), self.reward.partial_credit)

    def test_call_custom_partial_credit(self):
        """Test __call__ uses custom partial credit value."""
        self.assertEqual(self.custom_reward("prompt", "no number", "42"), 0.2)
        self.assertEqual(self.custom_reward("prompt", "42", "invalid"), 0.2)

    def test_call_zero_values(self):
        """Test __call__ with zero values."""
        self.assertEqual(self.reward("prompt", "0", "0"), 1.0)
        self.assertEqual(self.reward("prompt", "The answer is 0", "0.0"), 1.0)

    def test_call_negative_values(self):
        """Test __call__ with negative values."""
        self.assertEqual(self.reward("prompt", "-42", "-42"), 1.0)
        self.assertEqual(self.reward("prompt", "#### -3.14", "-3.14"), 1.0)
        self.assertEqual(self.reward("prompt", "-5", "-4.9"), 0.0)

    def test_call_large_numbers(self):
        """Test __call__ with large numbers."""
        self.assertEqual(self.reward("prompt", "1000000", "1000000"), 1.0)
        self.assertEqual(self.reward("prompt", "1e6", "1000000"), 1.0)
        self.assertEqual(self.reward("prompt", "1000001", "1000000"), 0.0)

    def test_call_small_numbers(self):
        """Test __call__ with very small numbers."""
        self.assertEqual(self.reward("prompt", "0.000001", "0.000001"), 1.0)
        self.assertEqual(self.reward("prompt", "1e-6", "0.000001"), 1.0)

    def test_call_complex_response_text(self):
        """Test __call__ with complex response text containing multiple elements."""
        response = """
        Let me solve this step by step:
        First, I calculate 2 + 3 = 5
        Then, I multiply by 4: 5 * 4 = 20
        Finally, I subtract 8: 20 - 8 = 12
        #### 12
        """
        self.assertEqual(self.reward("prompt", response, "12"), 1.0)

    def test_call_with_units_and_formatting(self):
        """Test __call__ with responses containing units and formatting."""
        self.assertEqual(self.reward("prompt", "The cost is $42.50", "42.5"), 1.0)
        self.assertEqual(self.reward("prompt", "Distance: 3.14 meters", "3.14"), 1.0)
        self.assertEqual(self.reward("prompt", "Temperature is -5.5°C", "-5.5"), 1.0)


if __name__ == "__main__":
    unittest.main()
