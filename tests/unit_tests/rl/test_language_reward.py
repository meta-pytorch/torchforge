# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from unittest.mock import patch


class TestLanguageReward(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import after patching to avoid ImportError
        from forge.data.rewards import LanguageReward

        self.LanguageReward = LanguageReward
        self.reward_en = LanguageReward(target_language="en")
        self.reward_ja = LanguageReward(target_language="ja")

    def test_init_default_values(self):
        """Test LanguageReward initialization with default values."""
        reward = self.LanguageReward()
        self.assertEqual(reward.target_language, "ja")
        self.assertEqual(reward.match_reward, 1.0)
        self.assertEqual(reward.no_match_reward, 0.0)

    def test_init_custom_values(self):
        """Test LanguageReward initialization with custom values."""
        reward = self.LanguageReward(
            target_language="ja",
            match_reward=0.9,
            no_match_reward=0.1,
        )
        self.assertEqual(reward.target_language, "ja")
        self.assertEqual(reward.match_reward, 0.9)
        self.assertEqual(reward.no_match_reward, 0.1)

    def test_init_missing_langid(self):
        """Test LanguageReward initialization without langid installed."""
        # Remove langid from modules if it exists
        langid_module = sys.modules.get("langid")
        if "langid" in sys.modules:
            del sys.modules["langid"]

        with patch.dict("sys.modules", {"langid": None}):
            with self.assertRaises(ImportError) as context:
                # Re-import to trigger the ImportError
                import importlib

                import forge.data.rewards

                importlib.reload(forge.data.rewards)
                forge.data.rewards.LanguageReward()

            self.assertIn("langid is required", str(context.exception))
            self.assertIn("pip install langid", str(context.exception))

        # Restore langid module if it existed
        if langid_module is not None:
            sys.modules["langid"] = langid_module

    def test_regex_pattern(self):
        """Test that regex pattern is compiled correctly."""
        reward = self.LanguageReward()
        self.assertIsNotNone(reward._THINK_BLOCK_RE)

    def test_call_with_english_thinking(self):
        """Test __call__ with English text in thinking blocks."""
        response = "<思考>This is English reasoning about math problems.</思考>"
        result = self.reward_en("prompt", response)
        self.assertEqual(result, 1.0)

    def test_call_with_japanese_thinking(self):
        """Test __call__ with Japanese text in thinking blocks."""
        response = "<思考>これは日本語で考えています。数学の問題を解きます。</思考>"
        result = self.reward_ja("prompt", response)
        self.assertEqual(result, 1.0)

        # English reward should give no_match_reward for Japanese text
        result = self.reward_en("prompt", response)
        self.assertEqual(result, 0.0)

    def test_call_with_chinese_thinking(self):
        """Test __call__ with Chinese text in thinking blocks."""
        response = "<思考>这是中文思考。我们需要解决这个数学问题。</思考>"
        reward_zh = self.LanguageReward(target_language="zh")
        result = reward_zh("prompt", response)
        # langid should detect this as Chinese (zh)
        self.assertEqual(result, 1.0)

    def test_call_with_spanish_thinking(self):
        """Test __call__ with Spanish text in thinking blocks."""
        response = "<思考>Este es un razonamiento en español sobre problemas matemáticos.</思考>"
        reward_es = self.LanguageReward(target_language="es")
        result = reward_es("prompt", response)
        # langid should detect this as Spanish (es)
        self.assertEqual(result, 1.0)

    def test_call_language_mismatch(self):
        """Test __call__ when detected language doesn't match target."""
        # Japanese reward with English text
        response = "<思考>This is English reasoning.</思考>"
        result = self.reward_ja("prompt", response)
        self.assertEqual(result, 0.0)

        # English reward with Japanese text
        response = "<思考>これは日本語です。</思考>"
        result = self.reward_en("prompt", response)
        self.assertEqual(result, 0.0)

    def test_call_with_no_thinking_tags(self):
        """Test __call__ with response containing no thinking tags but correct language."""
        result = self.reward_en(
            "prompt", "This is just a regular response without any thinking tags."
        )
        # No thinking blocks -> detect whole response, English detected -> match_reward
        self.assertEqual(result, 1.0)

    def test_call_with_no_thinking_tags_wrong_language(self):
        """Test __call__ with response containing no thinking tags and wrong language."""
        result = self.reward_en("prompt", "これは日本語の応答です。タグはありません。")
        # No thinking blocks -> detect whole response, Japanese detected -> no_match_reward
        self.assertEqual(result, 0.0)

    def test_call_with_empty_thinking_block(self):
        """Test __call__ with empty thinking block."""
        result = self.reward_en("prompt", "<思考></思考>")
        self.assertEqual(result, 0.0)

    def test_call_with_whitespace_only_thinking_block(self):
        """Test __call__ with whitespace-only thinking block."""
        result = self.reward_en("prompt", "<思考>   \n  \t  </思考>")
        self.assertEqual(result, 0.0)

    def test_call_with_proper_tags(self):
        """Test __call__ with properly formatted thinking tags."""
        response = "<思考>This is English reasoning.</思考>"
        result = self.reward_en("prompt", response)
        self.assertEqual(result, 1.0)

        # Japanese content should also work
        response = "<思考>これは日本語です。</思考>"
        result = self.reward_ja("prompt", response)
        self.assertEqual(result, 1.0)

    def test_call_multiple_thinking_blocks(self):
        """Test __call__ with multiple thinking blocks - detects whole response language."""
        response = """
        <思考>First thought in English.</思考>
        Some text in between.
        <思考>Second thought also in English.</思考>
        """
        result = self.reward_en("prompt", response)
        # Multiple blocks -> detect whole response, English detected -> match_reward
        self.assertEqual(result, 1.0)

    def test_call_multiline_thinking_block(self):
        """Test __call__ with multiline thinking blocks."""
        response = """<思考>
        This is a multiline
        thinking block with
        lots of English content
        about solving problems
        </思考>"""
        result = self.reward_en("prompt", response)
        self.assertEqual(result, 1.0)

    def test_call_empty_response(self):
        """Test __call__ with empty response."""
        result = self.reward_en("prompt", "")
        self.assertEqual(result, 0.0)

    def test_call_none_response(self):
        """Test __call__ with None response."""
        result = self.reward_en("prompt", None)
        self.assertEqual(result, 0.0)

    def test_call_custom_reward_values(self):
        """Test __call__ with custom reward values."""
        response_ja_single = "<思考>これは日本語です。</思考>"
        response_ja_multiple = "<思考>最初の考え。</思考><思考>次の考え。</思考>"
        response_ja_no_tags = "これはタグなしの日本語です。"
        response_en = "<思考>This is English.</思考>"
        response_none = ""

        custom_reward = self.LanguageReward(
            target_language="ja",
            match_reward=0.9,
            no_match_reward=0.1,
        )
        # Test custom match reward (single block, correct language)
        self.assertEqual(custom_reward("prompt", response_ja_single), 0.9)
        # Test custom match reward (multiple blocks -> whole response, correct language)
        self.assertEqual(custom_reward("prompt", response_ja_multiple), 0.9)
        # Test custom match reward (no blocks -> whole response, correct language)
        self.assertEqual(custom_reward("prompt", response_ja_no_tags), 0.9)
        # Test custom no_match reward (wrong language)
        self.assertEqual(custom_reward("prompt", response_en), 0.1)
        # Test empty response
        self.assertEqual(custom_reward("prompt", response_none), 0.1)

    def test_call_with_special_characters(self):
        """Test __call__ with special characters in thinking blocks."""
        response = (
            "<思考>English with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?`~</思考>"
        )
        result = self.reward_en("prompt", response)
        self.assertEqual(result, 1.0)

    def test_call_with_mixed_content_outside_tags(self):
        """Test __call__ with mixed language content outside thinking tags."""
        # Content outside think tags should be ignored
        response = """
        これは日本語のテキストです。
        <思考>But this is English reasoning inside the tags.</思考>
        もっと日本語のテキスト。
        """
        result = self.reward_en("prompt", response)
        # Should detect English from thinking block only
        self.assertEqual(result, 1.0)

    def test_call_with_numbers_and_symbols(self):
        """Test __call__ with thinking blocks containing mostly numbers."""
        response = "<思考>Calculate: 2 + 2 = 4, then 4 * 3 = 12</思考>"
        result = self.reward_en("prompt", response)
        # Should still detect as English due to words like "Calculate" and "then"
        self.assertEqual(result, 1.0)

    def test_call_with_code_in_thinking(self):
        """Test __call__ with code snippets in thinking blocks."""
        response = """<思考>
        Let me write some Python code to solve this:
        def calculate(x):
            return x * 2
        The function doubles the input value.
        </思考>"""
        result = self.reward_en("prompt", response)
        # Should detect as English due to surrounding text
        self.assertEqual(result, 1.0)

    def test_different_language_codes(self):
        """Test __call__ with various ISO 639-1 language codes."""
        # Test a few common languages
        languages = {
            "fr": "Ceci est un texte en français avec beaucoup de contenu.",
            "de": "Dies ist ein deutscher Text mit viel Inhalt.",
            "it": "Questo è un testo italiano con molto contenuto.",
            "pt": "Este é um texto em português com muito conteúdo.",
        }

        for lang_code, text in languages.items():
            reward = self.LanguageReward(target_language=lang_code)
            response = f"<思考>{text}</思考>"
            result = reward("prompt", response)
            # langid should detect these correctly
            self.assertEqual(
                result,
                1.0,
                f"Failed to detect {lang_code} language: '{text[:50]}...'",
            )


if __name__ == "__main__":
    unittest.main()
