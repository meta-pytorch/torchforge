#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Debug script to test LanguageReward behavior."""

from forge.data.rewards import LanguageReward

# Create reward for Japanese
reward = LanguageReward(target_language="ja")

# Test cases mimicking what the model might generate
test_cases = [
    # Case 1: Perfect - Japanese in single thinking block
    ("<思考>これは数学の問題です。2+2=4です。</思考><answer>4</answer>", "Perfect Japanese"),
    # Case 2: English thinking (most likely during training)
    (
        "<思考>This is a math problem. 2+2=4.</思考><answer>4</answer>",
        "English thinking",
    ),
    # Case 3: No thinking blocks at all
    ("The answer is 4.<answer>4</answer>", "No thinking blocks"),
    # Case 4: Empty thinking blocks
    ("<思考></思考><answer>4</answer>", "Empty thinking block"),
    # Case 5: Multiple thinking blocks in Japanese
    (
        "<思考>最初の考え。</思考><思考>次の考え。</思考><answer>4</answer>",
        "Multiple Japanese blocks",
    ),
    # Case 6: Just the answer, no thinking
    ("<answer>4</answer>", "Just answer tag"),
    # Case 7: Thinking with mostly numbers/symbols
    ("<思考>2 + 2 = 4</思考><answer>4</answer>", "Mostly numbers"),
    # Case 8: Mixed English and Japanese
    ("<思考>Let me think... これは簡単です。</思考><answer>4</answer>", "Mixed languages"),
]

print("=" * 80)
print("LanguageReward Debug Output (target_language='ja')")
print("=" * 80)

for response, description in test_cases:
    score = reward(prompt="", response=response, target=None)

    import re

    # Try to detect what langid thinks
    import langid

    # Extract thinking content if exists
    think_match = re.findall(r"<\s*思考\s*>(.*?)<\s*/\s*思考\s*>", response, re.DOTALL)

    if think_match:
        content = " ".join(think_match)
        detected_lang, confidence = langid.classify(content)
        print(f"\n{description}:")
        print(f"  Response: {response[:60]}...")
        print(f"  Reward: {score}")
        print(f"  Thinking blocks found: {len(think_match)}")
        print(f"  Detected language: {detected_lang} (confidence: {confidence:.3f})")
    else:
        # Check fallback
        response_text = re.sub(r"<\s*/?\s*思考\s*>", "", response).strip()
        if response_text:
            detected_lang, confidence = langid.classify(response_text)
            print(f"\n{description}:")
            print(f"  Response: {response[:60]}...")
            print(f"  Reward: {score}")
            print("  Thinking blocks found: 0")
            print(
                f"  Fallback detection on response text: {detected_lang} (confidence: {confidence:.3f})"
            )
        else:
            print(f"\n{description}:")
            print(f"  Response: {response[:60]}...")
            print(f"  Reward: {score}")
            print("  No content to analyze")

print("\n" + "=" * 80)
print("Expected rewards:")
print("  full_reward (1.0): Single Japanese thinking block")
print("  partial_reward (0.5): Multiple Japanese thinking blocks")
print("  fallback_reward (0.2): No blocks but Japanese response text")
print("  no_match_reward (0.0): Wrong language")
print("=" * 80)
