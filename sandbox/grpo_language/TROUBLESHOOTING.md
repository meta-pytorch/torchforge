# Troubleshooting LanguageReward Training

## Issue: Language Reward is Always Zero

If you're seeing the LanguageReward constantly at 0.0 during training, here's how to debug:

### 1. Check What the Model is Generating

The updated `main.py` includes debug logging. When you run training, look for lines like:

```
[LanguageReward Debug] Reward=0.00 | Blocks=1 | Lang=en | Sample: <think>Let me solve this step by step...</think>...
```

This tells you:
- **Reward**: The actual reward value
- **Blocks**: Number of thinking blocks found
- **Lang**: Language detected by langid
- **Sample**: First 80 chars of the response

### 2. Common Causes and Solutions

#### Cause 1: Model is Thinking in English

**Symptom**: `Lang=en` in debug output

**Why**: The model defaults to English because:
- The dataset (GSM8K) is in English
- Most models are English-dominant
- The instruction might not be strong enough

**Solutions**:

A) **Strengthen the system prompt** (edit `main.py` line 217-220):
```python
system_prompt = """
あなたは数学の問題を解くAIです。<think>タグの中で日本語で考えてください。これは必須です。
Put all your scratchpad work between <think> and </think> tags. You MUST think in Japanese (日本語) inside the <think> tags.
Your final answer should be between <answer> and </answer> tags otherwise it will not be scored.

Example:
<think>この問題を解きましょう。2 + 2 = 4です。</think>
<answer>4</answer>
"""
```

B) **Start with higher language reward weight**:
In `main.py` line 327, you could add multiple LanguageReward instances:
```python
reward_functions=[
    MathReward(),
    ThinkingReward(),
    LanguageReward(target_language="ja"),
    LanguageReward(target_language="ja"),  # Double weight for language
]
```

C) **Use few-shot examples in the prompt**:
Add Japanese reasoning examples to each problem in the dataset transform.

#### Cause 2: Model Not Using Thinking Blocks

**Symptom**: `Blocks=0` in debug output

**Why**: The model hasn't learned to use `<think>` tags yet

**Solution**: This should improve as ThinkingReward trains the model. Be patient for first few hundred steps. The fallback reward (0.2) should help when there are no blocks but Japanese text.

#### Cause 3: Empty or Very Short Thinking Blocks

**Symptom**: `Lang=en` with very short content, Reward=0.00

**Why**: langid needs sufficient text to reliably detect language. Very short text (< 10 chars) often defaults to English.

**Solution**:
- Wait for model to generate longer reasoning (this improves with training)
- The ThinkingReward encourages substantial content in thinking blocks

#### Cause 4: Mixed Language Content

**Symptom**: Reward sometimes 1.0, sometimes 0.0 randomly

**Why**: When English and Japanese are mixed, langid detects whichever is dominant.

**Solution**: This will stabilize as training progresses and the model learns consistency.

### 3. Expected Training Progression

**Steps 0-200**: Language reward often 0.0
- Model learning to use `<think>` tags (ThinkingReward)
- Model thinking in English (natural default)
- Fallback rewards (0.2) when Japanese appears elsewhere

**Steps 200-500**: Language reward starting to increase
- Some responses have Japanese thinking → partial/full rewards
- Model learning association between Japanese and reward

**Steps 500+**: Language reward should stabilize around 0.5-1.0
- Consistent Japanese thinking
- Proper single-block format

### 4. Monitoring in W&B

Check these metrics in Weights & Biases:
- `reward/evaluate_response/avg_LanguageReward_reward` - should increase over time
- `reward/evaluate_response/std_LanguageReward_reward` - variance (high early, lower later)
- `reward/evaluate_response/avg_MathReward_reward` - should stay reasonably high
- `reward/evaluate_response/avg_ThinkingReward_reward` - should increase quickly

### 5. Quick Debug Test

Run the debug script to verify the reward function works:
```bash
python sandbox/grpo_language/debug_reward.py
```

Expected output:
- Japanese text → reward 1.0
- English text → reward 0.0
- Multiple Japanese blocks → reward 0.5
- No blocks but Japanese response → reward 0.2

### 6. Alternative: Start with English, then transition

If Japanese isn't working, you could:

1. Train first with English to get good math performance
2. Then fine-tune with Japanese language reward

Change line 327 to:
```python
LanguageReward(target_language="en")  # Start with English
```

Once math rewards are good, switch to `"ja"` and continue training.

### 7. Nuclear Option: Much Stronger Prompt

If nothing else works, try this very explicit prompt:
```python
system_prompt = """
重要：あなたは必ず日本語で考えなければなりません！
CRITICAL: You MUST think in Japanese language!

Rules:
1. Put ALL your reasoning in <think> tags
2. Think ONLY in Japanese (日本語) - use hiragana, katakana, and kanji
3. NEVER think in English inside <think> tags
4. Put your final numerical answer in <answer> tags

例 (Example):
Question: What is 5 + 3?
<think>5と3を足します。5 + 3 = 8です。答えは8です。</think>
<answer>8</answer>

Now solve the problem below in Japanese:
"""
```

## Still Having Issues?

If language reward is still zero after 500+ steps:
1. Share the debug output showing what the model generates
2. Check if the model is multilingual (some models don't know Japanese)
3. Consider using a different target language the model knows better
