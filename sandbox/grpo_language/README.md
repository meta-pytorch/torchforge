# GRPO with Language Reward

This sandbox app demonstrates using GRPO training with a language reward that encourages the model to think in a specific target language.

## Overview

This app extends the standard GRPO training (from `apps/grpo/`) by adding a `LanguageReward` that evaluates whether the model's thinking (text within `<思考></思考>` tags) is in the target language.

**Key Insight**: Uses Japanese tags `<思考>` (shikō = "thinking") instead of English `<think>` tags to break the model's association between thinking tags and English language. This helps encourage multilingual thinking.

## Key Features

- **Multi-objective training**: Combines three rewards:
  - `MathReward`: Evaluates correctness of math answers
  - `ThinkingReward`: Encourages use of `<思考>` tags
  - `LanguageReward`: Rewards thinking in target language (Japanese by default)

- **Japanese thinking tags**: Uses `<思考>` instead of `<think>` to encourage non-English reasoning

- **Language detection**: Uses `langid` to detect the language of thinking blocks

- **Configurable target language**: While this app defaults to Japanese (`ja`), the `LanguageReward` can be configured for any ISO 639-1 language code

- **Configurable tags**: Both rewards support custom tag names via the `tag` parameter

## Requirements

Before running this app, install the required language detection library:

```bash
pip install langid
```

## Usage

```bash
python -m sandbox.grpo_language.main --config apps/grpo/qwen3_1_7b.yaml
```

You can use any of the config files from `apps/grpo/` (e.g., `qwen3_1_7b.yaml`, `qwen3_8b.yaml`, `qwen3_32b.yaml`).

## How It Works

1. The model receives a math problem and is instructed to use `<思考>` tags for reasoning
2. During training, the model generates responses with thinking blocks
3. Three rewards are computed:
   - Math correctness (did it get the right answer?)
   - Thinking usage (did it use `<思考>` tags properly?)
   - Language usage (did it think in Japanese?)
4. The model is trained to maximize all three rewards

## Configuration

### Target Language

The target language is configured as Japanese in `main.py`:

```python
LanguageReward(target_language="ja", tag="思考")
ThinkingReward(tag="思考")
```

To use a different language:
1. Change `target_language` to the appropriate ISO 639-1 code:
   - English: `"en"`
   - Chinese: `"zh"`
- Spanish: `"es"`
- French: `"fr"`
- etc.

## Expected Behavior

Over the course of training, the model should learn to:
1. Solve math problems correctly
2. Use `<思考></思考>` tags for its reasoning
3. Write its thinking in Japanese (or the configured target language)

## Metrics

The following metrics are logged to W&B:
- `reward/evaluate_response/avg_LanguageReward_reward`: Average language reward
- `reward/evaluate_response/avg_MathReward_reward`: Average math reward
- `reward/evaluate_response/avg_ThinkingReward_reward`: Average thinking reward
- `reward/evaluate_response/avg_total_reward`: Average of all rewards

## Differences from Standard GRPO

This is a modified version of `apps/grpo/main.py` with:
1. Added import: `from forge.data.rewards import LanguageReward`
2. Modified reward functions list to include `LanguageReward(target_language="ja")`
3. Updated config to use different W&B group name

All other training logic remains the same.
