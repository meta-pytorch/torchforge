# On-Policy Distillation for Math Reasoning

This app implements on-policy distillation (OPD) following the approach described in the [Thinking Machines blog post](https://thinkingmachines.ai/blog/on-policy-distillation/). OPD combines the benefits of on-policy training with dense reward signals for efficient post-training.

## Overview

On-policy distillation trains a student model by:
1. Sampling trajectories from the student model itself
2. Using a teacher model to grade each token with dense rewards (per-token KL divergence)
3. Training the student to minimize reverse KL with the teacher

This approach is **10-30x more compute efficient** than traditional RL while achieving comparable or better performance.

## Experimental Setup

### Models
- **Student**: Qwen3-1.7B-Base (or Qwen3-8B for larger experiments)
- **Teacher**: Qwen3-8B (or Qwen3-32B)
- **Evaluation**: AIME'24 benchmark

### Training Pipeline

#### Phase 1: Supervised Fine-Tuning (SFT)
First, establish a strong baseline through off-policy distillation:

```bash
python -m apps.sft.main --config apps/sft/qwen3_1_7b.yaml
```

- **Dataset**: OpenThoughts3-1.2M (400k prompts)
- **Expected Performance**: ~40% on AIME'24
- **Purpose**: Teaches the model basic math reasoning patterns

#### Phase 2: On-Policy Distillation
Refine the model using on-policy learning with dense supervision:

```bash
python -m apps.on-policy-distillation.main --config apps/on-policy-distillation/qwen_opd.yaml
```

- **Starting Point**: SFT checkpoint from Phase 1
- **Dataset**: Math prompts (from OpenThoughts3 or DeepMath, but only prompts - not solutions)
- **Training**: ~150-200 steps (77k prompts with 4 samples each)
- **Expected Performance**: ~50% on AIME'24

### Key Implementation Details

1. **Loss Function**: Per-token reverse KL divergence
   ```python
   reverse_kl = -(student_logprobs - teacher_logprobs)
   ```

2. **Sampling**: Generate multiple trajectories per prompt (n=16 in config)

3. **No Discount Factor**: Optimize only immediate next token (discount=0)

4. **Efficient Batching**: Can use smaller batch sizes than RL due to dense rewards

## Key Advantages

- **Compute Efficiency**: 10-30x reduction vs traditional RL
- **Dense Supervision**: Learns from every token, not just final rewards
- **Data Efficiency**: Can reuse prompts multiple times effectively
- **Stability**: More stable training than sparse RL rewards

## Notes for Reproduction

1. **Ensure proper initialization**: Load the SFT checkpoint before starting OPD
2. **Use prompts only**: During OPD, sample completions from student, don't use dataset solutions
3. **Teacher quality matters**: Better teachers provide better supervision
4. **Monitor reverse KL**: Should go to near-zero as training progresses

## References

- [On-Policy Distillation Blog Post](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [OpenThoughts3 Dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
