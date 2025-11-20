# KL Clipping Implementation Summary

## Changes Made to `apps/blackjack/main_v2.py`

### 1. KL Divergence Clipping (Line 1327-1333)

**Before:**
```python
kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
```

**After:**
```python
# Following VERL's approach: clip log difference before exp for numerical stability
logprob_diff_clipped = torch.clamp(logprob_diff, min=-20.0, max=20.0)
kl = torch.exp(logprob_diff_clipped) - logprob_diff_clipped - 1
# Clip final KL to prevent extreme values
kl = torch.clamp(kl, min=-10.0, max=10.0)
```

**Why This Works:**
- **First clamp [-20, 20]**: Prevents numerical overflow/underflow in `exp()`
  - exp(-20) ≈ 2e-9 (very small but not zero)
  - exp(20) ≈ 485M (large but not inf)
- **Second clamp [-10, 10]**: Bounds the final KL divergence
  - Prevents extreme KL values from dominating the loss
  - Your previous KL was **61 million** → now capped at 10.0

**Based on:** VERL's `kl_penalty_forward()` with "low_var_kl" estimator

---

## Additional Recommendations

### 2. Add Gradient Clipping to Config

Your config doesn't have gradient clipping. Add this to `apps/blackjack/*.yaml`:

```yaml
trainer:
  optimizer:
    name: AdamW
    lr: 1e-5
    eps: 1e-8
  gradient_clipping:
    max_norm: 1.0  # Clip gradients to max norm of 1.0
  lr_scheduler:
    warmup_steps: 1
```

**Why:** Prevents large gradient updates that can cause policy divergence (especially at step 2).

**Typical values:**
- `max_norm: 0.5` - Conservative (used by many RL papers)
- `max_norm: 1.0` - Standard (good starting point)
- `max_norm: 5.0` - Lenient

---

### 3. Consider Increasing Batch Size

Your current config:
- `group_size: 4` (4 games per rollout)
- `local_batch_size: 8` (8 sequences per batch)

With such small batches, a single bad episode can cause large gradient updates.

**Recommendations:**
- Increase `group_size` to 8 or 16
- This provides more stable advantage estimates
- Reduces variance in gradient updates

---

### 4. Monitor These Metrics

After the fix, watch these metrics in your training logs:

```
loss_debug/logprob_diff_mean   # Should be close to 0
loss_debug/logprob_diff_max    # Should be < 20 (clipped)
loss_debug/kl_mean             # Should be < 1.0 typically
loss_debug/kl_max              # Should be = 10.0 (clipped) initially
```

If `kl_max` stays at 10.0 for many steps, it means clipping is active. You may need to:
- Reduce learning rate
- Increase beta (KL coefficient)
- Add stronger gradient clipping

---

## What Was Causing the Explosion?

Looking at your dump:
- **Position 221**: Token `\n\n` (271) predicting next token `<H` (73585)
- **Policy logprob**: -19.44 (policy is very uncertain)
- **Ref logprob**: -1.50 (ref model is confident)
- **Logprob diff**: -1.50 - (-19.44) = **17.94**
- **Unclipped KL**: exp(17.94) - 17.94 - 1 ≈ **61 million**
- **Clipped KL**: exp(17.94 clipped to 10) - 10 - 1 = exp(10) - 11 ≈ **22,015**

Still large, but not catastrophic!

---

## Testing the Fix

Run your training and check if:
1. ✅ KL no longer explodes to millions
2. ✅ Training is stable past step 2
3. ✅ Policy doesn't diverge too far from ref model

You can verify by running:
```bash
python debug/analyze_explosion_point.py
```

This will show you what the policy is predicting at the explosion points and whether clipping is working.

---

## Alternative: Token-Level Ratio Clipping (TRL/Prime-RL Approach)

If KL clipping doesn't fully solve it, consider adding importance ratio masking:

```python
# After computing per_token_loss
importance_ratio = torch.exp(logprobs - ref_logprobs)
is_masked = (importance_ratio < 0.125) | (importance_ratio > 8.0)
per_token_loss = per_token_loss * (~is_masked).float()
```

This masks tokens where the policy has diverged too far (outside [1/8, 8] ratio).
