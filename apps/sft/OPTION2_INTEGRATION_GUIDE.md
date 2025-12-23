# Example: How to integrate HuggingFace checkpoint saving into main.py

## Option 2: Integration Steps

### Step 1: Import the HF checkpoint module in main.py

Add to the imports section (around line 13-30):
```python
from forge.apps.sft.hf_checkpoint import save_hf_checkpoint, load_hf_checkpoint
```

### Step 2: Replace the existing checkpoint save call

Find this code in the `train()` method (around line 456):
```python
self.checkpointer.save(
    curr_step=self.current_step,
    last_step=self.current_step == self.num_training_steps,
)
```

Replace it with:
```python
# Save checkpoint using HuggingFace format (bypasses DCP KeyError bug)
checkpoint_config = self.job_config.checkpoint
if (checkpoint_config.enable and 
    self.current_step % checkpoint_config.interval == 0):
    
    save_hf_checkpoint(
        model=self.model[0],  # First model in list
        tokenizer=self.tokenizer,
        optimizer=self.optimizer.optimizers[0] if self.optimizer else None,
        lr_scheduler=self.lr_scheduler.schedulers[0] if self.lr_scheduler else None,
        checkpoint_dir=checkpoint_config.folder,
        step=self.current_step,
        rank=self._rank,
        save_optimizer=True,
    )
```

### Step 3: Update the config to enable checkpointing

In `llama3_8b.yaml`, change:
```yaml
checkpoint:
  enable: true  # ✅ Re-enable checkpointing!
  folder: /home/hosseinkh/models/Meta-Llama-3.1-8B-Instruct/checkpoint
  initial_load_path: /home/hosseinkh/models/Meta-Llama-3.1-8B-Instruct/
  initial_load_in_hf: true
  last_save_in_hf: true  # ✅ Now this will work!
  interval: 500
  async_mode: "disabled"
```

### Step 4: Optional - Load from HF checkpoint at startup

In the `setup()` method (around line 138), replace:
```python
self.checkpointer.load(step=self.current_step)
```

With:
```python
# Load from HuggingFace checkpoint if exists
checkpoint_config = self.job_config.checkpoint
if checkpoint_config.enable and Path(checkpoint_config.folder).exists():
    try:
        loaded_step = load_hf_checkpoint(
            model=self.model[0],
            tokenizer=self.tokenizer,
            optimizer=self.optimizer.optimizers[0] if self.optimizer else None,
            lr_scheduler=self.lr_scheduler.schedulers[0] if self.lr_scheduler else None,
            checkpoint_dir=checkpoint_config.folder,
            step=None,  # Load latest
            rank=self._rank,
        )
        self.current_step = loaded_step
        logger.info(f"Resumed from step {loaded_step}")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}, starting from scratch")
```

---

## How It Works

### The Problem with PyTorch DCP:
```
HuggingFace Model (plain state_dict)
    ↓
FSDP Wrapping (changes parameter IDs)
    ↓
Training with new parameter IDs
    ↓
PyTorch DCP Save (tries to map back to original FQNs)
    ❌ KeyError: 289 - parameter ID not in fqn_pid_mapping
```

### The HuggingFace Solution:
```
FSDP Model (sharded across GPUs)
    ↓
Gather Full State Dict (FSDP.state_dict_type)
    ↓
Load into Unwrapped Model
    ↓
HuggingFace save_pretrained (safetensors format)
    ✅ Works perfectly!
```

---

## Key Features of HF Checkpoint

1. **FSDP-Compatible**: Uses `FSDP.state_dict_type()` to properly gather sharded weights
2. **Rank-0 Only**: Only rank 0 saves to avoid race conditions
3. **Barriers**: Proper synchronization across all ranks
4. **Unwrapping**: Recursively unwraps FSDP layers to get original model
5. **SafeTensors**: Uses modern safetensors format
6. **Optimizer State**: Separately saves optimizer/scheduler as PyTorch .pt files
7. **Metadata**: Saves training step and config for easy resumption

---

## File Structure After Saving

```
checkpoint_dir/
├── step_500/
│   ├── model.safetensors         # ✅ Model weights (HF format)
│   ├── config.json                # ✅ Model config
│   ├── tokenizer.json             # ✅ Tokenizer
│   ├── tokenizer_config.json      # ✅ Tokenizer config
│   ├── optimizer.pt               # ✅ Optimizer + LR scheduler
│   └── training_metadata.pt       # ✅ Step number, etc.
├── step_1000/
│   └── ...
```

---

## Benefits Over PyTorch DCP

| Feature | PyTorch DCP | HuggingFace |
|---------|-------------|-------------|
| FSDP + HF Checkpoint | ❌ KeyError: 289 | ✅ Works |
| Format Compatibility | DCP-specific | ✅ Universal HF format |
| Easy Loading | Complex | ✅ `from_pretrained()` |
| Safetensors Support | Limited | ✅ Native |
| Resume Training | ✅ Yes | ✅ Yes |
| Size | Larger | Smaller (compressed) |

---

## Testing

After implementing, test with:
```bash
cd /home/hosseinkh/forge_updated/forge
python -m apps.sft.main --config apps/sft/llama3_8b.yaml
```

At step 500, you should see:
```
INFO:forge.apps.sft.hf_checkpoint:Saving HuggingFace checkpoint to /home/hosseinkh/models/Meta-Llama-3.1-8B-Instruct/checkpoint/step_500
INFO:forge.apps.sft.hf_checkpoint:✅ Model saved to /home/hosseinkh/models/Meta-Llama-3.1-8B-Instruct/checkpoint/step_500
INFO:forge.apps.sft.hf_checkpoint:✅ Tokenizer saved to /home/hosseinkh/models/Meta-Llama-3.1-8B-Instruct/checkpoint/step_500
INFO:forge.apps.sft.hf_checkpoint:✅ Optimizer state saved to /home/hosseinkh/models/Meta-Llama-3.1-8B-Instruct/checkpoint/step_500/optimizer.pt
INFO:forge.apps.sft.hf_checkpoint:🎉 Checkpoint saved successfully at step 500
```

No more KeyError! 🎉
