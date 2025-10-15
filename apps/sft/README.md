# SFT (Supervised Fine-Tuning) Training

This directory contains code for supervised fine-tuning of large language models using PyTorch and the Monarch actor framework.

## Quick Start

### Option 1: Command Line (Production)

```bash
# Run training with a config file
python -m forge.apps.sft.main --config forge/apps/sft/llama3_8b.yaml
```

### Option 2: Interactive Notebook (Experimentation)

Use the Jupyter notebook for interactive configuration and testing:

```bash
# Open the notebook in VS Code or JupyterLab
code forge/apps/sft/interactive_config_notebook.ipynb

# Or start JupyterLab
cd /home/hosseinkh/TorchForge/forge/apps/sft
jupyter lab
```

---

## Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `main.py` | Main training script | Production training runs |
| `trainer_actor.py` | Training actor implementation | Core training logic |
| `spawn_actor.py` | Actor spawning utilities | Distributed setup |
| `utils.py` | Utility functions | Helper functions for data/training |
| `interactive_config_notebook.ipynb` | Interactive configuration | Quick experiments & learning |
| `llama3_8b.yaml` | Default config for Llama 3 8B | Production configuration |
| `qwen3_8b.yaml` | Config for Qwen 3 8B | Alternative model |

---

## Interactive Notebook Guide

The `interactive_config_notebook.ipynb` allows you to:

1. **Interactively build configurations** - Modify settings in real-time
2. **Visualize configurations** - See YAML output before saving
3. **Test configurations** - Quickly iterate on experiments
4. **Save to YAML** - Export your configuration for production runs

### Notebook Structure

```
Step 1: Setup Environment
  └─ Import libraries, set working directory

Step 2: Configure Model and Process Settings
  └─ Model type, number of GPUs/processes

Step 3: Configure Optimizer and LR Scheduler
  └─ Learning rate, warmup steps

Step 4: Configure Training Settings
  └─ Batch size, sequence length, training steps

Step 5: Configure Checkpoint and Activation Checkpointing
  └─ Where to save, how often to save

Step 6: Configure Communication and Parallelism Settings
  └─ FSDP, tensor parallelism, etc.

Step 7: Combine All Configurations
  └─ Merge into complete config

Step 8: Save to YAML File
  └─ Export for use with main.py

Step 9: Optional - Run Training
  └─ Launch training from notebook
```

### How to Use the Notebook

1. **Open the notebook:**
   ```bash
   code forge/apps/sft/interactive_config_notebook.ipynb
   ```

2. **Run cells sequentially** (Shift+Enter) from top to bottom

3. **Modify parameters** in each step according to your needs:
   - **Model path**: Point to your model weights
   - **Training steps**: How long to train
   - **Batch size**: Adjust for your GPU memory
   - **Learning rate**: Tune for your task
   - **Checkpoint folder**: Where to save checkpoints

4. **View the configuration** in Step 7 to see the complete YAML

5. **Save to YAML** in Step 8 to export your config

6. **Run training**:
   - Option A: Use the notebook (Step 9)
   - Option B: Use command line with your saved config

---

## Configuration Files

### YAML Structure

```yaml
comm:
  trace_buf_size: 0

model:
  name: llama3
  flavor: 8B
  hf_assets_path: /path/to/model

processes:
  procs: 8           # Number of processes (usually = number of GPUs)
  with_gpus: true

optimizer:
  name: AdamW
  lr: 1e-5
  eps: 1e-8

lr_scheduler:
  warmup_steps: 200

training:
  local_batch_size: 1
  seq_len: 2048
  max_norm: 1.0
  steps: 1000
  compile: false
  dataset: c4

parallelism:
  data_parallel_replicate_degree: 1
  data_parallel_shard_degree: -1    # -1 = use all GPUs for FSDP
  tensor_parallel_degree: 1
  pipeline_parallel_degree: 1
  context_parallel_degree: 1
  expert_parallel_degree: 1
  disable_loss_parallel: false

checkpoint:
  enable: true
  folder: /path/to/checkpoints
  initial_load_path: /path/to/model
  initial_load_in_hf: true
  last_save_in_hf: true
  interval: 500
  async_mode: disabled

activation_checkpoint:
  mode: selective
  selective_ac_option: op
```

### Key Parameters to Adjust

#### For Quick Testing:
```yaml
training:
  steps: 10                  # Just 10 steps
  dataset: c4_test           # Tiny test dataset

processes:
  procs: 1                   # Single GPU
```

#### For Production Training:
```yaml
training:
  steps: 10000               # Full training run
  dataset: c4                # Full C4 dataset

processes:
  procs: 8                   # All 8 GPUs
```

#### For Memory-Constrained GPUs:
```yaml
training:
  local_batch_size: 1        # Smallest batch size
  seq_len: 1024              # Shorter sequences

activation_checkpoint:
  mode: selective            # Enable activation checkpointing
```

---

## Running Training

### Single Node (8 GPUs)

```bash
python -m forge.apps.sft.main --config forge/apps/sft/llama3_8b.yaml
```

### Multi-Node (32 GPUs across 4 nodes)

1. **Update config:**
   ```yaml
   processes:
     procs: 32        # 4 nodes × 8 GPUs
     with_gpus: true
   ```

2. **Submit to cluster:**
   ```bash
   sbatch --nodes=4 --gpus-per-node=8 \
     python -m forge.apps.sft.main --config forge/apps/sft/llama3_8b.yaml
   ```

---

## Datasets

### Available Datasets

| Dataset | Size | Use Case | Config Value |
|---------|------|----------|--------------|
| `c4` | ~750 GB | Production training | `dataset: c4` |
| `c4_test` | ~Few MB | Quick testing | `dataset: c4_test` |
| `c4_validation` | ~30 GB | Validation | `dataset: c4_validation` |

### Using Custom Datasets

Modify `utils.py` or `trainer_actor.py` to add your custom dataset loading logic.

---

## Monitoring Training

### Logs

Training logs show:
- Loss per step
- Learning rate
- Checkpoint saves
- GPU memory usage

Example output:
```
[Trainer-0/8] 2025-10-14 13:20:00 INFO 1 / 1000|Loss: 3.245
[Trainer-0/8] 2025-10-14 13:20:05 INFO 2 / 1000|Loss: 3.189
[Trainer-0/8] 2025-10-14 13:20:10 INFO 3 / 1000|Loss: 3.134
```

### Checkpoints

Checkpoints are saved to `checkpoint.folder` every `checkpoint.interval` steps:
```
/path/to/checkpoints/
├── step_500/
│   ├── model.pt
│   └── optimizer.pt
├── step_1000/
│   ├── model.pt
│   └── optimizer.pt
```

---

## Architecture

### Actor-Based Distributed Training

```
Orchestrator (main.py)
    │
    ├─ spawn_actor.run_actor()
    │      │
    │      └─ TrainerActor × 8 (one per GPU)
    │            │
    │            ├─ setup()       → Load data, model, optimizer
    │            ├─ train()       → Training loop
    │            └─ cleanup()     → Save final checkpoint
```

### Key Components

#### 1. TrainerActor (`trainer_actor.py`)
- Main training class
- Inherits from `ForgeActor` and `ForgeEngine`
- Handles:
  - Data loading
  - Model training
  - Checkpointing
  - Distributed coordination

#### 2. SpawnActor (`spawn_actor.py`)
- Utility for spawning actors
- Handles:
  - Actor configuration
  - Distributed setup
  - Actor lifecycle management

#### 3. Utils (`utils.py`)
- Helper functions for:
  - Data loading
  - Tokenization
  - Metrics logging

---

## Parallelism Strategies

### FSDP (Fully Sharded Data Parallelism)
- **Default**: `data_parallel_shard_degree: -1`
- Shards model parameters across all GPUs
- Reduces memory per GPU
- Recommended for large models

### Tensor Parallelism
- **Config**: `tensor_parallel_degree: N`
- Splits individual layers across N GPUs
- Useful for very large models that don't fit on single GPU

### Pipeline Parallelism
- **Config**: `pipeline_parallel_degree: N`
- Splits model into N stages
- Each stage on different GPU
- Good for very deep models

### Example Configurations

#### Small Model (7B), 8 GPUs:
```yaml
parallelism:
  data_parallel_shard_degree: 8   # FSDP across all 8 GPUs
  tensor_parallel_degree: 1
  pipeline_parallel_degree: 1
```

#### Large Model (70B), 8 GPUs:
```yaml
parallelism:
  data_parallel_shard_degree: 4   # FSDP across 4 GPUs
  tensor_parallel_degree: 2       # TP across 2 GPUs
  pipeline_parallel_degree: 1
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size:
   ```yaml
   training:
     local_batch_size: 1
   ```

2. Reduce sequence length:
   ```yaml
   training:
     seq_len: 1024  # or even 512
   ```

3. Enable activation checkpointing:
   ```yaml
   activation_checkpoint:
     mode: selective
   ```

4. Increase FSDP sharding:
   ```yaml
   parallelism:
     data_parallel_shard_degree: -1  # Use all GPUs
   ```

### Issue: Training is Slow

**Check:**
1. Are you using all GPUs? (`processes.procs`)
2. Is dataset loading slow? (try `c4_test` for quick test)
3. Is compile enabled? (can be slow on first run)

### Issue: Loss is NaN

**Solutions:**
1. Reduce learning rate:
   ```yaml
   optimizer:
     lr: 1e-6  # Smaller LR
   ```

2. Increase warmup:
   ```yaml
   lr_scheduler:
     warmup_steps: 500
   ```

3. Enable gradient clipping:
   ```yaml
   training:
     max_norm: 1.0  # Already enabled by default
   ```

### Issue: Can't Import `forge.apps.sft`

**Solution:**
Make sure you're running from the repo root:
```bash
cd /home/hosseinkh/TorchForge/forge
python -m forge.apps.sft.main --config forge/apps/sft/llama3_8b.yaml
```

---

## Comparison with GRPO

| Feature | SFT | GRPO |
|---------|-----|------|
| **Training Type** | Supervised learning | Reinforcement learning |
| **Number of Actors** | 1 type (Trainer) | 7 types (Policy, Trainer, Buffer, etc.) |
| **Complexity** | Simple | Complex |
| **Use Case** | Pre-training, instruction tuning | RL fine-tuning, RLHF |
| **Data** | (input, output) pairs | Prompts + rewards |
| **Architecture** | Single actor | Multi-actor pipeline |

---

## Best Practices

### 1. Start Small
- Use `c4_test` dataset first
- Run for 10 steps to verify setup
- Then scale up

### 2. Monitor Memory
- Check GPU memory usage
- Adjust batch size and sequence length accordingly
- Use activation checkpointing if needed

### 3. Save Regularly
- Set reasonable `checkpoint.interval` (e.g., 500)
- Save initial checkpoint to test loading

### 4. Use Version Control
- Commit your config files
- Track changes to training parameters
- Document experiments

### 5. Test Configurations
- Use the interactive notebook first
- Validate YAML before production runs
- Run quick tests (10 steps) before long runs

---

## Additional Resources

### Documentation Files

See the `apps/sft_v2/` directory for additional documentation:
- `EVALUATION_GUIDE.md` - How to add evaluation
- `C4_DATASET_EXPLAINED.md` - Dataset details
- `ENDPOINT_DECORATOR_EXPLAINED.md` - Understanding `@endpoint`
- `MULTI_NODE_EXPLAINED.md` - Multi-node training guide
- `GRPO_VS_SFT_COMPARISON.md` - SFT vs GRPO architecture

### Related Code

- `forge/src/forge/controller/` - ForgeActor base class
- `forge/src/forge/data/` - Data loading utilities
- `forge/apps/grpo/` - GRPO (RL) training for comparison

---

## Quick Reference

### Start Training
```bash
python -m forge.apps.sft.main --config forge/apps/sft/llama3_8b.yaml
```

### Test Configuration
```bash
# Use notebook
code forge/apps/sft/interactive_config_notebook.ipynb

# Or quick test with CLI
python -m forge.apps.sft.main --config forge/apps/sft/llama3_8b.yaml \
  --training.steps=10 \
  --training.dataset=c4_test
```

### Resume from Checkpoint
```yaml
# In config:
checkpoint:
  enable: true
  initial_load_path: /path/to/checkpoint/step_500/
```

### Multi-Node Training
```yaml
# In config:
processes:
  procs: 32  # 4 nodes × 8 GPUs
```

---

## Summary

**SFT provides simple, scalable supervised fine-tuning** with:
- ✅ Interactive notebook for configuration
- ✅ Single-command training launch
- ✅ Multi-node support out of the box
- ✅ FSDP/TP/PP parallelism options
- ✅ Automatic checkpointing
- ✅ Production-ready architecture

**Get started in 3 steps:**
1. Open `interactive_config_notebook.ipynb`
2. Configure your settings
3. Run `python -m forge.apps.sft.main --config your_config.yaml`

Happy training! 🚀
