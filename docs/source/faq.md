# FAQ

## Common Installation Issues

### Issue: `gh: command not found`

**Solution**: Install GitHub CLI:
```bash
# On Ubuntu/Debian
sudo apt install gh

# On Fedora
sudo dnf install gh

# Then authenticate
gh auth login
```

### Issue: `CUDA out of memory`

**Solution**: Reduce batch size in your config file:
```yaml
training:
  batch_size: 2  # Reduced from 4
  gradient_accumulation_steps: 8  # Increased to maintain effective batch size
```

### Issue: `ImportError: No module named 'torch'`

**Solution**: Ensure you activated the conda environment:
```bash
conda activate forge
```

### Issue: vLLM wheel download fails

**Solution**: The vLLM wheel is hosted on GitHub releases. Ensure you're authenticated with `gh auth login` and have internet access.

### Issue: `Unsupported GPU architecture`

**Solution**: Check your GPU compute capability:
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```

TorchForge requires compute capability 7.0 or higher (Volta architecture or newer).

### Issue: Monarch actor spawn failures

**Symptom**: Errors like "Failed to spawn actors" or "Process allocation failed"

**Solution**: Verify your GPU count matches your configuration:
```bash
nvidia-smi  # Check available GPUs
```

Ensure your config requests fewer processes than available GPUs.
