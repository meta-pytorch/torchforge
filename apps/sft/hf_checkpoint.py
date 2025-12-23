# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HuggingFace-native checkpoint saving that bypasses PyTorch DCP.

This module provides checkpoint saving functionality that works with FSDP
by gathering the full model state and using HuggingFace's native save_pretrained.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

logger = logging.getLogger(__name__)


def save_hf_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_dir: str,
    step: int,
    rank: int = 0,
    save_optimizer: bool = True,
) -> None:
    """
    Save checkpoint in HuggingFace format, bypassing PyTorch DCP.
    
    This function gathers the full model state from FSDP shards and saves
    it using HuggingFace's native format, avoiding the KeyError bug in DCP.
    
    Args:
        model: The model to save (can be FSDP-wrapped)
        tokenizer: The tokenizer to save
        optimizer: Optional optimizer to save
        lr_scheduler: Optional LR scheduler to save
        checkpoint_dir: Base directory for checkpoints
        step: Current training step
        rank: Process rank (only rank 0 saves)
        save_optimizer: Whether to save optimizer state
    """
    # Create checkpoint directory
    ckpt_path = Path(checkpoint_dir) / f"step_{step}"
    
    if rank == 0:
        ckpt_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving HuggingFace checkpoint to {ckpt_path}")
    
    # Wait for directory creation
    if dist.is_initialized():
        dist.barrier()
    
    # Handle FSDP model
    if isinstance(model, FSDP):
        # Configure FSDP to gather full state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=save_policy,
        ):
            model_state = model.state_dict()
    else:
        model_state = model.state_dict()
    
    # Only rank 0 saves to avoid race conditions
    if rank == 0:
        # Get the unwrapped model for HF saving
        if isinstance(model, FSDP):
            # Access the original model inside FSDP wrapper
            unwrapped_model = model._fsdp_wrapped_module
            # If it's still wrapped (nested FSDP), keep unwrapping
            while hasattr(unwrapped_model, '_fsdp_wrapped_module'):
                unwrapped_model = unwrapped_model._fsdp_wrapped_module
        else:
            unwrapped_model = model
        
        # Load the full state dict into unwrapped model
        unwrapped_model.load_state_dict(model_state, strict=True)
        
        # Save using HuggingFace's native method
        try:
            unwrapped_model.save_pretrained(
                ckpt_path,
                safe_serialization=True,  # Use safetensors format
            )
            logger.info(f"✅ Model saved to {ckpt_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
        
        # Save tokenizer
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(ckpt_path)
                logger.info(f"✅ Tokenizer saved to {ckpt_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save tokenizer: {e}")
        
        # Save optimizer and scheduler (as PyTorch native - no HF format for these)
        if save_optimizer and optimizer is not None:
            try:
                opt_path = ckpt_path / "optimizer.pt"
                optimizer_state = {
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                }
                if lr_scheduler is not None:
                    optimizer_state['lr_scheduler'] = lr_scheduler.state_dict()
                
                torch.save(optimizer_state, opt_path)
                logger.info(f"✅ Optimizer state saved to {opt_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save optimizer: {e}")
        
        # Save training metadata
        try:
            meta_path = ckpt_path / "training_metadata.pt"
            metadata = {
                'step': step,
                'model_config': unwrapped_model.config.to_dict() if hasattr(unwrapped_model, 'config') else {},
            }
            torch.save(metadata, meta_path)
            logger.info(f"✅ Metadata saved to {meta_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save metadata: {e}")
        
        logger.info(f"🎉 Checkpoint saved successfully at step {step}")
    
    # Synchronize all ranks
    if dist.is_initialized():
        dist.barrier()


def load_hf_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_dir: str,
    step: Optional[int] = None,
    rank: int = 0,
) -> int:
    """
    Load checkpoint from HuggingFace format.
    
    Args:
        model: The model to load into
        tokenizer: The tokenizer (not modified, HF loads from config)
        optimizer: Optional optimizer to load state into
        lr_scheduler: Optional LR scheduler to load state into
        checkpoint_dir: Base directory for checkpoints
        step: Specific step to load (None = latest)
        rank: Process rank
        
    Returns:
        The step number that was loaded
    """
    ckpt_base = Path(checkpoint_dir)
    
    # Find checkpoint to load
    if step is None:
        # Find latest checkpoint
        checkpoints = sorted([d for d in ckpt_base.glob("step_*") if d.is_dir()])
        if not checkpoints:
            logger.warning(f"No checkpoints found in {checkpoint_dir}")
            return 0
        ckpt_path = checkpoints[-1]
        step = int(ckpt_path.name.split("_")[1])
    else:
        ckpt_path = ckpt_base / f"step_{step}"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    if rank == 0:
        logger.info(f"Loading HuggingFace checkpoint from {ckpt_path}")
    
    # Get unwrapped model
    if isinstance(model, FSDP):
        unwrapped_model = model._fsdp_wrapped_module
        while hasattr(unwrapped_model, '_fsdp_wrapped_module'):
            unwrapped_model = unwrapped_model._fsdp_wrapped_module
    else:
        unwrapped_model = model
    
    # Load model using HuggingFace's from_pretrained
    try:
        from transformers import AutoModelForCausalLM
        loaded_model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=unwrapped_model.dtype if hasattr(unwrapped_model, 'dtype') else torch.float32,
        )
        unwrapped_model.load_state_dict(loaded_model.state_dict(), strict=True)
        if rank == 0:
            logger.info(f"✅ Model loaded from {ckpt_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Load optimizer and scheduler
    if optimizer is not None:
        opt_path = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            try:
                optimizer_state = torch.load(opt_path, map_location='cpu')
                optimizer.load_state_dict(optimizer_state['optimizer'])
                if lr_scheduler is not None and 'lr_scheduler' in optimizer_state:
                    lr_scheduler.load_state_dict(optimizer_state['lr_scheduler'])
                if rank == 0:
                    logger.info(f"✅ Optimizer state loaded from {opt_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load optimizer: {e}")
    
    if rank == 0:
        logger.info(f"🎉 Checkpoint loaded successfully from step {step}")
    
    return step
