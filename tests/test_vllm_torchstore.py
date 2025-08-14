#!/usr/bin/env python3
"""
Test script to:
1. Initialize Llama 3 8B model from HuggingFace transformers
2. Write its state dict to torchstore
3. Initialize Policy with torchstore
4. Call update to load model weights into Policy
5. Verify the model works correctly
"""

import asyncio
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import push_state_dict

from forge.actors.policy import Policy


async def test_llama3_torchstore_write():
    """
    First phase: Load Llama 3.1 8B and write state dict to torchstore
    """
    print("=== PHASE 1: Writing Llama 3.1 8B to TorchStore ===")
    print("Initializing MultiProcessStore...")
    store = MultiProcessStore()
    
    print("Loading Llama 3.1 8B model from local path...")
    # Load from local directory instead of HuggingFace download
    model_path = "/tmp/Meta-Llama-3.1-8B"
    
    try:
        # Load the model from local path - using device_map="auto" for efficient loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # Ensure we don't try to download
        )
        
        # Also load tokenizer for completeness
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True  # Ensure we don't try to download
        )
        
        print(f"Model loaded successfully. Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Get the model's state dict
        print("Getting model state dict...")
        state_dict = model.state_dict()
        print(f"State dict contains {len(state_dict)} parameters")
        
        # Print some info about the state dict
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameters: {total_params:,}")
        
        # Sample of parameter names
        param_names = list(state_dict.keys())[:10]
        print(f"Sample parameter names: {param_names}")
        
        # Write state dict to torchstore
        print("Writing state dict to torchstore...")
        key = "llama3_8b_state_dict"
        await push_state_dict(store, state_dict, key)
        print(f"Successfully wrote state dict to torchstore with key: {key}")
        
        # Test a simple forward pass to verify original model works
        print("Testing original model with a simple forward pass...")
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        
        # Move input to same device as model
        device = next(model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            outputs = model(**test_input)
            print(f"Original model forward pass successful. Output shape: {outputs.logits.shape}")
            # Store first few logits for comparison
            original_logits = outputs.logits[0, -1, :10].cpu()
            print(f"Original model sample logits: {original_logits}")
        
        return store, key, original_logits, tokenizer
        
    except Exception as e:
        print(f"Error during model loading or processing: {e}")
        raise
    
    finally:
        # Clean up original model
        try:
            model_var = locals().get('model')
            if model_var is not None:
                del model_var
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def test_policy_integration(store, state_dict_key, original_logits, tokenizer):
    """
    Second phase: Initialize Policy with torchstore and test update functionality
    """
    print("\n=== PHASE 2: Testing Policy Integration ===")
    
    # Set up environment variables for vLLM distributed initialization
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    try:
        print("Initializing Policy with torchstore...")
        # Initialize Policy with torchstore integration
        policy = Policy(
            model="meta-llama/Meta-Llama-3-8B",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enforce_eager=True,
            resources=1,
            torchstore=store,
            state_dict_key=state_dict_key
        )
        
        print("Setting up Policy...")
        await policy.setup()
        print("Policy setup completed successfully!")
        
        # Test that the policy is working before update
        print("Testing Policy before update...")
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        
        # Get model from policy worker
        policy_model = policy.worker.model_runner.model
        device = next(policy_model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            outputs_before = policy_model(**test_input)
            print(f"Policy model (before update) forward pass successful. Output shape: {outputs_before.logits.shape}")
            before_logits = outputs_before.logits[0, -1, :10].cpu()
            print(f"Policy model (before update) sample logits: {before_logits}")
        
        # Now call update to load weights from torchstore
        print("Calling Policy.update() to load weights from torchstore...")
        success = await policy.update()
        
        if success:
            print("✅ Policy update successful!")
            
            # Test the model after update
            print("Testing Policy model after update...")
            with torch.no_grad():
                outputs_after = policy_model(**test_input)
                print(f"Policy model (after update) forward pass successful. Output shape: {outputs_after.logits.shape}")
                after_logits = outputs_after.logits[0, -1, :10].cpu()
                print(f"Policy model (after update) sample logits: {after_logits}")
            
            # Compare logits to verify the update worked
            logits_diff = torch.abs(after_logits - before_logits).max()
            print(f"Max difference in logits after update: {logits_diff}")
            
            # The logits should be very close to the original model's logits
            original_diff = torch.abs(after_logits - original_logits).max()
            print(f"Max difference from original model logits: {original_diff}")
            
            if original_diff < 1e-3:  # Should be very close due to same weights
                print("✅ Model weights appear to be correctly loaded from torchstore!")
            else:
                print("⚠️  Model weights may not have been loaded correctly - large difference detected")
                
        else:
            print("❌ Policy update failed!")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error during Policy testing: {e}")
        raise


def setup_distributed_fsdp():
    """Initialize distributed environment for FSDP with world_size=2"""
    if not dist.is_initialized():
        # Set up environment variables for FSDP=2
        os.environ["RANK"] = str(int(os.environ.get("RANK", "0")))
        os.environ["WORLD_SIZE"] = "2"
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12356")
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"])
        )
        print(f"Initialized distributed for FSDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}")


async def test_llama3_fsdp_torchstore_write():
    """
    FSDP Phase 1: Load Llama 3 8B with FSDP=2 and write state dict to torchstore
    """
    print("\n=== FSDP PHASE 1: Writing Llama 3 8B with FSDP=2 to TorchStore ===")
    
    # Setup distributed environment for FSDP
    print("Setting up distributed environment for FSDP=2...")
    setup_distributed_fsdp()
    
    # Create device mesh for FSDP with 2 shards
    device_mesh = init_device_mesh("cuda", (2,))
    print(f"Created device mesh: {device_mesh}")
    
    print("Initializing MultiProcessStore...")
    store = MultiProcessStore()
    
    print("Loading Llama 3.1 8B model from local path...")
    model_path = "/tmp/Meta-Llama-3.1-8B"
    
    try:
        # Load the model from local path - NOT using device_map since we'll use FSDP
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True  # Ensure we don't try to download
        )
        
        # Move model to current device before FSDP wrapping
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Wrap model with FSDP (shard_degree=2)
        print("Wrapping model with FSDP...")
        fsdp_model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,  # Preserves original parameter names
        )
        
        # Also load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True  # Ensure we don't try to download
        )
        
        print(f"FSDP Model loaded successfully. Model type: {type(fsdp_model)}")
        print(f"Model device: {next(fsdp_model.parameters()).device}")
        print(f"Model dtype: {next(fsdp_model.parameters()).dtype}")
        
        # Get the model's state dict from FSDP model
        print("Getting FSDP model state dict...")
        with FSDP.state_dict_type(fsdp_model, FSDP.StateDictType.FULL_STATE_DICT):
            state_dict = fsdp_model.state_dict()
        print(f"FSDP state dict contains {len(state_dict)} parameters")
        
        # Print some info about the state dict (only on rank 0)
        if dist.get_rank() == 0:
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"Total parameters: {total_params:,}")
            
            param_names = list(state_dict.keys())[:10]
            print(f"Sample parameter names: {param_names}")
        
        # Write state dict to torchstore (only on rank 0)
        if dist.get_rank() == 0:
            print("Writing FSDP state dict to torchstore...")
            key = "llama3_8b_fsdp_state_dict"
            await push_state_dict(store, state_dict, key)
            print(f"Successfully wrote FSDP state dict to torchstore with key: {key}")
        else:
            key = "llama3_8b_fsdp_state_dict"
        
        # Test a simple forward pass to verify FSDP model works
        print("Testing FSDP model with a simple forward pass...")
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        
        # Move input to same device as FSDP model
        device = next(fsdp_model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            outputs = fsdp_model(**test_input)
            print(f"FSDP model forward pass successful. Output shape: {outputs.logits.shape}")
            # Store first few logits for comparison (only on rank 0)
            if dist.get_rank() == 0:
                original_logits = outputs.logits[0, -1, :10].cpu()
                print(f"FSDP model sample logits: {original_logits}")
            else:
                original_logits = None
        
        return store, key, original_logits, tokenizer
        
    except Exception as e:
        print(f"Error during FSDP model loading or processing: {e}")
        raise
    
    finally:
        # Clean up FSDP model
        try:
            fsdp_model_var = locals().get('fsdp_model')
            if fsdp_model_var is not None:
                del fsdp_model_var
            
            model_var = locals().get('model')
            if model_var is not None:
                del model_var
        except:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def test_policy_integration_fsdp(store, state_dict_key, original_logits, tokenizer):
    """
    FSDP Phase 2: Initialize Policy with tensor_parallel_size=2 and test update functionality
    """
    print("\n=== FSDP PHASE 2: Testing Policy Integration with Tensor Parallel Size 2 ===")
    
    # Set up environment variables for vLLM distributed initialization
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")  # Different port to avoid conflicts
    
    try:
        print("Initializing Policy with tensor_parallel_size=2 and torchstore...")
        # Initialize Policy with tensor parallel size 2 and torchstore integration
        policy = Policy(
            model="meta-llama/Meta-Llama-3-8B",
            tensor_parallel_size=2,  # Use tensor parallelism instead of FSDP for vLLM
            pipeline_parallel_size=1,
            enforce_eager=True,
            resources=2,  # 2 resources for 2 GPUs
            torchstore=store,
            state_dict_key=state_dict_key
        )
        
        print("Setting up Policy with distributed configuration...")
        await policy.setup()
        print("Policy setup completed successfully!")
        
        # Get model from policy worker (available on all ranks)
        policy_model = policy.worker.model_runner.model
        
        # Test that the policy is working before update (only on rank 0)
        before_logits = None
        if dist.get_rank() == 0:
            print("Testing Policy before update...")
            test_input = tokenizer("Hello, how are you?", return_tensors="pt")
            
            device = next(policy_model.parameters()).device
            test_input = {k: v.to(device) for k, v in test_input.items()}
            
            with torch.no_grad():
                outputs_before = policy_model(**test_input)
                print(f"Policy model (before update) forward pass successful. Output shape: {outputs_before.logits.shape}")
                before_logits = outputs_before.logits[0, -1, :10].cpu()
                print(f"Policy model (before update) sample logits: {before_logits}")
        
        # Now call update to load weights from torchstore
        print("Calling Policy.update() to load weights from torchstore...")
        success = await policy.update()
        
        if success:
            print("✅ Policy update successful!")
            
            # Test the model after update (only on rank 0)
            if dist.get_rank() == 0:
                print("Testing Policy model after update...")
                test_input = tokenizer("Hello, how are you?", return_tensors="pt")
                device = next(policy_model.parameters()).device
                test_input = {k: v.to(device) for k, v in test_input.items()}
                
                with torch.no_grad():
                    outputs_after = policy_model(**test_input)
                    print(f"Policy model (after update) forward pass successful. Output shape: {outputs_after.logits.shape}")
                    after_logits = outputs_after.logits[0, -1, :10].cpu()
                    print(f"Policy model (after update) sample logits: {after_logits}")
                
                # Compare logits to verify the update worked
                if before_logits is not None:
                    logits_diff = torch.abs(after_logits - before_logits).max()
                    print(f"Max difference in logits after update: {logits_diff}")
                
                # The logits should be very close to the original FSDP model's logits
                if original_logits is not None:
                    original_diff = torch.abs(after_logits - original_logits).max()
                    print(f"Max difference from original FSDP model logits: {original_diff}")
                    
                    if original_diff < 1e-2:  # Slightly higher tolerance for distributed differences
                        print("✅ FSDP model weights appear to be correctly loaded from torchstore!")
                    else:
                        print("⚠️  Model weights may not have been loaded correctly - large difference detected")
                else:
                    print("⚠️  Cannot compare with original logits (not available on this rank)")
                    
        else:
            print("❌ Policy update failed!")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error during FSDP Policy testing: {e}")
        raise


async def test_llama3_fsdp_torchstore():
    """
    Complete FSDP test: Write FSDP model to torchstore, then test Policy integration with tensor parallelism
    """
    try:
        # Phase 1: Write FSDP model to torchstore
        store, key, original_logits, tokenizer = await test_llama3_fsdp_torchstore_write()
        
        # Phase 2: Test Policy integration with tensor parallelism
        success = await test_policy_integration_fsdp(store, key, original_logits, tokenizer)
        
        if success:
            print("\n🎉 Complete FSDP test passed! Llama 3 8B FSDP model successfully loaded into Policy via TorchStore!")
        else:
            print("\n❌ FSDP test failed during Policy integration phase")
            
        return success
            
    except Exception as e:
        print(f"\n💥 FSDP test failed with error: {e}")
        raise
    
    finally:
        # Clean up distributed process group
        if dist.is_initialized():
            dist.destroy_process_group()
            print("Cleaned up distributed process group")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nFSDP test cleanup completed.")


async def test_llama3_torchstore():
    """
    Complete test: Write to torchstore, then test Policy integration
    """
    try:
        # Phase 1: Write model to torchstore
        store, key, original_logits, tokenizer = await test_llama3_torchstore_write()
        
        # Phase 2: Test Policy integration
        success = await test_policy_integration(store, key, original_logits, tokenizer)
        
        if success:
            print("\n🎉 Complete test passed! Llama 3 8B model successfully loaded into Policy via TorchStore!")
        else:
            print("\n❌ Test failed during Policy integration phase")
            
        return success
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        raise
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nTest cleanup completed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Llama 3 8B with TorchStore and Policy integration")
    parser.add_argument("--test", choices=["single", "fsdp", "both"], default="single",
                        help="Which test to run: single (default), fsdp, or both")
    args = parser.parse_args()
    
    async def run_tests():
        if args.test in ["single", "both"]:
            print("🚀 Starting Llama 3 8B torchstore test (single GPU)...")
            try:
                await test_llama3_torchstore()
            except Exception as e:
                print(f"Single GPU test failed: {e}")
            
        if args.test in ["fsdp", "both"]:
            print("\n🚀 Starting Llama 3 8B FSDP torchstore test (world_size=2)...")
            try:
                await test_llama3_fsdp_torchstore()
            except Exception as e:
                print(f"FSDP test failed: {e}")
                
        print("\n✨ All requested tests completed!")
    
    asyncio.run(run_tests())
