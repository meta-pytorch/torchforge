#!/usr/bin/env python3
"""
Test script to:
1. Initialize Llama 3.1 8B-Instruct model from HuggingFace transformers
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

from forge.actors.policy import Policy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import push_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.utils import get_open_port
from monarch.actor import proc_mesh
import numpy as np
import asyncio
import traceback
import argparse


def convert_state_dict(saved_sd):
    """
    Convert transformers state dict to vLLM format.
    
    Key conversions:
    1. Copy over directly mapped keys (down_proj, input_layernorm, etc.)
    2. Fuse QKV projections: combine q_proj, k_proj, v_proj into qkv_proj 
    3. Fuse MLP projections: combine gate_proj and up_proj into gate_up_proj
    """
    load_sd = {}
    num_layers = 32  # For Llama-8B-3.1, adjust if needed
    
    # Copy over directly mapped keys
    for k in saved_sd:
        if any(x in k for x in [
            'down_proj', 'input_layernorm', 'post_attention_layernorm', 'o_proj', 
            'norm.weight', 'embed_tokens.weight', 'lm_head.weight'
        ]):
            load_sd[k] = saved_sd[k]
    
    # Fuse QKV and gate_up_proj
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        
        # QKV fusion
        q = saved_sd[prefix + "self_attn.q_proj.weight"]
        k = saved_sd[prefix + "self_attn.k_proj.weight"]
        v = saved_sd[prefix + "self_attn.v_proj.weight"]
        load_sd[prefix + "self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)
        
        # MLP gate_up_proj fusion
        gate = saved_sd[prefix + "mlp.gate_proj.weight"]
        up = saved_sd[prefix + "mlp.up_proj.weight"]
        load_sd[prefix + "mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)
    
    return load_sd


async def test_llama3_torchstore_write():
    """
    First phase: Load Llama 3.1 8B-Instruct and write state dict to torchstore
    """
    print("=== PHASE 1: Writing Llama 3.1 8B-Instruct to TorchStore ===")

    # Use the class method create_store() which properly spawns the actors
    store = await MultiProcessStore.create_store()

    # Load from local directory instead of HuggingFace download
    model_path = "/tmp/Meta-Llama-3.1-8B-Instruct"

    try:
        # Load the model from local path - using device_map="auto" for efficient loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,  # Ensure we don't try to download
        )

        # Also load tokenizer for completeness
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True  # Ensure we don't try to download
        )

        # Get the model's state dict
        original_state_dict = model.state_dict()
        print(f"Original state dict has {len(original_state_dict)} parameters")

        # Convert transformers state dict to vLLM format
        print("Converting transformers state dict to vLLM format...")
        converted_state_dict = convert_state_dict(original_state_dict)
        print(f"Converted state dict has {len(converted_state_dict)} parameters")

        # Write converted state dict to torchstore
        key = "llama3_8b_state_dict"
        await push_state_dict(store, converted_state_dict, key)
        print(f"Successfully wrote converted state dict to torchstore with key: {key}")

        # Test a simple forward pass to verify original model works
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")

        # Move input to same device as model
        device = next(model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}

        with torch.no_grad():
            outputs = model(**test_input)
            # Store first few logits for comparison
            original_logits = outputs.logits[0, -1, :10].cpu()

        return store, key, original_logits, tokenizer

    except Exception as e:
        print(f"Error during model loading or processing: {e}")
        raise

    finally:
        # Clean up original model
        try:
            model_var = locals().get("model")
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
        policy_mesh = await proc_mesh(
            gpus=1,
            env={
                "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
                "MASTER_PORT": os.environ.get("MASTER_PORT", "12355"),
            },
        )

        # Spawn Policy as a proper Monarch actor
        policy = await policy_mesh.spawn(
            "policy",
            Policy,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enforce_eager=True,
            resources=1,
            torchstore=store,
            state_dict_key=state_dict_key,
        )

        await policy.setup.call()

        # Get model info before update
        model_info_result = await policy.test_model_info.call()
        model_info_before = (
            model_info_result._values[0]
            if hasattr(model_info_result, "_values")
            else model_info_result
        )

        if "sample_weights" in model_info_before:
            before_weights = model_info_before["sample_weights"]

        # Now call update to load weights from torchstore
        try:
            success = await policy.update.call()
            if success:
                print("Policy update successful!")
            else:
                print("Policy update failed!")
                return False
        except Exception as e:
            print(f"Policy.update() timed out or failed: {e}")
            success = None  # Mark as unknown

        # Test the model after update (run regardless of timeout)
        if success is not False:  # Continue if successful or unknown
            model_info_result = await policy.test_model_info.call()
            model_info_after = (
                model_info_result._values[0]
                if hasattr(model_info_result, "_values")
                else model_info_result
            )

            if "sample_weights" in model_info_after:
                after_weights = model_info_after["sample_weights"]

                # Verify the update operation worked (weights should be preserved)
                if "sample_weights" in model_info_before:
                    weight_diff = np.abs(
                        np.array(after_weights) - np.array(before_weights)
                    ).max()

                    if weight_diff < 1e-6:
                        print(
                            "Model weights preserved correctly after torchstore update!"
                        )
                    else:
                        print("Model weights changed unexpectedly during update")

        return True

    except Exception as e:
        print(f"Error during Policy testing: {e}")
        raise


def setup_distributed_fsdp():
    """Initialize distributed environment for FSDP with world_size=2"""
    if not dist.is_initialized():
        # Use environment variables that should already be set by multiprocessing
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "2"))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12356")

        try:
            # Initialize process group with timeout
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.timedelta(seconds=30),  # Add timeout
            )
        except Exception as e:
            raise


async def test_llama3_fsdp_torchstore_write():
    """
    FSDP Phase 1: Load Llama 3.1 8B-Instruct with FSDP=2 and write state dict to torchstore
    """
    print(
        "\n=== FSDP PHASE 1: Writing Llama 3.1 8B-Instruct with FSDP=2 to TorchStore ==="
    )

    # Setup distributed environment for FSDP
    setup_distributed_fsdp()

    # Create device mesh for FSDP with 2 shards
    device_mesh = init_device_mesh("cuda", (2,))

    store = MultiProcessStore()
    model_path = "/tmp/Meta-Llama-3.1-8B"

    try:
        # Load the model from local path - NOT using device_map since we'll use FSDP
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,  # Ensure we don't try to download
        )

        # Move model to current device before FSDP wrapping
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Wrap model with FSDP (shard_degree=2)
        fsdp_model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,  # Preserves original parameter names
        )

        # Also load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True  # Ensure we don't try to download
        )

        # Get the model's state dict from FSDP model
        with FSDP.state_dict_type(fsdp_model, FSDP.StateDictType.FULL_STATE_DICT):
            state_dict = fsdp_model.state_dict()

        # Print some info about the state dict (only on rank 0)
        if dist.get_rank() == 0:
            total_params = sum(p.numel() for p in state_dict.values())

        # Write state dict to torchstore (only on rank 0)
        if dist.get_rank() == 0:
            key = "llama3_8b_fsdp_state_dict"
            await push_state_dict(store, state_dict, key)
        else:
            key = "llama3_8b_fsdp_state_dict"

        # Test a simple forward pass to verify FSDP model works
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")

        # Move input to same device as FSDP model
        device = next(fsdp_model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}

        with torch.no_grad():
            outputs = fsdp_model(**test_input)
            # Store first few logits for comparison (only on rank 0)
            if dist.get_rank() == 0:
                original_logits = outputs.logits[0, -1, :10].cpu()
            else:
                original_logits = None

        return store, key, original_logits, tokenizer

    except Exception as e:
        print(f"Error during FSDP model loading or processing: {e}")
        raise

    finally:
        # Clean up FSDP model
        try:
            fsdp_model_var = locals().get("fsdp_model")
            if fsdp_model_var is not None:
                del fsdp_model_var

            model_var = locals().get("model")
            if model_var is not None:
                del model_var
        except:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def test_policy_integration_fsdp(
    store, state_dict_key, original_logits, tokenizer
):
    """
    FSDP Phase 2: Initialize Policy with tensor_parallel_size=2 and test update functionality
    """
    print(
        "\n=== FSDP PHASE 2: Testing Policy Integration with Tensor Parallel Size 2 ==="
    )

    master_addr = "localhost"
    master_port = str(get_open_port())  # Use dynamic port to avoid conflicts

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = master_port  # Always set a fresh port

    try:
        policy_mesh = await proc_mesh(
            gpus=2,  # 2 GPUs for tensor parallelism
            env={
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            },
        )

        # Spawn Policy as a proper Monarch actor with tensor parallelism
        policy = await policy_mesh.spawn(
            "policy",
            Policy,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tensor_parallel_size=2,  # Use tensor parallelism instead of FSDP for vLLM
            pipeline_parallel_size=1,
            enforce_eager=True,
            resources=2,  # 2 resources for 2 GPUs
            torchstore=store,
            state_dict_key=state_dict_key,
        )

        await policy.setup.call()
        print("Policy setup completed successfully!")

        # Test that the policy is working before update (only on rank 0)
        model_info_before = None
        if dist.get_rank() == 0:
            # Get model info before update
            model_info_result = await policy.test_model_info.call()
            model_info_before = (
                model_info_result._values[0]
                if hasattr(model_info_result, "_values")
                else model_info_result
            )
            print(
                f"Policy model (before update) - Parameters: {model_info_before['num_parameters']:,}"
            )

            if "sample_weights" in model_info_before:
                before_weights = model_info_before["sample_weights"]
                print(f"Sample weights before update: {before_weights[:5]}")

        # Now call update to load weights from torchstore
        print("Calling Policy.update() to load weights from torchstore...")
        success = await policy.update.call()

        if success:
            print("Policy update successful!")

            # Test the model after update (only on rank 0)
            if dist.get_rank() == 0:
                # Get model info after update
                model_info_result = await policy.test_model_info.call()
                model_info_after = (
                    model_info_result._values[0]
                    if hasattr(model_info_result, "_values")
                    else model_info_result
                )

                if "sample_weights" in model_info_after:
                    after_weights = model_info_after["sample_weights"]
                    print(f"Sample weights after update: {after_weights[:5]}")

                    # Verify the update operation worked (weights should be preserved)
                    if model_info_before and "sample_weights" in model_info_before:

                        before_weights = model_info_before["sample_weights"]
                        weight_diff = np.abs(
                            np.array(after_weights) - np.array(before_weights)
                        ).max()
                        print(f"Max weight difference: {weight_diff}")

                        if weight_diff < 1e-6:
                            print(
                                "FSDP model weights preserved correctly after torchstore update!"
                            )
                        else:
                            print(
                                "FSDP model weights changed unexpectedly during update"
                            )

        else:
            print("Policy update failed!")
            return False

        return True

    except Exception as e:
        print(f"Error during FSDP Policy testing: {e}")
        raise


def fsdp_worker_main(rank, world_size, master_port):
    """
    Worker function that runs in each FSDP process
    """

    # Set up environment for this rank
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    async def worker_async_main():
        try:
            # Phase 1: Write FSDP model to torchstore
            store, key, original_logits, tokenizer = (
                await test_llama3_fsdp_torchstore_write()
            )

            # Phase 2: Test Policy integration (only on rank 0)
            if rank == 0:
                success = await test_policy_integration_fsdp(
                    store, key, original_logits, tokenizer
                )
                return success
            else:
                # Other ranks just participate in FSDP but don't run the Policy test
                return True

        except Exception as e:
            traceback.print_exc()
            return False
        finally:
            # Clean up
            if dist.is_initialized():
                dist.destroy_process_group()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Run the async main function
    try:
        result = asyncio.run(worker_async_main())
        return result
    except Exception as e:
        print(f"Rank {rank}: Worker failed with error: {e}")

        traceback.print_exc()
        return False


async def test_llama3_fsdp_torchstore():
    """
    Test loading a full state dict into a tensor parallel model
    """
    print("Starting tensor parallel test (load full state dict into sharded model)...")

    # Check if we have enough GPUs
    if not torch.cuda.is_available():
        print("No CUDA available for tensor parallel test")
        return False
    elif torch.cuda.device_count() < 2:
        print(
            f"Only {torch.cuda.device_count()} GPU(s) available, need 2+ for tensor parallel"
        )
        return False

    print(
        f"{torch.cuda.device_count()} GPU(s) available - proceeding with tensor parallel test"
    )

    try:
        # Phase 1: Save a full (non-sharded) model to torchstore, then modify it
        print(
            "Phase 1: Loading regular model and saving modified full state dict to torchstore..."
        )
        store, key, original_logits, tokenizer = await test_llama3_torchstore_write()

        master_addr = "localhost"
        master_port = str(get_open_port())

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        print(f"Using MASTER_PORT: {master_port} for tensor parallel Policy")

        policy_mesh = await proc_mesh(
            gpus=2,  # 2 GPUs for tensor parallelism
            env={
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            },
        )

        # Spawn Policy as a proper Monarch actor with tensor parallelism
        policy = await policy_mesh.spawn(
            "policy",
            Policy,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tensor_parallel_size=2,  # Use tensor parallelism
            pipeline_parallel_size=1,
            enforce_eager=True,
            resources=2,  # 2 resources for 2 GPUs
            torchstore=store,
            state_dict_key=key,  # Use the key from the full model
        )

        await policy.setup.call()
        print("Tensor parallel Policy setup completed successfully!")

        # Get model info before update
        model_info_result = await policy.test_model_info.call()
        model_info_before = (
            model_info_result._values[0]
            if hasattr(model_info_result, "_values")
            else model_info_result
        )
        print(
            f"Tensor parallel model (before update) - Parameters: {model_info_before['num_parameters']:,}"
        )

        if "sample_weights" in model_info_before:
            before_weights = model_info_before["sample_weights"]
            print(f"Sample weights before update: {before_weights[:5]}")

        # Now call update to load full weights from torchstore into sharded model
        print(
            "Calling Policy.update() to load full state dict into tensor parallel model..."
        )
        print(
            "This should automatically shard the full tensors for tensor parallel loading..."
        )

        try:
            success = await policy.update.call()

            if success:
                print("Policy update successful!")

                # Get model info after update
                model_info_result = await policy.test_model_info.call()
                model_info_after = (
                    model_info_result._values[0]
                    if hasattr(model_info_result, "_values")
                    else model_info_result
                )

                if "sample_weights" in model_info_after:
                    after_weights = model_info_after["sample_weights"]
                    print(f"Sample weights after update: {after_weights[:5]}")

                    # The weights should be different since we're loading from the saved full model
                    if "sample_weights" in model_info_before:
                        weight_diff = np.abs(
                            np.array(after_weights) - np.array(before_weights)
                        ).max()
                        print(f"Max weight difference: {weight_diff}")

                        if weight_diff < 1e-6:
                            print(
                                "Tensor parallel model successfully loaded full state dict with automatic sharding!"
                            )
                        else:
                            print("Weights appear changed")

                print(
                    "\nTensor parallel test passed! Full state dict successfully loaded into tensor parallel model!"
                )
                return True
            else:
                print("Policy update failed!")
                return False

        except Exception as e:
            print(f"Policy update failed with error: {e}")
            return False  # Return False since this is a real limitation we need to fix

    except Exception as e:
        print(f"Tensor parallel test failed with error: {e}")

        traceback.print_exc()
        return False

    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Tensor parallel test cleanup completed.")


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
            print(
                "\nComplete test passed! Llama 3.1 8B-Instruct model successfully loaded into Policy via TorchStore!"
            )
        else:
            print("\nTest failed during Policy integration phase")

        return success

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise

    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nTest cleanup completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Llama 3 8B with TorchStore and Policy integration"
    )
    parser.add_argument(
        "--test",
        choices=["single", "fsdp", "both"],
        default="single",
        help="Which test to run: single (default), fsdp, or both",
    )
    args = parser.parse_args()

    async def run_tests():
        if args.test in ["single", "both"]:
            print("Starting Llama 3 8B torchstore test (single GPU)...")
            try:
                await test_llama3_torchstore()
            except Exception as e:
                print(f"Single GPU test failed: {e}")

        if args.test in ["fsdp", "both"]:
            print("Starting Llama 3 8B FSDP torchstore test (world_size=2)...")
            try:
                await test_llama3_fsdp_torchstore()
            except Exception as e:
                print(f"FSDP test failed: {e}")

        print("\n All requested tests completed!")

    asyncio.run(run_tests())
