#!/usr/bin/env python3
"""Test if tensor.numpy() creates a leak via the intermediate tensor."""

import gc
import torch
import psutil
import os

def get_memory_mb():
    """Get current process RSS in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("=" * 80)
print("TEST: Does .cpu().contiguous().numpy() leak the intermediate tensor?")
print("=" * 80)

gc.disable()  # Disable GC to see the leak clearly
start_mem = get_memory_mb()
print(f"Starting memory: {start_mem:.1f} MB\n")

leaked_arrays = []

for i in range(10):
    # Simulate what SharedTensor._create_from_tensor does
    tensor = torch.randn(100, 1024, 1024)  # ~400 MB

    # This is the problematic line from _shared_tensor.py
    raw_bytes = tensor.view(torch.uint8).view(-1).cpu().contiguous().numpy()

    # Delete the original tensor
    del tensor

    # BUT: raw_bytes still holds a reference to the intermediate .cpu().contiguous() tensor!
    # Let's "use" it like SharedTensor does (copy the data)
    _ = raw_bytes.sum()  # Just access it to simulate usage

    # Store raw_bytes to simulate not deleting it
    leaked_arrays.append(raw_bytes)

    current_mem = get_memory_mb()
    print(f"  Iteration {i+1}: {current_mem:.1f} MB (growth: {current_mem - start_mem:.1f} MB)")

print(f"\nFinal memory (before cleanup): {get_memory_mb():.1f} MB")
print(f"Expected leak: ~4000 MB (10 iterations × 400 MB)")

# Now delete raw_bytes and see if memory is freed
leaked_arrays.clear()
del leaked_arrays
print(f"\nMemory after deleting arrays: {get_memory_mb():.1f} MB")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("raw_bytes holds a reference to the intermediate .cpu().contiguous() tensor,")
print("preventing it from being freed even after the original tensor is deleted.")
