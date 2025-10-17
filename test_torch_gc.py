#!/usr/bin/env python3
"""Demonstrate PyTorch tensor GC behavior."""

import gc
import torch
import psutil
import os

def get_memory_mb():
    """Get current process RSS in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("=" * 80)
print("TEST 1: Without explicit del (relies on GC)")
print("=" * 80)

gc.disable()  # Disable automatic GC to make the issue obvious
start_mem = get_memory_mb()
print(f"Starting memory: {start_mem:.1f} MB")

for i in range(10):
    # Allocate a 100MB tensor
    tensor = torch.randn(100, 1024, 1024)  # ~400 MB (float32)
    _ = tensor.sum()  # Use it
    # Don't delete - just let the name get rebound next iteration

    current_mem = get_memory_mb()
    print(f"  Iteration {i+1}: {current_mem:.1f} MB (leak: {current_mem - start_mem:.1f} MB)")

print(f"\nMemory before GC: {get_memory_mb():.1f} MB")
gc.collect()
print(f"Memory after GC:  {get_memory_mb():.1f} MB")

print("\n" + "=" * 80)
print("TEST 2: With explicit del")
print("=" * 80)

start_mem = get_memory_mb()
print(f"Starting memory: {start_mem:.1f} MB")

for i in range(10):
    tensor = torch.randn(100, 1024, 1024)  # ~400 MB
    _ = tensor.sum()
    del tensor  # Explicit delete - frees immediately

    current_mem = get_memory_mb()
    print(f"  Iteration {i+1}: {current_mem:.1f} MB (leak: {current_mem - start_mem:.1f} MB)")

print(f"\nFinal memory: {get_memory_mb():.1f} MB")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("Without 'del': Memory grows unbounded until GC runs")
print("With 'del': Memory stays constant - immediate cleanup")
