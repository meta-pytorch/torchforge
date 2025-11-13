#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration test comparing torchtitan and Hugging Face RefModel outputs.
This script generates logits from both implementations and verifies they are close.

Example:
>>> python tests/integration_tests/test_titan_fwd_vs_hf_fwd.py \
        --model_name "Qwen/Qwen3-1.7B" \
        --titan-model-family "qwen3" \
        --titan-model-flavor "1.7B" \
"""

import argparse
import asyncio

from dataclasses import dataclass

import numpy as np
import torch

from forge.actors.reference_model import ReferenceModel
from forge.controller import ForgeActor
from forge.controller.provisioner import shutdown
from forge.util.config import _resolve_hf_model_path
from monarch.actor import endpoint
from torchtitan.config.job_config import Checkpoint, Compile, Model, Parallelism
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HfReferenceModel(ForgeActor):
    model_name: str
    device: torch.device | None = None
    dtype: torch.dtype = torch.float32

    @endpoint
    async def setup(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def forward(self, input_ids) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits
        return logits


def create_titan_config(model_name: str, model_family: str, model_flavor: str) -> dict:
    """Create torchtitan configuration for the given model."""
    resolved_hf_model_path = _resolve_hf_model_path(f"hf://{model_name}")
    config = {
        "model": Model(
            name=model_family,
            flavor=model_flavor,
            hf_assets_path=resolved_hf_model_path,
        ),
        "parallelism": Parallelism(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        "checkpoint": Checkpoint(
            enable=True,
            initial_load_path=resolved_hf_model_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
        ),
        "compile": Compile(
            enable=False,
        ),
    }
    return config


async def initialize_models(
    model_name: str, titan_model_family: str, titan_model_flavor: str
) -> tuple[ReferenceModel, HfReferenceModel]:
    """Initialize both torchtitan and HF models."""
    # Initialize torchtitan model
    titan_config = create_titan_config(
        model_name, titan_model_family, titan_model_flavor
    )
    titan_model = await ReferenceModel.options(
        procs=1, num_replicas=1, with_gpus=True
    ).as_service(**titan_config)

    # Initialize HF model
    hf_model = await HfReferenceModel.options(
        num_replicas=1, procs=1, with_gpus=True
    ).as_service(model_name=model_name)

    print("Both models initialized successfully")
    return titan_model, hf_model


def create_test_inputs(
    model_name: str, batch_size: int = 1, seq_len: int = 64
) -> tuple[torch.Tensor, AutoTokenizer]:
    """Create test inputs for the models."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create test prompts
    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning was the Word, and the Word was with God.",
        "To be or not to be, that is the question.",
        "Hello world! This is a test prompt for model comparison.",
    ]

    # Use first batch_size prompts
    prompts = test_prompts[:batch_size]

    # Tokenize
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=seq_len
    )

    input_ids = inputs["input_ids"]
    print(f"Created test inputs with shape: {input_ids.shape}")

    return input_ids, tokenizer


async def generate_logits(
    titan_model: ReferenceModel, hf_model: HfReferenceModel, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate logits from both models."""
    print("Generating logits from torchtitan model...")
    titan_logits = await titan_model.forward.route(input_ids)

    print("Generating logits from HF model...")
    hf_logits = await hf_model.forward.route(input_ids)

    return titan_logits, hf_logits


def compare_logits(
    titan_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = True,
) -> dict[str, float]:
    """Compare logits from both models and compute metrics."""
    # Move to CPU for comparison
    titan_logits_cpu = titan_logits.detach().cpu().float()
    hf_logits_cpu = hf_logits.detach().cpu().float()

    # Basic shape check
    assert (
        titan_logits_cpu.shape == hf_logits_cpu.shape
    ), f"Shape mismatch: titan {titan_logits_cpu.shape} vs hf {hf_logits_cpu.shape}"

    # Compute various metrics
    diff = titan_logits_cpu - hf_logits_cpu
    abs_diff = torch.abs(diff)
    rel_diff = abs_diff / (torch.abs(hf_logits_cpu) + 1e-8)

    metrics = {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            titan_logits_cpu.flatten(), hf_logits_cpu.flatten(), dim=0
        ).item(),
    }

    # Check if tensors are close
    is_close = torch.allclose(titan_logits_cpu, hf_logits_cpu, rtol=rtol, atol=atol)
    metrics["is_close"] = is_close

    if verbose:
        print("=== Logits Comparison Results ===")
        print(f"Shapes: {titan_logits_cpu.shape}")
        print(f"Max absolute difference: {metrics['max_abs_diff']:.6f}")
        print(f"Mean absolute difference: {metrics['mean_abs_diff']:.6f}")
        print(f"Max relative difference: {metrics['max_rel_diff']:.6f}")
        print(f"Mean relative difference: {metrics['mean_rel_diff']:.6f}")
        print(f"Cosine similarity: {metrics['cosine_similarity']:.6f}")
        print(f"Are close (rtol={rtol}, atol={atol}): {is_close}")

        if not is_close:
            # Find positions with largest differences
            flat_abs_diff = abs_diff.flatten()
            top_diff_indices = torch.topk(
                flat_abs_diff, k=min(5, len(flat_abs_diff))
            ).indices
            print("Top 5 positions with largest absolute differences:")
            for i, idx in enumerate(top_diff_indices):
                pos = np.unravel_index(idx.item(), titan_logits_cpu.shape)
                titan_val = titan_logits_cpu[pos].item()
                hf_val = hf_logits_cpu[pos].item()
                diff_val = abs_diff[pos].item()
                print(
                    f"  {i+1}. Position {pos}: titan={titan_val:.6f}, hf={hf_val:.6f}, diff={diff_val:.6f}"
                )

    return metrics


def compare_probabilities(
    titan_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 10,
    verbose: bool = True,
) -> dict[str, float]:
    """Compare top-k probabilities from both models."""
    # Convert logits to probabilities
    titan_probs = torch.softmax(titan_logits.detach().cpu().float(), dim=-1)
    hf_probs = torch.softmax(hf_logits.detach().cpu().float(), dim=-1)

    # Get top-k predictions for the last token of first sequence
    titan_top_k = torch.topk(titan_probs[0, -1], k=top_k)
    hf_top_k = torch.topk(hf_probs[0, -1], k=top_k)

    if verbose:
        print("\n=== Top-K Token Predictions Comparison ===")
        print("TorchTitan Top-K:")
        for i, (prob, token_id) in enumerate(
            zip(titan_top_k.values, titan_top_k.indices)
        ):
            token = tokenizer.decode([token_id.item()])
            print(f"  {i+1}. '{token}' (id={token_id.item()}): {prob.item():.6f}")

        print("\nHugging Face Top-K:")
        for i, (prob, token_id) in enumerate(zip(hf_top_k.values, hf_top_k.indices)):
            token = tokenizer.decode([token_id.item()])
            print(f"  {i+1}. '{token}' (id={token_id.item()}): {prob.item():.6f}")

    # Calculate overlap in top-k predictions
    titan_top_tokens = set(titan_top_k.indices.tolist())
    hf_top_tokens = set(hf_top_k.indices.tolist())
    overlap = len(titan_top_tokens.intersection(hf_top_tokens))
    overlap_ratio = overlap / top_k

    metrics = {
        "top_k_overlap": overlap,
        "top_k_overlap_ratio": overlap_ratio,
        "top1_match": titan_top_k.indices[0].item() == hf_top_k.indices[0].item(),
    }

    if verbose:
        print(f"\nTop-{top_k} overlap: {overlap}/{top_k} ({overlap_ratio:.2%})")
        print(f"Top-1 prediction match: {metrics['top1_match']}")

    return metrics


async def run_comparison(
    model_name: str,
    titan_model_family: str,
    titan_model_flavor: str,
    batch_size: int = 1,
    seq_len: int = 64,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Run the full comparison pipeline."""
    titan_model, hf_model = await initialize_models(
        model_name, titan_model_family, titan_model_flavor
    )
    input_ids, tokenizer = create_test_inputs(model_name, batch_size, seq_len)
    titan_logits, hf_logits = await generate_logits(titan_model, hf_model, input_ids)
    logits_metrics = compare_logits(titan_logits, hf_logits, rtol, atol, verbose)
    prob_metrics = compare_probabilities(
        titan_logits, hf_logits, tokenizer, verbose=verbose
    )
    all_metrics = {**logits_metrics, **prob_metrics}
    return all_metrics


async def main():
    parser = argparse.ArgumentParser(
        description="Compare TorchTitan and HF model outputs"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name/path")
    parser.add_argument(
        "--titan-model-family", type=str, help="Model family from Torchtitan spec"
    )
    parser.add_argument(
        "--titan-model-flavor", type=str, help="Model size from Torchtitan spec"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for testing"
    )
    parser.add_argument(
        "--seq_len", type=int, default=64, help="Sequence length for testing"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3, help="Relative tolerance for comparison"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-3, help="Absolute tolerance for comparison"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    try:
        metrics = await run_comparison(
            model_name=args.model_name,
            titan_model_family=args.titan_model_family,
            titan_model_flavor=args.titan_model_flavor,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            rtol=args.rtol,
            atol=args.atol,
            verbose=not args.quiet,
        )
        print("\n=== FINAL SUMMARY ===")
        print(f"All close (rtol={args.rtol}, atol={args.atol}): {metrics['is_close']}")
        print(f"Max absolute difference: {metrics['max_abs_diff']:.6f}")
        print(f"Cosine similarity: {metrics['cosine_similarity']:.6f}")
        print(f"Top-k overlap: ({metrics['top_k_overlap_ratio']:.2%})")
    finally:
        await shutdown()


if __name__ == "__main__":
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
