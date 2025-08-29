# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:

python -m apps.vllm.main --guided-decoding --num-samples 3

"""

import argparse
import asyncio
from argparse import Namespace
from typing import List

from forge.actors.policy import Policy, PolicyConfig, SamplingOverrides, WorkerConfig
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from vllm.outputs import CompletionOutput, RequestOutput


async def main():
    """Main application for running vLLM policy inference."""
    args = parse_args()

    # Create configuration objects
    policy_config, service_config = get_configs(args)

    # Resolve the Prompts
    if args.prompt is None:
        prompt = "What is 3+5?" if args.guided_decoding else "Tell me a joke"
    else:
        prompt = args.prompt

    # Run the policy
    await run_vllm(service_config, policy_config, prompt)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="VLLM Policy Inference Application")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--num-samples", type=int, default=2, help="Number of samples to generate"
    )
    parser.add_argument(
        "--guided-decoding", action="store_true", help="Enable guided decoding"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Custom prompt to use for generation"
    )
    return parser.parse_args()


def get_configs(args: Namespace) -> (PolicyConfig, ServiceConfig):

    worker_size = 2
    worker_params = WorkerConfig(
        model=args.model,
        tensor_parallel_size=worker_size,
        pipeline_parallel_size=1,
        enforce_eager=True,
        vllm_args=None,
    )

    sampling_params = SamplingOverrides(
        n=args.num_samples,
        guided_decoding=args.guided_decoding,
        max_tokens=16,
    )

    policy_config = PolicyConfig(
        worker_params=worker_params, sampling_params=sampling_params
    )
    service_config = ServiceConfig(
        procs_per_replica=worker_size, num_replicas=1, with_gpus=True
    )

    return policy_config, service_config


async def run_vllm(service_config: ServiceConfig, config: PolicyConfig, prompt: str):
    print("Spawning service...")
    policy = await spawn_service(service_config, Policy, config=config)

    async with policy.session():
        print("Requesting generation...")
        request_output: RequestOutput = await policy.generate.choose(prompt=prompt)
        responses: List[CompletionOutput] = request_output.outputs
        print("\nGeneration Results:")
        print("=" * 80)
        for batch, response in enumerate(responses):
            print(f"Sample {batch + 1}:")
            print(f"User: {prompt}")
            print(f"Assistant: {response.text}")
            print("-" * 80)
            print("\nShutting down...")

    await shutdown_service(policy)


if __name__ == "__main__":
    asyncio.run(main())
