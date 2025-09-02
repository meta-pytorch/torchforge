# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1
python -m apps.vllm.main --config apps/vllm/config.yaml
"""

import asyncio
import sys
from typing import Any

import yaml

from forge.actors.policy import Policy, PolicyConfig
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_configs(cfg: dict) -> tuple[PolicyConfig, ServiceConfig, str]:
    # Instantiate PolicyConfig and ServiceConfig from nested dicts
    policy_config = PolicyConfig.from_dict(cfg["policy_config"])
    service_config = ServiceConfig(**cfg["service_config"])
    if "prompt" in cfg and cfg["prompt"] is not None:
        prompt = cfg["prompt"]
    else:
        gd = policy_config.sampling_params.guided_decoding
        prompt = "What is 3+5?" if gd else "Tell me a joke"
    return policy_config, service_config, prompt


async def run_vllm(service_config: ServiceConfig, config: PolicyConfig, prompt: str):
    print("Spawning service...")
    policy = await spawn_service(service_config, Policy, config=config)

    async with policy.session():
        print("Requesting generation...")
        response_output = await policy.generate.choose(prompt=prompt)

        print("\nGeneration Results:")
        print("=" * 80)
        for batch, response in enumerate(response_output.outputs):
            print(f"Sample {batch + 1}:")
            print(f"User: {prompt}")
            print(f"Assistant: {response.text}")
            print("-" * 80)

        print("\nShutting down...")

    await shutdown_service(policy)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Policy Inference Application")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    policy_config, service_config, prompt = get_configs(cfg)
    asyncio.run(run_vllm(service_config, policy_config, prompt))


if __name__ == "__main__":
    sys.exit(main())
