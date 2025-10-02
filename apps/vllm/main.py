# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1
python -m apps.vllm.main --config apps/vllm/llama3_8b.yaml
"""

import asyncio

import os
import re
import time

from forge.actors.coder import SandboxedCoder

from forge.actors.policy import Policy
from forge.cli.config import parse
from forge.controller.provisioner import shutdown

from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from omegaconf import DictConfig

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


def extract_python_code(text: str) -> str:
    """Extract Python code from ```python``` markdown code blocks.

    Args:
        text: Text that may contain Python code in markdown blocks

    Returns:
        Extracted Python code, or original text if no code blocks found
    """
    # Look for ```python code blocks
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Return the first match, stripped of extra whitespace
        return matches[0].strip()
    else:
        # If no python blocks found, return the original text (fallback)
        return text.strip()


async def run(cfg: DictConfig):
    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # Use different prompts based on whether we're using coder
    if cfg.with_coder:
        if (prompt := cfg.get("prompt")) is None:
            prompt = "Write a Python function that calculates the factorial of a number and test it with factorial(5). Include the test call and print the result. Please wrap your code in ```python``` code blocks."
    else:
        if (prompt := cfg.get("prompt")) is None:
            gd = cfg.policy.get("sampling_config", {}).get("guided_decoding", False)
            prompt = "What is 3+5?" if gd else "Tell me a joke"

    print("Spawning services...")
    policy = await Policy.options(**cfg.services.policy).as_service(**cfg.policy)

    coder = None
    with_coder = cfg.get("with_coder", False)
    n = cfg.get("num_calls", 100)
    if with_coder:
        print("Setting up coder...")
        coder = await SandboxedCoder.options(**cfg.services.coder).as_service(
            docker_image="docker://python:3.10",
            sqsh_image_path="python-image.sqsh",
            container_name="sandbox",
        )

    print("Requesting generation...")
    n = cfg.num_calls
    start = time.time()
    response_outputs: list[Completion] = await asyncio.gather(
        *[policy.generate.route(prompt=prompt) for _ in range(n)]
    )
    end = time.time()

    print(f"Generation of {n} requests completed in {end - start:.2f} seconds.")
    print(
        f"Generation with procs {cfg.services.policy.procs}, replicas {cfg.services.policy.num_replicas}"
    )

    print(f"\nGeneration Results (last one of {n} requests):")
    print("=" * 80)
    for batch, response in enumerate(response_outputs[-1]):
        print(f"Sample {batch + 1}:")
        print(f"User: {prompt}")
        print(f"Assistant: {response.text}")

        # If we have a coder, try to execute the generated code
        if coder and with_coder:
            print(f"Parsing and executing generated code...")
            try:
                # Extract Python code from tags
                python_code = extract_python_code(response.text)
                print(f"Extracted Code:\n{python_code}")
                print("-" * 40)

                # Execute the extracted code
                execution_result = await coder.execute.route(code=python_code)
                print(f"Execution Output:\n{execution_result}")
            except Exception as e:
                print(f"Execution Error: {e}")

        print("-" * 80)

    print("\nShutting down...")
    await policy.shutdown()
    if coder:
        await coder.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()
