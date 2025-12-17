# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for vLLM tool parsing in forge.

Tests the full tool-calling workflow: model generates tool call -> parse -> execute -> return result.

Requires GPU access.

Run:
    pytest tests/integration_tests/test_tool_parsing.py -v -s
"""

import json
import logging

import pytest
import pytest_asyncio
import torch

from forge.rl import Policy
from huggingface_hub import snapshot_download
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

MODEL_NAME = "Qwen/Qwen3-0.6B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical equation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "The mathematical equation to evaluate",
                    },
                },
                "required": ["equation"],
            },
        },
    },
]


def calculator(equation: str) -> str:
    """Safely evaluate a mathematical equation."""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/().^ ")
        if all(c in allowed for c in equation):
            result = eval(equation.replace("^", "**"))
            return str(result)
        return "Error: Invalid characters in equation"
    except Exception as e:
        return f"Error: {e}"


@pytest.fixture(scope="module")
def model_path():
    """Download model once for all tests in this module."""
    logger.info(f"Downloading model checkpoint: {MODEL_NAME}")
    cached_dir = snapshot_download(repo_id=MODEL_NAME)
    logger.info(f"Model downloaded to: {cached_dir}")
    return cached_dir


@pytest.fixture(scope="module")
def tokenizer():
    """Create tokenizer once for all tests in this module."""
    return get_tokenizer(MODEL_NAME)


@pytest_asyncio.fixture
async def policy(model_path):
    """Create and teardown policy service for each test."""
    logger.info("Setting up policy service...")
    policy = await Policy.options(
        procs=1,
        num_replicas=1,
        with_gpus=True,
    ).as_service(
        engine_args={"model": model_path},
        sampling_params={"n": 1, "max_tokens": 256},
        tool_call_parser="hermes",
    )

    yield policy

    # Teardown
    logger.info("Shutting down policy service...")
    await policy.shutdown()


@requires_cuda
@pytest.mark.asyncio
async def test_tool_parsing_multi_turn(policy, tokenizer):
    """
    Multi-turn conversation: tool call -> execute -> feed result back -> final answer.
    """
    messages = [
        {
            "role": "system",
            "content": "/no_think Use the calculator tool for math.",
        },
        {"role": "user", "content": "Calculate 123 + 456"},
    ]

    # First turn - get tool call
    formatted = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    response = await policy.generate.route(formatted)
    completion = response[0]

    assert completion.has_tool_calls, "Expected tool calls"
    tool_call = completion.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = calculator(args["equation"])

    # Add assistant response and tool result to conversation
    messages.append(
        {
            "role": "assistant",
            "content": completion.text,
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        }
    )

    # Second turn - get final answer
    formatted = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    response = await policy.generate.route(formatted)
    final = response[0]

    logger.info(f"Final answer: {final.text}")
    assert "579" in final.text, "Expected 123 + 456 = 579"

    logger.info("✅ test_tool_parsing_multi_turn passed!")


@requires_cuda
@pytest.mark.asyncio
async def test_content_without_tool_calls(policy, tokenizer):
    """
    Test that content equals text when no tool calls are made.

    When a request doesn't trigger tool usage, the completion's content
    field should equal the raw text output.
    """
    # Ask a non-math question that won't trigger the calculator tool
    messages = [
        {
            "role": "system",
            "content": "/no_think You are a helpful assistant.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]

    formatted_request = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = await policy.generate.route(formatted_request)
    completion = response[0]

    logger.info(f"Response text: {completion.text}")
    logger.info(f"Response content: {completion.content}")

    assert completion.tool_calls == [], "Should have no tool calls"
    assert completion.content is not None, "Should have content when no tools called"
    assert (
        completion.content == completion.text
    ), "Content should equal text when no tools"

    logger.info("✅ test_content_without_tool_calls passed!")
