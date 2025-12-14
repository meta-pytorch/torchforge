# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

from forge.rl import Policy
from vllm.transformers_utils.tokenizer import get_tokenizer

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical equation. Uses Python's eval() to compute the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "The mathematical equation to evaluate, e.g. '2 + 2' or '(3 * 4) / 2'",
                    },
                },
                "required": ["equation"],
                "additionalProperties": False,
            },
        },
    },
]

model_name = "Qwen/Qwen3-1.7B"


@pytest.mark.asyncio
async def test_generator_without_tool_parsing():
    """Test generator without tool parsing - raw output includes tool tags."""
    policy = await Policy.options(
        procs=1,
        num_replicas=1,
        with_gpus=True,
    ).as_service(
        engine_args={"model": model_name},
        sampling_params={"n": 1, "max_tokens": 64},
    )

    tokenizer = get_tokenizer(model_name)
    as_chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can evaluate mathematical equations.",
        },
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    formatted_request = tokenizer.apply_chat_template(
        as_chat,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = await policy.generate.route(formatted_request)

    assert response[0].tool_calls == []
    assert response[0].content is None

    return policy


@pytest.mark.asyncio
async def test_generator_with_tool_parsing():
    """Test generator with tool parsing - tool calls are extracted into structured format."""
    policy = await Policy.options(
        procs=1,
        num_replicas=1,
        with_gpus=True,
    ).as_service(
        engine_args={"model": model_name},
        sampling_params={"n": 1, "max_tokens": 256},
        tool_call_parser="hermes",
    )

    tokenizer = get_tokenizer(model_name)
    as_chat = [
        {
            "role": "system",
            "content": "/no_think You are a helpful assistant that can evaluate mathematical equations.",
        },
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    formatted_request = tokenizer.apply_chat_template(
        as_chat,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = await policy.generate.route(formatted_request)
    completion = response[0]

    if completion.has_tool_calls:
        for i, tool_call in enumerate(completion.tool_calls):
            print(f"Tool Call {i + 1}:")
            print(f"  ID: {tool_call.id}")
            print(f"  Type: {tool_call.type}")
            print(f"  Function: {tool_call.function.name}")
            print(f"  Arguments: {tool_call.function.arguments}")

            args = json.loads(tool_call.function.arguments)
            print(f"  Parsed args: {args}")

    if "<tool_call>" in completion.text:
        assert completion.has_tool_calls, "Tool calls should be parsed"
        calc_calls = [
            tc for tc in completion.tool_calls if tc.function.name == "calculator"
        ]
        if calc_calls:
            args = json.loads(calc_calls[0].function.arguments)
            assert "equation" in args, "Calculator should have equation argument"

    # Test request with no tools
    as_chat_no_tools = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": "/no_think What is the capital of France?"},
    ]
    formatted_request_no_tools = tokenizer.apply_chat_template(
        as_chat_no_tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    response_no_tools = await policy.generate.route(formatted_request_no_tools)
    completion_no_tools = response_no_tools[0]

    assert completion_no_tools.tool_calls == [], "Should have no tool calls"
    assert (
        completion_no_tools.content is not None
    ), "Should have content when no tools called"
    assert (
        completion_no_tools.content == completion_no_tools.text
    ), "Content should equal text when no tools"

    return policy
