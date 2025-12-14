# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json

from forge.rl import Policy
from vllm.transformers_utils.tokenizer import get_tokenizer

# Tool definitions - passed to apply_chat_template so the model knows what tools exist
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


async def test_generator_without_tool_parsing():
    """Test generator without tool parsing - raw output includes tool tags."""
    policy = await Policy.options(
        procs=1,
        num_replicas=1,
        with_gpus=True,
    ).as_service(
        engine_args={"model": model_name},
        sampling_params={"n": 1, "max_tokens": 2048},
    )

    tokenizer = get_tokenizer(model_name)
    as_chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can evaluate mathematical equations.",
        },
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    # Tools are passed HERE to the chat template - this tells the model what tools are available
    formatted_request = tokenizer.apply_chat_template(
        as_chat,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = await policy.generate.route(formatted_request)
    print("=" * 100)
    print("WITHOUT TOOL PARSING (raw output):")
    print("=" * 100)
    print(response[0].text)
    print("=" * 100)

    # No tool_calls parsed since we didn't enable the parser
    assert response[0].tool_calls == []
    assert response[0].content is None

    return policy  # Return for cleanup


async def test_generator_with_tool_parsing():
    """Test generator with tool parsing - tool calls are extracted into structured format."""
    policy = await Policy.options(
        procs=1,
        num_replicas=1,
        with_gpus=True,
    ).as_service(
        engine_args={"model": model_name},
        sampling_params={"n": 1, "max_tokens": 2048},
        # Enable tool parsing - specify the parser for your model
        # The parser extracts <tool_call> tags into structured ToolCall objects
        tool_call_parser="hermes",
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
    completion = response[0]

    print("=" * 100)
    print("WITH TOOL PARSING (structured output):")
    print("=" * 100)
    print(f"Raw text: {completion.text}")
    print("-" * 50)
    print(f"Has tool calls: {completion.has_tool_calls}")
    print(f"Content (thinking): {completion.content}")
    print("-" * 50)

    if completion.has_tool_calls:
        for i, tool_call in enumerate(completion.tool_calls):
            print(f"Tool Call {i + 1}:")
            print(f"  ID: {tool_call.id}")
            print(f"  Type: {tool_call.type}")
            print(f"  Function: {tool_call.function.name}")
            print(f"  Arguments: {tool_call.function.arguments}")

            # Parse arguments JSON
            args = json.loads(tool_call.function.arguments)
            print(f"  Parsed args: {args}")

    print("=" * 100)

    # If the model called a tool, we should have structured data
    if "<tool_call>" in completion.text:
        assert completion.has_tool_calls, "Tool calls should be parsed"
        # Check that calculator was called
        calc_call = completion.get_tool_call_by_name("calculator")
        if calc_call:
            args = json.loads(calc_call.function.arguments)
            assert "equation" in args, "Calculator should have equation argument"

    return policy  # Return for cleanup


async def main():
    """Run tool parsing test."""
    print("\n" + "=" * 100)
    print("TEST: Generator with tool parsing")
    print("=" * 100 + "\n")
    policy = await test_generator_with_tool_parsing()

    # Note: Proper shutdown would require the underlying actor, not the service interface
    # For now, just let the process exit naturally
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
