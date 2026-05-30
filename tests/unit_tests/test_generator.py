# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Generator's _to_completions and _extract_tool_calls logic."""

import json
from unittest.mock import MagicMock

import pytest
from vllm.outputs import CompletionOutput, RequestOutput


def _import_error():
    """Check if there are import errors that would cause CI failures."""
    try:
        import forge.actors.generator  # noqa: F401

        return False
    except ImportError:
        return True


class _StubTokenizer:
    """Minimal stub tokenizer for initializing the Hermes tool parser in tests.

    The Hermes tool parser from vLLM requires:
    - get_vocab(): Returns vocab dict mapping tokens to ids
    - vocab: Direct vocab attribute
    - eos_token_id: End of sequence token id
    - encode(text, add_special_tokens=False): Encode text to token ids
    - decode(token_ids): Decode token ids to text
    - <tool_call> and </tool_call> tokens in vocab (for streaming support)
    """

    def __init__(self):
        # Include tool call tokens that Hermes parser validates in __init__
        # (needed for streaming, but validated even for non-streaming use)
        self.vocab = {
            "<tool_call>": 1,
            "</tool_call>": 2,
        }
        self._id_to_token = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        """Return vocabulary dict (required by Hermes tool parser)."""
        return self.vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token ids. Returns ids for known tokens, empty otherwise."""
        if text in self.vocab:
            return [self.vocab[text]]
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids to text."""
        return "".join(self._id_to_token.get(tid, chr(tid)) for tid in token_ids)


@pytest.fixture(scope="module")
def stub_tokenizer():
    """Create a stub tokenizer compatible with Hermes tool parser."""
    return _StubTokenizer()


@pytest.fixture
def generator_with_hermes(stub_tokenizer):
    """Create Generator with hermes parser properly initialized."""
    from forge.actors.generator import Generator

    generator = Generator(
        engine_args={"model": "Qwen/Qwen3-0.6B"},
        sampling_params={"max_tokens": 64},
        tool_call_parser="hermes",
    )
    generator._tool_parser = generator._init_tool_parser(stub_tokenizer)
    generator.generator_version = 1

    return generator


def make_mock_request_output(
    prompt: str = "test prompt",
    outputs: list[dict] | None = None,
) -> RequestOutput:
    """Create a mock vLLM RequestOutput for testing _to_completions."""
    if outputs is None:
        outputs = [
            {"text": "test response", "token_ids": [1, 2, 3], "finish_reason": "stop"}
        ]

    mock_outputs = []
    for out in outputs:
        mock_output = MagicMock(spec=CompletionOutput)
        mock_output.text = out.get("text", "")
        mock_output.token_ids = out.get("token_ids", [1, 2, 3])
        mock_output.finish_reason = out.get("finish_reason", "stop")
        mock_output.logprobs = out.get("logprobs", None)
        mock_outputs.append(mock_output)

    mock_request_output = MagicMock(spec=RequestOutput)
    mock_request_output.prompt = prompt
    mock_request_output.prompt_token_ids = [100, 101, 102]
    mock_request_output.outputs = mock_outputs
    mock_request_output.num_cached_tokens = 0

    return mock_request_output


@pytest.mark.skipif(
    _import_error(),
    reason="Import error, likely due to missing dependencies on CI.",
)
class TestInitToolParser:
    """Test the _init_tool_parser method of Generator."""

    def test_init_hermes_parser(self, stub_tokenizer):
        """Test that passing tool_call_parser='hermes' initializes the parser."""
        from forge.actors.generator import Generator

        generator = Generator(
            engine_args={"model": "Qwen/Qwen3-0.6B"},
            sampling_params={"max_tokens": 64},
            tool_call_parser="hermes",
        )

        parser = generator._init_tool_parser(stub_tokenizer)

        assert parser is not None
        assert hasattr(parser, "extract_tool_calls")

    def test_init_parser_none_when_not_configured(self):
        """Test that no parser is created when tool_call_parser is None."""
        from forge.actors.generator import Generator

        generator = Generator(
            engine_args={"model": "Qwen/Qwen3-0.6B"},
            sampling_params={"max_tokens": 64},
            tool_call_parser=None,
        )

        assert generator.tool_call_parser is None

    def test_init_parser_invalid_parser_name(self, stub_tokenizer):
        """Test that invalid parser name returns None."""
        from forge.actors.generator import Generator

        generator = Generator(
            engine_args={"model": "Qwen/Qwen3-0.6B"},
            sampling_params={"max_tokens": 64},
            tool_call_parser="nonexistent_parser",
        )

        parser = generator._init_tool_parser(stub_tokenizer)
        assert parser is None


@pytest.mark.skipif(
    _import_error(),
    reason="Import error, likely due to missing dependencies on CI.",
)
class TestExtractToolCalls:
    """Test _extract_tool_calls with real parser initialization."""

    def test_extract_single_tool_call(self, generator_with_hermes):
        """Test extracting a single tool call."""
        generator = generator_with_hermes

        model_output = """<tool_call>
{"name": "calculator", "arguments": {"equation": "2 + 2"}}
</tool_call>"""

        result = generator._extract_tool_calls(model_output)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "calculator"

        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["equation"] == "2 + 2"

    def test_extract_tool_call_with_content_prefix(self, generator_with_hermes):
        """Test extracting tool call when there's content before it."""
        generator = generator_with_hermes

        model_output = """Let me calculate that for you.
<tool_call>
{"name": "calculator", "arguments": {"equation": "15 * 7"}}
</tool_call>"""

        result = generator._extract_tool_calls(model_output)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "calculator"
        assert "Let me calculate" in (result.content or "")

    def test_extract_tool_call_with_think_prefix(self, generator_with_hermes):
        """Test extracting tool call when there's <think> tags before it."""
        generator = generator_with_hermes

        model_output = """<think>
The user is asking for a math calculation. I should use the calculator tool.
Let me compute 2 + 2.
</think>
<tool_call>
{"name": "calculator", "arguments": {"equation": "2 + 2"}}
</tool_call>"""

        result = generator._extract_tool_calls(model_output)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "calculator"
        # <think> content should be preserved in the content field
        assert result.content is not None
        assert """<think>
The user is asking for a math calculation. I should use the calculator tool.
Let me compute 2 + 2.
</think>""" in (
            result.content
        )

    def test_extract_multiple_tool_calls(self, generator_with_hermes):
        """Test extracting multiple tool calls."""
        generator = generator_with_hermes

        model_output = """<tool_call>
{"name": "calculator", "arguments": {"equation": "2 + 2"}}
</tool_call>
<tool_call>
{"name": "calculator", "arguments": {"equation": "3 * 4"}}
</tool_call>"""

        result = generator._extract_tool_calls(model_output)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2

        equations = [
            json.loads(tc.function.arguments)["equation"] for tc in result.tool_calls
        ]
        assert "2 + 2" in equations
        assert "3 * 4" in equations

    def test_no_tool_call_in_output(self, generator_with_hermes):
        """Test when model output has no tool calls."""
        generator = generator_with_hermes

        model_output = "The capital of France is Paris."

        result = generator._extract_tool_calls(model_output)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_extract_tool_calls_no_parser(self):
        """Test _extract_tool_calls returns content as-is when no parser."""
        from forge.actors.generator import Generator

        generator = Generator(
            engine_args={"model": "Qwen/Qwen3-0.6B"},
            sampling_params={"max_tokens": 64},
            tool_call_parser=None,
        )
        generator._tool_parser = None

        result = generator._extract_tool_calls("Hello, world!")

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, world!"


@pytest.mark.skipif(
    _import_error(),
    reason="Import error, likely due to missing dependencies on CI.",
)
class TestToCompletions:
    """Test _to_completions with real parser initialization."""

    def test_to_completions_without_tool_parser(self):
        """Test _to_completions when no tool parser is configured."""
        from forge.actors.generator import Generator

        generator = Generator(
            engine_args={"model": "Qwen/Qwen3-0.6B"},
            sampling_params={"max_tokens": 64},
            tool_call_parser=None,
        )
        generator._tool_parser = None
        generator.generator_version = 1

        request_output = make_mock_request_output(
            prompt="What is 2 + 2?",
            outputs=[{"text": "The answer is 4.", "token_ids": [10, 20, 30]}],
        )

        completions = generator._to_completions(request_output, request_output.prompt)

        assert len(completions) == 1
        completion = completions[0]

        assert completion.tool_calls == []
        assert completion.content is None
        assert completion.text == "The answer is 4."
        assert not completion.has_tool_calls

    def test_to_completions_no_tool_call_with_parser(self, generator_with_hermes):
        """Test _to_completions when parser finds no tool calls."""
        generator = generator_with_hermes

        request_output = make_mock_request_output(
            prompt="What is the capital of France?",
            outputs=[
                {"text": "Paris is the capital of France.", "token_ids": [10, 20]}
            ],
        )

        completions = generator._to_completions(request_output, request_output.prompt)

        assert len(completions) == 1
        completion = completions[0]

        assert not completion.has_tool_calls
        assert completion.tool_calls == []
        assert completion.content == "Paris is the capital of France."

    def test_to_completions_multiple_outputs(self, generator_with_hermes):
        """Test _to_completions with multiple outputs (n > 1)."""
        generator = generator_with_hermes

        request_output = make_mock_request_output(
            prompt="Calculate something",
            outputs=[
                {
                    "text": """<tool_call>
{"name": "calculator", "arguments": {"equation": "1 + 1"}}
</tool_call>""",
                    "token_ids": [1, 2],
                },
                {"text": "The answer is obviously 2.", "token_ids": [3, 4]},
            ],
        )

        completions = generator._to_completions(request_output, request_output.prompt)

        assert len(completions) == 2
        # First completion has tool call
        assert completions[0].has_tool_calls
        assert completions[0].tool_calls[0].function.name == "calculator"
        args = json.loads(completions[0].tool_calls[0].function.arguments)
        assert args["equation"] == "1 + 1"
        # Second completion has no tool call
        assert not completions[1].has_tool_calls
        assert completions[1].content == "The answer is obviously 2."
