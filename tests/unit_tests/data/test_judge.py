# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from forge.controller.service.interface import ServiceInterface

from forge.data.judge import EvaluationMethodology, LLMJudge


# Mock classes to simulate VLLM RequestOutput structure
@dataclass
class MockCompletionOutput:
    text: str


@dataclass
class MockRequestOutput:
    outputs: List[MockCompletionOutput]


class TestLLMJudge:
    @pytest.fixture
    def mock_service(self):
        """Create a mock ServiceInterface for testing."""
        service = Mock(spec=ServiceInterface)
        service.generate = AsyncMock()
        return service

    @pytest.fixture
    def judge_majority(self, mock_service):
        return LLMJudge(
            judge_model=mock_service, methodology=EvaluationMethodology.MAJORITY
        )

    @pytest.fixture
    def judge_first_sample(self, mock_service):
        """Create an LLMJudge with FIRST_SAMPLE methodology."""
        return LLMJudge(
            judge_model=mock_service, methodology=EvaluationMethodology.FIRST_SAMPLE
        )

    @pytest.fixture
    def judge_pass_n(self, mock_service):
        """Create an LLMJudge with PASS_N methodology."""
        return LLMJudge(
            judge_model=mock_service, methodology=EvaluationMethodology.PASS_N
        )

    @pytest.mark.asyncio
    async def test_majority_vote_true_case(self, judge_majority):
        mock_outputs = [
            MockCompletionOutput(text="yes"),  # matches
            MockCompletionOutput(text="no"),  # doesn't match
            MockCompletionOutput(text="YES"),  # matches (case insensitive)
            MockCompletionOutput(text="yes "),  # matches (stripped)
            MockCompletionOutput(text="maybe"),  # doesn't match
        ]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(
            judge_majority, "_generate", return_value=mock_request_output
        ):
            result = await judge_majority.evaluate_response("What is 2+2?", "yes")
            assert result is True

    @pytest.mark.asyncio
    async def test_majority_vote_false_case(self, judge_majority):
        mock_outputs = [
            MockCompletionOutput(text="yes"),  # matches
            MockCompletionOutput(text="no"),  # doesn't match
            MockCompletionOutput(text="no"),  # doesn't match
            MockCompletionOutput(text="maybe"),  # doesn't match
            MockCompletionOutput(text="YES"),  # matches (case insensitive)
        ]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(
            judge_majority, "_generate", return_value=mock_request_output
        ):
            result = await judge_majority.evaluate_response("What is 2+2?", "yes")
            assert result is False

    @pytest.mark.asyncio
    async def test_first_sample_true_case(self, judge_first_sample):
        mock_outputs = [
            MockCompletionOutput(text="YES"),  # matches (case insensitive)
            MockCompletionOutput(text="no"),  # doesn't matter
            MockCompletionOutput(text="maybe"),  # doesn't matter
        ]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(
            judge_first_sample, "_generate", return_value=mock_request_output
        ):
            result = await judge_first_sample.evaluate_response("What is 2+2?", "yes")
            assert result is True

    @pytest.mark.asyncio
    async def test_first_sample_false_case(self, judge_first_sample):
        mock_outputs = [
            MockCompletionOutput(text="no"),  # doesn't match
            MockCompletionOutput(text="yes"),  # doesn't matter
            MockCompletionOutput(text="YES"),  # doesn't matter
        ]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(
            judge_first_sample, "_generate", return_value=mock_request_output
        ):
            result = await judge_first_sample.evaluate_response("What is 2+2?", "yes")
            assert result is False

    @pytest.mark.asyncio
    async def test_pass_n_true_case(self, judge_pass_n):
        mock_outputs = [
            MockCompletionOutput(text="no"),  # doesn't match
            MockCompletionOutput(text="maybe"),  # doesn't match
            MockCompletionOutput(text="YES"),  # matches (case insensitive)
            MockCompletionOutput(text="no"),  # doesn't match
        ]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(judge_pass_n, "_generate", return_value=mock_request_output):
            result = await judge_pass_n.evaluate_response("What is 2+2?", "yes")
            assert result is True

    @pytest.mark.asyncio
    async def test_pass_n_false_case(self, judge_pass_n):
        mock_outputs = [
            MockCompletionOutput(text="no"),  # doesn't match
            MockCompletionOutput(text="maybe"),  # doesn't match
            MockCompletionOutput(text="four"),  # doesn't match
            MockCompletionOutput(text="nope"),  # doesn't match
        ]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(judge_pass_n, "_generate", return_value=mock_request_output):
            result = await judge_pass_n.evaluate_response("What is 2+2?", "yes")
            assert result is False

    @pytest.mark.asyncio
    async def test_case_insensitive_and_whitespace_handling(self, judge_majority):
        mock_outputs = [
            MockCompletionOutput(text="YES"),  # matches
            MockCompletionOutput(text=" yes "),  # matches (with whitespace)
            MockCompletionOutput(text="Yes"),  # matches
            MockCompletionOutput(text="no"),  # doesn't match
            MockCompletionOutput(text="NO"),  # doesn't match
        ]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(
            judge_majority, "_generate", return_value=mock_request_output
        ):
            result = await judge_majority.evaluate_response("What is 2+2?", "  YES  ")
            assert result is True

    @pytest.mark.asyncio
    async def test_empty_outputs_handling(self, judge_majority):
        """Test handling of empty outputs list."""
        mock_outputs = []
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(
            judge_majority, "_generate", return_value=mock_request_output
        ):
            result = await judge_majority.evaluate_response("What is 2+2?", "yes")
            assert result is False  # 0 out of 0 match, which is not > 0//2 = 0

    @pytest.mark.asyncio
    async def test_unknown_evaluation_methodology(self, mock_service):
        """Test that unknown evaluation methodology raises ValueError."""
        judge = LLMJudge(judge_model=mock_service, methodology="INVALID")

        mock_outputs = [MockCompletionOutput(text="yes")]
        mock_request_output = MockRequestOutput(outputs=mock_outputs)

        with patch.object(judge, "_generate", return_value=mock_request_output):
            with pytest.raises(
                ValueError, match="Unknown evaluation methodology: INVALID"
            ):
                await judge.evaluate_response("What is 2+2?", "yes")
