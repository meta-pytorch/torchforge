# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum

try:
    from vllm.outputs import RequestOutput
except ImportError as e:
    print(f"Failed to import RequestOutput from vllm.outputs: {e}")
    RequestOutput = "RequestOutput"

from forge.controller.service.interface import ServiceInterface


class EvaluationMethodology(str, Enum):
    """Evaluation methodology for LLM Judge."""

    MAJORITY = "Majority"
    FIRST_SAMPLE = "First"
    PASS_N = "Pass N"


@dataclass
class LLMJudge:
    """Simple interface for Judges utilizing LLMs."""

    judge_model: ServiceInterface
    methodology: EvaluationMethodology = EvaluationMethodology.MAJORITY

    async def _generate(self, prompt: str) -> RequestOutput:
        """Internally generate responses."""
        return await self.judge_model.generate.choose(prompt=prompt)

    async def evaluate_response(self, prompt: str, response: str) -> float:
        """Evaluate a response to a prompt."""
        outputs: RequestOutput = await self._generate(prompt)
        match self.methodology:
            case EvaluationMethodology.MAJORITY:
                return await self._majority_vote(response, outputs)
            case EvaluationMethodology.FIRST_SAMPLE:
                return await self._first_sample(response, outputs)
            case EvaluationMethodology.PASS_N:
                return await self._pass_n(response, outputs)
            case _:
                raise ValueError(f"Unknown evaluation methodology: {self.methodology}")

    async def _majority_vote(self, response: str, outputs: RequestOutput) -> bool:
        """
        Return whether at least half of the outputs match the response
        """
        matching = 0
        response_normalized = response.lower().strip()

        for output in outputs.outputs:
            output_normalized = output.text.lower().strip()
            if response_normalized == output_normalized:
                matching += 1
            print(output.text)

        return matching > (len(outputs.outputs) // 2)

    async def _first_sample(self, response: str, outputs: RequestOutput) -> bool:
        """
        Returns whether there is a match to the first output
        """
        first_output = outputs.outputs[0]
        output_normalized = first_output.text.lower().strip()
        response_normalized = response.lower().strip()

        return output_normalized == response_normalized

    async def _pass_n(self, response: str, outputs: RequestOutput) -> bool:
        """
        Return whether any of the outputs match the response
        """
        response_normalized = response.lower().strip()

        for output in outputs.outputs:
            output_normalized = output.text.lower().strip()
            if response_normalized == output_normalized:
                return True

        return False
