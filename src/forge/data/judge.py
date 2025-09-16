# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from enum import Enum

from forge.controller.service.interface import ServiceInterface


class EvaluationMethodology(str, Enum):
    """Evaluation methodology for LLM Judge."""

    MAJORITY = "Majority"
    FIRST_SAMPLE = "First sample"
    PASS = "Pass"


@dataclass
class LLMJudge:
    """Simple interface for Judges utilizing LLMs."""

    judge_model: ServiceInterface = judge_model
    methodology: EvaluationMethodology = EvaluationMethodology.MAJORITY

    async def _generate(self, prompt: str) -> RequestOutput:
        """Internally generate responses."""
        return await self.judge_model.generate.call(prompt=prompt)

    async def evaluate_response(self, prompt: str, response: str) -> float:
        """Evaluate a response to a prompt."""
        outputs: RequestOutput = await self._generate(prompt)
        match self.methodology:
            case EvaluationMethodology.MAJORITY:
                return await self._majority_vote(response, outputs)
            case EvaluationMethodology.FIRST_SAMPLE:
                return await self._first_sample(response, outputs)
            case EvaluationMethodology.PASS:
                return await self._pass(response, outputs)
            case _:
                raise ValueError(f"Unknown evaluation methodology: {self.methodology}")
