# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class GenerativeJudge:
    """
    Wrapper with custom prompting and post processing used for generative
    Judging. Represents a single verifier which could be LLM based
    RewardModels or LLM based judges.

    - `RewardModels` are typically discriminative models posttrained to
    evaluate responses. These models are specialized and need less prompting

    - `LLM-based Judges` are typically generative models which are then prompted
    to evaluate responses. These models NEED prompt engineering to evaluate
    and may require more postprocessing
    """

    # Typically Policy, but effectively any service with `generate` method
    generator: ServiceInterface
    prompt_wrapper: Optional[Callable[[str, list[str]], str]] = None
    output_postprocessor: Optional[Callable[[Any], Any]] = None

    def _wrap_prompt(self, prompt: str, responses: list[str]) -> str:
        """
        Construct the string being passed to the generator
        """
        if self.prompt_wrapper:
            return self.prompt_wrapper(prompt, responses)
        return prompt

    def _postprocess_output(self, output: Any) -> Any:
        """
        Postprocess generation results (metrics, aggregation, reducing)
        """
        if self.output_postprocessor:
            return self.output_postprocessor(output)
        return output

    async def generate(self, prompt: str, responses: list[str], priority: int = 0):
        wrapped_prompt: str = self._wrap_prompt(prompt, responses)
        response = await self.generator.generate.choose(prompt=wrapped_prompt)
        return self._postprocess_output(response)
