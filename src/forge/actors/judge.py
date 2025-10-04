# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import auto, Enum

from monarch.actor import endpoint

from forge.actors.policy import Policy
from forge.data_models.completion import Completion


class EvaluationMode(Enum):
    """Enum for selecting how a judge should evaluate the provided args"""

    BEST_RESPONSE = auto()
    RESPONSE_CHECK = auto()


@dataclass
class Judge(Policy):
    """
    `LLM-based Judges` are typically generative models which are then prompted
    to evaluate responses. These models NEED prompt engineering to evaluate
    and may require more postprocessing
    """

    def _response_check(self, prompt: str, responses: list[str]) -> str:
        """
        Construct the generator input. Formats the request such that the generator
        will return a comma separated list with a [[GOOD]] or [[BAD]] evaluation
        for each response, corresponding to whether the model thinks it correct
        answers the prompt.

        Note: This is not a "good" prompt, it just demonstrates how to make one
        """

        system_prompt = f"""
        You are an expert fact checker. Given a prompt and response attempts, evaluate
        each attempt and return whether it accurately answers the prompt.
        Each response is formatted as [Response #<N>], where <N> represents the
        attempt.

        Your answer should be a comma separated list of "[[GOOD]]" or "[[BAD]]",
        corresponding to the same order as the reponses provided.

        - If the answer is irrelevant to the prompt, return "[[BAD]]".
        - If you are not confident that the answer accurately answers the prompt, return "[[BAD]]"
        - Only return "[[GOOD]]" if the attempt accurately answers the prompt

        Do not explain your reasoning, just provide your evaluations.
        Here is the prompt that generated the responses: {prompt}.
        """
        response_str = "\n".join(
            [f"[Response #{i+1}] {resp}" for i, resp in enumerate(responses)]
        )
        as_chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": response_str},
        ]
        tokenizer = self.processor.tokenizer.tokenizer
        formatted_request = tokenizer.apply_chat_template(
            as_chat, tokenize=False, add_generation_prompt=True
        )
        return formatted_request

    def _best_response(self, prompt: str, responses: list[str]) -> str:
        """
        Construct the generator input. Format the request such that the generator
        will respond with a single integer corresponding to the response the model
        thinks is most factually correct.

        Note: This is not a "good" prompt, it just demonstrates how to make one
        """

        system_prompt = f"""
        You are an expert evaluator. Evaluate the responses provided and return
        a single integer indicating which response is the most factually correct.
        Each response is formatted as [Response #<N>], where <N> represents the
        selection. Do not explain your reasoning, just provide a number.

        Here is the prompt that generated the responses: {prompt}.
        """
        response_str = "\n".join(
            [f"[Response #{i+1}] {resp}" for i, resp in enumerate(responses)]
        )
        as_chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": response_str},
        ]
        tokenizer = self.processor.tokenizer.tokenizer
        formatted_request = tokenizer.apply_chat_template(
            as_chat, tokenize=False, add_generation_prompt=True
        )
        return formatted_request

    def _postprocess_output(self, outputs: list[Completion]) -> list[str]:
        return [output.text for output in outputs]

    @endpoint
    async def evaluate(
        self,
        prompt: str,
        responses: None | list[str] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.BEST_RESPONSE,
    ) -> list[str]:
        _prompting: dict = {
            EvaluationMode.BEST_RESPONSE: self._response_check,
            EvaluationMode.ANSWER_CHECK: self._answer_check,
        }

        wrapped_prompt: str = _prompting[evaluation_mode](prompt, responses)
        response: List[Completion] = await self.generate._method(self, wrapped_prompt)
        return self._postprocess_output(response)


@dataclass
class RewardModelJudge(Policy):
    """
    `RewardModels` are typically discriminative models, post trained to
    evaluate responses without further prompting required.
    """

    # TODO: Add reward models formatting
    def wrapped_prompt(self, prompt: str, responses: list[str]) -> str:
        return prompt

    def _postprocess_output(self, outputs: list[Completion]) -> list[str]:
        return [output.text for output in outputs]

    @endpoint
    async def evaluate(
        self,
        prompt: str,
        responses: list[str],
    ) -> list[str]:
        wrapped_prompt: str = self._wrap_prompt(prompt, responses)
        response: List[Completion] = await self.generate._method(self, wrapped_prompt)
        return self._postprocess_output(response)
