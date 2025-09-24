# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from monarch.actor import endpoint, ProcMesh

from vllm.transformers_utils.tokenizer import get_tokenizer

from forge.actors.policy import Policy
from forge.controller import ForgeActor, get_proc_mesh, stop_proc_mesh
from forge.data_models.completion import Completion


@dataclass
class LLMJudge(ForgeActor):
    """
    `LLM-based Judges` are typically generative models which are then prompted
    to evaluate responses. These models NEED prompt engineering to evaluate
    and may require more postprocessing
    """

    # Typically Policy, but effectively any service with `generate` method
    generator: ServiceInterface | None = None

    def __post_init__(self):
        super().__init__()
        self._judge_proc: ProcMesh | None = None

    @classmethod
    async def launch(
        cls: type["LLMJudge"],
        *,
        process_config: ProcessConfig,
        policy_cfg: Mapping,
        **kwargs,
    ):
        judge_procs = await get_proc_mesh(process_config=process_config)
        policy = await Policy.options(**policy_cfg.services.policy).as_service(
            **policy_cfg.policy
        )
        print("Launch policy type: ", type(policy))

        actor_name = kwargs.pop("name", cls.__name__)
        llm_judge = await judge_procs.spawn(actor_name, cls, generator=policy)
        llm_judge._judge_proc = judge_procs

        await llm_judge.setup.call()
        return llm_judge

    @endpoint
    async def setup(self):
        assert self.generator is not None, "Generator not initialized correctly"
        self.tokenizer = get_tokenizer(self.generator.engine_config.model)

    @classmethod
    async def shutdown(cls: type["LLMJudge"], actor: "LLMJudge"):
        assert (
            actor.generator is not None
        ), "Tried to shutdown a generator that was not initialized correctly"
        assert (
            actor._judge_proc is not None
        ), "Tried to shutdown a LLMJudge that was not initialized correctly"

        await actor.generator.shutdown()
        await stop_proc_mesh(actor._judge_proc)

    def _wrap_prompt(self, prompt: str, responses: list[str]) -> str:
        """
        Construct the string being passed to the generator

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
        formatted_request = self.tokenizer.apply_chat_template(
            as_chat, tokenize=False, add_generation_prompt=True
        )
        return formatted_request

    def _postprocess_output(self, output: List[Completion]) -> list[str]:
        return [output.text for output in response.outputs]

    @endpoint
    async def generate(
        self, prompt: str, responses: list[str], priority: int = 0
    ) -> list[str]:
        wrapped_prompt: str = self._wrap_prompt(prompt, responses)
        response: List[Completion] = await self.generator.generate.choose(
            prompt=wrapped_prompt
        )
        return self._postprocess_output(response)


@dataclass
class RewardModelJudge(ForgeActor):
    """
    `RewardModels` are typically discriminative models, post trained to
    evaluate responses without further prompting required.
    """

    # Typically Policy, but effectively any service with `generate` method
    generator: ServiceInterface | None = None

    def __post_init__(self):
        super().__init__()
        self._judge_proc: ProcMesh | None = None

    @classmethod
    async def launch(
        cls: type["LLMJudge"],
        *,
        process_config: ProcessConfig,
        policy_cfg: Mapping,
        **kwargs,
    ):
        judge_procs = await get_proc_mesh(process_config=process_config)
        policy = await Policy.options(**policy_cfg.services.policy).as_service(
            **policy_cfg.policy
        )

        actor_name = kwargs.pop("name", cls.__name__)
        llm_judge = await judge_procs.spawn(actor_name, cls, generator=policy)
        llm_judge._judge_proc = judge_procs

        return llm_judge

    # TODO: Add formatting for reward models
    def _wrap_prompt(self, prompt: str, responses: list[str]) -> str:
        return prompt

    @endpoint
    async def generate(
        self, prompt: str, responses: list[str], priority: int = 0
    ) -> list[str]:
        wrapped_prompt: str = self._wrap_prompt(prompt, responses)
        return await self.generator.generate.choose(prompt=wrapped_prompt)
