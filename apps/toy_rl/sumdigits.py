# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import random
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torchstore as ts
from datasets import load_dataset
from forge.actors.policy import Policy
from forge.actors.reference_model import ReferenceModel  # noqa: F401
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import _qwen3_hf_to_vllm
from forge.cli.config import parse
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import shutdown
from forge.data.rewards import MathReward, ThinkingReward
from forge.losses.grpo_loss import SimpleGRPOLoss
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint
from omegaconf import DictConfig

from torch.utils.data import IterableDataset
from torchstore.state_dict_utils import DELIM
from transformers import AutoModelForCausalLM
from vllm.transformers_utils.tokenizer import get_tokenizer


def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    return selective_log_softmax(logits, input_ids)


@dataclass
class Episode:
    # TODO: add adtional layer for multi-turn
    episode_id: str
    request: str
    policy_version: int
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    # processed data
    response: str | None = None
    request_tokens: list[int] | None = None
    response_tokens: list[int] | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None

    @property
    def request_tensor(self, slice_obj=None):
        """Returns the full request tensor"""
        return self.get_request_tensor()

    @property
    def response_tensor(self):
        """Returns the full response tensor"""
        return self.get_response_tensor()

    def get_request_tensor(self, slice_obj=None):
        tokens = (
            self.request_tokens[slice_obj]
            if slice_obj is not None
            else self.request_tokens
        )
        tensor = torch.tensor(tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    def get_response_tensor(self, slice_obj=None):
        tokens = (
            self.response_tokens[slice_obj]
            if slice_obj is not None
            else self.response_tokens
        )
        tensor = torch.tensor(tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


@dataclass
class Group:
    group_id: str
    episodes: list[Episode]

    @classmethod
    def new_group(
        cls,
        group_id: int,
        group_size: int,
        request: str,
        policy_version: int,
        pad_id: int,
        request_len: int,
        response_len: int,
        target: Any = None,
    ):
        episodes = []
        for _ in range(group_size):
            episodes.append(
                Episode(
                    episode_id=str(uuid.uuid4()),
                    request=request,
                    policy_version=policy_version,
                    pad_id=pad_id,
                    request_len=request_len,
                    response_len=response_len,
                    target=target,
                )
            )
        return cls(str(group_id), episodes)


@dataclass
class Trainer(ForgeActor):
    """GRPO Trainer implementation for policy optimization."""

    model_name: str
    learning_rate: float = 1e-5
    beta: float = 0.1
    device: torch.device | None = None
    state_dict_key: str = "model_state_dict"
    dp_rank: int = 0  # TODO: support data parallelism, hard code it for now

    @endpoint
    async def setup(self):
        # import debugpy

        # debugpy.listen(5681)
        # print("[LEARNER MESH] Waiting for VS Code debugger to attach...")
        # debugpy.wait_for_client()
        # print("Attached!")
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.optimizer.zero_grad()

        self.loss = SimpleGRPOLoss(self.beta)
        self._tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

        self.logger.info(f"Trainer model initialized on {self.device}")

    @endpoint
    async def train_step(self, batch: list[list[Episode]]):
        microbatch = batch[self.dp_rank]
        pad_id = microbatch[0].pad_id

        # prepare batch
        request = [e.request_tensor for e in microbatch]
        request = torch.stack(request).to(self.device)  # [b x s]

        response = [e.get_response_tensor(slice(None, -1)) for e in microbatch]
        response = torch.stack(response).to(self.device)  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in microbatch]
        ref_logprobs = torch.stack(ref_logprobs).to(self.device).squeeze()  # [b x s]

        advantages = [e.advantage for e in microbatch]
        advantages = torch.tensor(advantages).to(self.device).unsqueeze(-1)  # [b x 1]
        del batch

        # prompt_ids = torch.LongTensor(microbatch[0].request_tokens)
        # token_ids = torch.LongTensor(microbatch[0].response_tokens)
        # ids = torch.cat([prompt_ids, token_ids])
        # input_ids = ids[:-1]
        # diff = 128 - input_ids.size(0)
        # input_ids = (
        #     F.pad(input_ids, (0, diff), value=pad_id).to(self.device).unsqueeze(0)
        # )

        # mask = input_ids != pad_id
        # logits = self.model(input_ids=input_ids, attention_mask=mask).logits

        # target_ids = ids[1:]
        # diff = 128 - target_ids.size(0)
        # target_ids = (
        #     F.pad(target_ids, (0, diff), value=pad_id).to(self.device).unsqueeze(0)
        # )

        # logprobs = compute_logprobs(logits, target_ids)
        # del logits

        # mask = target_ids != pad_id
        # loss = self.loss(logprobs, ref_logprobs, advantages, mask)

        # Process all microbatches
        batch_input_ids = []
        batch_target_ids = []
        batch_masks = []
        for microbatch_item in microbatch:
            prompt_ids = torch.LongTensor(microbatch_item.request_tokens)
            token_ids = torch.LongTensor(microbatch_item.response_tokens)
            ids = torch.cat([prompt_ids, token_ids])

            input_ids = ids[:-1]
            diff = 128 - input_ids.size(0)
            input_ids = F.pad(input_ids, (0, diff), value=pad_id)

            target_ids = ids[1:]
            diff = 128 - target_ids.size(0)
            target_ids = F.pad(target_ids, (0, diff), value=pad_id)

            batch_input_ids.append(input_ids)
            batch_target_ids.append(target_ids)
        # Stack into batched tensors
        input_ids = torch.stack(batch_input_ids).to(self.device)  # [batch_size, 128]
        target_ids = torch.stack(batch_target_ids).to(self.device)  # [batch_size, 128]
        # Create attention mask
        mask = input_ids != pad_id
        # Forward pass
        logits = self.model(input_ids=input_ids, attention_mask=mask).logits
        # Compute logprobs
        logprobs = compute_logprobs(logits, target_ids)
        del logits
        # Create target mask for loss computation
        target_mask = target_ids != pad_id
        # Compute loss

        loss = self.loss(logprobs, ref_logprobs, advantages, target_mask)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    @endpoint
    async def push_weights(self, version: int):
        """Update policy model weights with trainer's current weights."""
        key = f"{self.state_dict_key}{DELIM}{version}"  # Use version as unique id
        new_sd = _qwen3_hf_to_vllm(
            self.model.state_dict(), num_layers=self.model.config.num_hidden_layers
        )
        start_time = time.time()
        await ts.put_state_dict(new_sd, key)
        end_time = time.time()
        self.logger.debug(
            f"Pushed weights to {key} in {end_time - start_time:.2f} seconds"
        )


@dataclass
class RewardActor(ForgeActor):
    """Reward actor that uses a list of scoring functions."""

    # reward_functions: list[Callable]

    def __init__(self, reward_functions: list[Callable]):
        super().__init__()
        # import debugpy

        # debugpy.listen(5687)
        # print("[Reward] Waiting for VS Code debugger to attach...")
        # debugpy.wait_for_client()
        # print("Attached!")
        self.reward_functions = reward_functions

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        total_rewards = 0.0
        for reward_fn in self.reward_functions:
            reward = reward_fn(prompt, response, target)
            total_rewards += reward
        return total_rewards / len(self.reward_functions)


class ComputeAdvantages(ForgeActor):
    """Compute advantages for GRPO using reward signals."""

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        # TODO: add batch processing
        rewards = torch.tensor([[e.reward for e in group.episodes]])
        mean = rewards.mean(1, keepdim=True)
        # Use unbiased=False to disable Bessel's correction
        std = rewards.std(1, keepdim=True, unbiased=False)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


class RefModel(ForgeActor):
    def __init__(self, model_name, device: torch.device | None = None):
        super().__init__()
        self.model_name = model_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.logger.info(f"Model initialized on {self.device}")
        self._tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    @endpoint
    async def forward(self, episode: Episode) -> torch.Tensor:
        req, res = episode.request_tensor, episode.response_tensor
        input_ids = torch.cat([req, res]).to(self.device).unsqueeze(0)
        mask = input_ids != episode.pad_id

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits

        input_ids = input_ids[:, len(req) :]
        return compute_logprobs(logits, input_ids)


@dataclass
class SumDigitsDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=1000):
        self.min_digit_length = 2  # TODO: needs to come from config/argument
        self.max_digit_length = 3  # TODO: needs to come from config/argument
        self.max_numbers = max_samples
        self.data = self.generate_random_number()
        self._tokenizer = tokenizer

    def __iter__(self) -> Iterator[Any]:
        for data in self.data:
            answer = str(sum(int(x) for x in data))
            system_prompt = """
            A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant only gives very concise answers.
            """
            request: str = f"What is the sum of the digits of {data}"
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            yield {
                "question": formatted_request,
                "request": formatted_request,
                "answer": answer,
                "target": answer,
            }

    def generate_random_number(self) -> Iterator[str]:
        while True:
            yield self.generate_one()

    def generate_one(self) -> str:
        return "".join(
            str(random.randint(0, 9))
            for _ in range(random.randint(self.min_digit_length, self.max_digit_length))
        )


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str = "openai/gsm8k"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = True
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        self._iterator = iter(SumDigitsDataset(self._tokenizer))

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            return next(self._iterator)
        except StopIteration:
            return None

    @endpoint
    async def pad_token(self):
        return self._tokenizer.pad_token_id


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    # Get parameters from config with fallbacks
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens
    mlogger = get_metric_logger(
        "wandb",
        freq=1,
        project="sumdigits-training",
    )

    print("Started setup services!")

    # ---- Setup services ---- #
    await ts.initialize()
    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.services.dataset).as_service(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        Trainer.options(**cfg.services.trainer).as_service(**cfg.trainer),
        ReplayBuffer.options(**cfg.services.replay_buffer).as_service(
            **cfg.replay_buffer
        ),
        ComputeAdvantages.options(**cfg.services.compute_advantages).as_service(),
        RefModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        RewardActor.options(**cfg.services.reward_actor).as_service(
            reward_functions=[MathReward()]
        ),
    )

    print("All services initialized successfully!")
    import debugpy

    debugpy.listen(5678)
    print("Waiting for VS Code debugger to attach...")
    debugpy.wait_for_client()
    print("Attached!")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        _tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        pad_id = await dataloader.pad_token.choose()
        while True:
            sample = await dataloader.sample.choose()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return
            prompt, target = sample["request"], sample["target"]
            responses = await policy.generate.choose(prompt)
            version = await policy.get_version.choose()
            group = Group.new_group(
                group_id=rollout_count,
                group_size=group_size,
                request=prompt,
                policy_version=version,
                pad_id=pad_id,
                request_len=max_req_tokens,
                response_len=max_res_tokens,
                target=target,
            )

            # TODO: Parallelize the following calculation
            for episode, response in zip(group.episodes, responses.outputs):
                episode.request_tokens = responses.prompt_token_ids
                episode.response_tokens = response.token_ids
                episode.response = response.text
                episode.ref_logprobs = await ref_model.forward.choose(episode)
                episode.reward = await reward_actor.evaluate_response.choose(
                    prompt=prompt, response=response.text, target=target
                )
            advantages = await compute_advantages.compute.choose(group)
            for episode, advantage in zip(group.episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.choose(episode)

            avg_response_len = (
                sum(len(e.response_tokens) for e in group.episodes) / group_size
            )
            mlogger.log("avg_response_len/rollout", avg_response_len, rollout_count)
            buffer_size = await replay_buffer._numel.choose()
            mlogger.log("buffer_size/rollout", buffer_size, rollout_count)
            avg_reward = sum(e.reward for e in group.episodes) / group_size
            mlogger.log("avg_reward/rollout", avg_reward, rollout_count)
            # print(f"avg_reward/rollout_count: {avg_reward} at {rollout_count}")

            rollout_count += 1

    async def continuous_training():
        training_step = 0
        policy_version = 0
        while True:
            batch = await replay_buffer.sample.choose(
                curr_policy_version=policy_version
            )
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                loss = await trainer.train_step.choose(batch)
                training_step += 1
                mlogger.log("loss/training_step", loss, training_step)
                print(f"loss/training_step: {loss} at {training_step}")
                if training_step % 10 == 0:
                    # push weights after every 35 training steps
                    await trainer.push_weights.call(policy_version)
                    policy_version += 1
                    await policy.update_weights.call()
                    await replay_buffer.clear.call()

    print("Starting GRPO training loops...")
    # TODO: Start multiple rollouts once all serivces support it
    rollout_task = asyncio.create_task(continuous_rollouts())
    training_task = asyncio.create_task(continuous_training())

    try:
        await asyncio.gather(rollout_task, training_task)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        rollout_task.cancel()
        training_task.cancel()
    finally:
        print("Shutting down...")
        await asyncio.gather(
            dataloader.shutdown(),
            policy.shutdown(),
            trainer.shutdown(),
            replay_buffer.shutdown(),
            compute_advantages.shutdown(),
            ref_model.shutdown(),
            reward_actor.shutdown(),
        )
        # TODO - add a global shutdown that implicitly shuts down all services
        # and remote allocations
        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
