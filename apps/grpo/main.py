# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torchstore as ts
from datasets import load_dataset
from forge.actors.policy import Policy
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import _qwen3_hf_to_vllm
from forge.cli.config import parse
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import shutdown
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from forge.data.rewards import MathReward, ThinkingReward
from forge.data.utils import exclude_service
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint
from omegaconf import DictConfig
from torch import nn
from torchstore.state_dict_utils import DELIM
from transformers import AutoModelForCausalLM
from vllm.transformers_utils.tokenizer import get_tokenizer


def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    context_length = logits.shape[1] - input_ids.shape[1]

    # Truncate request logits and drop last
    logits = logits[:, context_length - 1 : -1]

    # Compute logprobs
    logprobs = torch.log_softmax(logits / temperature, dim=-1)
    logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)

    return logprobs


class SimpleGRPOLoss(nn.Module):
    """Simplified GRPO Loss for simplified single step updates
    Inspired by the Hugging Face TRL implementation:
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1624.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logprobs, ref_logprobs, advantages, padding_mask):
        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
        per_token_loss = -(per_token_policy_loss - self.beta * kl)
        loss = (
            ((per_token_loss * padding_mask).sum(dim=1))
            / (padding_mask.sum(dim=1).clamp(min=1.0))
        ).mean()
        return loss


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
    def request_tensor(self):
        tensor = torch.tensor(self.request_tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self):
        tensor = torch.tensor(self.response_tokens, dtype=torch.long)
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

        self.logger.info(f"Trainer model initialized on {self.device}")

    @endpoint
    async def train_step(self, batch: list[list[Episode]]):
        microbatch = batch[self.dp_rank]
        pad_id = microbatch[0].pad_id

        # prepare batch
        request = [e.request_tensor for e in microbatch]
        request = torch.stack(request).to(self.device)  # [b x s]

        response = [e.response_tensor for e in microbatch]
        response = torch.stack(response).to(self.device)  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in microbatch]
        ref_logprobs = torch.stack(ref_logprobs).to(self.device).squeeze()  # [b x s]

        advantages = [e.advantage for e in microbatch]
        advantages = torch.tensor(advantages).to(self.device).unsqueeze(-1)  # [b x 1]
        del batch

        input_ids = torch.cat([request, response], dim=1)
        mask = input_ids != pad_id
        logits = self.model(input_ids=input_ids, attention_mask=mask).logits
        logprobs = compute_logprobs(logits, response)
        del logits

        mask = response != pad_id
        loss = self.loss(logprobs, ref_logprobs, advantages, mask)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    @endpoint
    async def push_weights(self, version: int):
        """Update policy model weights with trainer's current weights."""
        key = f"{self.state_dict_key}{DELIM}{version}"  # Use version as unique id
        new_sd = _qwen3_hf_to_vllm(self.model.state_dict(), num_layers=28)
        start_time = time.time()
        await ts.put_state_dict(new_sd, key)
        end_time = time.time()
        self.logger.debug(
            f"Pushed weights to {key} in {end_time - start_time:.2f} seconds"
        )


@dataclass
class RewardActor(ForgeActor):
    """Reward actor that uses a list of scoring functions."""

    reward_functions: list[Callable]

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
        std = rewards.std(1, keepdim=True)
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

        def gsm8k_transform(sample):
            system_prompt = """
            Put all your scratchpad work between <think> and </think> tags.
            Your final answer should be between <answer> and </answer> tags otherwise it will not be scored.
            """
            request: str = sample["question"]
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            target: str = sample["answer"]
            formatted_target = target.split("#### ")[1]
            return {"request": formatted_request, "target": formatted_target}

        ds = load_dataset(
            self.path, self.revision, split=self.data_split, streaming=self.streaming
        )
        ds = ds.map(gsm8k_transform)
        ds = ds.shuffle()
        self._iterator = iter(ds)

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
    model = cfg.model
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens
    mlogger = get_metric_logger(
        "wandb",
        freq=1,
        project="grpo-training",
    )

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
        spawn_service(
            ServiceConfig(**cfg.dataset.service),
            DatasetActor,
            **exclude_service(cfg.dataset),
        ),
        spawn_service(
            ServiceConfig(**cfg.policy.service),
            Policy,
            **exclude_service(cfg.policy),
        ),
        spawn_service(
            ServiceConfig(**cfg.trainer.service),
            Trainer,
            **exclude_service(cfg.trainer),
        ),
        spawn_service(
            ServiceConfig(**cfg.replay_buffer.service),
            ReplayBuffer,
            **exclude_service(cfg.replay_buffer),
        ),
        spawn_service(
            ServiceConfig(**cfg.compute_advantages.service),
            ComputeAdvantages,
        ),
        spawn_service(
            ServiceConfig(**cfg.ref_model.service),
            RefModel,
            model_name=model,
        ),
        spawn_service(
            ServiceConfig(**cfg.reward_actor.service),
            RewardActor,
            reward_functions=[MathReward(), ThinkingReward()],
        ),
    )

    print("All services initialized successfully!")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
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
                await trainer.push_weights.call(policy_version)
                policy_version += 1
                await policy.update_weights.call()

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
            shutdown_service(policy),
            shutdown_service(trainer),
            shutdown_service(replay_buffer),
            shutdown_service(dataloader),
            shutdown_service(compute_advantages),
            shutdown_service(ref_model),
            shutdown_service(reward_actor),
            return_exceptions=True,
        )
        # TODO - add a global shutdown that implicitly shuts down all services
        # and remote allocations
        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
