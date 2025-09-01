# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from datasets import load_dataset
from forge.actors.policy import Policy, PolicyConfig, SamplingOverrides, WorkerConfig
from forge.actors.replay_buffer import ReplayBuffer
from forge.controller import ServiceConfig, spawn_service
from forge.controller.actor import ForgeActor
from forge.data.rewards import MathReward, ThinkingReward
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint
from torch import nn
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
    """Simplified GRPO Loss for simplified single step updates"""

    def __init__(self, epsilon=0.1, beta=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, logprobs, ref_logprobs, advantages, padding_mask):
        log_ratio = ref_logprobs.detach() - logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1

        pl = torch.exp(logprobs - logprobs.detach()) * advantages
        loss = -pl + self.beta * kl

        # Compute mean
        loss = (loss * padding_mask).sum() / (padding_mask.sum() + 1e-8)
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
    target: Optional[Any] = None
    # processed data
    response: Optional[str] = None
    request_tokens: Optional[list[int]] = None
    response_tokens: Optional[list[int]] = None
    ref_logprobs: Optional[torch.Tensor] = None
    reward: Optional[float] = None
    advantage: Optional[float] = None

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
            diff = self.request_len - tensor.shape[0]
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
        for i in range(group_size):
            Episode(
                episode_id=str(uuid.uuid4()),
                request=copy.deepcopy(messages),
                policy_version=policy_version,
                pad_id=pad_iddd,
                request_len=request_len,
                response_len=response_len,
                target=target,
            )
        return cls(group_id, episodes)


@dataclass
class Trainer(ForgeActor):
    """GRPO Trainer implementation for policy optimization."""

    model_name: str
    learning_rate: float = 1e-5
    beta: float = 0.1
    epsilon: float = 0.1
    device: torch.device | None = None

    def setup(self):
        # Set device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.optimizer.zero_grad()

        # Initialize loss
        self.loss = SimpleGRPOLoss(self.epsilon, self.beta)

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def train_step(self, batch: list[Episode]):
        total_loss = 0.0
        num_episodes_processed = 0
        pad_id = batch[0].pad_id

        # prepare batch
        request = [e.response_tokens for e in batch]
        request = torch.stack(request).to(self.device)

        response = [e.response_tokens for e in batch]
        response = torch.stack(response).to(self.device)

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs).to(self.device)

        advantages = [e.advantages for e in batch]
        advantages = torch.tensor(advantages).to(self.device).unsqueeze(-1)
        del batch

        # compute policy logprobs
        input_ids = torch.cat([request, response])
        mask = input_ids[1] != pad_id
        logits = self.model(input_ids=input_ids, attention_mask=mask).logits
        logprobs = compute_logprobs(logits, response)
        del logits

        # compute loss
        mask = (response != pad_id).unsqueeze(-1)
        loss = self.loss(logprobs, ref_logprobs, advantages, mask)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional but recommended for stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / len(batch) if batch else 0.0

        return {"loss": avg_loss, "episodes_processed": num_episodes_processed}

    @endpoint
    async def update_weights(self, policy_actor):
        """Update policy model weights with trainer's current weights."""
        # Time how long it takes to update weights
        start_time = time.time()

        # Set model to eval mode for weight extraction
        self.model.eval()

        # Extract current model state dict
        model_state_dict = self.model.state_dict()

        # Convert tensors to CPU for transfer (if they're on GPU)
        cpu_state_dict = {}
        for key, tensor in model_state_dict.items():
            cpu_state_dict[key] = tensor.cpu() if tensor.is_cuda else tensor

        # Update the policy actor's model weights
        await policy_actor.update_model_weights.choose(cpu_state_dict)

        # Set model back to training mode
        self.model.train()

        # Log the time taken
        end_time = time.time()
        self.logger.info(f"Updating weights took {end_time - start_time:.2f} seconds")


class RewardActor(ForgeActor):
    """Reward actor that uses a list of scoring functions."""

    def __init__(self, reward_functions: list[Callable]):
        super().__init__()
        self.reward_functions = reward_functions

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        total_reward = 0.0
        for reward_fn in self.reward_functions:
            reward = reward_fn(prompt, response, target)
            total_reward += reward
        return total_reward


class ComputeAdvantages(ForgeActor):
    """Compute advantages for GRPO using reward signals."""

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        # TODO: add batch processing
        rewards = torch.Tensor([[e.reward for e in group.episodes]])
        advantages = (rewards - rewards.me / an(1, keepdim=True)) / (
            rewards.std(1, keepdim=True) + 1e-4
        )
        return advantages.squeeze(0)


class RefModel(ForgeActor):
    def __init__(self, model_name, device: torch.device | None = None):
        super().__init__()
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        # Set model to eval mode for reference computations
        self.model.eval()

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def forward(self, episode: Episode) -> torch.Tensor:

        # Convert tokens to tensor
        req, res = episode.request_tensor, episode.response_tensor
        input_ids = torch.cat([request, response]).to(self.device).unsqueeze(0)
        mask = input_ids[1] != episode.pad_id

        # Compute logits
        with torch.inference():
            logits = model(input_ids=input_ids, attention_mask=mask).logits

        # Compute logprobs
        input_ids = input_ids[:, request.shape[1] :]
        logprobs = compute_logprobs(logits, input_ids)

        return logprobs


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str
    name: str
    data_split: str
    streaming: bool
    model: str

    @endpoint
    def setup(self):
        self.tokenizer = get_tokenizer(self.model)

        def gsm8k_transform(sample):
            request: str = sample["question"]
            formatted_request = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": request}],
                tokenize=False,
                add_generation_prompt=True,
            )
            target: str = sample["answer"]
            formatted_target = target.split("#### ")[1]
            return {"request": formatted_request, "target": formatted_target}

        ds = load_dataset(
            self.path, self.name, split=self.data_split, streaming=self.streaming
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
    def pad_token(self):
        if self.tokenizer.pad_token is None:
            return self.tokenizer.eos_token
        return self.tokenizer.pad_token


async def main():
    """Main GRPO training loop with rollout and training processes."""
    group_size = 1
    model = "Qwen/Qwen3-1.7B"
    max_req_tokens = 512
    max_res_tokens = 128

    # ---- Setup WandB Logger ---- #
    logger = get_metric_logger(
        "wandb",
        freq=1,
        project="grpo-training",
    )

    # ---- Setup services ---- #
    default_service_cfg = ServiceConfig(
        procs_per_replica=1,
        num_replicas=1,
    )

    policy = await spawn_service(
        default_service_cfg,
        Policy,
        PolicyConfig(
            num_workers=1,
            worker_params=WorkerConfig(model=model),
            sampling_params=SamplingOverrides(n=group_size, max_tokens=max_res_tokens),
            available_devices="3",
        ),
    )

    trainer = await spawn_service(
        default_service_cfg,
        Trainer,
        learning_rate=1e-5,
        beta=0.1,
        model_name=model,
        device=torch.device("cuda:1"),
    )

    replay_buffer = await spawn_service(
        default_service_cfg,
        ReplayBuffer,
        batch_size=4,
        max_policy_age=1,
    )

    dataloader = await spawn_service(
        default_service_cfg,
        DatasetActor,
        "openai/gsm8k",
        "main",
        data_split="train",
        streaming=True,
        model=model,
    )

    compute_advantages = await spawn_service(
        default_service_cfg,
        ComputeAdvantages,
        gamma=0.99,
        lambda_=0.95,
    )

    ref_model = await spawn_service(
        default_service_cfg,
        RefModel,
        model_name=model,
        device=torch.device("cuda:2"),
    )

    reward_actor = await spawn_service(
        default_service_cfg,
        RewardActor,
        reward_functions=[MathReward(), ThinkingReward()],
    )

    print("All services initialized successfully!")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = dataloader.pad_token.choose()
        # TODO: Move this into setup
        asyncio.create_task(policy.run_processing.call())
        while True:
            sample = await dataloader.sample.choose()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return
            prompt, target = sample["request"], sample["target"]
            version = 0  # await policy.get_current_version.choose()
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
            responses = await policy.generate.choose(prompt)
            for episode, response in zip(group.episodes, responses.outputs):
                episode.request_tokens = responses.prompt_token_ids
                episode.response_tokens = response.token_ids
                episode.ref_logprobs = await ref_model.forward.choose(episode)
                episode.reward = await reward_actor.evaluate_response.choose(
                    prompt=prompt, response=response.text, target=target
                )
            advantages = await compute_advantages.compute.choose(group)
            for episode, advantage in zip(group.episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.choose(episode)

            rollout_count += 1
            if rollout_count % 10 == 0:
                avg_reward = sum(e.reward for e in group.episodes) / len(group.episodes)
                print(
                    f"Generated {rollout_count} rollouts w/ average reward {avg_reward}"
                )
                logger.log("reward/rollout", avg_reward, rollout_count)

    async def continuous_training():
        training_step = 0
        while True:
            batch = await replay_buffer.sample.choose(curr_policy_version=0)
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                training_result = await trainer.train_step.choose(batch)
                training_step += 1
                if training_step % 10 == 0:
                    print(f"Completed {training_step} training steps")
                    if training_result:
                        loss_value = training_result.get("loss", 0.0)
                        print(f"Latest loss: {loss_value}")
                        logger.log("loss/training_step", loss_value, training_step)
                # await trainer.update_weights(policy)

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


if __name__ == "__main__":
    asyncio.run(main())
