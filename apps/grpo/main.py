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
from typing import Callable

import torch
from datasets import load_dataset
from forge.actors.policy import Policy, PolicyConfig, SamplingOverrides, WorkerConfig
from forge.actors.replay_buffer import ReplayBuffer
from forge.controller import ServiceConfig, spawn_service
from forge.controller.actor import ForgeActor
from forge.data.rewards import MathReward, ThinkingReward
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
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


@dataclass
class Episode:
    # TODO: add adtional layer for multi-turn
    episode_id: str
    request: str
    policy_version: int
    target: Optinoal[Any]
    # processed data
    response: Optional[str]
    request_tokens: Optional[list[int]]
    response_tokens: Optional[list[int]]
    ref_logprobs: Optional[torch.Tensor] = None
    reward: Optional[float] = None
    advantage: Optional[float] = None
    policy_version: Optional[int] = None

    @property
    def tokens(self):
        return self.request_tokens + self.response_tokens

    @property
    def mask(self):
        return [0] * len(self.request_tokens) + [1] * len(self.response_tokens)


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
        target: Any = None,
    ):
        episodes = []
        for i in range(group_size):
            Episode(
                episode_id=str(uuid.uuid4()),
                request=copy.deepcopy(messages),
                policy_version=policy_version,
                target=target,
            )
        return cls(group_id, episodes)


class Trainer(ForgeActor):
    """GRPO Trainer implementation for policy optimization."""

    def __init__(
        self,
        learning_rate: float = 1e-5,
        beta: float = 0.1,
        model_name: str = "",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta = beta  # KL penalty coefficient
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
        self.model.train()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def train_step(self, batch: list[Episode]):
        total_loss = 0.0
        num_groups_processed = 0

        # Batch logic -> move to replay replay_buffer
        input_ids = []
        advantages = []
        ref_logprobs = []
        for episode in batch:
            # collect infomration and batch + pad sequences
            input_ids.append(episode.response_tokens + episode.request_tokens)
            # TODO philip you are here !!!!!!!!!!!!!

        # loss reference:
        # https://github.com/pytorch/torchtune/blob/
        #   67ab86b94de9e7ac7dd9850113ebe69e2bbd307c/torchtune/dev/grpo/loss.py#L123
        # input_ids = tokenized["input_ids"].to(self.device)
        # attention_mask = tokenized["attention_mask"].to(self.device)
        #
        # # Compute current policy log probabilities using the model
        # current_logprobs = compute_sequence_logprobs(
        #     self.model, input_ids, attention_mask, requires_grad=True
        # )
        #
        # # Convert ref_logprobs and advantages to tensors
        # ref_logprobs_tensor = torch.stack(ref_logprobs_list).to(self.device)
        # advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32).to(
        #     self.device
        # )
        #
        # # Compute GRPO loss components
        # # Ratio between current policy and reference policy
        # ratio = torch.exp(current_logprobs - ref_logprobs_tensor)
        #
        # # Policy gradient loss weighted by advantages
        # pg_loss = -torch.mean(ratio * advantages_tensor)
        #
        # # KL penalty to prevent policy from deviating too far from reference
        # kl_penalty = self.beta * torch.mean(
        #     (current_logprobs - ref_logprobs_tensor) ** 2
        # )
        #
        # # Total GRPO loss
        # loss = pg_loss + kl_penalty
        # total_loss += loss.item()
        # num_groups_processed += len(groups)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional but recommended for stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        avg_loss = total_loss / len(batch) if batch else 0.0

        return {"loss": avg_loss, "groups_processed": num_groups_processed}

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
        # philip: compare against
        # https://github.com/pytorch/torchtune/blob/
        #   67ab86b94de9e7ac7dd9850113ebe69e2bbd307c/torchtune/dev/rl/rewards.py#L270
        total_reward = 0.0
        for reward_fn in self.reward_functions:
            reward = reward_fn(prompt, response, target)
            total_reward += reward
        return total_reward


class ComputeAdvantages(ForgeActor):
    """Compute advantages for GRPO using reward signals."""

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        super().__init__()
        # self.gamma = gamma  # Discount factor
        # self.lambda_ = lambda_  # GAE lambda parameter

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        # TODO: add batch processing
        rewards = torch.Tensor([[e.reward for e in group.episodes]])
        advantages = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True) + 1e-4
        )

        # # Extract rewards from groups
        # rewards = [group.reward for group in groups]
        # num_groups = len(groups)
        #
        # # For simplicity, use reward-to-go as advantages
        # # This is a valid advantage estimator: A(s,a) = Q(s,a) - V(s)
        # # where Q(s,a) ≈ reward-to-go and V(s) ≈ average reward
        #
        # # Compute discounted reward-to-go for each step
        # reward_to_go = []
        # running_reward = 0.0
        #
        # # Calculate discounted returns (reward-to-go)
        # for t in reversed(range(num_groups)):
        #     running_reward = rewards[t] + self.gamma * running_reward
        #     reward_to_go.insert(0, running_reward)
        #
        # # Compute baseline (mean of rewards) and advantages
        # baseline = sum(rewards) / len(rewards) if rewards else 0.0
        # advantages = [rtg - baseline for rtg in reward_to_go]
        #
        # # Normalize advantages to have zero mean and unit variance
        # if len(advantages) > 1:
        #     mean_adv = sum(advantages) / len(advantages)
        #     var_adv = sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)
        #     std_adv = (var_adv**0.5) if var_adv > 1e-8 else 1.0
        #     advantages = [(a - mean_adv) / std_adv for a in advantages]

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
    async def forward(self, request: list[int], response: list[int]) -> torch.Tensor:

        # Convert tokens to tensor
        input_ids = torch.tensor(
            request + response, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # Compute logits
        with torch.inference():
            logits = model(input_ids=input_ids).logits

        # Compute logprobs
        input_ids = input_ids[:, len(response) :]
        logprobs = compute_logprobs(logits, input_ids)

        return logprobs


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str
    name: str
    split: str
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
            self.path, self.name, split=self.split, streaming=self.streaming
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


async def main():
    """Main GRPO training loop with rollout and training processes."""
    group_size = 1
    model = "Qwen/Qwen3-1.7B"

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
            sampling_params=SamplingOverrides(n=group_size, max_tokens=16),
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
        split="train",
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
                target=target,
            )
            responses = await policy.generate.choose(prompt)
            for episode, response in zip(group.episodes, responses.outputs):
                episode.request_tokens = responses.prompt_token_ids
                episode.response_tokens = response.token_ids
                episode.ref_logprobs = await ref_model.forward.choose(
                    request=episode.request_tokens, response=episode.response_tokens
                )
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
