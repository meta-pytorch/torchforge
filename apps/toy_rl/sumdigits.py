# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchstore as ts
from forge.actors.policy import Policy
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import _qwen3_hf_to_vllm
from forge.cli.config import parse
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import shutdown

from forge.losses.reinforce_loss import ReinforceLoss
from forge.types import Episode, Group
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint
from omegaconf import DictConfig

from torch.utils.data import IterableDataset
from torchstore.state_dict_utils import DELIM
from transformers import AutoModelForCausalLM
from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class Trainer(ForgeActor):
    """Reinforce Loss Trainer implementation for policy optimization."""

    model_name: str
    learning_rate: float = 1e-5
    device: torch.device | None = None
    state_dict_key: str = "model_state_dict"

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
        self.loss = ReinforceLoss()
        self.logger.info(f"Trainer model initialized on {self.device}")

    @endpoint
    def train_step(self, episodes: list[Episode]) -> float:
        pad_id = episodes[0].pad_id

        # Calculate batch maximum length
        max_seq_len = max(ep.max_seq_len - 1 for ep in episodes)
        batch_input_ids = []
        batch_target_ids = []
        batch_loss_masks = []
        batch_weights = []
        batch_sampling_log_probs = []
        for episode in episodes:
            input_ids = self.pad_sequence(episode.input_ids, max_seq_len, pad_id)
            target_ids = self.pad_sequence(episode.target_ids, max_seq_len, pad_id)
            loss_mask = self.pad_sequence(episode.loss_mask, max_seq_len, 0.0)
            sampling_log_probs = self.pad_sequence(
                episode.sampling_log_probs, max_seq_len, 0.0
            )
            weights = self.pad_sequence(episode.weighted_advantages, max_seq_len, 0.0)

            # Exclude padded response tokens from loss
            valid_mask = target_ids != pad_id
            loss_mask = loss_mask * valid_mask.float()
            weights = weights * valid_mask.float()
            sampling_log_probs = sampling_log_probs * valid_mask.float()

            batch_input_ids.append(input_ids)
            batch_target_ids.append(target_ids)
            batch_loss_masks.append(loss_mask)
            batch_weights.append(weights)
            batch_sampling_log_probs.append(sampling_log_probs)

        # Stack into batched tensors
        input_ids = torch.stack(batch_input_ids).to(self.device)
        target_ids = torch.stack(batch_target_ids).to(self.device)
        loss_masks = torch.stack(batch_loss_masks).to(self.device)
        weights = torch.stack(batch_weights).to(self.device)
        sampling_log_probs = torch.stack(batch_sampling_log_probs).to(self.device)

        # Create attention mask
        attention_mask = input_ids != pad_id

        # Forward pass
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Compute loss only on response tokens
        loss = self.loss(logits, target_ids, loss_masks, weights, sampling_log_probs)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

    def pad_sequence(
        self, tensor: torch.Tensor, target_len: int, pad_value: float = 0.0
    ) -> torch.Tensor:
        diff = target_len - tensor.size(0)
        if diff > 0:
            return F.pad(tensor, (0, diff), value=pad_value)
        return tensor


@dataclass
class RewardActor(ForgeActor):
    """Reward actor that uses a list of scoring functions."""

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        if response == target:
            return 1.0
        return 0.0


@dataclass
class SumDigitsDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=1000):
        self.min_digit_length = 2
        self.max_digit_length = 3
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

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"

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
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.services.dataset).as_service(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        Trainer.options(**cfg.services.trainer).as_service(**cfg.trainer),
        ReplayBuffer.options(**cfg.services.replay_buffer).as_service(
            **cfg.replay_buffer
        ),
        RewardActor.options(**cfg.services.reward_actor).as_service(),
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
                episode.response_logprobs = torch.tensor(
                    [
                        top_k_dict[token].logprob
                        for token, top_k_dict in zip(
                            response.token_ids,
                            response.logprobs,
                        )
                    ]
                )
                episode.reward = await reward_actor.evaluate_response.choose(
                    prompt=prompt, response=response.text, target=target
                )
                episode.advantage = episode.reward  # simple case for now
            for episode in group.episodes:
                await replay_buffer.add.choose(episode)

            avg_response_len = (
                sum(len(e.response_tokens) for e in group.episodes) / group_size
            )
            mlogger.log("avg_response_len/rollout", avg_response_len, rollout_count)
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
                loss = await trainer.train_step.choose(batch[0])
                training_step += 1
                mlogger.log("loss/training_step", loss, training_step)
                print(f"loss/training_step: {loss} at {training_step}")
                if training_step % 5 == 0:
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
