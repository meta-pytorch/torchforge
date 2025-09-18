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
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint
from omegaconf import DictConfig

from torch.utils.data import IterableDataset
from torchstore.state_dict_utils import DELIM
from transformers import AutoModelForCausalLM
from vllm.transformers_utils.tokenizer import get_tokenizer


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
    response_logprobs: torch.Tensor | None = None

    @property
    def request_tensor(self, slice_obj=None):
        """Returns the full request tensor"""
        return self.get_request_tensor()

    @property
    def response_tensor(self):
        """Returns the full response tensor"""
        return self.get_response_tensor()


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

        self.loss = ReinforceLoss()
        self._tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

        self.logger.info(f"Trainer model initialized on {self.device}")

    @endpoint
    async def train_step(self, batch: list[list[Episode]]):
        microbatch = batch[self.dp_rank]
        pad_id = microbatch[0].pad_id
        max_seq_len = microbatch[0].request_len + microbatch[0].response_len - 1

        # Process all microbatches
        batch_input_ids = []
        batch_target_ids = []
        batch_loss_masks = []
        batch_weights = []
        batch_old_log_probs = []
        for microbatch_item in microbatch:
            prompt_ids = torch.LongTensor(microbatch_item.request_tokens)
            token_ids = torch.LongTensor(microbatch_item.response_tokens)
            ids = torch.cat([prompt_ids, token_ids])

            input_ids = ids[:-1]  # truncate EOS
            diff = max_seq_len - input_ids.size(0)
            input_ids = F.pad(input_ids, (0, diff), value=pad_id)

            target_ids = ids[1:]  # truncate BOS
            diff = max_seq_len - target_ids.size(0)
            target_ids = F.pad(target_ids, (0, diff), value=pad_id)

            # Simple loss mask: 0 for prompt, 1 for response
            loss_mask = torch.cat(
                [
                    torch.zeros(
                        len(prompt_ids), dtype=torch.float32
                    ),  # Don't compute loss on prompt
                    torch.ones(
                        len(token_ids), dtype=torch.float32
                    ),  # Compute loss on response
                ]
            )

            # Get advantage and create weights
            advantage = microbatch_item.reward  # Single float value
            weights = loss_mask * advantage  # [B]

            # log probs
            old_log_probs = torch.cat(
                [
                    torch.zeros(prompt_ids.shape, dtype=torch.float32),
                    microbatch_item.response_logprobs,
                ]
            )

            # Handle shifting (target_ids = ids[1:])
            loss_mask = loss_mask[1:]  # Shift to align with target_ids
            weights_shifted = weights[1:]  # Shift weights
            old_log_probs_shifted = old_log_probs[1:]  # Shift log probs

            # Pad the loss mask
            diff = max_seq_len - loss_mask.size(0)
            loss_mask = F.pad(loss_mask, (0, diff), value=0.0)  # Pad with 0.0
            weights_shifted = F.pad(weights_shifted, (0, diff), value=0.0)
            old_log_probs_shifted = F.pad(old_log_probs_shifted, (0, diff), value=0.0)

            # Exclude padded response tokens from loss
            valid_mask = target_ids != pad_id
            loss_mask = loss_mask * valid_mask.float()
            weights_shifted = weights_shifted * valid_mask.float()
            old_log_probs_shifted = old_log_probs_shifted * valid_mask.float()

            batch_input_ids.append(input_ids)
            batch_target_ids.append(target_ids)
            batch_loss_masks.append(loss_mask)
            batch_weights.append(weights_shifted)
            batch_old_log_probs.append(old_log_probs_shifted)

        # Stack into batched tensors
        input_ids = torch.stack(batch_input_ids).to(self.device)
        target_ids = torch.stack(batch_target_ids).to(self.device)
        loss_masks = torch.stack(batch_loss_masks).to(self.device)
        weights = torch.stack(batch_weights).to(self.device)
        old_log_probs = torch.stack(batch_old_log_probs).to(self.device)

        # Create attention mask
        attention_mask = input_ids != pad_id

        # Forward pass
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Compute loss only on response tokens
        loss = self.loss(logits, target_ids, loss_masks, weights, old_log_probs)
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
    # import debugpy

    # debugpy.listen(5678)
    # print("Waiting for VS Code debugger to attach...")
    # debugpy.wait_for_client()
    # print("Attached!")

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
            for episode in group.episodes:
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
