# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import random
import time
from dataclasses import dataclass

import torch
import torchstore as ts
from forge.actors.policy import Policy
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import _qwen3_hf_to_vllm
from forge.cli.config import parse
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import shutdown
from forge.data_models.episode import Episode, from_completion
from forge.losses.grpo_loss import SimpleGRPOLoss
from forge.util.metric_logging import get_metric_logger

from forge.util.ops import pad_sequence, selective_log_softmax
from monarch.actor import endpoint
from omegaconf import DictConfig

from torchstore.state_dict_utils import DELIM
from transformers import AutoModelForCausalLM
from vllm.transformers_utils.tokenizer import get_tokenizer


def to_batch(episodes: list[Episode], device: torch.device):
    batch = {}
    pad_id = episodes[0].pad_id
    max_seq_len = max(ep.max_seq_len - 1 for ep in episodes)

    batch_input_ids = []
    batch_target_ids = []
    batch_loss_masks = []
    batch_weights = []
    batch_ref_logprobs = []
    for episode in episodes:
        input_ids = pad_sequence(episode.input_ids, max_seq_len, pad_id)
        target_ids = pad_sequence(episode.target_ids, max_seq_len, pad_id)
        loss_mask = pad_sequence(episode.loss_mask, max_seq_len, 0.0)
        weights = pad_sequence(episode.weighted_advantages, max_seq_len, 0.0)
        ref_logprobs = episode.ref_logprobs

        # Exclude padded response tokens from loss
        valid_mask = target_ids != pad_id
        loss_mask = loss_mask * valid_mask.float()
        weights = weights * valid_mask.float()

        batch_input_ids.append(input_ids)
        batch_target_ids.append(target_ids)
        batch_loss_masks.append(loss_mask)
        batch_weights.append(weights)
        batch_ref_logprobs.append(ref_logprobs)

    # Stack into batched tensors
    batch["input_ids"] = torch.stack(batch_input_ids).to(device)
    batch["target_ids"] = torch.stack(batch_target_ids).to(device)
    batch["loss_masks"] = torch.stack(batch_loss_masks).to(device)
    batch["weights"] = torch.stack(batch_weights).to(device)
    batch["ref_logprobs"] = torch.stack(batch_ref_logprobs).to(device)

    return batch


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
        input_ids = (
            pad_sequence(episode.input_ids, episode.max_seq_len - 1, episode.pad_id)
            .to(self.device)
            .unsqueeze(0)
        )
        target_ids = (
            pad_sequence(episode.target_ids, episode.max_seq_len - 1, episode.pad_id)
            .to(self.device)
            .unsqueeze(0)
        )
        mask = input_ids != episode.pad_id

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits

        return selective_log_softmax(logits, target_ids).squeeze(0)


@dataclass
class Trainer(ForgeActor):
    """Reinforce Loss Trainer implementation for policy optimization."""

    model_name: str
    learning_rate: float = 1e-5
    device: torch.device | None = None
    state_dict_key: str = "model_state_dict"

    @endpoint
    async def setup(self):
        import debugpy

        debugpy.listen(5681)
        print("[LEARNER] Waiting for VS Code debugger to attach...")
        debugpy.wait_for_client()
        print("Attached!")
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

        # beta = 0.01 for quicker convergence
        self.loss = SimpleGRPOLoss(0.01)
        self.logger.info(f"Trainer model initialized on {self.device}")

    @endpoint
    def train_step(self, episodes: list[Episode]) -> float:
        pad_id = episodes[0].pad_id
        batch = to_batch(episodes, self.device)

        # Create attention mask
        attention_mask = batch["input_ids"] != pad_id

        # Forward pass
        logits = self.model(
            input_ids=batch["input_ids"], attention_mask=attention_mask
        ).logits

        trainer_log_probs = selective_log_softmax(logits, batch["target_ids"])

        # Compute loss only on response tokens
        loss = self.loss(
            trainer_log_probs,
            batch["ref_logprobs"],
            batch["weights"],
            batch["loss_masks"],
        )
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
    async def evaluate_response(self, response: str, target: str) -> float:
        reward = 1.0 if response.strip() == target else 0.0
        return reward


@dataclass
class SumDigitsDataset:
    def __init__(self, tokenizer, max_samples=1000):
        self.max_numbers = max_samples
        self._tokenizer = tokenizer

    def generate_sample(self, step: int) -> dict[str, str]:
        """Generate a single sample based on training step for progressive difficulty."""
        data = self.generate_one(step)
        answer = str(sum(int(x) for x in data))

        system_prompt = """
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
        The assistant only gives very concise answers (just the number, no explanation).
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
        return {
            "question": formatted_request,
            "request": formatted_request,
            "answer": answer,
            "target": answer,
        }

    def generate_one(self, step: int) -> str:
        """Generate number based on training step for curriculum learning."""
        min_val, max_val = 10, 100

        number = random.randint(min_val, max_val)
        return str(number)


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        self._dataset = SumDigitsDataset(self._tokenizer)

    @endpoint
    async def sample(self, step: int = 0) -> dict[str, str] | None:
        """Sample with progressive difficulty based on training step."""
        try:
            return self._dataset.generate_sample(step)
        except Exception as e:
            self.logger.error(f"Error generating sample: {e}")
            return None

    @endpoint
    async def pad_token(self):
        return self._tokenizer.pad_token_id


async def main(cfg: DictConfig):
    """Main Sumgits app training loop with rollout and training processes."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
    max_seq_len = cfg.max_seq_len
    mlogger = get_metric_logger(
        "wandb",
        freq=1,
        project="sumdigits-training",
    )

    # ---- Setup services ---- #
    await ts.initialize()
    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        reward_actor,
        ref_model,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.services.dataset).as_service(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        Trainer.options(**cfg.services.trainer).as_service(**cfg.trainer),
        ReplayBuffer.options(**cfg.services.replay_buffer).as_service(
            **cfg.replay_buffer
        ),
        RewardActor.options(**cfg.services.reward_actor).as_service(),
        RefModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
    )

    print("All services initialized successfully!")

    import debugpy

    debugpy.listen(5678)
    print("[MAIN] Waiting for VS Code debugger to attach...")
    debugpy.wait_for_client()
    print("Attached!")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await dataloader.pad_token.choose()
        while True:
            # Pass rollout_count for curriculum learning
            sample = await dataloader.sample.choose(rollout_count)
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return
            prompt, target = sample["request"], sample["target"]
            completions = await policy.generate.choose(prompt)

            episodes = []
            version = await policy.get_version.choose()
            for completion in completions:
                # calculates reward
                reward = await reward_actor.evaluate_response.choose(
                    response=completion.text, target=target
                )

                episode = from_completion(
                    completion,
                    policy_verson=version,
                    pad_id=pad_id,
                    max_seq_len=max_seq_len,
                    reward=reward,
                )

                # calculates ref log probs
                # TODO: make a batched forward instead of forward per episode
                episode.ref_logprobs = await ref_model.forward.choose(episode)
                episodes.append(episode)

            for episode in episodes:
                await replay_buffer.add.choose(episode)
            sample_size = len(episodes)
            avg_response_len = sum(len(e.token_ids) for e in completions) / sample_size
            mlogger.log("avg_response_len/rollout", avg_response_len, rollout_count)
            avg_reward = sum(e.reward for e in episodes) / sample_size
            mlogger.log("avg_reward/rollout", avg_reward, rollout_count)

            rollout_count += 1

    async def continuous_training():
        training_step = 0
        while True:
            batch = await replay_buffer.sample.choose(curr_policy_version=training_step)
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                loss = await trainer.train_step.choose(batch[0])
                training_step += 1
                mlogger.log("loss/training_step", loss, training_step)
                print(f"loss/training_step: {loss} at training step {training_step}")
                if training_step % 100 == 0:
                    await trainer.push_weights.call(training_step)
                    await policy.update_weights.call(training_step)
                    # NOTE: hard-coded to be on-policy for faster convergence
                    await replay_buffer.clear.call()

    print("Starting training loop.")
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
