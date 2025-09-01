# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import safetensors.torch as safetensors
import torch

from absl import app, flags
from datasets import load_dataset
from forge.actors.policy import Policy, PolicyConfig, SamplingOverrides, WorkerConfig
from forge.actors.replay_buffer import ReplayBuffer
from forge.controller.actor import ForgeActor
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from forge.data.rewards import MathReward, ThinkingReward
from forge.data.weights_handle import WeightsHandle, WeightsHandleType
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    apply_mask_to_padding_states,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 8, "")
flags.DEFINE_integer("update_period", 5, "")
flags.DEFINE_integer("group_size", 16, "")
flags.DEFINE_integer("max_policy_age", 10, "")


def clean_up_temp_dir(temp_dir: str) -> None:
    """Clean up temporary directory."""
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def compute_sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    requires_grad: bool = True,
) -> torch.Tensor:
    context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

    with context_manager:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probabilities for the actual tokens (excluding the first token for next-token prediction)
        shifted_input_ids = input_ids[:, 1:]  # Remove first token
        shifted_log_probs = log_probs[:, :-1, :]  # Remove last logit

        # Gather log probabilities for actual tokens
        token_log_probs = torch.gather(
            shifted_log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probabilities across sequence (masked by attention)
        shifted_attention_mask = attention_mask[:, 1:]
        sequence_log_probs = (token_log_probs * shifted_attention_mask).sum(dim=-1)

        return sequence_log_probs


@dataclass
class Group:
    response: str  # The response text for tokenization
    ref_logprobs: torch.Tensor
    reward: float
    advantage: float = 0.0


class Episode:
    """Episode container for GRPO rollouts."""

    def __init__(self, episode_id: int, prompt: str, target: str, policy_version: int):
        self.episode_id = episode_id
        self.prompt = prompt
        self.target = target
        self.policy_version = policy_version
        self.groups: list[Group] = []

    def add_group(self, group: Group):
        self.groups.append(group)

    def add_groups(self, groups: list[Group]):
        self.groups.extend(groups)


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

        for episode in batch:
            groups = episode.groups

            # Collect all response texts and corresponding data
            response_texts = []
            ref_logprobs_list = []
            advantages_list = []

            for group in groups:
                response_texts.append(group.response)
                ref_logprobs_list.append(group.ref_logprobs)
                advantages_list.append(group.advantage)

            # Tokenize all responses in batch
            tokenized = self.tokenizer(
                response_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,  # Adjust based on your needs
            )

            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            # Compute current policy log probabilities using the model
            current_logprobs = compute_sequence_logprobs(
                self.model, input_ids, attention_mask, requires_grad=True
            )

            # Convert ref_logprobs and advantages to tensors
            ref_logprobs_tensor = torch.stack(ref_logprobs_list).to(self.device)
            advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32).to(
                self.device
            )

            # Compute GRPO loss components
            # Ratio between current policy and reference policy
            ratio = torch.exp(current_logprobs - ref_logprobs_tensor)

            # Policy gradient loss weighted by advantages
            pg_loss = -torch.mean(ratio * advantages_tensor)

            # KL penalty to prevent policy from deviating too far from reference
            kl_penalty = self.beta * torch.mean(
                (current_logprobs - ref_logprobs_tensor) ** 2
            )

            # Total GRPO loss
            loss = pg_loss + kl_penalty
            total_loss += loss.item()
            num_groups_processed += len(groups)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional but recommended for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        avg_loss = total_loss / len(batch) if batch else 0.0

        return {"loss": avg_loss, "groups_processed": num_groups_processed}

    @endpoint
    async def export_weights(self, step: int) -> WeightsHandle:
        """Export weights to a temp file and return the handle."""
        # Time how long it takes to update weights
        start_time = time.time()

        # Set model to eval mode for weight extraction
        self.model.eval()

        # Save weights to a memory-backed temporary directory using /dev/shm
        import os

        shm_path = "/dev/shm"
        if os.path.exists(shm_path):
            # Use shared memory filesystem for memory-backed storage
            temp_dir = tempfile.mkdtemp(
                prefix=f"model_weights_step_{step:08d}_", dir=shm_path
            )
        else:
            # Fallback to system temp if /dev/shm not available
            temp_dir = tempfile.mkdtemp(prefix=f"model_weights_step_{step:08d}_")

        try:
            # Save weights directly to SafeTensors file
            weights_file = os.path.join(temp_dir, "model_weights.safetensors")
            state_dict = {name: param for name, param in self.model.named_parameters()}
            safetensors.save_file(state_dict, weights_file)

            # Create weights handle with the SafeTensors file path
            param_names = list(state_dict.keys())
            weights_handle = WeightsHandle(
                handle_type=WeightsHandleType.FILE,
                version=step,
                payload={
                    "param_names": param_names,
                    "model_path": weights_file,
                    "model_name": self.model_name,
                },
            )

        except Exception as e:
            # Clean up temporary directory if something goes wrong
            clean_up_temp_dir(temp_dir)
            raise e

        # Set model back to training mode
        self.model.train()

        # Log the time taken
        end_time = time.time()
        self.logger.info(f"Updating weights took {end_time - start_time:.2f} seconds")
        return weights_handle


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

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        super().__init__()
        self.gamma = gamma  # Discount factor
        self.lambda_ = lambda_  # GAE lambda parameter

    @endpoint
    async def __call__(self, groups: list[Group]) -> list[float]:
        # Extract rewards from groups
        rewards = [group.reward for group in groups]
        num_groups = len(groups)

        # For simplicity, use reward-to-go as advantages
        # This is a valid advantage estimator: A(s,a) = Q(s,a) - V(s)
        # where Q(s,a) ≈ reward-to-go and V(s) ≈ average reward

        # Compute discounted reward-to-go for each step
        reward_to_go = []
        running_reward = 0.0

        # Calculate discounted returns (reward-to-go)
        for t in reversed(range(num_groups)):
            running_reward = rewards[t] + self.gamma * running_reward
            reward_to_go.insert(0, running_reward)

        # Compute baseline (mean of rewards) and advantages
        baseline = sum(rewards) / len(rewards) if rewards else 0.0
        advantages = [rtg - baseline for rtg in reward_to_go]

        # Normalize advantages to have zero mean and unit variance
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            var_adv = sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)
            std_adv = (var_adv**0.5) if var_adv > 1e-8 else 1.0
            advantages = [(a - mean_adv) / std_adv for a in advantages]

        return advantages


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
    async def forward(self, token_ids: list[int]) -> torch.Tensor:
        # Use provided token_ids directly
        input_ids = (
            torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        # Create attention mask of all 1s since we have actual tokens (no padding)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        # Compute log probabilities using shared utility function
        sequence_log_probs = compute_sequence_logprobs(
            self.model, input_ids, attention_mask, requires_grad=False
        )

        return (
            sequence_log_probs.squeeze()
        )  # Remove batch dimension for single response


class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    def __init__(
        self, path: str, config_name: str, split: str, streaming: bool, **kwargs
    ):
        super().__init__()

        def gsm8k_to_messages(sample):
            question = sample["question"]
            full_answer: str = sample["answer"]
            answer = full_answer.split("#### ")[1]
            return {"question": question, "answer": answer}

        ds = load_dataset(path, config_name, split=split, streaming=streaming)
        ds = ds.map(gsm8k_to_messages)
        ds = ds.shuffle()
        self._iterator = iter(ds)

    @endpoint
    async def __next__(self) -> dict[str, str] | None:
        try:
            return next(self._iterator)
        except StopIteration:
            return None


async def _main():
    """Main GRPO training loop with rollout and training processes."""
    group_size = 16
    model = "Qwen/Qwen3-1.7B"

    # ---- Setup WandB Logger ---- #
    logger = get_metric_logger(
        "wandb",
        freq=1,
        project="grpo-training",
    )

    # ---- Setup services ---- #
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
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            DatasetActor,
            path="openai/gsm8k",
            config_name="main",
            split="train",
            streaming=True,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=1),
            Policy,
            config=PolicyConfig(
                worker_params=WorkerConfig(model=model),
                sampling_params=SamplingOverrides(
                    num_samples=group_size, max_tokens=16
                ),
            ),
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=1),
            Trainer,
            learning_rate=1e-5,
            beta=0.1,
            model_name=model,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            ReplayBuffer,
            batch_size=4,
            max_policy_age=1,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            ComputeAdvantages,
            gamma=0.99,
            lambda_=0.95,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1, with_gpus=True),
            RefModel,
            model_name=model,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            RewardActor,
            reward_functions=[MathReward(), ThinkingReward()],
        ),
    )
    print("All services initialized successfully!")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        while True:
            sample = await dataloader.__next__.choose()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return
            prompt, target = sample["question"], sample["answer"]

            response = await policy.generate.choose(prompt)
            actions = response.completions
            version = response.policy_version
            episode = Episode(
                episode_id=rollout_count,
                prompt=prompt,
                target=target,
                policy_version=version,
            )

            async def _get_group(action):
                ref_logprobs, reward = await asyncio.gather(
                    ref_model.forward.choose(action.token_ids),
                    reward_actor.evaluate_response.choose(
                        prompt=prompt, response=action.text, target=target
                    ),
                )
                return Group(
                    response=action.text, ref_logprobs=ref_logprobs, reward=reward
                )

            groups = await asyncio.gather(*[_get_group(action) for action in actions])

            episode.add_groups(groups)

            advantages = await compute_advantages.__call__.choose(episode.groups)
            for advantage, group in zip(advantages, episode.groups):
                group.advantage = advantage

            await replay_buffer.add.choose(episode)

            rollout_count += 1
            rewards = []
            if rollout_count % 10 == 0:
                episode_avg_reward = sum(
                    group.reward for group in episode.groups
                ) / len(episode.groups)
                rewards.append(episode_avg_reward)
                avg_reward = sum(rewards) / len(rewards)
                rewards.clear()
                print(
                    f"Generated {rollout_count} rollouts, average reward of last 10 = {avg_reward}"
                )
                logger.log("reward/rollout", avg_reward, rollout_count)

    async def continuous_training():
        training_step = 0
        update_period = FLAGS.update_period
        # using training_step as the policy version for now, open to suggestions
        while True:
            batch = await replay_buffer.sample.choose(curr_policy_version=training_step)
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                # why is call_one not defined?
                # training_result = await trainer.train_step.call_one(batch)
                training_result = await trainer.train_step.choose(batch)
                training_step += 1
                print(f"Completed {training_step} training steps")
                if training_result:
                    loss_value = training_result.get("loss", 0.0)
                    print(f"Latest loss: {loss_value}")
                    logger.log("loss/training_step", loss_value, training_step)
                if training_step % update_period == 0:
                    print(f"Exporting policy weights @ {training_step=}")
                    weights_handle = await trainer.export_weights.choose(training_step)
                    print(f"Exported weights @ {training_step=}")
                    await policy.update_weights.call(weights_handle)
                    print(f"Updated policy weights to version @ {training_step=}")
                    clean_up_temp_dir(weights_handle.payload["model_path"])

    print("Starting GRPO training loops...")

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
        )


def main(argv):
    asyncio.run(_main())


if __name__ == "__main__":
    app.run(main)
