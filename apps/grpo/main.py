import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import torch

from datasets import load_dataset
from forge.actors.policy import Policy, PolicyRouter
from forge.controller import ServiceConfig, spawn_service
from forge.data.replay_buffer import ReplayBuffer
from monarch.actor import Actor, endpoint
from vllm import CompletionOutput, SamplingParams


@dataclass
class Group:
    response: str
    current_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    reward: float
    advantage: float = 0.0


class Episode:
    """Episode container for GRPO rollouts."""

    def __init__(self):
        self.groups: list[Group] = []

    def add_group(self, group: Group):
        self.groups.append(group)


class Trainer(Actor):
    """GRPO Trainer implementation for policy optimization."""

    def __init__(self, learning_rate: float = 1e-5, beta: float = 0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta = beta  # KL penalty coefficient
        self.model = None  # Will be set during initialization
        self.optimizer = None

    def _initialize_model_and_optimizer(self, model):
        """Initialize model and optimizer - called when model is available."""
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

    @endpoint
    async def train_step(self, batch: list[Episode]):
        total_loss = 0.0
        num_groups_processed = 0

        for episode in batch:
            groups = episode.groups

            current_logprobs_list = []
            ref_logprobs_list = []
            advantages_list = []

            for group in groups:
                current_logprobs_list.append(group.current_logprobs)
                ref_logprobs_list.append(group.ref_logprobs)
                advantages_list.append(group.advantage)

            if not current_logprobs_list:
                continue

            # Convert to tensors
            current_logprobs_tensor = torch.stack(current_logprobs_list)
            ref_logprobs_tensor = torch.stack(ref_logprobs_list)
            advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32)

            # Ensure current_logprobs requires gradients for training
            current_logprobs_tensor.requires_grad_(True)

            # Compute GRPO loss components
            # Ratio between current policy and reference policy
            ratio = torch.exp(current_logprobs_tensor - ref_logprobs_tensor)

            # Policy gradient loss weighted by advantages
            pg_loss = -torch.mean(ratio * advantages_tensor)

            # KL penalty to prevent policy from deviating too far from reference
            kl_penalty = self.beta * torch.mean(
                (current_logprobs_tensor - ref_logprobs_tensor) ** 2
            )

            # Total GRPO loss
            loss = pg_loss + kl_penalty
            total_loss += loss.item()
            num_groups_processed += len(groups)

            # Backward pass and optimization
            if loss.requires_grad and self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        avg_loss = total_loss / len(batch) if batch else 0.0
        print(
            f"Training step completed. Average loss: {avg_loss:.4f}, Groups processed: {num_groups_processed}"
        )
        return {"loss": avg_loss, "groups_processed": num_groups_processed}

    @endpoint
    async def update_weights(self):
        """Update policy weights - placeholder for policy distribution."""
        # In a distributed setting, this would sync weights across workers
        print("Policy weights updated")
        return {"status": "weights_updated"}


def math_scoring_function(prompt: str, response: str, target: str) -> float:
    """Function to score math correctness."""
    import re

    # Extract expected answer from target
    expected_answer = (
        float(target.strip())
        if target.strip().replace(".", "").replace("-", "").isdigit()
        else None
    )

    # Extract model answer from response
    patterns = [
        r"####\s*([+-]?\d+(?:\.\d+)?)",  # GSM8K style answer format
        r"(?:the\s+)?answer\s+is\s*([+-]?\d+(?:\.\d+)?)",
        r"(?:answer:|result:)\s*([+-]?\d+(?:\.\d+)?)",
        r"=\s*([+-]?\d+(?:\.\d+)?)\s*(?:\.|$)",  # equals near end
        r"\b([+-]?\d+(?:\.\d+)?)\s*(?:\.|$)",  # number at end
        r"([+-]?\d+(?:\.\d+)?)",  # any number (fallback)
    ]

    model_answer = None
    response_lower = response.lower().strip()
    for pattern in patterns:
        matches = re.findall(pattern, response_lower)
        if matches:
            model_answer = float(matches[-1])
            break

    if expected_answer is None or model_answer is None:
        return 0.1  # Partial credit for attempting

    # Check if answers match (with some tolerance for floating point)
    if abs(expected_answer - model_answer) < 1e-6:
        return 1.0  # Correct answer
    else:
        return 0.0  # Incorrect answer


def thinking_scoring_function(prompt: str, response: str, target: str) -> float:
    """Function to score thinking tag usage."""
    # Check if response contains <think></think> tags
    if "<think>" in response.lower() and "</think>" in response.lower():
        return 0.5
    else:
        return 0.0


class RewardActor(Actor):
    """Reward actor that uses a list of scoring functions."""

    def __init__(self, scoring_functions: list[Callable]):
        super().__init__()
        self.scoring_functions = scoring_functions

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        total_reward = 0.0
        for scoring_fn in self.scoring_functions:
            reward = scoring_fn(prompt, response, target)
            total_reward += reward
        return total_reward


class ComputeAdvantages(Actor):
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


class RefModel(Actor):
    def __init__(self):
        super().__init__()
        self.model = None

    @endpoint
    async def forward(self, tokens: list[int]) -> torch.Tensor:
        # Placeholder implementation - in practice would use actual reference model
        # to compute log probabilities for the tokens
        logprob_value = -torch.log(torch.tensor(float(len(tokens))))
        logprobs = torch.full((len(tokens),), logprob_value.item(), dtype=torch.float32)
        return logprobs


class DatasetActor(Actor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    def __init__(self, **kwargs):
        super().__init__()
        self._setup_dataset(**kwargs)

    def _setup_dataset(self, **kwargs):
        def gsm8k_to_messages(sample):
            question = sample["question"]
            full_answer: str = sample["answer"]
            answer = full_answer.split("####")[1]
            return (question, answer)

        ds = load_dataset(**kwargs)
        ds.map(gsm8k_to_messages)
        ds.shuffle()
        self._iterator = ds

    @endpoint
    async def __next__(self) -> tuple[str, str] | None:
        try:
            sample = next(self._iterator)
            return sample  # Return full sample with question and answer
        except StopIteration:
            return None


async def main():
    """Main GRPO training loop with rollout and training processes."""
    group_size = 5

    # ---- Setup services ---- #
    default_service_cfg = ServiceConfig(
        procs_per_replica=1,
        min_replicas=1,
        max_replicas=1,
        default_replicas=1,
    )
    policy = await spawn_service(
        default_service_cfg,
        PolicyRouter,
        policy=Policy(model="Deepseek/Deepseek-v3"),
        sampling_params=SamplingParams(n=group_size),
    )

    trainer = await spawn_service(
        default_service_cfg,
        Trainer,
        learning_rate=1e-5,
        beta=0.1,
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
        path="openai/gsm8k",
        split="train",
        streaming=True,
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
        model_name="reference_model",
    )

    reward_actor = await spawn_service(
        default_service_cfg,
        RewardActor,
        scoring_functions=[math_scoring_function, thinking_scoring_function],
    )

    print("All services initialized successfully")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        while True:
            sample = await dataloader.__next__.call()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            prompt, target = sample
            version = await policy.get_current_version.choose()
            episode = Episode()

            async with policy.session(version=version):
                actions: list[CompletionOutput] = await policy.generate.call(prompt)

            for action in actions:
                current_logprobs = await policy.compute_logprobs.call(action.tokens)
                ref_logprobs = await ref_model.forward.call(action.tokens)
                reward = await reward_actor.evaluate_response.call(
                    prompt, action.text, target
                )
                episode.add_group(
                    Group(action.text, current_logprobs, ref_logprobs, reward)
                )

            advantages = await compute_advantages.__call__.call(episode.groups)
            for advantage, group in zip(advantages, episode.groups):
                group.advantage = advantage

            await replay_buffer.add.call(episode)

            rollout_count += 1
            if rollout_count % 10 == 0:
                print(f"Generated {rollout_count} rollouts")

    async def continuous_training():
        training_step = 0
        while True:
            batch = await replay_buffer.sample.call()
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                training_result = await trainer.train_step.call(batch)
                training_step += 1
                if training_step % 10 == 0:
                    print(f"Completed {training_step} training steps")
                    if training_result:
                        print(f"Latest loss: {training_result.get('loss', 'N/A')}")
                await trainer.update_weights.call()

    print("Starting GRPO training loops...")
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
