import asyncio
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from forge.actors.policy import Policy, PolicyRouter
from forge.controller import ServiceConfig, spawn_service
from forge.data.datasets.hf_dataset import HfIterableDataset
from forge.data.replay_buffer import ReplayBuffer
from monarch.actor import Actor, endpoint


class Trainer(Actor):
    """GRPO Trainer implementation for policy optimization."""

    def __init__(self, learning_rate: float = 1e-5, beta: float = 0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta = beta  # KL penalty coefficient
        self.policy_model = None  # Will be set during initialization
        self.optimizer = None

    def _initialize_model_and_optimizer(self, model):
        """Initialize model and optimizer - called when model is available."""
        self.policy_model = model
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.learning_rate
        )

    @endpoint
    async def train_step(self, batch: List[Dict[str, Any]]):
        """
        Perform a single GRPO training step.

        Args:
            batch: List of episodes with advantages, current_logprobs, ref_logprobs, etc.
        """
        if self.policy_model is None:
            print("Warning: Policy model not initialized, skipping training step")
            return

        total_loss = 0.0

        for episode in batch:
            # Extract episode data
            advantages = episode.get("advantages", [])
            current_logprobs = episode.get("current_logprobs", [])
            ref_logprobs = episode.get("ref_logprobs", [])

            if not advantages or not current_logprobs or not ref_logprobs:
                continue

            # Convert to tensors
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
            current_logprobs_tensor = torch.tensor(
                current_logprobs, dtype=torch.float32, requires_grad=True
            )
            ref_logprobs_tensor = torch.tensor(ref_logprobs, dtype=torch.float32)

            # Compute GRPO loss components
            ratio = torch.exp(current_logprobs_tensor - ref_logprobs_tensor)

            # Policy gradient loss
            pg_loss = -torch.mean(ratio * advantages_tensor)

            # KL penalty (simplified)
            kl_penalty = self.beta * torch.mean(
                (current_logprobs_tensor - ref_logprobs_tensor) ** 2
            )

            # Total loss
            loss = pg_loss + kl_penalty
            total_loss += loss.item()

            # Backward pass
            if loss.requires_grad:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print(f"Training step completed. Average loss: {total_loss / len(batch):.4f}")
        return {"loss": total_loss / len(batch)}

    @endpoint
    async def update_weights(self):
        """Update policy weights - placeholder for policy distribution."""
        if self.policy_model is None:
            print("Warning: Policy model not initialized")
            return

        # In a distributed setting, this would sync weights across workers
        print("Policy weights updated")
        return {"status": "weights_updated"}


class Episode:
    """Episode container for GRPO rollouts."""

    def __init__(self):
        self.turns = []
        self._metadata = {}

    def add_turn(self, turn):
        """Add a (prompt, action) turn to the episode."""
        self.turns.append(turn)

    def add_transform_info(self, key: str, data: Any):
        """Add metadata to the episode."""
        setattr(self, key, data)
        self._metadata[key] = data

    def get_tokens(self) -> List[int]:
        """Extract tokens from episode turns for model processing."""
        tokens = []
        for prompt, action in self.turns:
            # Simplified tokenization - in practice would use proper tokenizer
            if isinstance(prompt, str):
                tokens.extend([hash(char) % 50000 for char in prompt])
            if isinstance(action, str):
                tokens.extend([hash(char) % 50000 for char in action])
        return tokens

    def get(self, key: str, default=None):
        """Get metadata from episode."""
        return self._metadata.get(key, default)


class MathRewardActor(Actor):
    """Math reward model that evaluates correctness of math problem solutions."""

    def __init__(self):
        super().__init__()

    @endpoint
    async def evaluate_math_response(self, prompt: str, response: str) -> float:
        """
        Evaluate if a math response is correct.

        Args:
            prompt: The math problem question
            response: The generated response/solution

        Returns:
            Reward value (1.0 for correct, 0.0 for incorrect, partial scores possible)
        """
        try:
            # Extract the expected answer from simple math problems
            expected_answer = self._extract_answer_from_prompt(prompt)

            # Extract the model's answer from the response
            model_answer = self._extract_answer_from_response(response)

            if expected_answer is None or model_answer is None:
                # If we can't parse either, give partial credit for attempting
                return 0.1

            # Check if answers match (with some tolerance for floating point)
            if abs(expected_answer - model_answer) < 1e-6:
                return 1.0  # Correct answer
            else:
                return 0.0  # Incorrect answer

        except Exception as e:
            print(f"Error evaluating math response: {e}")
            return 0.1  # Small reward for attempting

    def _extract_answer_from_prompt(self, prompt: str) -> Optional[float]:
        """Extract expected answer from simple arithmetic problems."""
        import re

        prompt_lower = prompt.lower().strip()

        # Handle simple arithmetic patterns
        patterns = [
            r"what is (\d+\.?\d*) \+ (\d+\.?\d*)\?",  # addition
            r"(\d+\.?\d*) \+ (\d+\.?\d*) = \?",  # addition alt
            r"what is (\d+\.?\d*) \* (\d+\.?\d*)\?",  # multiplication
            r"(\d+\.?\d*) \* (\d+\.?\d*) = \?",  # multiplication alt
            r"what is (\d+\.?\d*) - (\d+\.?\d*)\?",  # subtraction
            r"calculate (\d+\.?\d*) - (\d+\.?\d*)",  # subtraction alt
            r"what is (\d+\.?\d*) / (\d+\.?\d*)\?",  # division
            r"(\d+\.?\d*) / (\d+\.?\d*) = \?",  # division alt
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                num1, num2 = float(match.group(1)), float(match.group(2))

                if "+" in pattern:
                    return num1 + num2
                elif "*" in pattern:
                    return num1 * num2
                elif "-" in pattern:
                    return num1 - num2
                elif "/" in pattern:
                    if num2 != 0:
                        return num1 / num2
                    else:
                        return None

        # Handle "find the value of X + Y" patterns
        value_pattern = r"find the value of (\d+\.?\d*) \+ (\d+\.?\d*)"
        match = re.search(value_pattern, prompt_lower)
        if match:
            return float(match.group(1)) + float(match.group(2))

        return None

    def _extract_answer_from_response(self, response: str) -> Optional[float]:
        """Extract numerical answer from model response."""
        import re

        # Look for numbers in the response, preferring those at the end
        # Common patterns: "The answer is 4", "= 4", "4.", "Answer: 4"
        patterns = [
            r"(?:answer is|equals?|=)\s*([+-]?\d+\.?\d*)",
            r"(?:answer:|result:)\s*([+-]?\d+\.?\d*)",
            r"\b([+-]?\d+\.?\d*)\s*(?:\.|$)",  # number at end
            r"([+-]?\d+\.?\d*)",  # any number (fallback)
        ]

        response_lower = response.lower().strip()

        for pattern in patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                try:
                    # Take the last match (often the final answer)
                    return float(matches[-1])
                except ValueError:
                    continue

        return None


class ComputeAdvantages(Actor):
    """Compute advantages for GRPO using reward signals."""

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        super().__init__()
        self.gamma = gamma  # Discount factor
        self.lambda_ = lambda_  # GAE lambda parameter

    @endpoint
    async def __call__(
        self, episode: Episode, rewards: List[float] = None
    ) -> List[float]:
        """
        Compute advantages for an episode using GAE.

        Args:
            episode: Episode containing turns and potential rewards
            rewards: Pre-computed rewards from reward model (optional)

        Returns:
            List of advantage values for each turn
        """
        num_turns = len(episode.turns)
        if num_turns == 0:
            return []

        # Use provided rewards or compute placeholder ones
        if rewards is None:
            rewards = self._compute_rewards(episode)

        # Placeholder value function - in practice would use critic model
        values = self._compute_values(episode)

        # Compute GAE advantages
        advantages = []
        gae = 0

        for t in reversed(range(num_turns)):
            if t == num_turns - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)

        return advantages

    def _compute_rewards(self, episode: Episode) -> List[float]:
        """Compute rewards for episode turns."""
        # Fallback dummy rewards if none provided
        return [1.0 for _ in episode.turns]

    def _compute_values(self, episode: Episode) -> List[float]:
        """Compute value estimates for episode states."""
        # Placeholder - would use critic/value model
        return [0.5 for _ in episode.turns]  # Dummy values


class RefModel(Actor):
    """Reference model for computing baseline log probabilities."""

    def __init__(self, model_name: str = "reference_model"):
        super().__init__()
        self.model_name = model_name
        self.model = None  # Placeholder for actual model

    @endpoint
    async def forward(self, tokens: List[int]) -> List[float]:
        """
        Compute log probabilities using reference model.

        Args:
            tokens: List of token IDs

        Returns:
            List of log probabilities for each token
        """
        if not tokens:
            return []

        # Placeholder implementation - in practice would use actual reference model
        # to compute log probabilities for the tokens
        logprobs = []
        for token in tokens:
            # Dummy log probability computation
            logprob = -torch.log(torch.tensor(float(len(tokens)))).item()
            logprobs.append(logprob)

        return logprobs


class DatasetActor(Actor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    def __init__(self, dataset_name: str = "gsm8k", split: str = "train"):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None
        self._iterator = None
        self._setup_dataset()

    def _setup_dataset(self):
        """Initialize the HuggingFace dataset."""
        try:
            self.dataset = HfIterableDataset(
                path=self.dataset_name,
                split=self.split,
                shuffle_buffer_size=1000,
                dataset_name=f"{self.dataset_name}_{self.split}",
            )
            self._iterator = iter(self.dataset)
        except Exception as e:
            print(f"Warning: Could not load dataset {self.dataset_name}: {e}")
            # Create a dummy dataset for testing
            self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """Create dummy dataset for testing when real dataset unavailable."""
        self.dummy_prompts = [
            "What is 2 + 2?",
            "Solve: 3 * 4 = ?",
            "Calculate 10 - 5",
            "What is 8 / 2?",
            "Find the value of 5 + 3",
        ]
        self.dummy_index = 0

    @endpoint
    async def __next__(self) -> Optional[str]:
        """Get next prompt from dataset."""
        if self.dataset and self._iterator:
            try:
                sample = next(self._iterator)
                # Extract prompt from sample - adjust based on actual dataset format
                return sample.get("question", sample.get("prompt", str(sample)))
            except StopIteration:
                # Restart iterator for infinite iteration
                self._iterator = iter(self.dataset)
                return await self.__next__()
            except Exception as e:
                print(f"Error getting next sample: {e}")
                return None
        else:
            # Use dummy dataset
            if hasattr(self, "dummy_prompts"):
                prompt = self.dummy_prompts[self.dummy_index % len(self.dummy_prompts)]
                self.dummy_index += 1
                return prompt
            return None


async def main():
    """Main GRPO training loop with rollout and training processes."""

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
        dataset_name="gsm8k",
        split="train",
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

    math_reward_actor = await spawn_service(
        default_service_cfg,
        MathRewardActor,
    )

    print("All services initialized successfully")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        while True:
            prompt = await dataloader.next.call()
            if prompt is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            version = await policy.get_current_version.choose()
            episode = Episode()

            # Generate response using policy
            async with policy.session(version=version):
                action = await policy.generate.call(prompt)
                episode.add_turn((prompt, action))

            # Get tokens for logprob computation
            tokens = episode.get_tokens()
            episode.add_transform_info("tokens", tokens)

            # Compute current policy logprobs (from the policy that generated the action)
            current_logprobs = await policy.compute_logprobs.call(tokens)
            episode.add_transform_info("current_logprobs", current_logprobs)

            # Compute reference model logprobs
            ref_logprobs = await ref_model.forward.call(tokens)
            episode.add_transform_info("ref_logprobs", ref_logprobs)

            # Compute math rewards for each turn in the episode
            rewards = []
            for prompt, response in episode.turns:
                reward = await math_reward_actor.evaluate_math_response.call(
                    prompt, response
                )
                rewards.append(reward)
            episode.add_transform_info("rewards", rewards)

            # Compute advantages using the actual math rewards
            advantages = await compute_advantages.__call__.call(episode, rewards)
            episode.add_transform_info("advantages", advantages)

            # Add to replay buffer
            await replay_buffer.add.call(episode)

            rollout_count += 1
            if rollout_count % 10 == 0:
                print(f"Generated {rollout_count} rollouts")
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                print(f"Average math reward for last rollout: {avg_reward:.2f}")

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
