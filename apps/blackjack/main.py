# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.blackjack.main --config apps/blackjack/qwen3_1_7b.yaml

import asyncio
import multiprocessing
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import requests
import torch
import torch.nn.functional as F
import torchstore as ts
from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
from forge.actors._torchstore_utils import (
    get_dcp_whole_state_dict_key,
    get_param_prefix,
)
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import TitanTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from forge.util.ops import compute_logprobs
from monarch.actor import endpoint
from omegaconf import DictConfig
from vllm.transformers_utils.tokenizer import get_tokenizer


def start_openspiel_server(game_name: str, port: int):
    """Start OpenSpiel server in background process."""
    os.environ["OPENSPIEL_GAME"] = game_name

    import uvicorn
    from envs.openspiel_env.server.app import app

    print(f"[SERVER] Starting uvicorn for game '{game_name}' on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def kill_process_on_port(port: int):
    """Kill any process using the specified port."""
    # Find process using the port
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.stdout.strip():
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
                print(f"[DEBUG] Killed existing process {pid} on port {port}")
            except ProcessLookupError:
                pass  # Process already dead
        time.sleep(0.5)  # Give OS time to release port
        return True
    return False


@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    # Processed data
    completion: Completion | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version

    @property
    def request_tensor(self) -> torch.Tensor:
        request_tokens: torch.Tensor = self.completion.prompt_ids
        # Use clone() instead of torch.tensor() to avoid UserWarning
        if isinstance(request_tokens, torch.Tensor):
            tensor = request_tokens.clone().detach()
        else:
            tensor = torch.tensor(request_tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        response_tokens: torch.Tensor = self.completion.token_ids
        # Use clone() instead of torch.tensor() to avoid UserWarning
        if isinstance(response_tokens, torch.Tensor):
            tensor = response_tokens.clone().detach()
        else:
            tensor = torch.tensor(response_tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


# Represents the group (G) of episodes in GRPO
Group = list[Episode]

# Represents the Policy Model to collect data from
Policy = Generator


def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates a list of batches into a single batch of inputs and targets.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    inputs = []
    targets = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs).squeeze()  # [b x s]

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]

        pad_id = batch[0].pad_id
        mask = response != pad_id

        input = {"tokens": torch.cat([request, response], dim=1)}
        target = {
            "response": response,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
            "padding_mask": mask,
        }
        inputs.append(input)
        targets.append(target)
    return inputs, targets


# Note: This is also available in losses.grpo_loss via `SimpleGRPOLoss`
def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    logprobs: torch.Tensor = compute_logprobs(logits, response)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()
    return loss


# Blackjack-specific helper functions
def format_prompt(step_num: int, action_history: list, obs, tokenizer) -> str:
    """
    Format game state as text prompt for LLM with full game information.

    Args:
        step_num: Current step number
        action_history: List of (action_name, player_total_after) tuples
        obs: OpenSpiel observation with metadata
        tokenizer: Tokenizer for chat template

    Returns:
        Formatted prompt string with game state
    """
    system = """You are an expert BlackJack player. Analyze the game state and output only 'HIT' or 'STAND'."""

    # Get game state from metadata (populated by OpenEnv server)
    player_total = obs.metadata.get("player_total", "?")
    dealer_card = obs.metadata.get("dealer_card", "?")

    state_desc = f"=== BlackJack Game (Step {step_num + 1}) ===\n\n"

    # Add game state information
    state_desc += "Current State:\n"
    state_desc += f"  Your hand total: {player_total}\n"

    # Format dealer card - just show the value (Ace or 2-10)
    if dealer_card == 1:
        dealer_str = "Ace"
    elif dealer_card != "?":
        dealer_str = str(dealer_card)
    else:
        dealer_str = "?"
    state_desc += f"  Dealer shows: {dealer_str}\n"
    state_desc += f"  Legal actions: {', '.join('HIT' if a == 0 else 'STAND' for a in obs.legal_actions)}\n"
    state_desc += "\n"

    # Add action history with hand totals for card counting
    if action_history:
        state_desc += "Previous actions:\n"
        for i, (action_name, hand_total) in enumerate(action_history):
            state_desc += f"  {i + 1}. {action_name} (hand became {hand_total})\n"
        state_desc += "\n"

    state_desc += "What do you do? Output only 'HIT' or 'STAND'. You have a small limit for thinking tokens, so avoid thinking for long."

    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": state_desc},
    ]

    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def parse_action(response_text: str, legal_actions: list[int]) -> int:
    """Parse action from model's text response."""
    text_lower = response_text.lower()

    if text_lower.endswith("hit"):
        action_id = 0
    elif text_lower.endswith("stand"):
        action_id = 1
    else:
        action_id = 2

    return action_id


@dataclass
class BlackJackReward(ForgeActor):
    """Reward actor for evaluating game outcomes."""

    @endpoint
    async def evaluate_response(
        self, prompt: str, response: str, game_reward: float
    ) -> float:
        """
        Evaluate episode reward with improved shaping.

        Args:
            prompt: Game state prompt
            response: Model's action
            game_reward: Raw game outcome (+1/-1/0)

        Returns:
            Shaped reward value
        """
        # Check if the response ends with a valid action
        response_lower = response.lower().strip()
        last_words = response_lower.split()[-3:] if response_lower else []

        has_valid_action = any(word in ["hit", "stand"] for word in last_words)

        # Base reward from game outcome
        reward = float(game_reward)

        # Penalize invalid format (didn't end with HIT or STAND)
        if not has_valid_action:
            reward -= 1.0  # Strong penalty for invalid format
            record_metric("reward/invalid_action_rate", 1, Reduce.MEAN)
        else:
            record_metric("reward/invalid_action_rate", 0, Reduce.MEAN)

        # Optional reward shaping: Scale up wins
        if game_reward > 0:
            reward = max(reward, 1.5)  # Make wins more valuable (but respect penalty)
        elif game_reward == 0:
            reward = max(reward, 0.3)  # Pushes better than losses (but respect penalty)

        record_metric("reward/evaluate_response/avg_reward", reward, Reduce.MEAN)

        return reward


@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    async def compute(self, group: Group) -> list[float]:
        # TODO: add batch processing
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


@dataclass
class EnvironmentActor(ForgeActor):
    """Actor that manages OpenEnv connections and tokenizer."""

    server_url: str = "http://localhost:8004"
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        print(f"EnvironmentActor initialized (server: {self.server_url})")

    @endpoint
    async def get_tokenizer(self):
        return self._tokenizer

    @endpoint
    async def pad_token(self):
        # Use pad_token_id if available, otherwise use eos_token_id
        # Llama models don't have a pad token by default
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        else:
            return self._tokenizer.eos_token_id


async def drop_weights(version: int):
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    # TODO: once we have something like `get_meta()` in torchstore, we can just
    # query the type of the object instead of relying on keys.
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


async def play_game(
    game_idx: int,
    game_id: str,
    server_url: str,
    policy: Generator,
    tokenizer,
    rollout_count: int = 0,
):
    """
    Play a single blackjack game and collect episode data.

    Args:
        game_idx: Index of this game in the rollout
        game_id: Unique game identifier
        server_url: OpenEnv server URL
        policy: Policy (Generator) for action selection
        tokenizer: Tokenizer for prompt formatting
        rollout_count: Current rollout iteration

    Returns:
        List of step results with prompts, responses, and final reward
    """
    env = OpenSpielEnv(base_url=server_url)

    # Bypass corporate proxy for localhost connections
    env._http.trust_env = False

    print(f"\n🎮 GAME {game_idx + 1} (Rollout #{rollout_count + 1}) - ID: {game_id}")

    try:
        result = env.reset()
        obs = result.observation
        done = False
        step_num = 0
        action_history = []
        game_steps = []

        while not done and step_num < 10:  # Max 10 steps per game
            # Format prompt with game state
            prompt = format_prompt(step_num, action_history, obs, tokenizer)

            # Generate action with policy (with timeout)
            try:
                responses = await asyncio.wait_for(
                    policy.generate.route(prompt), timeout=60.0
                )
            except asyncio.TimeoutError:
                print(
                    f"[ERROR] Policy generation timed out for {game_id} at step {step_num}"
                )
                raise

            response = responses[0]

            # Parse and execute action
            action_id = parse_action(response.text, obs.legal_actions)
            action_name = "HIT" if action_id == 0 else "STAND"

            # Store step data (reward assigned later)
            game_steps.append(
                {
                    "step_num": step_num,
                    "prompt": prompt,
                    "response": response,
                }
            )

            # Take action in environment
            result = env.step(
                OpenSpielAction(action_id=action_id, game_name="blackjack")
            )
            obs = result.observation
            done = result.done

            # Add action to history with the resulting hand total (for card counting)
            hand_total_after = obs.metadata.get("player_total", "?")
            action_history.append((action_name, hand_total_after))

            step_num += 1

        # Get final game outcome
        final_game_reward = result.reward  # +1 (win), -1 (loss), or 0 (push)

        outcome_text = (
            "WIN"
            if final_game_reward > 0
            else ("LOSS" if final_game_reward < 0 else "PUSH")
        )
        print(
            f"  Result: {outcome_text} (reward={final_game_reward}, steps={len(game_steps)})"
        )

        # Print all steps with full model thinking
        if game_steps:
            print(f"\n  === GAME SUMMARY ===")
            for step_data in game_steps:
                print(f"\n  Step {step_data['step_num'] + 1}:")

                # Parse prompt to show key information
                prompt_lines = step_data["prompt"].split("\n")
                for line in prompt_lines:
                    if "Your hand total:" in line or "Dealer shows:" in line:
                        print(f"    {line.strip()}")

                # Show action taken
                action_text = step_data["response"].text
                if "hit" in action_text.lower():
                    action_taken = "HIT"
                elif "stand" in action_text.lower():
                    action_taken = "STAND"
                else:
                    action_taken = "UNKNOWN"
                print(f"    Action: {action_taken}")

                # Show full thinking process
                print(f"\n    Full AI thinking:")
                print(f"    {'-' * 60}")
                # Print the complete response text with proper indentation
                for line in step_data["response"].text.split("\n"):
                    print(f"    {line}")
                print(f"    {'-' * 60}")

            print(f"\n  Final outcome: {outcome_text} (reward={final_game_reward})")
            print(f"  ===================\n")

        # Assign final reward to all steps
        all_step_results = []
        total_steps = len(game_steps)
        for step_data in game_steps:
            all_step_results.append(
                {
                    "game_id": game_id,
                    "final_reward": final_game_reward,
                    "total_steps": total_steps,
                    **step_data,
                }
            )

        # Record game outcome metrics with clearer names
        record_metric("game/total_games_played", 1, Reduce.SUM)
        record_metric("game/average_game_length_in_steps", len(game_steps), Reduce.MEAN)

        # Average reward: +1 for win, -1 for loss, 0 for push
        record_metric("game/average_reward", final_game_reward, Reduce.MEAN)

        # Track wins, losses, pushes separately
        if final_game_reward > 0:
            record_metric("game/count_wins", 1, Reduce.SUM)
            record_metric("game/win_rate", 1, Reduce.MEAN)  # 1 = win, 0 = not win
        elif final_game_reward < 0:
            record_metric("game/count_losses", 1, Reduce.SUM)
            record_metric("game/win_rate", 0, Reduce.MEAN)  # 0 = loss
        else:
            record_metric("game/count_pushes", 1, Reduce.SUM)
            record_metric("game/win_rate", 0, Reduce.MEAN)  # 0 = push (not a win)

        # Parse the last observation before game ended to get final state
        # Note: We use the observation from the last step (before done=True)
        if game_steps:
            # Get the observation from the last action step
            last_step_obs = obs  # This is the final obs after the last step

            player_final = last_step_obs.metadata.get("player_total")
            dealer_card = last_step_obs.metadata.get("dealer_card")

            if player_final is not None and dealer_card is not None:
                # Record final state metrics
                record_metric(
                    "game/average_player_final_hand", player_final, Reduce.MEAN
                )
                record_metric("game/average_dealer_upcard", dealer_card, Reduce.MEAN)

                # Player busted if > 21
                if player_final > 21:
                    record_metric("game/bust_rate", 1, Reduce.MEAN)
                else:
                    record_metric("game/bust_rate", 0, Reduce.MEAN)

                # Track average hand totals by outcome (for strategy analysis)
                if final_game_reward > 0:  # Win
                    record_metric(
                        "game/average_winning_hand_total", player_final, Reduce.MEAN
                    )
                elif final_game_reward < 0:  # Loss
                    record_metric(
                        "game/average_losing_hand_total", player_final, Reduce.MEAN
                    )

        return all_step_results

    except Exception as e:
        print(f"[ERROR] play_game {game_id} failed with {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # ---- Start OpenSpiel Server ---- #
    game_name = cfg.blackjack_env.get("game_name", "blackjack")
    server_port = cfg.blackjack_env.get("server_port", 8004)

    # Clean up any existing server on this port
    if kill_process_on_port(server_port):
        print(f"Cleaned up existing server on port {server_port}")

    print(f"Starting OpenSpiel server for game '{game_name}' on port {server_port}...")
    server_process = multiprocessing.Process(
        target=start_openspiel_server, args=(game_name, server_port)
    )
    server_process.start()

    # Wait for server to be ready
    print("Waiting for OpenSpiel server to be ready...")
    server_ready = False
    for i in range(30):  # Try for 30 seconds
        # Check if server process is still alive
        if not server_process.is_alive():
            print(f"[ERROR] Server process died unexpectedly!")
            print(f"[ERROR] Exit code: {server_process.exitcode}")
            raise RuntimeError(
                f"OpenSpiel server process crashed during startup (exit code: {server_process.exitcode})"
            )

        try:
            # Skip proxy for localhost to avoid corporate proxy blocking with 403
            resp = requests.get(
                f"http://localhost:{server_port}/health",
                timeout=1,
                proxies={"http": None, "https": None},  # Bypass proxy
            )
            print(f"[DEBUG] Health check attempt {i+1}: status={resp.status_code}")
            if resp.status_code == 200:
                server_ready = True
                print(f"✓ OpenSpiel server ready (took {i+1}s)")
                break
        except Exception as e:
            print(f"[DEBUG] Health check attempt {i+1} failed: {type(e).__name__}: {e}")
            time.sleep(1)

    if not server_ready:
        server_process.terminate()
        raise RuntimeError(f"OpenSpiel server never became ready on port {server_port}")

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup services ---- #

    # Extract only the fields needed for EnvironmentActor
    env_actor_config = {
        "server_url": cfg.blackjack_env.server_url,
        "model": cfg.blackjack_env.model,
    }

    (
        env_actor,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        EnvironmentActor.options(**cfg.actors.blackjack_env).as_actor(
            **env_actor_config
        ),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        BlackJackReward.options(**cfg.services.reward_actor).as_service(),
    )

    # Set max_steps to the configured value, or -1 if not specified or Null
    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()
    # Here we spawn a torchstore storage volume per trainer process.
    # We initialize after service initialization because torchstore currently
    # requires access to the underlying proc meshes in the local rank strategy.
    # We should be able to hide this in the future.
    # TODO: support multiple host meshes
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # ---- Warmup policy ---- #
    print("Warming up policy with test generation...")
    test_prompt = "Test prompt to warm up the model."
    try:
        test_response = await asyncio.wait_for(
            policy.generate.route(test_prompt), timeout=120.0
        )
        print(f"✓ Policy ready, test response: '{test_response[0].text[:50]}...'")
    except asyncio.TimeoutError:
        raise RuntimeError("Policy warmup timed out after 120s")
    except Exception as e:
        raise RuntimeError(f"Policy warmup failed: {e}")

    # ---- Test OpenSpiel server ---- #
    print("Testing OpenSpiel server connection...")
    test_env = OpenSpielEnv(
        base_url=cfg.blackjack_env.get("server_url", "http://localhost:9000")
    )
    # Bypass corporate proxy for localhost - must set trust_env=False
    test_env._http.trust_env = False
    try:
        print(
            f"[DEBUG] Test env base_url={test_env._base}, timeout={test_env._timeout}"
        )
        print(f"[DEBUG] Test env trust_env={test_env._http.trust_env}")
        print(f"[DEBUG] Calling test_env.reset()...")
        test_result = test_env.reset()
        print(
            f"✓ OpenSpiel server test successful, legal_actions={test_result.observation.legal_actions}"
        )
        test_env.close()
    except Exception as e:
        print(f"[ERROR] OpenSpiel server test failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"OpenSpiel server test failed: {e}")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await env_actor.pad_token.call_one()
        tokenizer = await env_actor.get_tokenizer.call_one()
        server_url = cfg.blackjack_env.get("server_url", "http://localhost:8004")

        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()

            # Play group_size games
            all_step_results = []
            for game_idx in range(group_size):
                game_id = str(uuid.uuid4())[:8]
                step_results = await play_game(
                    game_idx=game_idx,
                    game_id=game_id,
                    server_url=server_url,
                    policy=policy,
                    tokenizer=tokenizer,
                    rollout_count=rollout_count,
                )
                all_step_results.extend(step_results)

            t.step("play_games")

            # Construct episodes and calculate rewards
            episodes = []
            input_ids = torch.ones(
                (len(all_step_results), max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )
            for i, step_result in enumerate(all_step_results):
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    target=None,
                    completion=step_result["response"],
                )
                episode.reward = await reward_actor.evaluate_response.route(
                    prompt=step_result["prompt"],
                    response=step_result["response"].text,
                    game_reward=step_result["final_reward"],
                )
                episodes.append(episode)

                # Build input_ids for reference logprobs
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

            t.step("reward_evaluation")

            ref_logprobs = await ref_model.forward.route(
                input_ids, max_req_tokens, return_logprobs=True
            )
            t.step("reference_model_calculate_logprobs")

            for i, episode in enumerate(episodes):
                episode.ref_logprobs = ref_logprobs[i]
            del ref_logprobs, input_ids

            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.call_one(episode)

            rollout_count += 1
            record_metric(
                "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
            )
            t.stop()

    async def continuous_training():
        training_step = 0
        restart_tracer = True  # Flag to control when to restart tracer

        while max_steps == -1 or training_step < max_steps:
            # Restart tracer when needed (initial start or after completing a training step)
            # Otherwise, we cannot measure time waiting for buffer
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                t.step("waiting_for_buffer")

                inputs, targets = batch
                await trainer.train_step.call(inputs, targets)
                training_step += 1
                t.step("train_step")

                await trainer.push_weights.call(training_step)
                t.step("push_weights")

                await policy.update_weights.fanout(training_step)
                t.step("update_weights")

                if training_step >= 2:
                    await drop_weights(training_step - 1)
                    t.step("drop_weights")

                t.stop()
                restart_tracer = True

                # Flush metrics every training step to WandB
                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
    )
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Shutting down... (this may take a few seconds)")
        shutdown_event.set()

        # Cancel rollout tasks
        try:
            # Give rollouts up to 5s to finish naturally
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            print("Timeout waiting for rollouts; forcing cancellation...")
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        # Cancel training task
        training_task.cancel()
        try:
            await asyncio.wait_for(training_task, timeout=2)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        # Shutdown forge actors/services with timeout
        print("Shutting down Forge actors...")
        try:
            await asyncio.wait_for(shutdown(), timeout=10)
            print("✓ Forge actors shut down")
        except asyncio.TimeoutError:
            print("⚠ Forge shutdown timed out after 10s, forcing exit...")

        # Shutdown OpenSpiel server
        print("Stopping OpenSpiel server...")
        server_process.terminate()
        server_process.join(timeout=2)
        if server_process.is_alive():
            print("⚠ Server didn't stop gracefully, killing...")
            server_process.kill()
            server_process.join(timeout=1)
        print("✓ OpenSpiel server stopped")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
