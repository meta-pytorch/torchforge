# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import uuid

import torch
import torchstore as ts
import yaml
from apps.grpo.data import DatasetActor
from apps.grpo.grading import MathReward, ThinkingReward
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl import collate, ComputeAdvantages, Episode, RewardActor
from forge.rl.loss import GRPOLoss
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.checkpoint import drop_weights
from forge.util.config import parse
from forge.util.logging import get_logger
from omegaconf import DictConfig, OmegaConf

logger = get_logger("INFO")


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    # Convert OmegaConf config to plain dict
    run_config_for_logging = OmegaConf.to_container(cfg, resolve=True)

    # Log config
    logger.info("=" * 30 + " CONFIGURATION " + "=" * 30)
    logger.info(
        yaml.dump(run_config_for_logging, default_flow_style=False, sort_keys=False)
    )

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        # Create launcher config with services and actors for pre-allocation
        launcher_config = LauncherConfig(
            **cfg.provisioner,
            services=cfg.get("services", {}),
            actors=cfg.get("actors", {}),
        )
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=launcher_config)
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})

    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(
        backend_config=metric_logging_cfg, run_config=run_config_for_logging
    )

    # ---- Setup loss function ---- #
    loss_fn = GRPOLoss(
        clip_low=0.2,
        clip_high=0.28,
        beta=0.1,
        agg_type="fixed_horizon",
    )

    # Fail-fast: Check loss/ref_model compatibility before spawning actors
    uses_ref_model = cfg.get("services", {}).get("ref_model") is not None
    if uses_ref_model and not isinstance(loss_fn, GRPOLoss):
        raise ValueError(
            f"ref_model is configured but {type(loss_fn).__name__} does not use ref_logprobs. "
            "Either remove the ref_model service config or use GRPOLoss with beta > 0."
        )
    if isinstance(loss_fn, GRPOLoss) and loss_fn.beta > 0 and not uses_ref_model:
        raise ValueError(
            f"GRPOLoss with beta={loss_fn.beta} requires ref_logprobs, but ref_model is not configured. "
            "Either add ref_model to services config or set beta=0."
        )

    # ---- Setup services ---- #

    async def noop():
        return None

    (
        dataloader,
        generator,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Generator.options(**cfg.services.generator).as_service(**cfg.generator),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=loss_fn
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        (
            ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model)
            if uses_ref_model
            else noop()
        ),
        RewardActor.options(**cfg.services.reward_actor).as_service(
            reward_functions=[MathReward(), ThinkingReward()]
        ),
    )

    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

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
    trainer_hosts = await provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await dataloader.pad_token.call_one()
        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()
            sample = await dataloader.sample.call_one()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            prompt, target = sample["request"], sample["target"]
            responses: list[Completion] = await generator.generate.route(prompt)
            t.step("policy_generation")

            # Construct episodes and calculate rewards
            episodes = []
            input_ids = torch.ones(
                (group_size, max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )
            seq_len = max_req_tokens + max_res_tokens

            for i, response in enumerate(responses):
                # Validate logprobs exist
                if response.logprobs is None:
                    raise ValueError(
                        "Completion.logprobs is None. "
                        "Ensure Generator returns logprobs by setting 'logprobs: 1' in sampling_params config."
                    )

                # Prepare generator_logprobs
                # Shift by -1 to align with next-token prediction
                actual_response_len = response.token_ids.shape[0]
                generator_logprobs = torch.zeros(seq_len, dtype=response.logprobs.dtype)
                generator_logprobs[
                    max_req_tokens : max_req_tokens + actual_response_len
                ] = response.logprobs
                generator_logprobs = torch.roll(generator_logprobs, shifts=-1, dims=0)
                generator_logprobs[-1] = 0.0

                # Prepare loss_mask
                response_mask = torch.zeros(seq_len, dtype=torch.float32)
                response_mask[max_req_tokens : max_req_tokens + actual_response_len] = (
                    1.0
                )
                loss_mask = torch.roll(response_mask, shifts=-1, dims=0)
                loss_mask[-1] = 0.0

                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    target=target,
                    request=prompt,
                    response=response.text,
                    completion=response,
                    generator_logprobs=generator_logprobs,
                    loss_mask=loss_mask,
                )

                (
                    episode.reward_breakdown,
                    episode.reward,
                ) = await reward_actor.evaluate_response.route(
                    prompt=prompt, response=response.text, target=target
                )
                episodes.append(episode)

                # Build input_ids for reference logprobs
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

                # Track token-based metrics
                prompt_tokens = episode.completion.prompt_ids.shape[0]
                response_tokens = episode.completion.token_ids.shape[0]

                record_metric("episode/avg_prompt_tokens", prompt_tokens, Reduce.MEAN)
                record_metric("episode/max_prompt_tokens", prompt_tokens, Reduce.MAX)
                record_metric("episode/min_prompt_tokens", prompt_tokens, Reduce.MIN)
                record_metric(
                    "episode/avg_response_tokens", response_tokens, Reduce.MEAN
                )
                record_metric(
                    "episode/max_response_tokens", response_tokens, Reduce.MAX
                )
                record_metric(
                    "episode/min_response_tokens", response_tokens, Reduce.MIN
                )

            # drop episodes if
            # 1> reward std-dev is very small (including all 0s and all 1s)
            # 2> any response was truncated (didn't end with EOS)
            # TODO: change it to filter only truncated episodes instead of dropping entire group
            rewards = [e.reward for e in episodes]
            rewards_std = torch.std(torch.tensor(rewards))
            is_low_variance = rewards_std < 1e-3
            num_truncated = sum(
                1 for e in episodes if e.completion.stop_reason == "length"
            )
            is_truncated = num_truncated > 0
            drop = is_low_variance or is_truncated

            n = len(episodes)
            record_metric(
                "main/continuous_rollouts/episodes_dropped/low_variance",
                n if is_low_variance else 0,
                Reduce.SUM,
            )
            record_metric(
                "main/continuous_rollouts/episodes_dropped/truncated",
                num_truncated,
                Reduce.SUM,
            )
            record_metric(
                "main/continuous_rollouts/episodes_dropped/total",
                n if drop else 0,
                Reduce.SUM,
            )

            if drop:
                del input_ids, episodes
                continue

            t.step("reward_evaluation")

            # Compute ref_logprobs only if ref_model is configured
            if ref_model is not None:
                ref_logprobs = await ref_model.forward.route(
                    input_ids, return_logprobs=True
                )
                t.step("reference_model_calculate_logprobs")

                for i, episode in enumerate(episodes):
                    episode.ref_logprobs = ref_logprobs[i]  # [seq_len]

                del ref_logprobs

            del input_ids

            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.call_one(episode)

                sample = episode.to_dict(
                    exclude=[
                        "completion",
                        "loss_mask",
                        "generator_logprobs",
                        "ref_logprobs",
                    ]
                )
                sample["score"] = sample["reward"]
                record_metric(
                    "main_samples/continuous_rollouts/sample_table",
                    sample,
                    Reduce.SAMPLE,
                )

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

                await generator.update_weights.fanout(training_step)
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

        training_task.cancel()

        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
