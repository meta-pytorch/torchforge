# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import getpass
import os
import uuid
from argparse import ArgumentParser, Namespace

from agent_tune.data_loaders.sum_digits_data_loader import SumDigitsDataLoader
from agent_tune.data_models.experience import from_scored_completions
from agent_tune.data_models.minibatch import from_experiences
from agent_tune.scorers.sum_digits_scorer import SumDigitsScorer

from forge.actors.learner import Learner
from forge.actors.policy_v1 import Policy
from monarch._src.actor.proc_mesh import proc_mesh as local_proc_mesh
from torch.utils.tensorboard import SummaryWriter


async def main(args: Namespace) -> None:
    master_port = int(os.environ.get("BASE_MASTER_PORT", "12345"))
    master_addr = "localhost"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    model_name = args.model_name
    log_dir = f"{_local_root_dir()}/tb"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"To launch a TensorBoard: tensorboard --logdir {log_dir}")
    learner_proc_mesh = await local_proc_mesh(
        hosts=1,
        gpus=1,
        env=dict(os.environ),
    )

    # change the master-port for policy mesh
    os.environ["MASTER_PORT"] = str(master_port + 1)
    policy_proc_mesh = await local_proc_mesh(
        hosts=1,
        gpus=1,
        env=dict(os.environ),
    )

    policy_mesh = await policy_proc_mesh.spawn(
        "policy",
        Policy,
        model_name,
    )
    trainer_mesh = await learner_proc_mesh.spawn(
        "trainer",
        Learner,
        model_name,
    )
    scorer = SumDigitsScorer()
    steps = 20
    total_prompts = 0
    total_completions = 0
    for step in range(steps):
        total_loss = 0.0
        loss_count = 0
        total_score = 0.0
        score_count = 0
        for prompt in SumDigitsDataLoader(batch_size=100):
            completions = await policy_mesh.generate.call_one(prompt)
            scored_completions = scorer.score_batch(completions)

            # Accumulate scores from all completions
            for scored_completion in scored_completions:
                total_score += scored_completion.score
                score_count += 1
                # print(f"score: {scored_completion.score}")
            experiences = from_scored_completions(scored_completions)
            mini_batch = from_experiences(experiences, 300)
            loss_output = await trainer_mesh.accummulate_gradients.call_one(mini_batch)
            # print(
            #     f"loss: {loss_output.loss.numerator.local()/loss_output.loss.denominator.local()};"
            # )
            # Accumulate loss
            current_loss = (
                loss_output.loss.numerator.local()
                / loss_output.loss.denominator.local()
            )
            total_loss += current_loss
            loss_count += 1

        # Update cumulative totals
        total_prompts += loss_count
        total_completions += score_count

        # Calculate and print means for this step
        mean_loss = total_loss / loss_count if loss_count > 0 else 0.0
        mean_score = total_score / score_count if score_count > 0 else 0.0

        # Log to TensorBoard
        writer.add_scalar("Loss/Mean", mean_loss, step)
        writer.add_scalar("Score/Mean", mean_score, step)
        writer.add_scalar("Metrics/Total_Prompts", total_prompts, step)
        writer.add_scalar("Metrics/Total_Completions", total_completions, step)

        print(
            f"Step {step + 1}: Mean Loss: {mean_loss:.4f} | Mean Score: {mean_score:.4f} | Total Prompts: {total_prompts} | Total Completions: {total_completions}"
        )

        await trainer_mesh.apply_gradients.call_one()
        weights_buffer = await trainer_mesh.snapshot_weights.call_one()
        await policy_mesh.update_weights.call_one(weights_buffer)

    # Close the writer when done
    writer.close()


def _local_root_dir() -> str:
    local_runs_root = os.path.expanduser("~/monarchrl_local_runs")
    job_name = f"rl_loca_run_{uuid.uuid4().hex}"
    path = os.path.join(local_runs_root, job_name)
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run a sync RL loop using monarch")
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default=f"/home/{getpass.getuser()}/pci-wsf/huggingface_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/model",
        help="Path to the model to use for training",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="sum_digits",
        choices=["sum_digits"],
        help="Type of experiment to run (currently only sum_digits is supported)",
    )
    args: Namespace = parser.parse_args()
    asyncio.run(main(args))
