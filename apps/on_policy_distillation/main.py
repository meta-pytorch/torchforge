# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import itertools
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchstore as ts
from datasets import load_dataset
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.trainer import RLTrainer
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.util.config import parse
from forge.util.ops import compute_logprobs
from omegaconf import DictConfig
from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class Trajectory:
    pad_id: int
    request_len: int
    response_len: int
    completion: Completion | None = None
    teacher_logprobs: torch.Tensor | None = None

    @property
    def request_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.prompt_ids.to(torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        elif tensor.shape[0] > self.request_len:  # truncate
            tensor = tensor[-self.request_len :]
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.token_ids.to(torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        elif tensor.shape[0] > self.response_len:  # truncate
            tensor = tensor[: self.response_len]
        return tensor


def collate(
    batches: list[list[Trajectory]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    targets = []
    for batch in batches:
        request = [t.request_tensor for t in batch]
        request = torch.stack(request)

        response = [t.response_tensor for t in batch]
        response = torch.stack(response)

        teacher_logprobs = [t.teacher_logprobs for t in batch]
        teacher_logprobs = torch.stack(teacher_logprobs)

        pad_id = batch[0].pad_id
        padding_mask = response != pad_id

        input = {"tokens": torch.cat([request, response], dim=1)}
        target = {
            "response": response,
            "teacher_logprobs": teacher_logprobs,
            "padding_mask": padding_mask,
        }
        inputs.append(input)
        targets.append(target)
    return inputs, targets


def reverse_kl_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    padding_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    student_logprobs = compute_logprobs(logits, response)

    reverse_kl = student_logprobs.detach() - teacher_logprobs
    advantages = -reverse_kl

    per_token = -(advantages * student_logprobs) * padding_mask
    loss = per_token.sum() / padding_mask.sum().clamp(min=1)

    return loss.mean()


async def main(cfg: DictConfig):
    train_batch_size = cfg.train_batch_size
    max_steps = cfg.trainer.training.steps
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    provisioner = await init_provisioner()
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(cfg.metric_logging)
    student_trainer, student_generator, teacher = await asyncio.gather(
        RLTrainer.options(**cfg.services.trainer).as_actor(
            **cfg.trainer, loss=reverse_kl_loss
        ),
        Generator.options(**cfg.services.student_generator).as_service(
            **cfg.student_generator
        ),
        ReferenceModel.options(**cfg.services.teacher).as_service(**cfg.teacher),
    )

    # Initialize torchstore for weight management
    trainer_num_procs = cfg.services.trainer["procs"]
    trainer_host_mesh_name = cfg.services.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )

    print("All services initialized successfully!")

    # Configure my dataset
    tokenizer = get_tokenizer(cfg.student_model)
    map_fn = lambda x: tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": x["question"]
                + "\n\nPlease reason step by step, and put your final answer within \boxed{}.",
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    dataset = load_dataset(cfg.dataset.path, split=cfg.dataset.split).map(map_fn)
    dataset_iter = iter(dataset)

    step = 0
    for epoch in range(max_steps):
        start = time.perf_counter()
        if step >= max_steps:
            break

        trajectories = []
        while len(trajectories) < train_batch_size:
            prompt = next(dataset_iter)
            completions = await student_generator.generate.fanout(prompt)
            for completion in itertools.chain(*completions):
                trajectory = Trajectory(
                    pad_id=tokenizer.pad_token_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    completion=completion,
                )
                input_ids = torch.cat(
                    [
                        trajectory.request_tensor.unsqueeze(0),
                        trajectory.response_tensor.unsqueeze(0),
                    ],
                    dim=1,
                )
                teacher_logprobs = await teacher.forward.route(
                    input_ids, max_req_tokens, return_logprobs=True
                )
                trajectory.teacher_logprobs = teacher_logprobs
                trajectories.append(trajectory)

        trajectories = [
            trajectories[i::train_batch_size] for i in range(train_batch_size)
        ]
        inputs, targets = collate(trajectories)
        await student_trainer.train_step.call(inputs, targets)

        await student_trainer.push_weights.call(step)
        await student_generator.update_weights.fanout(step)

        step += 1

        end = time.perf_counter()
        print(f"Step {step} took {end - start} seconds")
        await mlogger.flush.call_one(step)

    print(f"Training completed after {step} steps")
    await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()
