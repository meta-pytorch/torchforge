import asyncio
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
    # Processed data
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


def importance_sampling_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    padding_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    student_logprobs = compute_logprobs(logits, response)
    reverse_kl = -(student_logprobs - teacher_logprobs)
    prob_ratio = torch.exp(teacher_logprobs - student_logprobs)
    per_token_loss = prob_ratio * reverse_kl

    # Apply mask and compute mean over valid tokens
    masked_loss = per_token_loss * padding_mask
    num_valid_tokens = padding_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    loss_per_sequence = masked_loss.sum(dim=1, keepdim=True) / num_valid_tokens
    loss = loss_per_sequence.mean()

    return loss


async def main(cfg: DictConfig):
    train_batch_size = cfg.train_batch_size
    max_steps = cfg.trainer.training.steps
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    provisioner = await init_provisioner()
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(
        {
            "wandb": {"project": "opd-v0", "logging_mode": "global_reduce"},
            "console": {"logging_mode": "global_reduce"},
        }
    )
    student_trainer, student_generator, teacher = await asyncio.gather(
        RLTrainer.options(**cfg.services.trainer).as_actor(
            **cfg.trainer, loss=importance_sampling_loss
        ),
        Generator.options(**cfg.services.student_generator).as_service(
            **cfg.student_generator
        ),
        ReferenceModel.options(**cfg.services.teacher).as_service(**cfg.teacher),
    )

    # Setup torchstore for weight management
    trainer_num_procs = cfg.services.trainer["procs"]
    trainer_host_mesh_name = cfg.services.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )

    # Load dataset
    tokenizer = get_tokenizer(cfg.student_model)
    pad_id = tokenizer.pad_token_id
    dataset = load_dataset(cfg.dataset.path, split=cfg.dataset.get("split", "train"))
    dataset = dataset.filter(lambda x: x["domain"] == cfg.dataset["domain"])
    dataset_iter = iter(dataset)

    print("All services initialized successfully!")

    step = 0
    for epoch in range(max_steps):
        if step >= max_steps:
            break

        # Collect rollout
        trajectories = []
        while len(trajectories) < train_batch_size:
            try:
                sample = next(dataset_iter)
                # Extract the human prompt from OpenThoughts format
                conversations = sample.get("conversations", [])
                if conversations and len(conversations) > 0:
                    prompt = conversations[0].get("value", "")
                else:
                    prompt = sample.get("prompt", sample.get("text", ""))

                print(f"Starting request with prompt: {prompt}")
                completions = await student_generator.generate.route(prompt)

                for completion in completions:
                    # Create trajectory with raw completion
                    trajectory = Trajectory(
                        pad_id=pad_id,
                        request_len=max_req_tokens,
                        response_len=max_res_tokens,
                        completion=completion,
                    )

                    # Build padded input for teacher using trajectory properties
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
            except StopIteration:
                print("Dataset exhausted, resetting iterator")
                dataset_iter = iter(dataset)

        # Train on collected trajectories
        trajectories = [
            trajectories[i::train_batch_size] for i in range(train_batch_size)
        ]
        inputs, targets = collate(trajectories)
        await student_trainer.train_step.call(inputs, targets)

        step += 1

        # Push weights to student generator
        await student_trainer.push_weights.call(step)
        await student_generator.update_weights.fanout(step)

        await mlogger.flush.call_one(step)

    print(f"Training completed after {step} steps")
    await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()
