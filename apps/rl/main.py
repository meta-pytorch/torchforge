# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A working example showcasing a practical example of forge with RL.

Run this with:
    python -m apps.rl.main --config apps/rl/llama3_8b.yaml

"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from forge.actors import ReplayBuffer, RLTrainer
from forge.cli.config import parse

from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    ref_logprobs: Tensor | None = None
    reward: float | None = None
    advantage: float | None = None

    @property
    def request_tensor(self):
        tensor = torch.tensor(self.request_tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self):
        tensor = torch.tensor(self.response_tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


def collate(batches: list[list[Episode]]):
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


def compute_logprobs(
    logits: Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> Tensor:
    context_length = logits.shape[1] - input_ids.shape[1]

    # Truncate request logits and drop last
    logits = logits[:, context_length - 1 : -1]

    # Compute logprobs
    logprobs = torch.log_softmax(logits / temperature, dim=-1)
    logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)

    return logprobs


def simple_grpo_loss(
    logits: Tensor,
    response: Tensor,
    ref_logprobs: Tensor,
    advantages: Tensor,
    padding_mask: Tensor,
    beta: float = 0.1,
):
    """Simplified GRPO Loss for simplified single step updates
    Copied from https://github.com/pytorch/torchtune/blob/main/torchtune/dev/grpo/loss.py.
    """
    logprobs = compute_logprobs(logits, response)
    per_token_kl = (
        torch.exp(ref_logprobs.detach() - logprobs)
        - (ref_logprobs.detach() - logprobs)
        - 1
    )
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * per_token_kl)
    loss = (
        (per_token_loss * padding_mask).sum(dim=1) / (padding_mask.sum(dim=1) + 1e-8)
    ).mean()
    return loss


async def run(cfg: DictConfig):
    trainer = await RLTrainer.options(**cfg.actors.trainer).as_actor(
        **cfg.trainer, loss=simple_grpo_loss
    )
    replay_buffer = await ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
        **cfg.replay_buffer, collate=collate
    )

    print("Services initialized....")

    print("Collecting Data...")
    g = torch.manual_seed(0)
    global_batch_size = cfg.replay_buffer.batch_size * cfg.replay_buffer.dp_size
    for i in range(global_batch_size):
        req_len, res_len = torch.randint(64, 256, (2,), generator=g).tolist()
        e = Episode(
            episode_id=i,
            request="",
            policy_version=0,
            pad_id=0,
            request_len=256,
            response_len=256,
            request_tokens=torch.randint(64_000, (req_len,), generator=g).tolist(),
            response_tokens=torch.randint(64_000, (res_len,), generator=g).tolist(),
            ref_logprobs=torch.randn((256,), generator=g),
            advantage=torch.randn((1,), generator=g),
        )
        await replay_buffer.add.choose(e)

    print("Train step...")
    inputs, targets = await replay_buffer.sample.choose(curr_policy_version=0)
    outputs = await trainer.train_step.choose(inputs, targets)
    print("Loss: ", outputs["loss"])

    print("Shutting down...")
    await trainer.shutdown()
    await replay_buffer.shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
