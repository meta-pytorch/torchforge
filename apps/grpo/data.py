# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from datasets import load_dataset
from forge.controller.actor import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from monarch.actor import endpoint
from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str = "openai/gsm8k"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = True
    model: str = ""
    seed: int = 42

    @endpoint
    async def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        self._epoch = 0

        def gsm8k_transform(sample):
            system_prompt = """
            Put all your scratchpad work between <think> and </think> tags.
            Your final answer should be between <answer> and </answer> tags otherwise it will not be scored.
            """
            request: str = sample["question"]
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            target: str = sample["answer"]
            formatted_target = target.split("#### ")[1]
            return {"request": formatted_request, "target": formatted_target}

        self._base_dataset = load_dataset(
            self.path, self.revision, split=self.data_split, streaming=self.streaming
        )
        self._base_dataset = self._base_dataset.map(gsm8k_transform)
        self._base_dataset = self._base_dataset.shuffle(seed=self.seed)
        self._base_dataset.set_epoch(self._epoch)
        self._iterator = iter(self._base_dataset)

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)

            record_metric("dataset/sample/current_epoch", self._epoch, Reduce.MAX)

            return sample
        except StopIteration:
            # Restart iterator for next epoch with reshuffling
            self._epoch += 1
            print(
                f"Dataset epoch {self._epoch - 1} completed. Starting epoch {self._epoch}"
            )
            self._base_dataset.set_epoch(self._epoch)
            self._iterator = iter(self._base_dataset)
            return next(self._iterator)

    @endpoint
    async def pad_token(self):
        # Use pad_token_id if available, otherwise use eos_token_id
        # Llama models don't have a pad token by default
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        else:
            return self._tokenizer.eos_token_id
