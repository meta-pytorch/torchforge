# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Iterator

from forge.data_models.api import PromptDataLoader, PromptDataset
from forge.data_models.prompt import Prompt, to_prompt

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "only gives very concise answers."
)

ANSWER_METADATA_KEY = "answer"


class SumDigitsDataset(PromptDataset):
    def __init__(self, max_samples=1000):
        self.min_digit_length = 2  # TODO: needs to come from config/argument
        self.max_digit_length = 3  # TODO: needs to come from config/argument
        self.max_numbers = max_samples
        self.data = self.generate_random_number()

    def __iter__(self) -> Iterator[Prompt]:
        for data in self.data:
            yield to_prompt(
                prompt=f"What is the sum of the digits of {data}",
                system_instruction=SYSTEM_PROMPT,
                metadata={
                    ANSWER_METADATA_KEY: str(sum(int(x) for x in data))
                },  # actual answer to be used in grading later in the flow
            )

    def generate_random_number(self) -> list[str]:
        return [self.generate_one() for _ in range(self.max_numbers)]

    def generate_one(self) -> str:
        return "".join(
            str(random.randint(0, 9))
            for _ in range(random.randint(self.min_digit_length, self.max_digit_length))
        )


class SumDigitsDataLoader(PromptDataLoader):
    def __init__(self, batch_size: int = 1000):
        self.dataset = SumDigitsDataset(max_samples=batch_size)
        super().__init__(self.dataset, batch_size=batch_size)

    def __iter__(self) -> Iterator[Prompt]:
        for data in self.dataset:
            yield data
