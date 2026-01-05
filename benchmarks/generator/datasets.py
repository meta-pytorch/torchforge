# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight dataset utilities for generator throughput benchmarking.
"""

import random
import uuid
from dataclasses import dataclass

from vllm import __version__ as vllm_version


if vllm_version >= "0.13.0":
    from vllm.tokenizers import TokenizerLike as Tokenizer
else:
    from vllm.transformers_utils.tokenizer import AnyTokenizer as Tokenizer


@dataclass
class BenchmarkRequest:
    """
    Attributes:
        prompt: The text prompt to generate from
        prompt_len: Length of the prompt in tokens
        expected_output_len: Expected length of generated output in tokens
        request_id: Unique identifier for this request.
    """

    prompt: str
    prompt_len: int
    expected_output_len: int
    request_id: str


class RandomDataset:
    """Generates prompts with random token sequences of specified lengths.

    Args:
        tokenizer: Tokenizer to use for encoding/decoding
        num_requests: Number of benchmark requests to generate
        input_len: Target input prompt length in tokens
        output_len: Target output generation length in tokens
        range_ratio: Variance ratio for input/output lengths (0.0-1.0).
                     0.0 means fixed lengths, 0.2 means ±20% variance.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_requests: int,
        input_len: int,
        output_len: int,
        range_ratio: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.num_requests = num_requests
        self.input_len = input_len
        self.output_len = output_len
        self.range_ratio = range_ratio
        self.vocab_size = tokenizer.vocab_size

    def _sample_length(self, target_len: int) -> int:
        """Sample a length with variance based on range_ratio."""
        if self.range_ratio == 0.0:
            return target_len

        min_len = int(target_len * (1 - self.range_ratio))
        max_len = int(target_len * (1 + self.range_ratio))
        return random.randint(min_len, max_len)

    def generate(self) -> list[BenchmarkRequest]:
        """Generate benchmark requests with random token sequences.

        Returns:
            List of BenchmarkRequest objects with random prompts
        """
        requests = []

        for i in range(self.num_requests):
            # Sample lengths with variance
            prompt_len = self._sample_length(self.input_len)
            output_len = self._sample_length(self.output_len)

            token_ids = [
                random.randint(0, self.vocab_size - 1) for _ in range(prompt_len)
            ]
            prompt = self.tokenizer.decode(token_ids)

            requests.append(
                BenchmarkRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=f"random-{i}-{uuid.uuid4().hex[:8]}",
                )
            )

        return requests


class FixedDataset:
    """Repeat a fixed prompt for baseline testing.

    Args:
        tokenizer: Tokenizer to use for encoding the prompt
        prompt: The fixed text prompt to repeat
        num_requests: Number of times to repeat the prompt
        output_len: Target output generation length in tokens
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        prompt: str,
        num_requests: int,
        output_len: int,
    ):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.num_requests = num_requests
        self.output_len = output_len
        self.prompt_len = len(tokenizer.encode(prompt))

    def generate(self) -> list[BenchmarkRequest]:
        """Generate benchmark requests with the same fixed prompt.

        Returns:
            List of BenchmarkRequest objects with the fixed prompt
        """
        requests = []

        for i in range(self.num_requests):
            requests.append(
                BenchmarkRequest(
                    prompt=self.prompt,
                    prompt_len=self.prompt_len,
                    expected_output_len=self.output_len,
                    request_id=f"fixed-{i}-{uuid.uuid4().hex[:8]}",
                )
            )

        return requests
