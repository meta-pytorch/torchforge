# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from forge.data_models.completion import Completion


@dataclass
class ScoredCompletion:
    """A completion with an associated score (from a reward model or human)."""

    completion: Completion
    score: float  # akin to reward

    # TODO: add more fields as needed.
