# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from apps.sumdigits.sum_digits_data_loader import ANSWER_METADATA_KEY
from forge.data_models.api import Scorer
from forge.data_models.completion import Completion
from forge.data_models.scored_completion import ScoredCompletion


class SumDigitsScorer(Scorer):
    def score(self, completion: Completion) -> ScoredCompletion:
        """
        Scores the completion:
        1.0 if correct, 0.0 if empty, -1.0 if incorrect.
        """
        sampled_text = (
            str(completion.text).strip() if completion.text is not None else ""
        )
        metadata = getattr(completion.prompt, "metadata", None)
        if not metadata or metadata.get(ANSWER_METADATA_KEY) is None:
            raise ValueError(
                "Completion prompt metadata with answer is required for SumDigitsScorer."
            )

        answer = str(metadata[ANSWER_METADATA_KEY]).strip()
        if not sampled_text:
            score = 0.0
        elif sampled_text == answer:
            score = 1.0
        else:
            score = -1.0
        return ScoredCompletion(completion=completion, score=score)
