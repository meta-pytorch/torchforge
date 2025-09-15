# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from forge.data_models.api import Generator, WeightsBuffer

from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt


class VLLMGenerator(Generator):
    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate(self, prompt: Prompt) -> List[Completion]:
        """
        Generate completions for a given prompt using vLLM.
        """
        return []

    def update_weights(self, weights_handle: WeightsBuffer) -> None:
        """
        Update the weights of the model.
        Args:
            weights: A remote handle to weights buffer
        """
        pass
