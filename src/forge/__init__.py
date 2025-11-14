# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__version__ = ""

# Enables faster downloading. For more info: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
# To disable, run `HF_HUB_ENABLE_HF_TRANSFER=0 tune download <model_config>`
try:
    import os

    import hf_transfer  # noqa

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass


# FIXME: remove this once wandb fixed this issue
# Patch importlib.metadata.distributions before wandb imports it
# to filter out packages with None metadata
import importlib.metadata

_original_distributions = importlib.metadata.distributions


def _patched_distributions():
    """Filter out distributions with None metadata"""
    for dist in _original_distributions():
        if dist.metadata is not None:
            yield dist


importlib.metadata.distributions = _patched_distributions
