# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator wrapper with automatic vLLM version detection.

src/forge/actors/
  ├── generator.py              # Version detection wrapper
  └── vllm/                     # vLLM integrations
      ├── v0/                   # vLLM == 0.10.0 package
      │   ├── __init__.py
      │   └── generator.py      # Original implementation
      └── v1/                   # vLLM >= 0.13.0 package
          ├── __init__.py
          └── generator.py      # AsyncLLM implementation

"""

from __future__ import annotations

import logging

from forge.actors._vllm_utils import get_vllm_version, use_generator_v1

logger = logging.getLogger(__name__)

if use_generator_v1():
    logger.info(f"vLLM version {get_vllm_version()} detected. Using Generator v1.")
    from forge.actors.vllm.v1 import Generator
else:
    logger.info(f"vLLM version {get_vllm_version()} detected. Using Generator v0.")
    from forge.actors.vllm.v0 import Generator

# Re-export Generator for public API
__all__ = ["Generator"]
