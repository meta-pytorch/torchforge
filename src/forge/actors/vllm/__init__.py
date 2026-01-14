# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM integrations for torchforge.

This package contains version-specific implementations of the Generator actor
that integrate with different versions of vLLM.

Current OSS stable version uses vllm v0.10.0 as the Generator backend in the /v0 folder;
there is ongoing work to use vllm v0.13.0 (and onwards) in the /v1 folder.
"""

__all__ = []
