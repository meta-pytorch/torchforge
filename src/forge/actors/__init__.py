# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .collector import Collector

__all__ = ["Collector"]

try:
    from .policy import Policy, PolicyRouter

    __all__.extend(["Policy", "PolicyRouter"])
except ImportError as e:
    # Create placeholder classes that give helpful error messages
    class Policy:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Policy requires vLLM to be installed. "
                "Install it with: pip install vllm"
            ) from e

    class PolicyRouter:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PolicyRouter requires vLLM to be installed. "
                "Install it with: pip install vllm"
            ) from e

    __all__.extend(["Policy", "PolicyRouter"])
