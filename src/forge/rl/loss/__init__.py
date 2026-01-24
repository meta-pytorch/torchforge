# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Types
from forge.rl.loss.cispo import CISPOLoss, pg_cispo
from forge.rl.loss.dapo import DAPOLoss, pg_dual_clip

# Loss classes
from forge.rl.loss.grpo import GRPOLoss
from forge.rl.loss.gspo import GSPOLoss

# Primitives
from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    create_shifted_targets,
    CROSS_ENTROPY_IGNORE_IDX,
    masked_mean,
    pg_ppo_clip,
)
from forge.rl.loss.sapo import pg_soft_gate, SAPOLoss
from forge.rl.loss.types import AggType, BaseLossConfig, KLType, LossOutput, RatioType

__all__ = [
    # Types
    "AggType",
    "RatioType",
    "KLType",
    "LossOutput",
    "BaseLossConfig",
    # Primitives
    "CROSS_ENTROPY_IGNORE_IDX",
    "masked_mean",
    "create_shifted_targets",
    "compute_logprobs",
    "compute_entropy",
    "compute_ratio",
    "compute_kl",
    "aggregate",
    # PG strategies
    "pg_ppo_clip",
    "pg_dual_clip",
    "pg_cispo",
    "pg_soft_gate",
    # Loss classes
    "GRPOLoss",
    "DAPOLoss",
    "GSPOLoss",
    "CISPOLoss",
    "SAPOLoss",
]
