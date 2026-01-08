# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from forge.rl.advantage import ComputeAdvantages
from forge.rl.collate import collate
from forge.rl.grading import RewardActor
from forge.rl.losses import (
    aggregate,
    AggType,
    BaseLossConfig,
    CISPOLoss,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    create_shifted_targets,
    CROSS_ENTROPY_IGNORE_IDX,
    DAPOLoss,
    GRPOLoss,
    GSPOLoss,
    KLType,
    LossOutput,
    masked_mean,
    pg_cispo,
    pg_dual_clip,
    pg_ppo_clip,
    pg_soft_gate,
    PolicyGradientLoss,
    RatioType,
    SAPOLoss,
)
from forge.rl.types import Episode, Group

__all__ = [
    "Episode",
    "Group",
    "collate",
    "ComputeAdvantages",
    "RewardActor",
    # Loss types
    "LossOutput",
    "BaseLossConfig",
    "PolicyGradientLoss",
    # Type aliases
    "AggType",
    "RatioType",
    "KLType",
    # Constants
    "CROSS_ENTROPY_IGNORE_IDX",
    # Losses
    "GRPOLoss",
    "DAPOLoss",
    "GSPOLoss",
    "CISPOLoss",
    "SAPOLoss",
    # Primitives
    "compute_logprobs",
    "compute_entropy",
    "compute_ratio",
    "compute_kl",
    "aggregate",
    "masked_mean",
    "create_shifted_targets",
    # PG strategies
    "pg_ppo_clip",
    "pg_dual_clip",
    "pg_soft_gate",
    "pg_cispo",
]
