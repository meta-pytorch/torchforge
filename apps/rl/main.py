# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A working example showcasing a practical example of forge with RL.

Run this with:
    HF_HUB_DISABLE_XET=1 python -m apps.rl.main --config apps/rl/llama3_8b.yaml

"""

import asyncio
import logging
import sys

from dataclasses import asdict

# from forge.actors import Collector, Policy, PolicyRouter, Reference, Trainer
from forge.actors import Policy, Trainer

from forge.cli.config import parse
from forge.controller import get_proc_mesh
from omegaconf import DictConfig, OmegaConf
from torchtitan.experiments.forge.job_config import ForgeJobConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def setup_entity(entity: str, cfg: DictConfig):
    logging.info("Initializing %s...", entity)
    assert hasattr(
        cfg, "scheduler"
    ), f"{entity} config does not have a scheduler definition"

    logging.info("Creating proc_mesh for %s, with config %s", entity, cfg.scheduler)
    p = await get_proc_mesh(cfg.scheduler)
    logging.info("Proc mesh created for %s: %s", entity, p)

    match entity:
        case "trainer":
            actor = await p.spawn("trainer", Trainer, cfg)
        case "policy":
            actor = await p.spawn("policy", Policy, cfg)
        case _:
            raise ValueError(f"Unknown entity {entity}")

    return p, actor


async def run(cfg: DictConfig):
    results = await asyncio.gather(
        setup_trainer(entity="trainer", cfg=cfg.trainer),
        setup_policy(cfg="policy", cfg=cfg.policy),
    )
    print(results)


@parse
def recipe_main(cfg: DictConfig) -> None:
    # TODO: this is a hack to get the defaults from ForgeJobConfig
    default_cfg = ForgeJobConfig()
    # Hack to deal with literal types from titan
    default_cfg = asdict(default_cfg)
    cfg = OmegaConf.merge(default_cfg, cfg)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
