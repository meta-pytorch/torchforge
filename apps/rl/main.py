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
from forge.actors import Policy, PolicyRouter, PostProcessor, Trainer

from forge.cli.config import parse
from forge.controller import get_proc_mesh
from omegaconf import DictConfig, OmegaConf
from torchtitan.experiments.forge.job_config import ForgeJobConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def setup_entity(entity: str, scheduler_cfg: DictConfig, *args, **kwargs):
    """Initialize an entity and return the proc_mesh and actor."""
    logging.info("Initializing %s...", entity)
    logging.info("Creating proc_mesh for %s, with config %s", entity, scheduler_cfg)
    p = await get_proc_mesh(scheduler_cfg)
    logging.info("Proc mesh created for %s: %s", entity, p)

    actor_type = None

    match entity:
        case "trainer":
            actor_type = Trainer
        case "policy":
            actor_type = Policy
        case "policy_router":
            actor_type = PolicyRouter
        case "postprocessor":
            actor_type = PostProcessor
        case _:
            raise ValueError(f"Unknown entity {entity}")

    actor = await p.spawn(entity, actor_type, *args, **kwargs)
    return p, actor


async def run(cfg: DictConfig):
    results = await asyncio.gather(
        setup_entity(
            entity="trainer",
            scheduler_cfg=cfg.trainer.scheduler,
            job_config=cfg.trainer,
        ),
        setup_entity(
            entity="policy",
            scheduler_cfg=cfg.policy.scheduler,
            model=cfg.policy.model,
            tensor_parallel_size=cfg.policy.tensor_parallel_size,
            pipeline_parallel_size=cfg.policy.pipeline_parallel_size,
            enforce_eager=cfg.policy.enforce_eager,
        ),
        setup_entity(entity="policy_router", scheduler_cfg=cfg.policy.scheduler),
        setup_entity(
            entity="postprocessor",
            scheduler_cfg=cfg.postprocessor.scheduler,
            job_config=cfg.postprocessor,
        ),
    )

    # Unzip results into individual proc and actor variables
    (
        (trainer_proc, trainer_actor),
        (policy_proc, policy_actor),
        (policy_router_proc, policy_router_actor),
        (postprocess_proc, postprocess_actor),
    ) = results

    print(f"Trainer proc: {trainer_proc}, actor: {trainer_actor}")
    print(f"Policy proc: {policy_proc}, actor: {policy_actor}")
    print(f"Policy router proc: {policy_router_proc}, actor: {policy_router_actor}")
    print(f"Postprocessor proc: {postprocess_proc}, actor: {postprocess_actor}")
    # Initialize everything
    await asyncio.gather(
        policy_actor.setup.call(),
        trainer_actor.setup.call(),
        policy_router_actor.setup.call(policy_actor),
        postprocess_actor.setup.call(),
    )
    print("Done initializing")

    print("shutting down...")
    # await asyncio.gather(
    #     *[p.stop() for p in [trainer_proc, policy_proc, policy_router_proc]]
    # )


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
