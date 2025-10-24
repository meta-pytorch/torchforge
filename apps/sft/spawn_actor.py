# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SpawnActor - Orchestrates the spawning and lifecycle management of actors.

This module provides a high-level interface for creating, setting up, running,
and cleaning up different types of actors (e.g., Trainer, Evaluator, etc.)
"""

import logging
from typing import Any, Type

from apps.sft.actor import BaseForgeActor
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SpawnActor:
    """
    Orchestrator for spawning and managing actor lifecycles.

    This class handles the creation, setup, execution, and cleanup of actors
    in a standardized way.
    """

    def __init__(self, actor_class: Type[BaseForgeActor], config: DictConfig):
        """
        Initialize the spawn actor orchestrator.

        Args:
            actor_class: The actor class to instantiate (must inherit from BaseForgeActor)
            config: Configuration dictionary for the actor
        """
        self.actor_class = actor_class
        self.config = config
        self.actor = None

        if not issubclass(actor_class, BaseForgeActor):
            raise TypeError(
                f"actor_class must be a subclass of BaseForgeActor, got {actor_class}"
            )

    async def spawn(self) -> Any:
        """
        Spawn the actor instance with the given configuration.

        Returns:
            The spawned actor instance
        """
        logger.info(f"Spawning {self.actor_class.__name__}...")

        process_cfg = self.config.pop("processes", {})

        self.actor = await self.actor_class.options(**process_cfg).as_actor(self.config)

        logger.info(f"{self.actor_class.__name__} spawned successfully.")
        return self.actor

    async def setup(self):
        """
        Setup the spawned actor (load data, checkpoints, etc.).
        """
        if self.actor is None:
            raise RuntimeError(
                "Actor must be spawned before setup. Call spawn() first."
            )

        logger.info(f"Setting up {self.actor_class.__name__}...")
        await self.actor.setup.call()
        logger.info(f"{self.actor_class.__name__} setup complete.")

    async def run(self):
        """
        Run the main execution logic of the actor.
        """
        if self.actor is None:
            raise RuntimeError(
                "Actor must be spawned before running. Call spawn() first."
            )

        logger.info(f"Running {self.actor_class.__name__}...")
        await self.actor.run.call()
        logger.info(f"{self.actor_class.__name__} execution complete.")

    async def cleanup(self):
        """
        Cleanup the actor resources and stop the mesh.
        """
        if self.actor is None:
            raise RuntimeError(
                "Actor must be spawned before cleanup. Call spawn() first."
            )

        logger.info(f"Cleaning up {self.actor_class.__name__}...")
        await self.actor.cleanup.call()

        if hasattr(self.actor, "mesh"):
            await self.actor.mesh.stop()

        logger.info(f"{self.actor_class.__name__} cleanup complete.")

    async def run_full_lifecycle(self):
        """
        Execute the complete actor lifecycle: spawn -> setup -> run -> cleanup.

        This is a convenience method that runs all phases in sequence.
        """
        logger.info(f"Starting full lifecycle for {self.actor_class.__name__}...")

        try:
            await self.spawn()
            await self.setup()
            await self.run()
        finally:
            if self.actor is not None:
                await self.cleanup()

        logger.info(f"Full lifecycle complete for {self.actor_class.__name__}.")


async def run_actor(
    actor_class: Type[BaseForgeActor],
    config: DictConfig,
) -> None:
    """
    Convenience function to run an actor with full lifecycle management.

    Args:
        actor_class: The actor class to instantiate
        config: Configuration dictionary for the actor
    """
    spawner = SpawnActor(actor_class, config)
    await spawner.run_full_lifecycle()
