# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio

from apps.grpo.main import main as grpo_main
from forge.actors.policy import Policy

from forge.actors.reference_model import ReferenceModel
from forge.cli.config import parse
from forge.controller.launcher.mast import lauch_mast_job

from omegaconf import DictConfig


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    import debugpy

    debugpy.listen(5678)
    print("[MAIN] Waiting for VS Code debugger to attach...")
    debugpy.wait_for_client()
    print("Attached!")
    # rithesh-forge-c8072a
    spec = await lauch_mast_job(cfg, job_name="rithesh-forge-c8072a")
    job_name = spec.name

    # rl_trainer = (
    #     await RLTrainer.options(**cfg.services.trainer).as_service(
    #         **cfg.trainer, loss=None
    #     ),
    # )

    await grpo_main(cfg)
    print("successful")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
