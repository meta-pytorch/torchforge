# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from apps.grpo.main import main as grpo_main
from forge.cli.config import parse
from forge.controller.launcher.mast import lauch_mast_job

from omegaconf import DictConfig


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""

    # TODO: Remove this once we have a better way to launch mast jobs
    harcoded_job_name = "rithesh-forge-c8072a"
    spec = await lauch_mast_job(cfg, job_name=harcoded_job_name)
    await grpo_main(cfg)


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
