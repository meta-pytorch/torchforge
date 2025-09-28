# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "1200"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


import asyncio

from apps.grpo.main import main as grpo_main
from forge.cli.config import parse
from forge.controller.provisioner import init_provisioner, SCHEDULER_KEY

from forge.types import Scheduler
from omegaconf import DictConfig


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    if cfg.get(SCHEDULER_KEY, Scheduler.MAST.value) != Scheduler.MAST.value:
        raise ValueError("Schuduler must be MAST.")

    # init mast provisioner
    await init_provisioner(cfg)
    await grpo_main(cfg)


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
