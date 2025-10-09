# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pip install --force-reinstall --no-deps . && python -m apps.mast.main --config apps/mast/qwen3_1_7b_mast.yaml

from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
configure(
    default_transport=ChannelTransport.MetaTlsWithHostname,
)

import asyncio
import getpass
import uuid

from apps.grpo.main import main as grpo_main
from forge.cli.config import parse
from forge.controller.provisioner import init_provisioner, JOB_NAME_KEY, SCHEDULER_KEY


from monarch.actor import context
from forge.types import Scheduler
from omegaconf import DictConfig

DEFAULT_CHECKPOINT_FOLDER_KEY = "checkpoint_folder"
DEFAULT_CHECKPOINT_FOLDER = "/mnt/wsfuse/teamforge/forge_runs/"


import monarch
from monarch.actor import Actor, endpoint

print(f"{str(context().actor_instance.actor_id)=}")


class Foo(Actor):
    @endpoint
    async def ping(self):
        print("pong")
        return "pong"

async def main(cfg: DictConfig):
    """Main module for launching mast jobs for GRPO training."""
    if cfg.get(SCHEDULER_KEY, Scheduler.MAST.value) != Scheduler.MAST.value:
        raise ValueError("Schuduler must be MAST.")

    if cfg.get(JOB_NAME_KEY, None) is not None:
        # prepend user name to the job to avoid name collision
        cfg[JOB_NAME_KEY] = f"{getpass.getuser()}-{cfg[JOB_NAME_KEY]}"
        print(f"Overriding mast job name to {cfg[JOB_NAME_KEY]}")

    if cfg.get(DEFAULT_CHECKPOINT_FOLDER_KEY, DEFAULT_CHECKPOINT_FOLDER) is not None:
        # append job_name and guid to CP folder path to avoid path collision
        if cfg[DEFAULT_CHECKPOINT_FOLDER_KEY] == DEFAULT_CHECKPOINT_FOLDER:
            cfg[
                DEFAULT_CHECKPOINT_FOLDER_KEY
            ] = f"{cfg[DEFAULT_CHECKPOINT_FOLDER_KEY]}{cfg[JOB_NAME_KEY]}-{uuid.uuid4().hex[:6]}"
        print(f"Overriding checkpoint folder to {cfg[DEFAULT_CHECKPOINT_FOLDER_KEY]}")

    # init mast provisioner
    await init_provisioner(cfg)

    await grpo_main(cfg)


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
