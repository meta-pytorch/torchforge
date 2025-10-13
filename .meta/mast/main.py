# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import getpass
import sys
import uuid

from apps.grpo.main import main as grpo_main
from forge.cli.config import parse
from forge.controller.launcher import (
    JOB_NAME_KEY,
    LAUNCHER_KEY,
    MastLauncher,
    mount_mnt_directory,
)
from forge.controller.provisioner import init_provisioner

from forge.types import (
    Launcher,
    LauncherConfig,
    ProcessConfig,
    ProvisionerConfig,
    ServiceConfig,
)
from omegaconf import DictConfig

DEFAULT_CHECKPOINT_FOLDER_KEY = "checkpoint_folder"
DEFAULT_CHECKPOINT_FOLDER = "/mnt/wsfuse/teamforge/forge_runs/"


async def main(cfg: DictConfig, mode: str = "detached"):
    """Main module for launching mast jobs for GRPO training.

    Args:
        cfg: Configuration dictionary
        mode: "detached" (default) launches MAST job with client in MAST,
              "remote" runs training directly (used when client runs in MAST)
    """
    if cfg.get(LAUNCHER_KEY, Launcher.MAST.value) != Launcher.MAST.value:
        raise ValueError("Launcher must be MAST.")

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

    launcher_config = LauncherConfig(
        launcher=Launcher(cfg.get(LAUNCHER_KEY, Launcher.MAST.value)),
        job_name=cfg.get(JOB_NAME_KEY, None),
        services={k: ServiceConfig(**v) for k, v in cfg.services.items()},
        actors={k: ProcessConfig(**v) for k, v in cfg.actors.items()},
    )

    if mode == "detached":
        # In detached mode, just launch the MAST job with client role included
        # Get the config file from sys.argv to pass to the client role
        config_file = None
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                config_file = sys.argv[i + 1]
                break

        if not config_file:
            raise ValueError("--config argument is required in detached mode")

        launcher = MastLauncher(launcher_config, detached=True, config_file=config_file)
        await launcher.launch_mast_job()
        print(f"MAST job {launcher.job_name} launched successfully with client role.")
        print("The client is running inside MAST and will execute the training.")
    else:
        # In remote mode, we're already running inside MAST, so mount directory, init provisioner and run training
        mount_mnt_directory("/mnt/wsfuse")
        await init_provisioner(ProvisionerConfig(launcher_config=launcher_config))
        await grpo_main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="detached",
        choices=["detached", "remote"],
        help="Run mode: 'detached' for launching MAST job with client in MAST, 'remote' for running training directly",
    )
    args, remaining = parser.parse_known_args()

    # Replace sys.argv with remaining args so @parse can work
    sys.argv = [sys.argv[0]] + remaining

    @parse
    def _main(cfg):
        asyncio.run(main(cfg, mode=args.mode))

    _main()  # @parse grabs the cfg from CLI
