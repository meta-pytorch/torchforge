# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher factory"""


from forge.types import Launcher, LauncherConfig

from .base import BaseLauncher
from .slurm_launcher import Slurmlauncher
from .ssh_launcher import SSHLauncher

JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher(cfg)
    elif cfg.launcher == Launcher.SSH:
        success, error_msg = SSHLauncher.validate_configuration(cfg)
        if not success:
            raise ValueError(error_msg)
        return SSHLauncher(cfg)
    elif cfg.launcher == Launcher.MAST:
        try:
            from forge.fb.mast_launcher import MastLauncher

            return MastLauncher(cfg, detached=False)
        except ImportError as err:
            raise ValueError("MAST is not available, cannot launch MAST jobs.") from err

    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")
