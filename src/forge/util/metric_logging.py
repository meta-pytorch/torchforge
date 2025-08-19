# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from typing import Mapping, Optional

import torch

from forge.interfaces import MetricLogger
from forge.types import Scalar


def get_metric_logger(logger: str = "stdout", **log_config):
    return METRIC_LOGGER_STR_TO_CLS[logger](**log_config)


class StdoutLogger(MetricLogger):
    """Logger to standard output."""

    def _log(self, name: str, data: Scalar, step: int) -> None:
        print(f"Step {step} | {name}:{data}")

    def _log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        print(f"Step {step} | ", end="")
        for name, data in payload.items():
            print(f"{name}:{data} ", end="")
        print("\n", end="")

    def close(self) -> None:
        sys.stdout.flush()


class TensorBoardLogger(MetricLogger):
    """Logger for use w/ PyTorch's implementation of TensorBoard (https://pytorch.org/docs/stable/tensorboard.html).

    Args:
        log_dir (str): torch.TensorBoard log directory
        organize_logs (bool): If `True`, this class will create a subdirectory within `log_dir` for the current
            run. Having sub-directories allows you to compare logs across runs. When TensorBoard is
            passed a logdir at startup, it recursively walks the directory tree rooted at logdir looking for
            subdirectories that contain tfevents data. Every time it encounters such a subdirectory,
            it loads it as a new run, and the frontend will organize the data accordingly.
            Recommended value is `True`. Run `tensorboard --logdir my_log_dir` to view the logs.
        **kwargs: additional arguments

    Example:
        >>> from torchtune.training.metric_logging import TensorBoardLogger
        >>> logger = TensorBoardLogger(log_dir="my_log_dir")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Note:
        This utility requires the tensorboard package to be installed.
        You can install it with `pip install tensorboard`.
        In order to view TensorBoard logs, you need to run `tensorboard --logdir my_log_dir` in your terminal.
    """

    def __init__(
        self,
        log_freq: Mapping[str, int],
        log_dir: str,
        organize_logs: bool = True,
        **kwargs,
    ):
        super().__init__(log_freq)

        from torch.utils.tensorboard import SummaryWriter

        self._writer: Optional[SummaryWriter] = None
        rank = _get_rank()

        # In case organize_logs is `True`, update log_dir to include a subdirectory for the
        # current run
        self.log_dir = (
            os.path.join(log_dir, f"run_{rank}_{time.time()}")
            if organize_logs
            else log_dir
        )

        # Initialize the log writer only if we're on rank 0.
        if rank == 0:
            self._writer = SummaryWriter(log_dir=self.log_dir)

    def _log(self, name: str, data: Scalar, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def _log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        for name, data in payload.items():
            self.log(name, data, step)

    def __del__(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


class WandBLogger(MetricLogger):
    """Logger for use w/ Weights and Biases application (https://wandb.ai/).
    For more information about arguments expected by WandB, see https://docs.wandb.ai/ref/python/init.

    Args:
        log_dir (Optional[str]): WandB log directory.
        project (str): WandB project name. Default is `torchtune`.
        entity (Optional[str]): WandB entity name. If you don't specify an entity,
            the run will be sent to your default entity, which is usually your username.
        group (Optional[str]): WandB group name for grouping runs together. If you don't
            specify a group, the run will be logged as an individual experiment.
        **kwargs: additional arguments to pass to wandb.init

    Example:
        >>> from torchtune.training.metric_logging import WandBLogger
        >>> logger = WandBLogger(log_dir="wandb", project="my_project", entity="my_entity", group="my_group")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Raises:
        ImportError: If ``wandb`` package is not installed.

    Note:
        This logger requires the wandb package to be installed.
        You can install it with `pip install wandb`.
        In order to use the logger, you need to login to your WandB account.
        You can do this by running `wandb login` in your terminal.
    """

    def __init__(
        self,
        log_freq: Mapping[str, int],
        log_dir: str,
        project: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(log_freq)

        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "``wandb`` package not found. Please install wandb using `pip install wandb` to use WandBLogger."
            ) from e
        self._wandb = wandb

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        rank = _get_rank()
        if self._wandb.run is None and rank == 0:
            # we check if wandb.init got called externally
            run = self._wandb.init(
                project=project,
                entity=entity,
                group=group,
                dir=log_dir,
                **kwargs,
            )

        if self._wandb.run:
            self._wandb.run._label(repo="torchtune")

        # define default x-axis (for latest wandb versions)
        if getattr(self._wandb, "define_metric", None):
            self._wandb.define_metric("step")
            self._wandb.define_metric("*", step_metric="step", step_sync=True)

        self.config_allow_val_change = kwargs.get("allow_val_change", False)

    def _log(self, name: str, data: Scalar, step: int) -> None:
        if self._wandb.run:
            self._wandb.log({name: data, "step": step})

    def _log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self._wandb.run:
            self._wandb.log({**payload, "step": step})

    def close(self) -> None:
        if hasattr(self, "_wandb") and self._wandb.run:
            self._wandb.finish()


# TODO: replace with direct instantiation via a path to the class in the config
METRIC_LOGGER_STR_TO_CLS = {
    "stdout": StdoutLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandBLogger,
}


def _get_rank():
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
