# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from abc import ABC, abstractmethod
from typing import Mapping, Optional

import torch

from forge.types import Scalar


def get_metric_logger(logger: str = "stdout", **log_config):
    return METRIC_LOGGER_STR_TO_CLS[logger](**log_config)


class MetricLogger(ABC):
    """Abstract metric logger.

    Args:
        log_freq (int): calls to `log` and `log_dict` will be ignored if `step % log_freq != 0`
    """

    def __init__(self, log_freq: int):
        self._log_freq = log_freq
        self._step = None

    def set_step(self, step: int) -> None:
        """Subsequent log calls will use this step number by default if not provided to the log call."""
        self._step = step

    def is_log_step(self, step: Optional[int] = None):
        """Returns true if the current step is a logging step.

        Args:
            step (int): current step. if not given, will use the one last provided via set_step()
        """
        if step is None:
            assert (
                self._step is not None
            ), "`step` arg required if `set_step` has not been called."
            step = self._step
        return step % self._log_freq == 0

    def log(
        self,
        name: str,
        data: Scalar,
        step: Optional[int] = None,
    ) -> None:
        """Log scalar data if this is a logging step.

        Args:
            name (str): tag name used to group scalars
            data (Scalar): scalar data to log
            step (int): step value to record. if not given, will use the one last provided via set_step()
        """
        if step is None:
            assert (
                self._step is not None
            ), "`step` arg required if `set_step` has not been called."
            step = self._step
        if step % self._log_freq == 0:
            self._log(name, data, step)

    def log_dict(
        self, payload: Mapping[str, Scalar], step: Optional[int] = None
    ) -> None:
        """Log multiple scalar values if this is a logging step.

        Args:
            payload (Mapping[str, Scalar]): dictionary of tag name and scalar value
            step (int): step value to record. if not given, will use the one last provided via set_step()
        """
        if step is None:
            assert (
                self._step is not None
            ), "`step` arg required if `set_step` has not been called."
            step = self._step
        if step % self._log_freq == 0:
            self._log_dict(payload, step)

    @abstractmethod
    def _log(self, name: str, data: Scalar, step: int) -> None:
        """Log scalar data.

        Args:
            name (str): tag name used to group scalars
            data (Scalar): scalar data to log
            step (int): step value to record
        """
        pass

    @abstractmethod
    def _log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values.

        Args:
            payload (Mapping[str, Scalar]): dictionary of tag name and scalar value
            step (int): step value to record
        """
        pass

    def close(self) -> None:
        """
        Close log resource, flushing if necessary.
        Logs should not be written after `close` is called.
        """
        pass


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
        log_dir: str,
        project: str = "torchforge",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        **kwargs,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "``wandb`` package not found. Please install wandb using `pip install wandb` to use WandBLogger."
            ) from e
        self._wandb = wandb

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
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

    def __del__(self) -> None:
        # extra check for when there is an import error
        if hasattr(self, "_wandb") and self._wandb.run:
            self._wandb.finish()

    def close(self) -> None:
        if self._wandb.run:
            self._wandb.finish()


METRIC_LOGGER_STR_TO_CLS = {
    "wandb": WandBLogger,
}
