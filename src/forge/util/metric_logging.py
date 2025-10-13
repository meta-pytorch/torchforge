# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from typing import Mapping, Union

from forge.interfaces import MetricLogger
from forge.types import Scalar
from forge.util.distributed import get_world_size_and_rank


def get_metric_logger(logger: str = "stdout", **log_config):
    return METRIC_LOGGER_STR_TO_CLS[logger](**log_config)


class StdoutLogger(MetricLogger):
    """Logger to standard output.

    Args:
        freq (Union[int, Mapping[str, int]]):
            If int, all metrics will be logged at this frequency.
            If Mapping, calls to `log` and `log_dict` will be ignored if `step % freq[metric_name] != 0`
    """

    def __init__(self, freq: Union[int, Mapping[str, int]]):
        self._freq = freq

    def is_log_step(self, name: str, step: int) -> bool:
        """Returns true if the current step is a logging step.

        Args:
            name (str): metric name (for checking the freq for this metric)
            step (int): current step
        """
        if isinstance(self._freq, int):
            return step % self._freq == 0
        return step % self._freq[name] == 0

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Log the metric if it is a logging step.

        Args:
            name (str): metric name
            data (Scalar): metric value
            step (int): current step
        """
        if not self.is_log_step(name, step):
            return
        print(f"Step {step} | {name}:{data}")

    def log_dict(self, metrics: Mapping[str, Scalar], step: int) -> None:
        """Log the metrics for which this is currently a logging step.

        Args:
            metrics (Mapping[str, Scalar]): dict of metric names and values
            step (int): current step
        """
        log_step_metrics = {
            name: value
            for name, value in metrics.items()
            if self.is_log_step(name, step)
        }
        if not log_step_metrics:
            return

        print(f"Step {step} | ", end="")
        for name, data in log_step_metrics.items():
            print(f"{name}:{data} ", end="")
        print("\n", end="")

    def close(self) -> None:
        sys.stdout.flush()


class TensorBoardLogger(MetricLogger):
    """Logger for use w/ PyTorch's implementation of TensorBoard (https://pytorch.org/docs/stable/tensorboard.html).

    Args:
        freq (Union[int, Mapping[str, int]]):
            If int, all metrics will be logged at this frequency.
            If Mapping, calls to `log` and `log_dict` will be ignored if `step % freq[metric_name] != 0`
        log_dir (str): torch.TensorBoard log directory
        organize_logs (bool): If `True`, this class will create a subdirectory within `log_dir` for the current
            run. Having sub-directories allows you to compare logs across runs. When TensorBoard is
            passed a logdir at startup, it recursively walks the directory tree rooted at logdir looking for
            subdirectories that contain tfevents data. Every time it encounters such a subdirectory,
            it loads it as a new run, and the frontend will organize the data accordingly.
            Recommended value is `True`. Run `tensorboard --logdir my_log_dir` to view the logs.
        **kwargs: additional arguments

    Example:
        >>> from forge.util.metric_logging import TensorBoardLogger
        >>> logger = TensorBoardLogger(freq={"loss": 10}, log_dir="my_log_dir")
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
        freq: Union[int, Mapping[str, int]],
        log_dir: str = "metrics_log",
        organize_logs: bool = True,
        **kwargs,
    ):
        from torch.utils.tensorboard import SummaryWriter

        self._freq = freq
        self._writer: SummaryWriter | None = None
        _, rank = get_world_size_and_rank()

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

    def is_log_step(self, name: str, step: int) -> bool:
        """Returns true if the current step is a logging step.

        Args:
            name (str): metric name (for checking the freq for this metric)
            step (int): current step
        """
        if isinstance(self._freq, int):
            return step % self._freq == 0
        return step % self._freq[name] == 0

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Log the metric if it is a logging step.

        Args:
            name (str): metric name
            data (Scalar): metric value
            step (int): current step
        """
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_dict(self, metrics: Mapping[str, Scalar], step: int) -> None:
        """Log the metrics for which this is currently a logging step.

        Args:
            metrics (Mapping[str, Scalar]): dict of metric names and values
            step (int): current step
        """
        for name, data in metrics.items():
            if self.is_log_step(name, step):
                self.log(name, data, step)

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


class WandBLogger(MetricLogger):
    """Logger for use w/ Weights and Biases application (https://wandb.ai/).
    For more information about arguments expected by WandB, see https://docs.wandb.ai/ref/python/init.

    Args:
        freq (Union[int, Mapping[str, int]]):
            If int, all metrics will be logged at this frequency.
            If Mapping, calls to `log` and `log_dict` will be ignored if `step % freq[metric_name] != 0`
        log_dir (str | None): WandB log directory.
        project (str): WandB project name. Default is `torchtune`.
        entity (str | None): WandB entity name. If you don't specify an entity,
            the run will be sent to your default entity, which is usually your username.
        group (str | None): WandB group name for grouping runs together. If you don't
            specify a group, the run will be logged as an individual experiment.
        **kwargs: additional arguments to pass to wandb.init

    Example:
        >>> from forge.util.metric_logging import WandBLogger
        >>> logger = WandBLogger(freq={"loss": 10}, log_dir="wandb", project="my_project")
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
        freq: Union[int, Mapping[str, int]],
        project: str,
        log_dir: str = "metrics_log",
        entity: str | None = None,
        group: str | None = None,
        **kwargs,
    ):
        self._freq = freq

        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "``wandb`` package not found. Please install wandb using `pip install wandb` to use WandBLogger."
            ) from e
        self._wandb = wandb

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        _, rank = get_world_size_and_rank()
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
            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("step")
                self._wandb.define_metric("*", step_metric="step", step_sync=True)

    def is_log_step(self, name: str, step: int) -> bool:
        """Returns true if the current step is a logging step.

        Args:
            name (str): metric name (for checking the freq for this metric)
            step (int): current step
        """
        if isinstance(self._freq, int):
            return step % self._freq == 0
        return step % self._freq[name] == 0

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Log the metric if it is a logging step.

        Args:
            name (str): metric name
            data (Scalar): metric value
            step (int): current step
        """
        if self._wandb.run and self.is_log_step(name, step):
            self._wandb.log({name: data, "step": step})

    def log_dict(self, metrics: Mapping[str, Scalar], step: int) -> None:
        """Log the metrics for which this is currently a logging step.

        Args:
            metrics (Mapping[str, Scalar]): dict of metric names and values
            step (int): current step
        """
        log_step_metrics = {
            name: value
            for name, value in metrics.items()
            if self.is_log_step(name, step)
        }
        if not log_step_metrics:
            return

        if self._wandb.run:
            self._wandb.log({**metrics, "step": step})

    def close(self) -> None:
        if hasattr(self, "_wandb") and self._wandb.run:
            self._wandb.finish()


# TODO: replace with direct instantiation via a path to the class in the config
METRIC_LOGGER_STR_TO_CLS = {
    "stdout": StdoutLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandBLogger,
}
