# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

from forge.data.transforms import Transform


@dataclass(frozen=True)
class Metric:
    source: str
    metric_name: str
    value: Union[int, float, str]
    agg_type: "AggregationType"


class AggregationType(Enum):
    """Defines how a metric's value should be aggregated by the MetricsAggregator.

    Each type corresponds to a specific AggregationHandler that implements the logic
    for initialization, updates, and distributed reduction.
    """

    SUM = "sum"
    MEAN = "mean"
    STATS = "distribution"
    CATEGORICAL_COUNT = "categorical_count"
    MAX = "max"
    MIN = "min"


class MetricTransform(Transform, ABC):
    """Applied to each dataset sample to generate per-sample metrics for training tracking.

    Creates Metric objects that are later aggregated by MetricsAggregator. This separation
    of concerns ensures metrics are correctly aggregated even with multiple dataloader
    workers and in distributed settings.

    The transform must be configured with a source via set_source() before use.
    Each call to __call__ adds metrics to the sample's "metrics" key.

    Example:
        >>> transform = DefaultTrainingMetricTransform()
        >>> transform.set_source("alpaca")
        >>> sample = {"tokens": [1, 2, 3]}
        >>> result = transform(sample)
        >>> # result["metrics"] contains list of Metric objects
    """

    def set_source(self, source: str) -> None:
        """Called by the dataset to set the namespace for metrics.

        This is used to differentiate metrics from multiple datasets, for example,
        "alpaca/tokens_seen" vs. "slim_orca/tokens_seen".

        Args:
            source (str): Name of the dataset, used for logging and disambiguation.
        """
        self.source = source

    @abstractmethod
    def _generate_metrics(self, sample: dict[str, Any]) -> list[Metric]:
        """Generate metrics for a single sample.

        Args:
            sample (dict[str, Any]): The sample dictionary to generate metrics from

        Returns:
            list[Metric]: List of metrics generated for this sample

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement _generate_metrics method")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self, "source"):
            raise RuntimeError(
                "'transform.set_source' must be called before using the transform."
            )

        # Generate metrics for this sample
        metrics = self._generate_metrics(sample)

        # Add to existing metrics list or create new one
        if "metrics" not in sample:
            sample["metrics"] = []
        sample["metrics"].extend(metrics)
        return sample


class DefaultTrainingMetricTransform(MetricTransform):
    """Generates common training metrics: samples seen, tokens seen, and sequence length.

    This transform detects the token key in a sample, checking for "tokens"
    first and then falling back to "input_ids".

    For details on the base class behavior, see MetricTransform.

    Tracked metrics:
      - samples_seen: Cumulative count of samples processed (SUM aggregation)
      - tokens_seen: Cumulative sum of all tokens processed (SUM aggregation)
      - seq_len: Distribution stats of sequence lengths (STATS aggregation)

    Example:
        >>> transform = DefaultTrainingMetricTransform()
        >>> transform.set_source("alpaca")
        >>>
        >>> sample = {"tokens": [1, 2, 3, 4, 5]}  # 5 tokens
        >>> metrics = transform._generate_metrics(sample)
        >>> # This generates the following Metric objects:
        >>> # [
        >>> #   Metric(source="alpaca", metric_name="samples_seen", value=1, agg_type=AggregationType.SUM),
        >>> #   Metric(source="alpaca", metric_name="tokens_seen", value=5, agg_type=AggregationType.SUM),
        >>> #   Metric(source="alpaca", metric_name="seq_len", value=5, agg_type=AggregationType.STATS)
        >>> # ]
    """

    def _generate_metrics(self, sample: dict[str, Any]) -> list[Metric]:
        # Determine token key
        token_key = "tokens" if "tokens" in sample else "input_ids"
        token_len = len(sample.get(token_key, []))

        # Create metrics for this sample
        return [
            Metric(
                source=self.source,
                metric_name="samples_seen",
                value=1,
                agg_type=AggregationType.SUM,
            ),
            Metric(
                source=self.source,
                metric_name="tokens_seen",
                value=token_len,
                agg_type=AggregationType.SUM,
            ),
            Metric(
                source=self.source,
                metric_name="seq_len",
                value=token_len,
                agg_type=AggregationType.STATS,
            ),
        ]
