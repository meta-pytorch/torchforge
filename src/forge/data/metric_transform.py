# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from forge.observability.metrics import Metric, Reduce


class MetricTransform:
    """
    Base class for transforms that collect observability metrics from dataset samples.

    This class provides a foundation for implementing dataset-level metric collection
    during data processing pipelines. Subclasses should override the __call__ method
    to add specific metrics to each sample that passes through the transform.

    Metrics are collected as `forge.observability.metrics.Metric` objects and made available
    in batch["metrics"].

    Attributes:
        source (str, optional): The source name for metrics, typically the dataset name.
            This is used as a prefix in metric keys to distinguish metrics from different
            data sources.

    Example:
        >>> transform = SomeMetricTransform()
        >>> transform.set_source("training_data")
        >>> processed_sample = transform(sample)
        >>> # Metrics are automatically added to sample["metrics"]
    """

    def __init__(self):
        self.source = None

    def set_source(self, source: str):
        """Set the source name for metrics (typically the dataset name)."""
        self.source = source

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Transform a sample by adding metrics to it."""
        return sample


class DefaultDatasetMetricTransform(MetricTransform):
    """
    Collects basic dataset processing metrics during data pipeline execution.

    Metrics collected:
    - samples_processed: Total number of samples that have passed through this transform (SUM)
    - tokens_processed: Total number of tokens processed across all samples (SUM)
    - mean_seq_len: Average sequence length across samples (MEAN)
    - max_seq_len: Maximum sequence length observed (MAX)
    - min_seq_len: Minimum sequence length observed (MIN)

    Note: Token-related metrics are only collected if the sample contains a 'tokens' field.
    Sequence length is measured as the number of tokens in each sample.

    Example:
        >>> collector = DefaultDatasetMetricTransform()
        >>> collector.set_source("training_data")
        >>> sample = {"tokens": ["hello", "world"]}
        >>> processed_sample = collector(sample)
        >>> # Metrics are automatically added to processed_sample["metrics"]
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if "metrics" not in sample:
            sample["metrics"] = []

        source_name = self.source or "unnamed_ds"

        # Add samples_processed metric
        sample["metrics"].append(
            Metric(
                key=f"dataset/{source_name}/samples_processed",
                value=1,
                reduction=Reduce.SUM,
            )
        )

        # Add token-based metrics if tokens are present
        if "tokens" in sample:
            token_count = len(sample.get("tokens", []))

            sample["metrics"].extend(
                [
                    Metric(
                        key=f"dataset/{source_name}/tokens_processed",
                        value=token_count,
                        reduction=Reduce.SUM,
                    ),
                    Metric(
                        key=f"dataset/{source_name}/mean_seq_len",
                        value=token_count,
                        reduction=Reduce.MEAN,
                    ),
                    Metric(
                        key=f"dataset/{source_name}/max_seq_len",
                        value=token_count,
                        reduction=Reduce.MAX,
                    ),
                    Metric(
                        key=f"dataset/{source_name}/min_seq_len",
                        value=token_count,
                        reduction=Reduce.MIN,
                    ),
                ]
            )

        return sample
