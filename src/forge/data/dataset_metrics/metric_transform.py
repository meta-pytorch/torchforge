# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from forge.interfaces import Transform
from forge.observability.metrics import Metric, Reduce


class MetricTransform(Transform):
    """
    Base class for metric transforms that collect metrics from dataset samples.

    Uses the new observability system instead of the old dataset_metrics system.
    Metrics are collected as observability.Metric objects and will be recorded
    in the main process using record_metric().
    """

    def __init__(self):
        self.source = None

    def set_source(self, source: str):
        """Set the source name for metrics (typically the dataset name)."""
        self.source = source

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Transform a sample by adding metrics to it."""
        return sample


class DefaultTrainingMetricTransform(MetricTransform):
    """
    Default metric transform that collects standard training metrics.

    Collects:
    - samples_seen: Count of samples processed (SUM reduction)
    - tokens_seen: Count of tokens processed (SUM reduction)
    - seq_len: Sequence length (MEAN, MAX, MIN reductions for different stats)

    Uses observability.Metric objects instead of the old system.
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if "metrics" not in sample:
            sample["metrics"] = []

        source_name = self.source or "dataset"

        # Add samples_seen metric
        sample["metrics"].append(
            Metric(
                key=f"dataset/{source_name}/samples_seen",
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
                        key=f"dataset/{source_name}/tokens_seen",
                        value=token_count,
                        reduction=Reduce.SUM,
                    ),
                    Metric(
                        key=f"dataset/{source_name}/seq_len_mean",
                        value=token_count,
                        reduction=Reduce.MEAN,
                    ),
                    Metric(
                        key=f"dataset/{source_name}/seq_len_max",
                        value=token_count,
                        reduction=Reduce.MAX,
                    ),
                    Metric(
                        key=f"dataset/{source_name}/seq_len_min",
                        value=token_count,
                        reduction=Reduce.MIN,
                    ),
                ]
            )

        return sample
