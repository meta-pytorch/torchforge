# Data Management

Comprehensive data handling utilities for training and inference, including datasets, data models, and various data processing utilities.

## Data Processing

```{eval-rst}
.. currentmodule:: forge.data

.. autosummary::
   :toctree: generated/
   :nosignatures:

   collate_packed
```

## Datasets

```{eval-rst}
.. currentmodule:: forge.data.datasets

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DatasetInfo
   HfIterableDataset
   InterleavedDataset
   InfiniteTuneIterableDataset
   PackedDataset
   SFTOutputTransform
   sft_iterable_dataset
```

## Data Models

```{eval-rst}
.. currentmodule:: forge.data_models

.. autosummary::
   :toctree: generated/
   :nosignatures:

   completion.Completion
   episode.Episode
   prompt.Message
   prompt.Prompt
   scored_completion.ScoredCompletion
```

## Dataset Metrics

```{eval-rst}
.. currentmodule:: forge.data.dataset_metrics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AggregationType
   AggregationHandler
   CategoricalCountAggHandler
   DefaultTrainingMetricTransform
   StatsAggHandler
   MaxAggHandler
   MeanAggHandler
   Metric
   MetricState
   MetricsAggregator
   MetricTransform
   MinAggHandler
   SumAggHandler
```
