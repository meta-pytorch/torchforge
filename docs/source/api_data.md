# Data Management

Comprehensive data handling utilities for training and inference, including datasets, data models, and various data processing utilities.

## Overview

```{eval-rst}
.. automodule:: forge.data
   :members:
   :undoc-members:
   :show-inheritance:
```

## Datasets

The datasets module provides various dataset implementations for different training scenarios.

### Base Dataset Module

```{eval-rst}
.. automodule:: forge.data.datasets
   :members:
   :undoc-members:
   :show-inheritance:
```

### Dataset Implementation

```{eval-rst}
.. automodule:: forge.data.datasets.dataset
   :members:
   :undoc-members:
   :show-inheritance:
```

### Hugging Face Dataset

Integration with Hugging Face datasets for seamless data loading.

```{eval-rst}
.. automodule:: forge.data.datasets.hf_dataset
   :members:
   :undoc-members:
   :show-inheritance:
```

### Packed Dataset

Efficient packed dataset implementation for optimized training.

```{eval-rst}
.. automodule:: forge.data.datasets.packed
   :members:
   :undoc-members:
   :show-inheritance:
```

### SFT Dataset

Supervised Fine-Tuning specific dataset implementation.

```{eval-rst}
.. automodule:: forge.data.datasets.sft_dataset
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data Models

Core data structures used throughout the training pipeline.

### Base Data Models

```{eval-rst}
.. automodule:: forge.data_models
   :members:
   :undoc-members:
   :show-inheritance:
```

### Completion

Data model for model completions and responses.

```{eval-rst}
.. automodule:: forge.data_models.completion
   :members:
   :undoc-members:
   :show-inheritance:
```

### Episode

Data model for training episodes in reinforcement learning.

```{eval-rst}
.. automodule:: forge.data_models.episode
   :members:
   :undoc-members:
   :show-inheritance:
```

### Prompt

Data model for input prompts and contexts.

```{eval-rst}
.. automodule:: forge.data_models.prompt
   :members:
   :undoc-members:
   :show-inheritance:
```

### Scored Completion

Data model for completions with associated scores and rewards.

```{eval-rst}
.. automodule:: forge.data_models.scored_completion
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data Utilities

Various utilities for data processing, tokenization, and manipulation.

### Collation

Data collation utilities for batching and padding.

```{eval-rst}
.. automodule:: forge.data.collate
   :members:
   :undoc-members:
   :show-inheritance:
```

### Rewards

Reward computation and processing utilities.

```{eval-rst}
.. automodule:: forge.data.rewards
   :members:
   :undoc-members:
   :show-inheritance:
```

### Sharding

Data sharding utilities for distributed training.

```{eval-rst}
.. automodule:: forge.data.sharding
   :members:
   :undoc-members:
   :show-inheritance:
```

### Tokenizer

Tokenization utilities and tokenizer management.

```{eval-rst}
.. automodule:: forge.data.tokenizer
   :members:
   :undoc-members:
   :show-inheritance:
```

### General Utilities

General data processing and utility functions.

```{eval-rst}
.. automodule:: forge.data.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## Dataset Metrics

Advanced metrics and monitoring for dataset performance.

### Base Metrics

```{eval-rst}
.. automodule:: forge.data.dataset_metrics
   :members:
   :undoc-members:
   :show-inheritance:
```

### Metric Aggregation Handlers

```{eval-rst}
.. automodule:: forge.data.dataset_metrics.metric_agg_handlers
   :members:
   :undoc-members:
   :show-inheritance:
```

### Metric Aggregator

```{eval-rst}
.. automodule:: forge.data.dataset_metrics.metric_aggregator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Metric Transform

```{eval-rst}
.. automodule:: forge.data.dataset_metrics.metric_transform
   :members:
   :undoc-members:
   :show-inheritance:
```
