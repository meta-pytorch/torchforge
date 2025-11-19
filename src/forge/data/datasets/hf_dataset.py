# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from forge.data.metric_transform import DefaultDatasetMetricTransform, MetricTransform
from forge.observability.metrics import Metric, Reduce

from .dataset import DatasetInfo, InfiniteTuneIterableDataset

logger = logging.getLogger(__name__)


class HfIterableDataset(InfiniteTuneIterableDataset):
    """HuggingFace dataset with infinite iteration and composable transforms.

    Transform pipeline: raw_data -> message_transform -> model_transform -> output_transform -> metric_transform

    This dataset is responsible for:
      - Loading and sharding the dataset
      - Shuffling at initialization and after each epoch
      - Applying transforms to the data
      - Returning an infinite iterator over the dataset

    Args:
        message_transform (Callable | None): Transforms raw data into a `Message`.
        model_transform (Callable | None): Prepares messages for the model,
            usually by tokenizing them.
        output_transform (Callable | None): Prepares tokenized inputs for the
            recipe, often by manipulating labels (e.g., setting an ignore index).
            This transform is recipe-dependent (e.g., SFT, DPO, etc.).
        metric_transform (MetricTransform | None): Computes metrics from a
            sample (e.g., token count). If ``None``, a default transform is used.
            To disable standard metric tracking, set this to ``lambda x: x``.
        shuffle_buffer_size (int | None): Size of the shuffle buffer.
            If ``None`` or 0, no shuffling is performed.
        weight (float | None): Weight for this dataset. Defaults to 1.0.
        seed (int): Seed for shuffling.
        num_shards_per_rank (int): The target number of shards per worker (GPU).
            The actual number of shards will be a multiple of
            ``world_size * dataloader_workers``.
        dataset_name (str | None): Name of the dataset. If ``None``, a name is
            generated from the ``path``, ``source``, and ``split``.
        filter_fn (Callable | None): A function to filter the dataset.
        filter_kwargs (dict[str, Any] | None): Keyword arguments for ``filter_fn``.
        **load_dataset_kwargs: Keyword arguments for the
            :func:`~datasets.load_dataset` function.
    """

    def __init__(
        self,
        *,
        message_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        model_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        output_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        metric_transform: MetricTransform | None = None,
        shuffle_buffer_size: int | None = 1000,
        weight: float | None = 1.0,
        seed: int = 42,
        num_shards_per_rank: int = 64,
        dataset_name: str | None = None,
        filter_fn: Callable | None = None,
        filter_kwargs: dict[str, Any] | None = None,
        dp_mesh: dist.ProcessGroup | None = None,
        **load_dataset_kwargs,
    ):
        # Store configuration
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._message_transform = message_transform
        self._model_transform = model_transform
        self._output_transform = output_transform
        self._weight = weight if weight is not None else 1.0
        self._dp_mesh = dp_mesh

        # Create default transform if not provided
        self._metric_transform = metric_transform or DefaultDatasetMetricTransform()

        # Auto-generate dataset name if not provided
        if dataset_name is None:
            path = load_dataset_kwargs.get("path", None)
            source = load_dataset_kwargs.get("source", None)
            split = load_dataset_kwargs.get("split", None)
            name_parts = []
            for item in [path, source, split]:
                if item is not None:
                    name_parts.append(str(item).replace("/", "_"))
            dataset_name = "_".join(name_parts)

        # Build info object for this dataset
        self._info = DatasetInfo(name=dataset_name, weight=self._weight)

        # Set dataset name on the transform if it supports it
        if hasattr(self._metric_transform, "set_source"):
            self._metric_transform.set_source(dataset_name)

        # Internal state for resumption
        # _start_epoch: The epoch to start from. Updated on resume from ckpt.
        # useful when doing iter(ds), which restarts dataset from original state.
        self._start_epoch = 0
        # _num_epochs: updated on every dataset exhaustion
        self._num_epochs = 0

        # Load and setup HF dataset
        self._setup_hf_dataset(
            load_dataset_kwargs, num_shards_per_rank, filter_fn, filter_kwargs
        )

    @property
    def info(self) -> DatasetInfo:
        """Returns info for this leaf dataset, which has no children."""
        return self._info

    def _apply_transforms(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transforms if they exist, otherwise return sample unchanged."""
        if self._message_transform is not None:
            sample = self._message_transform(sample)
        if self._model_transform is not None:
            sample = self._model_transform(sample)
        if self._output_transform is not None:
            sample = self._output_transform(sample)
        if self._metric_transform is not None:
            sample = self._metric_transform(sample)
        return sample

    def _setup_hf_dataset(
        self,
        load_dataset_kwargs: dict[str, Any],
        num_shards_per_rank: int,
        filter_fn: Callable | None = None,
        filter_kwargs: dict[str, Any] | None = None,
    ):
        """
        One-time setup of HuggingFace dataset that handles Handles distributed sharding,
        shuffle configuration, and filtering. Called once during __init__.
        """

        # Extract rank/world_size from DP mesh
        world_size, rank = 1, 0
        if self._dp_mesh is not None:
            world_size = dist.get_world_size(group=self._dp_mesh)
            rank = dist.get_rank(group=self._dp_mesh)
            logger.debug(
                f"Using DP mesh for sharding: rank={rank}, world_size={world_size}"
            )
        elif dist.is_initialized():
            # Fallback to global rank (may not respect TP/PP)
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # TODO: is there a way to detect this and raise error instead?
            logger.warning(
                f"Using global rank for sharding: rank={rank}, world_size={world_size}. "
                f"If using other types of parallelsim (CP/TP/PP), pass dp_mesh for correct sharding."
            )

        # Load and shard dataset
        ds = load_dataset(**load_dataset_kwargs)

        # Use to_iterable_dataset for non-streaming datasets
        is_streaming = load_dataset_kwargs.get("streaming", False)
        if is_streaming:
            logger.warning(
                f"Streaming datasets were not yet tested for distributed training. "
                f"Dataset '{self.info.name}' has "
                f"{getattr(ds, 'num_shards', 'unknown')} shards, and your training has {world_size} ranks."
                f"See: https://huggingface.co/docs/datasets/en/package_reference/main_classes?#datasets.IterableDataset.shard"
                f"Consider setting streaming=False, which should also be faster."
            )
        if not is_streaming:
            # Define number of shards based on (world_size, num of shards per GPU, dataloader workers)
            # E.g. world_size=2, num_shards_per_rank=16, dataloader_workers=3
            # we will try 2*16 = 32 shards. Since 32 is not a multiple of 2*3=6, we will do 36 shards.
            # Each rank gets 18 shards, each dataloader worker in that rank gets 6 shards.
            worker_info = torch.utils.data.get_worker_info()
            num_dataloader_workers = worker_info.num_workers if worker_info else 1

            # Calculate total workers across all ranks and dataloader processes
            total_workers = world_size * num_dataloader_workers

            # Find minimum shards that satisfies our target while being divisible by workers
            desired_shards = world_size * num_shards_per_rank

            # Round up to next multiple of total_workers for even distribution
            if desired_shards % total_workers == 0:
                num_shards = desired_shards
            else:
                num_shards = total_workers * (
                    (desired_shards + total_workers - 1) // total_workers
                )

            # If the dataset has a defined length,
            # assert num_shards < dataset_size.
            if hasattr(ds, "__len__"):
                dataset_size = len(ds)
                if num_shards > dataset_size:
                    raise ValueError(
                        f"Number of shards ({num_shards}) is greater than the dataset size ({dataset_size})."
                        f"Please decrease one of {num_shards_per_rank=} or dataloader.num_workers={num_dataloader_workers}"
                    )

            ds = ds.to_iterable_dataset(num_shards=num_shards)

        # Shuffle the dataset
        # We shuffle after sharding and before splitting so shards can be shuffled well
        if self._shuffle_buffer_size and self._shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=self._seed, buffer_size=self._shuffle_buffer_size)

        # Distribute across ranks
        if world_size > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

        # Apply filtering if specified
        if filter_fn:
            filter_kwargs = filter_kwargs or {}
            ds = ds.filter(filter_fn, **filter_kwargs)

        self._ds = ds

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Infinite iteration over dataset samples.

        Behavior:
        - Restarts from beginning when dataset is exhausted
        - Reshuffles at start of each epoch (if enabled)
        - Applies full transform pipeline to each sample
        - Adds 'num_epochs' metric to track dataset progress
        - Yields samples indefinitely for continuous training
        """
        self._num_epochs = self._start_epoch

        while True:  # Infinite iteration
            self._ds.set_epoch(self._num_epochs)
            epoch_iterator = iter(self._ds)
            samples_yielded = 0

            try:
                for sample in epoch_iterator:
                    # NOTE: We apply transforms here instead of using .map() to work around
                    # HuggingFace datasets bug where .map() causes incorrect checkpoint resumption.
                    # See: https://github.com/huggingface/datasets/issues/7630
                    # .map is applied lazily and the advantage would be to leverage caching.
                    sample = self._apply_transforms(sample)

                    # Track the number of epochs completed for each dataset. This is
                    # especially useful when interleaving multiple datasets, but
                    # also necessary to track dataset-level metrics.
                    if "metrics" not in sample:
                        sample["metrics"] = []

                    sample["metrics"].append(
                        Metric(
                            key=f"dataset/{self.info.name}/num_epochs",
                            value=self._num_epochs,
                            reduction=Reduce.MAX,
                        )
                    )

                    samples_yielded += 1
                    yield sample

            except StopIteration:
                # Expected when dataset is exhausted
                pass
            except Exception as e:
                logger.error(
                    f"Dataset {self.info.name} encountered an unexpected error: {e}."
                )
                raise

            # Check if we got zero samples - this might indicate an issue
            if samples_yielded == 0:
                logger.warning(
                    f"Dataset {self.info.name} epoch {self._num_epochs} yielded 0 samples - potential issue!"
                )

            # Epoch complete - increment and continue infinite loop
            self._num_epochs += 1

    def state_dict(self) -> dict[str, Any]:
        hf_state = self._ds.state_dict()
        state = {
            "num_epochs": self._num_epochs,
            "hf_dataset_state": hf_state,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._start_epoch = state_dict["num_epochs"]
        hf_state = state_dict["hf_dataset_state"]

        # HF is responsible for resuming the dataset state
        # where it last left off
        self._ds.load_state_dict(hf_state)
