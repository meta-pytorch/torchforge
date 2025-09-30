# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

__all__ = [
    "TuneIterableDataset",
    "InfiniteTuneIterableDataset",
    "InterleavedDataset",
    "DatasetInfo",
]


@dataclass(frozen=True)
class DatasetInfo:
    """Hierarchical metadata for datasets, enabling composition and weight tracking.

    Used to build tree structures when composing datasets. For example, a nested
    InterleavedDataset dataset would have this structure:

    Example::

        DatasetInfo(name='parent_interleaved',
            weight=1.0,
            children=(DatasetInfo(name='child_interleaved',
                                  weight=0.7,
                                  children=(DatasetInfo(name='dataset_a',
                                                        weight=0.6,
                                                        children=()),
                                            DatasetInfo(name='dataset_b',
                                                        weight=0.4,
                                                        children=()))),
                      DatasetInfo(name='dataset_c', weight=0.3, children=())))

    This hierarchical structure is used for validation (ensuring unique dataset
    names) and for logging metrics.

    Attributes:
        name (str): Unique identifier for the dataset
        weight (float): Sampling weight for dataset selection (default: 1.0)
        children (tuple[DatasetInfo, ...]): Nested datasets for composed structures
    """

    name: str
    weight: float = 1.0
    children: tuple["DatasetInfo", ...] = field(default_factory=tuple)


class TuneIterableDataset(IterableDataset, ABC):
    """Base class for all torchtune iterable datasets.

    Datasets are composable, enabling complex structures such as:
    ``PackedDataset(InterleavedDataset([InterleavedDataset([ds1, ds2]), ds3]))``

    Each dataset implementation must:
    - Track hierarchical metadata via the ``info`` property
    - Handle checkpointing: parents resume children's state
    """

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Returns a hierarchical structure of all dataset information, including
        this dataset and its children."""
        pass

    def _validate_unique_dataset_names(self) -> None:
        """Traverses the DatasetInfo tree and raises ValueError on duplicate names."""
        root_info = self.info
        names = []
        to_process = [root_info]

        while to_process:
            node = to_process.pop(0)
            names.append(node.name)
            to_process.extend(node.children)

        # Check for duplicates after traversing the whole tree
        duplicates = [name for name in set(names) if names.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate dataset names found in hierarchy: {duplicates=}, all names={names}"
            )

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Returns an iterator over the dataset. Each implementation is responsible
        for its own iteration logic, including shuffling, distribution of data across ranks,
        and making it an infinite stream."""
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Returns checkpoint state for dataset resumption."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores dataset state from checkpoint."""
        pass


class InfiniteTuneIterableDataset(TuneIterableDataset):
    """Base class for infinite datasets that never exhaust.

    Prevents distributed training hangs by ensuring all ranks always
    have data available. Datasets restart from beginning when exhausted.
    """

    pass


class InterleavedDataset(InfiniteTuneIterableDataset):
    """Infinitely interleaves multiple datasets according to their sampling weights.

    The weights are extracted from each dataset's ``info.weight`` property and
    normalized to sum to 1.0. This dataset manages the state of its child
    datasets to ensure correct checkpointing and resumption.

    Args:
        datasets (list[InfiniteTuneIterableDataset]): A list of datasets to interleave.
        seed (int): The seed for sampling.
        weight (float): The weight for this dataset. Defaults to 1.0.
        dataset_name (str): The name of the dataset. Defaults to "interleaved_dataset".
        sampling_log_maxlen (int): The maximum length of the sampling log. This gets dumped to the
            checkpoint and can be used for debugging and analysis. Defaults to 10000.
    """

    def __init__(
        self,
        datasets: list[InfiniteTuneIterableDataset],
        seed: int,
        weight: float = 1.0,
        dataset_name: str = "interleaved_dataset",
        sampling_log_maxlen: int = 10000,
    ):
        self._datasets = sorted(datasets, key=lambda ds: ds.info.name)
        self._sampling_log_maxlen = sampling_log_maxlen

        # Build the hierarchical info object for this dataset
        self._info = DatasetInfo(
            name=dataset_name,
            weight=weight,
            children=tuple(ds.info for ds in self._datasets),
        )

        # Validate the entire hierarchy using the base class method
        self._validate_unique_dataset_names()

        # Extract weights from direct children and normalize them
        child_weights = [info.weight for info in self._info.children]
        total_weight = sum(child_weights)
        if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
            logger.warning(
                f"Interleaved dataset normalized weights to sum to 1.0. "
                f"Previous weights={child_weights}, "
                f"new weights={[w / total_weight for w in child_weights]}"
            )
        self._normalized_weights = torch.tensor(
            [w / total_weight for w in child_weights], dtype=torch.float
        )

        # Track sampling decisions for debugging and analysis
        self._sampling_log: deque[tuple[int, str]] = deque(
            maxlen=self._sampling_log_maxlen
        )
        self._iteration_count = 0
        self._sampling_generator = torch.Generator().manual_seed(seed)

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Interleave samples from child infinite datasets"""
        # Create a dictionary of iterators for each child dataset
        child_iters = {ds.info.name: iter(ds) for ds in self._datasets}

        while True:
            # Sample a child dataset based on the normalized weights
            ds_idx: int = torch.multinomial(
                self._normalized_weights,
                1,
                replacement=True,
                generator=self._sampling_generator,
            ).item()

            selected_ds = self._datasets[ds_idx]
            ds_name = selected_ds.info.name

            # Log
            self._sampling_log.append((self._iteration_count, ds_name))
            self._iteration_count += 1

            # Yield the next sample from the selected child iterator
            yield next(child_iters[ds_name])

    def state_dict(self) -> dict[str, Any]:
        """Save interleaver state and all children dataset states."""
        # The parent is responsible for namespacing the child states
        child_states = {ds.info.name: ds.state_dict() for ds in self._datasets}
        return {
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states,
            "sampling_log": list(self._sampling_log),
            "iteration_count": self._iteration_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state for the interleaver and its children."""
        self._sampling_generator.set_state(state_dict["sampling_generator_state"])
        child_states = state_dict["child_states"]

        for ds in self._datasets:
            ds.load_state_dict(child_states[ds.info.name])

        # Load sampling log and iteration count
        self._sampling_log = deque(
            state_dict.get("sampling_log", []), maxlen=self._sampling_log_maxlen
        )
        self._iteration_count = state_dict.get("iteration_count", 0)
