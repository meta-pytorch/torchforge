# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for StopAfterOneEpoch iterator and extract_epoch_from_batch helper."""
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from forge.data.datasets import HfIterableDataset

from forge.data.utils import extract_epoch_from_batch, StopAfterOneEpoch
from forge.observability.metrics import Metric, Reduce

from tests.test_utils import gpu_test
from torch.testing._internal.common_fsdp import FSDPTest
from torchdata.stateful_dataloader import StatefulDataLoader


def create_test_json_file(path: Path, num_samples: int) -> None:
    """Create test data file with simple samples."""
    with open(path, "w") as f:
        for i in range(num_samples):
            f.write(f'{{"id": {i}, "tokens": [{i}, {i + 1}]}}\n')


def simple_collate(batch):
    """Simple collate function that mimics collate_packed behavior.

    Stacks tensors, extends metrics list, keeps other fields as lists.
    """
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
        elif key == "metrics":
            # Extend all metrics into a single list
            collated[key] = []
            for sample in batch:
                collated[key].extend(sample[key])
        else:
            collated[key] = [sample[key] for sample in batch]
    return collated


class TestExtractEpochFromBatch:
    """Test extract_epoch_from_batch helper function."""

    def test_extract_epoch_from_batch_success(self):
        """Test extracting epoch from valid batch with metrics."""
        batch = {
            "tokens": torch.tensor([1, 2, 3]),
            "metrics": [
                Metric(key="dataset/test/num_epochs", value=2, reduction=Reduce.MAX),
                Metric(
                    key="dataset/test/other_metric", value=42, reduction=Reduce.MEAN
                ),
            ],
        }
        epoch = extract_epoch_from_batch(batch)
        assert epoch == 2

    def test_extract_epoch_missing_metrics_field(self):
        """Test error when batch has no 'metrics' field."""
        batch = {"tokens": torch.tensor([1, 2, 3])}
        with pytest.raises(ValueError, match="Batch missing 'metrics' field"):
            extract_epoch_from_batch(batch)

    def test_extract_epoch_no_num_epochs_metric(self):
        """Test error when no num_epochs metric found."""
        batch = {
            "metrics": [
                Metric(
                    key="dataset/test/other_metric", value=42, reduction=Reduce.MEAN
                ),
            ]
        }
        with pytest.raises(ValueError, match="No 'num_epochs' metric found"):
            extract_epoch_from_batch(batch)


class TestStopAfterOneEpochSingleProcess:
    """Test StopAfterOneEpoch in single-process mode (no distributed)."""

    def test_stop_after_one_epoch(self, tmp_path):
        """Verify iterator stops after exactly one epoch completes."""
        # Create small dataset (10 samples)
        data_file = tmp_path / "data.json"
        create_test_json_file(data_file, num_samples=10)

        dataset = HfIterableDataset(
            path="json",
            data_files=str(data_file),
            split="train",
            shuffle_buffer_size=0,
            num_shards_per_rank=1,
        )

        dataloader = StatefulDataLoader(
            dataset, batch_size=2, collate_fn=simple_collate
        )

        # Wrap with StopAfterOneEpoch
        batch_iter = StopAfterOneEpoch(
            iter=iter(dataloader),
            device=torch.device("cpu"),
            dp_mesh=None,
        )

        # Collect all batches until StopIteration
        batches = []
        for batch in batch_iter:
            batches.append(batch)
            # Verify all batches are from epoch 0
            epoch = extract_epoch_from_batch(batch)
            assert epoch == 0, f"Expected epoch 0, got {epoch}"

        # Should have consumed exactly one epoch (5 batches of size 2)
        assert len(batches) == 5


class TestStopAfterOneEpochDistributed(FSDPTest):
    """Test StopAfterOneEpoch with distributed synchronization."""

    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_epoch_sync_across_ranks(self):
        """Verify all ranks stop when any rank detects epoch change."""
        import shutil
        import tempfile

        rank = dist.get_rank()
        temp_dir = tempfile.mkdtemp(prefix=f"stop_epoch_test_rank{rank}_")

        try:
            data_file = Path(temp_dir) / "data.json"
            # Create dataset with 20 samples, split across 2 ranks (10 each)
            create_test_json_file(data_file, num_samples=20)

            dataset = HfIterableDataset(
                path="json",
                data_files=str(data_file),
                split="train",
                shuffle_buffer_size=0,
                num_shards_per_rank=1,
            )

            dataloader = StatefulDataLoader(
                dataset, batch_size=2, collate_fn=simple_collate
            )

            # Get DP process group (use global group for this test)
            dp_mesh = dist.group.WORLD

            batch_iter = StopAfterOneEpoch(
                iter=iter(dataloader),
                device=torch.device("cuda"),
                dp_mesh=dp_mesh,
            )

            # Collect batches
            batches = []
            for batch in batch_iter:
                batches.append(batch)
                # All should be epoch 0
                assert extract_epoch_from_batch(batch) == 0

            # All ranks should have processed exactly one epoch
            # Since dataset is split across ranks, each rank gets 10 samples = 5 batches
            assert (
                len(batches) == 5
            ), f"Rank {rank} expected 5 batches, got {len(batches)}"

            # Synchronize to ensure both ranks completed
            dist.barrier()

        finally:
            shutil.rmtree(temp_dir)
