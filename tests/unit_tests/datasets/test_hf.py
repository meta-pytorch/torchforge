# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for HfIterableDataset core functionality.

This module tests the foundational iterable dataset capabilities including:
- Basic iteration and data loading
- Epoch boundary handling and tracking
- Shuffling behavior across epochs
- Checkpointing and state restoration
- Distributed training scenarios

Uses synthetic JSON data with predictable patterns to verify correct behavior.
"""

import math
import shutil
import tempfile
from itertools import islice
from pathlib import Path

import pytest
import torch.distributed as dist

from forge.data.datasets import HfIterableDataset
from forge.data.metric_transform import DefaultDatasetMetricTransform

from tests.test_utils import gpu_test
from torch.testing._internal.common_fsdp import FSDPTest

from torchdata.stateful_dataloader import StatefulDataLoader

from .test_iterable_utils import collate_with_metrics, generate_ckpt

# Test Constants - Avoid perfect divisions
SMALL_DATASET_SIZE = 23
MEDIUM_DATASET_SIZE = 35
SEED = 42
BATCH_SIZE = 5
DEFAULT_SHUFFLE_BUFFER_SIZE = 8


def create_test_json_file(path: Path, num_samples: int, offset: int = 0) -> None:
    """Creates a dummy JSON test data file with token samples of varying lengths.

    Args:
        path (Path): The path to the file to create
        num_samples (int): The number of samples to create
        offset (int): The offset to add to the sample ID to ensure unique IDs in different datasets
    """
    with open(path, "w") as f:
        for i in range(num_samples):
            sample_id = i + offset
            # Realistic token length variation (1-3 tokens)
            token_len = (i % 3) + 1
            tokens = list(range(sample_id, sample_id + token_len))
            f.write(
                f'{{"id": {sample_id}, "tokens": {tokens}, "text": "sample_{sample_id}", "labels": {tokens}}}\n'
            )


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide temporary directory for test data files."""
    return tmp_path


@pytest.fixture
def small_dataset_file(tmp_data_dir):
    path = tmp_data_dir / "small_data.json"
    create_test_json_file(path, SMALL_DATASET_SIZE, offset=0)
    return str(path)


@pytest.fixture
def dataset_factory():
    """Factory for creating HfIterableDataset instances with common defaults."""

    def _create_dataset(
        data_file: str,
        dataset_name: str = "test_dataset",
        shuffle: bool = False,
        **kwargs,
    ) -> HfIterableDataset:
        return HfIterableDataset(
            path="json",
            data_files=data_file,
            split="train",
            dataset_name=dataset_name,
            seed=SEED,
            shuffle_buffer_size=10 if shuffle else 0,
            metric_transform=DefaultDatasetMetricTransform(),
            num_shards_per_rank=2,
            **kwargs,
        )

    return _create_dataset


class TestHfIterableDataset:
    """Tests for HfIterableDataset basic functionality."""

    def test_default_dataset_name(self, small_dataset_file):
        """Test that dataset name is auto-generated from path when not provided."""
        # Create dataset without specifying name
        dataset = HfIterableDataset(
            path="json",
            data_files=small_dataset_file,
            split="train",
            # dataset_name not provided - should auto-generate
            seed=SEED,
            metric_transform=DefaultDatasetMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should generate name from path and split
        assert dataset.info.name == "json_train"
        # Test default sampling weight
        assert dataset.info.weight == 1.0

        # Test giving a name and custom weight
        custom_weight = 2.5
        dataset2 = HfIterableDataset(
            path="json",
            data_files=small_dataset_file,
            split="train",
            dataset_name="my_dataset",
            weight=custom_weight,
            seed=SEED,
            metric_transform=DefaultDatasetMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should use provided name and weight
        assert dataset2.info.name == "my_dataset"
        # Test custom sampling weight
        assert dataset2.info.weight == custom_weight

    @pytest.mark.parametrize("num_epochs", [0.5, 1.0, 2.5])
    def test_epoch_boundaries_and_checkpointing(
        self, num_epochs, dataset_factory, small_dataset_file
    ):
        """
        Tests that for N epochs, each sample appears exactly N times (rounded down),
        the epoch metric is correct, and checkpointing works as expected.
        """

        # 1. Setup Dataloaders for original and resumed runs
        def create_loader():
            dataset = dataset_factory(small_dataset_file, shuffle=False)
            loader = StatefulDataLoader(
                dataset, batch_size=BATCH_SIZE, collate_fn=collate_with_metrics
            )
            return loader

        loader1 = create_loader()
        loader2 = create_loader()

        # 2. Calculate steps for the test run
        total_samples = int(SMALL_DATASET_SIZE * num_epochs)
        total_steps = total_samples // BATCH_SIZE

        steps_before_checkpoint = max(1, total_steps // 2)
        steps_after_checkpoint = total_steps - steps_before_checkpoint

        # 3. Generate checkpoint and resume
        result = generate_ckpt(
            loader1,
            steps_before_checkpoint=steps_before_checkpoint,
            steps_after_checkpoint=steps_after_checkpoint,
            resume_dataloader=loader2,
        )

        # 4. Verify checkpointing and resumption
        orig_post_ids = [b["id"].tolist() for b in result["post_checkpoint_batches"]]
        resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
        assert (
            orig_post_ids == resumed_ids
        ), "Resumed batches should be identical for deterministic run"

        assert (
            result["post_checkpoint_metrics"] == result["resumed_metrics"]
        ), "Resumed training should produce same metrics as original training"

    def test_shuffling_behavior(self, dataset_factory, small_dataset_file):
        """Tests that shuffling changes data order between epochs but preserves the set of samples."""
        # Test unshuffled dataset
        unshuffled_ds = dataset_factory(
            small_dataset_file, dataset_name="unshuffled", shuffle=False
        )

        # Get samples from two passes through the dataset
        epoch_samples = list(islice(iter(unshuffled_ds), SMALL_DATASET_SIZE * 2))

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # Unshuffled should have same order in both epochs
        first_epoch_ids = [sample["id"] for sample in first_epoch_samples]
        second_epoch_ids = [sample["id"] for sample in second_epoch_samples]
        assert first_epoch_ids == list(range(SMALL_DATASET_SIZE))
        assert second_epoch_ids == list(range(SMALL_DATASET_SIZE))

        # Test shuffled dataset
        shuffled_ds = dataset_factory(
            small_dataset_file, dataset_name="shuffled", shuffle=True
        )

        # Collect full epochs to compare
        epoch_samples = list(islice(iter(shuffled_ds), SMALL_DATASET_SIZE * 2))

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # Extract IDs for comparison
        first_epoch_ids = [sample["id"] for sample in first_epoch_samples]
        second_epoch_ids = [sample["id"] for sample in second_epoch_samples]

        # Shuffled epochs should have different order
        assert first_epoch_ids != list(
            range(SMALL_DATASET_SIZE)
        ), f"Shuffled should not be sorted, got {first_epoch_ids}"
        assert (
            first_epoch_ids != second_epoch_ids
        ), f"Shuffled epochs should be shuffled differently, got {first_epoch_ids} and {second_epoch_ids}"

        # But should contain the same set of IDs
        assert set(first_epoch_ids) == set(
            range(SMALL_DATASET_SIZE)
        ), f"First epoch samples should be (0-{SMALL_DATASET_SIZE - 1}), got {first_epoch_ids}"
        assert set(second_epoch_ids) == set(
            range(SMALL_DATASET_SIZE)
        ), f"Second epoch samples should be (0-{SMALL_DATASET_SIZE - 1}), got {second_epoch_ids}"

    def test_epoch_tracking(self, dataset_factory, small_dataset_file):
        """Test that epoch number is correctly tracked across dataset restarts."""
        dataset = dataset_factory(small_dataset_file, shuffle=False)

        # Two epoch samples
        epoch_samples = list(islice(iter(dataset), SMALL_DATASET_SIZE * 2))

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # All should have epoch 0
        first_epoch_metrics = []
        for sample in first_epoch_samples:
            first_epoch_metrics.extend(sample["metrics"])
        epoch_values = [
            metric.value for metric in first_epoch_metrics if "num_epochs" in metric.key
        ]
        assert all(
            epoch_value == 0 for epoch_value in epoch_values
        ), f"Epoch values should be 0, got {epoch_values}"

        # All should have epoch 1
        second_epoch_metrics = []
        for sample in second_epoch_samples:
            second_epoch_metrics.extend(sample["metrics"])
        epoch_values = [
            metric.value
            for metric in second_epoch_metrics
            if "num_epochs" in metric.key
        ]
        assert all(
            epoch_value == 1 for epoch_value in epoch_values
        ), f"Epoch values should be 1, got {epoch_values}"

    def test_multiple_iter_calls_after_resume(
        self, dataset_factory, small_dataset_file
    ):
        """Test that calling iter() multiple times after resuming restarts from checkpoint epoch.

        1. Resume from checkpoint at epoch 2
        2. Consume one epoch (now at epoch 3)
        3. Call iter(ds) again to create a new iterator
        4. The new iterator should restart from epoch 2 (checkpoint epoch), not 0 or 3

        This ensures datasets can be re-iterated from their checkpoint state.
        """
        dataset = dataset_factory(small_dataset_file, shuffle=False)

        # consume 2 epochs
        it1 = iter(dataset)
        samples = list(islice(it1, SMALL_DATASET_SIZE * 2))

        # Save checkpoint after 2 epochs
        state = dataset.state_dict()

        # Continue training for 1 more epoch on the same iterator
        more_samples = list(islice(it1, SMALL_DATASET_SIZE))

        # Create a new dataset instance and load the checkpoint
        dataset2 = dataset_factory(small_dataset_file, shuffle=False)
        dataset2.load_state_dict(state)

        # First iter() call should start from epoch 2 (the checkpoint epoch)
        it2 = iter(dataset2)
        first_iter_samples = list(islice(it2, SMALL_DATASET_SIZE))
        first_iter_epochs = [
            metric.value
            for sample in first_iter_samples
            for metric in sample["metrics"]
            if "num_epochs" in metric.key
        ]
        assert all(
            epoch == 2 for epoch in first_iter_epochs
        ), f"First iter() should start at checkpoint epoch 2, got {set(first_iter_epochs)}"

        # Consume one more epoch from the same iterator (now at epoch 3)
        second_epoch_samples = list(islice(it2, SMALL_DATASET_SIZE))
        second_epoch_epochs = [
            metric.value
            for sample in second_epoch_samples
            for metric in sample["metrics"]
            if "num_epochs" in metric.key
        ]
        assert all(
            epoch == 3 for epoch in second_epoch_epochs
        ), f"Second epoch should be 3, got {set(second_epoch_epochs)}"

        # Call iter() again - it should restart from epoch 2, not continue from 4
        it3 = iter(dataset2)
        new_iter_samples = list(islice(it3, SMALL_DATASET_SIZE))
        new_iter_epochs = [
            metric.value
            for sample in new_iter_samples
            for metric in sample["metrics"]
            if "num_epochs" in metric.key
        ]
        assert all(
            epoch == 2 for epoch in new_iter_epochs
        ), f"New iter() should restart from checkpoint epoch 2, got {set(new_iter_epochs)}"


class TestDistributedHfIterableDataset(FSDPTest):
    """Test HfIterableDataset with 2-GPU distributed setup."""

    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_distributed_epoch_boundary_checkpointing(self):
        """
        Test epoch boundary handling with checkpointing in distributed setting.
        Ensures proper handling of:
        - Checkpointing at 0.9, 1.0, and 2.5 epoch boundaries
        - Correct sample distribution across epochs
        - Proper state restoration after checkpointing
        """
        rank = dist.get_rank()

        # Each rank creates its own local temp dir and files
        temp_dir = tempfile.mkdtemp(prefix=f"epoch_test_rank{rank}_")
        tmp_path = Path(temp_dir)

        try:
            medium_dataset_file = tmp_path / "medium_data.json"

            # Each rank creates its own file
            create_test_json_file(medium_dataset_file, MEDIUM_DATASET_SIZE)

            # Test multiple epoch boundaries
            for num_epochs in [0.9, 1.0, 2.5]:

                def create_loader():
                    dataset = HfIterableDataset(
                        path="json",
                        data_files=str(medium_dataset_file),
                        split="train",
                        dataset_name="epoch_test",
                        seed=SEED,
                        shuffle_buffer_size=0,  # No shuffle for determinism
                        metric_transform=DefaultDatasetMetricTransform(),
                        num_shards_per_rank=2,
                    )
                    loader = StatefulDataLoader(
                        dataset,
                        batch_size=BATCH_SIZE,
                        collate_fn=collate_with_metrics,
                        num_workers=0,
                    )
                    return loader

                loader1 = create_loader()
                loader2 = create_loader()

                # Calculate steps to reach desired epoch boundary
                samples_per_rank = MEDIUM_DATASET_SIZE // dist.get_world_size()
                total_samples = int(samples_per_rank * num_epochs)
                total_steps = total_samples // BATCH_SIZE

                if total_steps < 2:
                    raise ValueError(
                        f"Not enough steps for meaningful test: {total_steps}"
                    )

                # Split steps between before and after checkpoint
                steps_before = max(1, total_steps // 2)
                steps_after = total_steps - steps_before

                result = generate_ckpt(
                    loader1,
                    steps_before_checkpoint=steps_before,
                    steps_after_checkpoint=steps_after,
                    resume_dataloader=loader2,
                )

                # Verify deterministic resumption - critical for distributed training
                orig_post_ids = [
                    b["id"].tolist() for b in result["post_checkpoint_batches"]
                ]
                resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
                assert orig_post_ids == resumed_ids, (
                    f"Rank {rank}: Non-deterministic resume for {num_epochs} epochs. "
                    f"This indicates checkpoint/resume state is not properly preserved."
                )

                # Verify epoch metric is correctly tracked
                final_metrics = result["final_metrics"]
                expected_epoch = math.floor(
                    num_epochs - 1e-9
                )  # -1e-9 so 1.0 epochs -> 0
                assert (
                    final_metrics["dataset/epoch_test/num_epochs"] == expected_epoch
                ), f"Epoch count incorrect for {num_epochs} epochs test scenario"

        finally:
            shutil.rmtree(temp_dir)


class TestDPShardingWithTP(FSDPTest):
    """Test DP sharding with TP replication (4-GPU setup)."""

    @property
    def world_size(self) -> int:
        return 4

    @gpu_test(gpu_count=4)
    def test_dp_sharding_with_tp_replication(self):
        """Verify DP sharding works correctly with TP/CP replication.

        This is a CRITICAL test that validates the core bug fix:
        - Previously: Each rank got different batches (incorrect)
        - Now: TP/CP ranks within same DP group get identical batches (correct)

        Setup: DP=2, TP=2 (4 GPUs total)
        - DP group 0: ranks [0, 1] - should see SAME batches (TP replication)
        - DP group 1: ranks [2, 3] - should see SAME batches (TP replication)
        - DP group 0 vs 1: should see DIFFERENT batches (DP sharding)

        Mesh structure:
        - TP rank 0 DP replicas: [0, 2] - shard across these
        - TP rank 1 DP replicas: [1, 3] - shard across these
        """
        import hashlib

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        temp_dir = tempfile.mkdtemp(prefix=f"dp_tp_test_rank{rank}_")

        try:
            data_file = Path(temp_dir) / "data.json"
            # Create dataset with enough samples for clear sharding
            # 40 samples / 2 DP groups = 20 samples per DP group
            create_test_json_file(data_file, MEDIUM_DATASET_SIZE, offset=0)

            # Create DP mesh for sharding
            # Key insight: Create groups across DP replicas for each TP rank
            # TP rank = rank % 2, so:
            # - TP rank 0: ranks [0, 2] (one from each DP group)
            # - TP rank 1: ranks [1, 3] (one from each DP group)
            tp_rank = rank % 2
            tp_world_size = 2
            dp_world_size = world_size // tp_world_size

            # Create DP groups for each TP rank
            dp_groups = []
            for tp_r in range(tp_world_size):
                # Ranks for this TP rank across DP groups
                ranks = [tp_r + i * tp_world_size for i in range(dp_world_size)]
                group = dist.new_group(ranks=ranks)
                dp_groups.append(group)

            dp_mesh = dp_groups[tp_rank]

            # - Rank 0 (tp_rank=0) uses group [0, 2], gets rank=0 → shard 0
            # - Rank 1 (tp_rank=1) uses group [1, 3], gets rank=0 → shard 0
            # - Rank 2 (tp_rank=0) uses group [0, 2], gets rank=1 → shard 1
            # - Rank 3 (tp_rank=1) uses group [1, 3], gets rank=1 → shard 1

            dataset = HfIterableDataset(
                path="json",
                data_files=str(data_file),
                split="train",
                dataset_name="dp_tp_test",
                shuffle_buffer_size=0,
                metric_transform=DefaultDatasetMetricTransform(),
                num_shards_per_rank=2,
                dp_mesh=dp_mesh,  # CRITICAL: Pass dp_mesh for correct sharding
            )

            dataloader = StatefulDataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                collate_fn=collate_with_metrics,
                num_workers=0,
            )

            # Collect batches and compute hashes
            batches = list(islice(iter(dataloader), 5))
            batch_hashes = []
            for batch in batches:
                # Hash the batch IDs to verify identity/difference
                batch_ids = batch["id"].cpu().tolist()
                batch_hash = hashlib.md5(str(batch_ids).encode()).hexdigest()
                batch_hashes.append(batch_hash)

            # Gather hashes from all ranks for comparison
            gathered_hashes = [None] * world_size
            dist.all_gather_object(gathered_hashes, batch_hashes)

            if rank == 0:
                # Verify TP replication within DP groups
                # Ranks 0 and 1 should have identical hashes (same DP group)
                assert gathered_hashes[0] == gathered_hashes[1], (
                    f"Ranks 0 and 1 (same DP group) should see identical batches!\n"
                    f"Rank 0 hashes: {gathered_hashes[0][:3]}...\n"
                    f"Rank 1 hashes: {gathered_hashes[1][:3]}..."
                )

                # Ranks 2 and 3 should have identical hashes (same DP group)
                assert gathered_hashes[2] == gathered_hashes[3], (
                    f"Ranks 2 and 3 (same DP group) should see identical batches!\n"
                    f"Rank 2 hashes: {gathered_hashes[2][:3]}...\n"
                    f"Rank 3 hashes: {gathered_hashes[3][:3]}..."
                )

                # Verify DP sharding across groups
                # Ranks 0/1 should see DIFFERENT batches from ranks 2/3
                assert gathered_hashes[0] != gathered_hashes[2], (
                    f"Ranks 0 and 2 (different DP groups) should see different batches!\n"
                    f"DP group 0 hashes: {gathered_hashes[0][:3]}...\n"
                    f"DP group 1 hashes: {gathered_hashes[2][:3]}..."
                )

            dist.barrier()

        finally:
            shutil.rmtree(temp_dir)
