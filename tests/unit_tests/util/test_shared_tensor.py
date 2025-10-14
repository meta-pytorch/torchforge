# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import time
from multiprocessing import Process, Queue

import pytest
import torch

# Assuming SharedTensor is in shared_tensor.py
from forge.util._shared_tensor import SharedTensor


class TestSharedTensorCreation:
    """Test tensor creation methods"""

    def test_empty_creation(self):
        """Test creating empty tensor"""
        shape = (100, 200)
        dtype = torch.float32

        shared = SharedTensor.empty(shape, dtype)

        assert shared.tensor.shape == torch.Size(shape)
        assert shared.tensor.dtype == dtype
        assert shared.tensor.shape == torch.Size(shape)
        assert shared.tensor.dtype == dtype

        shared.cleanup()

    def test_empty_with_bfloat16(self):
        """Test creating empty bfloat16 tensor"""
        shape = (50, 50)
        shared = SharedTensor.empty(shape, torch.bfloat16)

        assert shared.tensor.dtype == torch.bfloat16
        assert shared.tensor.dtype == torch.bfloat16

        shared.cleanup()

    def test_zeros_creation(self):
        """Test creating zero-initialized tensor"""
        shape = (10, 20)
        shared = SharedTensor.zeros(shape, torch.float32)

        tensor = shared.tensor
        assert torch.all(tensor == 0)
        assert tensor.sum().item() == 0.0

        shared.cleanup()

    def test_ones_creation(self):
        """Test creating ones-initialized tensor"""
        shape = (10, 20)
        shared = SharedTensor.ones(shape, torch.float32)

        tensor = shared.tensor
        assert torch.all(tensor == 1)
        assert tensor.sum().item() == 200.0

        shared.cleanup()

    def test_from_tensor_creation(self):
        """Test creating from existing tensor"""
        original = torch.randn(50, 50)
        shared = SharedTensor(tensor=original)

        assert shared.tensor.shape == original.shape
        assert shared.tensor.dtype == original.dtype
        assert torch.allclose(shared.tensor, original)

        shared.cleanup()

    def test_from_handle_creation(self):
        """Test creating from handle"""
        # Create original
        original = SharedTensor.empty((10, 10), torch.float32)
        original.tensor.fill_(5.0)

        # Get handle
        handle = original.get_handle()

        # Create from handle
        reconstructed = SharedTensor(handle=handle)

        assert torch.all(reconstructed.tensor == 5.0)
        assert reconstructed.tensor.shape == original.tensor.shape
        assert reconstructed.tensor.dtype == original.tensor.dtype

        original.cleanup()

    def test_creation_requires_argument(self):
        """Test that creation without arguments raises error"""
        with pytest.raises(ValueError, match="Must provide either tensor or handle"):
            SharedTensor()

    @pytest.mark.parametrize(
        "shape",
        [
            (10,),
            (10, 20),
            (5, 10, 15),
            (2, 3, 4, 5),
        ],
    )
    def test_various_shapes(self, shape):
        """Test creation with various shapes"""
        shared = SharedTensor.empty(shape, torch.float32)
        assert shared.tensor.shape == torch.Size(shape)
        assert shared.tensor.shape == torch.Size(shape)
        shared.cleanup()


class TestSharedTensorDtypes:
    """Test all supported dtypes"""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ],
    )
    def test_all_dtypes(self, dtype):
        """Test that all dtypes work correctly"""
        shape = (10, 10)
        shared = SharedTensor.empty(shape, dtype)

        assert shared.tensor.dtype == dtype
        assert shared.tensor.dtype == dtype

        # Test that we can write to it
        if dtype == torch.bool:
            shared.tensor.fill_(True)
        elif dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
            shared.tensor.fill_(42)
        else:
            shared.tensor.fill_(3.14)

        shared.cleanup()

    def test_dtype_conversion_in_handle(self):
        """Test dtype is preserved through handle"""
        for dtype in [torch.float32, torch.bfloat16, torch.int64]:
            shared1 = SharedTensor.empty((5, 5), dtype)
            handle = shared1.get_handle()

            shared2 = SharedTensor(handle=handle)
            assert shared2.tensor.dtype == dtype

            shared1.cleanup()


class TestSharedTensorOperations:
    """Test tensor operations"""

    def test_copy_from(self):
        """Test copying data from another tensor"""
        source = torch.randn(20, 30)
        shared = SharedTensor.empty((20, 30), torch.float32)

        shared.copy_from(source)

        assert torch.allclose(shared.tensor, source)
        shared.cleanup()

    def test_copy_from_shape_mismatch(self):
        """Test copy_from raises error on shape mismatch"""
        source = torch.randn(10, 10)
        shared = SharedTensor.empty((20, 20), torch.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            shared.copy_from(source)

        shared.cleanup()

    def test_clone(self):
        """Test cloning creates independent copy"""
        original = SharedTensor.empty((10, 10), torch.float32)
        original.tensor.fill_(5.0)

        cloned = original.clone()

        # Verify data is same
        assert torch.all(cloned.tensor == 5.0)

        # Verify they're independent
        original.tensor.fill_(10.0)
        assert torch.all(cloned.tensor == 5.0)
        assert torch.all(original.tensor == 10.0)

        original.cleanup()
        cloned.cleanup()

    def test_tensor_modifications(self):
        """Test that modifications to tensor are reflected"""
        shared = SharedTensor.zeros((10, 10), torch.float32)
        tensor = shared.tensor

        tensor[0, 0] = 99.0
        tensor[5:, :] = 42.0

        # Get tensor again and verify changes persist
        tensor2 = shared.tensor
        assert tensor2[0, 0].item() == 99.0
        assert torch.all(tensor2[5:, :] == 42.0)

        shared.cleanup()

    def test_inplace_operations(self):
        """Test in-place operations work"""
        shared = SharedTensor.empty((100, 100), torch.float32)
        tensor = shared.tensor

        tensor.normal_(0, 1)
        mean = tensor.mean().item()

        tensor.add_(5.0)
        new_mean = tensor.mean().item()

        assert abs(new_mean - (mean + 5.0)) < 0.1

        shared.cleanup()


class TestSharedTensorSerialization:
    """Test pickling and handle serialization"""

    def test_handle_is_picklable(self):
        """Test that handle can be pickled"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        handle = shared.get_handle()

        # Pickle and unpickle
        pickled = pickle.dumps(handle)
        unpickled_handle = pickle.loads(pickled)

        assert unpickled_handle == handle
        assert unpickled_handle["shm_name"] == handle["shm_name"]
        assert unpickled_handle["shape"] == handle["shape"]
        assert unpickled_handle["dtype"] == handle["dtype"]

        shared.cleanup()

    def test_handle_small_size(self):
        """Test that handle is small (efficient for RPC)"""
        shared = SharedTensor.empty((10000, 10000), torch.float32)
        handle = shared.get_handle()

        pickled = pickle.dumps(handle)

        # Handle should be < 1KB even for huge tensors
        assert len(pickled) < 1024

        shared.cleanup()

    def test_data_integrity_after_pickle(self):
        """Test data is preserved through handle pickling"""
        # Create and fill tensor
        shared1 = SharedTensor.empty((50, 50), torch.bfloat16)
        shared1.tensor.normal_(0, 1)
        original_data = shared1.tensor.clone()

        # Pickle handle
        handle = shared1.get_handle()
        pickled = pickle.dumps(handle)
        unpickled_handle = pickle.loads(pickled)

        # Reconstruct
        shared2 = SharedTensor(handle=unpickled_handle)

        # Verify data is same
        assert torch.allclose(shared2.tensor.float(), original_data.float(), rtol=1e-3)

        shared1.cleanup()


class TestSharedTensorMemory:
    """Test memory management and cleanup"""

    def test_cleanup(self):
        """Test cleanup removes shared memory"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        shm_name = shared._shm_name

        # Verify shared memory exists
        tensor = shared.tensor
        tensor.fill_(5.0)

        # Cleanup
        shared.cleanup()

        # Trying to attach should fail
        from multiprocessing import shared_memory

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)

    def test_multiple_views_same_memory(self):
        """Test multiple tensor views point to same memory"""
        shared = SharedTensor.empty((10, 10), torch.float32)

        tensor1 = shared.tensor
        tensor1.fill_(5.0)

        tensor2 = shared.tensor
        assert torch.all(tensor2 == 5.0)

        # Modify through tensor2
        tensor2.fill_(10.0)

        # Verify tensor1 sees the change
        assert torch.all(tensor1 == 10.0)

        shared.cleanup()

    def test_handle_reconstruction_shares_memory(self):
        """Test that handle reconstruction shares same memory"""
        shared1 = SharedTensor.empty((20, 20), torch.float32)
        shared1.tensor.fill_(7.0)

        handle = shared1.get_handle()
        shared2 = SharedTensor(handle=handle)

        # Modify through shared2
        shared2.tensor.fill_(14.0)

        # Verify shared1 sees the change
        assert torch.all(shared1.tensor == 14.0)

        shared1.cleanup()


class TestSharedTensorEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_shape(self):
        """Test scalar tensor (empty shape)"""
        shared = SharedTensor.ones((), torch.float32)
        assert shared.tensor.shape == ()
        assert shared.tensor.numel() == 1
        assert torch.allclose(
            shared.tensor,
            torch.ones(
                (),
            ),
        )
        shared.cleanup()

    def test_single_element_tensor(self):
        """Test 1-element tensor"""
        shared = SharedTensor.empty((1,), torch.float32)
        shared.tensor.fill_(42.0)
        assert shared.tensor.item() == 42.0
        shared.cleanup()

    def test_large_tensor(self):
        """Test large tensor (1GB)"""
        # 1GB tensor: 250M float32 elements
        shape = (250_000_000,)
        shared = SharedTensor.empty(shape, torch.float32)

        assert shared.tensor.shape == shape
        assert shared.tensor.numel() == 250_000_000

        shared.cleanup()

    def test_non_contiguous_tensor_conversion(self):
        """Test that non-contiguous tensors are handled"""
        # Create non-contiguous tensor
        original = torch.randn(10, 10).t()  # Transpose makes it non-contiguous
        assert not original.is_contiguous()

        # Should work (internally makes contiguous)
        shared = SharedTensor(tensor=original)

        # Result should match
        assert torch.allclose(shared.tensor, original)

        shared.cleanup()

    def test_repr(self):
        """Test string representation"""
        shared = SharedTensor.empty((10, 20), torch.float32)
        repr_str = repr(shared)

        assert "SharedTensor" in repr_str
        assert "10, 20" in repr_str
        assert "float32" in repr_str
        assert shared._shm_name in repr_str

        shared.cleanup()


class TestSharedTensorMultiprocess:
    """Test multiprocess scenarios"""

    def test_multiprocess_read(self):
        """Test reading shared tensor from another process"""

        def reader_process(handle_dict, result_queue):
            shared = SharedTensor(handle=handle_dict)
            tensor = shared.tensor
            result_queue.put(tensor.sum().item())

        # Create shared tensor in main process
        shared = SharedTensor.empty((100, 100), torch.float32)
        shared.tensor.fill_(5.0)

        # Read from child process
        result_queue = Queue()
        handle = shared.get_handle()

        p = Process(target=reader_process, args=(handle, result_queue))
        p.start()
        p.join()

        result = result_queue.get()
        expected = 5.0 * 100 * 100

        assert abs(result - expected) < 1e-5

        shared.cleanup()

    def test_multiprocess_write(self):
        """Test writing to shared tensor from another process"""

        def writer_process(handle_dict, value):
            shared = SharedTensor(handle=handle_dict)
            shared.tensor.fill_(value)

        # Create empty shared tensor
        shared = SharedTensor.empty((50, 50), torch.float32)
        shared.tensor.zero_()

        # Write from child process
        handle = shared.get_handle()

        p = Process(target=writer_process, args=(handle, 42.0))
        p.start()
        p.join()

        # Verify in main process
        assert torch.all(shared.tensor == 42.0)

        shared.cleanup()

    def test_multiprocess_bidirectional(self):
        """Test bidirectional communication"""

        def worker_process(input_handle, output_handle):
            input_tensor = SharedTensor(handle=input_handle).tensor
            output_tensor = SharedTensor(handle=output_handle).tensor

            # Compute: output = input * 2
            output_tensor.copy_(input_tensor * 2)

        # Create input and output tensors
        input_shared = SharedTensor.empty((100, 100), torch.float32)
        input_shared.tensor.normal_(0, 1)
        input_data = input_shared.tensor.clone()

        output_shared = SharedTensor.empty((100, 100), torch.float32)

        # Process in child
        p = Process(
            target=worker_process,
            args=(input_shared.get_handle(), output_shared.get_handle()),
        )
        p.start()
        p.join()

        # Verify result
        expected = input_data * 2
        assert torch.allclose(
            output_shared.tensor, expected
        ), "output: {}, expected: {}".format(output_shared.tensor, expected)

        input_shared.cleanup()
        output_shared.cleanup()


class TestSharedTensorPerformance:
    """Performance-related tests"""

    def test_empty_faster_than_from_tensor(self):
        """Test that empty() is faster than from tensor"""
        shape = (1000, 1000)

        # Time empty creation
        start = time.time()
        for _ in range(10):
            shared = SharedTensor.empty(shape, torch.float32)
            shared.cleanup()
        empty_time = time.time() - start

        # Time from_tensor creation
        start = time.time()
        for _ in range(10):
            tensor = torch.randn(shape)
            shared = SharedTensor(tensor=tensor)
            shared.cleanup()
        from_tensor_time = time.time() - start

        # empty() should be faster (no data copying)
        assert empty_time < from_tensor_time

    def test_handle_serialization_fast(self):
        """Test that handle serialization is fast"""
        shared = SharedTensor.empty((10000, 10000), torch.float32)
        handle = shared.get_handle()

        start = time.time()
        for _ in range(1000):
            pickled = pickle.dumps(handle)
            unpickled = pickle.loads(pickled)
        elapsed = time.time() - start

        # Should be able to do 1000 round trips in < 0.1 seconds
        assert elapsed < 0.1

        shared.cleanup()


class TestSharedTensorBfloat16:
    """Specific tests for bfloat16 support"""

    def test_bfloat16_creation(self):
        """Test bfloat16 tensor creation"""
        shared = SharedTensor.empty((100, 100), torch.bfloat16)
        assert shared.tensor.dtype == torch.bfloat16
        shared.cleanup()

    def test_bfloat16_from_tensor(self):
        """Test creating shared tensor from bfloat16 tensor"""
        original = torch.randn(50, 50, dtype=torch.bfloat16)
        shared = SharedTensor(tensor=original)

        assert shared.tensor.dtype == torch.bfloat16
        assert torch.allclose(shared.tensor.float(), original.float(), rtol=1e-3)

        shared.cleanup()

    def test_bfloat16_handle_preservation(self):
        """Test bfloat16 dtype preserved through handle"""
        shared1 = SharedTensor.empty((20, 20), torch.bfloat16)
        shared1.tensor.normal_(0, 1)

        handle = shared1.get_handle()
        shared2 = SharedTensor(handle=handle)

        assert shared2.tensor.dtype == torch.bfloat16
        assert torch.allclose(shared1.tensor.float(), shared2.tensor.float(), rtol=1e-3)

        shared1.cleanup()

    def test_bfloat16_operations(self):
        """Test operations on bfloat16 tensors"""
        shared = SharedTensor.empty((100, 100), torch.bfloat16)
        tensor = shared.tensor

        tensor.normal_(0, 1)
        mean = tensor.float().mean().item()

        # Mean should be close to 0
        assert abs(mean) < 0.1

        shared.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
