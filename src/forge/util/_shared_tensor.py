# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class SharedTensorHandle:
    shm_name: str
    shape: Tuple[int, ...]
    dtype: str

    def to_shared_tensor(self) -> SharedTensor:
        """
        Create a SharedTensor from this handle.

        Returns:
            SharedTensor instance attached to the shared memory referenced by this handle
        """
        return SharedTensor(handle=self)


class SharedTensor:
    """Wrapper class for tensors backed my shared memory"""

    def __init__(
        self,
        *,
        tensor: Optional[torch.Tensor] = None,
        handle: Optional[SharedTensorHandle] = None,
    ):
        if tensor is not None:
            self._create_from_tensor(tensor)
        elif handle is not None:
            self._create_from_handle(handle)
        else:
            raise ValueError("Must provide either tensor or handle")

    @classmethod
    def empty(
        cls,
        shape: Union[Tuple[int, ...], torch.Size],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create an empty tensor directly in shared memory (no copy/allocation overhead)

        Args:
            shape: Shape of the tensor
            dtype: PyTorch dtype (supports bfloat16, float32, etc.)

        Returns:
            SharedTensor instance with uninitialized data
        """
        instance = cls.__new__(cls)
        instance._create_empty(shape, dtype)
        return instance

    @classmethod
    def zeros(
        cls,
        shape: Union[Tuple[int, ...], torch.Size],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create a zero-initialized tensor in shared memory

        Args:
            shape: Shape of the tensor
            dtype: PyTorch dtype

        Returns:
            SharedTensor instance with zeros
        """
        shared_tensor = cls.empty(shape, dtype)
        shared_tensor.tensor.zero_()
        return shared_tensor

    @classmethod
    def ones(
        cls,
        shape: Union[Tuple[int, ...], torch.Size],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create a ones-initialized tensor in shared memory

        Args:
            shape: Shape of the tensor
            dtype: PyTorch dtype

        Returns:
            SharedTensor instance with ones
        """
        shared_tensor = cls.empty(shape, dtype)
        shared_tensor.tensor.fill_(1)
        return shared_tensor

    def _create_empty(self, shape, dtype):
        """Initialize with empty tensor in shared memory"""
        # Store metadata
        self._shape = tuple(shape) if not isinstance(shape, tuple) else shape
        self._dtype = dtype
        self._dtype_str = str(dtype)

        # Calculate size
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = int(np.prod(self._shape))
        byte_size = total_elements * element_size

        # Create shared memory (uninitialized - fast!)
        shm_name = f"shared_tensor_{uuid.uuid4().hex}"
        self._shm = shared_memory.SharedMemory(
            create=True, size=byte_size, name=shm_name
        )
        self._shm_name = shm_name

    def _create_from_tensor(self, tensor):
        """Initialize from an existing tensor"""
        tensor = tensor.contiguous()

        # Store metadata
        self._shape = tuple(tensor.shape)
        self._dtype = tensor.dtype
        self._dtype_str = str(tensor.dtype)

        # Create shared memory
        byte_size = tensor.numel() * tensor.element_size()
        shm_name = f"shared_tensor_{uuid.uuid4().hex}"

        self._shm = shared_memory.SharedMemory(
            create=True, size=byte_size, name=shm_name
        )
        self._shm_name = shm_name

        # Copy data as raw bytes
        raw_bytes = tensor.view(torch.uint8).view(-1).cpu().contiguous().numpy()
        self._shm.buf[:byte_size] = raw_bytes

    def _create_from_handle(self, handle: SharedTensorHandle):
        """Initialize from a handle"""
        self._shm_name = handle.shm_name
        self._shape = handle.shape
        self._dtype_str = handle.dtype
        self._dtype = self._parse_dtype(self._dtype_str)

        # Attach to existing shared memory
        self._shm = shared_memory.SharedMemory(name=self._shm_name)

    def _create_tensor_view(self):
        """Create tensor view of shared memory."""
        element_size = torch.tensor([], dtype=self._dtype).element_size()
        total_elements = int(np.prod(self._shape))
        byte_size = total_elements * element_size

        # Create numpy array that shares the buffer
        np_array = np.ndarray(shape=(byte_size,), dtype=np.uint8, buffer=self._shm.buf)
        # Create torch tensor from numpy (shares memory)
        uint8_tensor = torch.from_numpy(np_array)
        tensor = uint8_tensor.view(self._dtype).reshape(self._shape)

        # Keep both the np array and the SharedTensor object alive
        tensor._forge_shared_tensor = self
        tensor._forge_np_array = np_array

        return tensor

    def _parse_dtype(self, dtype_str):
        """Parse dtype string"""
        dtype_str = dtype_str.replace("torch.", "")
        return getattr(torch, dtype_str)

    def get_handle(self):
        """Get picklable handle"""
        return SharedTensorHandle(
            shm_name=self._shm_name,
            shape=self._shape,
            dtype=self._dtype_str,
        )

    @functools.cached_property
    def tensor(self):
        """Get the underlying tensor"""
        return self._create_tensor_view()

    def copy_from(self, source_tensor):
        """
        Copy data from another tensor into this shared tensor
        Useful when you create empty tensor first, then fill it

        Args:
            source_tensor: Source tensor to copy from
        """
        if source_tensor.shape != self._shape:
            raise ValueError(f"Shape mismatch: {source_tensor.shape} vs {self._shape}")
        # Copy data
        self.tensor.copy_(source_tensor)

    def clone(self):
        """Create a new SharedTensor with copied data"""
        new_shared = SharedTensor.empty(self._shape, self._dtype)
        new_shared.tensor.copy_(self.tensor)
        return new_shared

    def drop(self):
        """
        Release and unlink the shared memory.

        This method closes the shared memory handle and removes the shared memory
        segment from the system. After calling this method, the shared memory
        will no longer be accessible by any process.

        Note:
            This should be called when the shared tensor is no longer needed.
            Failing to call this method may result in shared memory leaks.
        """
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, "shm"):
            try:
                self._shm.close()
            except Exception:
                pass

    def __repr__(self):
        return f"SharedTensor(shape={self._shape}, dtype={self._dtype}, shm_name={self._shm_name})"
