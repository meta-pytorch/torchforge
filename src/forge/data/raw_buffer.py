# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, TypeVar

from forge.interfaces import RawBuffer

K = TypeVar("K")
V = TypeVar("V")


class SimpleRawBuffer(RawBuffer[K, V]):
    """Simple in-memory RawBuffer backed by a Python dictionary."""

    def __init__(self) -> None:
        self._buffer: dict[K, V] = {}

    def __len__(self) -> int:
        """Return the number of key-value pairs in the buffer."""
        return len(self._buffer)

    def __getitem__(self, key: K) -> V:
        """Get a value from the buffer using the specified key."""
        return self._buffer[key]

    def __iter__(self) -> Iterator[tuple[K, V]]:
        """Iterate over the key-value pairs in the buffer."""
        for k, v in self._buffer.items():
            yield k, v

    def keys(self) -> Iterator[K]:
        """Iterate over the keys in the buffer."""
        for k in self._buffer.keys():
            yield k

    def add(self, key: K, val: V) -> None:
        """Add a key-value pair to the buffer."""
        if key in self._buffer:
            raise KeyError(f"Key {key} already exists in the buffer.")
        self._buffer[key] = val

    def pop(self, key: K) -> V:
        """Remove and return a value from the buffer using the specified key."""
        if key not in self._buffer:
            raise KeyError(f"Key {key} does not exist in the buffer.")
        val = self._buffer[key]
        del self._buffer[key]
        return val

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
