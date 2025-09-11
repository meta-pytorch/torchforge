# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

from src.forge.interfaces import StoreInterface


class KVStore(StoreInterface):
    """
    A simple single-node key-value (KV) store implementation of StoreInterface.

    This acts as a temporary backend for the replay buffer until torchstore
    supports the full set of operations we need (delete, pop, keys, numel, etc.).
    """

    def __init__(self):
        self._store = {}

    async def put(self, key: str, value: Any) -> None:
        self._store[key] = value

    async def get(self, key: str) -> Any:
        return self._store[key]

    async def exists(self, key: str) -> bool:
        # Check if a key exists in the KV store
        return key in self._store

    async def keys(self, prefix: str | None = None) -> list[str]:
        # Return all keys, optionally filtered by prefix
        if prefix is None:
            return list(self._store.keys())
        return [k for k in self._store if k.startswith(prefix)]

    async def numel(self, prefix: str | None = None) -> int:
        # Return the number of key-value pairs, optionally filtered by prefix
        return len(await self.keys(prefix))

    async def delete(self, key: str) -> None:
        # Delete a key-value pair from the store
        del self._store[key]

    async def pop(self, key: str) -> Any:
        # Remove and return a key-value pair (get + delete)
        return self._store.pop(key)
