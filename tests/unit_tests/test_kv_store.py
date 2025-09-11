# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test for forge/data/stores.py"""

import pytest
import pytest_asyncio
from forge.data.stores import KVStore


class TestKVStore:
    @pytest_asyncio.fixture
    async def store(self) -> KVStore:
        return KVStore()

    @pytest.mark.asyncio
    async def test_put_different_types(self, store: KVStore) -> None:
        """Test put and get with different value types."""
        await store.put("string_key", "string_value")
        await store.put("int_key", 42)
        await store.put("dict_key", {"nested": "dict"})
        await store.put("list_key", [1, 2, 3])

        assert await store.get("string_key") == "string_value"
        assert await store.get("int_key") == 42
        assert await store.get("dict_key") == {"nested": "dict"}
        assert await store.get("list_key") == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, store: KVStore) -> None:
        """Test getting a key that doesn't exist raises KeyError."""
        with pytest.raises(KeyError):
            await store.get("nonexistent_key")

    @pytest.mark.asyncio
    async def test_exists(self, store: KVStore) -> None:
        """Test exists method."""
        assert not await store.exists("key1")
        await store.put("key1", "value1")
        assert await store.exists("key1")

    @pytest.mark.asyncio
    async def test_keys_with_prefix(self, store: KVStore) -> None:
        """Test keys method with prefix."""
        await store.put("user.001", "user1")
        await store.put("user.002", "user2")
        await store.put("post.001", "post1")
        await store.put("comment.001", "comment1")

        user_keys = await store.keys("user")
        assert set(user_keys) == {"user.001", "user.002"}

        post_keys = await store.keys("post")
        assert set(post_keys) == {"post.001"}

        empty_keys = await store.keys("nonexistent")
        assert empty_keys == []

        # Test delete_all with prefix
        deleted_count = await store.delete_all("user")
        assert deleted_count == 2
        assert await store.numel("user") == 0
        assert await store.numel() == 2  # post and comment remain

        # Test delete_all with non-existent prefix
        deleted_count = await store.delete_all("nonexistent")
        assert deleted_count == 0

    @pytest.mark.asyncio
    async def test_keys_empty_store(self, store: KVStore) -> None:
        """Test keys method on empty store."""
        keys = await store.keys()
        assert keys == []

        keys_with_prefix = await store.keys("prefix")
        assert keys_with_prefix == []

    @pytest.mark.asyncio
    async def test_numel_with_prefix(self, store: KVStore) -> None:
        """Test numel method with prefix."""
        await store.put("user.001", "user1")
        await store.put("user.002", "user2")
        await store.put("post.001", "post1")

        assert await store.numel("user") == 2
        assert await store.numel("post") == 1
        assert await store.numel("nonexistent") == 0
        assert await store.numel() == 3

    @pytest.mark.asyncio
    async def test_delete(self, store: KVStore) -> None:
        """Test delete method."""
        await store.put("key1", "value1")
        assert await store.exists("key1")

        await store.delete("key1")
        assert not await store.exists("key1")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, store: KVStore) -> None:
        """Test deleting a key that doesn't exist raises KeyError."""
        with pytest.raises(KeyError):
            await store.delete("nonexistent_key")

    @pytest.mark.asyncio
    async def test_pop(self, store: KVStore) -> None:
        """Test pop method."""
        await store.put("key1", "value1")

        result = await store.pop("key1")
        assert result == "value1"
        assert not await store.exists("key1")

    @pytest.mark.asyncio
    async def test_pop_nonexistent_key(self, store: KVStore) -> None:
        """Test popping a key that doesn't exist raises KeyError."""
        with pytest.raises(KeyError):
            await store.pop("nonexistent_key")

    @pytest.mark.asyncio
    async def test_none_values(self, store: KVStore) -> None:
        """Test storing and retrieving None values."""
        await store.put("none_key", None)

        assert await store.exists("none_key")
        result = await store.get("none_key")
        assert result is None

        popped_result = await store.pop("none_key")
        assert popped_result is None

        # Test delete_all on empty store
        deleted_count = await store.delete_all()
        assert deleted_count == 0
        assert await store.numel() == 0
