# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test for data/raw_buffer.py"""

import pytest

from forge.data.raw_buffer import SimpleRawBuffer


class TestSimpleRawBuffer:
    """Test suite for SimpleRawBuffer class."""

    def test_init_empty_buffer(self):
        """Test that a new buffer is initialized empty."""
        buffer = SimpleRawBuffer[str, int]()
        assert len(buffer) == 0

    def test_add_single_item(self):
        """Test adding a single key-value pair."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("key1", 100)

        assert len(buffer) == 1
        assert buffer["key1"] == 100

    def test_add_multiple_items(self):
        """Test adding multiple key-value pairs."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("key1", 100)
        buffer.add("key2", 200)
        buffer.add("key3", 300)

        assert len(buffer) == 3
        assert buffer["key1"] == 100
        assert buffer["key2"] == 200
        assert buffer["key3"] == 300

    def test_add_duplicate_key_raises_error(self):
        """Test that adding a duplicate key raises KeyError."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("key1", 100)

        with pytest.raises(KeyError, match="Key key1 already exists in the buffer"):
            buffer.add("key1", 200)

    def test_getitem_existing_key(self):
        """Test retrieving an existing key."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("test_key", 42)

        assert buffer["test_key"] == 42

    def test_getitem_missing_key_raises_error(self):
        """Test that accessing a non-existent key raises KeyError."""
        buffer = SimpleRawBuffer[str, int]()

        with pytest.raises(KeyError):
            _ = buffer["missing_key"]

    def test_pop_existing_key(self):
        """Test removing and returning a value for an existing key."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("key1", 100)
        buffer.add("key2", 200)

        value = buffer.pop("key1")

        assert value == 100
        assert len(buffer) == 1
        assert "key1" not in buffer.keys()
        assert buffer["key2"] == 200

    def test_pop_missing_key_raises_error(self):
        """Test that popping a non-existent key raises KeyError."""
        buffer = SimpleRawBuffer[str, int]()

        with pytest.raises(
            KeyError, match="Key missing_key does not exist in the buffer"
        ):
            buffer.pop("missing_key")

    def test_keys_iteration(self):
        """Test iterating over keys."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("key1", 100)
        buffer.add("key2", 200)
        buffer.add("key3", 300)

        keys = list(buffer.keys())

        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_keys_empty_buffer(self):
        """Test iterating over keys in an empty buffer."""
        buffer = SimpleRawBuffer[str, int]()

        keys = list(buffer.keys())

        assert keys == []

    def test_iter_key_value_pairs(self):
        """Test iterating over key-value pairs."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("key1", 100)
        buffer.add("key2", 200)

        items = list(buffer)

        assert len(items) == 2
        assert ("key1", 100) in items
        assert ("key2", 200) in items

    def test_iter_empty_buffer(self):
        """Test iterating over an empty buffer."""
        buffer = SimpleRawBuffer[str, int]()

        items = list(buffer)

        assert items == []

    def test_clear_buffer(self):
        """Test clearing the buffer."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("key1", 100)
        buffer.add("key2", 200)
        buffer.add("key3", 300)

        assert len(buffer) == 3

        buffer.clear()

        assert len(buffer) == 0
        assert list(buffer.keys()) == []
        assert list(buffer) == []

    def test_clear_empty_buffer(self):
        """Test clearing an already empty buffer."""
        buffer = SimpleRawBuffer[str, int]()

        assert len(buffer) == 0

        buffer.clear()

        assert len(buffer) == 0

    def test_different_value_types(self):
        """Test buffer with different value types."""
        buffer = SimpleRawBuffer[str, list[int]]()
        buffer.add("list1", [1, 2, 3])
        buffer.add("list2", [4, 5, 6])

        assert buffer["list1"] == [1, 2, 3]
        assert buffer["list2"] == [4, 5, 6]

    def test_different_key_types(self):
        """Test buffer with different key types."""
        buffer = SimpleRawBuffer[int, str]()
        buffer.add(1, "value1")
        buffer.add(2, "value2")

        assert buffer[1] == "value1"
        assert buffer[2] == "value2"

    def test_complex_workflow(self):
        """Test a complex workflow with multiple operations."""
        buffer = SimpleRawBuffer[str, int]()

        # Add some items
        buffer.add("a", 1)
        buffer.add("b", 2)
        buffer.add("c", 3)
        assert len(buffer) == 3

        # Pop one item
        value = buffer.pop("b")
        assert value == 2
        assert len(buffer) == 2

        # Add another item
        buffer.add("d", 4)
        assert len(buffer) == 3

        # Verify remaining items
        assert buffer["a"] == 1
        assert buffer["c"] == 3
        assert buffer["d"] == 4

        # Clear and verify empty
        buffer.clear()
        assert len(buffer) == 0

    def test_len_consistency(self):
        """Test that len() remains consistent with add/pop operations."""
        buffer = SimpleRawBuffer[str, int]()

        # Initially empty
        assert len(buffer) == 0

        # Add items and check length
        for i in range(5):
            buffer.add(f"key{i}", i)
            assert len(buffer) == i + 1

        # Remove items and check length
        for i in range(5):
            buffer.pop(f"key{i}")
            assert len(buffer) == 4 - i

    def test_none_values(self):
        """Test storing None values."""
        buffer = SimpleRawBuffer[str, int | None]()
        buffer.add("none_value", None)
        buffer.add("int_value", 42)

        assert buffer["none_value"] is None
        assert buffer["int_value"] == 42

    def test_empty_string_key(self):
        """Test using empty string as key."""
        buffer = SimpleRawBuffer[str, int]()
        buffer.add("", 42)

        assert buffer[""] == 42
        assert "" in list(buffer.keys())
