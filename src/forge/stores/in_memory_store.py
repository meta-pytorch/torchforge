# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from forge.data_models.api import Store


class InMemoryStore(Store):
    """
    Simple in-memory key-value store implementation.
    Stores values in a Python dictionary, keyed by strings.
    Suitable for testing, prototyping, or single-process use cases.
    """

    def __init__(self):
        self._store = {}

    def put(self, key: str, value):
        """
        Store a value under the specified key.
        Args:
            key (str): The key under which to store the value.
            value: The value to store.
        """
        self._store[key] = value

    def get(self, key: str):
        """
        Retrieve the value associated with the specified key.
        Args:
            key (str): The key for which to retrieve the value.
        Returns:
            The value associated with the key, or None if not found.
        """
        return self._store.get(key, None)
