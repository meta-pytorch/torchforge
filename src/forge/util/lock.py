# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Add the ability to use as context manager
class RWLock:
    """
    Basic read-write lock with write lock priority.
    """

    def __init__(self):
        self._cond = asyncio.Condition()
        self._readers = 0

        # Used to indicate that an exclusive lock is held
        self._exclusive = False
        # Number of waiters for an exclusive lock
        self._exclusive_waiters = 0

    async def acquire(self):
        async with self._cond:
            while self._exclusive or self._exclusive_waiters > 0:
                await self._cond.wait()
            self._readers += 1

    async def release(self):
        async with self._cond:
            if self._readers == 0:
                raise RuntimeError("Cannot release an unacquired lock")
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    async def acquire_write_lock(self):
        async with self._cond:
            self._exclusive_waiters += 1
            while self._exclusive or self._readers > 0:
                await self._cond.wait()
            self._exclusive_waiters -= 1
            self._exclusive = True

    async def release_write_lock(self):
        async with self._cond:
            self._exclusive = False
            self._cond.notify_all()
