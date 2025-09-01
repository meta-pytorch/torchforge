"""Asynchronous reader-writer lock implementation for concurrent access control."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator


class AsyncRWLock:
    """
    An asynchronous reader-writer lock that allows multiple readers or a single writer.

    This implementation prioritizes writers - when a writer is waiting, new read requests
    are rejected to prevent writer starvation.

    Usage:
        lock = AsyncRWLock()

        async with lock.read_lock():
            # Multiple readers can access concurrently
            pass

        async with lock.write_lock():
            # Only one writer can access at a time
            pass
    """

    def __init__(self) -> None:
        """Initialize the reader-writer lock."""
        self._readers = 0
        self._writer = False
        self._writer_waiting = False
        self._condition = asyncio.Condition()

    async def acquire_read(self) -> None:
        """
        Acquire a read lock.

        Raises:
            RuntimeError: If a writer is waiting (prevents writer starvation).
        """
        async with self._condition:
            if self._writer_waiting:
                raise RuntimeError("Read lock request canceled: writer is waiting")

            while self._writer:
                await self._condition.wait()

            self._readers += 1

    async def release_read(self) -> None:
        """Release a read lock."""
        async with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_write(self) -> None:
        """
        Acquire a write lock.

        Waits until no other readers or writers are active.
        """
        async with self._condition:
            self._writer_waiting = True

            while self._writer or self._readers > 0:
                await self._condition.wait()

            self._writer = True
            self._writer_waiting = False

    async def release_write(self) -> None:
        """Release a write lock."""
        async with self._condition:
            self._writer = False
            self._condition.notify_all()

    @asynccontextmanager
    async def read_lock(self) -> AsyncGenerator[None, None]:
        """
        Context manager for acquiring a read lock.

        Example:
            async with lock.read_lock():
                # Multiple readers can access this section concurrently
                data = shared_resource.read()
        """
        await self.acquire_read()
        try:
            yield
        finally:
            await self.release_read()

    @asynccontextmanager
    async def write_lock(self) -> AsyncGenerator[None, None]:
        """
        Context manager for acquiring a write lock.

        Example:
            async with lock.write_lock():
                # Only one writer can access this section at a time
                shared_resource.write(data)
        """
        await self.acquire_write()
        try:
            yield
        finally:
            await self.release_write()
