# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest
from forge.util.lock import Lock


class TestLock:
    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_basic_acquire_release(self):
        """Test basic acquire and release functionality."""
        lock = Lock()
        assert lock._readers == 0
        assert lock._exclusive is False
        assert lock._exclusive_waiters == 0

        await lock.acquire()
        assert lock._readers == 1

        await lock.release()
        assert lock._readers == 0

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_multiple_readers(self):
        """Test that multiple readers can acquire the lock simultaneously."""
        lock = Lock()

        await lock.acquire()
        await lock.acquire()
        await lock.acquire()
        assert lock._readers == 3

        await lock.release()
        await lock.release()
        await lock.release()
        assert lock._readers == 0

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_exclusive_lock_basic(self):
        """Test basic exclusive lock functionality."""
        lock = Lock()
        assert lock._exclusive_waiters == 0

        await lock.acquire_exclusive_lock()
        assert lock._exclusive is True
        assert lock._exclusive_waiters == 0

        await lock.release_exclusive_lock()
        assert lock._exclusive is False

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_exclusive_lock_blocks_readers(self):
        """Test that exclusive lock blocks new readers."""
        lock = Lock()

        await lock.acquire_exclusive_lock()
        assert lock._exclusive is True

        results = []

        async def try_acquire_reader():
            await lock.acquire()
            results.append("reader_acquired")
            await lock.release()

        reader_task = asyncio.create_task(try_acquire_reader())
        await asyncio.sleep(0.1)
        assert len(results) == 0

        await lock.release_exclusive_lock()
        assert lock._exclusive is False

        await reader_task
        assert len(results) == 1
        assert results[0] == "reader_acquired"

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_exclusive_lock_waits_for_readers(self):
        """Test that exclusive lock waits for existing readers to finish."""
        lock = Lock()
        results = []

        reader_release_event = asyncio.Event()

        async def reader_with_delay():
            await lock.acquire()
            results.append("reader_acquired")
            await reader_release_event.wait()
            await lock.release()
            results.append("reader_released")

        async def exclusive_acquirer():
            await lock.acquire_exclusive_lock()
            results.append("exclusive_acquired")
            await lock.release_exclusive_lock()

        # Start reader first
        reader_task = asyncio.create_task(reader_with_delay())
        await asyncio.sleep(0.1)

        # Now try to acquire exclusive lock (should wait)
        exclusive_task = asyncio.create_task(exclusive_acquirer())
        await asyncio.sleep(0.1)

        # At this point, reader should be acquired but exclusive should be waiting
        assert "reader_acquired" in results
        assert "exclusive_acquired" not in results
        assert lock._exclusive is False
        assert lock._exclusive_waiters == 1

        # Signal reader to release
        reader_release_event.set()

        # Wait for both tasks to complete
        await reader_task
        await exclusive_task

        assert results == ["reader_acquired", "reader_released", "exclusive_acquired"]
        assert lock._readers == 0
        assert lock._exclusive is False

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_concurrent_readers_before_exclusive(self):
        """Test multiple readers before exclusive lock acquisition."""
        lock = Lock()
        results = []
        reader_release_events = [asyncio.Event() for _ in range(3)]

        async def reader_with_id(reader_id, release_event):
            await lock.acquire()
            results.append(f"reader_{reader_id}_acquired")
            await release_event.wait()
            await lock.release()
            results.append(f"reader_{reader_id}_released")

        # Start multiple readers
        reader_tasks = [
            asyncio.create_task(reader_with_id(i, reader_release_events[i]))
            for i in range(3)
        ]
        await asyncio.sleep(0.1)  # Let all readers acquire

        # Now try to acquire exclusive lock
        exclusive_task = asyncio.create_task(lock.acquire_exclusive_lock())
        await asyncio.sleep(0.1)

        assert lock._exclusive is False
        assert lock._exclusive_waiters == 1
        assert lock._readers == 3

        # Release readers one by one
        for event in reader_release_events:
            event.set()
            await asyncio.sleep(0.05)  # Small delay between releases

        # Wait for all tasks to complete
        await asyncio.gather(*reader_tasks)
        await exclusive_task

        # Verify exclusive lock was acquired after all readers released
        assert lock._readers == 0
        assert lock._exclusive is True
        assert lock._exclusive_waiters == 0

        await lock.release_exclusive_lock()
        assert lock._exclusive is False

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_reader_counter_accuracy(self):
        """Test that reader counter is accurate with rapid acquire/release."""
        lock = Lock()

        async def rapid_acquire_release(count):
            for _ in range(count):
                await lock.acquire()
                await lock.release()

        # Run multiple tasks that rapidly acquire/release
        tasks = [asyncio.create_task(rapid_acquire_release(10)) for _ in range(5)]

        await asyncio.gather(*tasks)

        # Final state should be clean
        assert lock._readers == 0

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_exclusive_lock_isolation(self):
        """Test that exclusive lock provides proper isolation."""
        lock = Lock()
        shared_resource = {"value": 0}

        async def reader_task(task_id):
            await lock.acquire()
            initial_value = shared_resource["value"]
            await asyncio.sleep(0.01)  # Small delay
            # Value shouldn't change during read (when exclusive isn't modifying)
            assert shared_resource["value"] == initial_value
            await lock.release()

        async def writer_task():
            await lock.acquire_exclusive_lock()
            # Simulate writing to shared resource
            for i in range(5):
                shared_resource["value"] = i
                await asyncio.sleep(0.01)
            await lock.release_exclusive_lock()

        # Start writer and readers concurrently
        writer = asyncio.create_task(writer_task())
        readers = [asyncio.create_task(reader_task(i)) for i in range(3)]
        await asyncio.gather(writer, *readers)

        assert shared_resource["value"] == 4
        assert lock._readers == 0
        assert lock._exclusive is False

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_release_without_acquire_error_handling(self):
        """Test behavior when releasing without acquiring."""
        lock = Lock()
        with pytest.raises(Exception):
            await lock.release()

    @pytest.mark.timeout(10)
    @pytest.mark.asyncio
    async def test_multiple_exclusive_waiters(self):
        """Test that multiple exclusive waiters are handled in and only one exclusive
        lock is held at a time."""
        lock = Lock()
        results = []

        async def exclusive_task(task_id, event_start, event_end):
            await event_start.wait()
            await lock.acquire_exclusive_lock()
            results.append(f"exclusive_{task_id}_acquired")
            await event_end.wait()
            await lock.release_exclusive_lock()
            results.append(f"exclusive_{task_id}_released")

        # Events to control when each exclusive task starts and ends
        start_events = [asyncio.Event(), asyncio.Event()]
        end_events = [asyncio.Event(), asyncio.Event()]

        tasks = [
            asyncio.create_task(exclusive_task(i, start_events[i], end_events[i]))
            for i in range(2)
        ]

        # Start the first exclusive waiter
        start_events[0].set()
        await asyncio.sleep(0.1)
        assert lock._exclusive is True
        assert lock._exclusive_waiters == 0

        # Start the second waiter
        start_events[1].set()
        await asyncio.sleep(0.1)
        assert lock._exclusive_waiters == 1

        # Release the first exclusive lock
        end_events[0].set()
        await asyncio.sleep(0.1)

        assert results[0] == "exclusive_0_acquired"
        assert results[1] == "exclusive_0_released"
        assert results[2] == "exclusive_1_acquired"
        assert lock._exclusive is True
        assert lock._exclusive_waiters == 0

        # Release the second exclusive lock
        end_events[1].set()
        await asyncio.sleep(0.1)
        assert results[3] == "exclusive_1_released"
        assert lock._exclusive is False
        assert lock._exclusive_waiters == 0
