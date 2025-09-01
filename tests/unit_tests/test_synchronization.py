"""Tests for AsyncRWLock synchronization primitives."""

import asyncio

import pytest
from forge.controller.synchronization import AsyncRWLock


class TestAsyncRWLock:
    """Test suite for AsyncRWLock reader-writer lock implementation."""

    @pytest.fixture
    def lock(self):
        """Create a fresh AsyncRWLock for each test."""
        return AsyncRWLock()

    @pytest.mark.asyncio
    async def test_writer_blocks_readers(self, lock):
        """Test that when writer has acquired the lock, no reader can read."""
        writer_acquired = asyncio.Event()
        reader_attempted = asyncio.Event()
        reader_acquired = asyncio.Event()
        writer_released = asyncio.Event()

        async def writer_task():
            """Writer that holds the lock for a period."""
            async with lock.write_lock():
                writer_acquired.set()
                # Wait for reader to attempt access
                await reader_attempted.wait()
                # Hold the lock briefly to ensure reader is blocked
                await asyncio.sleep(0.1)
            writer_released.set()

        async def reader_task():
            """Reader that attempts to acquire lock after writer."""
            # Wait for writer to acquire lock first
            await writer_acquired.wait()
            reader_attempted.set()

            # This should block until writer releases
            async with lock.read_lock():
                reader_acquired.set()

        # Start both tasks
        writer_task_handle = asyncio.create_task(writer_task())
        reader_task_handle = asyncio.create_task(reader_task())

        # Wait for writer to acquire and reader to attempt
        await writer_acquired.wait()
        await reader_attempted.wait()

        # Reader should not have acquired yet (writer still holds lock)
        assert not reader_acquired.is_set()

        # Wait for writer to finish
        await writer_task_handle
        assert writer_released.is_set()

        # Now reader should be able to acquire
        await reader_task_handle
        assert reader_acquired.is_set()

    @pytest.mark.asyncio
    async def test_waiting_writer_blocks_new_readers(self, lock):
        """Test that when writer is waiting for lock, new readers can't acquire lock."""
        reader1_acquired = asyncio.Event()
        writer_waiting = asyncio.Event()
        reader2_attempted = asyncio.Event()
        reader2_failed = asyncio.Event()

        async def reader1_task():
            """First reader that holds the lock."""
            async with lock.read_lock():
                reader1_acquired.set()
                # Hold lock to make writer wait
                await asyncio.sleep(0.2)

        async def writer_task():
            """Writer that waits for reader1 to finish."""
            # Wait for reader1 to acquire lock
            await reader1_acquired.wait()
            writer_waiting.set()

            # This will block until reader1 releases
            async with lock.write_lock():
                pass

        async def reader2_task():
            """Second reader that should be rejected due to waiting writer."""
            # Wait for writer to start waiting
            await writer_waiting.wait()
            reader2_attempted.set()

            # This should raise RuntimeError
            try:
                async with lock.read_lock():
                    # Should not reach here
                    assert (
                        False
                    ), "Reader2 should not acquire lock when writer is waiting"
            except RuntimeError as e:
                assert "writer is waiting" in str(e)
                reader2_failed.set()

        # Start all tasks
        reader1_task_handle = asyncio.create_task(reader1_task())
        writer_task_handle = asyncio.create_task(writer_task())
        reader2_task_handle = asyncio.create_task(reader2_task())

        # Wait for sequence of events
        await reader1_acquired.wait()
        await writer_waiting.wait()
        await reader2_attempted.wait()
        await reader2_failed.wait()

        # Verify reader2 was properly rejected
        assert reader2_failed.is_set()

        # Clean up
        await reader1_task_handle
        await writer_task_handle
        await reader2_task_handle

    @pytest.mark.asyncio
    async def test_writer_blocks_second_writer(self, lock):
        """Test that when writer has acquired lock, second writer can't write."""
        writer1_acquired = asyncio.Event()
        writer2_attempted = asyncio.Event()
        writer2_acquired = asyncio.Event()
        writer1_released = asyncio.Event()

        async def writer1_task():
            """First writer that holds the lock."""
            async with lock.write_lock():
                writer1_acquired.set()
                # Wait for writer2 to attempt access
                await writer2_attempted.wait()
                # Hold the lock to ensure writer2 is blocked
                await asyncio.sleep(0.1)
            writer1_released.set()

        async def writer2_task():
            """Second writer that should wait for first writer."""
            # Wait for writer1 to acquire lock
            await writer1_acquired.wait()
            writer2_attempted.set()

            # This should block until writer1 releases
            async with lock.write_lock():
                writer2_acquired.set()

        # Start both tasks
        writer1_task_handle = asyncio.create_task(writer1_task())
        writer2_task_handle = asyncio.create_task(writer2_task())

        # Wait for writer1 to acquire and writer2 to attempt
        await writer1_acquired.wait()
        await writer2_attempted.wait()

        # Writer2 should not have acquired yet (writer1 still holds lock)
        assert not writer2_acquired.is_set()

        # Wait for writer1 to finish
        await writer1_task_handle
        assert writer1_released.is_set()

        # Now writer2 should be able to acquire
        await writer2_task_handle
        assert writer2_acquired.is_set()

    @pytest.mark.asyncio
    async def test_multiple_readers_concurrent_access(self, lock):
        """Test that multiple readers can access the resource concurrently."""
        num_readers = 3
        readers_acquired = []
        all_readers_active = asyncio.Event()

        async def reader_task(reader_id):
            """Reader task that signals when it has acquired the lock."""
            async with lock.read_lock():
                readers_acquired.append(reader_id)
                # Wait for all readers to be active
                if len(readers_acquired) == num_readers:
                    all_readers_active.set()
                await all_readers_active.wait()
                # Hold lock briefly to ensure concurrent access
                await asyncio.sleep(0.1)

        # Start multiple reader tasks
        reader_tasks = [asyncio.create_task(reader_task(i)) for i in range(num_readers)]

        # Wait for all readers to complete
        await asyncio.gather(*reader_tasks)

        # Verify all readers were able to acquire locks
        assert len(readers_acquired) == num_readers
        assert set(readers_acquired) == set(range(num_readers))

    @pytest.mark.asyncio
    async def test_lock_state_after_exception(self, lock):
        """Test that lock state is properly cleaned up after exceptions."""
        reader_acquired = asyncio.Event()
        exception_handled = asyncio.Event()

        async def reader_with_exception():
            """Reader that raises an exception while holding lock."""
            try:
                async with lock.read_lock():
                    reader_acquired.set()
                    raise ValueError("Test exception")
            except ValueError:
                exception_handled.set()

        async def subsequent_reader():
            """Reader that should be able to acquire lock after exception."""
            await exception_handled.wait()
            async with lock.read_lock():
                # Should be able to acquire lock successfully
                pass

        # Run tasks
        await asyncio.gather(
            asyncio.create_task(reader_with_exception()),
            asyncio.create_task(subsequent_reader()),
        )

        # Verify exception was handled and lock is available
        assert exception_handled.is_set()

        # Verify lock state is clean (no readers or writers)
        assert lock._readers == 0
        assert lock._writer is False
        assert lock._writer_waiting is False

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, lock):
        """Test that context managers properly clean up lock state."""
        # Test read lock cleanup
        async with lock.read_lock():
            assert lock._readers == 1
        assert lock._readers == 0

        # Test write lock cleanup
        async with lock.write_lock():
            assert lock._writer is True
        assert lock._writer is False

        # Test nested operations don't interfere
        async with lock.read_lock():
            assert lock._readers == 1
            async with lock.read_lock():
                assert lock._readers == 2
            assert lock._readers == 1
        assert lock._readers == 0
