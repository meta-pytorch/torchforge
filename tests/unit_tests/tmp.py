import asyncio
import logging

import pytest
from forge.controller import ForgeActor
from forge.controller.service import (
    LeastLoadedRouter,
    Replica,
    ReplicaState,
    RoundRobinRouter,
    ServiceConfig,
    SessionRouter,
)
from forge.types import ProcessConfig
from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Counter(ForgeActor):
    """Test actor that maintains a counter with various endpoints."""

    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        """Increment the counter."""
        self.v += 1

    @endpoint
    async def value(self) -> int:
        """Get the current counter value."""
        return self.v

    @endpoint
    async def fail_me(self):
        """Endpoint that always fails to test error handling."""
        raise RuntimeError("I was asked to fail")

    @endpoint
    async def slow_incr(self):
        """Slow increment to test queueing."""
        await asyncio.sleep(1.0)
        self.v += 1

    @endpoint
    async def add_to_value(self, amount: int, multiplier: int = 1) -> int:
        """Add an amount (optionally multiplied) to the current value."""
        logger.info(f"adding {amount} with {multiplier}")
        self.v += amount * multiplier
        return self.v


async def test():
    service = await Counter.options(procs=1, num_replicas=1).as_service(1)
    # async with service.session() as session:
    #     # All calls within this block use the same replica
    #     result1 = await service.incr.route()
    #     result2 = await service.value.fanout()

    session_id = await service.start_session()
    result = await service.incr.route(sess_id=session_id)
    await service.terminate_session(session_id)


asyncio.run(test())
