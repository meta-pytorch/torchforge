# test_dataclass_service.py

from dataclasses import asdict, dataclass, field

import pytest
from forge.controller import ForgeActor

from forge.interfaces import Policy as PolicyInterface
from monarch.actor import endpoint


@dataclass
class DataCounter(ForgeActor):
    """ForgeActor implemented as a dataclass."""

    v: int = field(default=1)

    @endpoint
    async def value(self) -> int:
        return self.v


@dataclass
class SimplePolicy(PolicyInterface):
    """Minimal concrete policy dataclass."""

    version: int = 0

    @endpoint
    async def generate(self, request):
        # Just return the request directly as a "dummy action"
        return request

    @endpoint
    async def update_weights(self, policy_version: int):
        # Store the new version
        self.version = policy_version
        return self.version


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_dataclass_as_service_initialization():
    """Test that dataclass actor can be initialized via as_service()."""
    service = await DataCounter.as_service(42)
    try:
        result = await service.value.choose()
        assert result == 42
    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_simple_policy_as_service_and_endpoints():
    """Test that SimplePolicy can be initialized and its endpoints work."""

    # Start service with initial version=1
    service = await SimplePolicy.as_service(version=1)
    try:
        # Initial version should be 1
        v = await service.update_weights.choose(1)
        assert v == 1

        # Call generate — should just echo back the input
        result = await service.generate.choose("obs")
        assert result == "obs"

        # Update weights to version 2
        v2 = await service.update_weights.choose(2)
        assert v2 == 2

    finally:
        await service.shutdown()
