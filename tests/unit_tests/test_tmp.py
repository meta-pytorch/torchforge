# test_dataclass_service.py

from dataclasses import dataclass, field

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

    version: int = field(default=1)
    enabled: bool = field(default=True)

    @endpoint
    async def generate(self, request):
        # Just return the request directly as a "dummy action"
        return request

    @endpoint
    async def update_weights(self, policy_version: int):
        # Store the new version
        self.version = policy_version
        return self.version

    @endpoint
    async def get_enabled(self) -> bool:
        """Get the enabled status."""
        return self.enabled

    @endpoint
    async def get_version(self) -> int:
        """Get the current version."""
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

    # Start service with initial version=1 and default enabled=True
    service = await SimplePolicy.as_service(version=1)
    try:
        # Check that the default enabled field is True
        enabled_status = await service.get_enabled.choose()
        assert enabled_status is True

        # Initial version should be 1
        initial_version = await service.get_version.choose()
        assert initial_version == 1

        # Update weights to version 2
        v = await service.update_weights.choose(2)
        assert v == 2

        # Verify version was updated
        updated_version = await service.get_version.choose()
        assert updated_version == 2

        # Call generate — should just echo back the input
        result = await service.generate.choose("obs")
        assert result == "obs"

    finally:
        await service.shutdown()

    # Test with explicit enabled=False
    service2 = await SimplePolicy.as_service(version=3, enabled=False)
    try:
        # Check that the enabled field is False
        enabled_status2 = await service2.get_enabled.choose()
        assert enabled_status2 is False

        # Verify version is set correctly
        version2 = await service2.get_version.choose()
        assert version2 == 3

    finally:
        await service2.shutdown()
