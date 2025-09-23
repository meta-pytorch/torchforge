# test_dataclass_service.py

from dataclasses import asdict, dataclass, field

import pytest
from forge.controller import ForgeActor
from monarch.actor import endpoint


@dataclass
class DataCounter(ForgeActor):
    """ForgeActor implemented as a dataclass."""

    v: int = field(default=1)

    @endpoint
    async def value(self) -> int:
        return self.v


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
