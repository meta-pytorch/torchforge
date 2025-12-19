from typing import Any

from monarch.actor import ProcMesh


class BaseLauncher:
    async def initialize(self) -> None:
        pass

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        pass

    async def remote_setup(self, procs: ProcMesh) -> None:
        pass
