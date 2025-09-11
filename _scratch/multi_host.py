import asyncio

from monarch.actor import Actor, endpoint, get_or_spawn_controller, ActorError, this_host, HostMesh, current_rank 
from monarch.tools.components import hyperactor
from monarch.tools import commands
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch._src.actor.shape import Shape, NDSlice
from monarch.tools.config import Config


class Provisioner:
    def __init__(self):
        pass

    async def initialize(self):
        pass

    async def create_allocator(self, name: str, num_hosts: int) -> RemoteAllocator:
        print(f"for {name}, creating {num_hosts}")
        appdef = hyperactor.host_mesh(
            image="test",
            meshes=[f"{name}:{num_hosts}:gpu.small"]
        )
        for role in appdef.roles:
            role.resource.memMB = 2062607
            role.resource.cpu = 128
            role.resource.gpu = 8

        server_config = Config(
            scheduler="slurm",
            appdef=appdef,
            workspace="",
        )
        print("Creating server")
        server_info = await commands.get_or_create(
            "forge_job",
            server_config,
            force_restart=False,
        )
        return RemoteAllocator(
            world_id=name,
            initializer=TorchXRemoteAllocInitializer(server_info.server_handle),
        ), f"slurm:///{server_info.name}"




class TestActor(Actor):
    def __init__(self, id: str):
        self.id = id

    @endpoint
    async def test(self):
        import socket
        print(f"id: {self.id}, host: {socket.gethostname()}")
        return current_rank()

    @endpoint
    async def call_other(self, other):
        await other.test.choose()


async def get_allocator(num_hosts: int, name: str):
    print(f"for {name}, creating {num_hosts}")
    appdef = hyperactor.host_mesh(
        image="test",
        meshes=[f"{name}:{num_hosts}:gpu.small"]
    )
    for role in appdef.roles:
        role.resource.memMB = 2062607
        role.resource.cpu = 128
        role.resource.gpu = 8

    server_config = Config(
        scheduler="slurm",
        appdef=appdef,
        workspace="",
    )
    print("Creating server")
    server_info = await commands.get_or_create(
        "forge_job",
        server_config,
        force_restart=False,
    )
    return RemoteAllocator(
        world_id=name,
        initializer=TorchXRemoteAllocInitializer(server_info.server_handle),
    ), f"slurm:///{server_info.name}"


async def main():
    print("Running")
    print("Creating allocator a")
    alloc_a, s_a = await get_allocator(num_hosts=1, name="a")
    print("Creating allocator b")
    alloc_b, s_b = await get_allocator(num_hosts=1, name="b")
    try:
        ha = HostMesh(Shape(["hosts"], NDSlice.new_row_major([1])), alloc_a)
        hb = HostMesh(Shape(["hosts"], NDSlice.new_row_major([1])), alloc_b)

        ma = ha.spawn_procs(per_host={"gpus": 1})
        mb = hb.spawn_procs(per_host={"gpus": 1})

        ta = await ma.spawn("test", TestActor, id="a")
        tb = await mb.spawn("test", TestActor, id="b")
        print(f"a: {await ta.test.call()}")
        print(f"b: {await tb.test.call()}")
    finally:
        commands.kill(s_a)
        commands.kill(s_b)


asyncio.run(main())
