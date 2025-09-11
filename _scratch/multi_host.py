import asyncio

import monarch
from monarch.actor import Actor, endpoint, get_or_spawn_controller, ActorError, this_host, HostMesh, current_rank 
from monarch.tools.components import hyperactor
from monarch.tools import commands
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch._src.actor.shape import Shape, NDSlice
from monarch.tools.config import Config
import functools
import socket
import uuid


def _get_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        addr = s.getsockname()
        port = addr[1]
        return str(port)


class _SetupActor(Actor):
    @endpoint
    def get_info(self) -> [str, str]:
        return socket.gethostname(), _get_port()


class Provisioner:
    def __init__(self):
        self._server_names = []

    async def create_host_mesh(self, name: str, num_hosts: int) -> RemoteAllocator:
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
            workspace=monarch.tools.config.workspace.Workspace(dirs=['']),
        )
        print("Creating server")
        server_info = await commands.get_or_create(
            "forge_job",
            server_config,
            force_restart=False,
        )
        alloc = RemoteAllocator(
            world_id=name,
            initializer=TorchXRemoteAllocInitializer(server_info.server_handle),
        )
        self._server_names.append(f"slurm:///{server_info.name}")
        return HostMesh(Shape(["hosts"], NDSlice.new_row_major([num_hosts])), alloc)

    async def get_proc_mesh(self, num_procs: int, with_gpus: bool = False, num_hosts: int | None = None):
        # TODO - issues/144
        # Known issues:
        # - We don't do any intelligent resource management here, just assuming every
        # resource asked for is placed on its own host.
        def bootstrap(num_procs: int, with_gpus: bool):
            if with_gpus:
                # These env variables will work for single host, but on multi-host
                # we will need to override env variables.
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_procs)))
                os.environ["MASTER_ADDR"] = socket.gethostname()
                os.environ["MASTER_PORT"] = _get_port()

        if num_hosts:
            host_mesh = await self.create_host_mesh(name=f"alloc-{num_hosts}", num_hosts=num_hosts)
        else:
            host_mesh = this_host()
        procs = host_mesh.spawn_procs(
            per_host={"gpus": num_procs},
            bootstrap=functools.partial(
                bootstrap, num_procs=num_procs, with_gpus=with_gpus),
        )
        setup = await procs.spawn(f"setup-{uuid.uuid1()}", _SetupActor)
        # Pick a random host/port, we'll feed this in afterwards
        # Once we have true HostMesh support, we can do this on proc 0 of each host
        # then spin up the proc meshes with the environment afterwards.
        hostname, port = await setup.get_info.choose()
        procs._host = hostname
        procs._addr = port
        return procs

    async def shutdown(self):
        for server_name in self._server_names:
            print(f"shutting down {server_name}")
            commands.kill(server_name)

_provisioner: Provisioner | None = None

def initialize():
    global _provisioner
    _provisioner = Provisioner()

def _get_provisioner():
    global _provisioner
    if not _provisioner:
        raise AssertionError("Provisioner was not initialized")
    return _provisioner


async def get_procs(num_procs: int, with_gpus: bool = False, num_hosts: int | None= None):
    # note - num_hosts = None indicates use local
    return await _get_provisioner().get_proc_mesh(num_procs=num_procs, with_gpus=with_gpus, num_hosts=num_hosts)


async def shutdown():
    await _get_provisioner().shutdown()


class TestActor(Actor):
    def __init__(self, id: str):
        self.id = id

    @endpoint
    async def test(self):
        import socket
        import os
        return self.id, socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES", None)

    @endpoint
    async def call_other(self, other):
        await other.test.choose()


async def main():
    initialize()
    print("Running")
    try:
        ma = await get_procs(num_procs=1, with_gpus=True, num_hosts=1)
        mb = await get_procs(num_procs=1)
        ta = await ma.spawn("test", TestActor, id="a")
        tb = await mb.spawn("test", TestActor, id="b")
        print(f"a: {await ta.test.call_one()}")
        print(f"b: {await tb.test.call_one()}")
    finally:
        await shutdown()


asyncio.run(main())
