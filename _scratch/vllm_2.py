
from monarch.actor import this_host, HostMesh
import asyncio
from forge.actors.policy import Policy, PolicyWorker, SamplingConfig, EngineConfig
from monarch.tools.components import hyperactor
from monarch.tools import commands
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch.tools.config import Config
from monarch._src.actor.shape import Shape, NDSlice
import monarch


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
        workspace=monarch.tools.config.workspace.Workspace(dirs=[""]),
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


def _get_port() -> str:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        addr = s.getsockname()
        port = addr[1]
        return str(port)


async def main():
    allocator, server_name = await get_allocator(num_hosts=1, name="vllm")

    def bootstrap():
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

    worker_host = HostMesh(Shape(["hosts"], NDSlice.new_row_major([1])), allocator)
    w = worker_host.spawn_procs(
        per_host={"gpus": 2},
        bootstrap=bootstrap)
    p = this_host().spawn_procs(
        per_host={"gpus": 1})

    try:
        engine_config = EngineConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            enforce_eager=True,
        )
        sampling_config = SamplingConfig(
            n=2,
            guided_decoding=False,
            max_tokens=512,
        )
        workers = await w.spawn(
            "vllm_worker", PolicyWorker, vllm_args=engine_config
        )
        policy = await p.spawn(
            "vllm", Policy, engine_config=engine_config, sampling_config=sampling_config, policy_worker=workers, store=None
        )

        print("controller: ", policy)
        print("worker: ", workers)
        await policy.setup.call()
        results = await policy.generate.choose(prompt="Hello world")
        print("Results: ", results)
    finally:
        commands.kill(server_name)



asyncio.run(main())