
from monarch.actor import this_host
import asyncio
from forge.actors.policy import Policy, PolicyWorker, SamplingConfig, EngineConfig


async def main():
    def bootstrap():
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

    w = this_host().spawn_procs(
        per_host={"gpus": 2},
        bootstrap=bootstrap)
    p = this_host().spawn_procs(
        per_host={"gpus": 1})

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


asyncio.run(main())