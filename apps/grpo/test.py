import asyncio

from datasets import load_dataset

from forge.actors.policy import Policy, PolicyConfig, SamplingOverrides, WorkerConfig
from forge.actors.reference_actor import HuggingFaceRefModel, RefModel, TitanRefModel

from forge.controller.actor import ForgeActor
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from monarch.actor import endpoint
from torchtitan.config.job_config import Model


class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    def __init__(
        self, path: str, config_name: str, split: str, streaming: bool, **kwargs
    ):
        super().__init__()

        def gsm8k_to_messages(sample):
            question = sample["question"]
            full_answer: str = sample["answer"]
            answer = full_answer.split("#### ")[1]
            return {"question": question, "answer": answer}

        ds = load_dataset(path, config_name, split=split, streaming=streaming)
        ds = ds.map(gsm8k_to_messages)
        ds = ds.shuffle()
        self._iterator = iter(ds)

    @endpoint
    async def __next__(self) -> dict[str, str] | None:
        return next(self._iterator)


# Sandbox; will be removed
async def main():
    group_size = 1

    vllm_model = "Qwen/Qwen3-0.6B"
    titan_model = Model(name="qwen3", flavor="0.6B")

    # vllm_model = "meta-llama/Meta-Llama-3.1-8B"
    # titan_model = Model()  # Defaults to LLama

    # Spawn Reference "Agents"
    # # Joe
    # hf_model = await spawn_service(
    #     ServiceConfig(procs_per_replica=1, num_replicas=1, with_gpus=True),
    #     HuggingFaceRefModel,
    #     model_name=model,
    # )

    # # Philip
    # hf_model = await spawn_service(
    #     ServiceConfig(procs_per_replica=1, num_replicas=1, with_gpus=True),
    #     RefModel,
    #     model_name=model,
    # )

    titan_model = await spawn_service(
        ServiceConfig(procs_per_replica=1, num_replicas=1, with_gpus=True),
        TitanRefModel,
        model=titan_model,  # Defaults to LLama
    )

    # Spawn Policy for getting responses
    policy = await spawn_service(
        ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=1),
        Policy,
        config=PolicyConfig(
            worker_params=WorkerConfig(model=vllm_model),
            sampling_params=SamplingOverrides(num_samples=group_size, max_tokens=16),
        ),
    )

    # Load Dataset
    dataloader = await spawn_service(
        ServiceConfig(procs_per_replica=1, num_replicas=1),
        DatasetActor,
        path="openai/gsm8k",
        config_name="main",
        split="train",
        streaming=True,
    )
    sample = await dataloader.__next__.choose()
    prompt, target = sample["question"], sample["answer"]
    print("Sample: ", sample)

    # Generate output from policy, then pass to reference agents
    responses = await policy.generate.choose(prompt)
    actions = responses.outputs
    for action in actions:
        request_tokens = responses.prompt_token_ids
        response_tokens = action.token_ids

        print("request_tokens: ", request_tokens)
        print("response_tokens: ", response_tokens)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print("HuggingFace Results")
        # hf_logprobs = await hf_model.forward.choose(
        #     request=request_tokens, response=response_tokens
        # )
        # print("HF logprob: ", hf_logprobs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        await asyncio.gather(
            shutdown_service(policy),
            shutdown_service(dataloader),
            # shutdown_service(hf_model),
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("Titan Results")
        titan_logprobs: float = await titan_model.forward.choose(
            request=request_tokens, response=response_tokens
        )
        print("Titan logprob: ", titan_logprobs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # await shutdown_service(titan_model)


async def test_titan():

    import math

    import os

    from monarch.actor import current_rank, current_size, endpoint

    rank = current_rank().rank
    size = math.prod(current_size().values())

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    env = {
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "LOCAL_WORLD_SIZE": str(size),
        "GROUP_RANK": str(size),
        "GROUP_WORLD_SIZE": str(size),
        "ROLE_RANK": str(rank),
        "ROLE_WORLD_SIZE": str(size),
        "ROLE_NAME": "rank",
        "WORLD_SIZE": str(size),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    os.environ.update(env)

    from torchtitan.experiments.forge.engine import ForgeEngine
    from torchtitan.experiments.forge.job_config import ForgeJobConfig

    config = {
        "model": Model(
            name="qwen3",
            flavor="0.6B",
            hf_assets_path="./tests/assets/tokenizer",
            tokenizer_path=None,
            converters=[],
            print_after_conversion=False,
        ),
        # "parallelism": Parallelism(
        #     data_parallel_replicate_degree=1,
        #     enable_compiled_autograd=False,
        #     data_parallel_shard_degree=-1,
        #     fsdp_reshard_after_forward="default",
        #     tensor_parallel_degree=1,
        #     disable_loss_parallel=False,
        #     enable_async_tensor_parallel=False,
        #     pipeline_parallel_degree=1,
        #     pipeline_parallel_split_points=[],
        #     module_fqns_per_model_part=None,
        #     pipeline_parallel_first_stage_less_layers=1,
        #     pipeline_parallel_last_stage_less_layers=1,
        #     pipeline_parallel_layers_per_stage=None,
        #     pipeline_parallel_schedule="1F1B",
        #     pipeline_parallel_schedule_csv="",
        #     pipeline_parallel_microbatch_size=1,
        #     context_parallel_degree=1,
        #     context_parallel_rotate_method="allgather",
        #     expert_parallel_degree=1,
        # ),
    }

    engine = ForgeEngine(ForgeJobConfig(**config))
    model_part = engine.model_parts[0]
    print(model_part)


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(test_titan())
