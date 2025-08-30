import asyncio

from datasets import load_dataset

from forge.actors.policy import Policy, PolicyConfig, SamplingOverrides, WorkerConfig
from forge.actors.reference_actor import HuggingFaceRefModel, TitanRefModel

from forge.controller.actor import ForgeActor
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from monarch.actor import endpoint


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

    # For Torchtitan
    model = "Qwen/Qwen3-1.7B"

    # Spawn Reference "Agents"
    hf_model = await spawn_service(
        ServiceConfig(procs_per_replica=1, num_replicas=1, with_gpus=True),
        HuggingFaceRefModel,
        model_name=model,
    )
    titan_model = await spawn_service(
        ServiceConfig(procs_per_replica=1, num_replicas=1, with_gpus=True),
        TitanRefModel,
    )

    # Spawn Policy for getting responses
    policy = await spawn_service(
        ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=1),
        Policy,
        config=PolicyConfig(
            worker_params=WorkerConfig(model=model),
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
    actions = await policy.generate.choose(prompt)
    for action in actions:
        print("Generated Action tok_ids: ", action.token_ids)

        print("HuggingFace Results")
        hf_logprobs: float = await hf_model.forward.choose(action.token_ids)
        print("HF logprob: ", hf_logprobs)

        print("Titan Results")
        titan_logprobs: float = await titan_model.forward.choose(action.token_ids)
        print("Titan logprob: ", titan_logprobs)
        # TODO: Update forward to convert probs (logits) to logprobs

    await asyncio.gather(
        shutdown_service(policy),
        shutdown_service(dataloader),
        shutdown_service(hf_model),
        shutdown_service(titan_model),
    )


if __name__ == "__main__":
    asyncio.run(main())
