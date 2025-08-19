import asyncio

from forge.actors.policy import Policy, PolicyRouter
from forge.controller import ServiceConfig, spawn_service
from forge.data.replay_buffer import ReplayBuffer
from monarch.actor import Actor, endpoint


class Trainer(Actor):
    def __init__(self):
        pass

    @endpoint
    async def train_step(self, batch):
        pass

    @endpoint
    async def update_weights(self):
        pass


class Episode:

    turns = []

    def add_turn(self, turn):
        self.turns.append(turn)

    def add_transform_info(self, key, data):
        setattr(self, key, data)


class ComputeAdvantages(Actor):
    def __call__(self, episode):
        pass


class RefModel(Actor):
    def forward(self, x):
        pass


async def main():
    # ---- Setup services ---- #
    default_service_cfg = ServiceConfig(
        procs_per_replica=1,
        min_replicas=1,
        max_replicas=1,
        default_replicas=1,
    )
    policy = await spawn_service(
        default_service_cfg,
        PolicyRouter,
        policy=Policy(model="Deepseek/Deepseek-v3"),
    )
    trainer = await spawn_service(
        default_service_cfg,
        Trainer,
    )
    replay_buffer = await spawn_service(
        default_service_cfg,
        ReplayBuffer,
        batch_size=4,
        max_policy_age=1,
    )
    dataloader = await spawn_service(
        default_service_cfg,
        ForgeDataset,
        path="gsm8k",
    )
    compute_advantages = await spawn_service(
        default_service_cfg,
        ComputeAdvantages,
    )
    ref_model = await spawn_service(default_service_cfg, RefModel)

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        while True:
            prompt = await dataloader.__next__.call()
            if prompt is None:
                print(f"Dataloader is empty, exiting rollout creation")
                return
            version = await policy.get_current_version.choose()
            episode = Episode()
            async with policy.session(version=version):
                action = await policy.generate.call(prompt)
                episode.add_turn((prompt, action))
            episode.add_advantages(await compute_advantages.__call__.call(episode))
            episode.add_logprobs(await ref_model.forward.call(episode.get_tokens()))
            await replay_buffer.add.call(episode)

    rollout_task = asyncio.create_task(continuous_rollouts())

    async def continuous_training():
        while True:
            batch = await replay_buffer.sample.call()
            if batch is not None:
                await trainer.train_step.call(batch)
                await trainer.update_policy.call()

    training_task = asyncio.create_task(continuous_training())

    await asyncio.gather(rollout_task, training_task)


if __name__ == "__main__":
    asyncio.run(main())
