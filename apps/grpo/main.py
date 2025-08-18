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


async def generate_rollout():
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

    async def continuous_rollouts():
        while True:
            version = await policy.get_current_version.choose()
            episode = await generate_rollout(version)
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
