import sys
from functools import partial
from typing import Any

import forge.titan_fork.train_spec as forge_train_spec

import torch

from forge.config.parse import parse
from forge.titan_fork.config_manager import ForgeJobConfig

from forge.titan_fork.train import ForgeEngine
from omegaconf import DictConfig, OmegaConf

from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims

# stubs for now
Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any


class ForgeSFTRecipe(ForgeEngine):
    job_config: ForgeJobConfig
    train_spec: forge_train_spec.ForgeTrainSpec
    parallel_dims: ParallelDims
    model: list[nn.Module]
    loss_fn: LossFunction
    optimizer: OptimizersContainer
    lr_scheduler: LRSchedulersContainer
    checkpointer: Checkpointer
    tokenizer: Tokenizer
    train_dataloader: Dataloader
    # val_dataloader: Dataloader
    metric_logger: MetricLogger
    profiler: Profiler
    device: torch.device
    step: int

    def __init__(self, train_config: ForgeJobConfig):
        self.train_config = train_config
        self.current_step = 0
        self.num_training_steps = 1000  # Example value, adjust as needed
        self.gradient_accumulation_steps = 1  # Example value, adjust as needed
        print(train_config)
        super().__init__(train_config)

    def setup(self):
        self.train_dataloader = self.setup_data(
            self.train_config.train_dataset_config,
            self.train_config.train_dataloader_config,
            self.train_config.packing_config,
        )
        # self.val_dataloader = self.setup_data(
        #     self.train_config.val_dataset_config,
        #     self.train_config.val_dataloader_config,
        #     self.train_config.packing_config,
        # )
        self.checkpointer.load(self.train_config.checkpoint_config)
        self.profiler = self.setup_profiler(self.train_config.profiler_config)
        self.logger = self.setup_logger(self.train_config.logger_config)

    # TODO: this needs to be hooked into config system and generalized
    def setup_data(self, dataset_config, dataloader_config, packing_config):
        tokenizer = HuggingFaceModelTokenizer()
        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            load_dataset_kwargs={"source": "yahma/alpaca-cleaned"},
        )
        min_seq_len_divisor = 2 * self.parallel_dims.tp * self.parallel_dims.cp
        collate_fn = partial(padded_collate_fn, pad_to_multiple_of=min_seq_len_divisor)
        sampler = StatefulDistributedSampler(
            dataset, num_replicas=self.parallel_dims.dp, shuffle=True, seed=0
        )
        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_fn,
        )

        # Ultimately we probably want something like this
        # packer = build_packing_strategy(packing_config)
        # dataset = build_dataset(dataset_config)
        # dataloader = build_dataloader(dataloader_config, dataset, packer)
        return dataloader

    # def validate(self) -> None:
    #     # Placeholder for now
    #     pass

    def forward_backward(self) -> None:
        pass
        # context_parallel_manager = create_context_parallel_manager(...)
        # Implement the forward and backward pass logic

    def train_step(self, batch) -> None:
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        loss = self.forward_backward_step(batch)

        self.optimizer.step()
        self.lr_scheduler.step()

    def train(self) -> None:
        self.optimizer.zero_grad()
        while self.current_step < self.num_training_steps:
            batch = next(self.dataloader)
            self.train_step(batch)
            self.profiler.step()
            self.current_step += 1
            # if self.current_step % self.train_config.val_every_n_steps == 0:
            #     self.validate()
            if (
                self.current_step
                % self.train_config.checkpoint_config.save_every_n_steps
                == 0
            ):
                self.checkpointer.save()

    def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metric_logger:
            self.metric_logger.close()


@parse
def recipe_main(cfg: DictConfig) -> None:
    # TODO: this is a hack to get the defaults from ForgeJobConfig
    default_cfg = ForgeJobConfig()
    cfg = OmegaConf.merge(default_cfg, cfg)
    recipe = ForgeSFTRecipe(cfg)
    recipe.setup()
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
