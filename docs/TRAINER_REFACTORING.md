# Trainer Refactoring RFC: Unified TitanTrainer

## Overview

This RFC defines the target class hierarchy for unifying SFT and RL trainers.

---

## Target Class Hierarchy

```
ForgeActor (base actor)
    │
    └── TitanTrainer (generic trainer with ForgeEngine composition)
            │
            └── ForgeSFTRecipe (SFT-specific: data loading, eval, training loop)

ForgeEngine (standalone training engine - used via composition, not inheritance)
```

---

## TitanTrainer: The Base Trainer

```python
@dataclass
class TitanTrainer(ForgeActor):
    """Base trainer that wraps ForgeEngine with composition pattern.

    Provides reusable components for any training workload:
    - _setup_engine(): Engine initialization
    - rank_should_record_loss: Only last PP stage records loss
    - record_batch_metrics(): Generic batch metric recording
    - setup_metric_logger(): Metric logger initialization
    - forward_backward(): Forward/backward with PP+CP support (uses engine.loss_fn)
    - forward_backward_rl(): Forward/backward for RL (uses custom self.loss)
    - train_step_sft(): SFT-style training step
    - train_step(): RL-style training step (endpoint)
    - push_weights(): Push weights to torchstore
    - cleanup(): Cleanup resources

    Uses COMPOSITION: access ForgeEngine via self.engine.X
    """

    # === ForgeJobConfig fields ===
    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(default_factory=ActivationCheckpoint)
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)

    # === Non-JobConfig fields ===
    loss: Callable = lambda logits, **targets: logits  # Custom loss for RL

    # === Engine (set during setup) ===
    engine: ForgeEngine  # Composition - not inheritance!
    rank_should_record_loss: bool

    # === Lifecycle Methods ===

    @endpoint
    async def setup(self):
        self._setup_engine()

    def _setup_engine(self):
        """Initialize ForgeEngine. Non-endpoint helper for subclasses."""
        self.engine = ForgeEngine(ForgeJobConfig(...))
        # Set rank_should_record_loss based on PP stage
        self.rank_should_record_loss = True
        if hasattr(self.engine, "pp_has_last_stage") and not self.engine.pp_has_last_stage:
            self.rank_should_record_loss = False

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()

    # === Reusable Utilities ===

    async def setup_metric_logger(self):
        """Get or create metric logger."""
        return await get_or_create_metric_logger()

    def record_batch_metrics(self, data_metrics: list):
        """Record metrics from batch."""
        for metric in data_metrics:
            record_metric(metric.key, metric.value, metric.reduction)

    # === Forward/Backward Passes ===

    def forward_backward(
        self,
        input_dict: dict[str, Tensor],
        labels: Tensor,
        skip_backward: bool = False
    ) -> Tensor:
        """Forward/backward for SFT with PP+CP support.

        Uses engine.loss_fn (built-in cross-entropy loss).
        Supports Pipeline Parallelism and Context Parallelism.
        """
        # ... PP and CP handling ...
        loss = self.engine.loss_fn(pred, labels)
        if not skip_backward:
            loss.backward()
        return loss

    def forward_backward_rl(
        self,
        inputs: dict[str, Tensor],
        targets: dict[str, Tensor]
    ) -> Tensor:
        """Forward/backward for RL with custom loss.

        Uses self.loss (custom RL loss function passed to trainer).
        """
        logits = model(**inputs)
        loss = self.loss(logits, **targets)  # Custom loss!
        loss.backward()
        return loss

    # === Training Steps ===

    def train_step_sft(self, batch: dict[str, Tensor]) -> None:
        """SFT training step. Called from internal training loop.

        Extracts labels, calls forward_backward, logs metrics, optimizer step.
        """
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)

        if self.rank_should_record_loss:
            record_metric("loss", loss.item(), Reduce.MEAN)

        self.engine.optimizers.step()
        self.engine.lr_schedulers.step()

    @endpoint
    async def train_step(
        self,
        inputs: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]]
    ) -> float:
        """RL training step endpoint. Called from external control loop.

        For GRPO and similar algorithms where data comes from external rollouts.
        """
        loss = self.forward_backward_rl(inputs[self.engine.dp_rank], targets[self.engine.dp_rank])
        torch.distributed.all_reduce(loss)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()

        return loss.item()
```

---

## ForgeSFTRecipe: SFT-Specific Recipe

```python
class ForgeSFTRecipe(TitanTrainer):
    """SFT Recipe that adds data loading, eval, and training loop.

    INHERITS from TitanTrainer:
    ✓ _setup_engine(), rank_should_record_loss
    ✓ setup_metric_logger(), record_batch_metrics()
    ✓ forward_backward(), train_step_sft()

    ADDS SFT-specific:
    + __init__: Config handling (datasets, eval config)
    + setup_data(): Create StatefulDataLoader with tokenizer
    + evaluate(): Evaluation loop
    + train(): Training loop with checkpointing and eval triggers
    """

    def __init__(self, config: DictConfig):
        # Extract SFT-specific config (datasets, eval)
        self.eval_config = config.get("eval", {})
        training_datasets = config.training.pop("datasets", None)

        # Initialize parent with ForgeJobConfig fields only
        super().__init__(**titan_config)

        # Restore datasets for SFT use
        self.job_config.training.datasets = training_datasets

    @endpoint
    async def setup(self):
        # Initialize engine (from TitanTrainer)
        self._setup_engine()

        # Initialize metric logger (from TitanTrainer)
        self.mlogger = await self.setup_metric_logger()

        # SFT-specific: Setup data loaders
        self.train_dataloader = self.setup_data(self.job_config.training.datasets)

        # SFT-specific: Setup eval dataloaders
        if self.validation_enabled:
            for dataset_config in self.eval_config.datasets:
                self.val_dataloaders[name] = self.setup_data([dataset_config])

    def setup_data(self, dataset_configs: list[dict]) -> StatefulDataLoader:
        """SFT-specific: Create dataloader with tokenizer."""
        tokenizer = HuggingFaceModelTokenizer(...)
        dataset = sft_iterable_dataset(tokenizer, ...)
        return StatefulDataLoader(dataset, ...)

    async def evaluate(self) -> None:
        """SFT-specific: Evaluation loop."""
        for model_part in self.engine.model_parts:
            model_part.eval()

        for batch in val_dataloader:
            labels = batch.pop("labels")
            # Use inherited forward_backward with skip_backward=True
            loss = self.forward_backward(batch, labels, skip_backward=True)

        for model_part in self.engine.model_parts:
            model_part.train()

    @endpoint
    async def train(self) -> None:
        """SFT-specific: Training loop with eval triggers."""
        self.engine.optimizers.zero_grad()

        while self.current_step < self.num_training_steps:
            batch = next(self.train_dataloader)

            # Use inherited utilities
            self.record_batch_metrics(batch.pop("metrics", []))

            # Move to device
            batch = {k: v.to(self.engine.device) for k, v in batch.items()}

            # Use inherited train_step_sft
            self.train_step_sft(batch)
            self.current_step += 1

            # Eval periodically
            if self.current_step % self.eval_every_n_steps == 0:
                await self.evaluate()

            # Checkpoint
            self.engine.checkpointer.save(curr_step=self.current_step, ...)
```

---

## GRPO Usage (External Control Loop)

```python
# GRPO uses TitanTrainer directly (no subclass needed for now)
# Data/rollouts managed by external Orchestrator

async def run_grpo():
    # Create trainer with custom GRPO loss
    trainer = await TitanTrainer.options(...).as_actor(
        loss=grpo_loss_fn,  # Custom RL loss
        model={"name": "llama3", ...},
    )
    await trainer.setup.call()

    # External control loop (Orchestrator)
    for epoch in range(num_epochs):
        # Generate rollouts (external)
        rollouts = await generate_rollouts(...)

        # Train on rollouts using train_step endpoint
        for batch in rollouts:
            inputs = [{"input_ids": batch.ids, "attention_mask": batch.mask}]
            targets = [{"advantages": batch.advs, "old_log_probs": batch.log_probs}]
            loss = await trainer.train_step.call(inputs, targets)

    await trainer.cleanup.call()
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Composition for ForgeEngine** | TitanTrainer wraps engine via `self.engine.X` - avoids diamond inheritance |
| **Inheritance for Recipes** | ForgeSFTRecipe inherits from TitanTrainer - gets all utilities automatically |
| **Two forward_backward variants** | `forward_backward()` uses engine.loss_fn (SFT); `forward_backward_rl()` uses custom self.loss (RL) |
| **Endpoint vs Non-endpoint** | `train_step()` is endpoint for external control (RL); `train_step_sft()` is regular method for internal loop (SFT) |
| **`_setup_engine()` helper** | Non-endpoint method so subclasses can call it from their own setup endpoints |

---

## What's Reusable in TitanTrainer

| Component | SFT Uses | RL Uses | Notes |
|-----------|----------|---------|-------|
| `_setup_engine()` | ✓ | ✓ | Initialize ForgeEngine |
| `rank_should_record_loss` | ✓ | ✓ | PP stage handling |
| `setup_metric_logger()` | ✓ | ✓ | Metric logging |
| `record_batch_metrics()` | ✓ | - | Batch metrics from dataloader |
| `forward_backward()` | ✓ | - | Uses engine.loss_fn |
| `forward_backward_rl()` | - | ✓ | Uses custom self.loss |
| `train_step_sft()` | ✓ | - | Internal training step |
| `train_step()` endpoint | - | ✓ | External training step |
| `push_weights()` | - | ✓ | Push to torchstore |
| `cleanup()` | ✓ | ✓ | Resource cleanup |

---

## Migration Path

1. **PR 1 (Current)**: Add reusable components to TitanTrainer, have ForgeSFTRecipe inherit from TitanTrainer
2. **PR 2 (Future)**: Migrate GRPO to use TitanTrainer if beneficial
3. **PR 3 (Future)**: Add more reusable components (gradient clipping, etc.)
