# Trainer

```{eval-rst}
.. currentmodule:: forge.actors.trainer
```

The Trainer manages model training in TorchForge, built on top of TorchTitan.
It handles forward/backward passes, weight updates, and checkpoint management for reinforcement learning workflows.

## TitanTrainer

```{eval-rst}
.. autoclass:: TitanTrainer
   :members: train_step, push_weights, cleanup
   :exclude-members: __init__
```

## Configuration

The TitanTrainer uses TorchTitan's configuration system with the following components:

### Job Configuration

```{eval-rst}
.. autoclass:: torchtitan.config.job_config.Job
   :members:
   :undoc-members:
```

### Model Configuration

```{eval-rst}
.. autoclass:: torchtitan.config.job_config.Model
   :members:
   :undoc-members:
```

### Optimizer Configuration

```{eval-rst}
.. autoclass:: torchtitan.config.job_config.Optimizer
   :members:
   :undoc-members:
```

### Training Configuration

```{eval-rst}
.. autoclass:: torchtitan.config.job_config.Training
   :members:
   :undoc-members:
```

### Parallelism Configuration

```{eval-rst}
.. autoclass:: torchtitan.config.job_config.Parallelism
   :members:
   :undoc-members:
```

### Checkpoint Configuration

```{eval-rst}
.. autoclass:: torchtitan.config.job_config.Checkpoint
   :members:
   :undoc-members:
```
