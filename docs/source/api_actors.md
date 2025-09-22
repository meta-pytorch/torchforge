# Actors

The actors module contains the core components for model training and inference in TorchForge. This includes policy actors, reference models, replay buffers, and trainers.

## Overview

```{eval-rst}
.. automodule:: forge.actors
   :members:
   :undoc-members:
   :show-inheritance:
```

## Policy Actor

The policy actor is responsible for model inference and policy interactions during training.

```{eval-rst}
.. automodule:: forge.actors.policy
   :members:
   :undoc-members:
   :show-inheritance:
```

## Reference Model

The reference model provides baseline comparisons for reinforcement learning algorithms.

```{eval-rst}
.. automodule:: forge.actors.reference_model
   :members:
   :undoc-members:
   :show-inheritance:
```

## Replay Buffer

The replay buffer manages storage and sampling of training experiences.

```{eval-rst}
.. automodule:: forge.actors.replay_buffer
   :members:
   :undoc-members:
   :show-inheritance:
```

## Trainer

The trainer orchestrates the training process and implements training algorithms.

```{eval-rst}
.. automodule:: forge.actors.trainer
   :members:
   :undoc-members:
   :show-inheritance:
```
