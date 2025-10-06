# Built-in Actors

On top of the services/actors foundation, Forge provides implementations of actors that are useful in RL workflows.

## Policy

Inference and generation via vLLM. The {class}`forge.actors.policy.Policy` is a key actor for generating completions from language models.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.actors.policy.Policy
   forge.actors.policy.PolicyWorker
   forge.actors.policy.EngineConfig
   forge.actors.policy.SamplingConfig
```

## Trainer

Training via torchtitan. The {class}`forge.actors.trainer.RLTrainer` handles reinforcement learning training loops.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.actors.trainer.RLTrainer
```

## ReplayBuffer

For storing experience and sampling to the trainer - the glue between policy and trainer. The {class}`forge.actors.replay_buffer.ReplayBuffer` manages experience data for RL training.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.actors.replay_buffer.ReplayBuffer
```

## ReferenceModel

Used for RL correctness. The {class}`forge.actors.reference_model.ReferenceModel` provides reference logprobs for RL algorithms.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.actors.reference_model.ReferenceModel
```
