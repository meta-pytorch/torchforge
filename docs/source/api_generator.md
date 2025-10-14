# Generator

```{eval-rst}
.. currentmodule:: forge.actors.policy
```

The Generator (Policy) is the core inference engine in TorchForge,
built on top of [vLLM](https://docs.vllm.ai/en/latest/).
It manages model serving, text generation, and weight updates for reinforcement learning workflows.

## Policy

```{eval-rst}
.. autoclass:: Policy
   :members: generate, update_weights, get_version, stop
   :exclude-members: __init__, launch
   :no-inherited-members:
```

## PolicyWorker

```{eval-rst}
.. autoclass:: PolicyWorker
   :members: execute_model, update, setup_kv_cache
   :show-inheritance:
   :exclude-members: __init__
```
