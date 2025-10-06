# Configuration and Types

## Core Type Definitions

Core type definitions used throughout Forge. The {mod}`forge.types` module contains fundamental types and configurations.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.types
```

## Configuration Classes

Configuration classes for actors and services.

### EngineConfig

Configuration for vLLM engines used in Policy actors. The {class}`forge.actors.policy.EngineConfig` extends vLLM's EngineArgs with worker-specific fields.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.actors.policy.EngineConfig
```

### SamplingConfig

Configuration for sampling parameters in Policy actors. The {class}`forge.actors.policy.SamplingConfig` provides overrides for vLLM's sampling parameters.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.actors.policy.SamplingConfig
```
