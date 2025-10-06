# API Reference

This API reference is organized by priority of concepts that users should be exposed to, based on how they're used in the core Forge workflows.

```{eval-rst}
.. toctree::
   :maxdepth: 2

   api_core
   api_actors
   api_data_models
   api_types
   api_util
   api_data
   api_losses
```

## Overview

The Forge API is structured around key concepts in order of priority:

1. **[Core Concepts](api_core.md)** - Actor System, ForgeActor, and ForgeService fundamentals
2. **[Built-in Actors](api_actors.md)** - Policy, Trainer, ReplayBuffer, and ReferenceModel
3. **[Data Models](api_data_models.md)** - Completion, Prompt, Episode, and other data structures
4. **[Configuration and Types](api_types.md)** - Core types and configuration classes
5. **[Utilities](api_util.md)** - Distributed computing, logging, and observability tools
6. **[Data Processing](api_data.md)** - Rewards, tokenization, and data handling utilities
7. **[Loss Functions](api_losses.md)** - GRPO and REINFORCE loss implementations

## Quick Start

To get started with Forge, begin with the [Core Concepts](api_core.md) to understand the actor system foundation, then explore the [Built-in Actors](api_actors.md) for common RL workflows.

For a practical example, see the GRPO implementation in `apps/grpo/main.py` which demonstrates how these components work together in a complete reinforcement learning training loop.
