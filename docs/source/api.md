# API Reference

This section provides comprehensive API documentation for TorchForge modules and classes.

TorchForge is organized into several key modules, each providing specialized functionality for post-training generative AI models:

## Module Overview

**Core Components**
- [Interfaces & Types](api_core.md) - Core interfaces and type definitions
- [Actors](api_actors.md) - Model training and inference components
- [Controller](api_controller.md) - Distributed training orchestration and resource management

**Data Management**
- [Data](api_data.md) - Data handling utilities, datasets, and data models

**Training Components**
- [Losses](api_losses.md) - Loss functions for reinforcement learning and supervised fine-tuning
- [Environments](api_envs.md) - Training and inference environments

**Tools & Utilities**
- [CLI](api_cli.md) - Command line interface
- [Utilities](api_util.md) - General utility functions and helpers

```{toctree}
:maxdepth: 2
:hidden:

api_core
api_actors
api_data
api_losses
api_envs
api_controller
api_cli
api_util
```

## Quick Reference

For developers looking for specific functionality:

- **Getting started with training**: See [Actors](api_actors.md) for policy actors, trainers, and replay buffers
- **Working with data**: See [Data](api_data.md) for datasets, data models, and utilities
- **Custom loss functions**: See [Losses](api_losses.md) for GRPO and REINFORCE implementations
- **Distributed training**: See [Controller](api_controller.md) for orchestration and resource management
- **Command line tools**: See [CLI](api_cli.md) for available commands and configuration options
