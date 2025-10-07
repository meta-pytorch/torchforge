# API Reference

This section provides comprehensive API documentation for TorchForge.

## Overview

TorchForge is a PyTorch native platform for post-training generative AI models,
designed to streamline reinforcement learning workflows for large language
models. The platform leverages PyTorch's distributed computing capabilities
and is built on top of [Monarch](https://meta-pytorch.org/monarch/),
making extensive use of actors for distributed computation and fault tolerance.

Key Features of TorchForge include:

- **Actor-Based Architecture**: TorchForge uses an actor-based system for distributed training, providing excellent scalability and fault tolerance.
- **PyTorch Native**: Built natively on PyTorch, ensuring seamless integration with existing PyTorch workflows.
- **Post-Training Focus**: Specifically designed for post-training techniques like RLHF, SFT, and other alignment methods.
- **Distributed by Design**: Supports multi-GPU and multi-node training out of the box.


For most use cases, you'll interact with the high-level service
interfaces, which handle the complexity of actor coordination and
distributed training automatically.

For advanced users who need fine-grained control, the individual actor
APIs provide direct access to the underlying distributed components.

```{toctree}
:maxdepth: 1
api_actors
api_service
```
