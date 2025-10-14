# Model

```{eval-rst}
.. currentmodule:: forge.actors.reference_model
```

The {class}`forge.actors.reference_model.ReferenceModel` provides a frozen
copy of the policy model used for computing advantages in reinforcement
learning. It performs inference on input sequences and returns logits or
log probabilities for computing KL divergence and other RL metrics.

## ReferenceModel

```{eval-rst}
.. autoclass:: forge.actors.reference_model.ReferenceModel
   :members:
   :undoc-members:
   :show-inheritance:
```

The ReferenceModel uses a subset of TorchTitan's configuration system:

- **model**: Model architecture settings (Model dataclass)
- **parallelism**: Parallelism configuration for distributed inference (Parallelism dataclass)
- **checkpoint**: Checkpoint loading settings (Checkpoint dataclass)
- **compile**: Model compilation settings (Compile dataclass)
- **training**: Training configuration for dtype and other settings (Training dataclass)

For detailed configuration options, refer to the [TorchTitan documentation](https://github.com/pytorch/torchtitan).
