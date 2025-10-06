# Core Concepts

## Actor System

Forge is built on top of Monarch and makes extensive use of actors.
Actors are the foundation for building distributed, fault-tolerant systems.

## ForgeActor

In Forge, everything centers around the {class}`forge.controller.actor.ForgeActor`, which is a customized version of a Monarch actor tailored for Forge-specific needs.

The {class}`forge.controller.actor.ForgeActor` differs from a standard Monarch actor by allowing you to specify resource requirements using the `options()` method. This API lets you define the resources your actor needs and create two types of constructs:

- A regular Monarch actor using `as_actor()`
- A service using `as_service()`

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.controller.actor.ForgeActor
```

### Resource Configuration Example

Options are important because they demonstrate the resource requirements and how we represent them in Forge:

```python
from forge.controller.actor import ForgeActor

class MyActor(ForgeActor):
    pass

# Create a service with specific resource requirements
service = MyActor.options(
    hosts=1,
    procs=8,
    replicas=1,
    with_gpus=True
).as_service()
```

This example creates a service that has 1 replica, where the replica consists of 1 remote host and 8 processes, using Monarch's remote allocations.

### Key Methods

**`options()`**
Configures resource requirements for the actor.

**`setup()`**
Sets up an actor. All actors should implement this for any heavyweight setup (like PyTorch distributed initialization, model checkpoint loading, etc.).

**`launch()`**
Logic to provision and deploy a new replica. This is what services use to spin up replicas.

## ForgeService

Services are replicated, fault-tolerant versions of actors. They provide high availability and load distribution across multiple actor instances.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.controller.service.Service
   forge.controller.service.ServiceInterface
```
