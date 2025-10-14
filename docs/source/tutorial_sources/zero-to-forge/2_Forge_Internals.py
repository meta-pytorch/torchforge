# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Part 2: Peeling Back the Abstraction - What Are Services?
==========================================================

**Author:** `Sanyam Bhutani <https://github.com/init27>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How Forge services work under the hood
       * Service communication patterns (route, fanout, call_one)
       * State management in distributed systems
       * Real-world service orchestration patterns

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * Complete :doc:`1_RL_and_Forge_Fundamentals`
       * Understanding of Python async/await
       * Basic distributed systems knowledge

We highly recommend reading Part 1 before this - it explains RL Concepts
and how they land in Forge.

Now that you see the power of the service abstraction, let's understand
what's actually happening under the hood. Grab your chai!
"""

######################################################################
# Service Anatomy: Beyond the Interface
# --------------------------------------
#
# When you call ``await policy_service.generate(question)``, here's what
# actually happens:
#
# (Don't worry, we will understand Services right in the next section!)
#
# .. mermaid::
#
#     graph TD
#         Call["Your Code:<br/>await policy_service.generate"]
#
#         subgraph ServiceLayer["Service Layer"]
#             Proxy["Service Proxy: Load balancing, Health checking"]
#             LB["Load Balancer: Replica selection, Circuit breaker"]
#         end
#
#         subgraph Replicas["Replica Management"]
#             R1["Replica 1: GPU 0, Healthy"]
#             R2["Replica 2: GPU 1, Overloaded"]
#             R3["Replica 3: GPU 2, Failed"]
#             R4["Replica 4: GPU 3, Healthy"]
#         end
#
#         subgraph Compute["Actual Computation"]
#             Actor["Policy Actor: vLLM engine, Model weights, KV cache"]
#         end
#
#         Call --> Proxy
#         Proxy --> LB
#         LB --> R1
#         LB -.-> R2
#         LB -.-> R3
#         LB --> R4
#         R1 --> Actor
#         R4 --> Actor
#
#         style Call fill:#4CAF50
#         style LB fill:#FF9800
#         style R3 fill:#F44336
#         style Actor fill:#9C27B0

######################################################################
# Service Components Deep Dive
# -----------------------------
#
# 1. Real Service Configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here's the actual ServiceConfig from Forge source code:

# Configuration pattern from apps/grpo/main.py:
# Policy.options(
#     procs=1,           # Processes per replica
#     num_replicas=4,    # Number of replicas
#     with_gpus=True     # Allocate GPUs
#     # Other available options:
#     # hosts=None   #  the number of remote hosts used per replica
# )

######################################################################
# 2. Real Service Creation
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Services are created using the ``.options().as_service()`` pattern
# from the actual GRPO implementation.
#
# The service creation automatically handles:
#
# * Spawning actor replicas across processes/GPUs
# * Load balancing with .route() method for services
# * Health monitoring and failure recovery
# * Message routing and serialization

import asyncio

# Mock imports for documentation build
try:
    from forge.actors.policy import Policy
except ImportError:

    class Policy:
        pass


async def example_service_creation():
    """Example of creating a Policy service."""
    model = "Qwen/Qwen3-1.7B"

    policy = await Policy.options(procs=1, with_gpus=True, num_replicas=1).as_service(
        engine_config={
            "model": model,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "enforce_eager": False,
        },
        sampling_config={
            "n": 1,
            "max_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    )

    prompt = "What is 3 + 5?"
    responses = await policy.generate.route(prompt)
    print(f"Response: {responses[0].text}")

    # Cleanup when done
    await policy.shutdown()


######################################################################
# 3. How Services Actually Work
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Forge services are implemented as ServiceActors that manage
# collections of your ForgeActor replicas.
#
# When you call ``.as_service()``, Forge creates a ``ServiceInterface``
# that manages N replicas of your ``ForgeActor`` class and gives you
# methods like ``.route()``, ``.fanout()``, etc.


async def service_interface_example(policy):
    """Your code sees this simple interface."""
    # Simple call - but Forge handles all complexity
    responses = await policy.generate.route(prompt="What is 2+2?")
    # Forge handles: replica management, load balancing, fault tolerance


######################################################################
# Communication Patterns: Quick Reference
# ----------------------------------------
#
# **API Summary:**
#
# * ``.route()`` - Send request to any healthy replica in a service (load balanced)
# * ``.call_one()`` - Send request to a single actor instance
# * ``.fanout()`` - Send request to ALL replicas in a service
#
# .. mermaid::
#
#     graph LR
#         subgraph Request["Your Request"]
#             Code["await service.method.ADVERB()"]
#         end
#
#         subgraph Patterns["Communication Patterns"]
#             Route[".route()<br/>→ One healthy replica"]
#             CallOne[".call_one()<br/>→ Single actor"]
#             Fanout[".fanout()<br/>→ ALL replicas"]
#         end
#
#         subgraph Replicas["Replicas/Actors"]
#             R1["Replica 1"]
#             R2["Replica 2"]
#             R3["Replica 3"]
#             A1["Actor"]
#         end
#
#         Code --> Route
#         Code --> CallOne
#         Code --> Fanout
#
#         Route --> R2
#         CallOne --> A1
#         Fanout --> R1
#         Fanout --> R2
#         Fanout --> R3
#
#         style Route fill:#4CAF50
#         style CallOne fill:#FF9800
#         style Fanout fill:#9C27B0

######################################################################
# Deep Dive: Service Communication Patterns
# ------------------------------------------
#
# These communication patterns ("adverbs") determine how your service
# calls are routed to replicas. Understanding when to use each pattern
# is key to effective Forge usage.
#
# 1. ``.route()`` - Load Balanced Single Replica
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **When to use**: Normal request routing where any replica can handle
# the request.


async def route_example(policy):
    """Using .route() for load-balanced requests."""
    question = "What is 2+2?"
    responses = await policy.generate.route(prompt=question)
    answer = responses[0].text  # Extract text from Completion object
    return answer


# Behind the scenes:
# 1. Health check eliminates failed replicas
# 2. Load balancer picks replica (currently round robin)
# 3. Request routes to that specific replica
# 4. Automatic retry on different replica if failure
#
# **Performance characteristics**:
#
# * **Latency**: Lowest (single network hop)
# * **Throughput**: Limited by single replica capacity
# * **Fault tolerance**: Automatic failover to other replicas
#
# **Critical insight**: ``.route()`` is your default choice for
# stateless operations in Forge services.

######################################################################
# 2. ``.fanout()`` - Broadcast with Results Collection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **When to use**: You need responses from ALL replicas.


async def fanout_example(policy):
    """Using .fanout() to broadcast to all replicas."""
    # Get version from all policy replicas
    current_versions = await policy.get_version.fanout()
    # Returns: [version_replica_1, version_replica_2, ...]

    # Update weights on all replicas
    await policy.update_weights.fanout(new_policy_version=1)
    # Broadcasts to all replicas simultaneously


# **Performance characteristics**:
#
# * **Latency**: Slowest replica determines total latency
# * **Throughput**: Network bandwidth × number of replicas
# * **Fault tolerance**: Fails if ANY replica fails (unless configured)
#
# **Critical gotcha**: Don't use ``.fanout()`` for high-frequency
# operations - it contacts all replicas.

######################################################################
# 3. Streaming Operations - Custom Implementation Pattern
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **When to use**: You want to process results as they arrive.


async def streaming_pattern_example(replay_buffer, trainer, step):
    """Streaming pattern for continuous training."""
    # CONCEPTUAL - Streaming requires custom implementation
    # Pattern from apps/grpo/main.py continuous training:

    while True:
        # This is the real API call pattern
        batch = await replay_buffer.sample.call_one(curr_policy_version=step)
        if batch is not None:
            # Process batch immediately
            loss = await trainer.train_step.call_one(batch)
            print(f"Training loss: {loss}")
        else:
            await asyncio.sleep(0.1)  # Wait for more data
        break  # Just for example


# **Performance characteristics**:
#
# * **Latency**: Process first result immediately
# * **Throughput**: Non-blocking async operations
# * **Fault tolerance**: Continues if some replicas fail
#
# **Critical insight**: Essential for high-throughput RL where
# you can't wait for batches.

######################################################################
# Service Sessions for Stateful Operations
# -----------------------------------------
#
# **When to use**: When you need multiple calls to hit the same replica
# (like KV cache preservation).
#
# **What are sticky sessions?** A session ensures all your service calls
# within the ``async with`` block go to the same replica, instead of
# being load-balanced across different replicas.

# Mock classes for example
try:
    from forge.controller import ForgeActor
    from monarch.actor import endpoint
except ImportError:

    class ForgeActor:
        pass

    def endpoint(func):
        return func


class ForgeCounter(ForgeActor):
    """Example counter to demonstrate sessions."""

    def __init__(self, initial_value: int):
        self.value = initial_value

    @endpoint
    def increment(self) -> int:
        self.value += 1
        return self.value

    @endpoint
    def get_value(self) -> int:
        return self.value

    @endpoint
    async def reset(self):
        self.value = 0


async def without_sessions_example():
    """WITHOUT SESSIONS: Each .route() goes to different replica."""
    counter_service = await ForgeCounter.options(procs=1, num_replicas=4).as_service(
        initial_value=0
    )

    # Each call might go to different replica
    await counter_service.increment.route()  # Might go to replica 2
    await counter_service.increment.route()  # Might go to replica 1
    await counter_service.increment.route()  # Might go to replica 3

    results = await counter_service.increment.fanout()
    print(f"All replica values: {results}")
    # Output: All replica values: [1, 2, 1, 1]
    # Each replica has different state!

    await counter_service.shutdown()


async def with_sessions_example():
    """WITH SESSIONS: All calls go to the SAME replica."""
    counter_service = await ForgeCounter.options(procs=1, num_replicas=4).as_service(
        initial_value=0
    )

    print("\nUsing sticky sessions:")
    async with counter_service.session():
        await counter_service.reset.route()
        print(await counter_service.increment.route())  # 1
        print(await counter_service.increment.route())  # 2
        print(await counter_service.increment.route())  # 3

        final_value = await counter_service.get_value.route()
        print(f"Final value on this replica: {final_value}")  # 3

    # Cleanup
    await counter_service.shutdown()


######################################################################
# Deep Dive: State Management Reality
# ------------------------------------
#
# The most complex challenge in distributed RL is maintaining state
# consistency while maximizing performance.
#
# The KV Cache Problem
# ~~~~~~~~~~~~~~~~~~~~
#
# **The challenge**: Policy inference is much faster with KV cache,
# but cache is tied to specific conversation history.


async def naive_multi_turn(policy_service):
    """This breaks KV cache optimization."""
    question1 = "What is 2+2?"

    # Each call might go to different replica = cache miss
    response1 = await policy_service.generate.route(prompt=question1)
    full_prompt = question1 + response1[0].text
    response2 = await policy_service.generate.route(prompt=full_prompt)  # Cache miss!
    conversation = full_prompt + response2[0].text
    response3 = await policy_service.generate.route(prompt=conversation)  # Cache miss!


async def optimized_multi_turn(policy):
    """The solution: Sticky sessions ensure same replica."""
    async with policy.session():
        # All calls guaranteed to hit same replica = cache hits
        question1 = "What is 2+2?"
        response1 = await policy.generate.route(prompt=question1)
        full_prompt = question1 + response1[0].text
        response2 = await policy.generate.route(prompt=full_prompt)  # Cache hit!
        conversation = full_prompt + response2[0].text
        response3 = await policy.generate.route(prompt=conversation)  # Cache hit!

    # Session ends, replica can be garbage collected or reused


# **Performance impact**: Maintaining KV cache across turns avoids
# recomputing previous tokens.

######################################################################
# Replay Buffer Consistency
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **The challenge**: Multiple trainers and experience collectors
# reading/writing concurrently.
#
# **Real Forge approach**: The ReplayBuffer actor handles concurrency
# internally:


async def replay_buffer_example(replay_buffer):
    """ReplayBuffer provides thread-safe operations."""
    # Add episodes (thread-safe by actor model)
    episode = {}  # Mock episode
    await replay_buffer.add.call_one(episode)

    # Sample batches for training
    batch = await replay_buffer.sample.call_one(
        curr_policy_version=0,
        batch_size=None,  # Optional, uses default from config
    )

    # Additional methods available:
    # await replay_buffer.clear.call_one()  # Clear buffer
    # await replay_buffer.evict.call_one(curr_policy_version)  # Remove old
    # state = await replay_buffer.state_dict.call_one()  # Checkpoint


# **Critical insight**: The actor model provides natural thread safety -
# each actor processes messages sequentially.

######################################################################
# Weight Synchronization Strategy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **The challenge**: Trainer updates policy weights, but policy service
# needs those weights.

import torch


async def real_weight_sync(trainer, policy, step):
    """Forge weight synchronization pattern from apps/grpo/main.py."""
    # Trainer pushes weights to TorchStore with version number
    await trainer.push_weights.call_one(policy_version=step + 1)

    # Policy service updates to new version from TorchStore
    # Use .fanout() to update ALL policy replicas
    await policy.update_weights.fanout(policy_version=step + 1)

    # Check current policy version
    current_version = await policy.get_version.route()
    print(f"Current policy version: {current_version}")


######################################################################
# Deep Dive: Asynchronous Coordination Patterns
# ----------------------------------------------
#
# **The real challenge**: Different services run at different speeds,
# but Forge's service abstraction handles the coordination complexity.
#
# The Forge Approach: Let Services Handle Coordination
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Instead of manual coordination, Forge services handle speed mismatches
# automatically:


async def simple_rl_step(
    dataloader,
    policy,
    ref_model,
    reward_actor,
    replay_buffer,
    compute_advantages,
    trainer,
):
    """Simple RL step showing service coordination."""
    # ===== Generate a rollout =====
    sample = await dataloader.sample.call_one()
    prompt, target = sample["request"], sample["target"]

    print(f"Prompt: {prompt}")
    print(f"Target: {target}")

    actions = await policy.generate.route(prompt=prompt)
    print(f"Policy response: {actions[0].text}")

    # Create input tensor for reference model (requires full context)
    input_ids = torch.cat([actions[0].prompt_ids, actions[0].token_ids])
    ref_logprobs = await ref_model.forward.route(
        input_ids.unsqueeze(0), max_req_tokens=512, return_logprobs=True
    )

    reward = await reward_actor.evaluate_response.route(
        prompt=prompt, response=actions[0].text, target=target
    )
    print(f"Reward: {reward}")

    # Create episode (simplified for example)
    episode = {
        "episode_id": "0",
        "request": prompt,
        "response": actions[0].text,
        "reward": reward,
        "ref_logprobs": ref_logprobs[0],
    }

    await replay_buffer.add.call_one(episode)
    print("Episode stored in replay buffer")

    # ===== Train on the batch =====
    batch = await replay_buffer.sample.call_one(curr_policy_version=0)
    if batch is not None:
        print("Training on batch...")
        inputs, targets = batch
        loss = await trainer.train_step.call(inputs, targets)
        print(f"Training loss: {loss}")
        return loss
    else:
        print("Not enough data in buffer yet")
        return None


######################################################################
# Handling Speed Mismatches with Service Scaling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **The insight**: Scale services independently based on their
# bottlenecks.

# Mock imports for example
try:
    from forge.actors.trainer import RLTrainer
except ImportError:

    class RLTrainer:
        pass


async def scaling_example():
    """Scale services independently based on bottlenecks."""
    model_name = "Qwen/Qwen3-1.7B"

    # Scale fast services with more replicas
    policy = await Policy.options(
        procs=1, num_replicas=8, with_gpus=True  # Many replicas for throughput
    ).as_service(engine_config={"model": model_name, "tensor_parallel_size": 1})

    # Reward evaluation might be CPU-bound
    reward_actor = await RewardActor.options(
        procs=1, num_replicas=16, with_gpus=False  # More CPU replicas
    ).as_service(reward_functions=[])

    # Training needs fewer but more powerful replicas
    trainer = await RLTrainer.options(
        procs=1, with_gpus=True  # Fewer but GPU-heavy
    ).as_actor(  # Trainer typically uses .as_actor() not .as_service()
        model={"name": "qwen3", "flavor": "1.7B"},
        optimizer={"name": "AdamW", "lr": 1e-5},
    )


######################################################################
# Service Implementation Example
# -------------------------------
#
# Let's see how a reward service is actually implemented:

# Mock imports
try:
    from forge.controller import ForgeActor
    from forge.data.rewards import MathReward, ThinkingReward
    from monarch.actor import endpoint
except ImportError:

    class ForgeActor:
        pass

    def endpoint(func):
        return func

    class MathReward:
        def __call__(self, prompt, response, target):
            return 1.0

    class ThinkingReward:
        def __call__(self, prompt, response, target):
            return 1.0


class RewardActor(ForgeActor):
    """Exact RewardActor from apps/grpo/main.py."""

    def __init__(self, reward_functions: list):
        self.reward_functions = reward_functions

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        """Evaluate response quality using multiple reward functions."""
        total_reward = 0.0

        for reward_fn in self.reward_functions:
            # Each reward function contributes to total score
            reward = reward_fn(prompt, response, target)
            total_reward += reward

        # Return average reward across all functions
        return (
            total_reward / len(self.reward_functions) if self.reward_functions else 0.0
        )


async def reward_service_example():
    """Create and use a reward service."""
    reward_actor = await RewardActor.options(procs=1, num_replicas=1).as_service(
        reward_functions=[MathReward(), ThinkingReward()]
    )

    prompt = "What is 15% of 240?"
    response = "15% of 240 is 36"
    target = "36"

    score = await reward_actor.evaluate_response.route(
        prompt=prompt, response=response, target=target
    )
    print(f"Reward score: {score}")  # Usually around 1.0 for correct answers

    # For production scaling - increase num_replicas:
    # RewardActor.options(procs=1, num_replicas=16)  # 16 parallel evaluators

    # Cleanup when done
    await reward_actor.shutdown()


######################################################################
# Service Orchestration: The Training Loop
# -----------------------------------------
#
# Now let's see how services coordinate in a real training loop:


async def production_training_loop():
    """Real training loop pattern from apps/grpo/main.py."""
    # Service creation pattern (abbreviated)
    print("Initializing all services...")

    # (Services initialization code here - see Part 1)

    step = 0

    while True:
        # Data generation
        sample = await dataloader.sample.call_one()

        # Policy generation service call
        responses = await policy.generate.route(sample["request"])

        # Reference computation service call
        input_ids = torch.cat([responses[0].prompt_ids, responses[0].token_ids])
        ref_logprobs = await ref_model.forward.route(
            input_ids.unsqueeze(0), max_req_tokens=512, return_logprobs=True
        )

        # Reward evaluation service call
        reward = await reward_actor.evaluate_response.route(
            prompt=sample["question"],
            response=responses[0].text,
            target=sample["answer"],
        )

        # Experience storage
        await replay_buffer.add.call_one(episode)

        # Training when ready
        batch = await replay_buffer.sample.call_one(curr_policy_version=step)
        if batch is not None:
            inputs, targets = batch
            loss = await trainer.train_step.call(inputs, targets)

            # Weight synchronization pattern
            await trainer.push_weights.call(step + 1)
            await policy.update_weights.fanout(step + 1)  # Fanout to all replicas

            print(f"Step {step}, Loss: {loss:.4f}")
            step += 1

        if step >= 100:
            break


# **Key observations:**
#
# 1. **Parallelism**: Independent operations run concurrently
# 2. **Load balancing**: Each ``.route()`` call automatically selects optimal replica
# 3. **Fault tolerance**: Failures automatically retry on different replicas
# 4. **Resource efficiency**: CPU and GPU services scale independently
# 5. **Coordination**: Services coordinate through shared state (replay buffer, weight versions)
#
# This is the power of the service abstraction - complex distributed
# coordination looks like simple async Python code.

######################################################################
# Conclusion
# ----------
#
# This tutorial covered:
#
# * How Forge services work under the hood
# * Communication patterns: ``.route()``, ``.fanout()``, ``.call_one()``
# * State management with sessions and actors
# * Service scaling and orchestration patterns
#
# **Key takeaways:**
#
# * Use ``.route()`` for stateless load-balanced operations
# * Use ``.fanout()`` for coordinated updates across all replicas
# * Use sessions for stateful operations like multi-turn conversations
# * Scale services independently based on bottlenecks
# * Let Forge handle coordination complexity
#
# In the next part we will learn about Monarch internals.
#
# Further Reading
# ---------------
#
# * Continue to :doc:`3_Monarch_101` (coming soon)
# * Check the `Forge source code <https://github.com/meta-pytorch/forge>`_
# * Review the :doc:`../../api_actors` documentation
# * Explore the `GRPO application <https://github.com/meta-pytorch/forge/tree/main/apps/grpo>`_
