# Refactoring Proposal 08: Align with GRPO Main.py Patterns

## Overview
Building on Proposals 01-07, this iteration aligns the code structure and patterns more closely with grpo/main.py to maintain consistency across the codebase while keeping blackjack-specific improvements.

## Key Changes

### 1. Add Type Aliases for Clarity
Follow grpo/main.py pattern of defining type aliases.

**In main_v2.py:**
```python
# Type aliases (like grpo/main.py)
Group = list[Episode]  # Group of episodes for GRPO
Policy = Generator     # Policy model for generation

# Then use throughout:
async def compute_advantages(group: Group) -> list[float]:
    """Compute advantages for a group of episodes."""
    # ...
```

### 2. Align ComputeAdvantages Actor
Current implementation is nearly identical to grpo/main.py. Make it exactly the same.

**Before:**
```python
@dataclass
class ComputeAdvantages(ForgeActor):
    """Compute advantages for a group of episodes."""

    @endpoint
    async def compute(self, group: list[Episode]) -> list[float]:
        """Compute advantages using reward standardization."""
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()
```

**After (exactly match grpo/main.py):**
```python
@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    async def compute(self, group: Group) -> list[float]:
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()
```

### 3. Standardize Async Function Signatures
Follow grpo/main.py's clean async function signatures.

**Before:**
```python
async def do_single_rollout(
    env: BlackjackEnv,
    policy,
    tokenizer,
    max_seq_len: int,
    max_turns: int,
    messages: list[dict],
    game_id: str | None = None,
) -> Episode:
```

**After (add type hints):**
```python
async def do_single_rollout(
    env: BlackjackEnv,
    policy: Policy,
    tokenizer: Any,
    max_seq_len: int,
    max_turns: int,
    messages: list[dict[str, str]],
    game_id: str | None = None,
) -> Episode:
```

### 4. Unify Service Initialization Pattern
Current code initializes services differently than grpo/main.py. Align the pattern.

**Before:**
```python
# First, initialize env_actor to get pad_id
env_actor = await EnvironmentActor.options(**cfg.actors.blackjack_env).as_actor(**env_actor_config)
pad_id = await env_actor.pad_token.call_one()

# Create collate function with pad_id
collate_fn = partial(collate, pad_id=pad_id)

# Now initialize remaining services
(policy, trainer, replay_buffer, compute_advantages, ref_model) = await asyncio.gather(...)
```

**After (get tokenizer directly, pass to collate):**
```python
# Get tokenizer for pad_id
tokenizer = get_tokenizer(cfg.blackjack_env.model)
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
collate_fn = partial(collate, pad_id=pad_id)

# Initialize all services together (like grpo/main.py)
(policy, trainer, replay_buffer, compute_advantages, ref_model) = await asyncio.gather(
    Generator.options(**cfg.services.policy).as_service(**cfg.policy),
    TitanTrainer.options(**cfg.actors.trainer).as_actor(
        **cfg.trainer, loss=simple_grpo_loss
    ),
    ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
        **cfg.replay_buffer, collate=collate_fn
    ),
    ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
    ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
)
```

### 5. Align Drop Weights Function
Make it exactly match grpo/main.py.

**Current in main_v2.py (lines 1494-1507):**
```python
async def drop_weights(version: int):
    """Drop old weights from torchstore."""
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")
```

**After (exactly match grpo/main.py lines 276-290):**
```python
async def drop_weights(version: int):
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    # TODO: once we have something like `get_meta()` in torchstore, we can just
    # query the type of the object instead of relying on keys.
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")
```

### 6. Standardize Main Function Structure
Align the main() function structure with grpo/main.py.

**Structure:**
```python
async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""

    # ---- Extract config values ---- #
    group_size = cfg.group_size
    max_seq_len = cfg.blackjack_env.max_seq_len
    max_turns = cfg.blackjack_env.max_turns
    max_steps = cfg.trainer.training.steps or -1

    # ---- Start environment servers ---- #
    server_processes = start_servers(...)

    # ---- Global setups ---- #
    provisioner = ...
    mlogger = ...

    # ---- Setup services ---- #
    tokenizer = get_tokenizer(cfg.blackjack_env.model)
    pad_id = ...
    (policy, trainer, replay_buffer, ...) = await asyncio.gather(...)

    # ---- Initialize torchstore ---- #
    await ts.initialize(...)

    # ---- Warmup policy ---- #
    # ...

    # ---- Core RL loops ---- #
    async def continuous_rollouts(thread_id: int):
        # ...

    async def continuous_training():
        # ...

    # ---- Run training ---- #
    rollout_tasks = [...]
    training_task = ...

    try:
        await training_task
    except KeyboardInterrupt:
        # ...
    finally:
        # ... cleanup
```

### 7. Remove Multi-Threading Support (Simplify)
The original grpo/main.py has `rollout_threads` but simpler implementation. Blackjack has one thread per server which is over-engineered for a simple game.

**Consideration:** For Blackjack, we could simplify to single rollout thread, or keep multiple but document why (parallel game collection).

## Impact
- **Consistency:** Code patterns match grpo/main.py closely
- **Maintainability:** Easier to understand for developers familiar with grpo/main.py
- **Type safety:** Better type hints throughout
- **Service init:** Cleaner, no EnvironmentActor hack
- **Risk:** Low - mostly alignment, few logic changes
