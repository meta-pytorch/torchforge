# Truncation V4: Abstraction Fixes and Design Corrections

**Date:** 2025-01-16
**Purpose:** Address critical issues in V3 and establish proper environment/dataset abstractions based on investigation of Tinker, VERL, OpenEnv, TRL, and NeMo-RL.

---

## Easy Fixes (Quick Wins)

### Issue 1: Redundant Initial Prompt Check ❌ DELETE

**Problem:** Decision 8 suggests checking initial prompt before while loop, but this is redundant.

**Why it doesn't work:**
- The while loop naturally handles this on first iteration
- Adds complexity for zero benefit
- First turn already checks budget before generation

**Fix:** Remove the initial prompt check entirely.

```python
# ❌ DELETE THIS (from V3)
initial_prompt = tokenizer.apply_chat_template(messages, ...)
initial_tokens = tokenizer.encode(initial_prompt, add_special_tokens=False)
if len(initial_tokens) >= max_seq_len:
    return Episode(is_truncated=True, ...)

# ✅ KEEP ONLY THIS (let while loop handle it)
while not result.done and turn_num < max_turns:
    # Build prompt
    prompt_text = tokenizer.apply_chat_template(messages, ...)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Check budget naturally
    remaining = max_seq_len - len(prompt_tokens)
    if remaining <= 0:
        is_truncated = True
        break
```

---

### Issue 2: Generator Version from Completion ✅ FIX

**Problem:** V3 hardcodes `generator_version=0`

**Solution:** Extract from completion object.

```python
# ✅ Correct way
response = await policy.generate.route([prompt_text], ...)
response = response[0]

episode = Episode(
    generator_version=response.generator_version,  # From completion!
    ...
)
```

---

### Issue 3: Timeout on Policy Generation ⚠️ OPTIONAL

**Investigation results:**
- **TRL:** No timeout
- **VERL:** Timeout only on reward computation (300s)
- **NeMo-RL:** YES - 600s default via env var `NRL_VLLM_ASYNC_TIMEOUT_SECONDS`
- **Tinker:** No timeout
- **Verifiers:** YES - 600s configurable via `generation_timeout`

**Recommendation:** Add timeout as **optional config**, not hardcoded.

```python
# ✅ Configurable timeout (optional)
timeout = cfg.blackjack_env.get("generation_timeout", None)  # None = no timeout

if timeout is not None:
    responses = await asyncio.wait_for(
        policy.generate.route([prompt_text], sampling_params={"max_tokens": remaining}),
        timeout=timeout
    )
else:
    responses = await policy.generate.route(
        [prompt_text],
        sampling_params={"max_tokens": remaining}
    )
```

**Config:**
```yaml
blackjack_env:
  generation_timeout: 600.0  # Optional, omit for no timeout
```

---

### Issue 4: Double Padding Bug ❌ CRITICAL

**Problem:** We pad in both `continuous_rollouts()` AND `collate()`.

**Root cause:** Misunderstanding of when to pad.

**Investigation:**
- **Reference model** should receive padded batch (for efficient batching)
- **Collate** also needs to pad (for training batch)
- But we're padding the SAME data twice!

**Fix:** Pad only ONCE for ref_model, store ref_logprobs unpadded, then pad again in collate.

```python
# ✅ In continuous_rollouts() - pad for ref_model
max_len = max(len(e.all_token_ids) for e in episodes)
padded_tokens = []
for episode in episodes:
    seq_len = len(episode.all_token_ids)
    pad_len = max_len - seq_len
    padded = F.pad(episode.all_token_ids, (0, pad_len), value=pad_id)
    padded_tokens.append(padded)

input_ids = torch.stack(padded_tokens)  # [batch, max_len]

# Get reference logprobs (padded)
ref_logprobs_padded = await ref_model.forward.route(input_ids, 0, return_logprobs=True)

# Assign ref_logprobs to episodes (UNPAD them!)
for i, episode in enumerate(episodes):
    seq_len = len(episode.all_token_ids)
    episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]  # Unpad!

# ✅ In collate() - pad AGAIN for training batch
# (Different episodes, different max_len)
for batch in batches:
    max_len = max(len(e.all_token_ids) for e in batch)
    # ... pad all_token_ids, ref_logprobs, response_mask, logprobs ...
```

This is correct because:
- Rollout groups may have different max lengths than training batches
- We need flexibility to batch differently during training
- Storing unpadded in Episode keeps data clean

---

### Issue 5: Naive Slicing Bug with Response Mask ❌ CRITICAL

**Problem from V3:**
```python
# ❌ WRONG - ignores response_mask!
episode.ref_logprobs = ref_logprobs[i, :seq_len]
```

**Why it's wrong:**
- `ref_logprobs` includes logprobs for ALL tokens (prompt + response)
- We only care about response tokens (where `response_mask=1`)
- Should NOT naively slice - must respect the mask

**Actually... wait, this is fine:**

The `ref_logprobs` tensor is `[batch, seq_len]` where `seq_len` includes both prompt and response tokens. The `response_mask` will be applied LATER during loss computation to zero out prompt token contributions.

**So the slicing is correct!** We store ref_logprobs for all tokens, and mask is applied during training.

**Re-verification:**
```python
# Episode stores:
all_token_ids:  [prompt1_tokens, response1_tokens, prompt2_tokens, response2_tokens]
response_mask:  [0, 0, 0, ...,   1, 1, 1, ...,    0, 0, 0, ...,    1, 1, 1, ...]
ref_logprobs:   [lp_p1, ...,     lp_r1, ...,      lp_p2, ...,      lp_r2, ...]

# During loss computation:
masked_ref_logprobs = ref_logprobs * response_mask  # Zeros out prompt logprobs
# This is correct!
```

**Conclusion:** Issue 5 is NOT a bug. The slicing is correct. The mask is applied during training.

---

## Complex Issue: Environment/Dataset Abstraction

### Investigation Summary

I investigated 5 frameworks to understand best practices:

| Framework | Env Abstraction | Who Builds Prompts | Multi-Turn | Dataset Role |
|-----------|-----------------|-------------------|------------|--------------|
| **Tinker** | ✅ Yes (`Env` ABC) | Environment (via Renderer) | ✅ Yes | Provides `EnvGroupBuilder` |
| **VERL** | ⚠️ Agent Loop (not Env) | Agent Loop | ✅ Yes | Provides messages + config |
| **OpenEnv** | ✅ Yes (`Environment` class) | Agent (outside env) | ✅ Yes | Separate from env |
| **TRL** | ❌ No | Dataset | ❌ No | Provides formatted prompts |
| **NeMo-RL** | ✅ Yes (`EnvironmentInterface`) | Env appends observations | ✅ Yes | Provides initial messages |

### Key Insights

#### 1. **Tinker's Approach (Best for Us)**

**Architecture:**
```
Dataset → EnvGroupBuilder → Env (with Renderer) → Rollout Loop
```

**Key principles:**
- **Observations are pre-formatted prompts** (`tinker.ModelInput` - already tokenized)
- **Environment owns prompt building** via injected `Renderer`
- **Renderer handles model-specific formatting** (Llama3 vs Qwen3)
- **Environment handles task-specific logic** (check answer, compute reward)
- **Rollout loop is 100% generic** - no task-specific code

**Example:**
```python
# Environment (task-specific)
class BlackjackEnv(Env):
    def __init__(self, renderer: Renderer, server_url: str):
        self.renderer = renderer
        self.server_url = server_url
        self.messages = [{"role": "system", "content": "You are an expert..."}]

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Reset game
        result = self.game_client.reset()
        # Build user message
        self.messages.append({"role": "user", "content": self._format_game_state(result)})
        # Render to tokenized prompt
        obs = self.renderer.build_generation_prompt(self.messages)
        return obs, self.renderer.stop_condition

    async def step(self, action: list[int]) -> StepResult:
        # Parse action using renderer
        message, parse_success = self.renderer.parse_response(action)

        # Extract action from parsed message (task-specific)
        action_name = self._parse_action(message["content"])

        # Execute in game (task-specific)
        result = self.game_client.step(action_name)

        # Compute reward (task-specific)
        reward = self._compute_reward(result)

        # Build next observation
        if not result.done:
            self.messages.append(message)
            self.messages.append({"role": "user", "content": self._format_game_state(result)})
            next_obs = self.renderer.build_generation_prompt(self.messages)
        else:
            next_obs = tinker.ModelInput.empty()

        return StepResult(
            reward=reward,
            episode_done=result.done,
            next_observation=next_obs,
            next_stop_condition=self.renderer.stop_condition,
        )

# Rollout loop (100% generic)
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(ob=ob, ac=ac_with_logprobs, reward=step_result.reward, ...)
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)
```

**Benefits:**
- Loop never touches tokenizer or chat templates
- Same loop works for blackjack, math, code, dialogue
- Swap renderer to support new model (Llama → Qwen)
- Environment encapsulates ALL task logic

#### 2. **OpenEnv's Approach (Most Modular)**

**Architecture:**
```
Dataset (separate) → Agent → Environment (structured observations)
```

**Key principles:**
- **Environment returns structured data**, NOT formatted prompts
- **Agent builds prompts** from structured observations
- **Environment and Dataset are completely separate**
- **Reusability:** Same env works across many datasets

**Example:**
```python
# Environment returns structured observation
@dataclass
class GameObservation(Observation):
    player_total: int
    dealer_card: int
    done: bool
    reward: float

# Agent builds prompt
def build_prompt(obs: GameObservation) -> str:
    return f"Your total: {obs.player_total}, Dealer shows: {obs.dealer_card}"
```

**Benefits:**
- Maximum separation of concerns
- Environment is pure game logic
- Agent controls prompt format
- Easy to swap prompt strategies

**Drawbacks:**
- More boilerplate (agent must format every observation)
- Tokenizer lives in agent, not env

#### 3. **VERL's Approach (Registry-Based)**

**Architecture:**
```
Dataset → Agent Loop (Registry) → Tools
```

**Key principles:**
- **No traditional Env** - `AgentLoopBase.run()` encapsulates everything
- **Registry pattern** - dataset specifies which agent loop via `agent_name`
- **State machine** - `AgentState` enum drives multi-turn logic

**Benefits:**
- Highly extensible via registry
- Supports mixing task types in one training run

**Drawbacks:**
- Less clear boundaries (agent loop does everything)
- Harder to understand data flow

---

### Recommendation for Blackjack

**Use Tinker's pattern** with slight adaptations:

**Reasons:**
1. **Clean separation:** Env handles game logic, Renderer handles formatting, Loop is generic
2. **Observation = formatted prompt:** Loop doesn't need tokenizer
3. **Future-proof:** When we add tool calling, same pattern works
4. **Proven:** Tinker uses this for math, code, dialogue, games

**Adaptations needed:**
1. **No dataset (yet):** Blackjack generates fresh games, not from dataset
2. **Env setup:** Create `BlackjackEnv` with server URL, renderer
3. **Renderer:** Use existing Forge renderer (Qwen3Renderer)

---

## Proposed Abstraction: Blackjack with Tinker Pattern

### Architecture

```
EnvBuilder → BlackjackEnv (with Renderer) → do_single_rollout() → Episode
                ↓
         OpenSpielClient
```

### Component Responsibilities

| Component | Responsibilities | NOT Responsible For |
|-----------|-----------------|---------------------|
| **BlackjackEnv** | Game state, reward logic, action parsing, message history | Tokenization, model formatting |
| **Renderer** | Chat template, tokenization, stop sequences, parsing tokens → messages | Game logic, rewards |
| **Rollout Loop** | Call policy, step env, record transitions | Formatting, parsing, game logic |
| **OpenSpielClient** | HTTP communication with game server | Prompt building, parsing |

### Code Structure

#### 1. Environment Class

```python
# apps/blackjack/env.py

from tinker_cookbook.rl.types import Env, StepResult, Observation, StopCondition
from tinker_cookbook.renderers import Renderer
import tinker

class BlackjackEnv(Env):
    """
    Blackjack environment following Tinker's pattern.

    Responsibilities:
    - Manage game state via OpenSpielClient
    - Build conversation messages (user/assistant)
    - Parse actions from assistant messages
    - Compute rewards
    - Format game state into user messages

    Renderer handles all tokenization and model formatting.
    """

    def __init__(
        self,
        renderer: Renderer,
        server_url: str,
        system_prompt: str | None = None,
    ):
        self.renderer = renderer
        self.server_url = server_url
        self.client = OpenSpielEnv(base_url=server_url)

        # Message history (task-specific)
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        # Metrics tracking
        self.turn_count = 0
        self.has_invalid_action = False

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Reset game and return first observation."""
        # Reset game state
        result = self.client.reset()

        # Build user message with game state (task-specific)
        user_message = self._format_game_state(result.observation)
        self.messages.append({"role": "user", "content": user_message})

        # Render to tokenized observation (renderer handles this)
        obs = self.renderer.build_generation_prompt(self.messages)

        return obs, self.renderer.stop_condition

    async def step(self, action: list[int]) -> StepResult:
        """
        Execute action and return next observation.

        Args:
            action: Token IDs from model generation

        Returns:
            StepResult with next observation, reward, done flag
        """
        # Parse tokens → message (renderer handles this)
        message, parse_success = self.renderer.parse_response(action)

        # Extract action from message content (task-specific)
        action_name = self._parse_action(message["content"])
        if action_name == "INVALID":
            self.has_invalid_action = True
            action_name = "STAND"  # Fallback

        # Add assistant message to history
        self.messages.append(message)

        # Execute action in game (task-specific)
        action_id = 0 if action_name == "HIT" else 1
        result = self.client.step(OpenSpielAction(action_id=action_id, game_name="blackjack"))

        self.turn_count += 1

        # Compute reward (task-specific)
        if result.done:
            reward = self._compute_reward(result.reward, self.has_invalid_action)
        else:
            reward = 0.0  # No intermediate rewards for blackjack

        # Build next observation
        if not result.done:
            user_message = self._format_game_state(result.observation)
            self.messages.append({"role": "user", "content": user_message})
            next_obs = self.renderer.build_generation_prompt(self.messages)
        else:
            next_obs = tinker.ModelInput.empty()

        return StepResult(
            reward=reward,
            episode_done=result.done,
            next_observation=next_obs,
            next_stop_condition=self.renderer.stop_condition,
            metrics={
                "turn_count": self.turn_count,
                "has_invalid_action": self.has_invalid_action,
            }
        )

    def _format_game_state(self, observation) -> str:
        """Format game state into user message (task-specific)."""
        player_total = observation.metadata.get("player_total", "?")
        dealer_card = observation.metadata.get("dealer_card", "?")
        dealer_str = "Ace" if dealer_card == 1 else str(dealer_card)

        return (
            f"=== BlackJack Game (Turn {self.turn_count + 1}) ===\n\n"
            f"Current State:\n"
            f"  Your hand total: {player_total}\n"
            f"  Dealer shows: {dealer_str}\n"
            f"  Legal actions: HIT, STAND\n\n"
            f"What do you do? Output only 'HIT' or 'STAND'."
        )

    def _parse_action(self, text: str) -> str:
        """Parse action from assistant text (task-specific)."""
        text_lower = text.lower().strip()
        if text_lower.endswith("hit"):
            return "HIT"
        elif text_lower.endswith("stand"):
            return "STAND"
        else:
            return "INVALID"

    def _compute_reward(self, env_reward: float, has_invalid: bool) -> float:
        """Compute final reward (task-specific)."""
        if env_reward > 0:  # Win
            return 3.0
        else:  # Loss or push
            return -1.0
```

#### 2. Environment Builder

```python
# apps/blackjack/env.py (continued)

from functools import partial
from tinker_cookbook.rl.types import EnvGroupBuilder

@dataclass(frozen=True)
class BlackjackEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for creating groups of blackjack environments.

    Each env in the group is independent (different game instance).
    """
    server_url: str
    renderer: Renderer
    system_prompt: str
    num_envs: int

    async def make_envs(self) -> list[Env]:
        """Create num_envs independent blackjack environments."""
        return [
            BlackjackEnv(
                renderer=self.renderer,
                server_url=self.server_url,
                system_prompt=self.system_prompt,
            )
            for _ in range(self.num_envs)
        ]
```

#### 3. Rollout Loop (Generic - Reuse Tinker's)

```python
# apps/blackjack/rollouts.py

from tinker_cookbook.rl.rollouts import do_single_rollout, do_group_rollout
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup

# ✅ Use Tinker's generic rollout functions directly!
# No need to rewrite them - they work with any Env implementation.

async def rollout_blackjack_group(
    env_builder: BlackjackEnvGroupBuilder,
    policy: TokenCompleter,
) -> TrajectoryGroup:
    """Rollout a group of blackjack games."""
    return await do_group_rollout(env_builder, policy)
```

#### 4. Convert Trajectory → Episode

```python
# apps/blackjack/main.py

def trajectory_to_episode(traj: Trajectory, game_id: str) -> Episode:
    """
    Convert Tinker Trajectory to Forge Episode.

    Trajectory stores transitions (per-turn), Episode stores concatenated sequence.
    """
    all_tokens = []
    all_logprobs = []
    response_mask = []

    for transition in traj.transitions:
        # Observation tokens (prompt)
        ob_tokens = transition.ob.input_ids.tolist()
        all_tokens.extend(ob_tokens)
        response_mask.extend([0] * len(ob_tokens))
        all_logprobs.extend([0.0] * len(ob_tokens))

        # Action tokens (response)
        ac_tokens = transition.ac.tokens
        ac_logprobs = transition.ac.logprobs
        all_tokens.extend(ac_tokens)
        response_mask.extend([1] * len(ac_tokens))
        all_logprobs.extend(ac_logprobs)

    # Final reward from last transition
    final_reward = traj.transitions[-1].reward if traj.transitions else 0.0

    return Episode(
        episode_id=game_id,
        task_name="blackjack",
        generator_version=0,  # TODO: Get from policy
        is_truncated=False,  # TODO: Add truncation tracking
        all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
        logprobs=torch.tensor(all_logprobs, dtype=torch.float),
        response_mask=torch.tensor(response_mask, dtype=torch.float),
        reward=final_reward,
        metadata={
            "num_turns": len(traj.transitions),
            "game_id": game_id,
        }
    )
```

#### 5. Updated Continuous Rollouts

```python
# apps/blackjack/main.py

async def continuous_rollouts():
    """Main rollout loop using Tinker pattern."""

    # Setup renderer (model-specific, task-agnostic)
    renderer = get_renderer(cfg.policy.model)  # Qwen3Renderer, Llama3Renderer, etc.

    # Setup env builder
    env_builder = BlackjackEnvGroupBuilder(
        server_url=cfg.blackjack_env.server_url,
        renderer=renderer,
        system_prompt="You are an expert BlackJack player...",
        num_envs=cfg.grpo.group_size,
    )

    while not shutdown_event.is_set():
        # ============ Step 1: Rollout group (Tinker's generic function) ============
        trajectory_group = await do_group_rollout(env_builder, policy)

        # ============ Step 2: Convert trajectories → episodes ============
        episodes = [
            trajectory_to_episode(traj, game_id=str(uuid.uuid4()))
            for traj in trajectory_group.trajectories
        ]

        # ============ Step 3: Filter groups (constant rewards) ============
        rewards = [e.reward for e in episodes]
        if len(set(rewards)) == 1:
            record_metric("groups/rate_dropped", 1, Reduce.MEAN)
            continue
        record_metric("groups/rate_dropped", 0, Reduce.MEAN)

        # ============ Step 4: Compute ref_model ============
        max_len = max(len(e.all_token_ids) for e in episodes)
        padded_tokens = [
            F.pad(e.all_token_ids, (0, max_len - len(e.all_token_ids)), value=pad_id)
            for e in episodes
        ]
        input_ids = torch.stack(padded_tokens)

        ref_logprobs_padded = await ref_model.forward.route(input_ids, 0, return_logprobs=True)

        # Assign unpadded ref_logprobs
        for i, episode in enumerate(episodes):
            seq_len = len(episode.all_token_ids)
            episode.ref_logprobs = ref_logprobs_padded[i, :seq_len]

        # ============ Step 5: Compute advantages ============
        advantages = await compute_advantages.compute.call_one(episodes)
        for episode, advantage in zip(episodes, advantages):
            episode.advantage = advantage

        # ============ Step 6: Episode-level acceptance ============
        accepted_episodes = []
        for episode in episodes:
            should_accept = True
            if episode.is_truncated and not cfg.grpo.accept_truncated:
                should_accept = False
                record_metric("buffer/rate_rejected_truncated", 1, Reduce.MEAN)
            else:
                record_metric("buffer/rate_rejected_truncated", 0, Reduce.MEAN)

            if should_accept:
                accepted_episodes.append(episode)

        # ============ Step 7: Add to buffer ============
        for episode in accepted_episodes:
            await replay_buffer.add.call_one(episode)
```

---

## Handling Truncation with Env Pattern

### Where Does max_seq_len Fit?

**Problem:** Tinker's `Env` doesn't know about token budgets - it returns `ModelInput` (already tokenized).

**Solution:** Add budget tracking to `StepResult` via `metrics`:

```python
class BlackjackEnv(Env):
    def __init__(self, renderer, server_url, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.cumulative_tokens = 0

    async def initial_observation(self):
        obs = self.renderer.build_generation_prompt(self.messages)
        self.cumulative_tokens = obs.length
        return obs, self.renderer.stop_condition

    async def step(self, action):
        # Track cumulative tokens
        self.cumulative_tokens += len(action)

        # Check if we're approaching budget
        if self.cumulative_tokens >= self.max_seq_len:
            # Mark episode as truncated via metrics
            return StepResult(
                reward=self._compute_reward(...),
                episode_done=True,  # Force termination
                next_observation=tinker.ModelInput.empty(),
                metrics={"is_truncated": True, "truncation_reason": "max_seq_len"},
                ...
            )

        # Normal step logic...
```

**Rollout loop extracts truncation info:**
```python
def trajectory_to_episode(traj: Trajectory, game_id: str) -> Episode:
    # Check last transition for truncation
    last_transition = traj.transitions[-1]
    is_truncated = last_transition.metrics.get("is_truncated", False)
    truncation_reason = last_transition.metrics.get("truncation_reason", None)

    return Episode(
        is_truncated=is_truncated,
        metadata={"truncation_reason": truncation_reason, ...},
        ...
    )
```

---

## Summary of Changes to V3

### Delete
1. ❌ Initial prompt check before while loop (Issue 1)
2. ❌ Hardcoded timeout=60.0 (Issue 3 - make configurable)
3. ❌ The entire `do_single_rollout()` function in V3 (use Tinker's instead)

### Fix
1. ✅ `generator_version` from `completion.generator_version` (Issue 2)
2. ✅ Double padding: Keep padding in both places but unpad when storing (Issue 4)
3. ✅ Slicing is actually correct (Issue 5 - no bug)

### Add
1. ✅ `BlackjackEnv(Env)` class following Tinker pattern
2. ✅ `BlackjackEnvGroupBuilder(EnvGroupBuilder)`
3. ✅ `trajectory_to_episode()` conversion function
4. ✅ Budget tracking via `StepResult.metrics`
5. ✅ Optional timeout config

### Refactor
1. ✅ Use Tinker's `do_single_rollout()` and `do_group_rollout()` directly
2. ✅ Move all game logic into `BlackjackEnv`
3. ✅ Move all formatting into `Renderer` (already exists in Forge)
4. ✅ Keep rollout loop 100% generic

---

## Final Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Training Loop                        │
│                  (continuous_rollouts)                       │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              BlackjackEnvGroupBuilder                        │
│  • Creates group_size BlackjackEnv instances                 │
│  • Injects Renderer (Qwen3Renderer, etc.)                    │
└────────┬────────────────────────────────────────────────────┘
         │ make_envs()
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   BlackjackEnv (Env)                         │
│  • Manages OpenSpielClient                                   │
│  • Builds messages (user/assistant)                          │
│  • Parses actions from text                                  │
│  • Computes rewards                                          │
│  • Tracks budget via cumulative_tokens                       │
│  • Returns tokenized observations via Renderer               │
└────────┬───────────────────────────┬────────────────────────┘
         │                           │
         │ initial_observation()     │ step(action_tokens)
         │ returns ModelInput        │ returns StepResult
         ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Tinker's Generic Rollout Loop                      │
│           (do_single_rollout, do_group_rollout)              │
│  • Calls policy(obs, stop_cond) → action_tokens              │
│  • Calls env.step(action_tokens) → StepResult                │
│  • Records Transition(ob, ac, reward, done)                  │
│  • Returns Trajectory (list of transitions)                  │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              trajectory_to_episode()                         │
│  • Concatenates all transitions into single sequence         │
│  • Builds response_mask (0 for prompts, 1 for responses)     │
│  • Extracts final reward                                     │
│  • Returns Episode (Forge format)                            │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              GRPO Training (same as V3)                      │
│  • Filter groups (constant rewards)                          │
│  • Compute ref_model                                         │
│  • Compute advantages                                        │
│  • Episode-level acceptance                                  │
│  • Add to replay buffer                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Config Schema (Updated)

```yaml
blackjack_env:
  server_url: "http://localhost:8004"
  max_seq_len: 2048              # Episode-level budget
  max_turns: 10                  # Hard limit on turns
  generation_timeout: null       # Optional (e.g., 600.0), null = no timeout

grpo:
  group_size: 16
  accept_truncated: true

truncation:
  # Note: drop_truncated_generation not needed with Env pattern
  # Env decides when to terminate via episode_done flag

policy:
  model: "Qwen/Qwen3-1.7B"
  engine_args:
    enable_prefix_caching: true
    max_model_len: 4096
```

---

## Migration Checklist

- [ ] Create `apps/blackjack/env.py` with `BlackjackEnv` class
- [ ] Create `BlackjackEnvGroupBuilder`
- [ ] Add `trajectory_to_episode()` conversion function
- [ ] Update `continuous_rollouts()` to use Tinker's pattern
- [ ] Remove hardcoded timeout, add optional config
- [ ] Fix `generator_version` to use `completion.generator_version`
- [ ] Verify padding logic (pad → unpad → pad again is correct)
- [ ] Add budget tracking via `StepResult.metrics`
- [ ] Test with single game
- [ ] Test with group rollout
- [ ] Verify truncation handling
- [ ] Verify metrics tracking

---

**End of Document**
