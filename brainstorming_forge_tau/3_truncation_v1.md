# Max Seq Len and Truncation Strategies Across Frameworks

## Key Findings: How Different Frameworks Handle max_seq_len and Truncation

### 1. TRL (Example: catch.py, wordle.py) - Token Concatenation Pattern

**File:** `4_examples_APIs.md:3062-3070`

```python
# EACH TURN adds to the same lists
episode_prompt_ids.extend(result["prompt_ids"][0])
episode_completion_ids.extend(result["completion_ids"][0])
episode_logprobs.extend(result["logprobs"][0])
```

**Key points:**
- Concatenates all turns into ONE sequence
- max_seq_len applies to ENTIRE episode (not per turn)
- Truncation happens at EPISODE level (if total tokens > max_seq_len)
- No explicit truncation handling shown in examples
- Risk: Long episodes could exceed model's context window

---

### 2. VERL - Explicit Max Length Tracking

**File:** `4_examples_APIs.md:1226-1228`

```python
# Check termination conditions
if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
    return AgentState.TERMINATED
```

**Key points:**
- Tracks cumulative response length across turns: `len(agent_data.response_mask)`
- Terminates episode when hitting `max_seq_len`
- `response_length` is the max allowed tokens for ENTIRE episode
- Prevents exceeding model limits by early termination

**Tool result truncation:**
```yaml
multi_turn:
  max_tool_response_length: 2048
  tool_response_truncate_side: "left"  # or "right" or "middle"
```

```python
# File: verl/experimental/agent_loop/tool_agent_loop.py:1360-1367
if len(tool_response_text) > self.max_tool_response_length:
    if self.tool_response_truncate_side == "left":
        tool_response_text = tool_response_text[:max_len] + "...(truncated)"
    elif self.tool_response_truncate_side == "right":
        tool_response_text = "(truncated)..." + tool_response_text[-max_len:]
    else:  # middle
        half = max_len // 2
        tool_response_text = tool_response_text[:half] + "...(truncated)..." + tool_response_text[-half:]
```

---

### 3. NeMo-RL - Dynamic Tool Result Truncation

**File:** `RL/nemo_rl/experience/rollouts.py:721-726`

```python
# Check for sequence length overflow
if input_lengths + gen_token_count + len(tokenized_obs) >= max_seq_len:
    # Truncate environment observation to fit budget
    max_env_tokens = max_seq_len - input_lengths - gen_token_count
    if max_env_tokens > 0:
        tokenized_obs = tokenized_obs[:max_env_tokens]
    else:
        tokenized_obs = torch.tensor([], dtype=torch.int64)
```

**Key points:**
- max_seq_len applies to full episode (all turns concatenated)
- max_rollout_turns limits number of turns (orthogonal to seq_len)
- Dynamic tool/env truncation: Truncates tool results to fit remaining budget
- Truncation strategy: Left-truncation (keeps most recent tokens)

---

### 4. Verifiers/PRIME-RL - Multi-Turn with Max Turns Limit

**File:** `4_examples_APIs.md:2660`

```python
class ToolEnv(MultiTurnEnv):
    def __init__(self, tools: list[Callable], max_turns: int = 10, **kwargs):
```

**Key points:**
- `max_turns` limits number of interactions (not token count!)
- No explicit `max_seq_len` - episodes end when:
  1. Assistant responds without tool calls
  2. Max turns reached
  3. Task completed
- Tool responses can be truncated:

```python
# File: 4_examples_APIs.md:1358-1368
if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
    if self.tool_response_truncate_side == "left":
        tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"
    elif self.tool_response_truncate_side == "right":
        tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
```

---

### 5. Tinker-Cookbook - All-or-Nothing Termination

**UPDATED WITH ACTUAL CODE ANALYSIS**

#### How Prompts are Built

**File:** `tinker-cookbook/tinker_cookbook/renderers.py` (Qwen3Renderer example)

```python
def build_generation_prompt(
    self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
) -> tinker.ModelInput:
    """Build prompt for generation from message history."""
    tokens: list[int] = []  # No BOS token for Qwen
    for idx, message in enumerate(messages):
        ob_part, action_part, _ = self._render_message(idx, message)
        tokens.extend(ob_part)  # Add observation part
        tokens.extend(action_part)  # Add action part
    # Add generation prompt
    new_partial_message = Message(role=role, content="")
    ob_part, _, _ = self._render_message(len(messages), new_partial_message)
    tokens.extend(ob_part)
    tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
    return tinker.ModelInput.from_ints(tokens)
```

**Key insight:** NO `apply_chat_template` - They manually build prompts by iterating messages!

#### How max_tokens is Enforced

**File:** `tinker-cookbook/tinker_cookbook/completers.py:50-74`

```python
@dataclass
class TinkerTokenCompleter(TokenCompleter):
    sampling_client: tinker.SamplingClient
    max_tokens: int

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(stop=stop, max_tokens=self.max_tokens),
        )
```

**Key points:**
- `max_tokens` is at completer level (not environment level)
- Passed to `SamplingParams(max_tokens=self.max_tokens)`
- Limits only generation length per turn, NOT prompt length
- No enforcement of total sequence length

#### Multi-Turn Truncation Strategy

**File:** `tinker-cookbook/tinker_cookbook/recipes/tool_use/search/search_env.py:185-191`

```python
async def step(self, action: Action) -> StepResult:
    message, parse_success = self.renderer.parse_response(action)
    self.past_messages.append(message)

    if "tool_calls" in message:
        tool_return_message = await self.call_search_tool(message["tool_calls"][0])
        self.past_messages.extend(tool_return_message)

        # Rebuild prompt from FULL history
        next_observation = self.renderer.build_generation_prompt(self.past_messages)

        # Check if exceeded max length
        if next_observation.length > self.max_trajectory_tokens:
            return StepResult(
                reward=0.0,
                episode_done=True,  # TERMINATE with failure
                next_observation=tinker.ModelInput.empty(),
            )
```

**Constructor:**
```python
class SearchEnv(ProblemEnv):
    def __init__(self, ..., max_trajectory_tokens: int = 32 * 1024):
        self.past_messages: list[renderers.Message] = []
        self.max_trajectory_tokens = max_trajectory_tokens
```

**Key points:**
- Full history maintained in `self.past_messages`
- Prompts rebuilt from scratch each turn with ALL messages
- All-or-nothing: If `next_observation.length > max_trajectory_tokens`, episode terminates with failure
- No tool result truncation - accepts results as-is
- Default: 8K tokens (configurable, code shows 32K max)

#### What They Track

**File:** `tinker-cookbook/tinker_cookbook/rl/rollouts.py:48-79`

```python
rows.append({
    "step": t_idx,
    "ob_len": t.ob.length,  # Prompt length at this step
    "ac_len": len(t.ac.tokens),  # Response length
    "reward": f"{t.reward:.3f}",
})
```

- Log `ob.length` and `ac_len` per step for diagnostics only
- NOT used for truncation decisions
- Only for metrics reporting

---

### 6. Your Current Plan (PLAN.md) - Detection but No Strategy

**File:** `PLAN.md:649-663`

```python
# Check if response was truncated by max_tokens
if response.stop_reason == "length":
    # Response was cut off by max_tokens
    has_truncated_response = True
    # Mark for tracking, but continue game
    record_metric("game/truncated_response_rate", 1, Reduce.MEAN)
```

**Issues:**
- Detects truncation but doesn't prevent episode from growing too long
- No cumulative token tracking across turns
- Risk: Episode could exceed total `max_seq_len` even if individual turns don't truncate

---

## Summary Table: How Libraries Handle max_seq_len

| Library | max_seq_len Scope | Truncation Strategy | Tool Result Handling | Prompt Building |
|---------|-------------------|---------------------|----------------------|-----------------|
| **TRL** | Entire episode | None - relies on vLLM max_model_len | No truncation | `apply_chat_template` per turn |
| **VERL** | Entire episode | Early termination + tool truncation | 3 modes: left/right/middle | Manual/SGLang |
| **NeMo-RL** | Entire episode | Dynamic tool truncation to fit budget | Left-truncate to remaining budget | `apply_chat_template` per turn |
| **PRIME-RL/Verifiers** | N/A (uses max_turns) | No episode-level limit | No truncation | `apply_chat_template` with tools |
| **Tinker** | 8K default | All-or-nothing termination | No truncation, episode fails if exceeded | Manual token concat |

---

## Answers to Your Questions

### Q1: "So we would only have max_seq_len, truncate prompt, and dynamically set limit to generate?"

**YES, with clarifications:**

**What "max_seq_len" means:**
- Total token budget for ENTIRE episode (all turns concatenated)
- Includes: all prompts + all responses + all tool results across ALL turns
- Example: `max_seq_len=2048` means episode terminates when cumulative tokens ≥ 2048

**Two patterns observed:**

#### **Option A: Tinker Pattern (Simpler)**
- Build prompt from full message history each turn
- Check if prompt exceeds `max_seq_len` → terminate if so
- Calculate remaining budget and set `max_tokens` dynamically
- NO prompt truncation - always use full history

#### **Option B: VERL Pattern (More Explicit)**
- Track cumulative tokens in lists: `all_token_ids`, `all_logprobs`, `response_mask`
- Check if adding next prompt would exceed limit → terminate early
- Calculate remaining budget per turn
- Build response masks for training
- More bookkeeping, but safer

### Q2: "Is this how others do it?"

**Yes, most libraries use one of these patterns:**

| Library | Approach |
|---------|----------|
| **Tinker** | Option A - Terminate if exceeds limit |
| **VERL** | Option B - Track cumulative, terminate early |
| **NeMo-RL** | Option B - Dynamic tool truncation |
| **TRL** | No explicit handling (relies on vLLM limits) |
| **Verifiers** | `max_turns` only, no token limit |

**Recommendation:** Start with Option A for simplicity. Use Option B if you need explicit token tracking for training.

### Q3: "We would need to truncate prompt?"

**NO - Don't truncate the prompt (no sliding window).**

**Why not:**
1. Tinker/VERL rebuild from full history every turn - no truncation
2. Truncating loses context (model can't see previous tool results)
3. Makes training inconsistent

**What to do instead:**
- Terminate episode early if prompt would exceed `max_seq_len`
- Track cumulative length (Option B) or check prompt length each turn (Option A)
- Adjust `max_turns` to keep episodes within budget
- Tune `max_seq_len` based on task requirements

**When you SHOULD truncate:**
- **Tool results** (VERL & NeMo-RL do this):
  ```python
  # Fixed-length truncation
  if len(tool_result) > 1024:
      tool_result = tool_result[:1024] + "...(truncated)"

  # Dynamic truncation to fit remaining budget
  remaining_budget = max_seq_len - (prompt_len + generated_len)
  if len(tool_result_tokens) > remaining_budget:
      tool_result_tokens = tool_result_tokens[:max(0, remaining_budget)]
  ```

### Q4: "For policy.generate, max_tokens is not an arg, but now we have sampling_params"

**CORRECT!** Pass `max_tokens` via `sampling_params` dict:

```python
# Correct way
response = await policy.generate.route(
    prompt_text,
    sampling_params={"max_tokens": turn_max_tokens}
)
```

**How it works:** The dict is unpacked into vLLM's `SamplingParams`:
```python
# Inside Generator._generate() in forge/actors/generator.py
outputs = await self._engine.generate(
    prompts=[prompt_ids],
    sampling_params=SamplingParams(**sampling_params),
)
```

**Available sampling_params:**
- `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, etc. (all vLLM SamplingParams)

## Recommended Strategy for Forge

### Simple Implementation Pattern

**Use Option B (explicit tracking) for better control:**

```python
async def play_game(
    game_idx: int,
    game_id: str,
    server_url: str,
    policy: Generator,
    tokenizer,
    max_seq_len: int = 2048,
    max_turns: int = 10,
    rollout_count: int = 0,
) -> Episode:
    messages = [{"role": "system", "content": "You are a blackjack expert..."}]

    # Track tokens
    all_tokens = []
    all_logprobs = []
    response_mask = []

    env = OpenSpielEnv(base_url=server_url)
    result = env.reset()

    for turn in range(max_turns):
        if result.done:
            break

        # Build prompt from messages
        user_message = format_game_state(result.observation)
        messages.append({"role": "user", "content": user_message})

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenize to check length
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Check if prompt exceeds budget
        if len(all_tokens) + len(prompt_tokens) >= max_seq_len:
            record_metric("game/truncated_episode_rate", 1, Reduce.MEAN)
            break

        # Calculate budget for response
        remaining = max_seq_len - (len(all_tokens) + len(prompt_tokens))
        turn_max_tokens = min(256, remaining)

        # Safety check for negative or very small budgets
        if turn_max_tokens <= 0:
            break

        # Generate
        responses = await policy.generate.route(
            [prompt_text],
            sampling_params={"max_tokens": turn_max_tokens}
        )
        response = responses[0]

        # Accumulate
        all_tokens.extend(prompt_tokens)
        all_tokens.extend(response.token_ids)
        response_mask.extend([0] * len(prompt_tokens))
        response_mask.extend([1] * len(response.token_ids))
        all_logprobs.extend([0.0] * len(prompt_tokens))
        all_logprobs.extend(response.logprobs)

        # Add assistant response
        messages.append({"role": "assistant", "content": response.text})

        # Parse action and step env
        action = parse_action(response.text)
        result = env.step(OpenSpielAction(action_id=action, game_name="blackjack"))

    # Create episode
    episode = Episode(
        episode_id=game_id,
        all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
        logprobs=torch.tensor(all_logprobs, dtype=torch.float),
        response_mask=torch.tensor(response_mask, dtype=torch.float),
        reward=result.reward,
        ...
    )

    return episode
```

**Key points:**
- Use `tokenizer.apply_chat_template()` each turn
- Track cumulative tokens
- Dynamically set `max_tokens` via `sampling_params`
- Terminate early if budget exceeded
- No prompt truncation, use full message history

### For Future Tool Calling

Same pattern, but add tool results to messages:

```python
# After generating
if has_tool_call(response.text):
    tool_call = parse_tool_call(response.text)
    messages.append({
        "role": "assistant",
        "content": response.text,
        "tool_calls": [tool_call]
    })

    # Execute tool
    tool_result = await execute_tool(tool_call)

    # Truncate long tool results (recommended!)
    max_tool_len = 1024
    if len(tool_result) > max_tool_len:
        tool_result = tool_result[:max_tool_len] + "...(truncated)"
        record_metric("tool/truncated_result_rate", 1, Reduce.MEAN)

    messages.append({
        "role": "tool",
        "content": tool_result
    })

    # Continue loop - reformats with updated messages
```

---

## Key Recommendations

1. Use explicit token tracking (Option B pattern) for better control
2. Set `max_seq_len` conservatively (e.g., 2048 for blackjack, 4096 for tool calling)
3. Always use `tokenizer.apply_chat_template()` in rollout loop
4. Pass `max_tokens` via `sampling_params` dict
5. Track cumulative tokens to prevent exceeding budget
6. Don't truncate prompts - terminate episode instead
7. DO truncate tool results to control their size
8. Log truncation events for debugging


# Key Takeaways & Follow-ups

## Critical Bugs to Address

### 1. Empty Budget Can Cause Negative max_tokens Error

**Problem:**
```python
remaining_budget = max_seq_len - (len(all_token_ids) + len(prompt_tokens))
turn_max_tokens = min(256, remaining_budget)  # Can be negative!
```

**Fix:**
```python
remaining = max_seq_len - (len(all_tokens) + len(prompt_tokens))
if remaining <= 0:  # Check BEFORE min()
    record_metric("episode/terminated_zero_budget", 1, Reduce.MEAN)
    break
turn_max_tokens = min(256, remaining)
```

### 2. Mid-Tool-Call Truncation Corrupts Training Data

**Problem:** If `max_tokens` cuts off response mid-tool-call:
```
<tool_call>{"name": "search", "args": {"query": "Pytho[TRUNCATED]
```
- Tool call is incomplete → parsing fails
- But `response_mask` still has `[1, 1, 1, ...]`
- We train on corrupted output!

**Fix:**
```python
if response.stop_reason == "length":
    # Detect incomplete tool call
    has_tool_start = "<tool_call>" in response.text
    has_tool_end = "</tool_call>" in response.text

    if has_tool_start and not has_tool_end:
        record_metric("episode/truncated_mid_tool_call", 1, Reduce.MEAN)
        break  # Terminate episode, don't add to buffer
```

### 3. Reference Model Variable Sequence Lengths

**Current issue:** `max_req_tokens` is fixed, but multi-turn episodes have variable lengths.

**Fix:** Pass actual sequence length to ref model:
```python
for episode in episodes:
    seq_len = len(episode.all_token_ids)
    ref_logprobs = await ref_model.forward.route(
        episode.all_token_ids.unsqueeze(0),  # [1, seq_len]
        prompt_len=0,  # Use response_mask instead
        return_logprobs=True
    )
```

---

## Important Implementation Details

### Multiple Tool Calls Count as 1 Turn

**Both VERL and Verifiers do this:**
- Execute all tool calls in parallel
- Add all tool results to messages at once
- Token budget: `len(assistant_msg) + sum(len(tool_result) for each tool)`

```python
if response.tool_calls:
    # Execute all
    tool_results = [await execute_tool(tc) for tc in response.tool_calls]
    # Truncate each
    tool_results = [tr[:max_len] + "..." if len(tr) > max_len else tr
                    for tr in tool_results]
    # Add all to messages
    messages.extend([{"role": "tool", "content": tr} for tr in tool_results])
```

### vLLM Prefix Caching - Must Enable!

**Critical optimization for multi-turn:**
```yaml
policy:
  engine_args:
    enable_prefix_caching: true  # 2-3x speedup
```

**How it works:** Caches KV tensors for shared prompt prefixes across turns
- Turn 1: `[system, user1]`
- Turn 2: `[system, user1, assist1, tool1, user2]` ← first 3 cached
- Turn 3: `[system, user1, assist1, tool1, user2, assist2, tool2, user3]` ← first 7 cached

---

## Required Config Changes

Add to `apps/blackjack/qwen3_1_7b.yaml`:

```yaml
blackjack_env:
  max_seq_len: 2048              # Total episode token budget
  max_turns: 10                  # Max turns per episode
  max_tool_result_length: 1024   # Truncate tool results

policy:
  engine_args:
    enable_prefix_caching: true  # Critical for multi-turn
    max_model_len: 4096
```

In `main.py`:
```python
max_seq_len = cfg.blackjack_env.get("max_seq_len", 2048)
max_turns = cfg.blackjack_env.get("max_turns", 10)
max_tool_result_length = cfg.blackjack_env.get("max_tool_result_length", 1024)

# Validation
assert max_seq_len <= cfg.policy.engine_args.max_model_len
```

---

## Environment-Specific Budgets (Future)

Different tasks need different budgets:

| Environment | `max_seq_len` | `max_tool_result_length` | Reason |
|------------|---------------|--------------------------|---------|
| **Blackjack** | 2048 | 0 (no tools) | Simple game, short episodes |
| **Coding** | 4096 | 1024 | Code output moderate length |
| **WebSearch** | 8192 | 2048 | Search results can be long |

**Implementation:** Use per-environment config or dynamic budgets per tool type.

---

## Key Metrics to Track

**For debugging truncation:**

```python
# Episode-level
record_metric("episode/total_tokens", len(all_tokens), Reduce.MEAN)
record_metric("episode/num_turns", num_turns, Reduce.MEAN)
record_metric("episode/truncation_rate", 1 if truncated else 0, Reduce.MEAN)

# Turn-level
record_metric("turn/remaining_budget", remaining_budget, Reduce.MEAN)

# Critical errors
record_metric("episode/truncated_mid_tool_call", 1, Reduce.MEAN)
record_metric("episode/terminated_zero_budget", 1, Reduce.MEAN)
```

---

## Follow-up Questions

1. **Training quality:** Should we filter out truncated episodes or down-weight their advantages?
2. **Tool result truncation:** Fixed-length (1024) or dynamic based on remaining budget?
3. **Truncation strategy:** Should we have per-tool budgets (e.g., search=2048, execute=512)?
4. **Episode metadata:** Do we need to track `truncated` flag and `truncation_reason` for debugging?

---

## Main Learnings

1. **No prompt truncation** - terminate episode instead (Tinker/VERL approach)
2. **Always check remaining budget before `min()`** - avoid negative max_tokens
3. **Detect incomplete tool calls** - don't train on corrupted data
4. **Enable prefix caching** - 2-3x speedup for multi-turn
5. **Truncate tool results** - they grow the prompt quickly
6. **Track cumulative tokens** - prevent exceeding budget mid-episode
7. **Use `sampling_params` dict** - pass `max_tokens` dynamically per turn

---

## Open Questions from User Discussion

### Q1: When to Call tokenizer.encode()? (Inside or Outside While Loop?)

**Current recommendation (line 393):**
```python
for turn in range(max_turns):
    # Build prompt from messages
    prompt_text = tokenizer.apply_chat_template(messages, ...)

    # Tokenize to check length
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)  # INSIDE loop

    if len(all_tokens) + len(prompt_tokens) >= max_seq_len:
        break
```

**User question:** Should we encode only once at the start (outside while loop) instead?

**Status:** NEEDS RESEARCH - Check how TRL, VERL, NeMo-RL, Tinker, Verifiers handle this:
- Do they re-encode the full prompt each turn?
- Or do they track message-by-message token counts?
- Performance implications of encoding vs tracking?

---

### Q2: max_tool_result_length - Global vs Tool-Specific?

**Current recommendation (line 599):**
```yaml
blackjack_env:
  max_tool_result_length: 1024   # Global for all tools
```

**User question:** What should the signature be for tool-calling? Per-tool limits? Global? Dynamic?

**Status:** NEEDS RESEARCH - Check how VERL, Verifiers, NeMo-RL configure tool result truncation:
- Is `max_tool_result_length` global or per-tool?
- Do they have different limits for different tool types?
- How do they specify this in configs?
- Example: search results (2048) vs code execution (512)?

---

### Q3: Mid-Tool-Call Truncation - Is It Really a Special Problem?

**Current recommendation (lines 516-536):**
```python
if response.stop_reason == "length":
    # Detect incomplete tool call
    has_tool_start = "<tool_call>" in response.text
    has_tool_end = "</tool_call>" in response.text

    if has_tool_start and not has_tool_end:
        record_metric("episode/truncated_mid_tool_call", 1, Reduce.MEAN)
        break  # Terminate episode, don't add to buffer
```

**User skepticism:** If we're already evicting truncated episodes via `is_truncated` flag, why is mid-tool-call truncation special?

**Counter-argument:** Mid-tool-call creates invalid JSON → unparseable → corrupt training signal even if we mark episode as truncated.

**Status:** NEEDS RESEARCH - Check how other libraries handle generation truncation during tool calls:
- Do VERL, Verifiers, NeMo-RL detect incomplete tool calls specifically?
- Or do they just rely on general truncation handling?
- Do they immediately terminate or try to continue?
- Do they filter these episodes from training?

---

### Q4: Multiple Tool Calls + Budget Overflow - What Happens?

**Current recommendation (lines 557-573):**
```python
if response.tool_calls:
    # Execute all
    tool_results = [await execute_tool(tc) for tc in response.tool_calls]
    # Truncate each
    tool_results = [tr[:max_len] + "..." if len(tr) > max_len else tr
                    for tr in tool_results]
    # Add all to messages
    messages.extend([{"role": "tool", "content": tr} for tr in tool_results])
```

**Problem scenario:**
- Model makes 3 tool calls in one turn
- Each truncated to `max_tool_result_length=1024`
- Total: 3072 tokens
- But remaining budget: 300 tokens
- What to do?

**Proposed options:**
1. **Terminate episode** (safest, all-or-nothing)
2. **Fair allocation** (divide remaining budget by num tools)
3. **Keep first N tools that fit** (drop later ones)

**User preference:** Allow truncated tool output, let user decide eviction policy via config.

**Status:** NEEDS RESEARCH - Check how VERL, Verifiers, NeMo-RL handle multiple tool calls when total exceeds budget:
- Do they terminate the episode?
- Do they truncate all tool results to fit remaining budget?
- Do they keep only tools that fit?
- Is this configurable?

---

### Q5: Deprecate prompt_len in Reference Model

**Current Episode class:**
```python
@dataclass
class Episode:
    pad_id: int
    request_len: int  # Fixed length (legacy)
    response_len: int  # Fixed length (legacy)
```

**New Episode class:**
```python
@dataclass
class Episode:
    all_token_ids: torch.Tensor  # Variable length
    response_mask: torch.Tensor  # Replaces request_len/response_len
```

**User decision:** Clean break, no backward compatibility. Add clear error message if old fields detected.

**Rationale:**
1. Multi-turn is fundamental change anyway
2. Adding backward compat adds noise (`if prompt_len > 0: ... else: ...`)
3. Only small number of users (easier migration)
4. Maintains single code path

**Status:** DECIDED - Break at once, no backward compat.

---

## Research Tasks (IN ORDER)

**Before implementing, we need to research the following libraries to answer the open questions:**

1. **TRL** (`trl/examples/scripts/openenv/`)
2. **VERL** (`verl/experimental/agent_loop/`)
3. **NeMo-RL** (`RL/nemo_rl/experience/rollouts.py`)
4. **Tinker-Cookbook** (`tinker-cookbook/recipes/tool_use/`)
5. **Verifiers** (`verifiers/envs/`)

**For each library, investigate:**
- **Q1:** Where do they call tokenizer.encode()? Inside or outside turn loop?
- **Q2:** How do they configure max_tool_result_length? Global or per-tool?
- **Q3:** Do they detect/handle mid-tool-call truncation specially?
- **Q4:** How do they handle multiple tool calls when total exceeds budget?

**Research output:** Add findings to new section below titled "## Research Findings"

---

## Research Findings

### Q1 Research: When/Where to Call tokenizer.encode()

**Finding: Libraries use TWO distinct patterns - re-encode everything vs. incremental tracking**

#### Pattern A: Re-Encode Full Prompt Each Turn (TRL, Tinker, Verifiers)

**TRL Catch** (`trl/examples/scripts/openenv/catch.py:177-196`):
```python
while not obs.done:  # INSIDE loop
    episode_msg = {"prompt": [{"role": "user", "content": f"{base_prompt}\n\n{obs.info_state}\n"}]}
    episode_prompt = apply_chat_template(episode_msg, processing_class)

    # vLLM server returns prompt_ids
    response = requests.post(gen_url, json=payload)
    result = response.json()

    # Accumulate tokens
    episode_prompt_ids.extend(result["prompt_ids"][0])
    episode_completion_ids.extend(result["completion_ids"][0])
```

**TRL Wordle** (`trl/examples/scripts/openenv/wordle.py:352-383`):
```python
for _turn in range(cli_args.max_turns):  # INSIDE loop
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    vllm_result = request_vllm_completion(...)
    prompt_ids.extend(vllm_result["prompt_ids"])
    completion_ids.extend(vllm_result["completion_ids"])
```

**Tinker** (`tinker-cookbook/tinker_cookbook/renderers.py:189-202`):
```python
def build_generation_prompt(self, messages: list[Message]) -> tinker.ModelInput:
    tokens: list[int] = []
    tokens.extend(self._bos_tokens)
    for message in messages:  # OUTSIDE loop - called once per generation
        ob_part, action_part, action_tail = self._render_message(message)
        tokens.extend(ob_part)
        tokens.extend(action_part)
    return tinker.ModelInput.from_ints(tokens)
```

**Key insight:** They call `apply_chat_template()` or build prompt from scratch each turn, but the vLLM/generator returns the token IDs, so they don't explicitly call `tokenizer.encode()` themselves.

#### Pattern B: Incremental Token Tracking (NeMo-RL, VERL)

**NeMo-RL** (`RL/nemo_rl/experience/rollouts.py:446-477`):
```python
for turn in range(max_rollout_turns):  # INSIDE loop
    # Only tokenize NEW environment observation
    tokenized_obs = tokenizer(
        env_obs_content,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids[0]

    # Check if adding new tokens would overflow
    if (len(tokenized_obs) + len(generated_ids[i]) + active_input_lengths[i] >= max_seq_len):
        tokens_left_for_obs = max_seq_len - (len(generated_ids[i]) + active_input_lengths[i])
        tokenized_obs = tokenized_obs[:tokens_left_for_obs]  # Truncate to fit
        truncation_mask[i] = True
```

**VERL** (`verl/experimental/agent_loop/tool_agent_loop.py:200-209, 351-358`):
```python
# Initial prompt - OUTSIDE loop
agent_data.prompt_ids = await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        agent_data.messages, tools=self.tool_schemas,
        add_generation_prompt=True, tokenize=True
    ),
)

# Tool responses - INSIDE loop
response_ids = await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        add_messages, add_generation_prompt=True, tokenize=True
    ),
)

# Check budget
if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
    return AgentState.TERMINATED
```

**Verifiers** (post-processing - `verifiers/utils/processing_utils.py:95-155`):
```python
# Initial prompt - OUTSIDE loop
prompt_ids = processing_class.apply_chat_template(
    conversation=prompt, add_generation_prompt=True, tools=oai_tools
)

# For each turn - uses prefix matching to get delta
while i < len(zipped):
    token_prefix = processing_class.apply_chat_template(
        conversation=messages_consumed, add_generation_prompt=False, tools=oai_tools
    )
    token_prefix_with_turn = processing_class.apply_chat_template(
        conversation=messages_consumed + consecutive_messages,
        add_generation_prompt=True, tools=oai_tools
    )
    # Extract ONLY the new tokens
    assert token_prefix_with_turn[:len(token_prefix)] == token_prefix
    completion_turn_ids = token_prefix_with_turn[len(token_prefix):]
```

#### **Recommendation for Forge:**

Use Pattern B (incremental) like NeMo-RL/VERL:

```python
for turn in range(max_turns):
    # Build prompt from messages
    prompt_text = tokenizer.apply_chat_template(messages, ...)

    # Encode ONLY to check length, not for generation
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Check budget BEFORE generating
    if len(all_tokens) + len(prompt_tokens) >= max_seq_len:
        break

    # Calculate remaining budget
    remaining = max_seq_len - (len(all_tokens) + len(prompt_tokens))
    turn_max_tokens = min(256, remaining)

    # Generate (vLLM returns token_ids)
    responses = await policy.generate.route([prompt_text],
                                           sampling_params={"max_tokens": turn_max_tokens})

    # Accumulate tokens from response object
    all_tokens.extend(prompt_tokens)
    all_tokens.extend(response.token_ids)
```

**Why this is best:**
- Explicit budget control before generating
- Only encodes once per turn (not redundant)
- vLLM/Generator handles actual generation
- Clear separation: encode for budget check, generate for response

---

### Q2 Research: max_tool_result_length Configuration

**Finding: ALL libraries use GLOBAL configuration, NONE support per-tool limits**

#### VERL: Global with Multiple Truncation Strategies

**Config:** `verl/verl/trainer/config/rollout/rollout.yaml:165-169`
```yaml
multi_turn:
  max_parallel_calls: 1
  max_tool_response_length: 256  # Global for all tools
  tool_response_truncate_side: middle  # left/middle/right
```

**Implementation:** `verl/experimental/agent_loop/tool_agent_loop.py:457-464`
```python
if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
    if self.tool_response_truncate_side == "left":
        tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"
    elif self.tool_response_truncate_side == "right":
        tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
    else:  # middle
        length = self.max_tool_response_length // 2
        tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]
```

**Key details:**
- Configurable via YAML
- Three truncation strategies
- No per-tool customization
- CHARACTER-based, not token-based

#### NeMo-RL: Environment-Level Token Budget

**Implementation:** `RL/nemo_rl/experience/rollouts.py:446-477`
```python
# Truncate environment observation (which includes tool results)
if len(tokenized_obs) + len(generated_ids[i]) + active_input_lengths[i] >= max_seq_len:
    tokens_left_for_obs = max_seq_len - (len(generated_ids[i]) + active_input_lengths[i])
    tokenized_obs = tokenized_obs[:tokens_left_for_obs]
    truncation_mask[i] = True
```

**Key details:**
- TOKEN-based (more accurate)
- Dynamic allocation based on remaining budget
- No explicit max_tool_result_length parameter
- No per-tool customization

#### Tinker: Trajectory-Level Termination

**Implementation:** `tinker-cookbook/recipes/tool_use/search/search_env.py:108-117, 186-187`
```python
class SearchEnv(ProblemEnv):
    def __init__(self, ..., max_trajectory_tokens: int = 32 * 1024):
        self.max_trajectory_tokens = max_trajectory_tokens

    async def step(self, action):
        # After adding tool result
        next_observation = self.renderer.build_generation_prompt(self.past_messages)
        if next_observation.length > self.max_trajectory_tokens:
            return failure_result  # Terminates episode
```

**Key details:**
- TOKEN-based
- No tool-specific limits, only total trajectory
- Terminates rather than truncates
- No per-tool customization

#### Verifiers: No Tool Result Truncation

**Implementation:** `verifiers/envs/tool_env.py:54-71`
```python
async def call_tool(self, tool_name: str, tool_args: dict, ...) -> Message:
    tool_func = self.tool_map[tool_name]
    result = await maybe_await(tool_func, **tool_args)
    return {
        "role": "tool",
        "content": str(result),  # No truncation!
        "tool_call_id": tool_call_id,
    }
```

**Key details:**
- No tool result truncation at all
- Relies on sequence-level truncation/masking
- No per-tool customization

#### **Summary Table**

| Library | Scope | Unit | Default | Per-Tool? | Config Type |
|---------|-------|------|---------|-----------|-------------|
| **VERL** | Global | Characters | 256 | No | YAML config |
| **NeMo-RL** | Environment observation | Tokens | Dynamic (based on max_seq_len) | No | Function param |
| **Tinker** | Trajectory | Tokens | 32,768 | No | Constructor arg |
| **Verifiers** | None | N/A | N/A | No | N/A |

#### **Recommendation for Forge:**

**Phase 1: Global configuration (like VERL)**
```yaml
blackjack_env:
  max_tool_result_length: 1024  # Global, token-based
```

**Phase 2: Per-tool if needed (NOT currently supported by any library)**
```yaml
tool_configs:
  search_pages:
    max_result_length: 2048
  execute_code:
    max_result_length: 512
```

**Implementation signature:**
```python
async def execute_tool(tool_call: dict, max_tool_len: int = 1024) -> str:
    """Execute tool and truncate result to max_tool_len tokens."""
    result = await tools[tool_call["name"]](**tool_call["args"])

    # Tokenize to check length
    result_tokens = tokenizer.encode(str(result), add_special_tokens=False)

    if len(result_tokens) > max_tool_len:
        # Truncate and decode back
        truncated_tokens = result_tokens[:max_tool_len]
        result = tokenizer.decode(truncated_tokens) + "...(truncated)"
        record_metric("tool/truncated_result_rate", 1, Reduce.MEAN)

    return result
```

**Why token-based over character-based:**
- More accurate for budget tracking
- Consistent with max_seq_len
- What actually matters for model context

---

### Q3 Research: Mid-Tool-Call Truncation Detection

**Finding: NO library properly detects mid-tool-call truncation when stop_reason == "length"**

#### VERL Agent Loop: Silent Failure

**Implementation:** `verl/experimental/agent_loop/tool_agent_loop.py:212-258`
```python
async def _handle_generating_state(self, agent_data, sampling_params):
    output = await self.server_manager.generate(...)

    agent_data.response_ids = output.token_ids

    # No finish_reason check here!
    if len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED

    # Attempts to extract tool calls - fails silently on incomplete
    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

    if agent_data.tool_calls:
        return AgentState.PROCESSING_TOOLS
```

**Tool Parser** (`tool_parser.py:82-106`):
```python
async def extract_tool_calls(self, responses_ids):
    text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

    # Missing start/end = no tool calls
    if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
        return text, []  # Silent failure

    matches = self.tool_call_regex.findall(text)
    for match in matches:
        try:
            function_call = json.loads(match)
        except Exception as e:
            logger.error(f"Failed to decode tool call: {e}")  # Logged but ignored

    return content, function_calls
```

**Result:** Incomplete tool calls return empty list, episode continues as if no tool was called.

#### VERL SGLang: Checks finish_reason BUT Before Parsing

**Implementation:** `verl/workers/rollout/sglang_rollout/sglang_rollout.py:920-965`
```python
finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])

if finish_reason_type == FinishReasonTypeEnum.LENGTH:
    # Terminates IMMEDIATELY, doesn't check for tool calls
    _req.add_assistant_message(...)
    break
else:
    # Only checks for tool calls if NOT truncated
    if self._function_call_parser.has_tool_call(content):
        try:
            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
        except JSONDecodeError:
            normed_content = content
            tool_calls = []
```

**Result:** If `finish_reason == "length"`, episode terminates before checking for tool calls.

#### NeMo-RL: No finish_reason Checking

**Implementation:** `RL/nemo_rl/experience/rollouts.py:440-490`
```python
# No stop_reason/finish_reason checking anywhere
env_output = calculate_rewards(active_batch, task_to_env)

# Only checks sequence length
if len(tokenized_obs) + len(generated_ids[i]) + active_input_lengths[i] >= max_seq_len:
    truncation_mask[i] = True
```

**Result:** Relies on environment to handle parsing failures.

#### Verifiers: Will CRASH on Incomplete JSON

**Implementation:** `verifiers/envs/tool_env.py:73-89`
```python
async def env_response(self, messages, state, **kwargs):
    for tool_call in messages[-1]["tool_calls"]:
        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args = json.loads(tool_call.get("function", {}).get("arguments", ""))  # Can crash here!
        tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
```

**Result:** If OpenAI API returns truncated tool call JSON, `json.loads()` raises exception and crashes.

#### Tinker: Best Handling via parse_success Flag

**Implementation:** `tinker-cookbook/recipes/tool_use/search/search_env.py:161-209`
```python
async def step(self, action):
    message, parse_success = self.renderer.parse_response(action)

    if "tool_calls" in message:
        # ... execute tool
    else:
        correct_format = float(parse_success) and float(self.check_format(message["content"]))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer
        # If parse_success = False, format penalty applied
```

**Parser** (`renderers.py:140-161, 412-430`):
```python
def parse_response_for_stop_token(response, tokenizer, stop_token):
    emt_count = response.count(stop_token)
    if emt_count == 0:
        # Missing stop token = parse failure
        return Message(...), False
    elif emt_count == 1:
        return Message(...), True

def parse_response(self, response):
    assistant_message, parse_success = parse_response_for_stop_token(...)
    if not parse_success:
        return assistant_message, False

    match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_message["content"])
    if match:
        tool_calls = self._parse_tool_call(match.group(1))
        if tool_calls is None:
            return assistant_message, False  # Invalid JSON = parse failure
```

**Result:** Detects incomplete responses via missing stop token or invalid JSON, applies format penalty.

#### **Summary Table**

| Library | Checks finish_reason? | Detects incomplete? | Action | Filters from training? |
|---------|----------------------|---------------------|--------|----------------------|
| **VERL (agent_loop)** | No | No | Silent failure, continues | No |
| **VERL (sglang)** | Yes | Partial | Terminates before parsing | No |
| **NeMo-RL** | No | No | Relies on env | No |
| **Verifiers** | Only for prompts | No | **Crashes** | No |
| **Tinker** | No | Yes (parse_success) | Format penalty | No |

#### **Recommendation for Forge:**

**User was right to be skeptical!** Libraries don't treat mid-tool-call truncation specially. But here's why we still should:

**Problem with incomplete tool calls:**
- Incomplete JSON → unparseable → can't execute
- But `response_mask = [1, 1, 1, ...]` → we TRAIN on garbage
- Model learns to produce `<tool_call>{"name": "search",` without closing

**Best practice (combining Tinker's approach with finish_reason check):**
```python
if response.stop_reason == "length":
    record_metric("episode/generation_truncated", 1, Reduce.MEAN)

    # Check if it looks like a tool call was truncated
    has_tool_start = "<tool_call>" in response.text
    has_tool_end = "</tool_call>" in response.text

    if has_tool_start and not has_tool_end:
        # Mid-tool-call truncation
        record_metric("episode/truncated_mid_tool_call", 1, Reduce.MEAN)
        # Mark episode as truncated, let eviction policy handle it
        episode.is_truncated = True
        episode.truncation_reason = "mid_tool_call"
        break  # Terminate episode
```

**Let user decide via config:**
```yaml
grpo:
  eviction_policy:
    evict_truncated: true  # Remove truncated episodes from buffer
    evict_mid_tool_call: true  # More aggressive for tool call corruption
```

---

### Q4 Research: Multiple Tool Calls + Budget Overflow

**Finding: Libraries use ALL-OR-NOTHING (terminate) or TRUNCATE-TO-FIT strategies. None use fair allocation.**

#### VERL: Pre-Truncate Each, Then Terminate if Total Exceeds

**Individual truncation:** `verl/experimental/agent_loop/tool_agent_loop.py:457-464`
```python
# Each tool response truncated BEFORE tokenization
if len(tool_response_text) > self.max_tool_response_length:
    if self.tool_response_truncate_side == "left":
        tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"
    # ... other strategies
```

**Total budget check:** `verl/experimental/agent_loop/tool_agent_loop.py:324-361`
```python
# All tool messages added
agent_data.messages.extend(add_messages)

# Tokenize together
response_ids = tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True)

# Check if total exceeds budget
if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
    return AgentState.TERMINATED  # Episode ends
```

**Multiple tools:** `verl/experimental/agent_loop/tool_agent_loop.py:267-272`
```python
# Parallel execution
tasks = []
for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
    tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))

responses = await asyncio.gather(*tasks)  # All execute in parallel
```

**Result:** Truncate each to `max_tool_response_length`, then if total still exceeds budget, TERMINATE.

#### NeMo-RL: Truncate-to-Fit Remaining Budget

**Implementation:** `RL/nemo_rl/experience/rollouts.py:446-477`
```python
# After tokenizing env observation
if len(tokenized_obs) + len(generated_ids[i]) + active_input_lengths[i] >= max_seq_len:
    # Calculate remaining budget
    tokens_left_for_obs = max_seq_len - (len(generated_ids[i]) + active_input_lengths[i])

    # Truncate to fit
    tokenized_obs = tokenized_obs[:tokens_left_for_obs]
    truncation_mask[i] = True
    sample_truncated[active_indices[i]] = True
```

**Result:** Dynamically truncates observation (which may contain multiple tool results) to fit remaining budget.

#### Tinker: All-or-Nothing Termination

**Implementation:** `tinker-cookbook/recipes/tool_use/search/search_env.py:186-189`
```python
# After adding tool result to messages
next_observation = self.renderer.build_generation_prompt(self.past_messages)

if next_observation.length > self.max_trajectory_tokens:
    return failure_result  # Episode terminates
```

**Multiple tools:** Only processes first tool call (line 179: `message["tool_calls"][0]`)

**Result:** If adding tool result exceeds budget, TERMINATE.

#### Verifiers: No Budget Checking

**Implementation:** `verifiers/envs/tool_env.py:73-89`
```python
# Processes all tool calls sequentially
for tool_call in messages[-1]["tool_calls"]:
    tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
    tool_messages.append(tool_message)  # No length check

return tool_messages, state
```

**Result:** No budget management, relies on OpenAI client.

#### **Summary Table**

| Library | Multiple Tools? | Pre-Truncate Each? | Total Budget Check? | Overflow Strategy | Configurable? |
|---------|----------------|-------------------|---------------------|-------------------|---------------|
| **VERL** | Yes (parallel) | Yes (max_tool_response_length) | Yes | **TERMINATE** | Yes |
| **NeMo-RL** | Single env obs | No | Yes | **TRUNCATE to fit** | Partial |
| **Tinker** | First only | No | Yes | **TERMINATE** | Yes |
| **Verifiers** | Yes (sequential) | No | No | Unknown | No |

#### **Recommendation for Forge:**

**Implement hybrid approach combining best practices:**

```python
# 1. Pre-truncate each tool result (like VERL)
max_tool_len = cfg.max_tool_result_length  # Global: 1024 tokens

for tool_call in tool_calls:
    result = await execute_tool(tool_call)
    result_tokens = tokenizer.encode(str(result), add_special_tokens=False)

    if len(result_tokens) > max_tool_len:
        result_tokens = result_tokens[:max_tool_len]
        result = tokenizer.decode(result_tokens) + "...(truncated)"
        record_metric("tool/individual_truncated", 1, Reduce.MEAN)

    tool_results.append(result)

# 2. Check if total fits in remaining budget
total_tool_tokens = sum(len(tokenizer.encode(r)) for r in tool_results)
remaining_budget = max_seq_len - len(all_tokens)

if total_tool_tokens > remaining_budget:
    # Option A: Terminate (safest, like VERL/Tinker)
    record_metric("episode/tool_overflow_terminated", 1, Reduce.MEAN)
    episode.is_truncated = True
    episode.truncation_reason = "tool_overflow"
    break

    # Option B: Fair allocation (new, user's preference)
    if cfg.truncation.fair_allocate_tools:
        per_tool_budget = remaining_budget // len(tool_results)
        tool_results = [
            tokenizer.decode(tokenizer.encode(r)[:per_tool_budget])
            for r in tool_results
        ]
        record_metric("episode/tool_fair_allocated", 1, Reduce.MEAN)

# 3. Add to messages
for result in tool_results:
    messages.append({"role": "tool", "content": result})
```

**Config:**
```yaml
blackjack_env:
  max_tool_result_length: 1024  # Per-tool pre-truncation

truncation:
  strategy: "terminate"  # or "fair_allocate"
  evict_truncated: true  # Remove from training buffer
```

**Why this is best:**
- Pre-truncation prevents individual tools from being too large
- Total budget check prevents episode overflow
- Configurable strategy (terminate vs fair allocate)
- Clear metrics for debugging
- User controls eviction policy

---

**End of Document**
