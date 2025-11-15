# Example APIs and Patterns

**Goal:** Understand existing patterns for tool calling + multi-turn to design our own clean API for Forge.

**UPDATED:** Now includes deep dive into TRL's low-level implementation of multi-turn with OpenEnv.

---

## 📊 Framework Comparison: Component Coverage Analysis

### Complete Multi-Turn Tool Calling RL Loop Components

Below is the breakdown of ALL components needed for a complete multi-turn tool calling RL system, organized into three phases:

#### **Phase 1: Episode Execution (Rollout)**

1. **Episode Initialization**
   - Create/reset environment
   - Set initial state
   - Build initial prompt

2. **Multi-Turn Generation Loop**
   - Format prompt with conversation history + tool definitions
   - Call generator/LLM
   - Parse response (tool call vs final answer)
   - Execute tools if tool call detected
   - Update conversation history
   - Determine continue vs terminate

3. **Token Collection & Tracking**
   - Store generated tokens per turn
   - Store logprobs per token
   - Track response mask (which tokens are LLM output vs tool results)
   - Concatenate multi-turn tokens OR store per-step

#### **Phase 2: Reward & Advantage**

4. **Reward Computation**
   - Score final outcome
   - Assign rewards (sparse or dense)
   - Handle multi-step credit assignment

5. **Reference Model (for KL penalty)**
   - Get reference logprobs for generated tokens
   - Compute KL divergence

6. **Advantage Computation**
   - Normalize rewards (e.g., group-relative for GRPO)
   - Compute advantages (GAE or other methods)

#### **Phase 3: Training**

7. **Training Data Preparation**
   - Create batches from episodes
   - Apply response masks
   - Format for loss function

8. **Training Step**
   - Forward pass through model
   - Compute loss (GRPO/PPO/Importance Sampling)
   - Backward pass
   - Optimizer step


**Note:** The examples below provide detailed implementations addressing all these components.

---

## Example 1: OpenEnv BlackJack (Forge Integration)

**Location:** `/home/felipemello/forge/OpenEnv/examples/grpo_blackjack/grpo_utils.py`

### Architecture

```
Forge GRPO → OpenEnv HTTP Server → Game Logic
    ↓
Generator (vLLM) → Text Response
    ↓
Parse Action → Execute in Environment
    ↓
Collect Episodes → Train
```

### Key Components

**1. Episode Structure**
```python
@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    game_id: str
    step_in_game: int
    completion: Completion | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None
```

**2. Rollout Loop (play_game)**
```python
async def play_game(game_idx, game_id, server_url, policy, tokenizer, game_log):
    env = OpenSpielEnv(base_url=server_url)
    result = env.reset()

    step_num = 0
    action_history = []
    game_steps = []
    done = False

    while not done and step_num < 10:
        # 1. Format prompt from game state
        prompt = format_prompt(step_num, action_history, tokenizer)

        # 2. Generate response with policy
        responses = await policy.generate.route(prompt)
        response = responses[0]

        # 3. Parse action from text
        action_id = parse_action(response.text, obs.legal_actions)
        action_name = "HIT" if action_id == 0 else "STAND"
        action_history.append((action_id, action_name))

        # 4. Store step data
        game_steps.append({
            "step_num": step_num,
            "prompt": prompt,
            "response": response,
        })

        # 5. Execute action in environment
        result = env.step(OpenSpielAction(action_id=action_id))
        obs = result.observation
        done = result.done
        step_num += 1

    # 6. Get final reward
    final_game_reward = result.reward  # +1, -1, or 0

    # 7. Assign final reward to all steps
    all_step_results = []
    for step_data in game_steps:
        all_step_results.append({
            "game_id": game_id,
            "final_reward": final_game_reward,
            **step_data,
        })

    return all_step_results
```

**3. Prompt Formatting**
```python
def format_prompt(step_num: int, action_history: list, tokenizer) -> str:
    system = "You are an expert BlackJack player. Output only 'HIT' or 'STAND'."

    state_desc = f"=== BlackJack Game (Step {step_num + 1}) ===\n\n"
    if action_history:
        state_desc += "Previous actions:\n"
        for i, (_, name) in enumerate(action_history):
            state_desc += f"  {i + 1}. {name}\n"
        state_desc += "\n"

    state_desc += "What do you do? (Output only 'HIT' or 'STAND')"

    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": state_desc},
    ]

    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
```

**4. Action Parsing**
```python
def parse_action(response_text: str, legal_actions: list[int]) -> int:
    text_lower = response_text.lower().strip()

    if "hit" in text_lower:
        action_id = 0
    elif "stand" in text_lower:
        action_id = 1
    else:
        action_id = 1  # Default: STAND

    # Ensure action is legal
    if action_id not in legal_actions:
        action_id = legal_actions[0]

    return action_id
```

**5. Episode Creation (in continuous_rollouts)**
```python
# Play multiple games
for game_idx in range(group_size):
    game_id = str(uuid.uuid4())[:8]
    step_results = await play_game(
        game_idx, game_id, server_url, policy, tokenizer, game_log
    )
    all_step_results.extend(step_results)

# Create episodes
episodes = []
for step_result in all_step_results:
    episode = Episode(
        episode_id=str(uuid.uuid4()),
        pad_id=pad_id,
        request_len=max_req_tokens,
        response_len=max_res_tokens,
        game_id=step_result["game_id"],
        step_in_game=step_result["step_num"],
        completion=step_result["response"],
    )

    # Evaluate reward (with optional shaping)
    episode.reward = await reward_actor.evaluate_response.route(
        prompt=step_result["prompt"],
        response=step_result["response"].text,
        game_reward=step_result["final_reward"],
    )

    episodes.append(episode)
```

**6. Integration with Forge GRPO**
```python
# Get reference logprobs
ref_logprobs = await ref_model.forward.route(
    input_ids, max_req_tokens, return_logprobs=True
)
for i, episode in enumerate(episodes):
    episode.ref_logprobs = ref_logprobs[i]

# Compute advantages (group-relative)
advantages = await compute_advantages.compute.call_one(episodes)
for episode, advantage in zip(episodes, advantages):
    episode.advantage = advantage
    await replay_buffer.add.call_one(episode)
```

### Key Insights

✅ **Text-based action parsing works**: No need for structured tool calling
✅ **Multi-step = multiple episodes**: One episode per step, shared final reward
✅ **Action history in prompt**: Previous actions included in context
✅ **Simple prompt formatting**: Chat template with system + user message
✅ **Async environment calls**: `await env.step()` wraps sync OpenEnv

### Episode Organization: Per-Step Strategy

**BlackJack uses Strategy A:** Each step = separate Episode

```python
# Game with 3 steps produces 3 Episodes:
Episode(game_id="abc123", step_in_game=0, reward=1.0)  # Step 1
Episode(game_id="abc123", step_in_game=1, reward=1.0)  # Step 2
Episode(game_id="abc123", step_in_game=2, reward=1.0)  # Final step
```

**Credit Assignment:**
- Final game reward (`+1`, `-1`, or `0`) is assigned to ALL steps
- Each step trains independently
- No gradient flow between steps

**Why this works:**
- Simpler implementation
- Each Episode is self-contained
- No need for response masks (each completion is pure LLM output)
- Matches existing Forge GRPO pattern

---

## Example 2: Tinker-Cookbook Search Tool (Multi-turn + Tools)

**Location:** `/home/felipemello/forge/tinker-cookbook/tinker_cookbook/recipes/tool_use/search/`

### Architecture

```
RL Training Loop → SearchEnv → ChromaDB Tool
    ↓
Model Generate → Parse Tool Calls
    ↓
Execute Tools → Return Results
    ↓
Continue or Terminate → Reward
```

### Key Components

**1. Tool Interface**
```python
class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Returns tool definitions"""
        ...

    @abstractmethod
    async def invoke(self, tool_call: ToolCall) -> list[Message]:
        """Executes tool and returns results"""
        ...
```

**2. Tool Schema**
```python
{
    "name": "search",
    "title": "Wikipedia search",
    "description": "Searches Wikipedia for relevant information...",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of fully-formed semantic queries...",
            }
        },
        "required": ["query_list"],
    },
    "outputSchema": {
        "type": "string",
        "description": "The search results in JSON format",
    },
}
```

**3. System Prompt with Tool Instructions**
```python
SEARCH_TOOL_SYSTEM_PROMPT = """
You are an expert assistant who solves tasks using a Wikipedia search tool.
Tool calling. Execute the tool by wrapping calls in <function_call>...</function_call>

The search tool you are given has the following schema:
{tool_schema}

Here are instructions for how to solve a problem:
1. Think step by step before calling the tool
2. Call the tool with the queries you have decided on
3. Think step by step again after you receive the result
4. If you have the information you need, provide your answer
5. Otherwise, come up with new queries
6. Include your final answer after the "Answer:" prefix

Example:
Question: "Between 2020 and 2025, which year did NYC see most growth?"
1. Think: I need to search for NYC population data 2020-2025
2. Tool call: <function_call>{"name": "search", "args": {"query_list": ["NYC population 2020-2025"]}}</function_call>
3. Think: Based on results, 2024 had most growth. Now check San Francisco...
4. Tool call: <function_call>{"name": "search", "args": {"query_list": ["SF population 2024"]}}</function_call>
5. Answer: NYC grew most in 2024, SF changed by XXXX.
"""
```

**4. Environment Step Function**
```python
class SearchEnv(ProblemEnv):
    async def step(self, action: Action) -> StepResult:
        # Parse response (text or tool call)
        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)

        # If tool call
        if "tool_calls" in message:
            if message["tool_calls"][0]["name"] == "search":
                self.current_num_calls += 1

                # Check max calls limit
                if self.current_num_calls > self.max_num_calls:
                    return StepResult(
                        reward=0.0,
                        episode_done=True,
                        next_observation=ModelInput.empty(),
                    )

                # Execute tool
                tool_return_message = await self.call_search_tool(
                    message["tool_calls"][0]
                )
                self.past_messages.extend(tool_return_message)

                # Continue episode with tool results
                next_observation = self.renderer.build_generation_prompt(
                    self.past_messages
                )
                return StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=next_observation,
                )

        # If final answer (no tool call)
        else:
            correct_format = self.check_format(message["content"])
            correct_answer = self.check_answer(message["content"])
            total_reward = format_coef * (correct_format - 1) + correct_answer

            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=ModelInput.empty(),
                metrics={"format": correct_format, "correct": correct_answer},
            )
```

**5. Message/History Management**
```python
class SearchEnv:
    def __init__(self, ...):
        self.past_messages: list[Message] = []
        self.convo_prefix: list[Message] = convo_prefix or []

    async def initial_observation(self):
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo)

    async def step(self, action):
        message = parse_response(action)
        self.past_messages.append(message)  # Add assistant message

        if is_tool_call(message):
            tool_result = await execute_tool(...)
            self.past_messages.extend(tool_result)  # Add tool result

            # Build next prompt with full history
            next_prompt = self.renderer.build_generation_prompt(
                self.past_messages
            )
            return StepResult(next_observation=next_prompt, ...)
```

**6. Renderer Pattern (Message → Prompt)**
```python
class Renderer:
    def build_generation_prompt(self, messages: list[Message]) -> ModelInput:
        """Convert message history to tokenized prompt"""
        # Format: [system, user, assistant, tool, user, assistant, ...]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return ModelInput(prompt=prompt_text, tokens=...)

    def parse_response(self, action: Action) -> tuple[Message, bool]:
        """Parse model output to Message (text or tool call)"""
        # Check for <function_call>...</function_call>
        if "<function_call>" in action.text:
            tool_call = extract_tool_call(action.text)
            return Message(
                role="assistant",
                tool_calls=[tool_call]
            ), True
        else:
            return Message(
                role="assistant",
                content=action.text
            ), True
```

**7. Tool Execution**
```python
async def call_search_tool(self, tool_call: ToolCall) -> list[Message]:
    # Validate tool call
    if tool_call["name"] != "search":
        return [Message(role="tool", content="Error: invalid tool")]

    # Execute tool (async)
    query_list = tool_call["args"]["query_list"]
    results = await self.chroma_tool_client.invoke(query_list)

    # Format results as tool message
    message_content = ""
    for query, documents in zip(query_list, results["documents"]):
        message_content += f"Query: {query}\n"
        for i, doc in enumerate(documents):
            message_content += f"Document {i + 1}:\n{doc}\n"

    return [Message(role="tool", content=message_content)]
```

### Key Insights

✅ **Tool calls wrapped in special tags**: `<function_call>...</function_call>`
✅ **Message history tracked explicitly**: `self.past_messages` grows each turn
✅ **Renderer abstracts prompt building**: Clean separation of concerns
✅ **Environment controls episode flow**: Decides when to continue vs terminate
✅ **Sparse rewards at end**: Intermediate tool calls get reward=0
✅ **Tool results added to history**: Next prompt includes tool outputs

### Response Masking Implementation

**File:** `tinker_cookbook/rl/data_processing.py:160-168`

**How Tinker builds the mask during trajectory→training data conversion:**

```python
# For each transition (observation → action):
def trajectory_to_data(traj: Trajectory, traj_advantage: float):
    for transition in traj.transitions:
        ob = transition.ob          # Environment observation (includes tool results)
        ac = transition.ac          # LLM-generated action

        delta_ob_len = len(observation_tokens)  # Tool results, env state
        ac_len = len(action_tokens)             # LLM output

        # Build mask: 0 for observations, 1 for actions
        SequenceAccumulator.mask.extend(
            [0.0] * delta_ob_len +  # DON'T train on observations
            [1.0] * ac_len           # TRAIN on LLM actions
        )

        # Also accumulate advantages (only for action tokens)
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len +           # No advantage for observations
            [traj_advantage] * ac_len       # Advantage for actions
        )
```

**Final training data:**
```python
tinker.Datum(
    model_input=input_tokens,
    loss_fn_inputs={
        "target_tokens": targets,
        "logprobs": sampled_logprobs,
        "advantages": advantages,      # Per-token advantages
        "mask": mask,                  # Per-token mask
    }
)
```

**Key points:**
- Per-token granularity: Each token has its own mask value
- Applied during loss computation via element-wise multiplication
- Observations (tool results) get `mask=0.0` → no gradient
- Actions (LLM output) get `mask=1.0` → full gradient

---

### Tinker-Cookbook Deep Dive: Low-Level Implementation Details

**NOW LET'S LOOK AT THE ACTUAL CODE** to see how Tinker-Cookbook implements multi-turn tool calling.

#### **1. Renderer: How Prompts Are Actually Built** (`renderers.py`)

The Renderer is KEY to understanding Tinker. Here's how it ACTUALLY works:

**Qwen3Renderer Example** (with tool calling support):

```python
class Qwen3Renderer(Renderer):
    def _render_message(self, idx: int, message: Message) -> tuple[list[int], list[int], list[int]]:
        """Render a message into three parts: observation, action, action_tail."""
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"

        # Handle tool calls
        ac_content = message["content"]
        if "tool_calls" in message:
            # Add tool call XML to content
            ac_content += "\n".join(
                [
                    f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        ac_content += "<|im_end|>"

        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),  # Observation
            self.tokenizer.encode(ac_content, add_special_tokens=False),  # Action
            self.tokenizer.encode("", add_special_tokens=False),  # Action tail (empty for Qwen)
        )

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

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse model output back to Message."""
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        # Parse tool calls from <tool_call>...</tool_call> tags
        match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_message["content"], re.DOTALL)
        if match:
            tool_calls = self._parse_tool_call(match.group(1))
            if tool_calls is None:
                return assistant_message, False
            else:
                assistant_message["tool_calls"] = tool_calls
                return assistant_message, True
        return assistant_message, True

    def _parse_tool_call(self, tool_call_str: str) -> list[ToolCall] | None:
        """Parse tool call JSON."""
        try:
            tool_call = json.loads(tool_call_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(tool_call, dict):
            return None
        if (
            "name" not in tool_call
            or "args" not in tool_call
            or not isinstance(tool_call["name"], str)
            or not isinstance(tool_call["args"], dict)
        ):
            return None

        return [ToolCall(**tool_call)]
```

**Key insights:**
- Renderer has THREE methods: `_render_message()`, `build_generation_prompt()`, `parse_response()`
- Tool calls are embedded as XML: `<tool_call>{"name": "search", "args": {...}}</tool_call>`
- Each message is split into: observation (prompt part) + action (completion part) + action_tail
- This allows separate training masks for supervised learning

#### **2. Environment: The Multi-Turn Loop** (`search_env.py`)

The SearchEnv shows how multi-turn actually works:

```python
class SearchEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: list[str],
        chroma_tool_client: ChromaToolClient,
        renderer: renderers.Renderer,
        max_num_calls: int = 4,
    ):
        self.problem = problem
        self.answer = answer
        self.chroma_tool_client = chroma_tool_client
        self.renderer = renderer
        self.past_messages: list[renderers.Message] = []
        self.current_num_calls = 0
        self.max_num_calls = max_num_calls

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Start episode with user question."""
        convo = [
            {"role": "system", "content": SEARCH_TOOL_SYSTEM_PROMPT},  # Tool instructions
            {"role": "user", "content": self.problem},
        ]
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Execute one step: either tool call or final answer."""
        # Parse model output
        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)

        # Check if tool call
        if "tool_calls" in message:
            if message["tool_calls"][0]["name"] == "search":
                self.current_num_calls += 1

                # Check max calls limit
                if self.current_num_calls > self.max_num_calls:
                    return StepResult(
                        reward=0.0,
                        episode_done=True,
                        next_observation=tinker.ModelInput.empty(),
                    )

                # Execute tool
                try:
                    tool_return_message = await self.call_search_tool(message["tool_calls"][0])
                    self.past_messages.extend(tool_return_message)  # Add tool result
                except Exception as e:
                    logger.error(f"Error calling search tool: {repr(e)}")
                    return StepResult(reward=0.0, episode_done=True, next_observation=tinker.ModelInput.empty())

                # Continue episode with tool results
                next_observation = self.renderer.build_generation_prompt(self.past_messages)
                return StepResult(
                    reward=0.0,  # Intermediate reward
                    episode_done=False,  # Continue
                    next_observation=next_observation,
                )
            else:
                # Invalid tool name
                return StepResult(reward=0.0, episode_done=True, next_observation=tinker.ModelInput.empty())
        else:
            # Final answer (no tool call)
            correct_format = float(parse_success) and float(self.check_format(message["content"]))
            correct_answer = float(self.check_answer(message["content"]))
            total_reward = self.format_coef * (correct_format - 1) + correct_answer
            return StepResult(
                reward=total_reward,  # Final reward
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                metrics={
                    "format": correct_format,
                    "correct": correct_answer,
                },
            )

    async def call_search_tool(self, tool_call: renderers.ToolCall) -> list[renderers.Message]:
        """Execute search tool and return result message."""
        async with _CONNECTION_SEMAPHORE:
            return await self.chroma_tool_client.invoke(tool_call)
```

**Key insights:**
- Environment maintains `self.past_messages` (full conversation history)
- `step()` returns different results based on tool call vs final answer
- Tool calls → `episode_done=False` (continue episode)
- Final answer → `episode_done=True` (end episode)
- Intermediate tool calls get `reward=0.0`, final answer gets scored

#### **3. Rollout Loop** (`rollouts.py:16-34`)

The actual rollout execution is SIMPLE:

```python
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    """Run one episode from start to finish."""
    transitions = []
    ob, stop_condition = await env.initial_observation()

    while True:
        # 1. Generate action from policy
        ac_with_logprobs = await policy(ob, stop_condition)

        # 2. Execute action in environment
        step_result = await env.step(ac_with_logprobs.tokens)

        # 3. Store transition
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)

        # 4. Update observation
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition

        # 5. Check if done
        if step_result.episode_done:
            break

    return Trajectory(transitions=transitions, final_ob=ob)
```

**Key insights:**
- Simple while loop: generate → step → store
- Environment (`env.step()`) handles ALL the complexity
- Policy is just a callable: `policy(observation) → action`
- Each step creates a Transition (observation, action, reward)

#### **4. Training Integration** (`train.py`)

How rollouts feed into training:

```python
# From train.py:138-193
async def train_step(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: Literal["importance_sampling", "ppo"],
) -> List[torch.Tensor]:
    """Train the model on collected trajectories."""
    batches_md = split_list(data_D, min(num_substeps, len(data_D)))
    training_logprobs_D: list[torch.Tensor] = []

    for batch_d in batches_md:
        training_logprobs = await forward_backward(training_client, batch_d, loss_fn)
        training_logprobs_D.extend(training_logprobs)
        await optim_step(training_client, learning_rate)

    return training_logprobs_D
```

**The full RL loop** (from `train.main()`):

```python
while True:
    # 1. Collect rollouts
    traj_groups = []
    for _ in range(groups_per_batch):
        traj_group = await do_group_rollout(env_group_builder, policy)
        traj_groups.append(traj_group)

    # 2. Process trajectories → training data
    advantages_G = compute_advantages(traj_groups)
    data_D, metadata_D = assemble_training_data(traj_groups, advantages_G)

    # 3. Train on data
    await train_step(data_D, training_client, learning_rate, num_substeps, loss_fn)

    # 4. Evaluate
    if eval_every > 0 and step % eval_every == 0:
        for evaluator in evaluators:
            metrics = await evaluator.evaluate(sampling_client)
```

**Key insights:**
- Rollouts → Trajectories → Advantages → Training Data → Train
- Advantages computed from trajectory rewards (GAE or similar)
- Training data includes: model_input, targets, advantages (for loss weighting)
- Uses Tinker's TrainingClient (abstracts distributed training)

#### **5. From Transitions to Training Examples**

How multi-turn episodes become training examples:

```python
# Each Transition has:
# - ob: tinker.ModelInput (the prompt)
# - ac: TokensWithLogprobs (the generated tokens)
# - reward: float
# - episode_done: bool

# For multi-turn:
# Transition 1: ob=[system, user], ac=[<tool_call>search(...)</tool_call>], reward=0.0
# Transition 2: ob=[system, user, assistant, tool], ac=[Answer: X], reward=1.0

# These become training examples:
# Example 1: input=[system, user], target=[<tool_call>search(...)</tool_call>], advantage=A1
# Example 2: input=[system, user, assistant, tool], target=[Answer: X], advantage=A2
```

**The advantage computation ensures:**
- Later steps (with actual rewards) get higher advantage
- Early steps (reward=0) get credit via bootstrapping
- Model learns the full multi-turn policy

---

## Key Design Decisions

1. **Text Parsing vs Native Tool Calling?** - BlackJack uses text parsing, Tinker uses tags. **Rec:** Start with text parsing (simpler).

2. **Episode Granularity?** - BlackJack: One episode per step. Tinker: One episode for full conversation. **Rec:** One episode per step (matches GRPO).

3. **Message History Management?** - BlackJack: Rebuilt in prompt. Tinker: Explicit list. **Rec:** Explicit list (clearer, easier to debug).

4. **Reward Assignment?** - BlackJack: Final reward to all steps. Tinker: Sparse reward at end. **Rec:** Final reward to all steps (simpler for GRPO).

5. **Environment Integration?** - BlackJack: Custom loop. Tinker: Environment manages flow. **Rec:** Custom loop (more control, matches BlackJack).

---

## Example 3: VERL Multi-turn + Tool Calling (SGLang)

**Location:** `/home/felipemello/forge/verl/`

VERL provides a production-ready implementation of multi-turn tool calling with SGLang backend. This is highly relevant as a reference for Forge.

### Architecture

```
Ray Trainer → SGLangRollout → SGLang Engine
    ↓
Agent Loop (State Machine) → Tool Execution
    ↓
AsyncRolloutRequest → Message History → Episodes
```

### Key Components

**1. State Machine Pattern**

```python
class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    INTERACTING = "interacting"
    TERMINATED = "terminated"

# Main loop
while state != AgentState.TERMINATED:
    if state == AgentState.PENDING:
        state = await _handle_pending_state(agent_data, sampling_params)
    elif state == AgentState.GENERATING:
        state = await _handle_generating_state(agent_data, sampling_params)
    elif state == AgentState.PROCESSING_TOOLS:
        state = await _handle_processing_tools_state(agent_data)
    elif state == AgentState.INTERACTING:
        state = await _handle_interacting_state(agent_data)
```

**2. Tool Definition (YAML Config)**

```yaml
# gsm8k_tool_config.yaml
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "Calculate reward for GSM8K answer"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer"
          required: ["answer"]
```

**3. Tool Base Class**

```python
class BaseTool:
    async def create(self, instance_id: str = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create tool instance for a trajectory"""
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict) -> tuple[ToolResponse, float, dict]:
        """Execute tool, return (response, step_reward, metrics)"""
        return ToolResponse(text="result"), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward for this instance"""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Cleanup tool instance"""
        pass
```

**4. Multi-turn Rollout Flow**

```python
async def _async_rollout_a_request(self, req: AsyncRolloutRequest, **kwargs):
    current_turns = 0

    while current_turns < max_assistant_turns:
        # Generate model response
        output = await self._engine.async_generate(
            input_ids=req.get_generation_prompt_ids(tokenizer),
            sampling_params=sampling_params,
            return_logprob=True
        )

        # Parse response for tool calls
        if self._function_call_parser.has_tool_call(output["text"]):
            # Parse tool calls
            _, tool_calls = self._function_call_parser.parse_non_stream(output["text"])

            # Execute tools in parallel
            tool_results = await asyncio.gather(*[
                self._tool_map[tc.name].execute(req.request_id, tc.arguments)
                for tc in tool_calls
            ])

            # Add tool responses to message history
            req.add_tool_response_messages(tokenizer, [resp for resp, _, _ in tool_results])

            # Continue generation
            current_turns += 1
        else:
            # No tool call, terminate or continue with user interaction
            break

    # Calculate final rewards from all tools
    tool_rewards = await asyncio.gather(*[
        tool.calc_reward(req.request_id) for tool in tools_used
    ])

    req.finalize(tokenizer, tool_rewards, finish_reason)
    return req
```

**5. Message History Management**

```python
class AsyncRolloutRequest:
    messages: list[Message]  # Full conversation history

    def add_assistant_message(self, tokenizer, content: str, tool_calls=None):
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        self.messages.append(msg)
        # Update token IDs
        new_ids = tokenizer.apply_chat_template([msg], add_generation_prompt=False)
        self.response_ids = torch.cat([self.response_ids, new_ids])
        self.response_mask += [1] * len(new_ids)  # LLM-generated tokens

    def add_tool_response_messages(self, tokenizer, tool_responses: list[ToolResponse]):
        for tool_resp in tool_responses:
            msg = Message(role="tool", content=tool_resp.text)
            self.messages.append(msg)
            # Tokenize tool response
            new_ids = tokenizer.apply_chat_template([msg], add_generation_prompt=True)
            self.prompt_ids = torch.cat([self.prompt_ids, new_ids])
            self.response_mask += [0] * len(new_ids)  # Not LLM-generated
```

**6. Response Mask Pattern**

```python
# For multi-turn with tools:
# responses:     |<- LLM gen ->|<- tool_calls ->|<- LLM gen ->|<- padding ->|
# response_mask: | 1, 1, 1, 1  | 0, 0, 0, 0     | 1, 1, 1, 1  | 0, 0, 0, 0  |
#
# 1 = LLM-generated tokens (train on these)
# 0 = Tool results, padding (don't train on these)

batch = {
    "prompts": prompt_ids,           # [batch, prompt_len]
    "responses": response_ids,        # [batch, response_len]
    "response_mask": response_mask,   # [batch, response_len] - key for multi-turn!
    "input_ids": input_ids,           # [batch, prompt_len + response_len]
    "attention_mask": attention_mask, # [batch, prompt_len + response_len]
    "position_ids": position_ids,     # [batch, prompt_len + response_len]
}
```

**7. Configuration**

```yaml
# Config file
multi_turn:
  enable: True
  max_assistant_turns: 5
  max_user_turns: 3
  max_parallel_calls: 5
  tool_config_path: "config/tool_config/gsm8k_tool_config.yaml"
  format: "hermes"  # or "gpt-oss"
  max_tool_response_length: 2048
  tool_response_truncate_side: "left"
```

### Key Insights

✅ **State machine is explicit**: Clear transition logic between PENDING → GENERATING → TOOL_CALLING → GENERATING
✅ **Tools are async**: Parallel execution with `asyncio.gather()`
✅ **Two-phase rewards**: Step rewards during execution + final reward at end
✅ **Response mask critical**: Distinguishes LLM tokens (train) from tool results (don't train)
✅ **Message history explicit**: Full OpenAI-style conversation in `messages` list
✅ **Tool lifecycle**: create() → execute() (multiple times) → calc_reward() → release()
✅ **Config-driven tools**: Tools loaded from YAML, making it easy to swap
✅ **SGLang integration**: Uses SGLang's native function calling parser

### Response Mask Construction (Concatenated Episodes)

**VERL uses Strategy B:** All turns concatenated into ONE Episode with response_mask

**How mask is built during generation:**
```python
# From tool_agent_loop.py:1370-1470

# When LLM generates (GENERATING state):
agent_data.response_ids = output.token_ids
agent_data.prompt_ids += agent_data.response_ids      # CONCATENATE
agent_data.response_mask += [1] * len(agent_data.response_ids)  # TRAIN

# When tool executes (PROCESSING_TOOLS state):
response_ids = tokenizer.apply_chat_template(tool_messages, ...)
agent_data.prompt_ids += response_ids                 # CONCATENATE
agent_data.response_mask += [0] * len(response_ids)  # DON'T TRAIN
```

**Example multi-turn sequence:**
```python
# prompt_ids:     [sys, user] + [llm_gen_1] + [tool_result_1] + [llm_gen_2]
# response_mask:  [0,   0   ] + [1,1,1,1   ] + [0,0,0,0      ] + [1,1,1,1  ]
#
# 1 = Train on these (LLM output)
# 0 = Ignore these (prompts, tool results)
```

### Loss Computation with Response Mask

**File:** `verl/trainer/ppo/core_algos.py:787-808`

**How VERL applies the mask during training:**

```python
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Args:
        loss_mat: (batch, seq_len) - per-token loss
        loss_mask: (batch, seq_len) - 1=train, 0=ignore
    """
    if loss_agg_mode == "token-mean":
        # Average over all unmasked tokens
        loss = masked_mean(loss_mat, loss_mask)

    elif loss_agg_mode == "seq-mean-token-mean":
        # Average tokens per sequence, then average sequences
        seq_token_count = torch.sum(loss_mask, dim=-1)  # Count per seq
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_count + 1e-8)
        loss = seq_losses.mean()

    return loss
```

**Usage in policy loss:**
```python
# Compute per-token policy gradient loss
pg_losses = -advantages * log_prob  # (batch, seq_len)

# Apply mask and aggregate
pg_loss = agg_loss(
    loss_mat=pg_losses,
    loss_mask=response_mask,  # Zeros out tool result tokens
    loss_agg_mode="token-mean"
)
```

**Key mechanism:**
1. Element-wise multiplication: `loss_mat * loss_mask` zeros out masked tokens
2. Only unmasked tokens contribute to loss
3. Gradient flows only through LLM-generated tokens

---

### VERL Deep Dive: Low-Level Implementation Details

**NOW LET'S LOOK AT THE ACTUAL CODE** to understand how VERL really works under the hood.

#### **State Machine Handlers** (`verl/experimental/agent_loop/tool_agent_loop.py:184-428`)

The state machine handlers are where the magic happens. Here's the ACTUAL implementation:

**1. PENDING → GENERATING: Prepare Prompt with Tools**

```python
async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict) -> AgentState:
    """Handle the pending state: prepare the prompt and start generation."""
    # Apply chat template with tools
    if self.processor is not None:
        # For multimodal models
        raw_prompt = await self.loop.run_in_executor(
            None,
            lambda: self.processor.apply_chat_template(
                agent_data.messages,
                tools=self.tool_schemas,  # <-- Tools passed here!
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
        agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
    else:
        # For text-only models
        agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                agent_data.messages,
                tools=self.tool_schemas,  # <-- Tools passed to tokenizer
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
    return AgentState.GENERATING
```

**Key insight:** VERL uses the tokenizer/processor's `apply_chat_template()` with `tools=` parameter. The formatting happens inside the tokenizer (model-specific).

**2. GENERATING: Call Model and Parse Tool Calls**

```python
async def _handle_generating_state(
    self, agent_data: AgentData, sampling_params: dict, ignore_termination: bool = False
) -> AgentState:
    """Handle the generating state: generate model response and check for tool calls."""

    # Generate using SGLang server
    with simple_timer("generate_sequences", agent_data.metrics):
        output = await self.server_manager.generate(
            request_id=agent_data.request_id,
            prompt_ids=agent_data.prompt_ids,
            sampling_params=sampling_params,
            image_data=agent_data.image_data,
        )

    # Track turn count
    agent_data.assistant_turns += 1

    # Accumulate response tokens
    agent_data.response_ids = output.token_ids
    agent_data.prompt_ids += agent_data.response_ids  # <-- Concatenate!
    agent_data.response_mask += [1] * len(agent_data.response_ids)  # <-- Mark as LLM output

    if output.log_probs:
        agent_data.response_logprobs += output.log_probs

    # Check termination conditions
    if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED
    if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
        return AgentState.TERMINATED

    # Extract tool calls using parser
    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

    # Determine next state
    if agent_data.tool_calls:
        return AgentState.PROCESSING_TOOLS  # <-- Has tool calls
    elif self.interaction_config_file:
        return AgentState.INTERACTING  # <-- Need user input
    else:
        return AgentState.TERMINATED  # <-- Done
```

**Key insights:**
- Response tokens are CONCATENATED to prompt_ids: `agent_data.prompt_ids += agent_data.response_ids`
- Response mask marks LLM output as `1` (train on these)
- Tool parser extracts tool calls from the generated token IDs

**3. PROCESSING_TOOLS: Execute Tools in Parallel**

```python
async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
    """Handle the processing tools state: execute tool calls and prepare tool responses."""
    add_messages: list[dict[str, Any]] = []
    new_images_this_turn: list[Any] = []

    # Create tasks for parallel execution
    tasks = []
    tool_call_names = []
    for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
        tool_call_names.append(tool_call.name)

    # Execute ALL tools in parallel
    with simple_timer("tool_calls", agent_data.metrics):
        responses = await asyncio.gather(*tasks)  # <-- Parallel execution!

    # Process tool responses
    for tool_response, tool_reward, _ in responses:
        # Create message from tool response
        if tool_response.image or tool_response.video:
            # Multimodal content
            content = []
            if tool_response.image:
                content.append({"type": "image"})
                new_images_this_turn.append(tool_response.image)
            if tool_response.text:
                content.append({"type": "text", "text": tool_response.text})
            message = {"role": "tool", "content": content}
        else:
            # Text-only content
            message = {"role": "tool", "content": tool_response.text or ""}

        add_messages.append(message)

        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)

    agent_data.messages.extend(add_messages)

    # Tokenize tool responses
    if self.processor is not None:
        raw_tool_response = await self.loop.run_in_executor(
            None,
            lambda: self.processor.apply_chat_template(
                add_messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        model_inputs = self.processor(text=[raw_tool_response], images=new_images_this_turn, return_tensors="pt")
        response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
    else:
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
        )
        response_ids = response_ids[len(self.system_prompt) :]

    # Accumulate tool result tokens
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)  # <-- Mark as NOT LLM output (don't train)
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(response_ids)

    agent_data.user_turns += 1
    return AgentState.GENERATING  # <-- Continue generation
```

**Key insights:**
- Tools execute in parallel using `asyncio.gather(*tasks)`
- Tool results are tokenized and added to prompt_ids
- Response mask = `[0]` for tool results (DON'T train on these)
- After tools, loop back to GENERATING state

**4. Tool Execution** (`_call_tool` method)

```python
async def _call_tool(
    self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
) -> tuple[ToolResponse, float, dict]:
    """Call tool and return tool response."""
    tool, instance_id = None, None
    try:
        # Parse tool call
        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments)

        # Get tool from map
        tool = self.tools[tool_name]
        kwargs = tools_kwargs.get(tool_name, {})

        # Tool lifecycle: create → execute → release
        instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
        tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)

    except Exception as e:
        logger.warning(f"Error when executing tool: {e}")
        return (
            ToolResponse(text=f"Error when executing tool: {e}"),
            0.0,
            {},
        )
    finally:
        if tool and instance_id:
            await tool.release(instance_id)

    # Truncate long responses
    tool_response_text = tool_execution_response.text
    if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
        if self.tool_response_truncate_side == "left":
            tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
        elif self.tool_response_truncate_side == "right":
            tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
        else:
            length = self.max_tool_response_length // 2
            tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

    return ToolResponse(text=tool_response_text, image=tool_execution_response.image), tool_reward, res
```

**Key insights:**
- Tool lifecycle: `create()` → `execute()` → `release()`
- Tool responses can be truncated
- Each tool can return a reward
- Error handling with try/finally to ensure cleanup

#### **Response Mask Pattern**

The response mask is CRITICAL for multi-turn training:

```python
# Example multi-turn sequence:
# prompt_ids:     [system, user, <tool_def>] + [llm_gen_1] + [tool_result_1] + [llm_gen_2] + ...
# response_mask:  [       0    ,    0      ] + [    1     ] + [      0      ] + [    1     ] + ...
#
# 1 = Train on these tokens (LLM output)
# 0 = Don't train on these (prompts, tool results)
```

In VERL, this is built incrementally:
- `agent_data.response_mask += [1] * len(agent_data.response_ids)` when LLM generates
- `agent_data.response_mask += [0] * len(response_ids)` when tool responds

#### **Generator Integration** (How SGLang is called)

The `server_manager.generate()` call abstracts the SGLang engine:

```python
# From sglang_rollout.py:
output = await self.server_manager.generate(
    request_id=agent_data.request_id,
    prompt_ids=agent_data.prompt_ids,
    sampling_params=sampling_params,
    image_data=agent_data.image_data,
)
# Returns: output.token_ids, output.log_probs
```

This uses SGLang's async engine internally, which handles:
- Native function calling (if model supports it)
- Tool call parsing (using FunctionCallParser)
- Structured output

---

## Example 4: NeMo-RL Async vLLM with Pipelined Tool Calling

**Location:** `/home/felipemello/forge/RL/`

NeMo-RL implements async vLLM engines with **sample-level concurrency** that enables pipelined tool calling. When one sample is waiting for a tool response, other samples continue generating without blocking.

### Architecture

```
Async GRPO Loop → run_async_multi_turn_rollout() → Per-Sample Async Tasks
    ↓
Sample 1: [Turn 1 Gen] → [Tool Call] → [Waiting...] → [Turn 2 Gen] → ...
Sample 2: [Turn 1 Gen] → [Turn 2 Gen] → [Tool Call] → [Waiting...] → ...
Sample 3: [Turn 1 Gen] → [Done]
    ↓
All run concurrently via asyncio.gather()
    ↓
vLLM AsyncLLM Engine handles multiple in-flight requests
```

### Key Configuration

**1. Enable Async vLLM Engine** (`grpo_math_1B.yaml:218`)
```yaml
policy:
  generation:
    backend: "vllm"
    vllm_cfg:
      async_engine: true  # Enable async mode for pipelining
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
```

**2. Worker Selection** (`vllm_generation.py:155-160`)
```python
if self.cfg["vllm_cfg"]["async_engine"]:
    worker_cls = "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
else:
    worker_cls = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
```

### Sample-Level Concurrency Pattern

**1. Top-level Async Rollout** (`rollouts.py:780-936`)
```python
def run_async_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
    """Run multi-turn rollouts with sample-level processing.

    Each sample in the batch proceeds through its interaction independently.
    Async generation is used internally when available.
    """

    async def _async_rollout_implementation():
        batch_size = len(input_batch["message_log"])

        # Prepare initial states for each sample
        sample_initial_states = [...]

        # Create tasks for all samples
        sample_tasks = [
            run_single_sample_with_error_handling(i, sample_state)
            for i, sample_state in enumerate(sample_initial_states)
        ]

        # Execute ALL sample rollouts CONCURRENTLY
        sample_results = await asyncio.gather(*sample_tasks, return_exceptions=False)

        return final_batch, rollout_metrics

    return asyncio.run(_async_rollout_implementation())
```

**Key Insight**: Each sample gets its own async task that runs independently. This is the foundation of pipelining.

**2. Per-Sample Multi-turn Loop** (`rollouts.py:611-777`)
```python
async def run_sample_multi_turn_rollout(
    sample_idx: int,
    initial_sample_state: dict,
    policy_generation: GenerationInterface,
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[dict, dict[str, Any]]:
    """Run a multi-turn rollout for a single sample.

    This function manages the complete lifecycle of one sample's interaction.
    """
    current_message_log = copy.deepcopy(initial_sample_state["message_log"])

    for turn in range(max_rollout_turns):
        if terminated or truncated:
            break

        # 1. Generate response using async generation
        (
            updated_message_log,
            generated_tokens,
            input_lengths,
            gen_metrics,
        ) = await async_generate_response_for_sample_turn(
            policy_generation,
            current_message_log,
            current_stop_strings,
            tokenizer,
            max_seq_len,
            greedy=greedy,
        )
        current_message_log = updated_message_log

        # 2. Execute tool call in environment
        sample_batch = BatchedDataDict[DatumSpec]({
            "message_log": [current_message_log],
            "extra_env_info": [current_extra_env_info],
            "task_name": [task_name],
        })

        env_output = calculate_rewards(sample_batch, task_to_env)

        # 3. Add environment response to message log
        env_message = {
            "role": env_output.observations[0]["role"],
            "content": env_obs_content,
            "token_ids": tokenized_obs,
        }
        current_message_log.append(env_message)

        # 4. Check termination and continue
        terminated = env_output.terminateds[0].item()

    return final_sample_state, sample_metrics
```

**Key Insight**: While this sample is waiting for `calculate_rewards()` (tool execution), other samples continue their own `async_generate_response_for_sample_turn()` calls.

**3. Async Generation Per Sample** (`rollouts.py:544-608`)
```python
async def async_generate_response_for_sample_turn(
    policy_generation: GenerationInterface,
    sample_message_log: list[dict],
    sample_stop_strings: list[str] | None,
    tokenizer: TokenizerType,
    max_seq_len: int,
    greedy: bool = False,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, dict[str, float]]:
    """Generate a response for a single sample's turn using async generation."""

    # Convert single sample to batch format
    batch_message_logs = [sample_message_log]

    # Generate response using async version
    updated_batch, generated_ids, gen_metrics = await generate_responses_async(
        policy_generation,
        generation_input_data,
        dummy_batch,
        tokenizer,
        input_lengths=input_lengths,
        include_logprobs=True,
        greedy=greedy,
    )

    return updated_message_log, generated_tokens, input_lengths, gen_metrics
```

**4. Async vLLM Generation** (`rollouts.py:120-222`)
```python
async def generate_responses_async(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], list[torch.Tensor], dict[str, float | int]]:
    """Async version of generate_responses that properly calls generate_async."""

    # Check if this is vLLM with async_engine enabled
    use_async_generation = (
        hasattr(policy_generation, "cfg")
        and "vllm_cfg" in policy_generation.cfg
        and policy_generation.cfg["vllm_cfg"]["async_engine"]
        and hasattr(policy_generation, "generate_async")
    )

    assert use_async_generation, (
        "Async generation is not enabled. Please enable async generation by setting "
        "async_engine=True in the vllm_cfg section of the policy config."
    )

    # Use async generation with per-sample streaming
    collected_indexed_outputs: list[
        tuple[int, BatchedDataDict[GenerationOutputSpec]]
    ] = []
    async for original_idx, single_item_output in policy_generation.generate_async(
        generation_input_data, greedy=greedy
    ):
        collected_indexed_outputs.append((original_idx, single_item_output))

    # Sort by original_idx to ensure order matches generation_input_data
    collected_indexed_outputs.sort(key=lambda x: x[0])

    # Extract in correct order
    ordered_batched_data_dicts = [item for _, item in collected_indexed_outputs]

    generation_outputs = BatchedDataDict.from_batches(
        ordered_batched_data_dicts,
        pad_value_dict={"output_ids": tokenizer.pad_token_id, "logprobs": 0.0},
    )

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        assistant_message = {
            "role": "assistant",
            "content": text,
            "token_ids": output_ids[i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            assistant_message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(assistant_message)

    # Track per-worker load balancing
    if "gen_leader_worker_idx" in generation_outputs:
        v = generation_outputs["gen_leader_worker_idx"][0]
        gen_metrics["gen_leader_worker_idx"] = (
            int(v[0]) if isinstance(v, list) else int(v)
        )

    return batch, generated_ids, gen_metrics
```

### vLLM Async Engine Implementation

**1. AsyncLLM Engine** (`vllm_worker_async.py:128-146`)
```python
def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs

    self.llm_async_engine_args = AsyncEngineArgs(**llm_kwargs)
    self.llm = AsyncLLM.from_engine_args(self.llm_async_engine_args)

    # Optionally expose HTTP server for OpenAI-compatible API
    if self.cfg["vllm_cfg"].get("expose_http_server"):
        self.server_thread, self.base_url, self.http_server = (
            self._setup_vllm_server()
        )
```

**2. Async Generation with Per-Sample Yielding** (`vllm_worker_async.py:496-714`)
```python
async def generate_async(
    self,
    data: BatchedDataDict[GenerationDatumSpec],
    greedy: bool = False,
) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
    """Generate a batch of data using vLLM's AsyncLLMEngine, yielding results as they are ready.

    Yields:
        Tuple of (original_index, BatchedDataDict for the single sequence)
    """
    if not self.cfg["vllm_cfg"]["async_engine"]:
        raise RuntimeError(
            "generate_async can only be used when async_engine is enabled in vLLM config."
        )

    batch_size = input_ids_batch.shape[0]

    # Ensure generate_async only receives single samples
    assert batch_size == 1, (
        f"generate_async is restricted to handle only single samples, "
        f"but received batch_size={batch_size}."
    )

    async def process_single_sample(sample_idx):
        """Process a single sample and return the result."""
        request_id = str(uuid.uuid4())

        # Generate using vLLM async engine
        vllm_request_generator = self.llm.generate(
            prompt=prompt,
            sampling_params=sampling_params_for_request,
            request_id=request_id,
        )

        # Get the final result from the generator
        final_request_output = None
        async for req_output in vllm_request_generator:
            final_request_output = req_output

        # Process the output
        generation_details = final_request_output.outputs[0]
        generated_token_ids = list(generation_details.token_ids)

        # Build result batch
        result_batch = BatchedDataDict[GenerationOutputSpec]({
            "output_ids": output_ids_single_item_batched,
            "logprobs": logprobs_single_item,
            "generation_lengths": generation_lengths_tensor,
            "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
        })

        return (sample_idx, result_batch)

    # Create tasks for all samples and yield results as they complete
    sample_tasks = [
        asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
    ]

    # Yield results as they become available (NOT in order!)
    for completed_task in asyncio.as_completed(sample_tasks):
        try:
            result = await completed_task
            yield result
        except Exception as e:
            # Cancel remaining tasks
            for task in sample_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*sample_tasks, return_exceptions=True)
            raise e
```

**Key Insight**:
- Uses `asyncio.as_completed()` to yield results as they finish
- This means faster samples don't wait for slower ones
- vLLM's async engine can handle multiple concurrent requests

### How Tool Calling is Pipelined

**Scenario: 4 samples in a batch, each doing multi-turn tool calling**

```
Time →

Sample 1: [Gen T1]─────────┐                [Gen T2]──────────┐
                           ↓                                  ↓
                    [Tool Exec T1]                     [Tool Exec T2]
                    (blocking)                         (blocking)

Sample 2:     [Gen T1]─────────┐          [Gen T2]──────────┐
                                ↓                            ↓
                         [Tool Exec T1]              [Tool Exec T2]

Sample 3:         [Gen T1]─────────┐  [Gen T2]──[Done]
                                    ↓
                             [Tool Exec T1]

Sample 4:             [Gen T1]──[Done]

vLLM AsyncLLM: [Req1]─[Req2]─[Req3]─[Req4]─[Req1.T2]─[Req2.T2]─[Req3.T2]
               All in-flight simultaneously, results streamed as ready
```

**Why This Works:**
1. Each sample has its own `async def run_sample_multi_turn_rollout()` task
2. When Sample 1 calls a tool and blocks on `calculate_rewards()`, its task yields control
3. Sample 2, 3, 4 continue executing their own generations
4. vLLM's `AsyncLLM` engine maintains a queue of in-flight generation requests
5. As soon as one generation completes, the next request starts processing
6. No sample blocks any other sample

### Comparison with Standard Batch Processing

**Standard (Synchronous) Approach:**
```
Batch of 4 samples → Generate all 4 → Wait for ALL to finish → Execute all 4 tools → Repeat
Problem: Slowest sample blocks the entire batch
```

**NeMo-RL Async Approach:**
```
Sample 1: Gen → Tool → Gen → Tool → Done
Sample 2:   Gen → Tool → Gen → Done
Sample 3:     Gen → Done
Sample 4:       Gen → Tool → Done

All happening concurrently!
Problem solved: Fast samples don't wait for slow ones
```

### Key Insights for vLLM Usage

✅ **Async engine is the foundation**: Must set `async_engine: true` in vLLM config

✅ **Sample-level concurrency**: Use `asyncio.gather()` to run all samples concurrently

✅ **vLLM handles the queue**: AsyncLLM engine manages multiple in-flight requests internally

✅ **Non-blocking tool calls**: Tool execution happens outside vLLM, doesn't block generation

✅ **Streaming results**: Use `async for` to stream results as they complete, not FIFO

✅ **Per-worker load balancing**: Engine tracks which worker handled each request

✅ **Message history tracking**: Each sample maintains its own message log independently

✅ **Response ordering**: Results can arrive out-of-order, must track original indices

### Message Log Structure (Concatenated Storage)

**File:** `nemo_rl/experience/rollouts.py:94-100`

**NeMo-RL stores token IDs in EACH message:**

```python
# After generation:
assistant_message = {
    "role": "assistant",
    "content": generated_text,
    "token_ids": output_ids[i, input_length:total_length],     # Store IDs
    "generation_logprobs": logprobs[i, input_length:total_length],  # Store logprobs
}
batch["message_log"][i].append(assistant_message)

# Full conversation example:
message_log = [
    {
        "role": "user",
        "content": "Task prompt",
        "token_ids": [101, 102, 103, ...]
    },
    {
        "role": "assistant",
        "content": "<tool_call>search(...)</tool_call>",
        "token_ids": [345, 346, 347, ...],           # LLM output
        "generation_logprobs": [-0.1, -0.2, ...]
    },
    {
        "role": "tool",
        "content": "Search results...",
        "token_ids": [456, 457, 458, ...]            # Tool result
    },
    {
        "role": "assistant",
        "content": "Answer: ...",
        "token_ids": [567, 568, 569, ...],           # LLM output
        "generation_logprobs": [-0.15, -0.18, ...]
    },
]
```

**Why this structure:**
- Enables later concatenation into single training sequence
- Preserves per-token logprobs for policy gradient
- Can build response_mask by checking message roles
- Each message is self-contained with all needed info

**Building response_mask from message_log:**
```python
response_mask = []
for msg in message_log:
    token_len = len(msg["token_ids"])
    if msg["role"] == "assistant":
        response_mask.extend([1] * token_len)  # TRAIN
    else:
        response_mask.extend([0] * token_len)  # IGNORE
```

---

### vLLM Async API Pattern

**Key Pattern from NeMo-RL:**
```python
# 1. Create AsyncLLM engine
from vllm.v1.engine.async_llm import AsyncLLM
llm = AsyncLLM.from_engine_args(args)

# 2. For each sample, submit async request
async def process_sample(sample):
    request_id = str(uuid.uuid4())

    # This returns an async generator
    vllm_generator = llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    )

    # Stream results (or just get final)
    final_output = None
    async for output in vllm_generator:
        final_output = output

    return final_output

# 3. Run all samples concurrently
tasks = [asyncio.create_task(process_sample(s)) for s in samples]

# 4. Yield results as they complete
for completed in asyncio.as_completed(tasks):
    result = await completed
    yield result
```

**What vLLM Does Internally:**
- Maintains a queue of active requests
- Schedules requests onto available GPU resources
- Streams tokens as they're generated
- Returns complete outputs when done
- Handles multiple concurrent requests without blocking

### Configuration for Async Tool Calling

**Minimal Config:**
```yaml
policy:
  generation:
    backend: "vllm"
    vllm_cfg:
      async_engine: true  # Enable async mode
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
      gpu_memory_utilization: 0.6
      max_model_len: 2048
```

**For Multi-turn with Tools:**
```yaml
grpo:
  max_rollout_turns: 10  # Allow up to 10 turns per sample

# Each sample can make multiple tool calls across turns
# All samples run concurrently without blocking each other
``

### Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│  Async GRPO Training Loop                               │
│  └─ run_async_multi_turn_rollout()                      │
│     └─ asyncio.gather([                                 │
│        run_sample_multi_turn_rollout(sample_1),         │
│        run_sample_multi_turn_rollout(sample_2),         │
│        run_sample_multi_turn_rollout(sample_3),         │
│        ...                                              │
│     ])                                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Per-Sample Multi-turn Loop (runs independently)        │
│  for turn in range(max_turns):                          │
│    1. await async_generate_response_for_sample_turn()   │
│       └─ await generate_responses_async()               │
│          └─ async for idx, output in                    │
│             policy_generation.generate_async()          │
│    2. calculate_rewards() - Execute tool                │
│    3. Add tool result to message log                    │
│    4. Continue if not done                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  vLLM AsyncLLM Engine (handles queue internally)        │
│  - Receives requests with unique request_id             │
│  - Maintains queue of in-flight requests                │
│  - Schedules onto available GPU resources               │
│  - Streams results as they complete (not FIFO)          │
│  - Multiple requests processed simultaneously           │
└─────────────────────────────────────────────────────────┘
```

### Key Takeaways for Forge

1. **Use async/await pattern**: Essential for non-blocking tool execution
2. **Sample-level tasks**: Each sample should be its own async task
3. **vLLM async engine**: Handles the queueing and scheduling internally
4. **Concurrent execution**: Use `asyncio.gather()` to run all samples together
5. **Independent message logs**: Each sample maintains its own conversation history
6. **Stream results**: Use `async for` to handle results as they arrive
7. **Tool calls don't block**: While one sample waits for tool response, others continue

**Critical for Performance:**
- Setting `async_engine: true` enables the pipelining
- Each sample runs independently, so fast samples don't wait for slow ones
- vLLM's async engine manages the GPU efficiently
- Tool execution happens outside vLLM, doesn't block the generation queue

---

---

## Example 5: PRIME-RL Wiki Search (Verifiers + vLLM Tool Calling)

**Location:** `/home/felipemello/forge/prime-rl/`

PRIME-RL is a production framework for async RL training that integrates with the `verifiers` environment library. The wiki-search example demonstrates multi-turn tool calling with native function calling support in vLLM.

### Architecture

```
Orchestrator (Rollout Generation)
    ↓
vLLM Inference Server (Native Tool Calling) ← BLACK BOX
    ↓
Verifiers Environment (ToolEnv) ← BLACK BOX
    ↓
Trainer (LoRA Fine-tuning)
```

### Key Philosophy

**Environment-Centric Design**: Unlike BlackJack/Tinker/VERL which implement rollout loops manually, PRIME-RL delegates multi-turn and tool calling to **external libraries** (`vLLM` for tool calling, `verifiers` for multi-turn loop). The framework just calls `env.generate()` and receives back complete rollouts.

**IMPORTANT:** Much of the implementation is in external libraries (vLLM and verifiers) whose source isn't in this codebase, so we can only see the API boundaries.

### Key Components

**1. vLLM Configuration - Enabling Native Tool Calling**

```toml
# examples/wiki_search/rl.toml
[inference.model]
enable_auto_tool_choice = true  # vLLM flag - enables tool calling
tool_call_parser = "hermes"     # Use Hermes format parser
```

**What this does (from prime-rl source):**
```python
# src/prime_rl/inference/config.py:79-91
enable_auto_tool_choice: bool = False  # Passed to vLLM as `--enable-auto-tool-choice`
tool_call_parser: str = "hermes"        # Passed to vLLM as `--tool-call-parser`

# src/prime_rl/inference/vllm/server.py:59-60
if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
    ToolParserManager.import_tool_parser(args.tool_parser_plugin)
```

**What we DON'T know (vLLM internals):**
- Exactly how the hermes parser works
- How vLLM formats tools in prompts
- The exact format of parsed tool calls

**What we DO know:**
- vLLM has built-in parsers for different tool formats
- "hermes" refers to Nous Hermes tool calling format
- These flags are just passed through to vLLM's engine

**2. Multi-turn Rollout Flow (From Tinker-Cookbook Example)**

The actual multi-turn logic is in the **verifiers library**. Here's how it's called:

```python
# tinker-cookbook/recipes/verifiers_rl/train.py:108-147

async def run_one_rollout():
    # Hook to capture each generation step
    recorded = []
    def hook(messages, model_input, tokens, logprobs):
        recorded.append((list(messages), model_input, list(tokens), list(logprobs)))

    local_client = TinkerAsyncOpenAIClient(sampling_client, renderer, tokenizer)
    local_client.set_generation_hook(hook)  # Track each turn

    # THE KEY CALL - environment handles multi-turn loop
    completion, state = await builder.vf_env.rollout(
        client=local_client,      # OpenAI-compatible client
        model="tinker",
        prompt=builder.prompt,    # Initial user message
        answer=builder.answer,
        task=builder.task,
        info=builder.info,
        sampling_args={},
    )

    # Score the final result
    rs = await builder.vf_env.rubric.score_rollout(
        prompt=builder.prompt,
        completion=completion,
        answer=builder.answer,
        state=state,
        task=builder.task,
        info=builder.info,
    )

    # Build trajectory from recorded turns
    transitions = []
    for _msgs, model_input, tokens, logprobs in recorded:
        transitions.append(Transition(
            ob=model_input,
            ac=TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs),
            reward=0.0,
            episode_done=False,
            metrics={},
        ))
    transitions[-1].reward = float(rs.reward)  # Assign final reward
    transitions[-1].episode_done = True
```

**What `vf_env.rollout()` does (we DON'T have the source):**
1. Calls `client.chat.completions.create()` in a loop
2. Parses model output for tool calls
3. Executes tools and adds results to conversation
4. Continues until task complete or max turns
5. Returns final completion + full state

**What we DO see:**
- Environment calls the client multiple times (hook records each turn)
- Each turn captures: messages, prompt, tokens, logprobs
- Final reward is assigned after full episode
- All turns get reward=0 except the last

**3. PRIME-RL's Simpler API**

PRIME-RL doesn't even track individual turns - it just calls env.generate():

```python
# src/prime_rl/utils/vf.py:81-99
async def generate_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    problem: dict,
    rollouts_per_example: int,
    sampling_args: dict,
) -> vf.GenerateOutputs:
    """Environment handles everything: multi-turn, tool calling, scoring."""
    semaphore = get_semaphore()

    return await env.generate(
        inputs=Dataset.from_list([problem] * rollouts_per_example),
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        semaphore=semaphore,
    )
```

**4. Processing Results - The ACTUAL Code (scheduler.py:71-86)**

This is where PRIME-RL processes the completed rollouts:

```python
def process_generate_outputs(self, generate_outputs: GenerateOutputs) -> list[Rollout]:
    # Call verifiers processing function (masks tool results)
    processed_outputs: ProcessedOutputs = self.env.process_env_results_vllm(
        prompts=generate_outputs.prompt,
        completions=generate_outputs.completion,
        states=generate_outputs.state,
        rewards=generate_outputs.reward,
        processing_class=self.tokenizer,
        max_seq_len=self.seq_len,
        mask_env_responses=self.config.mask_env_responses,  # KEY: Don't train on tool results
        zero_truncated_completions=self.config.zero_truncated_completions,
        mask_truncated_completions=self.config.mask_truncated_completions,
    )

    # Rest is standard RL processing
    advantages = compute_advantages(...)
    rollouts = make_rollouts(generate_outputs, processed_outputs, advantages, is_truncated)
    self.buffer.update(rollouts)
    accepted_rollouts = self.buffer.sample_rollouts(n=num_problems)
    return accepted_rollouts
```

**What `mask_env_responses` does (from verifiers library):**
- Similar to VERL's `response_mask` concept
- Marks which tokens to train on vs ignore
- Tool results are masked out (set to ignore)
- Only LLM-generated tokens are trained on

**5. Rollout Data Structure (utils/vf.py:136-148)**

```python
class Rollout(TypedDict):
    example_id: int
    task: str
    prompt_ids: list[int]
    prompt_mask: list[int]          # What to compute loss on in prompt
    completion_ids: list[int]
    completion_mask: list[int]      # What to compute loss on in completion (masking applied here)
    completion_logprobs: list[float]
    reward: float
    advantage: float
    is_truncated: bool
    metrics: dict[str, float]
```

### Verifiers Implementation Details (Now We Have The Source!)

#### **The Multi-Turn Rollout Loop** (multiturn_env.py:55-149)

```python
async def rollout(self, client: AsyncOpenAI, model: str, prompt: Messages, ...) -> tuple[Messages, State]:
    """Generate a multi-turn rollout with the environment."""
    is_completed = False
    state = await self.init_state(prompt, completion, answer, task, info, example_id)

    while not is_completed:
        # Build context from prompt + completion so far
        context_messages = await self.get_context_messages(state)

        if await self.is_completed(context_messages, state, **kwargs):
            break

        # Call the LLM with tools
        response = await self.get_model_response(
            client, model, context_messages,
            oai_tools=info.get("oai_tools", None),  # <-- Tools passed here
            sampling_args=sampling_args,
        )
        state["responses"].append(response)

        # Extract assistant message + tool calls
        response_message = {"role": "assistant", "content": response_text}
        if response.choices[0].message.tool_calls:
            response_message["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        state["completion"].append(response_message)

        state["turn"] += 1

        # Check if done
        if await self.is_completed(context_messages, state, **kwargs):
            is_completed = True
        else:
            # Execute tools and get results
            env_msgs, state = await self.env_response(context_messages, state, **kwargs)
            state["completion"] += env_msgs  # Add tool results to history

    return state["completion"], state
```

#### **Tool Execution** (tool_env.py:43-89)

```python
class ToolEnv(MultiTurnEnv):
    def __init__(self, tools: list[Callable], max_turns: int = 10, **kwargs):
        # Convert Python functions to OpenAI tool schemas
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        super().__init__(oai_tools=self.oai_tools, max_turns=max_turns, **kwargs)

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Episode ends when assistant responds without tool calls."""
        is_assistant_message = messages[-1]["role"] == "assistant"
        no_tool_calls = "tool_calls" not in messages[-1] or messages[-1]["tool_calls"] is None
        return await super().is_completed(...) or (is_assistant_message and no_tool_calls)

    async def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        """Execute all tool calls from the last assistant message."""
        tool_messages = []
        for tool_call in messages[-1]["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]

            # Execute the tool
            result = await self.tool_map[tool_name](**tool_args)
            tool_messages.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call_id,
            })
        return tool_messages, state
```

#### **Calling OpenAI API with Tools** (environment.py:285-296)

```python
async def get_model_response(self, client: AsyncOpenAI, model: str, prompt: Messages,
                             oai_tools: list[ChatCompletionToolParam] | None = None, ...) -> ModelResponse:
    if oai_tools:
        response = await client.chat.completions.create(
            model=model,
            messages=prompt,
            tools=oai_tools,  # <-- Tool schemas passed to OpenAI API
            **sampling_args,
        )
    else:
        response = await client.chat.completions.create(
            model=model, messages=prompt, **sampling_args
        )
    return response
```

#### **Example: Defining Tools** (wiki_search.py:99-128)

```python
# Just write normal Python functions with type hints and docstrings!
async def search_pages(query: str) -> list[dict]:
    """Search for top 10 relevant articles using title embedding similarity.

    args:
        query (str): The query to search for.
    """
    results = await collection.query(query_texts=[query], n_results=10)
    return [{"page_id": results["ids"][0][i], "title": results["metadatas"][0][i]["title"]}
            for i in range(len(results["ids"][0]))]

# Create environment
env = vf.ToolEnv(
    dataset=dataset,
    rubric=rubric,
    tools=[search_pages, view_sections, read_section],  # <-- Just pass functions!
    max_turns=10,
)
```

**How tool conversion works:**
- Parses type hints: `query: str` → `{"type": "string"}`
- Uses docstring for description
- Generates OpenAI tool schema automatically

#### **Complete Flow**

```
1. ToolEnv.__init__(tools=[search_pages, ...])
   └─ convert to OpenAI schemas → store in self.oai_tools

2. rollout() loop starts:
   ├─ Turn 1: User asks "Find info on AI"
   │   ├─ get_model_response(messages=[user msg], tools=oai_tools)
   │   │   └─ client.chat.completions.create(messages=[...], tools=[...])  # vLLM formats tools in prompt
   │   ├─ Response: assistant calls search_pages(query="AI")
   │   ├─ is_completed()? No (has tool_calls)
   │   ├─ env_response():
   │   │   ├─ Parse tool_call: {function: {name: "search_pages", arguments: "{\"query\":\"AI\"}"}}
   │   │   ├─ Execute: result = await search_pages(query="AI")
   │   │   └─ Return: [{"role": "tool", "content": "[page1, page2,...]", "tool_call_id": "123"}]
   │   └─ Append tool result to completion
   │
   ├─ Turn 2: Context now includes user + assistant tool call + tool result
   │   ├─ get_model_response(messages=[user, assistant, tool, ...], tools=oai_tools)
   │   ├─ Response: assistant provides answer (no tool_calls)
   │   ├─ is_completed()? YES (no tool_calls)
   │   └─ Exit loop
   │
   └─ Return (completion, state)
```

**📊 Updated Comparison:**

| Component | BlackJack | Tinker | VERL | Verifiers/PRIME-RL |
|-----------|-----------|--------|------|----------|
| Rollout loop | ✅ Visible | ✅ Visible | ✅ Visible | ✅ **NOW VISIBLE** |
| Tool calling | N/A | ✅ Visible | ✅ Visible | ✅ **NOW VISIBLE** |
| Tool execution | N/A | ✅ Visible | ✅ Visible | ✅ **NOW VISIBLE** |
| Prompt formatting | ✅ Visible | ✅ Visible | ✅ Visible | ❌ In vLLM server |
| Response masking | N/A | N/A | ✅ Visible | ✅ Visible |

**What's STILL in vLLM (black box):**
- How tools are formatted in the prompt (model-specific)
- How tool calls are parsed from model output (hermes/mistral/llama format)
- The actual "hermes" parser implementation

### Key Insights

✅ **Clean multi-turn loop**: Simple while loop with `is_completed()` check

✅ **Tool execution is straightforward**: Parse tool_calls → execute function → return result

✅ **OpenAI API compatibility**: Just pass `tools` parameter to `client.chat.completions.create()`

✅ **vLLM handles formatting**: Server formats tools in prompt based on model

✅ **Episode termination**: Ends when assistant doesn't request tools

✅ **Response masking**: Verifiers has `process_env_results_vllm()` to mask tool results

✅ **Simple tool definition**: Just write Python functions with type hints!

### Response Masking for Multi-Turn

**File:** `verifiers/utils/processing_utils.py:72-151`

**How Verifiers builds mask by processing chat turns:**

```python
def process_chat_format_vllm(
    prompt: list[ChatMessage],
    completion: list[ChatMessage],
    state: State,
    processing_class: TokenizerBase,
    mask_env_responses: bool = False,  # KEY FLAG
):
    completion_ids = []
    completion_mask = []

    for message in completion:
        if message["role"] == "assistant":
            # LLM output - get tokens from vLLM response
            tokens = parse_chat_completion_tokens(response)
            logprobs = parse_chat_completion_logprobs(response)

            completion_ids.extend(tokens)
            completion_mask.extend([1] * len(tokens))  # TRAIN on assistant

        elif message["role"] in ["user", "tool"]:
            # Environment/tool response
            tokens = tokenizer.apply_chat_template(
                conversation=messages_consumed + [message],
                add_generation_prompt=True,
                tools=oai_tools
            )

            completion_ids.extend(tokens)

            if mask_env_responses:
                completion_mask.extend([0] * len(tokens))  # MASK for RL
            else:
                completion_mask.extend([1] * len(tokens))  # TRAIN for SFT

    return prompt_ids, prompt_mask, completion_ids, completion_mask, completion_logprobs
```

**Key points:**
- **RL training:** `mask_env_responses=True` → tool results get `mask=0`
- **SFT training:** `mask_env_responses=False` → train on everything
- Mask is built incrementally as conversation progresses
- Returned to PRIME-RL scheduler for training

**Used by PRIME-RL:**
```python
# From prime_rl scheduler.py:71-86
processed_outputs = env.process_env_results_vllm(
    prompts=generate_outputs.prompt,
    completions=generate_outputs.completion,
    states=generate_outputs.state,
    rewards=generate_outputs.reward,
    processing_class=tokenizer,
    mask_env_responses=self.config.mask_env_responses,  # TRUE for RL
)
```

---

### For Forge: What's Actionable Now

**1. You CAN implement the multi-turn loop yourself (it's simple!):**
```python
# Based on verifiers multiturn_env.py
async def play_task(env, generator, task_prompt):
    messages = [{"role": "user", "content": task_prompt}]
    done = False
    turn = 0

    while not done and turn < MAX_TURNS:
        # Call LLM with tools
        response = await generator.sample(
            messages=messages,
            tools=env.get_tools(),  # OpenAI tool schemas
        )

        # Add assistant message
        assistant_msg = {"role": "assistant", "content": response.text}
        if response.tool_calls:
            assistant_msg["tool_calls"] = response.tool_calls
        messages.append(assistant_msg)

        # Check if done
        if not response.tool_calls:
            done = True
        else:
            # Execute tools
            for tool_call in response.tool_calls:
                result = await env.execute_tool(
                    tool_call["function"]["name"],
                    json.loads(tool_call["function"]["arguments"])
                )
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call["id"],
                })

        turn += 1

    return messages
```

**2. You CAN use vLLM's native tool calling:**
```python
# In your Generator vLLM config:
vllm_config = {
    "enable_auto_tool_choice": True,
    "tool_call_parser": "hermes",  # or "mistral", "llama"
}
```

**3. You SHOULD implement response masking:**
```python
# Like VERL and verifiers:
# Track which tokens are LLM output vs tool results
response_mask = [1] * len(llm_tokens) + [0] * len(tool_result_tokens)
```

**4. You CAN define tools like verifiers:**
```python
def search_wiki(query: str) -> list[str]:
    """Search Wikipedia for relevant articles.

    Args:
        query: The search query string.

    Returns:
        List of article titles matching the query.
    """
    return wikipedia.search(query)

# Convert to OpenAI schema
tool_schema = convert_func_to_oai_tool(search_wiki)
# Use verifiers' utility or implement yourself (parse type hints + docstring)
```

**5. Consider integrating verifiers:**
- **Pros**: Clean API, tool support, community environments, masking built-in
- **Cons**: Another dependency, less control over rollout loop
- **Middle ground**: Use verifiers' tool utilities (`convert_func_to_oai_tool`) but implement your own rollout loop

### Comparison: All Five Examples

| Aspect | BlackJack | Tinker | VERL | PRIME-RL | **Verifiers** |
|--------|-----------|--------|------|----------|-----------|
| **Rollout Loop** | Manual | Env step | State machine | Delegates | **Simple while loop** |
| **Tool Calling** | No tools | Tag-based | Native + manual | vLLM native | **OpenAI native** |
| **Tool Definition** | N/A | Functions | Functions | Functions | **Type-hinted funcs** |
| **Tool Execution** | N/A | Manual async | Manual async | In env | **tool_map lookup** |
| **Prompt Formatting** | Manual | Renderer | Manual | vLLM | **vLLM** |
| **Response Masking** | No | No | Explicit | Flag | **process_env_results** |
| **Abstraction Level** | Low | Medium | Medium | High | **Medium-High** |

**Verifiers' Sweet Spot:**
- Higher level than BlackJack/VERL (clean API, tool utilities)
- Lower level than fully delegated PRIME-RL (rollout loop is visible)
- Practical tool definition (just type-hinted functions)
- Production-ready (used by PRIME-RL, Tinker, others)

---

## Performance & Async Patterns: Complete Library Comparison

### Overview: Async Execution Across All Libraries

| Library | Async Support | vLLM Flags | Concurrency Pattern | Key Efficiency Features |
|---------|--------------|------------|---------------------|------------------------|
| **BlackJack (Forge)** | ✅ Partial | None | `asyncio` coroutines | Async env.step(), but sequential episodes |
| **Tinker-Cookbook** | ✅ Partial | None | `asyncio` coroutines | Async tool execution, sequential rollouts |
| **VERL** | ✅ Full | SGLang (not vLLM) | `asyncio.gather()` for parallel tools | Parallel tool execution, state machine |
| **NeMo-RL** | ✅ **Full Pipeline** | **`async_engine: true`** | **Per-sample async tasks** | **Sample-level pipelining, non-blocking tools** |
| **PRIME-RL/Verifiers** | ✅ Full | **`enable_auto_tool_choice: true`**<br>**`tool_call_parser: "hermes"`** | `asyncio.gather()` | Native vLLM tool parsing, async tools |
| **TRL** | ❌ None | External server | Blocking HTTP | Simple but slower, no pipelining |

---

### Library-by-Library Async Details

#### **1. BlackJack (Forge OpenEnv) - Basic Async**

**Async Pattern:**
```python
# File: OpenEnv/examples/grpo_blackjack/grpo_utils.py:197-244
async def play_game(game_idx, game_id, server_url, policy, tokenizer, game_log):
    # Async generation
    responses = await policy.generate.route(prompt)  # ✅ Non-blocking

    # Async environment step
    result = env.step(OpenSpielAction(action_id=action_id))  # ✅ Non-blocking
```

**Concurrency Level:** Sequential episodes
- Episodes run one-at-a-time within a batch
- Each episode's steps are async, but episodes don't overlap

**vLLM Configuration:** None (uses Forge Generator defaults)

**Performance:**
- ✅ Non-blocking I/O for env
- ❌ No sample-level pipelining
- ❌ No parallel tool execution

**Best for:** Simple prototyping, full control over loop

---

#### **2. Tinker-Cookbook - Async Tools, Sequential Rollouts**

**Async Pattern:**
```python
# File: tinker_cookbook/rl/rollouts.py:16-34
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    while True:
        # Async generation
        ac_with_logprobs = await policy(ob, stop_condition)  # ✅ Non-blocking

        # Async environment step (includes tool execution)
        step_result = await env.step(ac_with_logprobs.tokens)  # ✅ Non-blocking

        if step_result.episode_done:
            break
```

**Tool Execution:**
```python
# File: tinker_cookbook/recipes/tool_use/search/search_env.py:789-791
async def call_search_tool(self, tool_call):
    async with _CONNECTION_SEMAPHORE:  # Rate limiting
        return await self.chroma_tool_client.invoke(tool_call)  # ✅ Async tool
```

**Concurrency Level:** Sequential rollouts
- Rollouts collected one-by-one
- Tools execute async but don't pipeline with generation

**vLLM Configuration:** None (uses Tinker's TrainingClient)

**Performance:**
- ✅ Async tool execution with rate limiting
- ✅ Non-blocking I/O
- ❌ No parallel rollouts
- ❌ No generation pipelining

**Best for:** Research, clean abstractions, moderate scale

---

#### **3. VERL - Full Async with Parallel Tools**

**Async Pattern:**
```python
# File: verl/experimental/agent_loop/tool_agent_loop.py:1368-1370
async def _handle_processing_tools_state(self, agent_data: AgentData):
    # Create parallel tool tasks
    tasks = [self._call_tool(tc, agent_data.tools_kwargs) for tc in agent_data.tool_calls]

    # Execute ALL tools in parallel
    responses = await asyncio.gather(*tasks)  # ✅ Parallel execution!
```

**Generation:**
```python
# File: verl/experimental/agent_loop/tool_agent_loop.py:1311-1317
async def _handle_generating_state(self, agent_data, sampling_params):
    # Async generation via SGLang
    output = await self.server_manager.generate(
        request_id=agent_data.request_id,
        prompt_ids=agent_data.prompt_ids,
        sampling_params=sampling_params,
    )  # ✅ Non-blocking
```

**Concurrency Level:** Parallel tools, sequential episodes
- Multiple tools execute concurrently
- Episodes still run sequentially

**vLLM Configuration:** Uses SGLang, not vLLM
- SGLang has its own async engine
- No vLLM-specific flags

**Performance:**
- ✅ Parallel tool execution within episode
- ✅ State machine for clean control flow
- ✅ Non-blocking generation
- ❌ No sample-level pipelining
- ❌ Episodes don't overlap

**Best for:** Complex tool workflows, production systems

---

#### **4. NeMo-RL - Full Pipelining (BEST PERFORMANCE)**

**vLLM Async Configuration:**
```yaml
# File: RL/examples/grpo_math_1B.yaml:218
policy:
  generation:
    backend: "vllm"
    vllm_cfg:
      async_engine: true  # ✅ CRITICAL FLAG - enables AsyncLLM
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
```

**Per-Sample Async Pattern:**
```python
# File: RL/nemo_rl/experience/rollouts.py:780-936
async def run_async_multi_turn_rollout(...):
    # Create one async task PER SAMPLE
    sample_tasks = [
        run_single_sample_with_error_handling(i, sample_state)
        for i, sample_state in enumerate(sample_initial_states)
    ]

    # ALL samples run concurrently!
    sample_results = await asyncio.gather(*sample_tasks)  # ✅ Full pipelining
```

**Per-Sample Loop:**
```python
# File: RL/nemo_rl/experience/rollouts.py:611-777
async def run_sample_multi_turn_rollout(sample_idx, ...):
    for turn in range(max_rollout_turns):
        # Async generation (doesn't block other samples)
        response = await async_generate_response_for_sample_turn(...)  # ✅

        # Execute tool (while this blocks, other samples continue!)
        env_output = calculate_rewards(sample_batch, task_to_env)  # Other samples proceed
```

**vLLM AsyncLLM Engine:**
```python
# File: RL/nemo_rl/models/generation/vllm/vllm_worker_async.py:128-146
def _create_engine(self, llm_kwargs):
    from vllm.v1.engine.async_llm import AsyncLLM
    self.llm = AsyncLLM.from_engine_args(self.llm_async_engine_args)  # ✅ Async engine

# File: RL/nemo_rl/models/generation/vllm/vllm_worker_async.py:1830-1840
async def generate_async(self, data, greedy=False):
    # Submit to vLLM async engine
    vllm_generator = self.llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    )  # ✅ Returns immediately, vLLM queues request

    # Stream results
    async for req_output in vllm_generator:
        final_output = req_output
```

**Concurrency Level:** **Per-sample pipelining** (HIGHEST)
- Each sample is independent async task
- While Sample 1 waits for tool, Samples 2/3/4 generate
- vLLM queues all requests internally

**Performance:**
- ✅ **Sample-level pipelining** (unique feature!)
- ✅ Non-blocking generation queue
- ✅ Fast samples don't wait for slow ones
- ✅ Maximum GPU utilization

**Speedup Example:**
```
Without pipelining (4 samples, 2 turns each, 10s per turn):
Sample 1: Turn 1 (10s) → Tool (5s) → Turn 2 (10s) = 25s
Sample 2: Turn 1 (10s) → Tool (5s) → Turn 2 (10s) = 25s
Sample 3: Turn 1 (10s) → Done = 10s
Sample 4: Turn 1 (10s) → Done = 10s
Total: 70s (sequential)

With NeMo-RL pipelining:
All samples overlap, max time ≈ 25s (longest sample)
Speedup: ~2.8x
```

**Best for:** Production RL at scale, maximum throughput

---

#### **5. PRIME-RL/Verifiers - Native vLLM Tool Calling**

**vLLM Tool Calling Configuration:**
```toml
# File: prime-rl/examples/wiki_search/rl.toml
[inference.model]
enable_auto_tool_choice = true  # ✅ vLLM native tool calling
tool_call_parser = "hermes"     # ✅ Use Hermes format parser
```

**What these flags do:**
- `enable_auto_tool_choice`: vLLM parses tool calls from model output automatically
- `tool_call_parser`: Specifies format (hermes/mistral/llama/internlm)
- vLLM handles prompt formatting with tools

**Async Pattern:**
```python
# File: verifiers/environment.py:55-149
async def rollout(self, client, model, prompt, ...):
    while not is_completed:
        # Async generation via OpenAI-compatible client
        response = await self.get_model_response(
            client, model, context_messages,
            oai_tools=info.get("oai_tools", None),  # ✅ Tools passed to vLLM
        )  # ✅ Non-blocking

        # Async tool execution
        env_msgs, state = await self.env_response(context_messages, state)  # ✅ Async
```

**Tool Execution:**
```python
# File: verifiers/tool_env.py:43-89
async def env_response(self, messages, state, **kwargs):
    tool_messages = []
    for tool_call in messages[-1]["tool_calls"]:
        # Execute tool (async)
        result = await self.tool_map[tool_name](**tool_args)  # ✅ Async
        tool_messages.append({...})
    return tool_messages, state
```

**Concurrency Level:** Sequential rollouts, async tools
- Rollouts run one-at-a-time
- Tools can be async within episode

**Performance:**
- ✅ vLLM native tool parsing (no manual regex)
- ✅ Async tool execution
- ✅ Clean OpenAI-compatible API
- ❌ No sample pipelining
- ❌ PRIME-RL delegates to verifiers (black box)

**Best for:** Standard tool calling tasks, clean abstractions

---

#### **6. TRL - Synchronous (Simple but Slow)**

**Pattern:**
```python
# File: trl/examples/scripts/openenv/catch.py:162-215
def rollout_func(prompts, args, processing_class, client, gen_url):
    for prompt in prompts:
        for _ in range(args.num_generations):
            while not obs.done:
                # Blocking HTTP request to vLLM server
                response = requests.post(gen_url, json=payload)  # ❌ BLOCKING
                response.raise_for_status()
                result = response.json()

                # Blocking environment step
                env_result = client.step(action)  # ❌ BLOCKING
```

**Concurrency Level:** None (fully synchronous)

**vLLM Configuration:** External HTTP server
- TRL doesn't configure vLLM directly
- Uses separate vLLM server process
- No async flags

**Performance:**
- ❌ Blocking HTTP calls
- ❌ No pipelining
- ❌ Sequential processing
- ✅ Simple to understand and debug

**Best for:** Prototyping, education, debugging

---

### Key Performance Insights

**1. vLLM Async Engine is Critical for Pipelining**
- Only NeMo-RL uses `async_engine: true`
- This enables `AsyncLLM` class in vLLM
- Without it, generation blocks even with async/await

**2. Sample-Level Pipelining is Unique to NeMo-RL**
- Most libraries: episodes run sequentially
- NeMo-RL: each sample is independent task
- Massive speedup when samples have variable length

**3. Tool Execution Async ≠ Generation Async**
- Tinker, VERL: async tools but sequential rollouts
- NeMo-RL: both tools AND generation are pipelined
- Big difference in throughput

**4. vLLM Native Tool Calling Reduces Overhead**
- PRIME-RL: `enable_auto_tool_choice` → vLLM parses tools
- Others: manual regex/tag parsing
- Native parsing is faster and more reliable

**5. Async/Await Alone Doesn't Pipeline**
- BlackJack/Tinker: async/await but sequential episodes
- Need `asyncio.gather()` with independent tasks
- NeMo-RL does this at sample level

---

### Recommendations for Forge

**For Maximum Performance:**
1. Enable vLLM async: `async_engine: true` (NeMo-RL pattern)
2. Per-sample async tasks: `asyncio.gather([play_task(s) for s in samples])`
3. Native tool calling: `enable_auto_tool_choice: true` (if using vLLM server)

**For Simplicity:**
1. Start with TRL pattern (synchronous)
2. Add async/await for tools (Tinker pattern)
3. Optimize later if bottlenecked

**For Production:**
1. Use NeMo-RL async patterns
2. Add PRIME-RL's vLLM tool calling flags
3. Implement VERL's parallel tool execution

---

## Example 6: TRL GRPO with OpenEnv (Low-Level Implementation)

**Location:** `/home/felipemello/forge/trl/examples/scripts/openenv/`

TRL implements multi-turn rollouts for GRPO using the **`rollout_func` pattern**. This is an experimental hook that allows custom generation logic to replace TRL's default single-turn generation.

### Key Insight: TRL GRPO is Single-Turn by Default

**CRITICAL:** TRL's `GRPOTrainer` does NOT have built-in multi-turn support. The core trainer (`trl/trainer/grpo_trainer.py`) implements only:
1. Single prompt → single completion
2. Score with reward function
3. Train

For multi-turn, you MUST use the `rollout_func` parameter.

### Architecture

```
TRL GRPO Trainer
    ↓
Custom rollout_func (USER PROVIDED)
    ↓
vLLM Server (HTTP) → Multi-turn Loop → OpenEnv Client (HTTP)
    ↓
Returns: prompt_ids, completion_ids, logprobs (concatenated across ALL turns)
    ↓
GRPO treats entire episode as ONE sequence for training
```

### The `rollout_func` Signature

```python
# From trl/trainer/grpo_trainer.py:113
RolloutFunc = Callable[[list[str], Any, Any], dict[str, Any]]

# Signature:
def rollout_func(
    prompts: list[str],           # Batch of prompts from dataset
    args: GRPOConfig,              # Training config (temperature, max_tokens, etc.)
    processing_class: Tokenizer,   # Tokenizer for encoding/decoding
) -> dict[str, Any]:
    # Must return:
    return {
        "prompt_ids": list[list[int]],      # Token IDs of prompts (per-episode)
        "completion_ids": list[list[int]],  # Token IDs of completions (per-episode)
        "logprobs": list[list[float]],      # Log probs (per-token, per-episode)
        # Optional: any extra fields for reward functions
        "custom_reward": list[float],
        ...
    }
```

### Example 1: Catch Game (Multi-Turn Episode Loop)

**File:** `trl/examples/scripts/openenv/catch.py:162-215`

This example shows the CORE pattern for multi-turn with TRL:

```python
def rollout_func(
    prompts: list[str],
    args: GRPOConfig,
    processing_class,
    client: OpenSpielEnv,  # Injected via lambda
    gen_url: str,          # Injected via lambda
) -> dict[str, list]:
    """Generate completions via vLLM and compute environment rewards."""
    env_rewards = []
    all_prompt_ids, all_completion_ids, all_logprobs = [], [], []

    # OUTER LOOP: Process each prompt from the dataset
    for base_prompt in prompts:
        # MIDDLE LOOP: Generate G rollouts per prompt (for GRPO group)
        for _ in range(args.num_generations):
            env_result = client.reset()
            obs = env_result.observation
            total_reward = 0.0

            # Storage for THIS episode's tokens (across ALL turns)
            episode_prompt_ids, episode_completion_ids, episode_logprobs = [], [], []

            # INNER LOOP: Multi-turn episode loop
            while not obs.done:
                # 1. Build prompt from current observation
                episode_msg = {
                    "prompt": [{
                        "role": "user",
                        "content": f"{base_prompt}\n\n{obs.info_state}\n"
                    }]
                }
                episode_prompt = apply_chat_template(episode_msg, processing_class)

                # 2. Generate via vLLM server (HTTP request)
                payload = {
                    "prompts": [episode_prompt["prompt"]],
                    "n": 1,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_completion_length,
                }
                response = requests.post(gen_url, json=payload)
                response.raise_for_status()
                result = response.json()

                # 3. CRITICAL: Accumulate token IDs across turns
                # This makes the entire episode ONE sequence for training
                episode_prompt_ids.extend(result["prompt_ids"][0])
                episode_completion_ids.extend(result["completion_ids"][0])
                episode_logprobs.extend(result["logprobs"][0])

                # 4. Parse action from completion text
                completion_text = processing_class.batch_decode(
                    result["completion_ids"],
                    skip_special_tokens=True
                )[0]
                numbers = re.findall(r"\b([0-2])\b", completion_text)
                action_id = int(numbers[0]) if numbers else obs.legal_actions[0]

                # 5. Step environment
                env_result = client.step(OpenSpielAction(action_id=action_id, game_name="catch"))
                total_reward += env_result.reward or 0.0
                obs = env_result.observation

            # Store the ENTIRE episode as ONE rollout
            env_rewards.append(total_reward)
            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": env_rewards,  # Extra field for reward function
    }
```

### Key Implementation Tricks

#### 1. **Token Concatenation** (THE CRITICAL TRICK)

```python
# EACH TURN adds to the same lists
episode_prompt_ids.extend(result["prompt_ids"][0])
episode_completion_ids.extend(result["completion_ids"][0])
episode_logprobs.extend(result["logprobs"][0])
```

**Why this works:**
- Multi-turn episode becomes ONE long sequence: `[turn1_prompt, turn1_completion, turn2_prompt, turn2_completion, ...]`
- GRPO trains on the ENTIRE sequence as if it were one completion
- Gradient flows through all turns
- Model learns the full multi-turn policy

**Example:**
```python
# Turn 1: "What's 2+2?" → "4"
# Turn 2: "What's 4+2?" → "6"
# Turn 3: "What's 6+2?" → "8"

# Becomes ONE sequence:
prompt_ids = [tok("What's 2+2?"), tok("4"), tok("What's 4+2?"), tok("6"), tok("What's 6+2?"), tok("8")]
# GRPO treats this as ONE generation and trains on ALL of it
```

#### 2. **vLLM Server Communication** (Synchronous HTTP)

```python
payload = {
    "prompts": [episode_prompt["prompt"]],
    "n": 1,  # Only 1 completion per request
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_tokens": args.max_completion_length,
}
response = requests.post(gen_url, json=payload)  # BLOCKING
result = response.json()
```

**Key details:**
- Uses external vLLM server (not the training model)
- HTTP POST request per turn
- **BLOCKING** call (no async)
- vLLM returns: `{"prompt_ids": [[...]], "completion_ids": [[...]], "logprobs": [[...]]}`
- Response format matches TRL's expected output

**Why external server:**
- Keeps generation separate from training
- Avoids memory conflicts
- Can use different devices

#### 3. **Nested Loop Structure**

```python
for base_prompt in prompts:              # Dataset prompts
    for _ in range(num_generations):     # G rollouts (GRPO group)
        while not obs.done:              # Multi-turn episode
            # Generate → Parse → Step
```

**Loop purposes:**
1. **Outer:** Batch of prompts from dataset (GRPO's dataloader)
2. **Middle:** Generate G completions per prompt (for group normalization)
3. **Inner:** Multi-turn loop until episode ends

**Output shape:**
- `len(prompts) * num_generations` total episodes
- Each episode: variable length (depends on turns to completion)

#### 4. **Chat Template Per Turn**

```python
episode_msg = {"prompt": [{"role": "user", "content": f"{base_prompt}\n\n{obs.info_state}\n"}]}
episode_prompt = apply_chat_template(episode_msg, processing_class)
```

**Important:**
- Each turn builds a FRESH prompt
- Does NOT maintain conversation history in the prompt
- Environment state (`obs.info_state`) provides context
- Chat template wraps it properly

**For tool calling, you'd do:**
```python
messages = [
    {"role": "system", "content": "You have access to tools..."},
    {"role": "user", "content": task},
    # Previous turns would go here
]
```

### Example 2: Wordle (More Sophisticated Multi-Turn)

**File:** `trl/examples/scripts/openenv/wordle.py:331-425`

Wordle demonstrates MORE advanced patterns:

#### **1. Conversation History Management** (wordle.py:254-273)

```python
def format_history(messages: Iterable[TextArenaMessage]) -> str:
    lines = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)

def make_user_prompt(prompt_text: str, messages: Iterable[TextArenaMessage]) -> str:
    history = format_history(messages)
    prompt_section = prompt_text.strip()
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )
```

**Key insight:** Environment maintains the message history, code formats it for each turn's prompt.

#### **2. Multiple Reward Signals** (wordle.py:394-425)

```python
for _turn in range(cli_args.max_turns):
    if result.done:
        break

    # Build prompt with history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, ...)

    # Generate
    vllm_result = request_vllm_completion(...)

    # Extract guess
    guess = extract_guess(completion_text)

    # Step environment
    result = env.step(TextArenaAction(message=guess))

    # MULTIPLE reward signals
    feedback = extract_wordle_feedback(observation)
    green_count, yellow_count = extract_feedback_counts(feedback)

    green_score = green_count / 5.0
    yellow_score = yellow_count / 5.0
    repetition_score = scale_repetition_score(...)
    correct_score = float(result.reward or 0.0)

    # Store for return
    green_scores.append(green_score)
    yellow_scores.append(yellow_score)
    repetition_scores.append(repetition_score)
    correct_scores.append(correct_score)

# Return FINAL rewards from each signal
return {
    "prompt_ids": prompt_ids,
    "completion_ids": completion_ids,
    "logprobs": logprobs,
    "correct_reward": correct_scores[-1],      # Final turn
    "green_reward": green_scores[-1],          # Final turn
    "yellow_reward": yellow_scores[-1],        # Final turn
    "repetition_reward": repetition_scores[-1],# Final turn
}
```

#### **3. Multiple Reward Functions** (wordle.py:484-509)

```python
def reward_correct(completions, **kwargs):
    return kwargs.get("correct_reward", [0.0] * len(completions))

def reward_greens(completions, **kwargs):
    return kwargs.get("green_reward", [0.0] * len(completions))

def reward_yellows(completions, **kwargs):
    return kwargs.get("yellow_reward", [0.0] * len(completions))

def reward_repetition(completions, **kwargs):
    return kwargs.get("repetition_reward", [0.0] * len(completions))

# In trainer:
trainer = GRPOTrainer(
    reward_funcs=[
        reward_correct,
        reward_greens,
        reward_yellows,
        reward_repetition,
    ],
    args=grpo_config,
    rollout_func=wrapped_rollout,
)
```

**How it works:**
1. `rollout_func` computes multiple reward signals, stores in dict
2. Each reward function extracts its signal from kwargs
3. GRPO sums all rewards: `total_reward = w1*r1 + w2*r2 + w3*r3 + w4*r4`
4. Can weight each signal with `reward_weights=[1.0, 0.5, 0.5, 0.2]`

#### **4. Max Turns Limit** (wordle.py:352)

```python
for _turn in range(cli_args.max_turns):  # Limit to 5 guesses
    if result.done:
        break
    # ... generate and step
```

**Important:**
- Prevents infinite loops
- Truncates long episodes
- Similar to `max_steps` in RL

### How to Use in Forge

**Step 1: Define rollout function**

```python
def custom_rollout(prompts, args, processing_class, env_client, gen_url):
    all_prompt_ids, all_completion_ids, all_logprobs = [], [], []
    rewards = []

    for prompt in prompts:
        for _ in range(args.num_generations):
            # Multi-turn loop here
            episode_prompt_ids, episode_completion_ids, episode_logprobs = [], [], []
            env_result = env_client.reset()

            while not env_result.done:
                # Generate → Parse → Step → Accumulate
                ...

            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)
            rewards.append(total_reward)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": rewards,
    }
```

**Step 2: Pass to trainer**

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=lambda completions, **kwargs: kwargs.get("env_reward", []),
    rollout_func=lambda p, a, pc: custom_rollout(p, a, pc, env, gen_url),
    args=grpo_config,
    train_dataset=dataset,
)
```

### TRL Does NOT Have Native Tool Calling for GRPO

**Important realization:**

1. **No built-in tool calling:** TRL's GRPO does NOT have native support for tool execution
2. **Environment IS the tool:** The OpenEnv client acts as the "tool executor"
   - `env.step(action)` = "execute tool"
   - `env.observation` = "tool result"
3. **Text parsing:** Actions are parsed from model output text (regex, etc.)
4. **No async:** Everything is synchronous (blocking HTTP calls)

**For actual tool calling (like function calling), you'd need to:**

```python
while not done:
    # Generate
    response = vllm_generate(prompt)

    # Parse tool calls from text
    if "<function_call>" in response.text:
        tool_call = parse_tool_call(response.text)

        # Execute tool (YOUR CODE)
        tool_result = execute_tool(tool_call["name"], tool_call["args"])

        # Add to history
        messages.append({"role": "assistant", "tool_calls": [tool_call]})
        messages.append({"role": "tool", "content": str(tool_result)})

        # Continue
        prompt = build_prompt(messages)
    else:
        # No tool call, end episode
        done = True
```

### Comparison: TRL vs Forge BlackJack vs VERL

| Aspect | Forge BlackJack | TRL + OpenEnv | VERL | Verifiers |
|--------|-----------------|---------------|------|-----------|
| **Multi-turn loop** | Manual in play_game() | In rollout_func | State machine | While loop in env |
| **Generator** | Forge Generator (vLLM) | External vLLM server | SGLang/vLLM | AsyncOpenAI |
| **Token accumulation** | Per step (not concat) | **Concatenate across turns** | Per turn | Per turn |
| **Episode structure** | One Episode per step | **One episode = full game** | One episode = full convo | One episode = full convo |
| **Environment** | OpenEnv (sync) | OpenEnv (sync HTTP) | Custom | Verifiers MultiTurnEnv |
| **Async** | AsyncIO in rollouts | **No async (blocking HTTP)** | Full async/await | Full async/await |
| **Tool execution** | N/A | env.step() | Manual | tool_map lookup |
| **Reward assignment** | Final → all steps | Final reward | Step + final | Sparse at end |

### Key Takeaways for Forge

1. **Token concatenation is THE trick**
   - Entire episode becomes one sequence
   - GRPO trains on all turns together
   - Simpler than per-step episodes

2. **vLLM server separation**
   - Keeps generation off training GPU
   - Uses HTTP (blocking is fine)
   - Returns prompt_ids, completion_ids, logprobs

3. **rollout_func is the hook**
   - Replaces TRL's default generation
   - Full control over multi-turn logic
   - Can inject environment, URL, etc.

4. **No async needed (yet)**
   - TRL examples use blocking HTTP
   - Works fine for simple cases
   - Async would enable pipelining (see NeMo-RL)

5. **Multiple reward functions**
   - Define separate functions for each signal
   - GRPO sums them automatically
   - Can weight with `reward_weights`

6. **For tool calling:**
   - Parse tool calls from text output
   - Execute tools in rollout loop
   - Concatenate all tokens
   - Return final reward

### Token Concatenation Pattern (Strategy B)

**File:** `trl/examples/scripts/openenv/catch.py:162-215`

**THE CRITICAL TRICK - How TRL concatenates multi-turn into one sequence:**

```python
def rollout_func(prompts, args, processing_class, client, gen_url):
    for base_prompt in prompts:
        for _ in range(args.num_generations):
            # Storage for THIS episode's tokens (across ALL turns)
            episode_prompt_ids = []
            episode_completion_ids = []
            episode_logprobs = []

            # Multi-turn loop
            while not obs.done:
                # 1. Generate this turn
                response = requests.post(gen_url, json={
                    "prompts": [current_prompt],
                    "max_tokens": args.max_completion_length,
                })
                result = response.json()

                # 2. CONCATENATE tokens from this turn
                episode_prompt_ids.extend(result["prompt_ids"][0])
                episode_completion_ids.extend(result["completion_ids"][0])
                episode_logprobs.extend(result["logprobs"][0])

                # 3. Parse action and step environment
                action = parse_action(result["completion_ids"])
                env_result = client.step(action)

            # Return ENTIRE episode as ONE sequence
            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
    }
```

**What GRPO sees:**
```python
# Multi-turn episode with 3 turns becomes:
episode_completion_ids = [
    # Turn 1
    [345, 346, 347],      # "Action: 2"
    # Turn 2
    [456, 457, 458],      # "Action: 1"
    # Turn 3
    [567, 568, 569],      # "Action: 0"
]
# Flattened to: [345, 346, 347, 456, 457, 458, 567, 568, 569]

# GRPO trains on this as ONE completion
# Gradient flows through all turns
```

**Note:** TRL doesn't use response_mask in these examples (trains on everything). For tool calling, you'd need to add masking.

---

## Updated Comparison: All Six Examples

| Aspect | BlackJack | Tinker | VERL | NeMo-RL | Verifiers | **TRL** |
|--------|-----------|--------|------|---------|-----------|---------|
| **Rollout Loop** | Manual | Env step | State machine | Per-sample async | While loop | **In rollout_func** |
| **Tool Calling** | No tools | Tag-based | Native | Native | OpenAI native | **Text parsing** |
| **Generator** | vLLM v1 | Model.generate | vLLM/SGLang | vLLM async | vLLM/AsyncOpenAI | **vLLM server (HTTP)** |
| **Token Handling** | Per step | Per turn | Concatenated | Concatenated | Per turn | **Concatenated** |
| **Episode = ** | Single step | Full convo | Full convo | Full convo | Full convo | **Full game** |
| **Async** | AsyncIO | No | Full | **Per-sample** | Full | **None (blocking)** |
| **Response Mask** | No | No | Explicit | Explicit | process_env_results | **No** |
| **Multi Rewards** | Single | Single | Tool lifecycle | Per-step | Single | **Multiple funcs** |
| **Abstraction** | Low | Medium | Medium | Medium | Medium-High | **Hook-based** |

---

## Recommendation for Forge: Hybrid Approach

Based on all six examples, here's the recommended approach for Forge + tool calling:

### Phase 1: Simple Implementation (Like TRL)

**Goal:** Get multi-turn tool calling working ASAP

**Pattern:** Adapt TRL's `rollout_func` approach to Forge

```python
async def play_task(
    task_prompts: list[str],
    args,
    generator,  # Forge Generator
    tokenizer,
    env_client,  # OpenEnv or custom tool executor
    max_turns: int = 10,
):
    """Multi-turn rollout with tool calling."""
    all_episodes = []

    for prompt in task_prompts:
        for _ in range(args.num_generations):
            # Reset environment
            env_result = env_client.reset(task=prompt)

            # Storage for entire episode
            episode_tokens = []
            episode_logprobs = []
            messages = [{"role": "user", "content": prompt}]
            total_reward = 0.0

            for turn in range(max_turns):
                if env_result.done:
                    break

                # 1. Build prompt from message history
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

                # 2. Generate via Forge Generator
                response = await generator.generate(prompt_text)

                # 3. Concatenate tokens (THE KEY TRICK)
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                completion_ids = response.token_ids
                episode_tokens.extend(prompt_ids + completion_ids)
                episode_logprobs.extend(response.logprobs)

                # 4. Parse tool calls from response
                if is_tool_call(response.text):
                    tool_call = parse_tool_call(response.text)

                    # Execute tool
                    tool_result = env_client.execute_tool(
                        tool_call["name"],
                        tool_call["args"]
                    )

                    # Add to message history
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result)
                    })

                    # Update env
                    env_result = env_client.step(tool_call)
                    total_reward += env_result.reward or 0.0
                else:
                    # Final answer
                    messages.append({
                        "role": "assistant",
                        "content": response.text
                    })
                    env_result = env_client.finalize(response.text)
                    total_reward += env_result.reward or 0.0
                    break

            all_episodes.append({
                "token_ids": episode_tokens,
                "logprobs": episode_logprobs,
                "reward": total_reward,
                "num_turns": turn + 1,
            })

    return all_episodes
```

**Key points:**
- Concatenate all turns into ONE sequence
- Use existing Forge Generator
- Synchronous execution (blocking is OK)
- Simple text parsing for tool calls

### Phase 2: Add Response Masking (Like VERL/NeMo-RL)

**Goal:** Don't train on tool results

```python
def build_episode_with_mask(messages, tokenizer):
    """Build episode with response mask to exclude tool results."""
    all_tokens = []
    response_mask = []

    for msg in messages:
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)

        if msg["role"] == "assistant":
            # Train on assistant tokens
            all_tokens.extend(tokens)
            response_mask.extend([1] * len(tokens))
        elif msg["role"] == "tool":
            # Don't train on tool results
            all_tokens.extend(tokens)
            response_mask.extend([0] * len(tokens))
        else:
            # Prompt tokens
            all_tokens.extend(tokens)
            response_mask.extend([0] * len(tokens))

    return all_tokens, response_mask
```

### Phase 3: Async Pipelining (Like NeMo-RL)

**Goal:** Don't block on tool execution

```python
async def play_task_async(task_prompts, ...):
    """Per-sample async tasks for pipelining."""
    # Create one task per sample
    tasks = [
        asyncio.create_task(play_single_task(prompt, ...))
        for prompt in task_prompts
    ]

    # Run concurrently
    episodes = await asyncio.gather(*tasks)
    return episodes

async def play_single_task(prompt, ...):
    """Single sample multi-turn loop."""
    while not done:
        # Generate (may block)
        response = await generator.generate_async(prompt_text)

        # Parse tool call
        tool_call = parse_tool_call(response.text)

        # Execute tool (async, doesn't block other samples)
        tool_result = await env_client.execute_tool_async(...)

        # Continue
```

**Benefit:** While sample 1 waits for tool result, sample 2/3/4 continue generating

### Summary

| Phase | Complexity | Performance | Features |
|-------|-----------|-------------|----------|
| 1: Simple | Low | OK | Multi-turn, text parsing, sync |
| 2: Masking | Medium | Better | + Don't train on tool results |
| 3: Async | High | Best | + Pipelined execution |

**Recommendation:** Start with Phase 1, add Phase 2 when working, consider Phase 3 if bottlenecked.

---

## Forge: Current Capabilities & Optimization Roadmap

This section consolidates information about Forge's current state, what optimizations are available, and how to add multi-turn tool calling.

### Current Forge Architecture

#### What You Have ✅

**Generator** (`src/forge/actors/generator.py`)
- **vLLM v1 Engine**: Manual implementation mirroring AsyncLLMEngine (lines 71-578)
- **Async Interface**: `async def generate()` endpoint (line 290)
- **Request Queueing**: Uses `asyncio.Future` for async request handling (line 357)
- **Run Loop**: Continuous `schedule() → execute() → process()` pattern (line 396)
- **Architecture**: Coordinator (CPU) + Workers (GPU) via Monarch proc meshes

**GRPO Main** (`apps/grpo/main.py`)
- **Parallel Rollout Threads**: Multiple `continuous_rollouts()` tasks (line 472)
- **Async Generation**: `await policy.generate.route()` (line 373)
- **Async Rewards**: `await reward_actor.evaluate_response.route()` (line 391)
- **Async Reference Model**: `await ref_model.forward.route()` (line 402)
- **Replay Buffer**: Decoupled rollout and training loops

#### What You're Missing ❌

**Critical Missing Pieces**

1. **vLLM AsyncLLM Engine**
   - Current: Synchronous scheduler with async wrapper
   - Missing: True `AsyncLLM` from `vllm.v1.engine.async_llm`
   - Impact: Can't pipeline requests at vLLM level

2. **Parallel Episode Execution**
   - Current: Episodes in a group process sequentially (main.py:382-398)
   - Missing: `asyncio.gather()` for parallel episode creation
   - Impact: Reward evaluation blocks each other

3. **Multi-turn / Tool Calling**
   - Missing: Turn loop in rollout
   - Missing: Message history tracking
   - Missing: Tool execution logic
   - Impact: Can't do multi-step reasoning tasks

4. **Response Masking**
   - Missing: Masks to exclude tool results from training
   - Impact: Would train on environment outputs (bad!)

### Quick Performance Wins (1-2 Days Implementation)

**Impact**: 8-12x speedup on rollout collection
**Effort**: Low (refactor existing code)
**Risk**: Very low

#### 1. Parallel Episode Processing

**Current Bottleneck** (`apps/grpo/main.py:382-398`):
```python
for i, response in enumerate(responses):
    episode.reward = await reward_actor.evaluate_response.route(...)  # Sequential!
```

**Fix**: Use `asyncio.gather()`
```python
# Create episodes in parallel
episode_tasks = [
    create_episode_async(response, prompt, target, ...)
    for response in responses
]
results = await asyncio.gather(*episode_tasks)
```

**Speedup**: `group_size`x on reward evaluation (8x if `group_size=8`)

**Complete Implementation**:
```python
async def create_episode_async(
    i: int,
    response: Completion,
    prompt: str,
    target: str,
    pad_id: int,
    max_req_tokens: int,
    max_res_tokens: int,
    reward_actor: Any,
) -> tuple[Episode, torch.Tensor]:
    """Create one episode with async reward evaluation."""
    import uuid

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        pad_id=pad_id,
        request_len=max_req_tokens,
        response_len=max_res_tokens,
        target=target,
        completion=response,
    )

    # Async reward evaluation (doesn't block other episodes!)
    episode.reward = await reward_actor.evaluate_response.route(
        prompt=prompt, response=response.text, target=target
    )

    # Build input_ids row for reference model
    input_ids_row = torch.ones(max_req_tokens + max_res_tokens, dtype=torch.long)
    input_ids_row[:max_req_tokens] = episode.request_tensor
    input_ids_row[max_req_tokens:] = episode.response_tensor

    return episode, input_ids_row
```

#### 2. Parallel Prompt Groups

**Current**: Process one prompt at a time
```python
sample = await dataloader.sample.call_one()
responses = await policy.generate.route(prompt)  # Then next prompt
```

**Fix**: Batch multiple prompts
```python
# Sample multiple prompts at once
samples = await asyncio.gather(*[
    dataloader.sample.call_one()
    for _ in range(concurrent_prompts)
])

# Process all prompts concurrently
prompt_tasks = [
    process_single_prompt_group(sample, ...)
    for sample in samples
]
episode_counts = await asyncio.gather(*prompt_tasks)
```

**Speedup**: ~4x if processing 4 prompts in parallel

**Expected Combined Speedup**: 8x (parallel episodes) × 4x (parallel prompts) = **32x total**

### What vLLM Flags You Can Use NOW

**✅ Supported (No Code Changes)**

Add these to `EngineArgs` in your config:

```yaml
# apps/grpo/qwen3_1_7b.yaml
policy:
  engine_args:
    model: "Qwen/Qwen3-1.7B"
    # Tool calling support (PRIME-RL pattern)
    enable_auto_tool_choice: true
    tool_call_parser: "hermes"  # or "mistral", "llama", "internlm"

    # Standard vLLM performance flags
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 4096
    enable_prefix_caching: true  # Helps with multi-turn!
```

**Impact**:
- `enable_auto_tool_choice`: vLLM parses tool calls natively (no regex needed)
- `tool_call_parser`: Specifies format (model-dependent)
- `enable_prefix_caching`: Caches prompt prefixes (useful for multi-turn)

**❌ NOT Supported (Requires Refactor)**

```python
# This requires AsyncLLM class (Phase 3.1):
async_engine: true  # ❌ Your Generator uses synchronous Scheduler
```

### Recommended Implementation Roadmap

#### Week 1: Quick Wins
1. ✅ Implement parallel episode processing (`asyncio.gather` for rewards)
2. ✅ Implement parallel prompt groups (process 4 prompts at once)
3. ✅ Add metrics to measure speedup
4. 🎯 **Target**: 8-32x speedup on rollout collection

#### Weeks 2-3: Multi-turn Foundation
5. ✅ Add multi-turn loop (TRL pattern, token concatenation)
6. ✅ Add simple tool calling (text parsing, function map)
7. ✅ Add response masking (don't train on tool results)
8. ✅ Test on simple tool task (e.g., calculator)
9. 🎯 **Target**: Working tool calling RL

#### Weeks 4-6: Production Multi-turn
10. ✅ Add vLLM native tool calling (`enable_auto_tool_choice`)
11. ✅ Add message history management (explicit list)
12. ✅ Add per-sample async tasks (NeMo-RL pattern)
13. ✅ Benchmark on Tau-bench or similar
14. 🎯 **Target**: Production-ready tool calling

#### Future: Advanced Optimization (If Needed)
15. ⚠️ Refactor Generator to use AsyncLLM (if bottlenecked)
16. ⚠️ Add sample-level pipelining (if tool latency is high)
17. 🎯 **Target**: Maximum throughput

### Comparison: Forge vs Other Libraries

| Feature | Forge (Current) | After Quick Wins | After Multi-turn | NeMo-RL | PRIME-RL |
|---------|----------------|------------------|------------------|---------|----------|
| **Async Generation** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Parallel Episodes** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Parallel Prompts** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Multi-turn** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Tool Calling** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Response Masking** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **vLLM Native Tools** | ❌ | ❌ | Optional | ❌ | ✅ |
| **vLLM AsyncLLM** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Per-Sample Pipeline** | ❌ | ❌ | ❌ | ✅ | ❌ |

### Risk Assessment

**Low Risk ✅**
- **Parallel episodes**: Just refactoring existing code
- **Parallel prompts**: Uses existing async API
- **Multi-turn loop**: Additive, doesn't change existing flow
- **Response masking**: Just modifies loss function

**Medium Risk ⚠️**
- **vLLM native tools**: Depends on model support
- **Per-sample tasks**: Changes concurrency model

**High Risk 🔴**
- **AsyncLLM refactor**: Major architectural change
- Recommendation: **Only do this if Quick Wins + Multi-turn aren't enough!**

### Expected Performance Gains

**Quick Wins (Week 1)**
- Baseline: 1 prompt with group_size=8 takes ~1 second
- Parallel episodes: 800ms → 100ms per group (8x)
- Parallel prompts: Process 4 groups in 100ms instead of 400ms (4x)
- **Total speedup**: ~32x on rollout collection

**Multi-turn (Weeks 2-6)**
- Baseline: Multi-turn with 3 turns, 2 tools per episode
- Without optimization: Sequential turns, sequential tool calls
- With async tools: Parallel tool execution (~2x)
- With per-sample tasks: While Sample 1 waits, Sample 2 generates (~1.5x)
- **Total speedup**: ~3x additional (96x total from baseline)

**AsyncLLM (Future)**
- Baseline: vLLM generation throughput
- Current: Synchronous scheduler
- AsyncLLM: Request pipelining at vLLM level
- **Additional speedup**: ~2x (if generation-bound)

### Next Steps

1. **Implement** Quick Wins (parallel episodes + parallel prompts)
2. **Test** speedup on your current GSM8K setup
3. **Measure** with existing metrics
4. **Add** multi-turn loop following TRL/BlackJack patterns above
5. **Avoid** AsyncLLM refactor unless absolutely necessary (high risk!)

### Key Questions to Answer Before Implementing

**What's your bottleneck?**
- If rollout collection: Quick Wins are enough
- If you need tool calling: Multi-turn required
- If generation-bound: Consider AsyncLLM (risky!)

**What tasks are you targeting?**
- Single-turn (math, coding): Quick Wins only
- Multi-turn reasoning: Multi-turn required
- Complex tool workflows: Multi-turn + async tools

**What's your timeline?**
- Need results this week: Quick Wins
- Research project (1-2 months): Multi-turn
- Production system: Multi-turn, consider AsyncLLM

**What's your risk tolerance?**
- Low: Quick Wins + Multi-turn (Phases 1-2)
- Medium: Full Multi-turn + vLLM native tools
- High: AsyncLLM refactor (only if truly needed!)

---

## Handling Multiple Environments (e.g., WebSearch + Coding)

This section addresses the question: **What happens if you have multiple environments/domains (e.g., websearch AND coding tasks)?**

Research conducted across all major frameworks: **Tinker-Cookbook (Meta)**, **Verifiers (Prime Intellect)**, **VERL**, and **NeMo-RL (Thinking Machines)**.

---

### 1. Tinker-Cookbook: `CompositeDataset` Pattern ⭐ RECOMMENDED

**Location**: `tinker-cookbook/distillation/datasets.py:45-84`

Tinker uses a **`CompositeDataset`** that mixes multiple `RLDataset`s at the batch level.

#### Core Abstraction: `EnvGroupBuilder`

Every environment implements this interface:

```python
# tinker_cookbook/rl/types.py:64-108

class EnvGroupBuilder(ABC):
    """
    Builds a group of environments. Can be used for:
    - Multi-agent environments
    - GRPO groups (e.g., 8 copies for one problem)
    """

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments (e.g., 8 copies for GRPO)"""
        pass

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute final reward looking at whole group (optional)"""
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        """Tags for logging (e.g., ['gsm8k'], ['websearch'])"""
        return []
```

**Example: Math Environment**
```python
# tinker_cookbook/recipes/math_rl/math_env.py

class Gsm8kDataset(RLDataset):
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        return [
            ProblemGroupBuilder(
                env_thunk=partial(MathEnv, problem, answer, self.renderer),
                num_envs=group_size,  # e.g., 8 for GRPO
                dataset_name="gsm8k"
            )
            for row in self.ds.select(range(batch_start, batch_end))
        ]
```

#### Mixing Multiple Environments: `CompositeDataset`

```python
# tinker_cookbook/distillation/datasets.py:45-84

class CompositeDataset:
    """Wraps multiple datasets and samples from each according to their groups_per_batch."""

    def __init__(self, datasets: List[RLDataset], groups_per_batch_list: List[int]):
        self.datasets = datasets
        self.groups_per_batch_list = groups_per_batch_list
        self.length = min(len(dataset) for dataset in datasets)

    def get_batch(self, i_batch: int) -> tuple[List[EnvGroupBuilder], List[int]]:
        """
        Get a batch by sampling from each dataset.

        Returns:
            env_group_builders: List of all env group builders (mixed!)
            dataset_indices: Which dataset each builder came from
        """
        all_env_group_builders = []
        all_dataset_indices = []

        for dataset_idx, (dataset, groups_per_batch) in enumerate(
            zip(self.datasets, self.groups_per_batch_list)
        ):
            env_group_builders = dataset.get_batch(i_batch)
            all_env_group_builders.extend(env_group_builders)
            all_dataset_indices.extend([dataset_idx] * groups_per_batch)

        return all_env_group_builders, all_dataset_indices
```

#### How Training Works with Mixed Environments

```python
# tinker_cookbook/rl/train.py:357

# Training loop
for i_batch in range(num_batches):
    # Get batch of EnvGroupBuilders (could be from different envs!)
    env_group_builders_P = dataset.get_batch(i_batch)

    # Rollout each group asynchronously
    for builder in env_group_builders_P:
        trajectory_group = await do_group_rollout(
            sampling_client,
            builder,  # Each builder knows its own env type!
            max_tokens=cfg.max_tokens,
        )

        # Training data assembly
        # Each trajectory_group has its own reward/metrics
        # Logging uses builder.logging_tags() to separate metrics
```

**Key insight:** Each `EnvGroupBuilder` is self-contained:
- Knows how to create its environments
- Knows how to compute rewards
- Has its own logging tags

#### Concrete Example: Mixing WebSearch and Coding

```python
from tinker_cookbook.rl.types import RLDataset, EnvGroupBuilder
from tinker_cookbook.distillation.datasets import CompositeDataset

# 1. Define WebSearch dataset
class WebSearchDataset(RLDataset):
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            ToolUseGroupBuilder(
                env_thunk=partial(
                    SearchEnv,
                    problem=row["query"],
                    answer=row["answer"],
                    tool_client=search_tool_client,  # search_pages, view_sections
                    renderer=renderer,
                ),
                num_envs=8,
                dataset_name="websearch"
            )
            for row in self.ds.select(batch_indices)
        ]

# 2. Define Coding dataset
class CodingDataset(RLDataset):
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            ToolUseGroupBuilder(
                env_thunk=partial(
                    CodeEnv,
                    problem=row["task"],
                    test_cases=row["tests"],
                    tool_client=code_tool_client,  # execute_code, debug
                    renderer=renderer,
                ),
                num_envs=8,
                dataset_name="coding"
            )
            for row in self.ds.select(batch_indices)
        ]

# 3. Mix them with CompositeDataset
mixed_dataset = CompositeDataset(
    datasets=[
        WebSearchDataset(...),
        CodingDataset(...),
    ],
    groups_per_batch_list=[
        50,  # 50 websearch groups per batch
        50,  # 50 coding groups per batch
    ]
)

# 4. Use in training
for i_batch in range(num_batches):
    env_group_builders, dataset_indices = mixed_dataset.get_batch(i_batch)
    # env_group_builders has 100 items: 50 websearch + 50 coding
    # Each knows its own tools, max_turns, reward function!
```

**Why this works:**
- ✅ **Batch-level mixing**: Each batch contains groups from multiple datasets
- ✅ **Decentralized**: Each `EnvGroupBuilder` is independent
- ✅ **Flexibility**: Control exact ratio per batch (`groups_per_batch_list=[50, 50]`)
- ✅ **Logging**: Each builder has its own tags for separate metrics

---

### 2. Verifiers (Prime Intellect): `EnvGroup` Pattern

**Location**: `verifiers/verifiers/envs/env_group.py`

Verifiers has an **`EnvGroup`** class specifically designed for mixing environments:

```python
# verifiers/verifiers/envs/env_group.py

class EnvGroup(Environment):
    """Environment group that acts as a mixture of multiple environments."""

    def __init__(self, envs: list[Environment], env_names: list[str] | None = None):
        self.envs = envs
        self.env_names = env_names or [f"env_{i}" for i in range(len(envs))]

        # Create mapping for quick lookup
        self.env_map = {name: env for name, env in zip(self.env_names, self.envs)}

        # Concatenate datasets with task labels
        for env, name in zip(self.envs, self.env_names):
            env_dataset = env.get_dataset().map(lambda x: {**x, "task": name})
            datasets.append(env_dataset)

        # Combine all datasets
        dataset = concatenate_datasets(datasets)
```

#### How EnvGroup Routes to Environments

```python
async def rollout(self, client, model, prompt, task, ...):
    # Route to appropriate environment based on task
    env = self.env_map[task]

    # Set tools for this task's environment
    if hasattr(env, "oai_tools") and env.oai_tools:
        info["oai_tools"] = env.oai_tools  # Different tools per env!

    # Execute rollout with task-specific environment
    completion, state = await env.rollout(client, model, prompt, ...)
```

#### Example Usage

```python
# Define environments
websearch_env = vf.ToolEnv(
    dataset=websearch_dataset,
    tools=[search_pages, view_sections],  # Web search tools
    max_turns=10
)

coding_env = vf.ToolEnv(
    dataset=coding_dataset,
    tools=[execute_code, debug_code],  # Coding tools
    max_turns=15
)

# Combine into EnvGroup
env = EnvGroup(
    envs=[websearch_env, coding_env],
    env_names=["websearch", "coding"]
)

# Training: samples automatically routed to correct environment
generate_outputs = await env.generate(
    inputs=mixed_dataset,  # Has both websearch and coding samples
    client=client,
    model=model_name
)
```

**How it works:**
1. Each sample gets a `task` field (e.g., `"websearch"` or `"coding"`)
2. `EnvGroup.rollout()` routes to appropriate environment based on task
3. Different tools, max_turns, reward functions per environment

**Key advantages:**
- ✅ **Sample-level routing**: Automatic based on task field
- ✅ **Centralized**: `EnvGroup` owns all sub-environments
- ✅ **Simpler API**: Just pass task name, routing is automatic
- ✅ **Different configurations**: Each environment has its own tools, max_turns, rubric

---

### 3. VERL: Separate Config Files (Manual Approach)

**Location**: `verl/examples/sglang_multiturn/config/tool_config/`

VERL uses **separate YAML config files** for different tool sets:

```yaml
# gsm8k_tool_config.yaml
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        parameters: {...}

# sandbox_fusion_tool_config.yaml  (for coding)
tools:
  - class_name: "verl.tools.sandbox_fusion_tools.SandboxFusionTool"
    config:
      sandbox_fusion_url: "..."
    tool_schema:
      type: "function"
      function:
        name: "code_interpreter"
        parameters: {...}
```

**How they handle multiple environments:**
- **Option A**: Run separate training jobs with different configs
  ```bash
  # Job 1: Math with calculator tool
  python main.py --tool_config gsm8k_tool_config.yaml

  # Job 2: Coding with sandbox tool
  python main.py --tool_config sandbox_fusion_tool_config.yaml
  ```

- **Option B**: Load tools dynamically based on task (manual implementation)

**Limitation:** Not designed for mixed datasets out-of-the-box.

---

### 4. NeMo-RL (Thinking Machines): Environment Registry

**Location**: `RL/nemo_rl/distributed/ray_actor_environment_registry.py`

NeMo-RL has an **`ACTOR_ENVIRONMENT_REGISTRY`** but it's for Python environments, not task routing:

```python
ACTOR_ENVIRONMENT_REGISTRY: dict[str, str] = {
    "nemo_rl.environments.math_environment.MathEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.code_environment.CodeEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.vlm_environment.VLMEnvironment": PY_EXECUTABLES.SYSTEM,
    ...
}
```

**This is different:** It maps environment classes to Python virtual environments (for dependency isolation), not for routing during training.

**How to handle multiple environments in NeMo-RL:**
```python
# In your config/code, you'd specify which environment to use
task_to_env = {
    "websearch": WebSearchEnvironment(...),
    "coding": CodeEnvironment(...),
}

# In rollout loop:
env = task_to_env[sample["task_type"]]
result = await env.step(action)
```

**Similar to Verifiers' approach but manual.**

---

### Framework Comparison Table

| Framework | Multi-Env Support | Routing Method | Tools Per Env | Best For |
|-----------|-------------------|----------------|---------------|----------|
| **Tinker (Meta)** | ✅ Built-in `CompositeDataset` | Batch-level mixing | ✅ Different tools | **Production multi-env** |
| **Verifiers (Prime)** | ✅ Built-in `EnvGroup` | `task` field in dataset | ✅ Different tools | **Production multi-env** |
| **VERL** | ⚠️ Manual | Separate configs | Config-based | Single env per job |
| **NeMo-RL** | ⚠️ Manual | Dict lookup | Code-based | Custom routing logic |

---

### Recommendation for Forge + Tau2Bench

**Use Tinker's `CompositeDataset` pattern** (most flexible for your use case):

```python
# 1. Define your environments
from tinker_cookbook.rl.types import RLDataset, EnvGroupBuilder
from tinker_cookbook.distillation.datasets import CompositeDataset

websearch_env_builder = ToolUseGroupBuilder(
    env_thunk=partial(WebSearchEnv, tools=[search_wiki, view_page], max_turns=10),
    num_envs=8,
    dataset_name="websearch"
)

coding_env_builder = ToolUseGroupBuilder(
    env_thunk=partial(CodingEnv, tools=[execute_python, execute_bash], max_turns=15),
    num_envs=8,
    dataset_name="coding"
)

# 2. Create datasets
websearch_dataset = Tau2BenchDataset(domain="websearch", builders=[websearch_env_builder])
coding_dataset = Tau2BenchDataset(domain="coding", builders=[coding_env_builder])

# 3. Combine into CompositeDataset
mixed_dataset = CompositeDataset(
    datasets=[websearch_dataset, coding_dataset],
    groups_per_batch_list=[50, 50]  # 50 websearch + 50 coding per batch
)

# 4. Use in Forge rollout
async def continuous_rollouts():
    while True:
        # Get mixed batch
        env_group_builders, dataset_indices = mixed_dataset.get_batch(batch_idx)

        # Each builder knows its own environment type!
        for builder in env_group_builders:
            episodes = await play_task_with_env_builder(
                policy=policy,
                env_builder=builder,  # Handles routing internally
            )
```

**Why this works:**
- ✅ **Different tools** per environment (websearch vs coding)
- ✅ **Different max_turns** per environment
- ✅ **Different rewards** per environment
- ✅ **Unified training loop** (no special casing needed)
- ✅ **Separate metrics** (via logging_tags)
- ✅ **Flexible mixing ratios** (control via groups_per_batch_list)

**Alternative (simpler but less flexible):**
Implement simple routing yourself:
```python
task_to_env = {
    "websearch": websearch_env,
    "coding": coding_env,
}

async def play_task(task_sample, policy, tokenizer):
    env = task_to_env[task_sample["task_type"]]
    # Use env-specific tools and max_turns
    ...
```

---

### Summary

**Best patterns for handling multiple environments:**

1. **Tinker's `CompositeDataset`**: Batch-level mixing, decentralized, flexible ratios
2. **Verifiers' `EnvGroup`**: Sample-level routing, centralized, automatic
3. **Manual routing**: Simple dict lookup, full control

**For Forge + Tau2Bench:** Start with Tinker's pattern for maximum flexibility, or implement simple dict-based routing if you want to keep it simple.
