# Tutorial: Multi-turn + Tool Calling in Forge for Tau2Bench

**Goal:** This document teaches you the fundamentals of multi-turn and tool calling, shows concrete examples from Tau2Bench, explains how to implement it in Forge with OpenEnv, and provides a clear implementation plan.

**For:** Junior developers new to RL and the Forge codebase

**Status:** Tutorial + Planning Document

---

## Table of Contents

1. [Part 1: The Fundamentals](#part-1-the-fundamentals)
2. [Part 2: Tau2Bench Deep Dive](#part-2-tau2bench-deep-dive)
3. [Part 3: How Forge Currently Works](#part-3-how-forge-currently-works)
4. [Part 4: How Other Libraries Do It](#part-4-how-other-libraries-do-it)
5. [Part 5: Implementation Plan for Forge](#part-5-implementation-plan-for-forge)
6. [Part 6: Performance & Async Patterns](#part-6-performance--async-patterns)
7. [Part 7: What's Already Supported vs What Needs to Be Added](#part-7-whats-already-supported-vs-what-needs-to-be-added)


## Part 2: Tau2Bench Deep Dive

### What is Tau2Bench?

Tau2Bench is a **benchmark** for evaluating conversational agents in customer service scenarios. It tests if agents can:

1. **Follow policies** (domain-specific rules)
2. **Use tools correctly** (call the right functions with right arguments)
3. **Communicate well** (talk to users naturally)
4. **Complete tasks** (achieve the goal)

**Key Insight:** Tau2 is ONLY for evaluation. We'll train a model on different dataset and then evaluate on Tau2.

---

### Tau2 Task Structure

**📁 Code Reference:** `tau2-bench/data/tau2/domains/mock/tasks.json:1-28`

Here's a complete task from the `mock` domain:

```json
{
  "id": "create_task_1",
  "description": {
    "purpose": "Test the create_task functionality",
    "notes": "Basic task creation test with a simple title"
  },
  "user_scenario": {
    "persona": "Professional and direct communicator",
    "instructions": "Create a new task called 'Important Meeting' for user_1."
  },
  "ticket": "User needs to create a task for an upcoming meeting. Create a new task called 'Important Meeting' for user_1.",
  "evaluation_criteria": {
    "actions": [
      {
        "action_id": "create_1",
        "name": "create_task",
        "arguments": {
          "user_id": "user_1",
          "title": "Important Meeting"
        },
        "info": "Create a new task for the meeting"
      }
    ],
    "nl_assertions": [
      "The agent confirmed the task was created successfully"
    ]
  }
}
```

**Key insight:** Evaluation is done by checking if expected tools were called and by having another LLM confirm that the task was created successfully.

---

### Tau2 Available Tools (Mock Domain)

**📁 Code Reference:** `tau2-bench/src/tau2/domains/mock/tools.py`

The `mock` domain has these tools:

```python
# Tool 1: Create a task
create_task(user_id: str, title: str, description: str = None) -> Task

# Tool 2: Get all users
get_users() -> list[User]

# Tool 3: Update task status
update_task_status(task_id: str, status: str) -> Task
# status can be "pending" or "completed"

# Tool 4: Transfer to human agent
transfer_to_human_agents(summary: str) -> str
```

**Other domains have different tools:**

- `airline` - Search flights, book tickets, cancel bookings, etc.
- `retail` - Product search, orders, returns, refunds
- `telecom` - Account management, troubleshooting, plan changes

---

### Example Multi-turn Interaction on Tau2

**Task:** Create a task and mark it as completed

**Full Conversation:**

```
[Turn 1 - User]
"Hi! I need to create a task called 'Team Standup' for user_1 and then mark it as completed."

[Turn 2 - Assistant]
<calls create_task(user_id="user_1", title="Team Standup")>

[Turn 3 - Tool Result]
{"task_id": "task_2", "title": "Team Standup", "status": "pending"}

[Turn 4 - Assistant]
"I've created the task 'Team Standup'. The task ID is task_2. Let me mark it as completed now."

[Turn 5 - Assistant]
<calls update_task_status(task_id="task_2", status="completed")>

[Turn 6 - Tool Result]
{"task_id": "task_2", "title": "Team Standup", "status": "completed"}

[Turn 7 - Assistant]
"Done! Task 'Team Standup' (task_2) is now marked as completed."

[Turn 8 - User]
"Thanks!"

[Turn 9 - Assistant]
<calls done()>  # Special tool to signal completion
```

**Episode ends when:**
- Agent calls `done()` tool
- User says stop keywords (like "bye", "thanks")
- Max turns reached

---

### How Tau2 Scores Episodes

Tau2 evaluates based on multiple criteria:

**1. ACTION Criteria** - Did the agent call the right tools with right arguments?

```python
"evaluation_criteria": {
  "actions": [
    {
      "name": "create_task",
      "arguments": {
        "user_id": "user_1",
        "title": "Important Meeting"
      }
    }
  ]
}

# Scoring: Agent must have called create_task with these arguments (order doesn't matter)
```

**2. ENV Criteria** - Is the database/environment state correct?

```python
"env_assertions": [
  {
    "func_name": "assert_task_status",
    "arguments": {"task_id": "task_2", "expected_status": "completed"}
  }
]

# Scoring: After episode, task_2 must have status="completed"
```

**3. NL_ASSERTIONS Criteria** - Did the agent communicate properly?

```python
"nl_assertions": [
  "The agent confirmed the task was created successfully"
]

# Scoring: LLM judges if this assertion is true based on conversation
```

**Final Score:**

```python
# Each criterion returns 0.0 or 1.0
action_score = 1.0 if all_actions_correct else 0.0
env_score = 1.0 if all_env_assertions_pass else 0.0
nl_score = 1.0 if all_nl_assertions_pass else 0.0

# Final reward is the product (all must pass!)
final_reward = action_score * env_score * nl_score
```

---

### Tau2 Modes

**1. Normal Mode** - Agent talks to user simulator

```
Agent ←→ User Simulator (another LLM)
  ↓
Environment (executes tools, tracks state)
```

**2. Solo Mode** - Agent works alone on a ticket

```
Agent gets ticket description
  ↓
Agent uses tools to complete task
  ↓
No user interaction
```

**For training:** Solo mode is simpler. Normal mode requires user simulation.
**For evaluatoin:** Both modes are valid in the leaderboard. Using an agent is more challenging and usually has lower score: https://taubench.com/#leaderboard


---

## Part 1: The Fundamentals

### What is Tool Calling?

**Tool calling** is when a language model can invoke external functions/APIs instead of just generating text.

**Simple Example:**

```
User: "What's the weather in NYC?"

WITHOUT tool calling:
Model: "I don't have access to real-time weather data..."

WITH tool calling:
Model: <tool_call>get_weather(location="NYC")</tool_call> # this gets parsed and executed
System: Returns "72°F, sunny"
Model: "It's 72°F and sunny in NYC!"
```

**Tool Definition Example (from Tau2 Mock domain):**

**📁 Code Reference:** `tau2-bench/src/tau2/domains/mock/tools.py:14-40`

```python
def create_task(user_id: str, title: str, description: str = None) -> Task:
    """
    Create a new task for a user.

    Args:
        user_id: The ID of the user creating the task
        title: The title of the task
        description: Optional description of the task

    Returns:
        The created task
    """
    task_id = f"task_{len(db.tasks) + 1}"
    task = Task(task_id=task_id, title=title, description=description, status="pending")
    db.tasks[task_id] = task
    return task
```

The tool description can be converted to an OpenAI-style tool schema and displayed in the system prompt, so models know which tools are available and how to call them:

```json
{
  "type": "function",
  "function": {
    "name": "create_task",
    "description": "Create a new task for a user.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": {"type": "string", "description": "The ID of the user creating the task"},
        "title": {"type": "string", "description": "The title of the task"},
        "description": {"type": "string", "description": "Optional description of the task"}
      },
      "required": ["user_id", "title"]
    }
  }
}
```

---

### What is Multi-turn?

**Multi-turn** means a conversation or interaction that spans multiple back-and-forth exchanges (turns).

**Visual Comparison:**

```
SINGLE-TURN (Current Forge GRPO):
┌─────────────┐
│ User Prompt │ → Model generates response → Episode ends
└─────────────┘

MULTI-TURN (What we need):
┌─────────────┐
│ User Prompt │ → Model response → Tool execution → Model response → Tool execution → ... → Done
└─────────────┘
     Turn 1          Turn 2             Turn 3          Turn 4             Turn 5
```

**NOTE**: Tau2bench ha a "SOLO" mode, as described above, where the agent interacts with the system by calling tools until the task is completed. Another mode, with solo=False, an LLM can act as an user. In their benchmark, results can be posted in both ways. For our implementation, I suggest we use solo=True. Leaderboard link: https://taubench.com/#leaderboard

**Concrete Example:**
```
Turn 1:
  User: "Create a task called 'Important Meeting' for user_1"

Turn 2:
  Assistant: <calls create_task(user_id="user_1", title="Important Meeting")>

Turn 3:
  System (Tool): Returns Task(task_id="task_2", title="Important Meeting", status="pending")

Turn 4:
  Assistant: "I've created the task 'Important Meeting' for you."

Turn 5:
  User: "Great! Now mark it as completed."

Turn 6:
  Assistant: <calls update_task_status(task_id="task_2", status="completed")>

Turn 7:
  System (Tool): Returns Task(task_id="task_2", title="Important Meeting", status="completed")

Turn 8:
  Assistant: "Done! Task_2 is now marked as completed."
```

**Key Insight:** Each turn builds on the conversation history. The model needs to see all previous turns to understand context.

---

### Message Format (OpenAI Standard)

Multi-turn conversations are represented as a list of messages:

```python
messages = [
    {"role": "system", "content": "You are a helpful task management assistant."},
    {"role": "user", "content": "Create a task called 'Important Meeting' for user_1"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "create_task",
                "arguments": '{"user_id": "user_1", "title": "Important Meeting"}'
            }
        }]
    },
    {
        "role": "tool",
        "content": '{"task_id": "task_2", "title": "Important Meeting", "status": "pending"}',
        "tool_call_id": "call_123"
    },
    {
        "role": "assistant",
        "content": "I've created the task 'Important Meeting' for you. It's task_2."
    }
]
```

**Message Roles:**
- `system` - Instructions for the model
- `user` - Human input
- `assistant` - Model's response (can be text or tool calls)
- `tool` - Result from tool execution

---

### Two Approaches to Tool Calling

**Approach 1: Native Function Calling (vLLM, OpenAI)**

The model is trained to output structured tool calls:

```python
# Model output is automatically parsed
response = {
    "content": None,
    "tool_calls": [{
        "function": {
            "name": "create_task",
            "arguments": '{"user_id": "user_1", "title": "Meeting"}'
        }
    }]
}
```
---

**Approach 2: Text-Based Parsing (BlackJack pattern)**

The model outputs text, and you parse it:

```python
# Model output is plain text
response_text = "create_task(user_id='user_1', title='Meeting')"

# You parse it
import re
match = re.search(r'(\w+)\((.*)\)', response_text)
if match:
    function_name = match.group(1)
    # Parse arguments...
```


---

## Part 3: How Forge Currently Works

### Current Forge GRPO Flow (GSM8K Example)

Forge currently does **single-turn** training on math problems:

```python
# apps/grpo/main.py - Simplified

# 1. Sample a math problem
prompt = "What is 25 * 4?"
target = "100"

# 2. Generate G responses using vllm
responses = await policy.generate(prompt, num_responses=G)  # G=8 typically
# responses = ["100", "100", "99", "100", "100", "101", "100", "100"]

# 3. Score each response
rewards = []
for response in responses:
    reward = 1.0 if extract_answer(response) == target else 0.0
    rewards.append(reward)
# rewards = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]

# 4. Get reference logprobs (for KL penalty)
ref_logprobs = await ref_model.forward(prompt, responses)

# 5. Compute advantages (group-relative), i.e. z-score normalized
# so we reward better answers and penalize worse ones
advantages = []
for i, reward in enumerate(rewards):
    advantage = reward - mean(rewards)  # Group-relative
    advantages.append(advantage)
# advantages = [0.125, 0.125, -0.875, 0.125, 0.125, -0.875, 0.125, 0.125]

# 6. Create episodes
episodes = []
for i in range(G):
    episode = Episode(
        prompt=prompt,
        response=responses[i],
        reward=rewards[i],
        advantage=advantages[i],
        ref_logprobs=ref_logprobs[i]
    )
    episodes.append(episode)

# 7. Add to replay buffer
await replay_buffer.add(episodes)

# 8. Training loop samples from buffer and trains
batch = await replay_buffer.sample(batch_size=32)
loss = grpo_loss(batch)
trainer.train_step(loss)
```

**Summary:**

We currently have: Single prompt → single response
[ ] No multi-turn support
[ ] No tool calling

---

### What Forge is Missing for Tool Calling

**Missing Pieces:**

1. **Tool Definition System**
   [ ] Need to define available tools
   [ ] Convert to OpenAI schema format
   [ ] Pass to vLLM during generation

2. **Response Parsing**
   [ ] Detect if response contains tool calls
   [ ] Parse tool name and arguments
   [ ] Handle both text format and native function calling

3. **Multi-turn Loop**
   [ ] Keep conversation history
   [ ] Execute tool calls
   [ ] Add tool results to history
   [ ] Continue generating until done

4. **Episode Structure for Multi-turn**
   [ ] Track which tokens are LLM-generated vs tool results
   [ ] Response mask (train only on LLM tokens, not tool results)
   [ ] Multiple turns per episode

5. **Environment Integration**
   [ ] Connect to OpenEnv (or other environment)
   [ ] Execute tool calls in sandboxed environment
   [ ] Get rewards from environment

---

## Part 4: How Other Libraries Do It

### Pattern 1: OpenEnv BlackJack (Simplest, Proven with Forge)

**📁 Code Reference:** `OpenEnv/examples/grpo_blackjack/grpo_utils.py` (search for `async def play_game`)

```python
async def play_game(game_id, server_url, policy, tokenizer):
    """Play a full BlackJack game, returning all steps."""

    # 1. Initialize environment
    env = OpenSpielEnv(base_url=server_url)
    result = env.reset()  # Start game

    # 2. Game loop
    step_num = 0
    action_history = []
    game_steps = []
    done = False

    while not done and step_num < MAX_STEPS:
        # 3. Format prompt with game state
        prompt = format_prompt(step_num, action_history, tokenizer)

        # 4. Generate response
        response = await policy.generate(prompt)

        # 5. Parse action from text
        action_id = parse_action(response.text, obs.legal_actions)
        # response.text might be "HIT" or "I choose to STAND"
        # parse_action extracts: 0 (HIT) or 1 (STAND)

        # 6. Store step data
        game_steps.append({
            "step_num": step_num,
            "prompt": prompt,
            "response": response,
        })

        # 7. Execute action in environment
        result = env.step(OpenSpielAction(action_id=action_id))
        obs = result.observation
        done = result.done

        action_history.append((action_id, "HIT" if action_id == 0 else "STAND"))
        step_num += 1

    # 8. Get final reward
    final_reward = result.reward  # +1 (win), -1 (loss), 0 (push)

    # 9. Assign final reward to ALL steps
    all_step_results = []
    for step_data in game_steps:
        all_step_results.append({
            "game_id": game_id,
            "final_reward": final_reward,
            **step_data,
        })

    return all_step_results
```

**Prompt Formatting:**

**📁 Code Reference:** `OpenEnv/examples/grpo_blackjack/grpo_utils.py`

```python
def format_prompt(step_num: int, action_history: list, tokenizer) -> str:
    system = "You are an expert BlackJack player. Output only 'HIT' or 'STAND'."

    state_desc = f"=== BlackJack Game (Step {step_num + 1}) ===\n\n"

    # Include previous actions in prompt
    if action_history:
        state_desc += "Previous actions:\n"
        for i, (_, name) in enumerate(action_history):
            state_desc += f"  {i + 1}. {name}\n"
        state_desc += "\n"

    state_desc += "What do you do? (Output only 'HIT' or 'STAND')"

    # Use chat template
    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": state_desc},
    ]

    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
```

**Action Parsing:**

**📁 Code Reference:** `OpenEnv/examples/grpo_blackjack/grpo_utils.py:205-229`

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

**Episode Creation:**

**📁 Code Reference:** `OpenEnv/examples/grpo_blackjack/grpo_utils.py` (in `continuous_rollouts` function)

```python
# In continuous_rollouts:

# Play {group_size} games
for game_idx in range(group_size):
    game_id = str(uuid.uuid4())[:8]
    step_results = await play_game(game_id, server_url, policy, tokenizer)
    all_step_results.extend(step_results)

# Create one episode PER STEP
episodes = []
for step_result in all_step_results:
    episode = Episode(
        episode_id=str(uuid.uuid4()),
        game_id=step_result["game_id"],
        step_in_game=step_result["step_num"],
        completion=step_result["response"],
        # ... other fields
    )

    # Assign reward (final game reward for all steps)
    episode.reward = step_result["final_reward"]

    episodes.append(episode)
```

**Key Takeaways:**

✅ **Text parsing works** - No need for complex function calling
✅ **One episode per step** - Each step in the game is a separate episode
✅ **Final reward for all steps** - Sparse reward assigned to entire trajectory
✅ **Action history in prompts** - Model sees what it did before
✅ **Simple, proven pattern** - This works with Forge today!

---

### Pattern 2: Verifiers ToolEnv (Production-Ready Tool Calling)

**Location:** `/home/felipemello/forge/verifiers/verifiers/envs/tool_env.py`

**Key Insight:** Clean API for tool calling with OpenAI-style function calling.

**Defining Tools:**

**📁 Code Reference:** See examples in `verifiers/environments/wiki_search/wiki_search.py:99-128`

```python
# Just write normal Python functions with type hints!
async def search_wiki(query: str) -> list[str]:
    """
    Search Wikipedia for relevant articles.

    Args:
        query: The search query string.

    Returns:
        List of article titles matching the query.
    """
    results = await wikipedia_api.search(query)
    return [article.title for article in results]

# Convert to OpenAI schema automatically
tool_schema = convert_func_to_oai_tool(search_wiki)
```

**Multi-turn Rollout Loop:**

**📁 Code Reference:** `verifiers/verifiers/envs/multiturn_env.py:55-149`

```python
# verifiers/envs/multiturn_env.py (simplified)

async def rollout(client, model, prompt, tools, max_turns=10):
    """Generate a multi-turn rollout with tools."""

    messages = [{"role": "user", "content": prompt}]
    turn = 0

    while turn < max_turns:
        # 1. Call LLM with tools
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,  # OpenAI tool schemas
        )

        # 2. Add assistant message
        assistant_msg = {
            "role": "assistant",
            "content": response.choices[0].message.content
        }

        # 3. Check for tool calls: append the tool calls -> execute -> append their results
        if response.choices[0].message.tool_calls:
            assistant_msg["tool_calls"] = [
                tc.model_dump() for tc in response.choices[0].message.tool_calls
            ]
            messages.append(assistant_msg)

            # 4. Execute tools
            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                result = await execute_tool(tool_name, tool_args)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call.id
                })
        else:
            # No tool calls, episode done
            messages.append(assistant_msg)
            break

        turn += 1

    return messages
```

**Tool Execution:**

**📁 Code Reference:** `verifiers/verifiers/envs/tool_env.py:43-89`

```python
class ToolEnv:
    def __init__(self, tools: list[Callable]):
        # Map function name to function
        self.tool_map = {tool.__name__: tool for tool in tools}

        # Convert to OpenAI schemas
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in tools]

    async def execute_tool(self, tool_name: str, arguments: dict):
        """Execute a tool and return the result."""
        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_func = self.tool_map[tool_name]
        result = await tool_func(**arguments)
        return result
```

**Key Takeaways:**

✅ **Simple tool definition** - Just type-hinted Python functions
✅ **OpenAI-compatible** - Uses standard OpenAI API format
✅ **Clean loop structure** - Easy to understand and modify
✅ **Automatic schema generation** - No manual JSON writing
✅ **Production-ready** - Used by PRIME-RL and others

---

### Pattern 3: VERL/NeMo-RL (Response Masking for Multi-turn)

**📁 Code References:**
- VERL: `verl/` repository (see `4_examples_APIs.md` for details)
- NeMo-RL: `RL/` repository (see `4_examples_APIs.md` for details)
- Verifiers: `verifiers/verifiers/utils/processing_utils.py` (has `process_env_results_vllm`)

**Key Insight:** When training on multi-turn with tools, you need to **mask out tool results** so the model only trains on its own generated tokens.

**Why Masking Matters:**

```
Conversation:
[User] "Search for AI"
[Assistant] <tool_call: search("AI")>     ← Train on this
[Tool] "Results: [AI article 1, 2, 3]"    ← DON'T train on this (not model output)
[Assistant] "I found 3 articles..."       ← Train on this

Response Mask:
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 ↑ LLM tokens  ↑ Tool result tokens     ↑ LLM tokens
```

**Response Mask Pattern:**

```python
# Building the response with mask
response_tokens = []
response_mask = []

# Turn 1: Assistant generates tool call
assistant_tokens = tokenize("<tool_call: search('AI')>")
response_tokens.extend(assistant_tokens)
response_mask.extend([1] * len(assistant_tokens))  # Train on these

# Turn 2: Tool result (not LLM output)
tool_result_tokens = tokenize("Results: [article 1, 2, 3]")
response_tokens.extend(tool_result_tokens)
response_mask.extend([0] * len(tool_result_tokens))  # DON'T train on these

# Turn 3: Assistant responds
assistant_tokens_2 = tokenize("I found 3 articles about AI...")
response_tokens.extend(assistant_tokens_2)
response_mask.extend([1] * len(assistant_tokens_2))  # Train on these

# In training:
loss = compute_loss(logits, response_tokens, response_mask)
# Only tokens with mask=1 contribute to loss
```

**Key Takeaways:**

✅ **Critical for multi-turn** - Prevents training on tool outputs
✅ **Simple concept** - Just track which tokens are LLM vs system
✅ **Used by all production systems** - VERL, NeMo-RL, Verifiers

---

### Pattern 4: Async vLLM for Pipelined Tool Calling (NeMo-RL)

**📁 Code References:**
- NeMo-RL: `RL/` (see `4_examples_APIs.md` lines 660-1190 for full details)
- Sample-level concurrency: `RL/.../rollouts.py:780-936`
- vLLM async worker: `RL/.../vllm_worker_async.py:496-714`

**Key Insight:** Use async/await pattern with sample-level concurrency so fast samples don't wait for slow ones.

**The Problem with Synchronous:**

```
Batch of 4 samples:
Sample 1: Gen[██████] → Tool[████] → Gen[████] → Done
Sample 2: Gen[████] → Tool[██] → Gen[██] → Done
Sample 3: Gen[██] → Done
Sample 4: Gen[████████] → Tool[██████] → Gen[██] → Done

Synchronous: Wait for ALL samples to finish each stage
Total time: Max(all samples) per stage
```

**Async Pattern:**

```python
async def run_rollout_batch(samples):
    # Create async task for each sample
    tasks = [
        run_single_sample(sample)
        for sample in samples
    ]

    # Run ALL samples concurrently
    results = await asyncio.gather(*tasks)
    return results

async def run_single_sample(sample):
    """Each sample runs independently."""
    messages = [sample.initial_prompt]

    for turn in range(MAX_TURNS):
        # Generate (async, doesn't block other samples)
        response = await policy.generate(messages)

        # If tool call
        if has_tool_call(response):
            # Execute tool (async, doesn't block other samples)
            result = await env.execute_tool(response.tool_call)
            messages.append({"role": "tool", "content": result})
        else:
            break

    return messages
```

**Benefits:**

```
Sample 1: Gen → Tool → Gen → Done
Sample 2:   Gen → Tool → Gen → Done
Sample 3:     Gen → Done
Sample 4:       Gen → Tool → Gen → Done

All happening CONCURRENTLY!
Total time: ~Max(single sample) not Sum(all samples)
```

**vLLM Configuration:**

```yaml
policy:
  vllm_cfg:
    async_engine: true  # Enable async mode
```

**Key Takeaways:**

✅ **Massive speedup** - 4-8x faster for multi-turn with tools
✅ **Simple to implement** - Just use async/await
✅ **vLLM handles queuing** - Engine manages multiple in-flight requests
✅ **Essential for production** - All modern RL systems use this

---

## Part 5: Implementation Plan for Forge

### High-Level Strategy

We'll adapt the **BlackJack pattern** (proven with Forge) and extend it for tool calling:

1. ✅ **Start simple** - Text-based tool call parsing (like BlackJack parses "HIT"/"STAND")
2. ✅ **Reuse BlackJack structure** - `play_game()` becomes `play_task()`
3. ✅ **Add tool execution** - Execute tools in environment (OpenEnv or custom)
4. ✅ **Track message history** - Build conversation context for each turn
5. ✅ **Add response masking** - Mark which tokens to train on
6. 🔄 **Upgrade to async** - Use async pattern for performance (optional initially)
7. 🔄 **Add native function calling** - Use vLLM's built-in support (optional later)

---

### API Design

**Core Function: `play_task()`**

**📁 Inspired by:**
- BlackJack's `play_game()`: `OpenEnv/examples/grpo_blackjack/grpo_utils.py`
- Verifiers' `rollout()`: `verifiers/verifiers/envs/multiturn_env.py:55-149`

**⚠️ NEW CODE** - This needs to be implemented

```python
async def play_task(
    task_id: str,
    task_prompt: str,
    tools: list[dict],  # OpenAI tool schemas
    env: ToolEnv,       # Environment with tool execution
    policy: Generator,  # Forge Generator
    tokenizer,
    max_turns: int = 10,
) -> list[dict]:
    """
    Play a complete multi-turn task with tool calling.

    Returns:
        List of step results, each containing:
        - turn: int
        - messages: list[dict] (conversation history at this turn)
        - prompt: str (tokenized prompt for this turn)
        - response: Completion (model response)
        - response_mask: list[int] (1 for LLM tokens, 0 for tool results)
        - is_final: bool (is this the last turn?)
    """
    messages = [
        {"role": "system", "content": format_system_prompt(tools)},
        {"role": "user", "content": task_prompt}
    ]

    task_steps = []
    turn = 0
    done = False

    while not done and turn < max_turns:
        # 1. Format prompt from message history
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. Generate response
        response = await policy.generate(prompt)

        # 3. Parse response (tool call or message)
        parsed = parse_response(response.text)

        # 4. Track tokens for masking
        response_mask = [1] * len(response.token_ids)  # All LLM tokens

        if parsed["type"] == "tool_call":
            # Tool call detected
            tool_name = parsed["name"]
            tool_args = parsed["arguments"]

            # Add assistant message with tool call
            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_call": {"name": tool_name, "arguments": tool_args}
            })

            # Execute tool in environment
            tool_result = await env.execute_tool(tool_name, tool_args)

            # Add tool result to messages
            tool_message = {"role": "tool", "content": str(tool_result)}
            messages.append(tool_message)

            # Extend response with tool result tokens (masked out)
            tool_tokens = tokenizer.encode(str(tool_result))
            response_mask.extend([0] * len(tool_tokens))  # Don't train on tool results

        else:
            # Regular message
            messages.append({
                "role": "assistant",
                "content": response.text
            })
            done = True  # Episode ends when model doesn't call tools

        # 5. Store step data
        task_steps.append({
            "turn": turn,
            "messages": list(messages),  # Copy current state
            "prompt": prompt,
            "response": response,
            "response_mask": response_mask,
            "is_final": done,
        })

        turn += 1

    # 6. Get final reward from environment
    final_reward = await env.calculate_reward(messages, task_id)

    # 7. Assign final reward to all steps
    for step in task_steps:
        step["final_reward"] = final_reward

    return task_steps
```

---

### Response Parsing Function

**📁 Inspired by:**
- BlackJack's `parse_action()`: `OpenEnv/examples/grpo_blackjack/grpo_utils.py:205-229`
- Tinker's parsing: `tinker-cookbook/tinker_cookbook/renderers.py` (search for parse_response)

**⚠️ NEW CODE** - This needs to be implemented

```python
def parse_response(response_text: str) -> dict:
    """
    Parse model response to detect tool calls.

    Supports two formats:
    1. Function call syntax: "create_task(user_id='user_1', title='Meeting')"
    2. JSON format: '{"name": "create_task", "arguments": {"user_id": "user_1", ...}}'

    Returns:
        {
            "type": "tool_call" or "message",
            "name": str (if tool_call),
            "arguments": dict (if tool_call),
            "text": str
        }
    """
    text = response_text.strip()

    # Try parsing as function call: func_name(arg1=val1, arg2=val2)
    func_pattern = r'(\w+)\((.*?)\)'
    match = re.search(func_pattern, text)

    if match:
        func_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        # Simple version: "key='value', key2='value2'"
        arguments = {}
        for arg in args_str.split(','):
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                arguments[key] = value

        return {
            "type": "tool_call",
            "name": func_name,
            "arguments": arguments,
            "text": text
        }

    # Try parsing as JSON
    if text.startswith('{'):
        try:
            parsed = json.loads(text)
            if "name" in parsed and "arguments" in parsed:
                return {
                    "type": "tool_call",
                    "name": parsed["name"],
                    "arguments": parsed["arguments"],
                    "text": text
                }
        except json.JSONDecodeError:
            pass

    # Default: regular message
    return {
        "type": "message",
        "text": text
    }
```

**Example Usage:**

```python
# Input 1: Function syntax
response = "create_task(user_id='user_1', title='Important Meeting')"
parsed = parse_response(response)
# Output: {
#     "type": "tool_call",
#     "name": "create_task",
#     "arguments": {"user_id": "user_1", "title": "Important Meeting"}
# }

# Input 2: JSON format
response = '{"name": "create_task", "arguments": {"user_id": "user_1", "title": "Meeting"}}'
parsed = parse_response(response)
# Output: same as above

# Input 3: Regular message
response = "I've created the task for you!"
parsed = parse_response(response)
# Output: {"type": "message", "text": "I've created the task for you!"}
```

---

### System Prompt for Tool Calling

**📁 Inspired by:**
- Tinker system prompts: `tinker-cookbook/tinker_cookbook/recipes/tool_use/search/train.py` (search for SYSTEM_PROMPT)
- Verifiers tool formatting: How it formats tools in prompts

**⚠️ NEW CODE** - This needs to be implemented

```python
def format_system_prompt(tools: list[dict]) -> str:
    """Format system prompt with tool definitions."""

    prompt = """You are a helpful assistant that can use tools to complete tasks.

When you need to use a tool, call it using this format:
tool_name(argument1='value1', argument2='value2')

Available tools:
"""

    # Add each tool
    for tool in tools:
        func = tool["function"]
        prompt += f"\n{func['name']}("

        # Add parameters
        params = func["parameters"]["properties"]
        required = func["parameters"].get("required", [])

        param_strs = []
        for param_name, param_info in params.items():
            param_str = param_name
            if param_name in required:
                param_str += " (required)"
            param_strs.append(param_str)

        prompt += ", ".join(param_strs)
        prompt += f")\n  Description: {func['description']}\n"

    prompt += """
Examples:
- To create a task: create_task(user_id='user_1', title='Important Meeting')
- To update status: update_task_status(task_id='task_2', status='completed')

When you're done with the task, just respond with a regular message (no tool call).
"""

    return prompt
```

---

### Tool Environment (Simple Version)

**📁 Inspired by:**
- Verifiers ToolEnv: `verifiers/verifiers/envs/tool_env.py:43-89`
- Tool schema conversion: `verifiers/verifiers/utils/tool_utils.py` (search for `convert_func_to_oai_tool`)

**⚠️ NEW CODE** - Simplified version for prototyping

```python
class SimpleToolEnv:
    """Simple tool calling environment for training."""

    def __init__(self, tools: list[Callable], reward_func: Callable):
        """
        Args:
            tools: List of Python functions to use as tools
            reward_func: Function that calculates reward from conversation
        """
        # Map function name to function
        self.tool_map = {tool.__name__: tool for tool in tools}

        # Convert to OpenAI schemas
        self.tool_schemas = [self._func_to_schema(tool) for tool in tools]

        self.reward_func = reward_func

    def _func_to_schema(self, func: Callable) -> dict:
        """Convert Python function to OpenAI tool schema."""
        # Use inspect to get signature
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        params = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Get type hint
            param_type = param.annotation
            if param_type == str:
                params[param_name] = {"type": "string"}
            elif param_type == int:
                params[param_name] = {"type": "integer"}
            # ... handle other types

            # Check if required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required
                }
            }
        }

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        if tool_name not in self.tool_map:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            tool_func = self.tool_map[tool_name]

            # Execute the tool
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**arguments)
            else:
                result = tool_func(**arguments)

            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def calculate_reward(self, messages: list[dict], task_id: str) -> float:
        """Calculate final reward for the episode."""
        return await self.reward_func(messages, task_id)
```

**Example Tools:**

**📁 Inspired by:** Tau2 mock tools at `tau2-bench/src/tau2/domains/mock/tools.py`

```python
# Define simple tools
def mock_create_task(user_id: str, title: str) -> str:
    """Create a new task for a user."""
    task_id = f"task_{random.randint(1, 100)}"
    return f"Created task '{title}' with ID {task_id}"

def mock_update_status(task_id: str, status: str) -> str:
    """Update task status."""
    return f"Task {task_id} status updated to {status}"

# Reward function
async def simple_reward(messages: list[dict], task_id: str) -> float:
    """Simple reward: 1.0 if task completed, 0.0 otherwise."""

    # Check if create_task was called
    created = any(
        msg.get("tool_call", {}).get("name") == "mock_create_task"
        for msg in messages if msg.get("role") == "assistant"
    )

    # Check if update_status was called
    updated = any(
        msg.get("tool_call", {}).get("name") == "mock_update_status"
        for msg in messages if msg.get("role") == "assistant"
    )

    # Reward if both tools were called
    return 1.0 if (created and updated) else 0.0

# Create environment
env = SimpleToolEnv(
    tools=[mock_create_task, mock_update_status],
    reward_func=simple_reward
)
```

---

### Updated Episode Structure

**📁 Based on:**
- Current Episode: `OpenEnv/examples/grpo_blackjack/grpo_utils.py:47-60`
- Response mask pattern: See VERL/NeMo-RL examples in `4_examples_APIs.md`

**⚠️ MODIFIED CODE** - Extends existing Episode with multi-turn fields

```python
@dataclass
class Episode:
    """Episode data for multi-turn tool calling RL training."""

    episode_id: str
    pad_id: int
    request_len: int
    response_len: int

    # Multi-turn specific
    task_id: str            # Which task this is from
    turn_in_task: int       # Which turn in the task (0, 1, 2, ...)

    # Standard fields
    completion: Completion   # Contains prompt_ids, token_ids, logprobs
    ref_logprobs: torch.Tensor
    reward: float
    advantage: float

    # NEW: Response mask
    response_mask: torch.Tensor | None = None  # 1=train on, 0=ignore (tool results)

    @property
    def masked_response_tensor(self) -> torch.Tensor:
        """Get response tensor with padding."""
        response_tokens = torch.tensor(self.completion.token_ids, dtype=torch.long)

        # Pad to response_len
        if response_tokens.shape[0] < self.response_len:
            diff = self.response_len - response_tokens.shape[0]
            response_tokens = F.pad(response_tokens, (0, diff), value=self.pad_id)

        return response_tokens

    @property
    def mask_tensor(self) -> torch.Tensor:
        """Get mask tensor with padding."""
        if self.response_mask is None:
            # No mask, train on all tokens
            mask = torch.ones(len(self.completion.token_ids), dtype=torch.long)
        else:
            mask = self.response_mask

        # Pad to response_len
        if mask.shape[0] < self.response_len:
            diff = self.response_len - mask.shape[0]
            mask = F.pad(mask, (0, diff), value=0)  # Padding is masked out

        return mask
```

---

### Integration with Forge GRPO

**📁 Based on:**
- Current rollouts: `OpenEnv/examples/grpo_blackjack/grpo_utils.py` (search for `continuous_rollouts`)
- Main GRPO: `apps/grpo/main.py`

**⚠️ MODIFIED CODE** - Extends existing continuous_rollouts for tool calling

**Updated `continuous_rollouts`:**

```python
async def continuous_rollouts(
    policy: Generator,
    replay_buffer: ReplayBuffer,
    reward_actor: RewardActor,
    ref_model: ReferenceModel,
    env: SimpleToolEnv,
    tokenizer,
    group_size: int = 8,
):
    """Continuous rollout loop with tool calling."""

    while True:
        # Sample tasks
        tasks = sample_tasks(group_size)  # Get G different tasks

        # Play all tasks
        all_step_results = []
        for task in tasks:
            task_id = task["id"]
            task_prompt = task["prompt"]

            # Play the task (multi-turn)
            step_results = await play_task(
                task_id=task_id,
                task_prompt=task_prompt,
                tools=env.tool_schemas,
                env=env,
                policy=policy,
                tokenizer=tokenizer,
                max_turns=10
            )

            all_step_results.extend(step_results)

        # Create episodes (one per turn)
        episodes = []
        for step_result in all_step_results:
            episode = Episode(
                episode_id=str(uuid.uuid4()),
                pad_id=tokenizer.pad_token_id,
                request_len=MAX_REQUEST_TOKENS,
                response_len=MAX_RESPONSE_TOKENS,
                task_id=step_result["task_id"],
                turn_in_task=step_result["turn"],
                completion=step_result["response"],
                response_mask=torch.tensor(step_result["response_mask"]),
            )

            # Simple reward (could add shaping)
            episode.reward = step_result["final_reward"]

            episodes.append(episode)

        # Get reference logprobs
        input_ids = [tokenizer.encode(ep.completion.prompt) for ep in episodes]
        ref_logprobs = await ref_model.forward(input_ids, return_logprobs=True)
        for i, episode in enumerate(episodes):
            episode.ref_logprobs = ref_logprobs[i]

        # Compute advantages (group-relative)
        # Group by task_id to compare different trajectories of same task
        task_groups = {}
        for episode in episodes:
            if episode.task_id not in task_groups:
                task_groups[episode.task_id] = []
            task_groups[episode.task_id].append(episode)

        for task_id, task_episodes in task_groups.items():
            rewards = [ep.reward for ep in task_episodes]
            mean_reward = sum(rewards) / len(rewards)

            for episode in task_episodes:
                episode.advantage = episode.reward - mean_reward

        # Add to replay buffer
        for episode in episodes:
            await replay_buffer.add(episode)
```

---

### Updated GRPO Loss (with masking)

**📁 Based on:**
- Current GRPO loss: `OpenEnv/examples/grpo_blackjack/grpo_utils.py:125-150` (`simple_grpo_loss`)
- Response masking pattern: See VERL in `4_examples_APIs.md:599-615`

**⚠️ MODIFIED CODE** - Adds response_mask parameter to existing loss

```python
def grpo_loss_with_masking(
    logits: torch.Tensor,
    response: torch.Tensor,
    response_mask: torch.Tensor,  # NEW!
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    GRPO loss with response masking for multi-turn.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        response: Response tokens [batch, seq_len]
        response_mask: Which tokens to train on [batch, seq_len] (1=train, 0=ignore)
        ref_logprobs: Reference model log probabilities [batch, seq_len]
        advantages: Normalized advantages [batch, 1]
        padding_mask: Mask for padded tokens [batch, seq_len]
        beta: KL penalty coefficient

    Returns:
        Scalar loss value
    """
    # Compute log probabilities
    logprobs = compute_logprobs(logits, response)

    # KL divergence
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1

    # Policy loss
    policy_loss = -logprobs * advantages

    # Total loss per token
    loss_per_token = policy_loss + beta * kl

    # IMPORTANT: Combine padding_mask AND response_mask
    combined_mask = padding_mask * response_mask  # Both must be 1

    # Apply combined mask
    masked_loss = loss_per_token * combined_mask

    # Average over non-masked tokens
    loss = masked_loss.sum() / combined_mask.sum()

    return loss
```

**Key Difference:** `response_mask` zeros out tool result tokens, so we only train on LLM-generated tokens.

---

## Part 6: Performance & Async Patterns

### Why Async Matters for Tool Calling

**Synchronous Problem:**

```python
# BAD: Blocks entire batch while waiting for tools
for sample in batch:
    response = policy.generate(sample.prompt)  # Blocks others
    if has_tool_call(response):
        result = env.execute_tool(response.tool_call)  # Blocks others!
    ...
```

**With async:**

```python
# GOOD: All samples run independently
async def process_sample(sample):
    response = await policy.generate(sample.prompt)  # Doesn't block
    if has_tool_call(response):
        result = await env.execute_tool(response.tool_call)  # Doesn't block!
    ...

# Run all samples concurrently
results = await asyncio.gather(*[process_sample(s) for s in batch])
```

**Speedup Example:**

```
Synchronous (4 samples, each takes 10s):
Sample 1 → 10s → Sample 2 → 10s → Sample 3 → 10s → Sample 4 → 10s
Total: 40 seconds

Asynchronous (all 4 samples in parallel):
Sample 1 ┐
Sample 2 ├ All run together → 10s
Sample 3 ┤
Sample 4 ┘
Total: ~10 seconds (4x speedup!)
```

---

### Enabling Async in Forge Generator

**Step 1: Enable vLLM async engine**

**📁 Code Reference:**
- Generator setup: `src/forge/actors/generator.py:71-99`
- NeMo-RL async config: See `4_examples_APIs.md:680-689`

```python
# In your config
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    # ... other args
)

# When creating Generator
generator = await Generator.options(
    procs=1,
    num_replicas=1,
    with_gpus=True
).as_service(
    engine_args=engine_args,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
)
```

**Note:** Forge's Generator already supports async! You just need to use `await` when calling it.

---

**Step 2: Make `play_task` async**

```python
async def play_task(task_id, task_prompt, tools, env, policy, tokenizer, max_turns=10):
    """Already async in our implementation above!"""
    messages = [{"role": "user", "content": task_prompt}]

    for turn in range(max_turns):
        # Async generation
        response = await policy.generate(prompt)  # await here!

        # Async tool execution
        if has_tool_call(parsed):
            result = await env.execute_tool(...)  # await here!
        ...
```

---

**Step 3: Run multiple tasks concurrently**

**📁 Code Reference:** See NeMo-RL pattern in `4_examples_APIs.md:719-735` (`run_async_multi_turn_rollout`)

```python
async def continuous_rollouts(...):
    while True:
        # Sample G tasks
        tasks = sample_tasks(group_size)

        # Create tasks for all
        task_coroutines = [
            play_task(
                task_id=task["id"],
                task_prompt=task["prompt"],
                tools=env.tool_schemas,
                env=env,
                policy=policy,
                tokenizer=tokenizer,
            )
            for task in tasks
        ]

        # Run ALL tasks concurrently
        all_step_results_per_task = await asyncio.gather(*task_coroutines)

        # Flatten results
        all_step_results = []
        for step_results in all_step_results_per_task:
            all_step_results.extend(step_results)

        # Continue with episode creation...
```

---

### Performance Best Practices

**1. Use async/await everywhere**

**📁 Code Reference:** NeMo-RL async patterns in `4_examples_APIs.md:803-830`

```python
# BAD
def execute_tool(self, tool_name, args):
    return tool_func(**args)  # Blocks

# GOOD
async def execute_tool(self, tool_name, args):
    if asyncio.iscoroutinefunction(tool_func):
        return await tool_func(**args)
    else:
        # Run sync function in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, tool_func, **args)
```

---

**2. Batch reference model calls**

```python
# BAD: One call per episode
for episode in episodes:
    ref_logprobs = await ref_model.forward(episode.prompt)
    episode.ref_logprobs = ref_logprobs

# GOOD: Batch all episodes
all_prompts = [ep.completion.prompt for ep in episodes]
all_ref_logprobs = await ref_model.forward(all_prompts)  # Single batched call
for episode, ref_logprobs in zip(episodes, all_ref_logprobs):
    episode.ref_logprobs = ref_logprobs
```

---

**3. Pipeline rollouts and training**

```python
# BAD: Wait for all rollouts before training
rollouts = await collect_rollouts()
await train_on_rollouts(rollouts)

# GOOD: Start training as soon as buffer has enough samples
async def rollout_loop():
    while True:
        rollouts = await collect_rollouts()
        await replay_buffer.add(rollouts)

async def training_loop():
    while True:
        if replay_buffer.size() >= min_size:
            batch = await replay_buffer.sample()
            await trainer.train_step(batch)
        await asyncio.sleep(0.1)

# Run both concurrently
await asyncio.gather(rollout_loop(), training_loop())
```

---

## Part 7: What's Already Supported vs What Needs to Be Added

### Already Supported in Forge ✅

**1. vLLM Async Generation**
- ✅ Forge Generator already uses vLLM v1
- ✅ Async generation works out of the box
- ✅ `await policy.generate(prompt)` is already async

**2. Multi-GPU and Distributed Training**
- ✅ Monarch handles distributed coordination
- ✅ Generator, Trainer, ReplayBuffer can run on different GPUs
- ✅ Weight syncing via torchstore

**3. GRPO Algorithm**
- ✅ Group-relative advantages
- ✅ KL penalty with reference model
- ✅ Replay buffer with sampling
- ✅ Async training loop

**4. Episode Management**
- ✅ Episode dataclass structure
- ✅ Collation for batching
- ✅ Tokenization and padding

**5. OpenEnv Integration**
- ✅ BlackJack example shows it works!
- ✅ HTTP-based environment communication
- ✅ Async environment calls (with wrapper)

---

### What Needs to Be Added ⚠️

**1. Response Parsing for Tool Calls**

**What:** Function to detect and parse tool calls from model output

**Complexity:** Low (see Part 5 for implementation)

**Example:**
```python
def parse_response(response_text: str) -> dict:
    # Detect: create_task(user_id='user_1', title='Meeting')
    # Return: {"type": "tool_call", "name": "create_task", "arguments": {...}}
```

**Status:** ❌ Not implemented
**Effort:** ~1-2 hours
**File:** Can be in `grpo_utils.py` or new `tool_calling_utils.py`

---

**2. Multi-turn Rollout Loop**

**What:** `play_task()` function (like `play_game()` in BlackJack)

**Complexity:** Medium

**Status:** ❌ Not implemented (but BlackJack provides template!)
**Effort:** ~4-6 hours
**File:** `grpo_utils.py` or new `tool_calling_rollouts.py`

**Implementation:** See Part 5, "API Design" section

---

**3. Tool Environment**

**What:** Environment that executes tools and returns results

**Complexity:** Medium-High (depends on tools)

**Options:**

**Option A:** Use existing OpenEnv environment
- ✅ Already has Docker sandboxing
- ❌ May not have tool calling support yet
- **Effort:** Check if OpenEnv has tool env, otherwise 8-12 hours to build

**Option B:** Build simple mock environment
- ✅ Easiest to get started
- ❌ Not realistic for production
- **Effort:** 2-4 hours
- **Implementation:** See Part 5, "Tool Environment (Simple Version)"

**Option C:** Integrate Verifiers ToolEnv
- ✅ Production-ready, clean API
- ✅ Tool schema generation built-in
- ❌ Another dependency
- **Effort:** 4-6 hours integration

**Recommendation:** Start with Option B (mock), upgrade to Option C (Verifiers) later

**Status:** ❌ Not implemented
**File:** `tool_env.py`

---

**4. Response Masking**

**What:** Track which tokens are LLM output vs tool results

**Complexity:** Medium

**Status:** ❌ Not implemented
**Effort:** 3-4 hours

**What needs to change:**
1. Add `response_mask` field to Episode dataclass (✅ shown in Part 5)
2. Track mask during rollout (✅ shown in Part 5)
3. Update GRPO loss to use mask (✅ shown in Part 5)

**Files to modify:**
- `Episode` dataclass
- `play_task()` function
- `grpo_loss()` function

---

**5. Tool Schema Generation**

**What:** Convert Python functions to OpenAI tool schemas

**Complexity:** Medium

**Status:** ❌ Not implemented (but can copy from Verifiers!)
**Effort:** 2-3 hours

**Implementation:**
```python
def func_to_schema(func: Callable) -> dict:
    # Use inspect.signature, inspect.getdoc
    # Return OpenAI tool schema
```

**Recommendation:** Copy from Verifiers library (it's well-tested)

---

**6. System Prompt Formatting**

**What:** Format system prompt with tool definitions

**Complexity:** Low

**Status:** ❌ Not implemented
**Effort:** 1-2 hours

**Implementation:** See Part 5, "System Prompt for Tool Calling"

---

**7. vLLM Native Tool Calling Support (Optional)**

**What:** Use vLLM's built-in function calling instead of text parsing

**Complexity:** Medium-High

**Status:** ❌ Not implemented (not needed initially!)
**Effort:** 6-8 hours

**vLLM Config:**
```python
engine_args = EngineArgs(
    model="...",
    enable_auto_tool_choice=True,  # Enable native tool calling
    tool_call_parser="hermes",      # Parser type
)
```

**Recommendation:** Skip initially, use text parsing. Add later if needed.

---

**8. Tau2 Evaluation Integration**

**What:** Run trained model on Tau2Bench for evaluation

**Complexity:** Medium

**Status:** ❌ Not implemented
**Effort:** 4-6 hours

**Two approaches:**

**Approach A:** Use Tau2 CLI
```bash
tau2 run --domain mock --agent-llm /path/to/checkpoint
```
Need to figure out how to point Tau2 to local model.

**Approach B:** Use Tau2's gym interface programmatically
```python
import gymnasium as gym
from tau2.gym import register_gym_agent

env = gym.make("Tau-v0", domain="mock")
# Run evaluation loop
```

**Recommendation:** Start with Approach A (simpler)

---

### Summary: Implementation Checklist

**Phase 1: Minimum Viable Tool Calling (1-2 days)**

- [ ] 1. Implement `parse_response()` function (1-2 hours)
- [ ] 2. Implement `SimpleToolEnv` with mock tools (2-4 hours)
- [ ] 3. Implement `play_task()` function (4-6 hours)
- [ ] 4. Test end-to-end on simple task (2-3 hours)

**Phase 2: Integration with Forge GRPO (2-3 days)**

- [ ] 5. Add `response_mask` to Episode (1 hour)
- [ ] 6. Update `continuous_rollouts` to use `play_task()` (2-3 hours)
- [ ] 7. Update GRPO loss with masking (2-3 hours)
- [ ] 8. Test training loop (4-6 hours)

**Phase 3: Production-Ready (3-5 days)**

- [ ] 9. Implement proper tool schema generation (2-3 hours)
- [ ] 10. Add system prompt formatting (1-2 hours)
- [ ] 11. Integrate Verifiers ToolEnv or build OpenEnv tool env (8-12 hours)
- [ ] 12. Add comprehensive logging and metrics (4-6 hours)

**Phase 4: Evaluation (1-2 days)**

- [ ] 13. Figure out Tau2 local model evaluation (2-4 hours)
- [ ] 14. Create evaluation script (2-3 hours)
- [ ] 15. Run full evaluation on Tau2 mock domain (2-4 hours)

**Total Estimated Effort:** 2-3 weeks for full implementation

---

## Appendix: Quick Reference

### Key Files to Create/Modify

**New Files:**
- `tool_calling_utils.py` - Response parsing, tool schemas
- `tool_env.py` - Tool execution environment
- `tool_calling_rollouts.py` - `play_task()` implementation

**Files to Modify:**
- `apps/grpo/main.py` - Update `continuous_rollouts`
- `grpo_utils.py` - Add response masking to Episode, update loss

---

### Key Concepts Recap

1. **Tool Calling** = Model invokes functions instead of just generating text
2. **Multi-turn** = Multiple back-and-forth exchanges in one episode
3. **Response Mask** = Track which tokens to train on (LLM) vs ignore (tools)
4. **Sparse Reward** = Reward only at episode end, not per turn
5. **Async Pattern** = Use async/await for concurrent sample processing

---

### Next Steps

1. **Start with BlackJack** - Understand how it works end-to-end
2. **Build Simple Mock Environment** - 2-3 tools, simple reward
3. **Prototype `play_task()`** - Single task, multi-turn, with tools
4. **Test Locally** - Run one episode, verify it works
5. **Integrate with GRPO** - Add to training loop
6. **Scale Up** - Add more tools, better reward functions
7. **Evaluate on Tau2** - Measure performance on benchmark

---

### Questions to Answer Next

1. **Which tool environment?** Mock, OpenEnv, or Verifiers?
2. **Text parsing or native function calling?** Start text, upgrade later?
3. **Reward function design?** Binary, shaped, or LLM-as-judge?
4. **Training tools = Tau2 tools?** Or different for generalization?

See `3_open_questions.md` for detailed discussion of these questions.

---

**End of Tutorial**

You should now have a solid understanding of:
- What tool calling and multi-turn are
- How Tau2Bench works
- How Forge currently operates
- How other libraries implement these features
- What needs to be added to Forge
- How to implement it step by step

Ready to start coding! 🚀
