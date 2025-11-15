# Requirements and Context

## Original User Prompt (Updated)

I work in torchforge, which is an RL training library. Here is an example on how we do GRPO at `apps/grpo/main.py`.

It is still early days and we have multiple blind spots.

**IMPORTANT UPDATE:** We want to train a model to perform well on tau2bench, but the approach is:
- **Training**: Use OpenEnv Docker sandboxes for tool calling and rewards (NOT Tau2)
- **Evaluation**: Use Tau2Bench to evaluate trained models

Tau2Bench is **ONLY** for evaluation. Training will happen on OpenEnv environments.

**My Questions:**
1. Once we have a trained model, how do I run taubench?
2. How do I prepare rewards or data to do well on taubench? Do I look at the scoring done by taubench? Do I try to support the same exact tools in my training?
3. How does taubench score?
4. We currently don't have multiturn or tool calling. How does it work and how do I incorporate it to main.py?
5. What else am I missing?

**Process Notes:**
- Clean code snippets help me a lot to understand the situation
- Since there is a lot of content here, we won't be able to figure this out in a single conversation, so we will have to do it in steps
- These docs (1, 2, 3) should have all info needed to continue executing and exploring
- I will NOT provide this prompt again explaining my motivations

**Main Goal:** Come up with **clean code** showing how to go from what we have in Forge (GRPO on single-turn) to a **rollout loop that uses tool calling + multi-turn**.

Specifically:
1. **Design clear APIs/abstractions** for tool calling episodes
2. **Show concrete code** (not just plans) for:
   - Prompt formatting with tools
   - Response parsing (tool calls vs messages)
   - Multi-turn conversation management
   - Episode creation from multi-turn tasks
   - Integration with existing Forge GRPO
3. **Enable Tau2Bench evaluation** of the trained model

**Approach:**
- Study existing examples: OpenEnv BlackJack, Tinker-cookbook tool use
- Extract patterns and best practices
- Synthesize into clean Forge-compatible code
- Provide working implementation, not just design docs

The deliverable is **code that works**, with clear examples and minimal abstraction complexity.

---

## What is Forge (torchforge)?

**Location:** `/home/felipemello/forge/`

Forge is a PyTorch-native agentic RL library focused on enabling rapid research while maintaining scalability.

### Key Concepts

**Architecture:**
- **Actors** - Distributed components running RL logic (Generators, Trainers, ReplayBuffers, etc.)
- **Monarch** - Underlying process mesh system for distributed coordination
- **Controllers** - Orchestrate actors and manage lifecycle

**Core Components:**
- `Generator` - vLLM-based text generation service (uses vLLM v1)
- `TitanTrainer` - Training service for model updates
- `ReplayBuffer` - Stores episodes for training
- `ReferenceModel` - Maintains reference model for KL divergence
- `ForgeActor` - Base class for all actors in the system

**Current Capabilities:**
- GRPO (Group Relative Policy Optimization) - see `apps/grpo/main.py`
- SFT (Supervised Fine-Tuning)
- Async/sync training modes
- Multi-GPU support with distributed training

**Current GRPO Flow (apps/grpo/main.py):**
```python
# 1. Setup services
policy = Generator(...)              # Generate completions
trainer = TitanTrainer(...)          # Train model
replay_buffer = ReplayBuffer(...)    # Store episodes
ref_model = ReferenceModel(...)      # Reference for KL
reward_actor = RewardActor(...)      # Calculate rewards

# 2. Rollout loop (continuous_rollouts)
prompt, target = sample from dataset
responses = policy.generate(prompt)  # Generate G responses
rewards = reward_actor.evaluate(...)  # Score each response
ref_logprobs = ref_model.forward(...) # Get reference logprobs
advantages = compute_advantages(...)  # Normalize rewards
replay_buffer.add(episode)           # Store episode

# 3. Training loop (continuous_training)
batch = replay_buffer.sample(...)
trainer.train_step(inputs, targets)  # Train on batch
trainer.push_weights(version)        # Save weights to torchstore
policy.update_weights(version)       # Update policy with new weights
```

**What Forge Currently Does NOT Have:**
- Multi-turn conversation handling
- Tool/function calling support
- Structured reward functions for tool-based tasks
- Environment interaction patterns (like gym environments)

---

## What is Tau2Bench?

**Location:** `/home/felipemello/forge/tau2-bench/`

Tau2Bench is a benchmark for evaluating conversational agents in customer service scenarios. It simulates realistic multi-turn conversations where agents must follow policies, use tools, and interact with users.

### Key Concepts

**Domains:**
- `mock` - Simple task management (create_task, update_task)
- `airline` - Flight booking and management
- `retail` - Product orders and returns
- `telecom` - Customer support with technical troubleshooting

**Two Modes:**
1. **Normal Mode** - Agent converses with user simulator
2. **Solo Mode** - Agent works independently on tickets (no user interaction)

**Architecture:**
```
Orchestrator
├── Agent (your model)
├── User Simulator (LLM playing customer)
└── Environment (domain-specific tools and state)
```

**Tool Calling Format:**
Agents can either:
- Send text message: `"I'll help you with that"`
- Make tool call: `"search_flights(origin='NYC', destination='LAX')"`
- JSON format: `{"name": "search_flights", "arguments": {"origin": "NYC", "destination": "LAX"}}`

**Task Structure:**
```json
{
  "id": "create_task_1",
  "user_scenario": {
    "persona": "Professional communicator",
    "instructions": "Create a task called 'Important Meeting' for user_1"
  },
  "ticket": "User needs to create a task...",
  "evaluation_criteria": {
    "actions": [
      {
        "action_id": "create_1",
        "name": "create_task",
        "arguments": {"user_id": "user_1", "title": "Important Meeting"}
      }
    ],
    "reward_basis": ["ACTION", "COMMUNICATE"]
  }
}
```

**Reward/Scoring System:**

Tau2 evaluates completed simulations based on multiple criteria:

1. **ENV** - Environment state checks:
   - Database state matches expectations
   - Environment assertions pass (e.g., task_id="task_2" has status="pending")

2. **ACTION** - Tool call verification:
   - Agent called the right tools
   - With the right arguments (or subset via `compare_args`)
   - In any order (not sequence-dependent)

3. **COMMUNICATE** - Communication checks:
   - Agent communicated required information to user

4. **NL_ASSERTIONS** - Natural language assertions (experimental):
   - LLM-based evaluation of conversation quality

**Final reward** = product of all reward components (0.0 or 1.0 typically, binary success)

Tasks must end with:
- `AGENT_STOP` - Agent calls `done()` tool
- `USER_STOP` - User says stop keywords
- Otherwise reward = 0.0

### Gymnasium Interface

Tau2 now includes RL training support via `AgentGymEnv`:

```python
import gymnasium as gym
from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID

register_gym_agent()
env = gym.make(TAU_BENCH_ENV_ID, domain="mock", task_id="create_task_1")

# Observation: conversation history as string
observation, info = env.reset()
# info contains: tools, policy, simulation_run

# Action: either message or tool call
action = "create_task(user_id='user_1', title='Important Meeting')"
observation, reward, terminated, truncated, info = env.step(action)

# reward is binary: 1.0 if all criteria met, 0.0 otherwise
```

**Key Insight:** The gym interface provides **sparse rewards** - you only get the final reward after the episode terminates (when agent/user stops).

### Task Splits

Domains have train/test splits for proper evaluation:
- `base` - Complete task set (original benchmark)
- `train` - Training tasks
- `test` - Held-out evaluation tasks

---

---

## What is OpenEnv?

**Location:** `/home/felipemello/forge/OpenEnv/`

OpenEnv is a framework for creating isolated execution environments (Docker containers) for agentic RL training. It provides a Gymnasium-style API for any environment.

### Key Concepts

**Architecture:**
```
Client (Forge)  ←─HTTP─→  Docker Container (OpenEnv Server)
                          └─ Environment Logic
                          └─ Reward Computation
                          └─ State Management
```

**API (Gym-style):**
```python
from envs.coding_env import CodingEnv, CodeAction

env = CodingEnv.from_docker_image("coding-env:latest")
result = env.reset()                    # Start episode
result = env.step(CodeAction(...))      # Take action
state = env.state()                     # Get state
env.close()                             # Cleanup
```

**StepResult:**
```python
@dataclass
class StepResult:
    observation: Observation  # Environment feedback
    reward: float            # Immediate reward (can be sparse or dense)
    done: bool              # Episode terminated?
```

**Existing Environments:**
- `echo_env` - Simple message echo (demo)
- `coding_env` - Python code execution
- `openspiel_env` - Games (BlackJack, Chess, TicTacToe, etc.)
- `browsergym_env` - Web browser interaction
- `atari_env` - Atari games
- Many more (70+ total)

**Important:** OpenEnv environments can run **synchronously** (blocking) or be wrapped for async use.

### Working Example: GRPO + BlackJack

A complete working example exists at `/home/felipemello/forge/OpenEnv/examples/grpo_blackjack/` showing Forge + OpenEnv integration. See `4_examples_APIs.md` for detailed analysis of the pattern.

---

## Comparison: Forge GRPO vs OpenEnv vs Tau2

| Aspect | Forge GRPO (GSM8K) | OpenEnv Training | Tau2 Evaluation |
|--------|-------------------|------------------|-----------------|
| **Purpose** | Current training | New training approach | Final evaluation |
| **Input** | Single prompt | Game/environment state | Multi-turn conversation |
| **Output** | Single completion | Actions (text or parsed) | Messages + tool calls |
| **Tools** | Not supported | Environment-specific | Domain-specific |
| **Reward** | Per-response | Per-step or per-episode | Sparse, end-of-episode |
| **Episode** | 1 prompt → 1 response | Multi-step game/task | Multi-turn conversation |
| **Use Case** | Math problems | Tool calling, games | Benchmark performance |

---

## File References

**Forge:**
- Main GRPO (GSM8K): `apps/grpo/main.py`
- Generator: `src/forge/actors/generator.py`
- Trainer: `src/forge/actors/trainer.py`
- Episode dataclass: `apps/grpo/main.py:43-74`

**OpenEnv (Training):**
- Main README: `OpenEnv/README.md`
- Environments: `OpenEnv/src/envs/`
- **BlackJack Example (KEY!)**: `OpenEnv/examples/grpo_blackjack/`
  - `grpo_utils.py` - Complete integration with Forge
  - `blackjack.yaml` - Training configuration
  - `play_game()` - Episode collection pattern
- Coding Environment: `OpenEnv/src/envs/coding_env/`

**Tinker-Cookbook (Tool Use Examples):**
- Tool interface: `tinker-cookbook/tinker_cookbook/recipes/tool_use/search/tools.py`
- Search environment: `tinker-cookbook/tinker_cookbook/recipes/tool_use/search/search_env.py`
- Training: `tinker-cookbook/tinker_cookbook/recipes/tool_use/search/train.py`
- Renderers: `tinker-cookbook/tinker_cookbook/renderers.py`

**Tau2 (Evaluation Only):**
- Main README: `tau2-bench/README.md`
- Evaluation command: `tau2 run --domain <domain> --agent-llm <model> --user-llm <model>`
- Gym README: `tau2-bench/src/tau2/gym/README.md`
- Evaluator: `tau2-bench/src/tau2/evaluator/evaluator.py`
- Task structure: `tau2-bench/src/tau2/data_model/tasks.py`
- Example tasks: `tau2-bench/data/tau2/domains/mock/tasks.json`

**Example APIs:**
- **4_examples_APIs.md** - Complete analysis of BlackJack and Tinker patterns with proposed Forge API
