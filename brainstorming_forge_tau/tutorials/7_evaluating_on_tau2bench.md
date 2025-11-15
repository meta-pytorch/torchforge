# Part 7: Evaluating Your Trained Model on Tau2Bench

Once you've trained a model with multi-turn tool calling, you need to evaluate it on Tau2Bench to measure performance.

## 7.1 Running Tau2Bench Evaluation

### Using tau2 CLI Command

**Basic usage:**
```bash
tau2 run \
  --domain mock \
  --agent-llm /path/to/your/trained/model \
  --mode solo
```

**Full options:**
```bash
tau2 run \
  --domain mock \
  --task-split test \
  --agent-llm /path/to/model \
  --mode solo \
  --output-dir ./results/tau2_eval \
  --num-workers 4
```

**Configuration options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--domain` | Which domain to evaluate (mock, airline, retail, telecom) | Required |
| `--agent-llm` | Path to your model | Required |
| `--mode` | solo or normal | solo |
| `--task-split` | train, test, or base | base |
| `--output-dir` | Where to save results | ./results |
| `--num-workers` | Parallel evaluation workers | 1 |
| `--max-turns` | Max turns per episode | 10 |

### How to Point to Your Trained Model

**Option 1: HuggingFace checkpoint path**
```bash
tau2 run \
  --domain mock \
  --agent-llm "felipemello/qwen-tau2-finetuned" \
  --mode solo
```

**Option 2: Local checkpoint directory**
```bash
tau2 run \
  --domain mock \
  --agent-llm "/home/felipemello/forge/checkpoints/tau2_grpo/step_1000" \
  --mode solo
```

**Option 3: Using Forge saved checkpoints**

Forge saves checkpoints via torchstore. Convert to HF format first:

```python
# Convert Forge checkpoint to HuggingFace format
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from torchstore
model_path = trainer.load_checkpoint(version=latest_version)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Save in HF format
model.save_pretrained("./checkpoints/hf_format")
tokenizer.save_pretrained("./checkpoints/hf_format")
```

Then use:
```bash
tau2 run \
  --domain mock \
  --agent-llm "./checkpoints/hf_format" \
  --mode solo
```

## 7.2 Programmatic Evaluation (Gym Interface)

For more control, use Tau2's Gym interface:

```python
# examples/tau2bench/evaluate.py

import gymnasium as gym
from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID
from transformers import AutoModelForCausalLM, AutoTokenizer

# Register Tau2 gym environment
register_gym_agent()

# Load your trained model
model_path = "./checkpoints/hf_format"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def evaluate_on_tau2(domain: str, task_split: str = "test"):
    """Evaluate model on Tau2Bench tasks."""

    # Get all tasks for this domain
    from tau2.data_model import load_tasks
    tasks = load_tasks(domain=domain, split=task_split)

    results = []

    for task in tasks:
        # Create environment for this task
        env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=domain,
            task_id=task["id"]
        )

        # Run episode
        observation, info = env.reset()
        done = False
        turn = 0
        max_turns = 10

        while not done and turn < max_turns:
            # Build prompt
            prompt = observation  # Tau2 provides formatted observation

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7
            )
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Step environment
            observation, reward, terminated, truncated, info = env.step(response_text)
            done = terminated or truncated
            turn += 1

        # Collect result
        results.append({
            "task_id": task["id"],
            "reward": reward,
            "num_turns": turn,
            "success": reward > 0.5
        })

    return results


# Run evaluation
results = evaluate_on_tau2(domain="mock", task_split="test")

# Print summary
successes = sum(1 for r in results if r["success"])
print(f"Success rate: {successes}/{len(results)} = {successes/len(results)*100:.1f}%")
print(f"Average reward: {sum(r['reward'] for r in results) / len(results):.3f}")
```

### Collecting Metrics

```python
# examples/tau2bench/evaluate.py (continued)

def aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics."""
    return {
        "total_tasks": len(results),
        "successes": sum(1 for r in results if r["success"]),
        "success_rate": sum(r["success"] for r in results) / len(results),
        "average_reward": sum(r["reward"] for r in results) / len(results),
        "average_turns": sum(r["num_turns"] for r in results) / len(results),
        "max_reward": max(r["reward"] for r in results),
        "min_reward": min(r["reward"] for r in results),
    }


def save_results(results: list[dict], metrics: dict, output_path: str):
    """Save evaluation results."""
    import json

    output = {
        "metrics": metrics,
        "per_task_results": results,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


# Use it
metrics = aggregate_metrics(results)
save_results(results, metrics, "./results/eval_results.json")
```

## 7.3 Interpreting Results

### Understanding Tau2Bench Scores

Tau2Bench computes multiple sub-scores that combine into final reward:

```python
# Example result breakdown
{
    "task_id": "create_task_1",
    "scores": {
        "ACTION": 1.0,      # Called correct tools with correct args
        "ENV": 1.0,         # Environment state is correct
        "COMMUNICATE": 1.0, # Communicated required info to user
        "NL_ASSERTIONS": 1.0  # (Optional) LLM-judged quality
    },
    "final_reward": 1.0  # Product of all scores
}
```

**Score meanings:**

**ACTION Score (0.0 or 1.0):**
- ✅ 1.0: Agent called all required tools with correct arguments
- ❌ 0.0: Missing tools or wrong arguments

**ENV Score (0.0 or 1.0):**
- ✅ 1.0: Environment state matches expectations
- ❌ 0.0: Database inconsistencies, wrong object states

**COMMUNICATE Score (0.0 or 1.0):**
- ✅ 1.0: Agent communicated all required information
- ❌ 0.0: Missing confirmations or key details

**NL_ASSERTIONS Score (0.0-1.0):**
- LLM-based evaluation (experimental)
- Checks conversation quality, tone, etc.

**Final Reward:**
```python
final_reward = ACTION * ENV * COMMUNICATE * NL_ASSERTIONS
```

If ANY component is 0, final reward is 0!

### Debugging Failed Episodes

**Inspect conversation history:**

```python
def debug_failed_episode(task_id: str, domain: str):
    """Inspect a failed episode."""
    env = gym.make(TAU_BENCH_ENV_ID, domain=domain, task_id=task_id)

    observation, info = env.reset()
    messages = []
    done = False

    while not done:
        # Generate (your model)
        response = generate_response(observation)
        messages.append({"role": "assistant", "content": response})

        # Step
        observation, reward, terminated, truncated, info = env.step(response)
        messages.append({"role": "environment", "content": observation})
        done = terminated or truncated

    # Print full conversation
    print(f"=== Episode: {task_id} ===")
    for i, msg in enumerate(messages):
        print(f"Turn {i}: [{msg['role']}] {msg['content']}")

    # Check what went wrong
    print(f"\n=== Evaluation ===")
    print(f"Final reward: {reward}")
    print(f"Score breakdown: {info.get('scores', {})}")

    # Compare to expected
    task_data = load_task(domain, task_id)
    print(f"\n=== Expected Actions ===")
    for action in task_data["evaluation_criteria"]["actions"]:
        print(f"- {action['name']}({action['arguments']})")
```

**Common failure modes:**

1. **Agent doesn't call tools** (ACTION=0)
   - **Symptom**: Model generates text response instead of tool call
   - **Fix**: Improve prompt engineering, more training on tool calling

2. **Wrong tool arguments** (ACTION=0)
   - **Symptom**: Tool called with incorrect parameters
   - **Fix**: Better parsing, more diverse training data

3. **Environment state wrong** (ENV=0)
   - **Symptom**: Tools executed but state inconsistent
   - **Fix**: Check tool execution logic, verify OpenEnv integration

4. **Missing communication** (COMMUNICATE=0)
   - **Symptom**: Agent completes task but doesn't confirm
   - **Fix**: Add confirmation prompts, train on communication examples

### Common Issues and Fixes

**Issue 1: Model generates text instead of tool calls**

```python
# Diagnosis:
# Response: "I'll create that task for you."
# Expected: <function_call>{"name": "create_task", ...}</function_call>

# Fixes:
# 1. Check system prompt includes tool format
system_prompt = build_tool_calling_system_prompt(tools)

# 2. Add few-shot examples
few_shot_examples = """
Example:
User: Create a task called "Meeting"
Assistant: <function_call>{"name": "create_task", "args": {"title": "Meeting"}}</function_call>
"""

# 3. Train on more tool calling data
```

**Issue 2: Environment state doesn't match expectations**

```python
# Diagnosis:
# ENV score = 0
# Expected: task_id="task_123" has status="completed"
# Actual: task_id="task_123" has status="pending"

# Fixes:
# 1. Check tool execution
result = env.execute_tool(tool_call)
print(f"Tool result: {result}")  # Verify success

# 2. Verify OpenEnv is properly integrated
# Make sure tools actually modify environment state

# 3. Check done() is called
# Tau2 requires explicit done() call to finalize
```

**Issue 3: Reward is always 0**

```python
# Diagnosis:
# All scores show 0.0

# Check:
# 1. Is episode ending properly?
if not (agent_called_done or user_stopped):
    # Episode didn't end correctly → reward = 0
    # Fix: Ensure done() tool is available and called

# 2. Check task_split
# Don't evaluate on 'train' split if you trained on it!
# Use task_split='test' for fair evaluation
```

**Issue 4: Parser doesn't detect tool calls**

```python
# Diagnosis:
# Model outputs: "I'll call create_task with title=Meeting"
# Parser returns: None

# Fix:
def parse_tool_call(text: str):
    # Add more robust parsing
    # Try multiple formats

    # Format 1: Tagged
    if "<function_call>" in text:
        match = re.search(r'<function_call>(.*?)</function_call>', text)
        if match:
            return json.loads(match.group(1))

    # Format 2: Plain JSON
    if '{"name":' in text:
        match = re.search(r'\{.*"name".*\}', text)
        if match:
            return json.loads(match.group(0))

    return None
```

### Example Evaluation Script

**Complete evaluation with debugging:**

```python
# examples/tau2bench/eval_with_debug.py

def evaluate_and_debug(
    model_path: str,
    domain: str,
    task_split: str = "test",
    debug_failures: bool = True,
):
    """Evaluate with automatic debugging of failures."""

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tasks = load_tasks(domain, task_split)
    results = []
    failures = []

    for task in tasks:
        env = gym.make(TAU_BENCH_ENV_ID, domain=domain, task_id=task["id"])

        # Run episode
        observation, info = env.reset()
        done = False
        messages = []

        while not done:
            prompt = build_prompt(observation, info["tools"])
            response = generate(model, tokenizer, prompt)

            messages.append({"role": "assistant", "content": response})
            observation, reward, terminated, truncated, info = env.step(response)
            messages.append({"role": "env", "content": observation})
            done = terminated or truncated

        # Record result
        result = {
            "task_id": task["id"],
            "reward": reward,
            "scores": info.get("scores", {}),
            "messages": messages
        }
        results.append(result)

        # Debug failures
        if reward < 0.5 and debug_failures:
            failures.append(result)
            print(f"\n❌ FAILED: {task['id']}")
            print(f"   Scores: {result['scores']}")
            print(f"   Last 3 turns:")
            for msg in messages[-3:]:
                print(f"   [{msg['role']}] {msg['content'][:100]}")

    # Summary
    success_rate = sum(r["reward"] > 0.5 for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    print(f"Average Reward: {sum(r['reward'] for r in results) / len(results):.3f}")

    if failures:
        print(f"\n{len(failures)} failures. Common issues:")
        action_fails = sum(1 for f in failures if f["scores"].get("ACTION", 1) == 0)
        env_fails = sum(1 for f in failures if f["scores"].get("ENV", 1) == 0)
        comm_fails = sum(1 for f in failures if f["scores"].get("COMMUNICATE", 1) == 0)
        print(f"  - ACTION failures: {action_fails}")
        print(f"  - ENV failures: {env_fails}")
        print(f"  - COMMUNICATE failures: {comm_fails}")

    return results
```

---

**Next**: Part 8 provides the complete implementation roadmap with effort estimates.
