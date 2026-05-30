# OpenEnv - Generic GRPO Training Framework

A centralized framework for training language models on any OpenEnv task using GRPO (Grouped Relative Policy Optimization) or DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization).

## Key Features

- **GenericEnvClient**: Works with ANY OpenEnv Docker image without requiring environment-specific packages locally
- **GenericAction**: Simple dict wrapper that maps to environment-specific actions at runtime
- **Single Main Script**: One `main.py` works for all OpenEnv tasks
- **Circuit Breaker Pattern**: Automatic detection and restart of unhealthy Docker containers
- **Episode Dropout**: Configurable filtering of low-quality training batches
- **GRPO/DAPO Loss**: Switchable loss functions with configurable parameters
- **Parallel Evaluation**: Multiple env_actors for isolated, parallel reward evaluation

## Folder Structure

```
apps/openenv/
  ├── main.py              # Generic training script (use this)
  ├── julia_utils.py       # Julia task utilities (GenericAction)
  ├── python_utils.py      # Python task utilities (GenericAction)
  ├── llama3_8b_julia.yaml # Julia training config
  ├── llama3_8b_coding.yaml# Python coding training config
  └── README.md                    # This file
```

## Quick Start

### Run Julia Training

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_julia.yaml
```

### Run Python Coding Training

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_coding.yaml
```

## YAML Configuration

### Minimal Configuration

Each task config needs at minimum:

```yaml
# Task-specific configuration
task:
  env_name: "julia"  # Environment name
  build_action: !function apps.openenv.julia_utils.build_julia_action
  evaluate_response: !function apps.openenv.julia_utils.evaluate_julia_response
  transform_sample: !function apps.openenv.julia_utils.transform_julia_sample

# OpenEnv configuration - only docker_image is required!
openenv_config:
  docker_image: "julia-env:latest"
```

### Full Configuration Reference

```yaml
# Global configuration
group_size: 8                    # Number of responses per prompt
batch_size: 2                    # Batches per training step
max_req_tokens: 1024             # Max prompt tokens
max_res_tokens: 1024             # Max response tokens
model: "path/to/model"           # Model path
off_by_n: 1                      # Max policy version age for episodes

# Loss configuration (GRPO or DAPO)
grpo:
  loss_type: grpo                # "grpo" or "dapo"
  clip_eps_low: 0.2              # Lower clipping bound
  clip_eps_high: 0.28            # Upper clipping bound
  agg_type: fixed_horizon        # "fixed_horizon" (GRPO) or "token_mean" (DAPO)
  beta: 0.1                      # KL penalty (GRPO only)
  dual_clip_c: 3.0               # Dual-clip constant (DAPO only)

# Episode dropout configuration
episode_dropout:
  enable_variance_dropout: true  # Drop low-variance batches
  enable_truncation_dropout: true # Drop batches with truncated responses
  variance_threshold: 0.001      # Std threshold for variance dropout

# Main loop configuration
rollout_threads: 1               # Parallel rollout threads
evaluation_timeout_s: 20.0       # Timeout for environment evaluation

# Circuit breaker configuration
circuit_breaker:
  threshold: 5                   # Timeouts before tripping
  window_s: 60.0                 # Time window for counting timeouts
  cooldown_s: 60.0               # Wait time after container restart

# Task configuration
task:
  env_name: "julia"
  build_action: !function apps.openenv.julia_utils.build_julia_action
  evaluate_response: !function apps.openenv.julia_utils.evaluate_julia_response
  transform_sample: !function apps.openenv.julia_utils.transform_julia_sample

# Dataset configuration
dataset:
  path: "path/to/dataset.parquet"  # Supports .parquet, .json, or HF datasets
  data_split: "train"
  streaming: false

# OpenEnv configuration
openenv_config:
  docker_image: "julia-env:latest"
  container_timeout_s: 180.0      # Container startup timeout
  container_memory_gb: 1024       # Container memory limit
  port: 8000                      # Starting port for containers
  num_env_actors: 2               # Number of parallel reward actors
  num_containers: 2               # Containers per actor
  num_connections: 12             # WebSocket connections per container
  request_timeout_s: 20.0         # Per-request timeout
  env_vars:                       # Environment variables for containers
    JULIA_EXECUTION_TIMEOUT: "15"
    JULIA_MAX_WORKERS: "16"
```

## Adding a New Language

To add support for a new language (e.g., Rust):

### 1. Create Utils File

Create `apps/openenv/rust_utils.py`:

```python
from typing import Any, Dict
from openenv import GenericAction
from forge.observability.metrics import record_metric, Reduce


def get_rust_system_prompt() -> str:
    """Get system prompt for Rust coding tasks."""
    return """You are an expert Rust programmer.
Write correct, safe Rust code that compiles and runs.
""".strip()


def build_rust_prompt(sample: Dict[str, Any], tokenizer) -> str:
    """Build prompt for Rust code generation."""
    system_prompt = get_rust_system_prompt()
    request = sample.get("prompt", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_rust_action(response: str, sample: Dict[str, Any]) -> GenericAction:
    """Build GenericAction from model response."""
    code = extract_rust_code(response)
    test_code = sample.get("target", "")

    # GenericAction fields must match what RustEnv expects
    return GenericAction(
        code=code,
        test_code=test_code,
    )


def evaluate_rust_response(result, response: str, sample: Dict[str, Any]) -> float:
    """Evaluate Rust code execution and return reward."""
    obs = result.observation
    if isinstance(obs, dict):
        exit_code = obs.get("exit_code", -1)
    else:
        exit_code = obs.exit_code

    reward = 1.0 if exit_code == 0 else 0.0
    record_metric("reward/rust/reward", reward, Reduce.MEAN)
    return reward


def extract_rust_code(response: str) -> str:
    """Extract Rust code from markdown blocks."""
    import re
    pattern = r"```rust\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def transform_rust_sample(sample: Dict[str, Any], tokenizer) -> Dict[str, Any] | None:
    """Transform dataset sample for Rust tasks."""
    if not sample.get("prompt"):
        return None

    return {
        "request": build_rust_prompt(sample, tokenizer),
        "target": sample.get("test", ""),
        "task_id": sample.get("task_id", ""),
    }
```

### 2. Create YAML Config

Create `apps/openenv/llama3_8b_rust.yaml`:

```yaml
# Rust training config using GenericEnvClient
group_size: 8
batch_size: 2
max_req_tokens: 1024
max_res_tokens: 1024
model: "path/to/model"

grpo:
  loss_type: grpo
  clip_eps_low: 0.2
  clip_eps_high: 0.28
  beta: 0.1

task:
  env_name: "rust"
  build_action: !function apps.openenv.rust_utils.build_rust_action
  evaluate_response: !function apps.openenv.rust_utils.evaluate_rust_response
  transform_sample: !function apps.openenv.rust_utils.transform_rust_sample

dataset:
  path: "path/to/rust/dataset"
  data_split: "train"

openenv_config:
  docker_image: "rust-env:latest"
  container_timeout_s: 180.0
  num_env_actors: 2
  num_containers: 2
  num_connections: 8

# ... rest of config (copy from existing configs)
```

### 3. Run Training

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_rust.yaml
```

## Task Utils API

Each task utils file should implement these functions:

### Required Functions

1. **`build_<env>_action(response: str, sample: dict) -> GenericAction`**
   - Builds GenericAction from model response
   - GenericAction fields must match what the environment expects

2. **`evaluate_<env>_response(result, response: str, sample: dict) -> float`**
   - Evaluates execution result and returns reward (0.0 to 1.0)
   - Works with both typed observations and raw dicts

3. **`transform_<env>_sample(sample: dict, tokenizer) -> dict | None`**
   - Transforms raw dataset sample into training format
   - Returns dict with 'request', 'target', 'task_id' or None if invalid

### Optional Helper Functions

- **`get_<env>_system_prompt() -> str`**: Get system prompt for the language
- **`build_<env>_prompt(sample: dict, tokenizer) -> str`**: Build formatted prompt
- **`extract_<env>_code(response: str) -> str`**: Extract code from markdown

## Architecture

### GenericEnvClient

The `OpenEnvActor` manages Docker containers and WebSocket connections:

```
┌─────────────────────────────────────────────────────────────┐
│                    GenericRewardActor                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ env_actor_0 │  │ env_actor_1 │  │ env_actor_2 │  ...     │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │          │
│  │ │Container│ │  │ │Container│ │  │ │Container│ │          │
│  │ │  WS x12 │ │  │ │  WS x12 │ │  │ │  WS x12 │ │          │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

- Each `env_actor` manages its own container pool
- Circuit breaker isolates failures per actor
- Unhealthy actors trigger automatic container restart

### Circuit Breaker

The circuit breaker pattern prevents cascading failures:

1. **Closed**: Normal operation, requests flow through
2. **Open**: Too many timeouts detected, actor marked unhealthy
3. **Half-Open**: After cooldown, actor retries with fresh container

Configuration:
```yaml
circuit_breaker:
  threshold: 5       # Timeouts before opening
  window_s: 60.0     # Counting window
  cooldown_s: 60.0   # Time before retry
```

### Episode Dropout

Batches are filtered based on quality:

1. **Variance Dropout**: Drops batches where all rewards are similar (e.g., all 0 or all 1)
2. **Truncation Dropout**: Drops batches with truncated responses (hit max_tokens)

This prevents training on uninformative gradients.


## Observability

### Metrics

Key metrics tracked:
- `reward/*/avg_reward`: Average reward per task
- `reward/*/pass_rate`: Test pass rate
- `circuit_breaker/*/tripped`: Circuit breaker activations
- `episode/avg_response_tokens`: Average response length
- `training/weight_update_duration_s`: Weight sync time

### Logging

Set log level via environment variable:
```bash
LOG_LEVEL=DEBUG python -m apps.openenv.main --config ...
```

### Weights & Biases

Configure in YAML:
```yaml
metric_logging:
  wandb:
    entity: "your-team"
    project: "your-project"
    logging_mode: global_reduce
```

## Performance Tuning

### GPU Memory

Enable expandable segments (set automatically in main.py):
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### Timeout Configuration

For Julia (with internal worker pool):
```yaml
# Julia kills workers at 15s, we wait 20s to allow recovery
evaluation_timeout_s: 20.0
openenv_config:
  request_timeout_s: 20.0
  env_vars:
    JULIA_EXECUTION_TIMEOUT: "15"
```

### Buffer Starvation

If training stalls waiting for episodes:
1. Increase `off_by_n` to accept older episodes
2. Increase `rollout_threads` for more parallel generation
3. Increase policy `num_replicas` for more generation capacity

Environment variables for debugging:
```bash
FORGE_MAX_EMPTY_BUFFER_WAIT_S=120  # Max wait before error
FORGE_BACKPRESSURE_TIMEOUT_S=30   # Max backpressure wait
```

## Debugging

### Common Issues

1. **No code extracted**: Model not following format
   - Check system prompt in utils file
   - Verify `extract_*_code()` handles model output format

2. **All evaluations timeout**: Container issues
   - Check container logs
   - Reduce `num_connections` to prevent overload
   - Increase `container_memory_gb`

3. **Circuit breaker keeps tripping**: Environment instability
   - Increase `threshold` for more tolerance
   - Check for memory leaks in environment
   - Add more `num_containers` for redundancy

4. **Buffer starvation**: Training faster than rollouts
   - Increase `off_by_n` (accept older episodes)
   - Increase `rollout_threads`
   - Add more policy replicas
