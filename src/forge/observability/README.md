# Observability in Forge

We aim to make distributed observability effortless. You can call `record_metric(key, val, reduce_type)` from anywhere, and it just works. We also provide memory/performance tracers, plug-and-play logging backends, and reduction types. You can visualize aggregated results globally, per-rank or as a stream. No boilerplate required - just call, flush, and visualize. Disable with `FORGE_DISABLE_METRICS=true`.

## 1. Your Superpowers

### 1.1 Call `record_metric` from Anywhere

Simple to use, with no need to pass dictionaries around. For example, users can simply write:

```python
def my_fn():
    record_metric(key, value, reduce)
```

Instead of:

```python
def my_fn(my_metrics):
    my_metrics[key] = value
    return my_metrics
```

Simple example (for a distributed one, check the next section)
```python
import asyncio
from forge.observability import get_or_create_metric_logger, record_metric, Reduce

async def main():
    # Setup logger
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one({"console": {"logging_mode": "global_reduce"}})

    # Have this in any process
    def my_fn(number):
        record_metric("my_sum_metric", number, Reduce.SUM)   # sum(1,2,3)
        record_metric("my_max_metric", number, Reduce.MAX)   # max(1,2,3)
        record_metric("my_mean_metric", number, Reduce.MEAN) # mean(1,2,3)

    # Accumulate metrics
    for number in range(1, 4): # 1, 2, 3
        my_fn(number)

    # Flush
    await mlogger.flush.call_one(global_step=0)

    # Shutdown when done
    await mlogger.shutdown.call_one()

if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```bash
=== [GlobalReduce] - METRICS STEP 0 ===
my_sum_metric:  6.0
my_max_metric:  3.0
my_mean_metric: 2.0
```

### 1.2 Track Performance: Timing and Memory

Use `Tracer` for tracking durations and memory usage. Overhead is minimal, and GPU timing is non-blocking. Set `timer="gpu"` for kernel-level precision. Tracer leverages `record_metric` in the backend.

```python
from forge.observability.perf_tracker import Tracer
import torch

# ... Initialize logger (as shown in previous example)

def my_fn():
    a = torch.randn(1000, 1000, device="cuda")

    t = Tracer(prefix="my_cuda_loop", track_memory=True, timer="gpu")
    t.start()
    for _ in range(3):
        torch.mm(a, a)
        t.step("my_metric_mm")
    t.stop()

# Accumulate metrics
for _ in range(2):
    my_fn()

await mlogger.flush(global_step=0) # Flush and reset
```

Output:
```bash
=== [GlobalReduce] - METRICS STEP 0 ===
my_cuda_loop/memory_delta_end_start_avg_gb:   0.015
my_cuda_loop/memory_peak_max_gb:              0.042
my_cuda_loop/my_metric_mm/duration_avg_s:     0.031
my_cuda_loop/my_metric_mm/duration_max_s:     0.186
my_cuda_loop/total_duration_avg_s:            0.094
my_cuda_loop/total_duration_max_s:            0.187
```

For convenience, you can also use `Tracer` as a context manager or decorator:

```python
from forge.observability.perf_tracker import trace

with trace(prefix="train_step", track_memory=True, timer="gpu") as t:
    t.step("fwd")
    loss = model(x)
    t.step("bwd")
    loss.backward()

@trace(prefix="my_reward_fn", track_memory=False, timer="cpu")
async def reward_fn(x):  # Supports both sync/async functions
    return 1.0 if x > 0 else 0.0
```
## 2. Logging Modes

Defined per backend. You have three options:

- **global_reduce**: N ranks = 1 chart. Reduces metrics across all ranks. Ideal for a single aggregated view (e.g., average loss chart).
- **per_rank_reduce**: N ranks = N charts. Each rank reduces locally and logs to its own logger. Ideal for per-rank performance debugging (e.g., GPU utilization).
- **per_rank_no_reduce**: N ranks = N charts.  Each rank streams to its own logger without reduction. Ideal for real-time streams.

Consider an example with an actor running on 2 replicas, each with 2 processes, for a total of 4 ranks. We will record the sum of the rank values. For example, rank_0 records 0, and rank_1 records 1.

```python
import asyncio

from forge.controller.actor import ForgeActor
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
from monarch.actor import current_rank, endpoint

# Your distributed actor
class MyActor(ForgeActor):
    @endpoint
    async def my_fn(self):
        rank = current_rank().rank # 0 or 1 per replica
        record_metric("my_sum_rank_metric", rank, Reduce.SUM) # <--- your metric

async def main():
    # Setup logger
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(
        {"console": {"logging_mode": "global_reduce"}} #  <--- Define logging_mode here
    )

    # Setup actor
    service_config = {"procs": 2, "num_replicas": 2, "with_gpus": False}
    my_actor = await MyActor.options(**service_config).as_service()

    # Accumulate metrics
    for _ in range(2):  # 2 steps
        await my_actor.my_fn.fanout()

    # Flush
    await mlogger.flush.call_one(global_step=0)  # Flush and reset

if __name__ == "__main__":
    asyncio.run(main())
```

Output when `"logging_mode": "global_reduce"`
```bash
=== [GlobalReduce] - METRICS STEP 0 ===
my_sum_rank_metric: 4.0 # (0 + 1) * 2 steps * 2 replicas
===============
```

Now, let’s set `"logging_mode": "per_rank_reduce"`:
```bash
# replica 1
=== [MyActor_661W_r0] - METRICS STEP 0 ===
my_sum_rank_metric: 0.0 # (rank_0) * 2 steps
===============
=== [MyActor_661W_r1] - METRICS STEP 0 ===
my_sum_rank_metric: 2.0 # (rank_1) * 2 steps
===============

# replica 2
=== [MyActor_wQ1g_r0] - METRICS STEP 0 ===
my_sum_rank_metric: 0.0 # (rank_0) * 2 steps
===============
=== [MyActor_wQ1g_r1] - METRICS STEP 0 ===
my_sum_rank_metric: 2.0 # (rank_1) * 2 steps
===============
```

Finally, with `"logging_mode": "per_rank_no_reduce"`, we have a stream with no reduction:
```bash
[0] [MyActor-0/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 0
[0] [MyActor-0/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 0
[1] [MyActor-1/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 1
[1] [MyActor-1/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 1
[0] [MyActor-0/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 0
[0] [MyActor-0/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 0
[1] [MyActor-1/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 1
[1] [MyActor-1/2] 2025-10-10 12:21:09 INFO my_sum_rank_metric: 1
```

## 3. Using Multiple Backends

For example, you can do `global_reduce` with Weights & Biases while using `per_rank_no_reduce` for debugging logs on the console.

```python
mlogger = await get_or_create_metric_logger(process_name="Controller")
await mlogger.init_backends.call_one({
    "console": {"logging_mode": "per_rank_no_reduce"},
    "wandb": {"logging_mode": "global_reduce"}
})
```

### 3.1 Adding a New Backend

Extend `LoggerBackend` for custom logging, such as saving data to JSONL files, sending Slack notifications when a metric hits a threshold, or supporting tools like MLFlow or Grafana. After writing your backend, register it with `forge.observability.metrics.get_logger_backend_class`.

```python
# Example of a custom backend
class ConsoleBackend(LoggerBackend):
    def __init__(self, logger_backend_config: dict[str, Any]) -> None:
        super().__init__(logger_backend_config)

    async def init(self, process_name: str | None = None, *args, **kwargs) -> None:
        self.process_name = process_name

    async def log_batch(self, metrics: list[Metric], global_step: int, *args, **kwargs) -> None:
        # Called on flush
        print(self.process_name, metrics)

    def log_stream(self, metric: Metric, global_step: int, *args, **kwargs) -> None:
        # Called on `record_metric` if "logging_mode": "per_rank_no_reduce"
        print(metric)
```

## 4. Adding a New Reduce Type

Metrics are accumulated each time `record_metric` is called. The following example implements the `Reduce.MEAN` accumulator. Users can extend this by adding custom reduce types, such as `WordCounterAccumulator` or `SampleAccumulator`, and registering them with `forge.observability.metrics.Reduce`. For details on how this is used, see `forge.observability.metrics.MetricCollector`.


```python
# Example of a custom reduce type
class MeanAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.sum = 0.0
        self.count = 0
        self.is_reset = True

    def append(self, value: Any) -> None:
        # Called after record_metric(key, value, reduce.TYPE)
        v = float(value.item() if hasattr(value, "item") else value)
        self.sum += v
        self.count += 1

    def get_value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def get_state(self) -> dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "sum": self.sum, "count": self.count}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        # Useful for global reduce; called before flush
        total_sum = sum(s["sum"] for s in states)
        total_count = sum(s["count"] for s in states)
        return total_sum / total_count if total_count > 0 else 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.is_reset = True
```

## 5. Behind the Scenes

We have two main requirements:
1. Metrics must be accumulated somewhere.
2. Metrics must be collected from all ranks.

To address #1, we use a `MetricCollector` per process to store state. For example, with 10 ranks, there are 10 `MetricCollector` instances. Within each rank, `MetricCollector` is a singleton, ensuring the same object is returned after the first call. This eliminates the need to pass dictionaries between functions.

To address #2, we automatically spawn a `LocalFetcherActor` for each process mesh and register it with the `GlobalLoggingActor`. This allows the `GlobalLoggingActor` to know which processes to call, and each `LocalFetcherActor` can access the local `MetricCollector`. This spawning and registration occurs in `forge.controller.provisioner.py::get_proc_mesh`.

The flow is generally:
GlobalLoggingActor.method() -> per-procmesh LocalFetcherActor.method() -> per-rank MetricCollector.method() -> logger

So you may ask: "what about the logging backends"? They live in two places:
- In each MetricCollector if the backend is marked as per_rank.
- In the GlobalLoggingActor if the backend is marked as global_reduce.

In summary:
1. One `GlobalLoggingActor` serves as the controller.
2. For each process, `forge.controller.provisioner.py::get_proc_mesh` spawns a `LocalFetcherActor`, so N ranks = N `LocalFetcherActor` instances. These are registered with the `GlobalLoggingActor`.
3. Each rank has a singleton `MetricCollector`, holding accumulated metrics and per_rank backends.
4. Calling `record_metric(key, value, reduce_type)` stores metrics locally in the `MetricCollector`.
5. When GlobalLoggingActor.flush() -> all LocalFetcherActor.flush() --> MetricCollector.flush()
