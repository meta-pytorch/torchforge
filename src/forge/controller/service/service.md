TODO
1. introduce actor and service, their diff
2. service or actor? link xs

# Service - Distributed Actor Service Controller

A robust service orchestration system for managing distributed actor-based workloads with fault tolerance and intelligent load balancing.

## Overview

The Service class provides a unified interface for deploying and managing multiple replicas of actor-based services across distributed compute resources. It automatically handles replica lifecycle, request routing, and session management.

## Key Features

### **Fault Tolerance**
- **Health Monitoring**: Continuous health checks with automatic replica recovery
- **Request Migration**: Seamless migration of requests from failed replicas
- **Session Preservation**: Maintains session state during replica failures
- **Graceful Degradation**: Continues operation with reduced capacity

### **Routing & Load Balancing**
- **Session Affinity**: Sticky sessions for stateful workloads.
- **Per-Endpoint Routing**: Each endpoint can declare its own router (e.g., round-robin, least-loaded, custom).
- **Batch Routing**: Endpoints may aggregate multiple requests into batches before routing (via `Batcher`), improving throughput.
- **Default Routing**: If no router is specified, endpoints use round-robin.
- **Extensible Routers**: Support for pluggable router classes, e.g. `RoundRobinRouter`, `LeastLoadedRouter`, or user-defined routers.

### **Comprehensive Metrics**
- **Request Metrics**: Throughput, latency, success/failure rates
- **Capacity Metrics**: Utilization, queue depth, active requests
- **Service Metrics**: Session counts, replica health, scaling events
- **Real-time Monitoring**: Sliding window metrics for responsive scaling

### **Session Management**
- **Context-Aware Sessions**: Automatic session context propagation
- **Session Lifecycle**: Managed session creation and cleanup
- **Batching and Sessions**: Batch routing is applied only for stateless calls; sticky sessions are always preserved
## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client API    │───▶│  Service Layer   │───▶│  Replica Pool   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │                        │
                               │                        │
                               ▼                        ▼
                       ┌──────────────┐         ┌─────────────┐
                       │ Routers /    │         │ Actor Mesh  │
                       │ Batchers per │         └─────────────┘
                       │   Endpoint   │                 │
                       └──────────────┘                 │
                              │                         ▼
                              ▼                 ┌─────────────┐
                       ┌──────────────┐         │   Health    │
                       │ Autoscaler   │         │  Monitor    │
                       └──────────────┘         └─────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   Metrics    │
                       │  Collector   │
                       └──────────────┘
```

## Actor vs Service

### Actor

* The raw execution unit provided by **Monarch**.
* Lightweight, single-instance, no routing.
* In **Forge**, we almost always use `ForgeActor`, a thin extension of `monarch.Actor` that adds configuration knobs:

```python
class ForgeActor(Actor):
    procs: int = 1
    hosts: int | None = None
    with_gpus: bool = False
    num_replicas: int = 1
    _extra_config: dict[str, Any] = {}
```

Actors are ideal for singleton components like trainers, replay buffers, or datasets — if they fail, the entire job should stop.

### Service

* A **Forge abstraction built on top of actors**.
* Provides **scaling, load balancing, and fault tolerance** automatically.
* Manages multiple replicas of an actor, handling lifecycle, health checks, and request routing.

```python
class Service:
    """
    Orchestrates a pool of actor replicas with routing, health monitoring,
    and fault tolerance.
    """
```

Services are best for components that might need to scale or tolerate transient failures, such as policies, environments, or reward actors.


## Usage

### Defining an Actor

```python
from forge.controller import ForgeActor
from monarch.actor import endpoint
from forge.controller.service import service_endpoint

class MyForgeActor(ForgeActor):
    def __init__(self, ...):
        ...

    @endpoint
    async def foo(self, x):
        return x + 1

    @service_endpoint(router=RoundRobinRouter(), batch_size=8, batch_timeout=0.01)
    async def bar(self, x):
        return heavy_model(x)
```

* `@endpoint`: plain actor endpoint with no batching, use default router (round robin).
* `@service_endpoint`: enables customized routing and batching when spawned as a Service.


### Spawning as a Service

TODO: some text explain here first


```python
# With explicit options
service = await MyForgeActor.options(num_replicas=2, procs=2).as_service()
# Graceful shutdown
await service.shutdown()

# Default (1 replica, 1 process)
service = await MyForgeActor.as_service()
# Graceful shutdown
await service.shutdown()
```

**`options` for services:**

| Config key                        | Type       | Default  | Description              |
| --------------------------------- | ---------- | -------- | ------------------------ |
| `procs`                           | int        | required | Number of processes to launch for each replica of the service.    |
| `num_replicas`                    | int        | required |Number of replicas to launch for the service.       |
| `with_gpus`                       | bool       | False    | Whether to allocate GPUs for the service processes.            |
| `hosts`                           | int | None | None     | Number of hosts to allocate for each replica.        |
| `health_poll_rate`                | float      | 0.2      | Frequency (in seconds) to poll for health status.   |
| `replica_max_concurrent_requests` | int        | 10       |  Maximum number of concurrent requests per replica. |
| `return_first_rank_result`        | bool       | True     | Whether to auto-unwrap ValueMesh to the first rank's result.    |

### Spawning as a single Actor

TODO: some text explain here first
```python
# With explicit options
actor = await MyForgeActor.options(procs=1, hosts=1).as_actor()
# Graceful shutdown
await MyForgeActor.shutdown(actor)

# Default (1 proc)
actor = await MyForgeActor.as_actor()
# Graceful shutdown
await MyForgeActor.shutdown(actor)
```

**`options` for actors:**

| Config key  | Type       | Default | Description         |
| ----------- | ---------- | ------- | ------------------- |
| `procs`     | int        | required       | Number of processes to launch for each replica of the service.    |
| `with_gpus` | bool       | False   | Whether to allocate GPUs for the service processes.            |
| `hosts`     | int | None     | Number of hosts to allocate for the actor.        |





### Calling Endpoints

When you access `service.some_endpoint` or `actor.some_endpoint`, you don’t get a plain function — you get an **endpoint object** with a specific API depending on whether you spawned as a **Service** or as an **Actor**.

#### Service Endpoints (`.as_service()`)

Service endpoints are defined with `@endpoint` or `@service_endpoint` and exposed as `ServiceEndpoint` objects.

They support only two methods:

* **`.route(*args, **kwargs)`** → send the request to a single replica, selected by the configured router (round-robin, least-loaded, custom, etc.).
* **`.fanout(*args, **kwargs)`** → broadcast the request to all healthy replicas and return a list of results.

Other Monarch actor APIs (`call`, `call_one`, `choose`, etc.) are not supported on services.

**Example:**

```python
# Route to one replica (load-balanced)
result = await service.predict.route(x)

# Broadcast to all replicas
results = await service.predict.fanout(x)
```

#### Actor Endpoints (`.as_actor()`)

When you spawn a raw actor via `.as_actor()`, its functions are [**Monarch `ActorEndpoint` objects**](https://github.com/meta-pytorch/monarch/blob/0a842ad3c4abfb3983eb7ba55ec327c42553ff44/python/monarch/_src/actor/actor_mesh.py#L279).

Highlights of supported APIs:
- **`.choose(...)`**: load-balanced call to one actor (may pick any if >1 exists).
- **`.call_one(...)`**: load-balanced call to one actor. Only valid if exactly one actor exists, otherwise raises error.
- **`.call(...)`** – call all actors, returns a `ValueMesh` of results
- ...


**Example:**

```python
# Route to one replica
result = await service.predict.call_one(x)

# Broadcast to all replicas
results = await service.predict.call(x)
```

### Session-Based Calls
```python
# Context manager for session lifecycle
async with service.session() as session:
    result1 = await service.my_endpoint.route(arg1, arg2)
    result2 = await service.another_endpoint.fanout(arg3)
    # Session automatically terminated on exit

# Manual session management
session_id = await service.start_session()
result = await service.my_endpoint.route(arg1, arg2, sess_id=session_id)
await service.terminate_session(session_id)
```

### Stateless Calls

```python
# Direct calls without sessions (uses the default router unless overridden)
result = await service.my_endpoint.route(arg1, arg2)
```


### Batch Routing
```python
class MyActor(ForgeActor):
    @service_endpoint(router=RoundRobinRouter(), batch_size=8, batch_timeout=0.01)
    async def predict(self, x):
        return model(x)

service = await MyActor.options(num_replicas=4).as_service()

# 16 requests → grouped into 2 batches of 8, routed round-robin across replicas
await asyncio.gather(*(service.predict.route(data) for data in inputs))
```
**Note:**
- Setting `batch_size=1` or not passing `batch_size` disables batching. Each request will be routed immediately using the underlying router.
- Supported routers: `RoundRobinRouter()`, `LeastLoadedRouter()`, or customized routers.
### Custom Routing

To implement custom routing logic, define your own `Router` by subclassing and overriding
the `get_replica` method. Then pass an instance to `@service_endpoint`.

```python
class MyCustomRouter(Router):
    """Example custom router: always picks the first replica."""

    def get_replica(self, healthy_replicas, sess_id=None, session_map=None):
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available")
        return healthy_replicas[0]


class MyActor(ForgeActor):
    @service_endpoint(router=MyCustomRouter())
    async def predict(self, x):
        return model(x)
```

### Monitoring and Metrics

```python
# Get detailed metrics
metrics = service.get_metrics()
print(f"Total request rate: {metrics.get_total_request_rate()}")
print(f"Average queue depth: {metrics.get_avg_queue_depth()}")
print(f"Capacity utilization: {metrics.get_avg_capacity_utilization(service._replicas)}")
print(f"Average sessions per replica: {metrics.get_sessions_per_replica()}")

# Get summary for monitoring dashboards
summary = service.get_metrics_summary()
print(f"Healthy replicas: {summary['service']['healthy_replicas']}")
print(f"Total sessions: {summary['service']['total_sessions']}")

# Per-replica metrics
for replica_idx, replica_metrics in summary['replicas'].items():
    print(f"Replica {replica_idx}: {replica_metrics['request_rate']:.1f} req/s")
```

**Service-level metrics:** sessions, healthy replicas, request rate, queue depth, capacity.
**Replica-level metrics:** request counts, latency, active requests, queue depth, session assignments.


TODO: double check correctness with code
#### Service-Level Metrics (`summary['service']`)
- **`total_sessions`**: Number of active sessions
- **`healthy_replicas`**: Number of operational replicas
- **`total_replicas`**: Total number of replicas
- **`total_request_rate`**: Requests per second across all replicas
- **`avg_queue_depth`**: Average pending requests per replica
- **`avg_capacity_utilization`**: Average resource usage across replicas
- **`sessions_per_replica`**: Distribution of sessions across replicas

#### Replica-Level Metrics (`summary['replicas'][replica.idx]`)
- **`total_requests`**: Total, successful, and failed requests
- **`successful_requests`**: Number of successful requests
- **`failed_requests`**: Number of failed requests
- **`request_rate`**: Requests per second (sliding window)
- **`avg_latency`**: Response time (sliding window)
- **`active_requests`**: Currently processing requests
- **`queue_depth`**: Pending requests in queue
- **`assigned_sessions`**: Number of sessions assigned to replica
- **`capacity_utilization`**: Current load vs maximum capacity



## Performance Characteristics

- **Low Latency**: Sub-millisecond request routing overhead
- **High Throughput**: Concurrent request processing across replicas
- **Elastic Scaling**: Responsive to traffic patterns with configurable thresholds
- **Batching**: Amortizes routing decisions across multiple requests, improving throughput under high load
- **Resource Efficient**: Intelligent replica management and load balancing
- **Fault Resilient**: Automatic recovery from replica failures
- **Session Aware**: Maintains state consistency for stateful workloads
## Best Practices

### Configuration
- Set `min_replicas` based on baseline load requirements
- Configure `max_replicas` based on resource constraints
- Tune autoscaling thresholds based on workload characteristics
- Use appropriate cooldown periods to prevent scaling oscillation

### Session Management
- Use sessions for stateful workloads requiring consistency
- Prefer stateless calls for better load distribution
- Implement custom routing for specialized workload requirements

### Monitoring
- Monitor key metrics: request rate, queue depth, capacity utilization
- Set up alerts for unhealthy replicas and scaling events
- Track session distribution for load balancing effectiveness

### Error Handling
- Implement proper error handling in actor endpoints
- Use try-catch blocks around service calls
- Monitor failed request rates for service health

## Dependencies

- `monarch.actor`: Actor framework for distributed computing
- `recoverable_mesh`: Fault-tolerant process mesh management
- `asyncio`: Asynchronous I/O support
- `contextvars`: Context variable support for session management

## Thread Safety

The Service class is designed for use in asyncio environments and is not thread-safe. All operations should be performed within the same event loop.
