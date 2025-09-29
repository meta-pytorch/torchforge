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
- **Routing Hints**: Custom session routing based on workload characteristics
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

## Usage

### Basic Service Setup

```python
# Pre-configure a service with multiple replicas
service = await MyForgeActor.options(num_replicas=2, procs=2).as_service(...)
await service.shutdown()

# Default usage without calling options
service = await MyForgeActor.as_service(...)
await service.shutdown()

# Pre-configure a single actor
actor = await MyForgeActor.options(procs=1, hosts=1).as_actor(...)
await actor.shutdown()

# Default usage without calling options
actor = await MyForgeActor.as_actor(...)
await actor.shutdown()
```

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
- **`.call_one(...)`** – RPC to a single actor (preferred for single-replica calls, stricter than `choose`)
- **`.call(...)`** – RPC to all actors, returns a `ValueMesh` of results
- ...


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

# Get summary for monitoring dashboards
summary = service.get_metrics_summary()
print(f"Healthy replicas: {summary['service']['healthy_replicas']}")
print(f"Total sessions: {summary['service']['total_sessions']}")

# Per-replica metrics
for replica_idx, replica_metrics in summary['replicas'].items():
    print(f"Replica {replica_idx}: {replica_metrics['request_rate']:.1f} req/s")
```

### Graceful Shutdown

```python
# Stop the service and all replicas
await service.shutdown()
```

For single actor:
```python
await MyForgeActor.shutdown(actor)
```

## Configuration

### ServiceConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpus_per_replica` | int | Number of GPUs allocated per replica |
| `min_replicas` | int | Minimum number of replicas to maintain |
| `max_replicas` | int | Maximum number of replicas allowed |
| `default_replicas` | int | Initial number of replicas to start |
| `replica_max_concurrent_requests` | int | Maximum concurrent requests per replica |
| `health_poll_rate` | float | Health check frequency in seconds |
| `return_first_rank_result` | bool | Auto-unwrap ValueMesh to first rank's result |
| `autoscaling` | AutoscalingConfig | Autoscaling configuration |

### AutoscalingConfig

#### Scale Up Triggers
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_up_queue_depth_threshold` | 5.0 | Average queue depth to trigger scale up |
| `scale_up_capacity_threshold` | 0.8 | Capacity utilization to trigger scale up |
| `scale_up_request_rate_threshold` | 10.0 | Requests/sec to trigger scale up |

#### Scale Down Triggers
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_down_capacity_threshold` | 0.3 | Capacity utilization to trigger scale down |
| `scale_down_queue_depth_threshold` | 1.0 | Average queue depth to trigger scale down |
| `scale_down_idle_time_threshold` | 300.0 | Seconds of low utilization before scale down |

#### Timing Controls
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_time_between_scale_events` | 60.0 | Minimum seconds between scaling events |
| `scale_up_cooldown` | 30.0 | Cooldown after scale up |
| `scale_down_cooldown` | 120.0 | Cooldown after scale down |

#### Scaling Behavior
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_up_step_size` | 1 | How many replicas to add at once |
| `scale_down_step_size` | 1 | How many replicas to remove at once |

#### Safety Limits
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_queue_depth_emergency` | 20.0 | Emergency scale up threshold |
| `min_healthy_replicas_ratio` | 0.5 | Minimum ratio of healthy replicas |

## Metrics

### Service-Level Metrics
- **Total Sessions**: Number of active sessions
- **Healthy Replicas**: Number of operational replicas
- **Total Request Rate**: Requests per second across all replicas
- **Average Queue Depth**: Average pending requests per replica
- **Average Capacity Utilization**: Average resource usage across replicas
- **Sessions Per Replica**: Distribution of sessions across replicas

### Replica-Level Metrics
- **Request Counts**: Total, successful, and failed requests
- **Request Rate**: Requests per second (sliding window)
- **Average Latency**: Response time (sliding window)
- **Active Requests**: Currently processing requests
- **Queue Depth**: Pending requests in queue
- **Assigned Sessions**: Number of sessions assigned to replica
- **Capacity Utilization**: Current load vs maximum capacity

## Use Cases

### ML Model Serving
```python
# High-throughput model inference with automatic scaling
config = ServiceConfig(
    gpus_per_replica=1,
    min_replicas=2,
    max_replicas=20,
    default_replicas=4,
    replica_max_concurrent_requests=8,
    autoscaling=AutoscalingConfig(
        scale_up_capacity_threshold=0.7,
        scale_up_queue_depth_threshold=3.0
    )
)
service = Service(config, ModelInferenceActor, model_path="/path/to/model")
```

### Batch Processing
```python
# Parallel job execution with fault tolerance
config = ServiceConfig(
    gpus_per_replica=2,
    min_replicas=1,
    max_replicas=10,
    default_replicas=3,
    replica_max_concurrent_requests=5,
    autoscaling=AutoscalingConfig(
        scale_up_queue_depth_threshold=10.0,
        scale_down_idle_time_threshold=600.0
    )
)
service = Service(config, BatchProcessorActor, batch_size=100)
```

### Real-time Analytics
```python
# Stream processing with session affinity
config = ServiceConfig(
    gpus_per_replica=1,
    min_replicas=3,
    max_replicas=15,
    default_replicas=5,
    replica_max_concurrent_requests=20,
    autoscaling=AutoscalingConfig(
        scale_up_request_rate_threshold=50.0,
        scale_up_capacity_threshold=0.6
    )
)
service = Service(config, StreamProcessorActor, window_size=1000)
```

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
