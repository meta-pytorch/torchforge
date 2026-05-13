# DNS-AID Service Discovery

Forge services can optionally register DNS-AID SVCB records on startup, enabling
peer discovery via DNS rather than hard-coded coordinator addresses.

## Installation

```bash
pip install forge[dns-aid]
```

## Configuration

DNS-AID requires **both** the `DNS_AID_ENABLED` environment variable and
the per-service `DnsAidConfig.enabled` flag to be true. This dual-guard
means the environment variable acts as a global kill switch.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DNS_AID_ENABLED` | `false` | Global toggle. Must be `true` for any DNS-AID operations. |
| `DNS_AID_ZONE` | — | DNS zone suffix (e.g. `_agents.svc.cluster.local`) |
| `DNS_AID_SERVER` | — | DNS server address (e.g. `10.0.0.53`) |
| `DNS_AID_PORT` | `853` | DNS server port |
| `DNS_AID_BACKEND` | — | DNS backend (`route53`, `cloudflare`, `ddns`, `mock`, etc.) |

### Per-Service Configuration

Add `DnsAidConfig` to your actor options:

```python
from forge.controller import ForgeActor
from forge.types import DnsAidConfig

dns_cfg = DnsAidConfig(
    enabled=True,
    name="generator",           # DNS service name (default: class name)
    domain="forge.internal",    # DNS domain
    port=8080,                  # Externally reachable port (required)
    ttl=30,                     # Record TTL in seconds
    capabilities=["gpu:8"],     # Extra capabilities to advertise
    category="rl-training",     # Discovery category
)

service = await MyGenerator.options(
    num_replicas=4,
    procs=2,
    with_gpus=True,
    dns_aid=dns_cfg,
).as_service(model_path="...")
```

The `port` field is required when `enabled` is True. It should be set to
the port that external systems use to reach this service (e.g. a load
balancer, gateway, or sidecar proxy port). Monarch services communicate
via actor RPC internally, so there is no auto-detected listener port.

### OmegaConf YAML

```yaml
# Requires DNS_AID_ENABLED=true in the environment
generator:
  procs: 2
  num_replicas: 4
  with_gpus: true
  dns_aid:
    enabled: true
    name: generator
    domain: forge.internal
    port: 8080
    ttl: 30
    capabilities:
      - "gpu:8"
      - "shard_count:4"
```

## How It Works

1. **Startup**: After the service is fully initialized, `publish_service()` creates
   a DNS-AID SVCB record advertising the service's hostname, port, role, and
   capabilities.

2. **Discovery**: Other services can call `discover_peers()` to find registered
   peers by name. Discovery retries with exponential backoff (max 5 attempts)
   to handle race conditions during cluster startup. Pass `retry_on_empty=False`
   if you want to return immediately when no peers are found.

3. **Shutdown**: `unpublish_service()` removes the DNS record. This is best-effort;
   if the process crashes, the record expires after the configured TTL (default 30s).

## Peer Discovery Example

```python
from forge.controller.dns_aid import discover_peers
from forge.types import DnsAidConfig

cfg = DnsAidConfig(enabled=True, domain="forge.internal")

# Find all trainer services (retries if not yet registered)
trainers = await discover_peers("trainer", cfg)
for agent in trainers:
    print(f"Found trainer at {agent.target_host}:{agent.port}")

# Check once without retrying
trainers = await discover_peers("trainer", cfg, retry_on_empty=False)
```

## Backward Compatibility

DNS-AID is fully opt-in. When `DNS_AID_ENABLED` is unset or `false` (the default),
no DNS operations are performed and the `dns-aid` package does not need to be
installed. Existing deployments are completely unaffected.
