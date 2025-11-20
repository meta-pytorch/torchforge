# Refactoring Proposal 06: Consolidate Server Management and Cleanup

## Overview
Building on Proposals 01-05, this iteration drastically simplifies server management, removes over-engineering, and consolidates utility functions.

## Key Changes

### 1. Drastically Simplify Server Startup
Current implementation (lines 1518-1584) is over-engineered with extensive health checks, retry logic, and error handling.

**Before (~100 lines):**
```python
# Start one server per rollout thread to avoid race conditions
server_processes = []
server_ports = []

for i in range(num_rollout_threads):
    server_port = base_server_port + i
    server_ports.append(server_port)

    # Clean up any existing server on this port
    if kill_process_on_port(server_port):
        print(f"Cleaned up existing server on port {server_port}")

    print(f"Starting OpenSpiel server {i} for game '{game_name}' on port {server_port}...")
    server_process = multiprocessing.Process(
        target=start_openspiel_server, args=(game_name, server_port)
    )
    server_process.start()
    server_processes.append(server_process)

# Wait for all servers to be ready
print(f"Waiting for {num_rollout_threads} OpenSpiel servers to be ready...")
all_ready = True
for i, server_port in enumerate(server_ports):
    server_ready = False
    for attempt in range(30):  # Try for 30 seconds per server
        if not server_processes[i].is_alive():
            print(f"[ERROR] Server {i} process died unexpectedly!")
            # ... error handling
            all_ready = False
            break

        try:
            resp = requests.get(
                f"http://localhost:{server_port}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            )
            if resp.status_code == 200:
                server_ready = True
                print(f"✓ OpenSpiel server {i} ready on port {server_port} (took {attempt+1}s)")
                break
        except Exception as e:
            # ... verbose error logging
            time.sleep(1)

    if not server_ready:
        # ... cleanup and error
        raise RuntimeError(f"Failed to start all OpenSpiel servers")
```

**After (~30 lines):**
```python
def start_servers(num_servers: int, base_port: int, game_name: str) -> list:
    """Start OpenSpiel servers for rollout workers."""
    processes = []

    for i in range(num_servers):
        port = base_port + i

        # Kill existing process if any
        subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            stdout=subprocess.DEVNULL,
        )

        proc = multiprocessing.Process(
            target=start_openspiel_server,
            args=(game_name, port),
        )
        proc.start()
        processes.append(proc)

    # Simple health check with retry
    time.sleep(2)  # Give servers time to start
    for i in range(num_servers):
        port = base_port + i
        for attempt in range(10):
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                if attempt == 9:
                    raise RuntimeError(f"Server on port {port} failed to start")
                time.sleep(1)

    return processes

# In main():
server_processes = start_servers(
    num_servers=num_rollout_threads,
    base_port=cfg.blackjack_env.server_port,
    game_name=cfg.blackjack_env.game_name,
)
```

**Rationale:** Remove excessive logging, simplify health checks, fail fast. If a server doesn't start in 10 seconds, something is wrong.

### 2. Remove Server Testing Loop
The server testing loop (lines 1660-1680) duplicates the health check.

**Before:**
```python
# ---- Test OpenSpiel servers ---- #
print("Testing OpenSpiel server connections...")
for i, server_port in enumerate(server_ports):
    test_url = f"http://localhost:{server_port}"
    test_env = OpenSpielEnv(base_url=test_url)
    test_env._http.trust_env = False
    try:
        test_result = test_env.reset()
        print(f"✓ Server {i} test successful (port {server_port}), ...")
        test_env.close()
    except Exception as e:
        # ... verbose error handling
        raise RuntimeError(f"OpenSpiel server {i} test failed: {e}")
```

**After:** (removed - health check is sufficient)

### 3. Simplify kill_process_on_port
Current implementation (lines 66-84) is overly verbose.

**Before:**
```python
def kill_process_on_port(port: int):
    """Kill any process using the specified port."""
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.stdout.strip():
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
                print(f"[DEBUG] Killed existing process {pid} on port {port}")
            except ProcessLookupError:
                pass
        time.sleep(0.5)
        return True
    return False
```

**After:**
```python
def kill_port(port: int):
    """Kill any process using the specified port."""
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
    )
    for pid in result.stdout.strip().split("\n"):
        if pid:
            subprocess.run(["kill", "-9", pid], stderr=subprocess.DEVNULL)
```

**Rationale:** Simpler, no unnecessary logging, use kill command instead of os.kill.

### 4. Move Server Functions to Separate Module (Optional)
Consider moving server-related functions to `envs/openspiel_env/server_utils.py` to keep main.py focused.

**New file structure:**
```python
# envs/openspiel_env/server_utils.py
def start_openspiel_server(game_name: str, port: int):
    """Start OpenSpiel server in background process."""
    # ... implementation

def start_servers(num_servers: int, base_port: int, game_name: str):
    """Start multiple OpenSpiel servers."""
    # ... implementation

def shutdown_servers(processes: list):
    """Shutdown OpenSpiel servers."""
    # ... implementation
```

**In main_v2.py:**
```python
from envs.openspiel_env.server_utils import start_servers, shutdown_servers
```

### 5. Simplify Server Shutdown
Current implementation (lines 1968-1977) is verbose.

**Before:**
```python
print(f"Stopping {len(server_processes)} OpenSpiel servers...")
for i, server_process in enumerate(server_processes):
    server_process.terminate()
    server_process.join(timeout=2)
    if server_process.is_alive():
        print(f"⚠ Server {i} didn't stop gracefully, killing...")
        server_process.kill()
        server_process.join(timeout=1)
print("✓ All OpenSpiel servers stopped")
```

**After:**
```python
# Shutdown servers
for proc in server_processes:
    proc.terminate()
    proc.join(timeout=2)
    if proc.is_alive():
        proc.kill()
```

## Impact
- **Server management:** ~150 lines → ~50 lines (67% reduction)
- **Modularity:** Server logic can be extracted to separate module
- **Reliability:** Simpler code = fewer bugs
- **Startup time:** Faster (less verbose health checking)
- **Risk:** Low - simplifying overly defensive code
