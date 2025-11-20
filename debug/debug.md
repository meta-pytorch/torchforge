# Blackjack main_v2.py Refactoring Progress

## Context
Refactoring `/home/felipemello/forge/apps/blackjack/main_v2.py` to be cleaner, simpler, and more maintainable. Goal is to align with `apps/grpo/main.py` patterns while removing over-engineering and debug code.

## File Organization (Current State)

### Files Created/Modified:
1. **`/home/felipemello/forge/apps/blackjack/token_accumulator.py`** ✅
   - Moved TokenAccumulator class and related enums (ValidationMode, TruncationReason, EpisodeData)
   - Has all necessary imports
   - Working correctly

2. **`/home/felipemello/forge/apps/blackjack/blackjack_env.py`** ✅
   - Moved BlackjackEnv class and EnvStepResult dataclass
   - Has all necessary imports
   - Fixed typo: `is_invalid` parameter (was `in_invalid`) - this was causing hangs!

3. **`/home/felipemello/forge/apps/blackjack/main_v2.py`** ✅
   - Imports from token_accumulator and blackjack_env
   - Significantly cleaned up (1987 lines → 1183 lines, ~800 lines removed)
   - Working correctly

## Completed Tasks

### ✅ Task 1: Fix All Imports
**Status:** COMPLETE
**Changes:**
- Added imports to `token_accumulator.py`: threading, dataclass, Enum, Optional, torch
- Added imports to `blackjack_env.py`: re, dataclass, field, Any, OpenSpielAction, OpenSpielEnv, record_metric, Reduce
- Added local imports to `main_v2.py`:
  ```python
  from apps.blackjack.blackjack_env import BlackjackEnv, EnvStepResult
  from apps.blackjack.token_accumulator import (
      TokenAccumulator,
      ValidationMode,
      TruncationReason,
      EpisodeData,
  )
  ```
- Updated usage comment from `main_v2` to `main`

**Key Issue Found & Fixed:**
- `blackjack_env.py` had typo `in_invalid` instead of `is_invalid` in `_compute_reward()` parameter - this was causing the import to hang!

### ✅ Task 2: Simplify Server Management in `async def main()`
**Status:** COMPLETE
**Changes:**
- Created helper functions (lines 74-161):
  - `kill_process_on_port()` - simplified (removed debug prints)
  - `_wait_for_server_health()` - extracted health check logic
  - `start_servers()` - consolidated server startup with health checks
  - `shutdown_servers()` - consolidated graceful shutdown

- **Server startup** (lines 801-806):
  ```python
  # Before: 67 lines of verbose code
  # After: 6 clean lines
  server_processes, server_ports = start_servers(
      num_servers=cfg.get("rollout_threads", 1),
      base_port=cfg.blackjack_env.server_port,
      game_name=cfg.blackjack_env.game_name,
  )
  ```

- **Server shutdown** (line 1191):
  ```python
  # Before: 10 lines
  # After: 1 line
  shutdown_servers(server_processes)
  ```

**Impact:** Removed ~70 lines from main(), much cleaner

### ✅ Task 3: Clean up `async def main()` debugging/checks
**Status:** COMPLETE
**Changes:**
- Created `print_episode_debug()` function (lines 164-193)
  - Reuses TokenAccumulator's `show_messages()` method
  - Creates temp TokenAccumulator, replaces internals with Episode data
  - Provides colorized token stream visualization

- **Removed redundant server testing** (deleted lines 915-935, ~22 lines)
  - Servers already tested in `start_servers()`, this was redundant

- **Simplified debug printing** (31 lines → 3 lines):
  ```python
  # Print episode details every 10 rollouts
  if episodes and rollout_count % 10 == 0:
      print_episode_debug(episodes[0], tokenizer, rollout_count)
  ```

**Impact:** Removed ~50 lines, cleaner console output (only every 10 rollouts)

## Current State Summary
- **File size:** 1183 lines (down from 1987, ~40% reduction)
- **All imports working:** ✅
- **Server management:** ✅ Simplified and extracted
- **Debug output:** ✅ Clean and using TokenAccumulator visualization
- **Tests:** ✅ All changes tested and working

## Next Task: Task 4 - Remove EnvironmentActor

### Current Problem:
`EnvironmentActor` exists only to provide tokenizer access (lines ~819-828 in main_v2.py):
```python
# First, initialize env_actor to get pad_id
env_actor = await EnvironmentActor.options(**cfg.actors.blackjack_env).as_actor(**env_actor_config)
pad_id = await env_actor.pad_token.call_one()

# Later in continuous_rollouts:
pad_id = await env_actor.pad_token.call_one()
tokenizer = await env_actor.get_tokenizer.call_one()
```

This is unnecessary overhead - we should just get the tokenizer directly and pass it where needed.

### Proposed Solution:
1. **Get tokenizer directly in main():**
   ```python
   tokenizer = get_tokenizer(cfg.blackjack_env.model)
   pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
   ```

2. **Pass tokenizer to continuous_rollouts:**
   ```python
   async def continuous_rollouts(thread_id: int, tokenizer):
       # Use tokenizer directly, no actor calls needed
   ```

3. **Remove EnvironmentActor class definition** (if it exists in main_v2.py)

4. **Remove threading locks from TokenAccumulator** (since tokenizer is no longer shared via actor):
   - Remove `self._lock = threading.Lock()` from TokenAccumulator.__init__
   - Remove `with self._lock:` blocks from tokenizer calls in TokenAccumulator
   - This simplifies TokenAccumulator significantly

### Files to Modify:
- `/home/felipemello/forge/apps/blackjack/main_v2.py`
- `/home/felipemello/forge/apps/blackjack/token_accumulator.py` (remove locks)

### Expected Impact:
- Remove EnvironmentActor abstraction (~20 lines)
- Simplify continuous_rollouts initialization
- Remove threading locks from TokenAccumulator (~5-10 places)
- Cleaner, more direct code

## Important Notes for Future Context

### Critical Bug Fixed:
- **Hang issue:** Was caused by typo `in_invalid` vs `is_invalid` in `blackjack_env.py:164`
- When importing BlackjackEnv caused hang, check for parameter name mismatches

### Testing Pattern:
- After each change, run: `python -m apps.blackjack.main_v2 --config apps/blackjack/qwen3_1_7b.yaml`
- Verify no hangs during initialization
- Check that colorized debug output appears every 10 rollouts

### Key Design Decisions:
- **Reuse TokenAccumulator visualization:** Don't duplicate colorization code, create temp instance and replace internals
- **Print every N rollouts:** Use `rollout_count % 10 == 0` to avoid console spam
- **Extract server logic:** Keep main() focused on training loop, not infrastructure

### File Line Counts:
- Start: 1987 lines
- After Task 1: ~1987 lines (just imports)
- After Task 2: ~1200 lines
- After Task 3: ~1183 lines
- Target: ~900-1000 lines after Task 4

### Remaining Tasks (Priority Order):
1. **Task 4:** Remove EnvironmentActor, pass tokenizer directly ⬅️ NEXT
2. Remove threading locks from TokenAccumulator (part of Task 4)
3. Any other cleanup identified during Task 4
