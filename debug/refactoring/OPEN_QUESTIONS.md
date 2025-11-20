# Open Questions for Refactoring Review

This document lists questions and decisions that need to be addressed before implementing the refactoring proposals.

## Architecture Decisions

### Q1: TokenAccumulator Module Location
**Question:** Should `TokenAccumulator` live in `src/forge/data/token_accumulator.py` or somewhere else?

**Options:**
1. `src/forge/data/token_accumulator.py` - Makes it available to all forge apps
2. `apps/blackjack/token_accumulator.py` - Keeps it local to blackjack
3. `envs/utils/token_accumulator.py` - Groups with environment utilities

**Recommendation:** Option 1 (forge-level) - TokenAccumulator is a general-purpose utility for any multi-turn RL task, not blackjack-specific.

**Decision needed:** ☐

---

### Q2: Server Management Module
**Question:** Should server management functions be extracted to a separate module?

**Options:**
1. Keep in main_v2.py (after simplification, only ~30 lines)
2. Move to `envs/openspiel_env/server_utils.py`
3. Move to `apps/blackjack/server_utils.py`

**Recommendation:** Option 2 - It's OpenSpiel-specific, not blackjack-specific.

**Decision needed:** ☐

---

### Q3: Rollout Module Location
**Question:** Should rollout functions be in `apps/blackjack/rollout.py` or elsewhere?

**Options:**
1. `apps/blackjack/rollout.py` - Keeps blackjack logic together
2. `apps/blackjack/main_v2.py` - Keep in main file (simpler)

**Recommendation:** Option 1 - Separates rollout logic from main loop, makes testing easier.

**Decision needed:** ☐

---

## Loss Function Questions

### Q4: Debug Metrics in simple_grpo_loss
**Question:** How much debug logging should we keep in `simple_grpo_loss`?

**Current state:** ~50 metrics, emergency dumps (280 lines)

**Options:**
1. **Minimal:** 3-5 essential metrics only (loss, KL, advantages)
2. **Moderate:** 10-15 metrics (add logprobs stats, per-token stats)
3. **Configurable:** All metrics controlled by `cfg.debug.loss_metrics_verbose` flag

**Recommendation:** Option 3 - Best of both worlds. Production uses minimal, debugging uses full.

**Decision needed:** ☐

---

### Q5: Emergency Tensor Dumps
**Question:** Should we keep the emergency tensor dump feature that triggers on huge loss values?

**Current state:** Lines 1432-1489 save all tensors to /tmp when loss > 1000

**Options:**
1. Remove completely - it's never triggered in practice
2. Keep but make configurable via `cfg.debug.emergency_dumps`
3. Keep and improve - save to a configured directory, add more context

**Recommendation:** Option 2 - Useful for debugging edge cases, but should be opt-in.

**Decision needed:** ☐

---

## Environment Questions

### Q6: Invalid Action Penalty
**Question:** Should the -10 penalty for invalid actions be configurable?

**Current state:** Hardcoded -10.0 penalty in `_compute_reward`

**Options:**
1. Keep hardcoded - it's a reasonable default
2. Make configurable via `cfg.blackjack_env.invalid_action_penalty`
3. Remove penalty entirely - let the model learn without artificial penalties

**Recommendation:** Option 2 - Different tasks may want different penalties.

**Decision needed:** ☐

---

### Q7: System Prompt Location
**Question:** Should the system prompt be in the config file or in code?

**Current state:** Hardcoded in main_v2.py (lines 1698-1720)

**Options:**
1. Move to config YAML - easier to iterate on prompts
2. Keep in code - simpler, less indirection
3. Both - default in code, override via config

**Recommendation:** Option 3 - Flexibility without losing simplicity.

**Decision needed:** ☐

---

## Validation and Testing

### Q8: TokenAccumulator Validation
**Question:** What should the default validation mode be?

**Current state:** `ValidationMode.OFF` in production code

**Options:**
1. `OFF` - No runtime cost, but harder to debug
2. `WARN` - Print warnings but don't fail
3. `STRICT` in development, `OFF` in production

**Recommendation:** Option 3 - Use config to control: `cfg.blackjack_env.token_validation`

**Decision needed:** ☐

---

### Q9: Message Log Storage
**Question:** Should message logs be stored in Episode objects by default?

**Current state:** Always stored, can be large for long episodes

**Options:**
1. Always store - useful for debugging
2. Never store - saves memory
3. Configurable via `cfg.debug.save_message_logs`

**Recommendation:** Option 3 - Only store when debugging.

**Decision needed:** ☐

---

## Performance Questions

### Q10: Sequential vs Parallel Rollouts
**Question:** Should games within a group be run sequentially or in parallel?

**Current state:** Sequential (one env per group, shared server)

**Options:**
1. Keep sequential - Simpler, avoids race conditions
2. Make parallel - Faster, but need one server per game
3. Configurable - Let config decide based on infrastructure

**Recommendation:** Option 1 - Blackjack games are fast enough that parallelism within a group doesn't matter.

**Decision needed:** ☐

---

### Q11: Number of Rollout Threads
**Question:** What's the recommended number of rollout threads for blackjack?

**Current state:** Configurable, each thread needs its own server

**Options:**
1. Single thread (simpler, fewer servers)
2. Multiple threads (one per CPU core)
3. Document recommendation in config

**Recommendation:** Option 3 - Add comment in config: `rollout_threads: 4  # One per CPU core`

**Decision needed:** ☐

---

## Configuration Questions

### Q12: Debug Configuration Defaults
**Question:** What should the default values be for debug configuration?

**Proposed defaults:**
```yaml
debug:
  enabled: false              # Disable all debug features by default
  print_episodes: false
  save_message_logs: false
  validate_tokens: false
  emergency_dumps: false
  rollout_interval: 100
  loss_metrics_verbose: false
```

**Are these reasonable?** ☐

---

### Q13: Backward Compatibility
**Question:** Should we maintain backward compatibility with existing checkpoints and configs?

**Options:**
1. Yes - Add migration logic for old configs
2. No - Breaking change, update configs manually
3. Support both for one release, then deprecate

**Recommendation:** Option 2 - This is internal research code, clean break is fine.

**Decision needed:** ☐

---

## Metric Naming

### Q14: Metric Naming Convention
**Question:** Should we standardize metric names?

**Current state:** Inconsistent naming (`groups/rate_dropped`, `buffer/episodes_accepted`, etc.)

**Proposed convention:**
```
loss/*          - Loss function metrics
episode/*       - Per-episode metrics
rollout/*       - Rollout loop metrics
buffer/*        - Replay buffer metrics
game/*          - Game environment metrics
policy/*        - Policy-related metrics
ref_model/*     - Reference model metrics
```

**Should we enforce this?** ☐

---

## Module Organization

### Q15: File Naming Convention
**Question:** Should we rename `main_v2.py` after refactoring?

**Options:**
1. Keep as `main_v2.py`
2. Rename to `main.py` (deprecate old main_v2.py)
3. Rename to `grpo_main.py` for clarity

**Recommendation:** Option 1 - Less disruption, clear that it's the second iteration.

**Decision needed:** ☐

---

### Q16: Import Organization
**Question:** Should we use absolute or relative imports in the new modules?

**Example:**
```python
# Absolute
from forge.data.token_accumulator import TokenAccumulator

# Relative
from ...data.token_accumulator import TokenAccumulator
```

**Recommendation:** Absolute imports - More explicit, easier to understand.

**Decision needed:** ☐

---

## Testing and Validation

### Q17: Testing Strategy
**Question:** What level of testing should we add during refactoring?

**Options:**
1. No tests - Just ensure existing code runs
2. Unit tests for extracted modules (TokenAccumulator, BlackjackEnv)
3. Integration test for full training loop
4. All of the above

**Recommendation:** Option 2 - Unit tests for new modules, smoke test for main loop.

**Decision needed:** ☐

---

### Q18: Regression Testing
**Question:** How do we verify the refactored code produces the same results?

**Options:**
1. Visual inspection - Run both versions, compare metrics
2. Automated comparison - Save outputs, assert equality
3. Don't validate - Trust the refactoring

**Recommendation:** Option 1 - Run a few short training runs, compare loss curves.

**Decision needed:** ☐

---

## Implementation Questions

### Q19: Implementation Order
**Question:** Which phase should we implement first?

**Proposed order:**
1. Phase 1: Critical simplifications (biggest impact, lowest risk)
2. Phase 2: Modular architecture (structural changes)
3. Phase 3: Polish and documentation

**Is this the right order?** ☐

---

### Q20: Rollback Strategy
**Question:** What if the refactoring breaks something?

**Options:**
1. Keep old main_v2.py as main_v2_old.py backup
2. Use git branches - feature branch for refactoring
3. Just commit frequently to main

**Recommendation:** Option 2 - Git branch is the right tool for this.

**Decision needed:** ☐

---

## Additional Considerations

### Q21: Documentation Updates
**Question:** What documentation needs to be updated?

**Items:**
- [ ] Update usage comment at top of file
- [ ] Update README for blackjack app
- [ ] Add docstrings to new modules
- [ ] Update config file comments

**All of these?** ☐

---

### Q22: Alignment with Future Changes
**Question:** Are there any upcoming changes to grpo/main.py that we should align with?

**Action needed:** Review recent commits to grpo/main.py for patterns to adopt.

**Decision needed:** ☐

---

## Summary of Decisions Needed

**High Priority (blocking refactoring):**
- Q4: Debug metrics level in loss function
- Q5: Emergency dump feature
- Q8: TokenAccumulator validation default
- Q9: Message log storage

**Medium Priority (affects architecture):**
- Q1: TokenAccumulator location
- Q2: Server management module
- Q3: Rollout module location

**Low Priority (nice to have):**
- Q6: Invalid action penalty configurability
- Q7: System prompt location
- Q14: Metric naming standardization
- Q15: File renaming

**For Documentation:**
- Q21: Documentation updates
- Q12: Debug config defaults

Please review and provide decisions on at least the high-priority questions before beginning implementation.
