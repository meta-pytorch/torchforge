# Tracker - Forge + Tau2 Integration

## Document Status

### Completed Documents
- ✅ `1_requirements_and_context.md` - UPDATED with clarified goal and OpenEnv approach
- ✅ `2_tracker.md` - This file
- ✅ `3_open_questions.md` - Open questions
- ✅ `4_examples_APIs.md` - **NEW!** Complete analysis of BlackJack + Tinker patterns

### In Progress
- 🔄 Understanding tool calling in vLLM and OpenEnv
- 🔄 Understanding BlackJack→ToolCalling adaptation

### Planned
- ⏳ Design doc: Tool calling environment for OpenEnv
- ⏳ Design doc: Adapting BlackJack pattern for tool calling
- ⏳ Design doc: Tau2 evaluation integration
- ⏳ Implementation plan: Step-by-step changes
- ⏳ Code snippets: Example implementations

---

## Current Focus

**MAJOR UPDATE: Training Strategy Changed!**

**Previous assumption:** Train using Tau2's gym environment
**New approach:**
- **Training**: Use OpenEnv Docker sandboxes (NOT Tau2)
- **Evaluation**: Use Tau2 to benchmark trained models

**Phase 1: Understand Patterns & Design API (Current)**
- ✅ Analyzed OpenEnv BlackJack example
- ✅ Analyzed Tinker-cookbook tool use example
- ✅ Created comprehensive comparison in `4_examples_APIs.md`
- 🔄 Next: Prototype the proposed API with actual code

---

## Next Steps

1. **Prototype Response Parsing** (Immediate)
   - Implement `parse_response()` function
   - Test both tag format and function-call format
   - Handle edge cases
   - Create: `5_response_parsing.py` (working code)

2. **Prototype `play_task()` Loop** (Immediate)
   - Implement multi-turn rollout function
   - Handle tool calls and messages
   - Track conversation history
   - Create: `6_play_task_loop.py` (working code)

3. **Create Simple Tool Environment** (Next)
   - Build minimal OpenEnv tool-calling environment
   - Support 2-3 simple tools (search, calculate, etc.)
   - Define reward function
   - Create: `7_simple_tool_env/` (working environment)

4. **Integration with Forge GRPO** (After prototypes work)
   - Adapt Episode dataclass
   - Integrate `play_task()` into continuous_rollouts
   - Test end-to-end training
   - Create: `8_forge_integration.py` (working example)

5. **Tau2 Evaluation** (Final)
   - Figure out local model evaluation
   - Create evaluation script
   - Document process
   - Create: `9_tau2_eval.py` (evaluation runner)

---

## Questions Resolved

*(None yet - see 3_open_questions.md)*

---

## Observations & Insights

**Key Patterns Identified** (See `4_examples_APIs.md` for detailed analysis):

1. **Working Integration Example**: OpenEnv BlackJack shows complete Forge + OpenEnv integration
2. **Training ≠ Evaluation**: Use OpenEnv for training (flexible, custom rewards), Tau2 for evaluation (standard benchmark)
3. **Text-based Actions**: Parsing actions from LLM text output works (proven in BlackJack)
4. **Sparse Rewards Pattern**: Final reward assigned to all steps (matches Tau2's structure)
5. **Multiple Reference Patterns**: BlackJack (simpler, Forge-proven) vs Tinker-cookbook (structured) vs VERL/NeMo-RL (production-scale)

See `4_examples_APIs.md` for complete code examples and detailed comparisons.

---

## Session Log

### Session 1
- **Date:** 2025-11-11 (Part 1)
- **Created:** Initial context docs (1, 2, 3)
- **Explored:**
  - Forge GRPO implementation
  - Tau2 gym interface and scoring
- **Major Update:** Learned that training will use OpenEnv (not Tau2)!
- **Discovered:** Working BlackJack example that integrates OpenEnv + Forge

### Session 1 (Continuation)
- **Date:** 2025-11-11 (Part 2)
- **Goal Clarified:** Need clean code showing rollout loop with tool calling + multi-turn
- **Created:** `4_examples_APIs.md` - Complete analysis of existing patterns
- **Analyzed:**
  - OpenEnv BlackJack: `play_game()` pattern, text parsing, episode structure
  - Tinker-cookbook: Tool schemas, message history, environment step flow
- **Proposed:** Synthesized Forge API combining best of both approaches
- **Next:** Prototype response parsing and play_task() loop with actual code
