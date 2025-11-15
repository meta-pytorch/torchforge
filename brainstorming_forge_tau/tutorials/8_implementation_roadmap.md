# Part 8: Implementation Roadmap

## 8.1 Already Supported in Forge ✅

Your Forge implementation already has:

- ✅ **vLLM v1 Engine** (Generator)
- ✅ **Async generation** via Monarch actors
- ✅ **Distributed training** (Monarch process mesh)
- ✅ **GRPO algorithm** (group relative policy optimization)
- ✅ **Replay buffer** (decoupled rollout/training)
- ✅ **Reference model** (for KL divergence)
- ✅ **Multi-GPU support**
- ✅ **Episode management** (dataclass structure)
- ✅ **Weight syncing** via torchstore
- ✅ **Async rollout loops** (`continuous_rollouts`)

**This is a solid foundation!** Multi-turn tool calling adds on top of this.

## 8.2 What Needs to Be Added 🔧

### 1. Response Parsing for Tool Calls (2-4 hours)

**What:** Detect and parse tool calls from model output

**Files to create:**
- `forge/utils/parsing.py`

**Functions:**
```python
def parse_tool_call(text: str) -> dict | None
def has_tool_call(text: str) -> bool
def parse_multiple_tool_calls(text: str) -> list[dict]
```

**Effort:** 2-4 hours (simple regex/JSON parsing)

### 2. Multi-turn Rollout Loop (6-8 hours)

**What:** Core `play_task()` function with multi-turn logic

**Files to create:**
- `forge/rollouts/multiturn.py`

**Functions:**
```python
async def play_task(
    task: str,
    policy: Generator,
    tokenizer,
    env: ToolEnv,
    max_turns: int
) -> Episode
```

**Effort:** 6-8 hours (core logic, testing, debugging)

### 3. Tool Environment (4-8 hours)

**What:** OpenEnv integration for Tau2Bench

**Files to create:**
- `forge/environments/tool_env.py` (base class)
- `examples/tau2bench/tau2_env.py` (Tau2-specific adapter)

**Classes:**
```python
class ToolEnv(ABC):
    async def initial_observation()
    async def step(action)
    def get_final_reward()

class Tau2OpenEnv(ToolEnv):
    # Tau2Bench-specific implementation
```

**Effort:** 4-8 hours (environment setup, tool execution, reward computation)

### 4. Response Masking (4-6 hours)

**What:** Track which tokens to train on

**Files to modify/create:**
- `forge/data/episode.py` (add `response_mask` field)
- `forge/losses/grpo_loss.py` (add masking to loss)
- `forge/utils/masking.py` (masking utilities)

**Functions:**
```python
def build_response_mask(messages: list[dict], tokenizer) -> list[int]
def apply_mask_to_loss(loss: Tensor, mask: Tensor) -> Tensor
```

**Effort:** 4-6 hours (dataclass updates, loss function modification, testing)

### 5. Tool Schema Generation (2-4 hours)

**What:** Convert Python functions to OpenAI tool schemas

**Files to create:**
- `forge/utils/tool_schemas.py`

**Functions:**
```python
def convert_func_to_oai_tool(func: callable) -> dict
def format_tools_for_prompt(tools: list[dict]) -> str
```

**Effort:** 2-4 hours (type hint parsing, schema generation)

### 6. System Prompt Formatting (2-3 hours)

**What:** Format prompts with tool definitions

**Files to create:**
- `forge/utils/prompts.py` (core templates)
- `examples/tau2bench/prompts.py` (task-specific)

**Functions:**
```python
def build_tool_calling_system_prompt(tools: list[dict]) -> str
def build_tau2_system_prompt(domain: str, tools: list[dict]) -> str
```

**Effort:** 2-3 hours (template creation, testing)

### 7. Tau2 Evaluation Integration (4-6 hours)

**What:** Scripts to evaluate on Tau2Bench

**Files to create:**
- `examples/tau2bench/evaluate.py`
- `examples/tau2bench/eval_with_debug.py`

**Functions:**
```python
def evaluate_on_tau2(model_path: str, domain: str) -> dict
def debug_failed_episode(task_id: str) -> None
```

**Effort:** 4-6 hours (evaluation loop, metrics, debugging tools)

## 8.3 Implementation Checklist

### Phase 1: Minimum Viable Tool Calling (1-2 days)

**Goal:** Get basic multi-turn working on one task

- [ ] **Step 1:** Implement `parse_tool_call()` in `forge/utils/parsing.py`
  - Test with sample responses
  - Handle edge cases (malformed JSON, etc.)

- [ ] **Step 2:** Create basic `ToolEnv` interface in `forge/environments/tool_env.py`
  - Abstract base class
  - Simple mock implementation for testing

- [ ] **Step 3:** Implement `play_task()` in `forge/rollouts/multiturn.py`
  - Start with Pattern A (simple concat)
  - No masking yet
  - Test with mock environment

- [ ] **Step 4:** Test end-to-end on simple task
  - Use mock domain
  - Single task: create_task
  - Verify multi-turn loop works
  - Check episode structure

**Validation:**
```bash
# Should complete without errors
python -m forge.rollouts.multiturn_test
```

### Phase 2: Integration with Forge GRPO (2-3 days)

**Goal:** Full training loop with masking

- [ ] **Step 5:** Add `response_mask` to Episode dataclass
  - Update `forge/data/episode.py`
  - Add helper methods (`mask_tensor()`, etc.)
  - Update serialization if needed

- [ ] **Step 6:** Implement response masking utilities
  - Create `forge/utils/masking.py`
  - Build masks during `play_task()`
  - Test mask correctness

- [ ] **Step 7:** Update GRPO loss with masking
  - Modify `forge/losses/grpo_loss.py`
  - Add `response_mask` parameter
  - Combine with padding mask
  - Verify gradients flow correctly

- [ ] **Step 8:** Update `continuous_rollouts` to use `play_task()`
  - Modify `examples/tau2bench/grpo/main.py`
  - Handle multi-turn episodes
  - Batch reference model calls
  - Test with small batch

- [ ] **Step 9:** Test training loop
  - Run 10 training steps
  - Verify loss decreases
  - Check GPU memory usage
  - Monitor metrics

**Validation:**
```bash
# Should train successfully
python examples/tau2bench/grpo/main.py --config config.yaml --steps 10
```

### Phase 3: Production-Ready (3-5 days)

**Goal:** Complete, robust implementation

- [ ] **Step 10:** Implement tool schema generation
  - Create `forge/utils/tool_schemas.py`
  - Support type-hinted functions
  - Generate OpenAI-compatible schemas
  - Test with Tau2 tools

- [ ] **Step 11:** Create system prompt templates
  - Core templates in `forge/utils/prompts.py`
  - Tau2-specific in `examples/tau2bench/prompts.py`
  - Test prompt quality

- [ ] **Step 12:** Implement Tau2OpenEnv
  - Create `examples/tau2bench/tau2_env.py`
  - Load Tau2 tasks
  - Execute tools via OpenEnv
  - Compute Tau2 rewards
  - Test on all mock domain tasks

- [ ] **Step 13:** Add comprehensive logging
  - Log episode details
  - Track multi-turn metrics (turns per episode, etc.)
  - Monitor tool call success rate
  - Save failed episodes for debugging

- [ ] **Step 14:** Error handling and edge cases
  - Tool execution timeouts
  - Malformed tool calls
  - Max turns limit
  - Environment errors
  - Graceful degradation

- [ ] **Step 15:** Refactor to Pattern B (Tinker-style)
  - Implement Renderer class
  - Clean up abstractions
  - Improve code organization
  - Add tests

**Validation:**
```bash
# Should handle all cases robustly
python examples/tau2bench/grpo/main.py --config config.yaml --steps 100
# Check logs for errors
```

### Phase 4: Tau2Bench Evaluation (1-2 days)

**Goal:** Evaluate trained model on benchmark

- [ ] **Step 16:** Implement evaluation script
  - Create `examples/tau2bench/evaluate.py`
  - Load trained checkpoint
  - Run on Tau2 test split
  - Collect metrics

- [ ] **Step 17:** Add debugging tools
  - Create `examples/tau2bench/eval_with_debug.py`
  - Inspect failed episodes
  - Analyze score breakdown
  - Generate debug reports

- [ ] **Step 18:** Create results analysis
  - Aggregate metrics (success rate, avg reward, etc.)
  - Per-domain breakdown
  - Per-task results
  - Visualizations (optional)

- [ ] **Step 19:** Run full evaluation on trained model
  - Train on mock domain (train split)
  - Evaluate on mock domain (test split)
  - Analyze results
  - Iterate on prompts/training based on failures

**Validation:**
```bash
# Evaluate on Tau2Bench
python examples/tau2bench/evaluate.py \
  --model ./checkpoints/tau2_grpo \
  --domain mock \
  --split test

# Should output success rate and detailed metrics
```

## Total Estimated Effort

| Phase | Days | Cumulative |
|-------|------|------------|
| Phase 1: MVP | 1-2 | 1-2 |
| Phase 2: Integration | 2-3 | 3-5 |
| Phase 3: Production | 3-5 | 6-10 |
| Phase 4: Evaluation | 1-2 | 7-12 |

**Total: 1.5 - 2.5 weeks** for complete implementation

**Breakdown by complexity:**
- **Simple** (Phase 1): Get it working
- **Medium** (Phase 2): Integrate with Forge
- **Complex** (Phase 3): Production-ready, robust
- **Validation** (Phase 4): Measure performance

## 8.4 Next Steps and Quick Reference

### Immediate Next Steps

1. **Choose a pattern** from Part 5
   - **Recommendation**: Start with Pattern A (simple concat)
   - Move to Pattern B (Tinker) when stable

2. **Set up environment**
   - Start OpenEnv Docker server
   - Load Tau2Bench data
   - Test basic connectivity

3. **Implement Phase 1** (MVP)
   - `parse_tool_call()` function
   - Basic `play_task()` loop
   - Mock environment for testing
   - Verify multi-turn works

4. **Test on one task**
   - Mock domain: create_task_1
   - Run end-to-end
   - Debug and iterate

5. **Scale up**
   - Add response masking
   - Integrate with GRPO
   - Train on full mock domain

### Key Files to Create

**Core Utilities** (reusable):
```
forge/
├── utils/
│   ├── parsing.py           # parse_tool_call(), has_tool_call()
│   ├── prompts.py           # build_tool_calling_system_prompt()
│   ├── renderers.py         # Renderer, Qwen3Renderer
│   ├── masking.py           # build_response_mask()
│   └── tool_schemas.py      # convert_func_to_oai_tool()
├── rollouts/
│   └── multiturn.py         # play_task(), do_rollout()
├── environments/
│   └── tool_env.py          # ToolEnv base class
├── data/
│   ├── episode.py           # Updated Episode with response_mask
│   └── trajectory_processing.py  # trajectory_to_episode()
└── losses/
    └── grpo_loss.py         # grpo_loss_with_masking()
```

**Tau2Bench Example** (task-specific):
```
examples/tau2bench/grpo/
├── main.py                  # Training script
├── tau2_env.py              # Tau2OpenEnv adapter
├── tau2_utils.py            # Task loading, reward computation
├── prompts.py               # Tau2-specific prompt templates
├── config.yaml              # Configuration
├── evaluate.py              # Evaluation script
└── eval_with_debug.py       # Debugging tools
```

### Key Concepts Recap

**Multi-turn** = multiple back-and-forth exchanges in one episode
- Loop until done or max_turns
- Accumulate conversation history
- Concatenate tokens from all turns

**Tool calling** = model invokes functions, not just text
- Parse tool calls from output
- Execute via environment
- Add results to history
- Continue loop

**Response mask** = which tokens to train on
- 1 = LLM-generated (train)
- 0 = Tool results, prompts (ignore)
- Apply during loss computation

**Environment** = executes tools, manages state, provides rewards
- `.reset()` - start episode
- `.step(action)` - execute tool
- `.get_final_reward()` - score episode

**Sparse reward** = only at episode end
- Intermediate steps: reward = 0.0
- Final step: reward from environment
- Matches Tau2Bench pattern

### Questions to Answer as You Implement

**Pattern Selection:**
- Start with Pattern A or B?
  - **A** if you want simplest path
  - **B** if you want clean code from start

**Code Organization:**
- Which utilities are core vs task-specific?
  - Use decision framework from Part 6.2

**OpenEnv Setup:**
- How to configure OpenEnv for Tau2Bench?
  - Docker container with Tau2 tools
  - See Tau2 docs for environment setup

**Evaluation:**
- When to evaluate on Tau2?
  - After Phase 3 (production-ready)
  - Use test split, not train

### Troubleshooting Tips

**If multi-turn loop doesn't work:**
- Check `parse_tool_call()` with print statements
- Verify environment returns correct observations
- Test with max_turns=1 first (single-turn)

**If training fails:**
- Check response_mask is correct shape
- Verify mask applied in loss function
- Start with small batch (batch_size=2)
- Monitor GPU memory

**If evaluation fails:**
- Check model outputs tool calls correctly
- Verify prompt includes tool definitions
- Test parser with model outputs
- Inspect failed episode conversation

**If Tau2 scores are low:**
- Check ACTION score (are tools called?)
- Check ENV score (is state correct?)
- Debug individual failed tasks
- Iterate on prompts and training

### Success Metrics

**Phase 1 (MVP):**
- ✅ Multi-turn loop completes without errors
- ✅ Episodes have correct token structure
- ✅ Can run on mock task

**Phase 2 (Integration):**
- ✅ Training runs for 100 steps
- ✅ Loss decreases
- ✅ Response masking applied correctly
- ✅ No GPU OOM errors

**Phase 3 (Production):**
- ✅ Handles all edge cases gracefully
- ✅ Clean, maintainable code
- ✅ Comprehensive logging
- ✅ All mock domain tasks work

**Phase 4 (Evaluation):**
- ✅ Success rate > 0% on Tau2 test split
- ✅ Can identify failure modes
- ✅ Metrics match expectations
- ✅ Model improves with training

### Final Checklist

Before considering implementation complete:

- [ ] Multi-turn loop works on all Tau2 mock tasks
- [ ] Response masking tested and verified
- [ ] Training loop stable for 1000+ steps
- [ ] Evaluation script produces meaningful results
- [ ] Code is clean and documented
- [ ] Tests pass
- [ ] Can reproduce results
- [ ] Performance metrics logged
- [ ] Ready to scale to other domains (airline, retail, etc.)

---

## 9. Open Questions for Further Research

Based on the tutorial creation, here are open questions to investigate:

### 1. Forge Async Engine Support
**Question:** Does Forge Generator support vLLM's `async_engine: true` flag, or does Monarch handle async differently?
**Action:** Check `forge/actors/generator.py` to understand async mechanism
**Impact:** Affects Pattern D implementation (async pipelining)

### 2. vLLM Configuration Flags in Forge
**Question:** Which vLLM flags work with Forge Generator? (`enable_auto_tool_choice`, `tool_call_parser`, etc.)
**Action:** Test different EngineArgs flags
**Impact:** Determines if Pattern E (native tools) is directly usable

### 3. Optimal Episode Strategy for Forge
**Question:** Strategy A (per-step) vs Strategy B (concatenated) - which performs better with Forge GRPO?
**Action:** Benchmark both on same task
**Impact:** Choose default pattern for production

### 4. Response Masking Performance
**Question:** How much does response masking improve sample efficiency?
**Action:** Train with/without masking, compare convergence
**Impact:** Validate masking is worth the complexity

### 5. OpenEnv + Tau2Bench Integration Details
**Question:** Best way to set up OpenEnv Docker containers with Tau2Bench tools?
**Action:** Create setup script and test
**Impact:** Ease of getting started

### 6. Memory Scaling
**Question:** How many concurrent samples can run with async pipelining before GPU OOM?
**Action:** Benchmark with different batch sizes
**Impact:** Production deployment planning

### 7. Model Tool Calling Capability
**Question:** Does Qwen2.5-1.5B need fine-tuning for tool calling, or can it zero-shot?
**Action:** Test base model on Tau2 before training
**Impact:** Determines if SFT phase needed before RL

### 8. Alternative Reward Shaping
**Question:** Can dense rewards (per-step) improve over sparse (end-of-episode)?
**Action:** Experiment with reward shaping on mock domain
**Impact:** Better credit assignment strategies

---

**You now have 8 complete tutorial documents!** Start with Part 1 and work through sequentially. Good luck with your implementation! 🚀
