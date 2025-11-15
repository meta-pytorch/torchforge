# Open Questions (UPDATED)

**MAJOR UPDATE:** Training approach changed from Tau2 to OpenEnv. Many questions are now obsolete or need reframing.

---

## Critical Path Questions

### Q1: How does vLLM support tool/function calling?
**Status:** 🔴 Not Answered

**What we need to know:**
- Does vLLM v1 natively support function calling?
- How to enable it in Forge's Generator?
- What's the output format?
- Can we parse tool calls from text output (like BlackJack does)?

**Why it matters:**
Tool calling is the core capability we're training. We need to know if vLLM handles it natively or if we parse from text.

**BlackJack shows text parsing works:**
```python
response = await policy.generate(prompt)  # "HIT" or "STAND"
action_id = parse_action(response.text)   # Parse from text
```

**Can we do similar for tools?**
```python
response = await policy.generate(prompt)  # "search_flights(origin='NYC')"
tool_call = parse_tool_call(response.text)
```

**Next steps:**
- Check vLLM v1 docs for function calling
- Test with simple tool call generation
- Document findings in `4_vllm_tool_calling.md`

---

### Q2: How to adapt BlackJack pattern for tool calling?
**Status:** 🔴 Not Answered

**What we need to know:**
- Current: `format_prompt()` → `generate()` → `parse_action()` → `env.step()`
- Needed: How to format prompts with tool definitions?
- How to parse tool calls from responses?
- How to map tool calls to OpenEnv actions?

**BlackJack Pattern:**
```python
async def play_game(...):
    env = OpenSpielEnv(base_url=server_url)
    result = env.reset()

    while not done:
        # 1. Format prompt
        prompt = format_prompt(step_num, action_history, tokenizer)

        # 2. Generate
        responses = await policy.generate.route(prompt)

        # 3. Parse action
        action_id = parse_action(response.text, obs.legal_actions)

        # 4. Execute
        result = env.step(OpenSpielAction(action_id=action_id))

        # Store step data
        game_steps.append({...})

    # Assign final reward to all steps
    return all_step_results
```

**Needed Tool Calling Pattern:**
```python
async def play_task(...):
    env = ToolCallingEnv(base_url=server_url)
    result = env.reset()
    while not done:
        prompt = format_prompt_with_tools(task, tools, history, tokenizer)
        responses = await policy.generate.route(prompt)

        if is_tool_call(response.text):
            tool_call = parse_tool_call(response.text)
            result = env.step(ToolCallAction(tool_call))
        else:
            result = env.step(MessageAction(response.text))

        task_steps.append({...})
    return all_step_results
```
*(See `4_examples_APIs.md` for complete implementation)*

**Next steps:**
- Study `format_prompt()` in grpo_utils.py
- Design `format_prompt_with_tools()`
- Implement `parse_tool_call()`
- Document pattern

---

### Q3: What tool-calling environment should we use for training?
**Status:** 🔴 Not Answered

**What we need to know:**
- Is there an existing OpenEnv tool-calling environment?
- Should we create one ourselves?
- What tools should it support?
- How should rewards work?

**Options:**

**Option A: Use coding_env**
- Already exists!
- Executes Python code
- Could frame tool calls as function executions
- Reward based on test passing?

**Option B: Create custom tool env**
- Define specific tools (search, book_flight, etc.)
- More aligned with Tau2 eval
- More work to build

**Option C: Wait for OpenEnv team to build one**
- Cleanest solution
- May take time
- Dependencies on external team

**Requirements for the environment:**
- Accept tool calls as actions
- Execute tools safely (Docker sandbox)
- Return observations (tool results)
- Provide rewards (task completion?)
- Support multiple tools per task

**Next steps:**
- Check if tool-calling env is being built
- Prototype simple version
- Define tool set and reward function
- Document in `5_tool_calling_env_design.md`

---

### Q4: How to run Tau2 evaluation on trained model?
**Status:** 🔴 Not Answered

**What we need to know:**
- How to point Tau2 CLI to local model checkpoint?
- Does it support local models or only API models?
- What format does checkpoint need to be in?
- Can we run programmatically (not just CLI)?

**From Tau2 README:**
```bash
tau2 run \
  --domain airline \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --task-split base
```

**Questions:**
- Can `--agent-llm` point to local model?
- Format: `--agent-llm /path/to/checkpoint`?
- Or need to serve via vLLM first?
- How to integrate with Forge checkpoints?

**Next steps:**
- Read Tau2 agent documentation
- Test with local model
- Document in `6_tau2_eval_integration.md`

---

### Q5: How to structure episodes for multi-step tool calling?
**Status:** 🟡 Partially Answered (BlackJack shows the way)

**What we know from BlackJack:**
- One Episode per step (not per game)
- All steps in a game get the same final reward
- Episode includes: episode_id, game_id, step_in_game, completion, ref_logprobs, reward, advantage

*(See BlackJack example in `4_examples_APIs.md` for full Episode dataclass)*

**What we still need:**
- How to handle tool results in prompts?
- Do we include tool results in the completion?
- How to track conversation history across steps?

**Next steps:**
- Prototype Episode structure for tool calling
- Test with simple example

---

## Secondary Questions

### Q6: Do we need vLLM's native tool calling or is text parsing enough?
**Status:** 🔴 Not Answered

**Trade-offs:**

**Text Parsing (BlackJack approach):**
- ✅ Simpler to implement
- ✅ Already proven to work
- ✅ Model learns to format correctly
- ❌ May have parsing errors
- ❌ Less structured

**Native vLLM Tool Calling:**
- ✅ More structured output
- ✅ Guaranteed valid JSON
- ✅ Industry standard
- ❌ More complex setup
- ❌ May not work with all models

**Recommendation:** Start with text parsing (proven), migrate to native if needed.

---

### Q7: How to align OpenEnv training tools with Tau2 evaluation tools?
**Status:** 🔴 Not Answered

**The dilemma:**
- Training: Custom tools in OpenEnv
- Evaluation: Fixed tools in Tau2 domains

**Should the tools match exactly?**

**Option A: Exact match**
- Training tools = Tau2 tools
- Ensures consistency
- But limits training flexibility

**Option B: Superset**
- Training includes Tau2 tools + more
- More diverse training
- May not transfer perfectly

**Option C: Different tools, same patterns**
- Focus on tool calling *skill*
- Not specific tools
- Rely on generalization

**Next steps:**
- List Tau2 tools by domain
- Design training tool set
- Decide on strategy

---

### Q8: What's the reward function for tool calling?
**Status:** 🔴 Not Answered

**BlackJack uses game outcome:** `reward = float(game_reward)  # +1 (win), -1 (loss), 0 (push)`

**For tool calling, options:**
- **Option A: Binary** - `1.0 if task_completed else 0.0`
- **Option B: Shaped** - Partial credit for correct tool + correct args + completion
- **Option C: LLM-as-judge** - `reward = llm_judge_quality(task, execution, output)`

**Next steps:**
- Experiment with reward functions
- Measure what works best
- Document findings

---

### Q9: How to run periodic Tau2 eval during training?
**Status:** 🔴 Not Answered (Nice to have, not required)

**Desired flow:** Run Tau2 evaluation every N training steps to track progress

**Challenges:**
- Tau2 eval may be slow
- May block training
- Need to run in separate process?

**Next steps:**
- Prototype tau2 eval wrapper
- Measure evaluation time
- Decide if worth implementing

---

## Questions for Admin

*(User decisions needed)*

### Admin Q1: Which tool-calling environment should we start with?
**Options:**
- (A) Use existing `coding_env` and frame tools as code execution
- (B) Build simple custom tool environment (e.g., search + book)
- (C) Wait for OpenEnv team to build proper tool env
- (D) Other suggestion?

**Recommendation:** (B) Build simple version to unblock training ASAP.

---

### Admin Q2: Should training tools match Tau2 evaluation tools exactly?
**Options:**
- (A) Yes, use identical tools for training and eval
- (B) No, use broader set in training, Tau2 tools in eval
- (C) Use different tools entirely, rely on generalization

**Implications:**
- (A) = Safest transfer, but limited training diversity
- (B) = More diverse training, may not transfer perfectly
- (C) = Most general, highest risk

**Recommendation:** Start with (A), expand to (B) if needed.

---

### Admin Q3: Reward function preference?
**Options:**
- (A) Binary (task completed or not)
- (B) Shaped rewards (partial credit)
- (C) LLM-as-judge
- (D) Hybrid

**Recommendation:** Start with (B) shaped rewards for faster learning.

---

### Admin Q4: Priority on periodic Tau2 eval?
**Options:**
- (A) High - implement in first version
- (B) Medium - add after basic training works
- (C) Low - only eval at end

**User said:** Nice to have, not must have → Answer is (B) or (C)

---

## Resolved Questions

### Q_RESOLVED: Should we use Tau2 for training?
**Answer:** No! Use OpenEnv for training, Tau2 only for evaluation.

**Source:** User clarification in conversation.

**Date:** 2025-11-11

**Implications:** Drastically simplifies the problem. We already have a working example (BlackJack) to build from.

---

### Q_RESOLVED: Do we need multi-turn conversation during training?
**Answer:** Depends on environment. BlackJack doesn't have "user" but plays full games. Tool-calling env may or may not need conversational user.

**Source:** BlackJack example analysis.

**Date:** 2025-11-11

**Implications:** Can use simpler task-based episodes without full Tau2-style user simulation.
