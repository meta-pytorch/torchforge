# Simplification Ideas: Token Accumulation in Multi-Turn RL Rollouts

## Problem Statement

Our current implementation in `clean_code.py` has significant complexity:

### Current Complexity Issues:

1. **Multiple `apply_chat_template` calls before generation:**
   - Call #1 (line 71): Extract new prompt tokens WITHOUT generation prompt
   - Call #2 (line 88): Check budget WITH generation prompt
   - Call #3 (line 102): Create prompt text WITH generation prompt (for actual generation)

   **Why this is complex:** We tokenize the same conversation 3 times with slightly different settings before we even generate.

2. **Multiple `apply_chat_template` calls after generation:**
   - Call #4 (line 120): Extract assistant tokens via prefix matching
   - Call #5 (line 166): Check if env obs would exceed budget
   - Call #6 (line 179): Extract env obs tokens via prefix matching

   **Total:** Up to 6 `apply_chat_template` calls per turn!

3. **Mismatch between `messages` and `all_tokens`:**
   When truncation occurs:
   - `messages[-1]` contains FULL observation content
   - `all_tokens` contains TRUNCATED version

   This mismatch is intentional but confusing.

4. **Cannot use `response.token_ids` directly:**
   - `response.token_ids` = [3 tokens] (just content like "HIT")
   - `assistant_tokens` = [7 tokens] (includes `<|im_start|>assistant\n` + content + `<|im_end|>\n`)

   Must re-tokenize full conversation to get role headers.

## What We're Trying To Do

**Goal:** Accumulate tokens incrementally during multi-turn RL episodes while:
1. Tracking budget (max_seq_len constraint)
2. Detecting truncation (generation or env observation)
3. Maintaining correct token sequences for training (all special tokens included)
4. Supporting variable-length episodes (env can end at any turn)

**Key Constraint:** `all_token_ids` must exactly match what `tokenizer.apply_chat_template(messages, ...)` would produce if called at the end. This is critical for:
- Reference model scoring (needs identical token sequence)
- Training (response_mask must align with actual tokens)

## Relevant Documents to Review

### Internal Documentation:
- `/home/felipemello/forge/brainstorming_forge_tau/changes/3_truncation_v6_token_accumulation_insights.md`
  - Analysis of how TRL, VERL, NeMo-RL, Verifiers, and Tinker handle token accumulation
  - Full library paths and code references

- `/home/felipemello/forge/brainstorming_forge_tau/changes/3_truncation_v5_simplified_env.md`
  - Previous attempt (incorrect approach using `tokenizer.encode()`)
  - Shows what NOT to do

- `/home/felipemello/forge/test_simple_vllm.py`
  - Comprehensive test suite validating current approach
  - 5 test cases covering all truncation scenarios

### Key Code References:
- Current implementation: `/home/felipemello/forge/clean_code.py`
- Generator: `/home/felipemello/forge/src/forge/actors/generator.py`
- GRPO trainer: `/home/felipemello/forge/apps/grpo/main.py`

## Research Questions for Future Investigation

**To be researched via subagents (NOT NOW - this is setup for future work):**

### 1. How do other libraries handle this?

**TRL (Transformers Reinforcement Learning):**
- Path:
- Questions:
  - How does accumulate tokens in PPOTrainer?
  - Do they use prefix matching or something else?
  - How do they handle truncation?
  - We know they use prefix matching (from v6 doc)
  - How many tokenization calls do they make per turn?
  - Do they have any optimizations we're missing?

`/home/felipemello/forge/verl/`
`/home/felipemello/forge/trl/`
/home/felipemello/forge/prime-rl
/home/felipemello/forge/RL
/home/felipemello/forge/tinker-cookbook
/home/felipemello/forge/verifiers

### 2. Can we avoid multiple tokenization calls?

**Idea A: Cache tokenized results**
- After call #1, can we reuse those tokens for calls #2 and #3?
- Problem: Call #2 and #3 have `add_generation_prompt=True`
- Could we manually append generation prompt tokens instead of re-tokenizing?

**Idea B: Tokenizer state/incremental tokenization**
- Does HF tokenizer support incremental tokenization?
- Can we tokenize just the new message and append?
- Problem: Chat template adds role headers that depend on position

**Idea C: Pre-compute generation prompt tokens**
- Tokenize generation prompt once at start
- Manually append when needed
- Saves 2 tokenization calls per turn

### 3. Can we use `response.token_ids` directly?

**Question:** Why doesn't vLLM return the full assistant message tokens (with role headers)?

**Investigate:**
- Is there a vLLM setting to include role headers in response?
- Do other inference engines (TGI, SGLang) include role headers?
- Could we modify Generator to add role headers to `response.token_ids`?

**Benefits if possible:**
- Eliminate call #4 (assistant token extraction via prefix matching)
- Reduce complexity significantly

### 4. Alternative token storage approach

**Current:** `all_tokens` stores everything, `response_mask` indicates trainable
**Alternative:** Store separately?
- `prompt_tokens`: List of prompt token lists per turn
- `response_tokens`: List of response token lists per turn
- Reconstruct `all_tokens` when needed

**Questions:**
- Would this simplify logic?
- Does it break compatibility with Episode schema?
- How would truncation work?

### 5. Can we eliminate the messages/all_tokens mismatch?

**Current issue:** When truncating env obs:
- `messages[-1]["content"]` = full text
- `all_tokens` = truncated tokens

**Alternative approaches:**
- Always update message content to match truncated tokens
- Keep two separate message logs (full vs truncated)
- Accept the mismatch but document it better

## How to Proceed with Research

**When ready to investigate (FUTURE WORK):**

1. **Launch exploration agents:**

2. **Analyze findings:**
   - Count tokenization calls in other libraries
   - Identify any clever optimizations
   - Check if our approach is unnecessarily complex

3. **Prototype simplifications:**
   - Test if proposed optimizations maintain correctness
   - Validate with test_simple_vllm.py test suite
   - Measure performance impact

## Success Criteria

A simplified implementation should:
1. ✅ Pass all 5 test cases in `test_simple_vllm.py`
2. ✅ Reduce number of `apply_chat_template` calls
3. ✅ Maintain exact token sequence correctness
4. ✅ Support all truncation scenarios
5. ✅ Be easier to understand and maintain

## Notes

- **Do NOT sacrifice correctness for simplicity**
- Token sequence MUST match `apply_chat_template` output exactly
- All truncation edge cases must still work
- Performance is secondary to correctness
