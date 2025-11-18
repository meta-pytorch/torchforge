# Truncation Strategy Investigation - V2 (Code-Based Analysis)

**Date:** 2025-01-16
**Context:** Multi-turn blackjack refactor - understanding how production libraries handle truncation, variable group sizes, and reference model timing.

---

## Table of Contents

1. [Investigation Questions](#investigation-questions)
2. [Library-by-Library Analysis](#library-by-library-analysis)
   - [TRL](#trl)
   - [VERL](#verl)
   - [NeMo-RL](#nemo-rl)
   - [Tinker-Cookbook](#tinker-cookbook)
   - [Verifiers](#verifiers)
3. [Cross-Library Comparison](#cross-library-comparison)
4. [Discussion & Design Decisions](#discussion--design-decisions)
5. [Blackjack Implementation](#blackjack-implementation)

---

## Investigation Questions

### Q1: Variable Group Sizes - Continue with Fewer or Resample?

**User's concern:** "I am a bit afraid of dynamic batch sizes. AFAIK, it's always better to have a fixed batch size for things like compile. I would prefer to keep the batch size fixed."

**Three possible behaviors when episodes are truncated/invalid:**
- **(a)** Continue with fewer episodes in the group (e.g., 15 instead of 16)
- **(b)** Sample more data until exactly GROUP_SIZE valid episodes
- **(c)** Filter at dataset level before rollout

**What we need to know:**
- How do libraries handle vectorization/batching with variable sizes?
- Do they maintain fixed batch sizes for training?
- How does this interact with compiled models?

---

### Q2: Dataset Filtering vs Rollout Checking

**User's perspective:** "We should absolutely filter in the dataset to not include initial prompts > max_seq_len. This type of case should never get to the rollout, since it wastes resources * group_size."

**BUT:** "A lot of times the prompt will contain extra info, such as tool calling, state of the environment, etc. These we would only know when at the start of the rollout."

**What we need to know:**
- Do libraries filter at dataset level or rollout level?
- How do they handle prompts that grow during rollout (multi-turn)?
- Is there a best practice?

---

### Q3: Train on Partial Tokens - What Does "Masked" Mean?

**User's confusion:** "You said 'most libraries train on partial tokens by default', but also said that all of them mask complete truncation. So they ACTUALLY train on those, right?"

**Clarification needed:**
- When they say "train on truncated", do they mean:
  - Train on partial text (e.g., "STA" instead of "STAND")?
  - Or keep all turns but mask the truncated one (no gradient)?
- What exactly does "masking" do - zero loss or exclude from batch?

---

### Q4: Reference Model Timing

**User's proposed flow:** "Set reward to partial or 0, then run the reference model, compute the advantages, and then decide if we put it in the buffer or not."

**What we need to know:**
- Do libraries compute ref_logprobs for ALL episodes (including ones they'll drop)?
- Or do they filter first, then compute ref_logprobs only for kept episodes?
- What's the exact flow: rollout → ref_model → buffer decision, or rollout → buffer decision → ref_model?

---

## Library-by-Library Analysis

---

## TRL

### Repository
`/home/felipemello/forge/trl/`

### Q1: Variable Group Sizes

**Answer: ❌ Assumes fixed size - will break with variable groups**

**Code Evidence:**

**File:** `trl/trainer/grpo_trainer.py` (lines 1594-1607)
```python
# Calculate rewards for each reward function
rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

# Apply weights to each reward function's output and sum
rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

# Compute grouped-wise rewards
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
# ^^^^ ASSUMES EXACTLY num_generations per prompt

# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
advantages = rewards - mean_grouped_rewards
```

**Critical line:** `rewards.view(-1, self.num_generations)` **requires** exactly `num_generations` samples per prompt. If you have variable group sizes (e.g., 15 instead of 16), this will crash with:
```
RuntimeError: shape '[-1, 16]' is invalid for input of size 15
```

**Batching for training:**

**File:** `trl/trainer/grpo_trainer.py` (lines 1685-1711)
```python
output = {
    "prompt_ids": prompt_ids,                    # [batch_size, seq_len]
    "prompt_mask": prompt_mask,                  # [batch_size, seq_len]
    "completion_ids": completion_ids,            # [batch_size, max_completion_length]
    "completion_mask": completion_mask,          # [batch_size, max_completion_length]
    "advantages": advantages,                    # [batch_size]
    "num_items_in_batch": num_items_in_batch,
}
if ref_per_token_logps is not None:
    output["ref_per_token_logps"] = ref_per_token_logps
```

All arrays are padded to fixed dimensions (`max_completion_length`), so training batch size is fixed.

**Conclusion:** TRL maintains fixed batch sizes for training, but **requires** fixed group sizes during rollout. Cannot handle variable groups.

---

### Q2: Dataset Filtering vs Rollout Checking

**Answer: No dataset-level filtering - checking happens during generation**

**Code Evidence:**

**File:** `trl/trainer/grpo_trainer.py` (lines 1396-1432)
```python
def _generate(self, prompts: list):
    device = self.accelerator.device
    mode = "train" if self.model.training else "eval"

    prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

    # Get completion length per sequence, used for logging
    prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
    completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)

    # Identify sequences that terminated with EOS and log their lengths
    eos_and_pad = [self.eos_token_id, self.pad_token_id]
    is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
    agg_is_truncated = self.accelerator.gather(is_truncated)
    self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
```

**Truncation detection:** A sequence is truncated if its **last token** is NOT `eos_token_id` or `pad_token_id`.

**No pre-filtering:** The dataset returns raw prompts, and truncation is only detected AFTER generation.

**Example from OpenEnv scripts:**

**File:** `trl/examples/scripts/openenv/catch.py` (lines 162-216)
```python
def rollout_func(
    prompts: list[str], args: GRPOConfig, processing_class, client: OpenSpielEnv, gen_url: str
) -> dict[str, list]:
    """Generate completions via vLLM and compute environment rewards."""
    env_rewards = []
    all_prompt_ids, all_completion_ids, all_logprobs = [], [], []

    for base_prompt in prompts:
        for _ in range(args.num_generations):  # Generate args.num_generations per prompt
            env_result = client.reset()
            obs = env_result.observation
            total_reward = 0.0

            episode_prompt_ids, episode_completion_ids, episode_logprobs = [], [], []

            while not obs.done:
                # Generate action
                episode_msg = {"prompt": [{"role": "user", "content": f"{base_prompt}\n\n{obs.info_state}\n"}]}
                episode_prompt = apply_chat_template(episode_msg, processing_class)

                # No prompt length check here!
                result = requests.post(gen_url, json=payload).json()

                episode_prompt_ids.extend(result["prompt_ids"][0])
                episode_completion_ids.extend(result["completion_ids"][0])
                episode_logprobs.extend(result["logprobs"][0])

                # Step environment
                # ...

            env_rewards.append(total_reward)
            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": env_rewards,
    }
```

**No budget checking** during rollout - episodes can grow unbounded.

**Conclusion:** TRL does NOT filter at dataset level. Truncation is detected post-generation, and there's no explicit budget enforcement during multi-turn rollouts.

---

### Q3: Train on Partial Tokens - What Does "Masked" Mean?

**Answer: By default, train on partial tokens. With `mask_truncated_completions=True`, zero out the ENTIRE episode's gradient.**

**Code Evidence:**

**File:** `trl/trainer/grpo_trainer.py` (lines 1480-1485)
```python
# If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
if self.mask_truncated_completions:
    eos_and_pad = [self.eos_token_id, self.pad_token_id]
    is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
    completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
    # ^^^^ Sets completion_mask = 0 for ALL tokens in truncated episodes
```

**What `completion_mask` does:**

**File:** `trl/trainer/grpo_trainer.py` (lines 1739-1752)
```python
def grpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    advantages: torch.FloatTensor,
    completion_masks: torch.FloatTensor,  # <-- Used here
) -> torch.FloatTensor:
    # ...
    per_token_loss = -advantages.unsqueeze(1) * policy_chosen_logps - beta * kl
    # Apply mask to zero out non-completion tokens and truncated sequences
    masked_loss = per_token_loss * completion_masks
    # ^^^^ Tokens where completion_mask=0 contribute zero loss

    # Average over non-masked tokens
    loss = masked_loss.sum() / completion_masks.sum()
    return loss
```

**Behavior:**

| Setting | Partial tokens (e.g., "STA") in batch? | Gradient computed? |
|---------|----------------------------------------|--------------------|
| `mask_truncated_completions=False` (default) | ✅ Yes | ✅ Yes - trains on "S", "T", "A" |
| `mask_truncated_completions=True` | ✅ Yes (still in batch) | ❌ No - `completion_mask=0` for entire episode |

**Config documentation:**

**File:** `trl/trainer/grpo_config.py` (lines 210-213)
```python
# mask_truncated_completions (`bool`, *optional*, defaults to `False`):
#     When enabled, truncated completions are excluded from the loss calculation, preventing them from being
#     incorrectly penalized and introducing noise during training. According to the
#     [DAPO](https://huggingface.co/papers/2503.14476) paper, this is a good practice for training stability.
```

**Conclusion:** By default, TRL **trains on partial tokens** like "STA". With masking enabled, it keeps the episode in the batch but zeros its gradient contribution.

---

### Q4: Reference Model Timing

**Answer: ref_model called AFTER generation, BEFORE buffer decision, for ALL episodes (including truncated ones)**

**Code Evidence:**

**File:** `trl/trainer/grpo_trainer.py` - Full flow (lines 1461-1711)

```python
# Step 1: Generation
prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
    self._generate(prompts)  # Line 1461-1463
)

# Step 2: Build completion_mask (initially all 1s for non-padding tokens)
completion_mask = torch.stack(
    [torch.tensor([token_id != self.pad_token_id for token_id in ids]) for ids in completion_ids_list]
).int()  # Line 1479

# Step 3: Apply truncation masking (BUFFER DECISION)
if self.mask_truncated_completions:
    eos_and_pad = [self.eos_token_id, self.pad_token_id]
    is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
    completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()  # Line 1480-1485

# Step 4: Compute reference model logprobs (AFTER masking decision, but FOR ALL EPISODES)
with torch.no_grad():
    if self.beta != 0.0:
        if self.ref_model is not None:
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                num_images=num_images,
                **forward_kwargs,
            )  # Lines 1545-1569
        else:
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )

# Step 5: Compute rewards
rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)  # Line 1597

# Step 6: Return to buffer (all episodes, with masking already applied)
output = {
    "prompt_ids": prompt_ids,
    "prompt_mask": prompt_mask,
    "completion_ids": completion_ids,
    "completion_mask": completion_mask,  # Truncated episodes have mask=0
    "advantages": advantages,
    "num_items_in_batch": num_items_in_batch,
}
if ref_per_token_logps is not None:
    output["ref_per_token_logps"] = ref_per_token_logps  # Lines 1685-1711
```

**Exact flow:**
```
1. rollout → generate episodes
2. detect truncation (is_truncated = last_token not in [eos, pad])
3. apply completion_mask (BUFFER DECISION: mask=0 for truncated if config enabled)
4. ← ref_model.forward() for ALL episodes (including masked ones)
5. compute rewards for ALL episodes
6. compute advantages
7. add to buffer (all episodes, some with mask=0)
```

**Key insight:** ref_model computes logprobs for **ALL** episodes, including truncated ones. The masking only affects gradient flow during loss computation, not whether ref_model runs.

**Conclusion:** TRL follows the pattern: rollout → masking decision → **ref_model (all episodes)** → buffer → train.

---

### TRL Summary

| Question | Answer | Key Mechanism |
|----------|--------|---------------|
| **Q1: Variable groups** | ❌ Cannot handle - assumes fixed size | `.view(-1, num_generations)` requires exact count |
| **Q2: Dataset filtering** | ❌ No filtering - truncation detected post-generation | Checking happens in `_generate()` |
| **Q3: Train on partial** | ✅ Yes by default, mask=0 if config enabled | `completion_mask` controls gradient, not batch membership |
| **Q4: Ref model timing** | After masking, before buffer, **for all episodes** | Single batched call processes everything |

---

## VERL

### Repository
`/home/felipemello/forge/verl/`

### Q1: Variable Group Sizes

**Answer: ✅ Continue with fewer episodes - handles variable sizes via sequence balancing**

**Code Evidence:**

**File:** `verl/trainer/ppo/ray_trainer.py` (lines 1031-1077)
```python
# Repeat prompts by rollout.n times
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
)

# ... generate sequences ...

# repeat to align with repeated responses in rollout
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

**No explicit GROUP_SIZE enforcement.** All generated episodes proceed to the next stage.

**Handling variable lengths:**

**File:** `verl/trainer/ppo/ray_trainer.py` (lines 1082-1086)
```python
if self.config.trainer.balance_batch:
    self._balance_batch(batch, metrics=metrics)
```

**File:** `verl/trainer/ppo/ray_trainer.py` (lines 919-954)
```python
def _balance_batch(self, batch: DataProto, metrics: dict = None):
    """Balance batch across DP ranks by total token count, not number of sequences"""

    # Get sequence lengths
    input_ids = batch.batch["input_ids"]
    seq_lens = (input_ids != self.tokenizer.pad_token_id).sum(dim=-1).cpu().numpy()

    # Partition sequences across DP ranks to balance total tokens
    dp_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes // self.config.trainer.ppo_mini_batch_size
    partitions = get_seqlen_balanced_partitions(seq_lens, dp_size)

    # Each rank gets a different number of sequences, but similar total tokens
    # ...
```

**Key insight:** VERL uses **sequence balancing**, NOT fixed batch sizes. Each DP rank gets different numbers of sequences, balanced by total token count.

**Truncation creates variable lengths:**

**File:** `verl/experimental/agent_loop/tool_agent_loop.py` (lines 165-182)
```python
# Finalize output
response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
output = AgentLoopOutput(
    prompt_ids=prompt_ids,
    response_ids=response_ids[: self.response_length],  # Truncate to response_length
    response_mask=agent_data.response_mask[: self.response_length],
    # ...
)
```

Episodes are truncated at `self.response_length`, creating variable-length sequences.

**Conclusion:** VERL explicitly handles variable group sizes and variable sequence lengths. It maintains dynamic batch sizes balanced by token count, not sequence count.

---

### Q2: Dataset Filtering vs Rollout Checking

**Answer: Rollout-level checking - budget enforced during generation**

**Code Evidence:**

**File:** `verl/experimental/agent_loop/tool_agent_loop.py` (lines 233-239)
```python
async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
    # ... generation ...

    # Check termination conditions
    if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED
    if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
        return AgentState.TERMINATED
    if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
        return AgentState.TERMINATED
```

**No dataset-level filtering.** Budget is checked **during rollout** after each turn:
- `len(agent_data.response_mask) >= self.response_length` → episode terminates
- Episodes can grow turn-by-turn until hitting budget

**Multi-turn prompt growth:**

**File:** `verl/experimental/agent_loop/tool_agent_loop.py` (lines 324-361)
```python
async def _handle_processing_tools_state(self, agent_data):
    # Execute tools
    add_messages = []
    for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
        tool_response = await self._call_tool(tool_call, agent_data.tools_kwargs)
        add_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": tool_response_text,
        })

    # Add all tool messages
    agent_data.messages.extend(add_messages)

    # Tokenize the new messages
    response_ids = await self.loop.run_in_executor(
        None,
        lambda: self.tokenizer.apply_chat_template(
            add_messages, add_generation_prompt=True, tokenize=True
        ),
    )

    # Check if total exceeds budget (ROLLOUT-LEVEL CHECK)
    if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
        return AgentState.TERMINATED  # Episode ends
```

**Conclusion:** VERL does NOT filter at dataset level. It checks budget during rollout, allowing prompts to grow multi-turn until hitting `response_length`.

---

### Q3: Train on Partial Tokens - What Does "Masked" Mean?

**Answer: ✅ VERL terminates cleanly at turn boundaries - NO partial tokens generated**

**Code Evidence:**

**File:** `verl/experimental/agent_loop/tool_agent_loop.py` (lines 233-239)
```python
# Check termination BEFORE generating next turn
if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
    return AgentState.TERMINATED  # Episode ends BEFORE generating partial tokens
```

**VERL is unique:** It checks budget **before** each generation, so it never generates partial tokens like "STA". The conversation ends cleanly with complete turns only.

**Example flow:**
```
Turn 1: prompt=100 tokens, response=50 tokens, total=150
Turn 2: prompt=150 tokens (includes turn 1), response=80 tokens, total=230
Turn 3: Check: prompt=230 tokens, would generate more
        → len(response_mask) >= response_length (250)
        → TERMINATE before generating
```

**Output truncation:**

**File:** `verl/workers/rollout/schemas.py` (lines 658-673)
```python
def truncate_output_ids(
    self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
) -> None:
    """Truncate sequences to max_model_len"""
    self.input_ids = self.input_ids[..., : self.max_model_len]
    self.attention_mask = self.attention_mask[..., : self.max_model_len]
    self.position_ids = self.position_ids[..., : self.max_model_len]
    self.loss_mask = self.loss_mask[..., : self.max_model_len]
    self.response_ids = self.input_ids[..., self.prompt_ids.shape[-1] :][..., : self.max_response_len]
    self.response_attention_mask = self.attention_mask[..., self.prompt_attention_mask.shape[-1] :][
        ..., : self.max_response_len
    ]
```

This is a **safety truncation** at the sequence level (if somehow it exceeds), not turn-level truncation.

**Conclusion:** VERL does NOT train on partial tokens. It terminates episodes cleanly at turn boundaries before generating partial text.

---

### Q4: Reference Model Timing

**Answer: ref_model called AFTER generation, for ALL episodes**

**Code Evidence:**

**File:** `verl/trainer/ppo/ray_trainer.py` (lines 1037-1144) - Full flow

```python
# Step 1: Generate sequences
with marked_timer("gen", timing_raw, color="red"):
    if not self.async_rollout_mode:
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
    else:
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

# Step 2: Combine with original batch
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)

# Step 3: Compute reward
with marked_timer("reward", timing_raw, color="yellow"):
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

# Step 4: Compute old_log_probs (if needed)
if need_recomputation:
    with marked_timer("old_log_prob", timing_raw, color="blue"):
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        batch = batch.union(old_log_prob)

# Step 5: Compute ref_log_prob (THIS IS THE KEY!)
if self.use_reference_policy:
    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
        if not self.ref_in_actor:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        else:
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)  # Lines 1082-1099

# Step 6: Compute values (critic)
if self.use_critic:
    with marked_timer("values", timing_raw, color="cyan"):
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)
```

**Exact flow:**
```
1. rollout → generate_sequences
2. union → combine with prompts
3. reward → compute rewards on ALL episodes
4. old_log_prob → compute current policy logprobs (for rollout correction)
5. ← ref_log_prob → compute reference policy logprobs on ALL episodes
6. values → compute critic values
7. train
```

**No selective ref_model computation.** Every episode that enters the batch goes through ref_model.

**Why this matters:** In VERL, there's no explicit "buffer decision" with accept/reject logic. ALL generated episodes are processed through the full pipeline unconditionally.

**Conclusion:** VERL follows: rollout → **ref_model (all episodes)** → train. No filtering before ref_model.

---

### VERL Summary

| Question | Answer | Key Mechanism |
|----------|--------|---------------|
| **Q1: Variable groups** | ✅ Continue with fewer - handles variable sizes | Sequence balancing by token count, not sequence count |
| **Q2: Dataset filtering** | ❌ Rollout-level checking | Budget checked during generation via `response_length` |
| **Q3: Train on partial** | ❌ No - clean turn termination | Checks budget BEFORE generating, never creates partial tokens |
| **Q4: Ref model timing** | After rollout, before training, **for all episodes** | Sequential pipeline processes everything |

---

## NeMo-RL

### Repository
`/home/felipemello/forge/RL/`

### Q1: Variable Group Sizes

**Answer: ✅ Sample more until exact size (in dynamic sampling mode), OR continue with fewer (standard mode)**

**Code Evidence:**

**Dynamic Sampling Mode:**

**File:** `RL/nemo_rl/algorithms/grpo.py` (lines 541-667)
```python
def dynamic_sampling(
    repeated_batch,
    std,
    baseline,
    master_config,
    batch_cache=None,
    dynamic_sampling_num_gen_batches=1,
):
    """
    Dynamic sampling: filter prompts with zero std, sample more batches until we have enough.
    """
    # Required batch size for training
    train_prompts_size = (
        master_config["grpo"]["num_prompts_per_step"]
        * master_config["grpo"]["num_generations_per_prompt"]
    )

    if master_config["grpo"]["use_dynamic_sampling"]:
        # Get the prompt indices with non-zero std
        non_zero_std_mask = std != 0.0
        keep_prompt_indices = torch.arange(len(non_zero_std_mask))[non_zero_std_mask].tolist()

        # Only select the inputs that have non-zero std
        filtered_repeated_batch = repeated_batch.select_indices(keep_prompt_indices)

        # If none of the prompts have non-zero std, skip this batch
        if filtered_repeated_batch.size > 0:
            # Concatenate with previous batch cache
            batch_cache = (
                filtered_repeated_batch if batch_cache is None
                else BatchedDataDict.from_batches([batch_cache, filtered_repeated_batch])
            )

        filtered_prompts_size = batch_cache.size if batch_cache is not None else 0

        # If insufficient, keep sampling more batches
        if filtered_prompts_size < train_prompts_size:
            if dynamic_sampling_num_gen_batches <= master_config["grpo"].get("dynamic_sampling_max_gen_batches", 10):
                is_batch_complete = False  # Signal to continue sampling
            else:
                raise ValueError(f"Reached max generation batches ({dynamic_sampling_max_gen_batches})")
        else:
            # We have enough! Slice to exact size
            batch_cache = batch_cache.select_indices(list(range(train_prompts_size)))
            is_batch_complete = True

        return batch_cache, is_batch_complete, batch_cache, metrics
    else:
        # Standard mode: no filtering
        return repeated_batch, True, None, {}
```

**Behavior:**
- **Dynamic mode:** Caches partial batches, samples more until exactly `num_prompts_per_step * num_generations_per_prompt` valid episodes
- **Standard mode:** No filtering, all episodes proceed

**Standard Mode (no dynamic sampling):**

**File:** `RL/nemo_rl/algorithms/grpo.py` (lines 924-927)
```python
# Always maintain exact group size by repeating prompts
repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
    master_config["grpo"]["num_generations_per_prompt"]
)
```

**Batching for training:**

**File:** `RL/nemo_rl/algorithms/grpo.py` (lines 1086-1123)
```python
# Convert to flat messages for training
flat_messages, input_lengths = batched_message_log_to_flat_message(
    repeated_batch["message_log"],
    truncate_to_max_len=master_config["grpo"]["truncate_to_max_len"],
)

train_data = BatchedDataDict[ClippedPGLossDataDict]({
    "input_ids": flat_messages["token_ids"],          # Variable length sequences
    "advantages": flat_messages["advantages"],
    "response_mask": flat_messages["response_mask"],  # Marks assistant tokens
    "loss_multiplier": repeated_batch["loss_multiplier"],  # Can be 0 for truncated
    # ...
})
```

**Fixed vs variable batch sizes:**
- Dynamic mode: **Fixed batch size** (resamples to exact count)
- Standard mode: **Fixed batch size** (repeats prompts exactly `num_generations_per_prompt` times)
- Within batch: **Variable sequence lengths** (handled by padding/masking)

**Conclusion:** NeMo-RL maintains fixed batch sizes by either resampling (dynamic mode) or fixed repetition (standard mode). Variable-length sequences within batches are handled via masking.

---

### Q2: Dataset Filtering vs Rollout Checking

**Answer: Rollout-level checking - budget enforced per-turn during multi-turn rollouts**

**Code Evidence:**

**File:** `RL/nemo_rl/experience/rollouts.py` (lines 444-470)
```python
# Multi-turn rollout loop
for turn_idx in range(max_rollout_turns):
    # ... generate response ...

    # Calculate reward and get environment observation
    env_output = calculate_rewards(active_batch, task_to_env)

    truncation_mask = torch.zeros_like(env_output.terminateds, dtype=torch.bool)

    for i, global_idx in enumerate(active_indices.tolist()):
        env_obs_content = env_output.observations[i]["content"]

        # Tokenize environment observation (tool result / game state)
        tokenized_obs = tokenizer(
            env_obs_content,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        # CHECK IF NEW MESSAGE OVERFLOWS max_seq_len
        if (len(tokenized_obs) + len(generated_ids[i]) + active_input_lengths[i] >= max_seq_len):
            # Calculate remaining budget
            tokens_left_for_obs = max_seq_len - (len(generated_ids[i]) + active_input_lengths[i])

            # Truncate the environment observation (not the generation!)
            tokenized_obs = tokenized_obs[:tokens_left_for_obs]
            truncation_mask[i] = True

            # Record truncation
            sample_truncated[active_indices[i]] = True
```

**No dataset-level filtering.** Episodes start from dataset prompts and grow turn-by-turn. Budget is checked **after each generation** to decide whether to truncate the environment observation.

**Truncation strategy:** Truncate **environment response** (tool results / game state), NOT the model generation. The model's text is kept intact.

**Conclusion:** NeMo-RL does NOT filter at dataset level. It checks budget during rollout and dynamically truncates environment observations to fit remaining budget.

---

### Q3: Train on Partial Tokens - What Does "Masked" Mean?

**Answer: Train on full generated text (e.g., "STAND"), but truncate environment response. Can zero loss via `loss_multiplier`.**

**Code Evidence:**

**Truncation detection (from Q2 above):**
- Sets `sample_truncated[i] = True` for episodes that hit `max_seq_len`
- Truncates **environment observation** to fit remaining budget
- Model's generated text is NOT truncated

**Overlong filtering:**

**File:** `RL/nemo_rl/algorithms/grpo.py` (lines 1066-1075)
```python
use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
if use_overlong_filtering:
    loss_multiplier = repeated_batch["loss_multiplier"].clone()
    truncated = repeated_batch["truncated"]

    if isinstance(truncated, list):
        truncated = torch.tensor(truncated, dtype=torch.bool)

    # Zero out loss for truncated samples
    loss_multiplier[truncated] = 0
    repeated_batch["loss_multiplier"] = loss_multiplier
```

**What `loss_multiplier` does:**

**File:** `RL/nemo_rl/algorithms/clipped_pg_loss.py` (lines 45-87)
```python
def clipped_policy_gradient_loss(
    logprobs,
    prev_logprobs,
    advantages,
    response_mask,
    loss_multiplier,  # <-- Used here
    eps=0.2,
):
    # Calculate importance ratio
    ratio = torch.exp(logprobs - prev_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)

    # Policy gradient loss
    pg_loss_unclipped = -advantages * ratio
    pg_loss_clipped = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss_unclipped, pg_loss_clipped)

    # Apply response_mask (only train on assistant tokens) and loss_multiplier (zero for truncated)
    masked_pg_loss = pg_loss * response_mask * loss_multiplier.unsqueeze(-1)
    # ^^^^ Tokens where loss_multiplier=0 contribute zero gradient

    # Average over non-masked tokens
    loss = masked_pg_loss.sum() / (response_mask * loss_multiplier.unsqueeze(-1)).sum().clamp(min=1.0)
    return loss
```

**Behavior:**

| Setting | Generated text in batch? | Env response truncated? | Gradient computed? |
|---------|-------------------------|-------------------------|-------------------|
| `overlong_filtering=False` (default) | ✅ Full (e.g., "STAND") | ✅ Yes (to fit budget) | ✅ Yes |
| `overlong_filtering=True` | ✅ Full (e.g., "STAND") | ✅ Yes (to fit budget) | ❌ No - `loss_multiplier=0` |

**Conclusion:** NeMo-RL does NOT train on partial tokens. It keeps full model generations but truncates environment observations. With `overlong_filtering=True`, it zeros `loss_multiplier` for truncated episodes (no gradient).

---

### Q4: Reference Model Timing

**Answer: Rollout → filter (optional) → ref_model (only for kept episodes)**

**Code Evidence:**

**File:** `RL/nemo_rl/algorithms/grpo.py` (lines 936-1132) - Full flow

```python
# Step 1: Generation (rollout)
with timer.time("generation"):
    repeated_batch, rollout_metrics = run_multi_turn_rollout(
        policy_generation=policy_generation,
        input_batch=repeated_batch,
        tokenizer=tokenizer,
        max_seq_len=master_config["grpo"]["max_seq_len"],
        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
        # ...
    )
    policy_generation.finish_generation()

# Step 2: Reward processing & filtering decision
with timer.time("reward_calculation"):
    rewards = repeated_batch["total_reward"]
    baseline, std = calculate_baseline_and_std_per_prompt(
        rewards,
        master_config["grpo"]["num_generations_per_prompt"],
    )

    # Dynamic sampling filtering happens HERE
    repeated_batch, is_batch_complete, batch_cache, ds_metrics = dynamic_sampling(
        repeated_batch, std, baseline, master_config, batch_cache, dynamic_sampling_num_gen_batches
    )

    # If not enough samples, skip to next batch WITHOUT calling ref_model
    if not is_batch_complete:
        continue  # <-- Skips ref_model!

# Step 3: Data preparation (still before ref_model)
with timer.time("data_processing"):
    # Add loss masks, advantages, etc.
    for i, message_log in enumerate(repeated_batch["message_log"]):
        for j, message in enumerate(message_log):
            if message["role"] == "assistant":
                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
            message["advantages"] = advantages[i].expand(message["token_ids"].shape)

    # Convert to training format
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        repeated_batch["message_log"],
        truncate_to_max_len=master_config["grpo"]["truncate_to_max_len"],
    )
    train_data = BatchedDataDict[ClippedPGLossDataDict]({
        "input_ids": flat_messages["token_ids"],
        "advantages": flat_messages["advantages"],
        "response_mask": flat_messages["response_mask"],
        "loss_multiplier": repeated_batch["loss_multiplier"],
        # ...
    })

# Step 4: Reference model logprobs (AFTER buffer decision, ONLY for kept episodes)
print("▶ Preparing for logprob inference...", flush=True)
with timer.time("logprob_inference_prep"):
    policy.prepare_for_lp_inference()

print("▶ Computing logprobs...", flush=True)
with timer.time("policy_and_reference_logprobs"):
    fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
    reference_logprobs = policy.get_reference_policy_logprobs(train_data)["reference_logprobs"]
    # ^^^^ ref_model called here, AFTER filtering, ONLY for is_batch_complete=True

    train_data["prev_logprobs"] = fprop_logprobs
    train_data["reference_policy_logprobs"] = reference_logprobs
```

**Exact flow:**
```
1. rollout → generate episodes
2. reward → compute rewards
3. filter (dynamic sampling) → keep only non-zero std prompts
4. if not enough samples: continue (skip ref_model)
5. if enough samples: data preparation
6. ← ref_model.get_reference_policy_logprobs() ONLY for kept episodes
7. train
```

**Key insight:** NeMo-RL skips ref_model for incomplete batches. Only batches with enough valid samples get ref_logprobs computed.

**Conclusion:** NeMo-RL follows: rollout → filter → **ref_model (only kept episodes)** → train.

---

### NeMo-RL Summary

| Question | Answer | Key Mechanism |
|----------|--------|---------------|
| **Q1: Variable groups** | ✅ Sample more (dynamic mode) OR fixed size (standard mode) | Dynamic sampling caches batches, resamples to exact size |
| **Q2: Dataset filtering** | ❌ Rollout-level checking | Budget checked per-turn, truncates env observations |
| **Q3: Train on partial** | ❌ No - keeps full model generation, truncates env | `loss_multiplier=0` for truncated if `overlong_filtering=True` |
| **Q4: Ref model timing** | After filter, before training, **only for kept episodes** | `continue` skips ref_model if batch incomplete |

---

## Tinker-Cookbook

### Repository
`/home/felipemello/forge/tinker-cookbook/`

### Q1: Variable Group Sizes

**Answer: ✅ Continue with fewer episodes - explicitly trains on smaller batches**

**Code Evidence:**

**File:** `tinker_cookbook/rl/train.py` (lines 987-1006)
```python
# Generate trajectory groups in parallel
trajectory_groups_P = await asyncio.gather(
    *[
        asyncio.create_task(
            do_group_rollout_and_filter_constant_reward(
                sampling_client,
                builder,
                max_tokens=cfg.max_tokens,
                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                enable_logging=i < cfg.num_groups_to_log,
            ),
            name=f"sample_task_{i}",
        )
        for i, builder in enumerate(env_group_builders_P)
    ],
)

# Filter out None groups (filtered due to constant rewards)
trajectory_groups_P = [
    trajectory_group
    for trajectory_group in trajectory_groups_P
    if trajectory_group is not None  # <-- Filter out dropped groups
]
```

**Filtering logic:**

**File:** `tinker_cookbook/rl/train.py` (lines 657-676)
```python
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
) -> TrajectoryGroup | None:
    """Rollout a group and optionally filter if all rewards are the same"""
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    with logtree.optional_enable_logging(enable_logging):
        trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Remove if all trajectories have the same reward (no gradient signal)
    trajectory_groups = [trajectory_group]
    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if len(trajectory_groups) == 0:
        return None  # <-- Returns None if filtered out
    return trajectory_groups[0]
```

**File:** `tinker_cookbook/rl/data_processing.py` (lines 198-209)
```python
def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    """Filter out groups where all rewards are identical (no learning signal)"""
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
    return new_groups
```

**Batching with variable sizes:**

**File:** `tinker_cookbook/rl/train.py` (lines 837-846)
```python
# Note: we may have removed trajectory groups that have the same reward.
# To have the same results as the sync implementation, we will
# remove these and train on a smaller batch.
wrapped_trajectory_groups = [g for g in wrapped_trajectory_groups if g is not None]

data_D, prepare_minibatch_metrics = await prepare_minibatch(
    [g.env_group_builder for g in wrapped_trajectory_groups],
    [g.trajectory_group for g in wrapped_trajectory_groups],
    tokenizer,
    service_client,
    model_name=cfg.model_name,
    kl_penalty_coef=cfg.kl_penalty_coef,
    kl_discount_factor=cfg.kl_discount_factor,
)
```

**Explicit comment:** "we will remove these and train on a smaller batch."

**Conclusion:** Tinker explicitly handles variable group sizes by training on smaller batches when groups are filtered. No resampling, no fixed size requirement.

---

### Q2: Dataset Filtering vs Rollout Checking

**Answer: Rollout-level checking - budget enforced during multi-turn episodes**

**Code Evidence:**

**File:** `tinker_cookbook/recipes/tool_use/search/search_env.py` (lines 161-195)
```python
async def step(self, action: Action) -> StepResult:
    """Execute one step of the environment"""
    message, parse_success = self.renderer.parse_response(action)

    self.past_messages.append(message)

    if "tool_calls" in message:
        failure_result = StepResult(
            reward=0.0,
            episode_done=True,  # <-- Episode terminates
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
        )

        if message["tool_calls"][0]["name"] == "search":
            self.current_num_calls += 1
            if self.current_num_calls > self.max_num_calls:
                return failure_result  # Too many calls

            try:
                tool_return_message = await self.call_search_tool(message["tool_calls"][0])
                self.past_messages.extend(tool_return_message)
            except Exception as e:
                logger.error(f"Error calling search tool: {repr(e)}")
                return failure_result  # Tool error

            # Rebuild prompt from FULL history
            next_observation = self.renderer.build_generation_prompt(self.past_messages)

            # CHECK BUDGET (ROLLOUT-LEVEL)
            if next_observation.length > self.max_trajectory_tokens:
                return failure_result  # <-- TRUNCATION: Episode ends with reward=0

            return StepResult(
                reward=0.0,
                episode_done=False,  # Continue if within budget
                next_observation=next_observation,
                next_stop_condition=self.stop_condition,
            )
```

**No dataset-level filtering.** Budget is checked **after adding tool results** to the conversation.

**Constructor:**

**File:** `tinker_cookbook/recipes/tool_use/search/search_env.py` (lines 108-117)
```python
class SearchEnv(ProblemEnv):
    def __init__(
        self,
        ...,
        max_trajectory_tokens: int = 32 * 1024,
        max_num_calls: int = 10,
    ):
        self.past_messages: list[renderers.Message] = []
        self.max_trajectory_tokens = max_trajectory_tokens
        self.current_num_calls = 0
```

**Conclusion:** Tinker does NOT filter at dataset level. It checks budget during rollout and terminates episodes when `next_observation.length > max_trajectory_tokens`.

---

### Q3: Train on Partial Tokens - What Does "Masked" Mean?

**Answer: Episode ends with failure reward when budget exceeded - full trajectory kept, but penalized**

**Code Evidence:**

**Truncation behavior (from Q2 above):**
- When budget exceeded: `return failure_result` with `reward=0.0` and `episode_done=True`
- The **entire trajectory** (all previous turns) is kept
- No partial tokens are generated (episode ends before next generation)

**Rollout structure:**

**File:** `tinker_cookbook/rl/rollouts.py` (lines 16-34)
```python
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    """Run a single episode until completion"""
    transitions = []
    ob, stop_condition = await env.initial_observation()

    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)

        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)

        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition

        if step_result.episode_done:  # <-- Breaks when truncated
            break

    return Trajectory(transitions=transitions, final_ob=ob)
```

All transitions (including the one that triggered truncation) are saved in the trajectory.

**No masking mechanism.** Episodes are penalized via `reward=0.0`, but all tokens contribute to loss.

**Conclusion:** Tinker does NOT train on partial tokens (episode ends before generating them) and does NOT mask truncated episodes. It penalizes them with `reward=0.0`.

---

### Q4: Reference Model Timing

**Answer: Rollout → filter → ref_model (only for kept episodes)**

**Code Evidence:**

**File:** `tinker_cookbook/rl/train.py` (lines 657-676) - Rollout and filtering

```python
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
) -> TrajectoryGroup | None:
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    with logtree.optional_enable_logging(enable_logging):
        trajectory_group = await do_group_rollout(env_group_builder, policy)
    # ^^^^ No ref_model called here - only current policy

    # Filter based on rewards
    trajectory_groups = [trajectory_group]
    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if len(trajectory_groups) == 0:
        return None  # Filtered out
    return trajectory_groups[0]
```

**File:** `tinker_cookbook/rl/train.py` (lines 702-740) - Reference model during training preparation

```python
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # ... assemble training data from trajectory_groups_P (ONLY kept episodes) ...

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=model_name),
                # ^^^^ THIS is where ref_model is called
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics
```

**File:** `tinker_cookbook/rl/metrics.py` (lines 86-131) - KL penalty computation

```python
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> Dict[str, float]:
    """
    Compute KL against base model. Adjust advantages in-place.
    """
    # Compute logprobs at all data items (ONLY for episodes in data_D)
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]

    # ← ref_model called here
    base_logprobs_D = await asyncio.gather(
        *[
            base_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )

    # ... compute KL penalty and adjust advantages ...
```

**Exact flow:**
```
1. rollout → do_group_rollout (current policy only)
2. filter → remove_constant_reward_groups (returns None for dropped)
3. if filtered: return None (no ref_model call)
4. if kept: prepare_minibatch
5.   ← ref_model.compute_logprobs_async() for ONLY kept episodes
6. train
```

**Key insight:** ref_model is called **only for episodes that will be trained on**, after the buffer decision.

**Conclusion:** Tinker follows: rollout → filter → **ref_model (only kept episodes)** → train.

---

### Tinker-Cookbook Summary

| Question | Answer | Key Mechanism |
|----------|--------|---------------|
| **Q1: Variable groups** | ✅ Continue with fewer - explicit support | Trains on smaller batches when groups filtered |
| **Q2: Dataset filtering** | ❌ Rollout-level checking | Budget checked after adding tool results |
| **Q3: Train on partial** | ❌ No partial tokens - episode ends with `reward=0.0` | Clean termination before next generation |
| **Q4: Ref model timing** | After filter, before training, **only for kept episodes** | KL penalty computed in `prepare_minibatch()` |

---

## Verifiers

### Repository
`/home/felipemello/forge/verifiers/`

### Q1: Variable Group Sizes

**Answer: ✅ Continue with fewer episodes - dynamic advantage computation**

**Code Evidence:**

**File:** `verifiers/rl/trainer/orchestrator.py` (lines 251-262)
```python
# Compute advantages per prompt group
for prompt_idx in range(prompts_in_batch):
    group_indices = [
        prompt_idx + k * prompts_in_batch
        for k in range(self.rollouts_per_example)
        if (prompt_idx + k * prompts_in_batch) < len(rewards)  # ← Allows partial groups
    ]
    if not group_indices:
        continue

    group = [rewards[i] for i in group_indices]
    gmean = sum(group) / float(len(group))  # ← Divides by actual group size

    for idx, r in zip(group_indices, group):
        advantages[idx] = r - gmean
```

**Key insight:** The condition `if (prompt_idx + k * prompts_in_batch) < len(rewards)` allows groups to have **fewer than `rollouts_per_example` episodes**. Advantages are computed as `r - gmean` where `gmean = sum(group) / float(len(group))`, dynamically adjusting to actual group size.

**Batching:**

**File:** `verifiers/rl/trainer/orchestrator.py` (lines 316-359)
```python
# Convert to microbatches
for mb_idx in range(num_microbatches):
    start_idx = mb_idx * microbatch_size
    end_idx = min((mb_idx + 1) * microbatch_size, len(all_prompt_ids))

    microbatch = {
        "prompt_ids": all_prompt_ids[start_idx:end_idx],
        "completion_ids": all_completion_ids[start_idx:end_idx],
        "advantages": torch.tensor(advantages[start_idx:end_idx]),
        # ...
    }
    microbatches.append(microbatch)
```

**Variable sizes handled by slicing** - each microbatch can have different sizes if total episodes don't divide evenly.

**Padding in trainer:**

**File:** `verifiers/rl/trainer/trainer.py` (lines 171-189)
```python
def pad(self, batch: dict) -> dict:
    """Pad sequences to max length in batch"""
    prompt_ids = batch["prompt_ids"]
    completion_ids = batch["completion_ids"]

    # Find max lengths
    max_prompt_len = max(len(p) for p in prompt_ids)
    max_completion_len = max(len(c) for c in completion_ids)

    # Right-pad with pad_token_id
    padded_prompts = [p + [self.pad_token_id] * (max_prompt_len - len(p)) for p in prompt_ids]
    padded_completions = [c + [self.pad_token_id] * (max_completion_len - len(c)) for c in completion_ids]

    # ...
```

**Conclusion:** Verifiers explicitly handles variable group sizes and uses dynamic padding for variable-length sequences.

---

### Q2: Dataset Filtering vs Rollout Checking

**Answer: Rollout-level checking - budget enforced during generation**

**Code Evidence:**

**File:** `verifiers/envs/environment.py` (lines 964-998) - Truncation during rollout

```python
# Process each response
for idx, response in enumerate(state["responses"]):
    # ... extract prompt_ids, completion_ids ...

    # CHECK BUDGET (ROLLOUT-LEVEL)
    is_truncated = False
    if max_seq_len > 0 and len(prompt_ids) + len(completion_ids) > max_seq_len:
        # Truncate prompt if it alone exceeds budget
        if len(prompt_ids) > max_seq_len:
            prompt_ids = prompt_ids[:max_seq_len]
            prompt_mask = prompt_mask[:max_seq_len]

        # Truncate completion to fit remaining budget
        completion_ids = completion_ids[: max_seq_len - len(prompt_ids)]
        completion_mask = completion_mask[: max_seq_len - len(prompt_ids)]
        completion_logprobs = completion_logprobs[: max_seq_len - len(prompt_ids)]
        is_truncated = True

    # Apply masking/zeroing based on config
    if is_truncated and mask_truncated_completions:
        completion_mask = [0] * len(completion_ids)  # ← Masks all completion tokens

    # ... later ...
    if zero_truncated_completions and is_truncated:
        all_rewards.append(0)  # ← Sets reward to 0
        all_is_truncated.append(True)
    else:
        all_rewards.append(reward)
        all_is_truncated.append(False)
```

**No dataset-level filtering.** Budget is checked **during rollout** after each response is generated.

**Conclusion:** Verifiers does NOT filter at dataset level. It checks budget during rollout and hard-truncates sequences at `max_seq_len`.

---

### Q3: Train on Partial Tokens - What Does "Masked" Mean?

**Answer: By default, train on partial tokens. With config flags, mask or zero-reward truncated episodes.**

**Code Evidence:**

**Truncation logic (from Q2 above):**
- Hard-truncate at `max_seq_len`: `completion_ids = completion_ids[: max_seq_len - len(prompt_ids)]`
- This creates partial tokens (e.g., "STA" if "STAND" was truncated)

**Two configuration options:**

**File:** `verifiers/rl/trainer/config.py` (lines 118-129)
```python
@dataclass
class GRPOTrainerConfig:
    # ...
    mask_truncated_completions: bool = False
    # When True: Sets completion_mask = [0] * len(completion_ids)
    # Effect: Excludes truncated tokens from loss calculation

    zero_truncated_completions: bool = False
    # When True: Sets reward = 0 for truncated episodes
    # Effect: Episode trains with negative advantage (if other episodes have positive rewards)
```

**File:** `verifiers/envs/environment.py` (lines 983-994)
```python
if is_truncated and mask_truncated_completions:
    completion_mask = [0] * len(completion_ids)  # ← Zero mask for all tokens

# ... later ...
if zero_truncated_completions and is_truncated:
    all_rewards.append(0)  # ← Zero reward
    all_is_truncated.append(True)
else:
    all_rewards.append(reward)
    all_is_truncated.append(False)
```

**Behavior:**

| Setting | Partial tokens (e.g., "STA") in batch? | Gradient computed? | Reward |
|---------|----------------------------------------|--------------------|--------|
| Both `False` (default) | ✅ Yes | ✅ Yes - trains on "S", "T", "A" | Original reward |
| `mask_truncated_completions=True` | ✅ Yes | ❌ No - `completion_mask=0` | Original reward (but no gradient) |
| `zero_truncated_completions=True` | ✅ Yes | ✅ Yes | `reward=0` (negative advantage) |

**Documentation:**

**File:** `verifiers/docs/training.md` (lines 69-70)
```toml
mask_truncated_completions = false
zero_truncated_completions = true
```

Recommended config: keep masked tokens in batch, but zero their rewards.

**Conclusion:** By default, Verifiers **trains on partial tokens**. With config flags, it can mask (zero gradient) or zero-reward truncated episodes while keeping them in the batch.

---

### Q4: Reference Model Timing

**Answer: No separate reference model - uses vLLM sampling logprobs**

**Code Evidence:**

**File:** `verifiers/rl/trainer/orchestrator.py` (lines 221-228) - Generation with logprobs

```python
# Generate with vLLM (includes logprobs in response)
env_results = await self.env.a_generate(
    repeated_ds,
    client=self.client,
    model=self.model_name,
    sampling_args=self.sampling_args,  # ← Includes logprobs=True
    score_rollouts=True,
    max_concurrent=self.max_concurrent,
)
```

**File:** `verifiers/rl/trainer/config.py` (lines 307-324) - Sampling args config

```python
self.sampling_args = {
    "temperature": self.temperature,
    "top_p": self.top_p,
    "max_tokens": self.max_tokens or self.max_seq_len,
    "n": 1,
    "logprobs": True,  # ← Request logprobs during generation
    "extra_body": {
        "return_tokens_as_token_ids": True,
    },
}
```

**vLLM returns logprobs during generation**, which are stored in `state["responses"]` and used as "reference logprobs".

**Training with importance sampling:**

**File:** `verifiers/rl/trainer/trainer.py` (lines 241-262) - Loss computation

```python
def compute_loss(
    self,
    batch: dict,
    trainer_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,  # ← From vLLM generation
) -> tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss with importance sampling
    """
    advantages = batch["advantages"]
    completion_mask = batch["completion_mask"]

    # Importance ratio: current policy vs inference policy
    log_importance_ratio = trainer_logprobs - inference_logprobs
    # ^^^^ inference_logprobs are the "reference" (from sampling time)

    # GRPO loss (similar to PPO)
    # ...
```

**No separate reference model forward pass.** The "reference" is the policy at the time of sampling, whose logprobs are captured by vLLM.

**Exact flow:**
```
1. rollout (vLLM with logprobs=True) → captures inference_logprobs
2. score rollout → compute rewards
3. process_env_results_vllm → apply truncation masks/rewards
4. create microbatches (all episodes, including masked ones)
5. trainer.forward() → compute trainer_logprobs (current policy)
6. compute_loss(trainer_logprobs, inference_logprobs) → importance sampling
```

**Conclusion:** Verifiers does NOT have a separate reference model call. It uses vLLM's sampling logprobs as the reference for importance sampling.

---

### Verifiers Summary

| Question | Answer | Key Mechanism |
|----------|--------|---------------|
| **Q1: Variable groups** | ✅ Continue with fewer - dynamic advantage computation | `gmean = sum(group) / float(len(group))` |
| **Q2: Dataset filtering** | ❌ Rollout-level checking | Hard-truncate at `max_seq_len` during generation |
| **Q3: Train on partial** | ✅ Yes by default, mask/zero-reward if config enabled | `completion_mask=0` or `reward=0` for truncated |
| **Q4: Ref model timing** | N/A - no separate ref model | Uses vLLM sampling logprobs for importance sampling |

---

## Cross-Library Comparison

### Q1: Variable Group Sizes

| Library | Continue with Fewer? | Resample to Exact Size? | Filter at Dataset? | Batching Strategy |
|---------|---------------------|------------------------|--------------------|-------------------|
| **TRL** | ❌ No - assumes fixed | ❌ No | ❌ No | Fixed batch size, `.view(-1, num_gen)` breaks with variable |
| **VERL** | ✅ Yes | ❌ No | ❌ No | Variable batch size, sequence balancing by token count |
| **NeMo-RL** | ✅ Yes (standard) | ✅ Yes (dynamic mode) | ❌ No | Fixed batch size (via resampling or fixed repetition) |
| **Tinker** | ✅ Yes | ❌ No | ❌ No | Variable batch size, explicit "train on smaller batch" |
| **Verifiers** | ✅ Yes | ❌ No | ❌ No | Variable batch size, dynamic padding |

**Majority pattern:** Continue with fewer episodes (4/5 libraries)

**Exception:** TRL assumes fixed size and will crash with variable groups

---

### Q2: Dataset Filtering vs Rollout Checking

| Library | Dataset Filtering? | Rollout Checking? | When is Budget Checked? |
|---------|-------------------|-------------------|------------------------|
| **TRL** | ❌ No | ⚠️ Partial (post-generation) | After generation, checks if last token is EOS |
| **VERL** | ❌ No | ✅ Yes | Before each turn, checks `len(response_mask) >= response_length` |
| **NeMo-RL** | ❌ No | ✅ Yes | After each turn, truncates env observation to fit budget |
| **Tinker** | ❌ No | ✅ Yes | After adding tool results, checks `observation.length > max_trajectory_tokens` |
| **Verifiers** | ❌ No | ✅ Yes | During generation, hard-truncates at `max_seq_len` |

**Unanimous:** **No dataset filtering** - all libraries check budget during rollout

**Reasoning:** Prompts grow during multi-turn rollouts (tool results, game state), so initial prompt length doesn't predict final length

---

### Q3: Train on Partial Tokens

| Library | Generates Partial Tokens? | Default Behavior | Masking Option? | How Masking Works |
|---------|--------------------------|------------------|-----------------|-------------------|
| **TRL** | ✅ Yes (e.g., "STA") | Train on partial | ✅ `mask_truncated_completions` | `completion_mask=0` → zero gradient |
| **VERL** | ❌ No - clean termination | N/A | ❌ N/A | Terminates before generating partial tokens |
| **NeMo-RL** | ❌ No - truncates env response | Train on full generation | ✅ `overlong_filtering` | `loss_multiplier=0` → zero gradient |
| **Tinker** | ❌ No - episode ends | Penalty via `reward=0.0` | ❌ No | No masking, just low reward |
| **Verifiers** | ✅ Yes (hard-truncated) | Train on partial | ✅ `mask_truncated_completions` | `completion_mask=0` → zero gradient |

**Key insight:** "Masked" means **zero gradient** (via `completion_mask=0` or `loss_multiplier=0`), NOT excluded from batch

**Clarification for user's Q3:**
- **"Train on partial tokens by default"**: TRL and Verifiers generate "STA" and compute gradients on it
- **"All of them mask"**: Libraries that generate partial tokens offer CONFIG OPTIONS to zero gradients
- **Default vs optional**: Most libraries train on partial by default, but allow masking via config

---

### Q4: Reference Model Timing

| Library | Flow | Ref Model Called for All Episodes? | Ref Model Called for Dropped Episodes? |
|---------|------|-------------------------------------|----------------------------------------|
| **TRL** | rollout → mask → **ref** → buffer | ✅ Yes | ✅ Yes (mask only affects gradient) |
| **VERL** | rollout → **ref** → train | ✅ Yes | N/A (no dropping) |
| **NeMo-RL** | rollout → filter → **ref** → train | ❌ No - only kept | ❌ No - skips if `is_batch_complete=False` |
| **Tinker** | rollout → filter → **ref** → train | ❌ No - only kept | ❌ No - filtered return `None` |
| **Verifiers** | rollout (captures logprobs) → train | N/A - no separate ref model | N/A |

**Two patterns:**
1. **TRL/VERL**: Compute ref_model for ALL episodes, masking/filtering affects only gradients
2. **NeMo-RL/Tinker**: Filter first, compute ref_model only for kept episodes (more efficient)

---

## Discussion & Design Decisions

### User's Questions & Answers

---

#### **Q1: Variable Group Sizes - "I'm afraid of dynamic batch sizes for compile"**

**Answer: You can maintain fixed batch sizes for training while handling variable rollout sizes**

**Evidence:**
- **TRL**: Pads all sequences to fixed dimensions (`max_completion_length`), so training batch is always fixed shape
- **NeMo-RL**: Uses dynamic sampling to resample until exactly `num_prompts * num_generations` episodes, maintaining fixed training batch size
- **VERL/Tinker/Verifiers**: Use variable batch sizes, but rely on padding/masking for fixed tensor shapes

**Recommendation for blackjack:**

```python
# Option A: Pad to fixed size (like TRL)
async def continuous_rollouts(tokenizer, pad_id):
    GROUP_SIZE = cfg.group_size  # e.g., 16

    while not shutdown_event.is_set():
        episodes = []
        for game_idx in range(GROUP_SIZE):
            episode = await play_game(...)
            episodes.append(episode)

        # Filter invalid episodes
        valid_episodes = [
            e for e in episodes
            if not (e.is_truncated and not cfg.grpo.include_truncated_in_buffer)
        ]

        if len(valid_episodes) < GROUP_SIZE:
            # Pad with dummy episodes (zero loss_multiplier)
            dummy_episode = create_dummy_episode(pad_id)
            dummy_episode.loss_multiplier = 0  # No gradient
            while len(valid_episodes) < GROUP_SIZE:
                valid_episodes.append(dummy_episode)

        # Now valid_episodes is always exactly GROUP_SIZE
        # Compute ref_logprobs, advantages, etc.
```

**Or simpler: Just continue with fewer episodes (like Tinker)**

Most libraries handle variable sizes fine. Compilation works with dynamic shapes in modern PyTorch (2.0+).

---

#### **Q2: Dataset Filtering - "Should we filter prompts > max_seq_len at dataset level?"**

**Answer: No - all libraries check at rollout level**

**Reasoning:**
1. **Multi-turn growth**: Initial prompt might be 500 tokens, but after 3 tool calls it's 2000 tokens
2. **Wasted filtering**: If you filter at dataset level, you'd drop potentially valid prompts that happen to have long initial messages but few turns
3. **Uniform pattern**: ALL 5 libraries check budget during rollout, NONE filter at dataset

**For blackjack:**
- Initial prompt is small (~100 tokens for system message)
- Grows turn-by-turn with game state
- **Don't filter at dataset level**
- Check budget before each generation in `play_game()`

**However:** You can add a **sanity check** to warn if initial prompts are unreasonably large:

```python
# In play_game()
prompt_text = tokenizer.apply_chat_template(messages, ...)
prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

if len(prompt_tokens) >= max_seq_len:
    logger.warning(f"Initial prompt ({len(prompt_tokens)} tokens) exceeds max_seq_len ({max_seq_len})")
    record_metric("episode/initial_prompt_too_large", 1, Reduce.MEAN)
    # Return truncated episode (don't crash)
    return Episode(is_truncated=True, truncation_reason="initial_prompt_exceeds_budget", ...)
```

---

#### **Q3: "They train on partial tokens but also mask. What's happening?"**

**Answer: "Masked" = zero gradient, NOT excluded from batch**

**Clarification:**

| Config | Partial Tokens in Batch? | Forward Pass Computed? | Gradient Computed? |
|--------|-------------------------|------------------------|-------------------|
| **Default** (no masking) | ✅ Yes ("STA") | ✅ Yes | ✅ Yes - trains on "STA" |
| **With masking** | ✅ Yes ("STA") | ✅ Yes | ❌ No - `completion_mask=0` zeros gradient |

**Example from TRL:**
```python
# Batch contains: ["STAND", "HIT", "STA"]  # "STA" is truncated
completion_mask = torch.tensor([[1,1,1,1,1], [1,1,1], [1,1,1]])  # Default: all 1s

if mask_truncated_completions:
    is_truncated = [False, False, True]
    completion_mask = completion_mask * (~is_truncated).unsqueeze(1)
    # Result: [[1,1,1,1,1], [1,1,1], [0,0,0]]  # "STA" tokens masked to 0

# Loss computation
masked_loss = per_token_loss * completion_mask
# "STA" tokens contribute zero to loss (but are still in batch)
```

**Summary:**
- **"Train on partial"** = partial tokens go through forward pass and loss computation
- **"Masked"** = their loss contribution is multiplied by 0 (no gradient)
- They still occupy space in the batch, still go through ref_model, etc.

---

#### **Q4: User's Proposed Flow - "Set reward, run ref_model, compute advantages, then decide buffer"**

**Answer: Two valid patterns - recommend Tinker/NeMo-RL (filter first, then ref_model)**

**User's proposed flow:**
```
rollout → set reward → ref_model → compute advantages → buffer decision
```

**This matches TRL/VERL** - compute ref_model for ALL episodes, including ones that might be dropped.

**Alternative (Tinker/NeMo-RL):**
```
rollout → set reward → filter → ref_model (only kept) → compute advantages → add to buffer
```

**Pros/cons:**

| Approach | Pros | Cons |
|----------|------|------|
| **Ref_model for all** (TRL/VERL) | Simpler code, no filtering logic | Wastes computation on episodes you'll drop |
| **Ref_model for kept** (Tinker/NeMo-RL) | More efficient (skip ref_model for dropped) | Slightly more complex (need to filter first) |

**Recommendation:** Use **filter-first approach** (Tinker/NeMo-RL) for efficiency:

```python
# In continuous_rollouts()
episodes = []
for game_idx in range(group_size):
    episode = await play_game(...)
    episodes.append(episode)

# Filter BEFORE ref_model
valid_episodes = [
    e for e in episodes
    if not e.is_truncated or cfg.grpo.include_truncated_in_buffer
]

if not valid_episodes:
    continue  # No valid episodes, skip entire rollout

# Compute ref_logprobs ONLY for valid episodes
# (pad to max_len, batch together)
max_len = max(len(e.all_token_ids) for e in valid_episodes)
padded_tokens = []
for episode in valid_episodes:
    seq_len = len(episode.all_token_ids)
    pad_len = max_len - seq_len
    padded = F.pad(episode.all_token_ids, (0, pad_len), value=pad_id)
    padded_tokens.append(padded)

input_ids = torch.stack(padded_tokens)
ref_logprobs = await ref_model.forward.route(input_ids, 0, return_logprobs=True)

# Unpad and assign
for i, episode in enumerate(valid_episodes):
    seq_len = len(episode.all_token_ids)
    episode.ref_logprobs = ref_logprobs[i, :seq_len]

# Compute advantages
advantages = await compute_advantages.compute.call_one(valid_episodes)
for episode, advantage in zip(valid_episodes, advantages):
    episode.advantage = advantage
    await replay_buffer.add.call_one(episode)
```

This skips ref_model for dropped episodes, saving computation.

---

## Blackjack Implementation

Based on the library investigation, here's the recommended implementation for blackjack.

---

### Configuration

**File:** `apps/blackjack/qwen3_1_7b.yaml`

```yaml
blackjack_env:
  server_url: "http://localhost:8004"
  server_port: 8004
  game_name: "blackjack"
  model: "Qwen/Qwen3-1.7B"
  max_seq_len: 2048              # Episode-level budget (all turns)
  max_turns: 10                  # Hard limit on turns per episode

grpo:
  group_size: 16                 # Number of games per group
  include_truncated_in_buffer: false  # Drop truncated episodes (configurable)

policy:
  engine_args:
    enable_prefix_caching: true  # Critical for multi-turn (2-3x speedup)
    max_model_len: 4096          # vLLM model context limit
```

---

### Episode Class

**File:** `apps/blackjack/episode.py` (new file)

```python
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Episode:
    """
    Episode data for GRPO training with multi-turn support.

    For blackjack (multi-turn game, single episode):
        - all_token_ids: [prompt1, resp1, prompt2, resp2, ...]
        - response_mask: [0, 0, ..., 1, 1, ..., 0, 0, ..., 1, 1, ...]
                         [  prompt1  ][  resp1  ][  prompt2  ][  resp2  ]
        - reward: Final game outcome (win/loss/push)

    One episode = one complete game with all turns.
    """

    # ============ Core Identifiers ============
    episode_id: str
    task_name: str | None = None  # e.g., "blackjack"

    # ============ Policy Version (for replay buffer eviction) ============
    generator_version: int = 0
    is_truncated: bool = False  # Hit max_seq_len or max_turns
    truncation_reason: str | None = None  # "max_seq_len", "initial_prompt_exceeds_budget", "max_turns"

    # ============ Token Data ============
    all_token_ids: torch.Tensor  # Shape: (seq_len,)
    logprobs: torch.Tensor       # Shape: (seq_len,)
    response_mask: torch.Tensor  # Shape: (seq_len,)
                                 # 1.0 = train on this token (response)
                                 # 0.0 = skip this token (prompt)

    # ============ Rewards & Training ============
    reward: float | None = None
    advantage: float | None = None
    ref_logprobs: torch.Tensor | None = None  # Shape: (seq_len,)

    # ============ Metadata ============
    metadata: dict[str, Any] = field(default_factory=dict)
    # Suggested fields:
    #   - num_turns: int
    #   - game_id: str
    #   - env_reward: float (raw from environment)

    # ============ Optional Debugging ============
    message_log: list[dict[str, Any]] | None = None
    # OpenAI-compatible messages for debugging/analysis


# Type alias for GRPO groups
Group = list[Episode]
```

---

### Unified Action Parser

**File:** `apps/blackjack/main.py`

```python
def parse_action(response_text: str) -> str:
    """
    Parse action from model's text response.

    Returns:
        "HIT", "STAND", or "INVALID"

    Note:
        INVALID actions default to STAND in play_game().
    """
    text_lower = response_text.lower().strip()

    if text_lower.endswith("hit"):
        return "HIT"
    elif text_lower.endswith("stand"):
        return "STAND"
    else:
        return "INVALID"
```

---

### Reward Calculation

**File:** `apps/blackjack/main.py`

```python
def calculate_reward(env_reward: float) -> float:
    """
    Reward structure:
        - Win: +3
        - Else: -1

    Args:
        env_reward: Raw environment reward (+1 win, 0 push, -1 loss)

    Returns:
        Final shaped reward for training
    """
    if env_reward > 0:  # Win
        return 3.0
    else:  # Loss or push
        return -1.0
```

---

### Multi-Turn Game Rollout

**File:** `apps/blackjack/main.py`

```python
async def play_game(
    game_idx: int,
    game_id: str,
    server_url: str,
    policy: Generator,
    tokenizer,
    pad_id: int,
    max_seq_len: int = 2048,
    max_turns: int = 10,
    rollout_count: int = 0,
) -> Episode:
    """
    Play a single blackjack game and return ONE episode with all turns.

    Key changes from single-turn:
    - Formats messages each turn (not once at start)
    - Tracks episode-level budget (max_seq_len)
    - Returns single Episode with concatenated tokens
    - Includes response_mask for training

    Returns:
        Episode with all turns concatenated
    """
    env = OpenSpielEnv(base_url=server_url)
    env._http.trust_env = False

    print(f"\n🎮 GAME {game_idx + 1} (Rollout #{rollout_count + 1}) - ID: {game_id}")

    # Initialize message history
    messages = [
        {
            "role": "system",
            "content": "You are an expert BlackJack player. Analyze the game state and output only 'HIT' or 'STAND'.",
        }
    ]

    # Track all tokens and masks across all turns
    all_tokens = []
    all_logprobs = []
    response_mask = []

    # Track for truncation
    is_truncated = False
    truncation_reason = None

    try:
        result = env.reset()
        obs = result.observation
        done = False
        turn_num = 0

        while not done and turn_num < max_turns:
            # Add user message with current game state
            player_total = obs.metadata.get("player_total", "?")
            dealer_card = obs.metadata.get("dealer_card", "?")
            dealer_str = "Ace" if dealer_card == 1 else str(dealer_card)

            state_desc = f"=== BlackJack Game (Turn {turn_num + 1}) ===\n\n"
            state_desc += "Current State:\n"
            state_desc += f"  Your hand total: {player_total}\n"
            state_desc += f"  Dealer shows: {dealer_str}\n"
            state_desc += f"  Legal actions: HIT, STAND\n\n"
            state_desc += "What do you do? Output only 'HIT' or 'STAND'."

            messages.append({"role": "user", "content": state_desc})

            # Format prompt from full message history
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # Encode to check budget (ROLLOUT-LEVEL CHECK, following all libraries)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Check if prompt exceeds budget (like VERL/Tinker/NeMo-RL)
            if len(prompt_tokens) >= max_seq_len:
                is_truncated = True
                truncation_reason = "max_seq_len"
                record_metric("episode/terminated_budget_exceeded", 1, Reduce.MEAN)
                print(f"  [TRUNCATED] Prompt length {len(prompt_tokens)} >= {max_seq_len}")
                break

            # Calculate remaining budget for this turn
            remaining = max_seq_len - len(prompt_tokens)

            # Safety check (like NeMo-RL)
            if remaining <= 0:
                is_truncated = True
                truncation_reason = "zero_budget"
                record_metric("episode/terminated_zero_budget", 1, Reduce.MEAN)
                break

            # Generate with remaining budget
            try:
                responses = await asyncio.wait_for(
                    policy.generate.route(
                        [prompt_text], sampling_params={"max_tokens": remaining}
                    ),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                print(f"[ERROR] Policy generation timed out for {game_id} at turn {turn_num}")
                raise

            response = responses[0]

            # Check if generation was cut off (like TRL/Verifiers)
            if response.stop_reason == "length":
                is_truncated = True
                truncation_reason = "generation_length"
                record_metric("episode/generation_truncated", 1, Reduce.MEAN)
                print(f"  [TRUNCATED] Generation hit max_tokens={remaining}")
                # Note: We continue to parse and execute, but mark episode as truncated
                # This follows VERL's pattern (but VERL terminates cleanly, we don't generate partial)

            # Accumulate tokens and build response mask
            all_tokens.extend(prompt_tokens)
            all_tokens.extend(response.token_ids)
            response_mask.extend([0] * len(prompt_tokens))  # Don't train on prompts
            response_mask.extend([1] * len(response.token_ids))  # Train on responses
            all_logprobs.extend([0.0] * len(prompt_tokens))
            all_logprobs.extend(response.logprobs)

            # Parse action
            action_name = parse_action(response.text)

            # Add assistant response to message history
            messages.append({"role": "assistant", "content": response.text})

            if action_name == "INVALID":
                action_name = "STAND"  # Fallback
                action_id = 1
            elif action_name == "HIT":
                action_id = 0
            elif action_name == "STAND":
                action_id = 1

            # Execute action
            result = env.step(OpenSpielAction(action_id=action_id, game_name="blackjack"))
            obs = result.observation
            done = result.done

            turn_num += 1

        # Check if hit max_turns
        if turn_num >= max_turns and not done:
            is_truncated = True
            truncation_reason = "max_turns"
            record_metric("episode/hit_max_turns", 1, Reduce.MEAN)

        # Get final game outcome
        final_game_reward = result.reward

        outcome_text = (
            "WIN" if final_game_reward > 0 else ("LOSS" if final_game_reward < 0 else "PUSH")
        )
        print(f"  Result: {outcome_text} (reward={final_game_reward}, turns={turn_num})")

        # Calculate final reward
        reward = calculate_reward(env_reward=final_game_reward)

        # Metrics
        record_metric("reward/env_reward", final_game_reward, Reduce.MEAN)
        record_metric("reward/final_reward", reward, Reduce.MEAN)
        record_metric("game/total_games_played", 1, Reduce.SUM)
        record_metric("game/average_game_length_in_turns", turn_num, Reduce.MEAN)
        record_metric("game/win_rate", 1 if final_game_reward > 0 else 0, Reduce.MEAN)

        # Create episode
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            task_name="blackjack",
            generator_version=0,  # TODO: Get from policy
            is_truncated=is_truncated,
            truncation_reason=truncation_reason,
            all_token_ids=torch.tensor(all_tokens, dtype=torch.long),
            logprobs=torch.tensor(all_logprobs, dtype=torch.float),
            response_mask=torch.tensor(response_mask, dtype=torch.float),
            reward=reward,
            advantage=None,  # Computed later
            ref_logprobs=None,  # Computed later
            message_log=messages,
            metadata={
                "num_turns": turn_num,
                "game_id": game_id,
                "env_reward": final_game_reward,
            },
        )

        return episode

    except Exception as e:
        print(f"[ERROR] play_game {game_id} failed with {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
```

---

### Continuous Rollouts

**File:** `apps/blackjack/main.py`

Following **Tinker/NeMo-RL pattern** - filter first, then compute ref_model only for kept episodes.

```python
async def continuous_rollouts(tokenizer, pad_id):
    """
    Continuous rollout loop following Tinker/NeMo-RL pattern:
    1. Generate episodes
    2. Filter invalid/truncated (if config)
    3. Compute ref_logprobs ONLY for kept episodes
    4. Compute advantages
    5. Add to buffer
    """
    rollout_count = 0
    server_url = cfg.blackjack_env.get("server_url", "http://localhost:8004")
    max_seq_len = cfg.blackjack_env.get("max_seq_len", 2048)
    max_turns = cfg.blackjack_env.get("max_turns", 10)
    group_size = cfg.grpo.get("group_size", 16)
    include_truncated = cfg.grpo.get("include_truncated_in_buffer", False)

    while not shutdown_event.is_set():
        t = Tracer("main_perf/continuous_rollouts")
        t.start()

        # Step 1: Generate group_size games
        episodes = []
        for game_idx in range(group_size):
            game_id = str(uuid.uuid4())[:8]
            episode = await play_game(
                game_idx=game_idx,
                game_id=game_id,
                server_url=server_url,
                policy=policy,
                tokenizer=tokenizer,
                pad_id=pad_id,
                max_seq_len=max_seq_len,
                max_turns=max_turns,
                rollout_count=rollout_count,
            )
            episodes.append(episode)

        t.step("play_games")

        # Metrics
        record_metric("rollout/episodes_generated", len(episodes), Reduce.SUM)

        # Step 2: Filter BEFORE ref_model (Tinker/NeMo-RL approach - more efficient)
        valid_episodes = [
            e for e in episodes if not e.is_truncated or include_truncated
        ]

        if not valid_episodes:
            print(f"[WARNING] No valid episodes in rollout {rollout_count}, skipping")
            record_metric("rollout/rollouts_with_no_valid_episodes", 1, Reduce.SUM)
            rollout_count += 1
            continue

        record_metric("rollout/episodes_kept", len(valid_episodes), Reduce.SUM)
        record_metric("rollout/episodes_dropped", len(episodes) - len(valid_episodes), Reduce.SUM)

        # Step 3: Compute ref_logprobs ONLY for valid episodes
        # Pad episodes to same length for batching
        max_len = max(len(e.all_token_ids) for e in valid_episodes)
        padded_tokens = []
        for episode in valid_episodes:
            seq_len = len(episode.all_token_ids)
            pad_len = max_len - seq_len
            padded = F.pad(episode.all_token_ids, (0, pad_len), value=pad_id)
            padded_tokens.append(padded)

        input_ids = torch.stack(padded_tokens)  # [num_valid_episodes, max_len]

        # Get reference logprobs
        ref_logprobs = await ref_model.forward.route(
            input_ids, 0, return_logprobs=True  # 0 = no separate prompt (mask handles it)
        )
        t.step("reference_model_calculate_logprobs")

        # Assign ref_logprobs to episodes (unpad)
        for i, episode in enumerate(valid_episodes):
            seq_len = len(episode.all_token_ids)
            episode.ref_logprobs = ref_logprobs[i, :seq_len]  # Unpad

        del ref_logprobs, input_ids

        # Step 4: Compute advantages
        advantages = await compute_advantages.compute.call_one(valid_episodes)
        t.step("compute_advantages")

        # Step 5: Add to buffer
        for episode, advantage in zip(valid_episodes, advantages):
            episode.advantage = advantage
            await replay_buffer.add.call_one(episode)

        rollout_count += 1
        record_metric("main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM)
        t.stop()
```

---

### Collate Function

**File:** `apps/blackjack/main.py`

```python
def collate(batches: list[Group]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates episodes into batches with dynamic padding.

    Each episode has variable length (different number of turns).
    Handles variable-length episodes like VERL/Tinker/Verifiers.
    """
    inputs = []
    targets = []

    for batch in batches:
        # Find max length in this batch
        max_len = max(len(e.all_token_ids) for e in batch)
        pad_id = 0  # Will be set via F.pad value parameter

        all_token_ids = []
        logprobs_list = []
        ref_logprobs_list = []
        advantages_list = []
        masks = []

        for e in batch:
            seq_len = len(e.all_token_ids)
            pad_len = max_len - seq_len

            # Right-pad tokens
            padded_tokens = F.pad(e.all_token_ids, (0, pad_len), value=pad_id)
            all_token_ids.append(padded_tokens)

            # Right-pad response_mask (0 for padding)
            padded_mask = F.pad(e.response_mask, (0, pad_len), value=0)
            masks.append(padded_mask)

            # Pad logprobs
            padded_logprobs = F.pad(e.logprobs, (0, pad_len), value=0)
            logprobs_list.append(padded_logprobs)

            # Pad ref_logprobs
            padded_ref = F.pad(e.ref_logprobs, (0, pad_len), value=0)
            ref_logprobs_list.append(padded_ref)

            advantages_list.append(e.advantage)

        input = {"tokens": torch.stack(all_token_ids)}
        target = {
            "response": torch.stack(all_token_ids),  # Full sequence
            "ref_logprobs": torch.stack(ref_logprobs_list),
            "advantages": torch.tensor(advantages_list).unsqueeze(-1),
            "padding_mask": torch.stack(masks),  # Combined response + padding mask
        }

        inputs.append(input)
        targets.append(target)

    return inputs, targets
```

---

### Main Setup

**File:** `apps/blackjack/main.py`

```python
async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    group_size = cfg.grpo.group_size
    max_req_tokens = cfg.max_req_tokens  # Deprecated, but keep for compatibility
    max_res_tokens = cfg.max_res_tokens  # Deprecated, but keep for compatibility

    # ---- Start OpenSpiel Server ---- #
    # ... (same as before) ...

    # ---- Global setups ---- #
    # ... (same as before) ...

    # ---- Setup services ---- #
    (
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
    ) = await asyncio.gather(
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(**cfg.trainer, loss=simple_grpo_loss),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(**cfg.replay_buffer, collate=collate),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
    )

    # Get tokenizer for rollout loop (following VERL/NeMo-RL/Tinker pattern)
    from vllm.transformers_utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(cfg.blackjack_env.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    print("All services initialized successfully!")

    # ... (rest of main setup) ...

    # ---- Core RL loops ---- #
    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)

    print(f"Starting GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads")

    rollout_tasks = [
        asyncio.create_task(continuous_rollouts(tokenizer, pad_id))
        for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # ... (shutdown logic same as before) ...
```

---

## Summary & Recommendations

### Key Findings

1. **Variable group sizes**:
   - **Majority (4/5)** continue with fewer episodes
   - **TRL** breaks with variable sizes (assumes fixed)
   - **Recommendation**: Continue with fewer (like Tinker), or pad to fixed size if needed for compile

2. **Dataset filtering**:
   - **ALL libraries** check budget at rollout level, NOT dataset level
   - **Recommendation**: Check budget during `play_game()`, don't filter at dataset

3. **Train on partial tokens**:
   - **"Masked" = zero gradient**, not excluded from batch
   - Libraries either generate partial tokens (TRL/Verifiers) or terminate cleanly (VERL/NeMo-RL/Tinker)
   - **Recommendation**: Follow VERL/Tinker - terminate before generating partial tokens

4. **Reference model timing**:
   - **TRL/VERL**: Compute for all episodes
   - **NeMo-RL/Tinker**: Filter first, compute only for kept episodes (more efficient)
   - **Recommendation**: Follow Tinker/NeMo-RL - filter first, then ref_model

### Implementation Checklist

- [x] New Episode class with `all_token_ids`, `response_mask`, `logprobs`
- [x] Unified `parse_action()` function
- [x] Separate `calculate_reward()` function
- [x] Multi-turn `play_game()` with budget tracking
- [x] `continuous_rollouts()` with filter-first pattern
- [x] Variable-length `collate()` function
- [x] Config parameters: `max_seq_len`, `max_turns`, `include_truncated_in_buffer`
- [ ] Remove old Episode class from main.py
- [ ] Remove `BlackJackReward` actor
- [ ] Remove `EnvironmentActor` class
- [ ] Test with single game
- [ ] Test with group_size > 1
- [ ] Monitor truncation metrics

---

**End of Document**
