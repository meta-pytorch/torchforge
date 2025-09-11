# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import vllm.envs as envs
from forge.data_models.api import Generator

# from monarch._src.tensor_engine.rdma import RDMABuffer
from forge.data_models.completion import Completion
from forge.data_models.prompt import Message, Prompt
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import (
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.inputs import TokensPrompt
from vllm.v1.engine.llm_engine import LLMEngine

envs.VLLM_USE_V1 = True


class VLLMGenerator(Generator):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.eos_token_id = self._tokenizer.eos_token_id
        self.llm_engine: LLMEngine = self._init_vllm_engine()

    def _init_vllm_engine(self) -> LLMEngine:
        """Initialize the vLLM engine using the v1 API."""
        cfg = VllmConfig(
            # TODO: Most of the hardcoded values needs to come form config
            model_config=ModelConfig(
                model=self.model_path,
                trust_remote_code=True,
                max_model_len=10240,
                dtype="auto",
                quantization=None,
                revision=None,
                enforce_eager=False,
                gpu_memory_utilization=0.3,
                swap_space=4.0,
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                distributed_executor_backend="external_launcher",
                disable_custom_all_reduce=True,
            ),
            scheduler_config=SchedulerConfig(
                scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler",
                max_num_batched_tokens=10240,
                max_num_seqs=8192,
                enable_chunked_prefill=False,
            ),
            observability_config=ObservabilityConfig(),
        )
        return LLMEngine.from_vllm_config(cfg)

    def generate(
        self,
        prompt: Prompt,
    ) -> list[Completion]:
        """
        Generate completions for a given prompt using vLLM.
        """
        prompt_ids = self._encode_prompt(prompt.messages)
        sampling_params = self._get_sampling_params()
        request_id = str(0)
        self.llm_engine.add_request(
            request_id,
            TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params,
        )
        return self._collect_completions(
            request_id, prompt, prompt_ids=torch.tensor(prompt_ids, dtype=torch.long)
        )

    def update_weights(
        self, weights_buffer: dict[str, tuple[torch.Tensor, torch.dtype, torch.Size]]
    ):
        # EPIC TODO: this assumes the model is NOT sharded.  We need to handle sharded models.
        # TODO 1: assumes the model uses FSDP
        # TODO 2: what about named_buffers()?
        # TODO 3: RDMABuffer integration
        self.llm_engine.reset_prefix_cache()
        # Access the model from the vLLM engine
        _model = self.llm_engine.model_executor.driver_worker.model_runner.model

        for name, param in _model.named_parameters():
            if "qkv_proj" in name:
                # VLLM may concat QKV into one tensor, even though Q, K, V are separate tensors
                # on the sender side.
                # Handle both qkv_proj.weight and qkv_proj.bias
                q_key = name.replace("qkv_proj", "q_proj")
                k_key = name.replace("qkv_proj", "k_proj")
                v_key = name.replace("qkv_proj", "v_proj")

                tensors = []
                q_param, q_dtype, q_shape = weights_buffer[q_key]
                k_param, k_dtype, k_shape = weights_buffer[k_key]
                v_param, v_dtype, v_shape = weights_buffer[v_key]

                tensors.append(q_param.view(q_dtype).reshape(q_shape))
                tensors.append(k_param.view(k_dtype).reshape(k_shape))
                tensors.append(v_param.view(v_dtype).reshape(v_shape))
                tensor_to_copy = torch.cat(tensors, dim=0)
            elif "gate_up_proj" in name:
                # VLLM may concat gate_proj and up_proj into one tensor, even though gate_proj and
                # up_proj are separate tensors on the sender side.
                gate_key = name.replace("gate_up_proj", "gate_proj")
                up_key = name.replace("gate_up_proj", "up_proj")

                tensors = []
                gate_param, gate_dtype, gate_shape = weights_buffer[gate_key]
                up_param, up_dtype, up_shape = weights_buffer[up_key]

                tensors.append(gate_param.view(gate_dtype).reshape(gate_shape))
                tensors.append(up_param.view(up_dtype).reshape(up_shape))
                tensor_to_copy = torch.cat(tensors, dim=0)
            else:
                t_param, t_dtype, t_shape = weights_buffer[name]
                tensor_to_copy = t_param.view(t_dtype).reshape(t_shape)

            param.data.copy_(tensor_to_copy)

        # TODO: Handle buffers
        # for name, buffer in _model.named_buffers():
        #     tensor_to_copy, _, _ = weights_buffer[name]
        #     buffer.data.copy_(tensor_to_copy)

    def _encode_prompt(self, messages: Sequence[Message]) -> list[int]:
        """
        Encode messages into token IDs and a mask indicating trainable tokens.
        """
        text_and_trainable = self.format_messages(
            messages, append_assistant_header=True
        )
        prompt_ids = []
        for text, _ in text_and_trainable:
            chunk_ids = self._tokenizer([text], add_special_tokens=False).input_ids[0]
            prompt_ids.extend(chunk_ids)
        return prompt_ids

    def _get_sampling_params(self) -> SamplingParams:
        """
        Return default sampling parameters.
        """
        return SamplingParams(
            n=4,
            temperature=1.0,
            max_tokens=16,
            logprobs=1,
            stop_token_ids=[self.eos_token_id],
        )

    def _collect_completions(
        self, request_id: str, prompt: Prompt, prompt_ids: torch.Tensor
    ) -> list[Completion]:
        """
        Collect completions from the LLM engine.
        """
        completions = []
        for _ in range(100):  # TODO: remove hardcoding
            if not self.llm_engine.has_unfinished_requests():
                break
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                for one_sample in output.outputs:
                    if one_sample.finish_reason or one_sample.stop_reason:
                        log_probs = self._extract_logprobs(one_sample)
                        completions.append(
                            Completion(
                                prompt=prompt,
                                text=self._tokenizer.decode(
                                    list(one_sample.token_ids),
                                    skip_special_tokens=True,
                                ),
                                prompt_ids=prompt_ids,
                                token_ids=torch.tensor(one_sample.token_ids),
                                log_probs=log_probs,
                            )
                        )
        return completions

    def _extract_logprobs(self, one_sample) -> torch.Tensor | None:
        """
        Extract log probabilities from a sample, if available.
        """
        if one_sample.logprobs is not None:
            # TODO: support returning more than 1 logprob
            return torch.tensor(
                [
                    top_k_dict[token].logprob
                    for token, top_k_dict in zip(
                        one_sample.token_ids, one_sample.logprobs
                    )
                ]
            )
        return None

    def format_message(self, message: Message) -> Sequence[tuple[str, bool]]:
        """Format a message to a list of string along with its trainable field."""
        role = message.role.name
        result = []
        result.append((f"<|im_start|>{role}\n", False))
        for chunk in message.chunks:
            if chunk is None:
                continue
            result.append((chunk, False))  # is_trainable is false
        last_chunk_trainable = result[-1][1]
        # The end token is trainable if the last chunk is trainable.
        result.append(("<|im_end|>\n", last_chunk_trainable))
        return result

    def format_messages(
        self, messages: Sequence[Message], append_assistant_header: bool
    ) -> Sequence[tuple[str, bool]]:
        result = []
        for message in messages:
            result.extend(self.format_message(message))
        if append_assistant_header:
            result.append(("<|im_start|>assistant\n", False))
        return result
