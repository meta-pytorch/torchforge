# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Protocol, Union

import torch


class TokenizerProtocol(Protocol):
    """Protocol for tokenizers used throughout Forge.

    This protocol defines the interface for tokenizers that support Hugging Face's
    apply_chat_template method with full signature support. This enables proper
    handling of turn boundaries, generation prompts, and message continuation.

    Key parameters for chat applications:
    - add_generation_prompt: Adds tokens that indicate the start of a response
    - continue_final_message: Removes EOS tokens to continue the final message
    - return_assistant_tokens_mask: Returns a mask for assistant-generated tokens
    """

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Apply chat template to messages with full Hugging Face signature.

        Args:
            conversation: A list of dicts with "role" and "content" keys, representing
                the chat history so far.
            tools: A list of tools (callable functions) that will be accessible to the model.
            documents: A list of dicts representing documents for RAG applications.
            chat_template: A Jinja template to use for this conversion.
            add_generation_prompt: If True, appends tokens that indicate the start of
                an assistant message. Useful for generating responses.
            continue_final_message: If True, formats the chat so the final message is
                open-ended without EOS tokens. Cannot be used with add_generation_prompt.
            tokenize: Whether to tokenize the output. If False, returns a string.
            padding: Strategy to pad returned sequences.
            truncation: Whether to truncate sequences at maximum length.
            max_length: Maximum length for padding or truncation.
            return_tensors: Framework for returned tensors ('pt', 'tf', 'np', 'jax').
            return_dict: Whether to return a dictionary with named outputs.
            return_assistant_tokens_mask: Whether to return a mask of assistant tokens.
            tokenizer_kwargs: Additional kwargs to pass to the tokenizer.
            **kwargs: Additional kwargs to pass to the template renderer.

        Returns:
            A list of token ids or a dict of tokenizer outputs, depending on parameters.
        """
        ...

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs
    ) -> str:
        """Decode token ids to a string.

        Args:
            token_ids: List of tokenized input ids. Can be obtained using the __call__ method.
            skip_special_tokens: Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces: Whether or not to clean up the tokenization spaces.
                If None, will default to self.clean_up_tokenization_spaces.
            kwargs: Additional keyword arguments to pass to the underlying model specific decode method.

        Returns:
            The decoded string.
        """
        ...
