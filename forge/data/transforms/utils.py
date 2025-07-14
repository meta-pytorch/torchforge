from typing import Any, Optional, Union

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
    "tool",  # Origin is return from a tool call
]


class Message:
    """
    This class represents individual messages in a fine-tuning dataset. It supports
    text-only content, text with interleaved images, and tool calls. The
    :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` will tokenize
    the content of the message using ``tokenize_messages`` and attach the appropriate
    special tokens based on the flags set in this class.

    Args:
        role (Role): role of the message writer. Can be "system" for system prompts,
            "user" for human prompts, "assistant" for model responses, or "ipython"
            for tool call returns.
        content (Union[str, list[dict[str, Any]]]): content of the message. If it is text only content,
            you can pass in a string. If it is multimodal content, pass in a list of dictionaries formatted
            as follows::

                [
                    {"type": "image", "content": torch.Tensor},
                    {"type": "text", "content": "What is in this image?"},
                ]

        masked (bool): whether the message is masked in the sample. If True, do not use
            in loss calculation. Default: False
        ipython (bool): whether the message is a tool call. Default: False
        eot (bool): whether the message corresponds to the end of a turn, where control is handed over
            to the assistant from the user or the user from the assistant. Default: True. Should be true
            in most cases except for:

            - For multiple consecutive assistant messages (i.e., tool calls
              by assistant), only the last assistant message will have ``eot=True``
            - All ipython messages (tool call returns) should set ``eot=False``.

    Note:
        Message class expects any image content to be a ``torch.Tensor``, as output
        by e.g. :func:`~torchtune.data.load_image`
    """

    def __init__(
        self,
        role: Role,
        content: Union[str, list[dict[str, Any]]],
        masked: bool = False,
        ipython: bool = False,
        eot: bool = True,
    ):
        self.role = role
        self.content = self._convert_to_list_of_dict(content)
        self.masked = masked
        self.ipython = ipython
        self.eot = eot

    def _convert_to_list_of_dict(self, content) -> list[dict[str, Any]]:
        """User is currently allowed to pass in a string for text-only content.
        This ensures that the content is formatted as a list of dictionaries."""
        if isinstance(content, str):
            return [{"type": "text", "content": content}]

        assert isinstance(
            content, list
        ), f"content must be of type list[dict[str, Any]], got {content}"

        return content

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """
        Construct a Message from a dictionary.

        Args:
            d (dict): dictionary containing the fields of the Message.

        Returns:
            Message: constructed Message.
        """
        return cls(
            role=d["role"],
            content=d["content"],
            masked=d.get("masked", False),
            ipython=d.get("ipython", False),
            eot=d.get("eot", True),
        )

    def __repr__(self) -> str:
        content_only = [content["content"] for content in self.content]
        return f"Message(role='{self.role}', content={content_only!r})"


def truncate(
    tokens: list[Any],
    max_seq_len: int,
    eos_id: Optional[Any] = None,
    truncation_type: str = "right",
) -> list[Any]:
    """
    Truncate a list of tokens to a maximum length. If eos_id is provided, the last
    token will be replaced with eos_id.

    Args:
        tokens (list[Any]): list of tokens to truncate
        max_seq_len (int): maximum length of the list
        eos_id (Optional[Any]): token to replace the last token with. If None, the
            last token will not be replaced. Default is None.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Returns:
        list[Any]: truncated list of tokens

    Raises:
        ValueError: if truncation_type is not "left" or "right"
    """

    if truncation_type == "left":
        tokens_truncated = tokens[-max_seq_len:]  # Take the last max_seq_len tokens
    elif truncation_type == "right":
        tokens_truncated = tokens[:max_seq_len]  # Take the first max_seq_len tokens
    else:
        raise ValueError(
            f"truncation_type must be 'left' or 'right', got {truncation_type}"
        )

    # Replace the last token with eos_id if necessary
    if eos_id is not None and tokens_truncated and tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id

    return tokens_truncated
