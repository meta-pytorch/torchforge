from typing import Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class Episode:
    # TODO: add adtional layer for multi-turn
    episode_id: str
    request: str
    policy_version: int
    pad_id: int
    request_len: int
    response_len: int
    target: Optional[Any] = None
    # processed data
    response: Optional[str] = None
    request_tokens: Optional[list[int]] = None
    response_tokens: Optional[list[int]] = None
    ref_logprobs: Optional[torch.Tensor] = None
    reward: Optional[float] = None
    advantage: Optional[float] = None

    @property
    def request_tensor(self):
        tensor = torch.tensor(self.request_tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self):
        tensor = torch.tensor(self.response_tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor
