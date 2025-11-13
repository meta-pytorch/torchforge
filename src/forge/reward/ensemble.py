from __future__ import annotations
from typing import Protocol, List, Literal, Optional
import torch

class RewardFn(Protocol):
    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        targets: Optional[List[str]] = None,
    ) -> torch.Tensor: ...

class EnsembleReward:
    """
    Wraps multiple reward functions and reduces their scores.
    Assumes each fn returns a 1D tensor [batch].
    """
    def __init__(
        self,
        fns: List[RewardFn],
        reduce: Literal["mean", "median", "max", "vote"] = "mean",
        eps: float = 1e-5,
    ):
        self.fns = fns
        self.reduce = reduce
        self.eps = eps

    @torch.inference_mode()
    def __call__(self, prompts, responses, targets=None) -> torch.Tensor:
        scores = []
        for fn in self.fns:
            s = fn(prompts, responses, targets)
            if not isinstance(s, torch.Tensor):
                s = torch.as_tensor(s, dtype=torch.float32)
            scores.append(s.float().cpu())  # keep device-agnostic; trainer can move later

        stacked = torch.stack(scores, dim=0)  # [n_models, batch]

        if self.reduce == "mean":
            # print("mean score is ", stacked.mean(0))
            return stacked.mean(0)
        if self.reduce == "median":
            # print("median score is ", stacked.median(0).values)
            return stacked.median(0).values
        if self.reduce == "max":
            # print("max score is ", stacked.max(0).values)
            return stacked.max(0).values
        if self.reduce == "vote":
            # print("vote score is ", (stacked > 0.0).float().mean(0))
            # Interpret >0 as "good"; vote => fraction of positives in [0,1]
            return (stacked > 0.0).float().mean(0)
        raise ValueError(f"Unknown reduce: {self.reduce}")