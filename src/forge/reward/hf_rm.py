from __future__ import annotations
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from forge.reward.rm_models import (
    GRMModel, SkyworksModel, URMModel, QRMModel, GPMModel,
    GRMLlama32Model, OffsetBiasModel, GRMGemmaModel, ArmorRMModel,
    QwenPRMModel, Qwen72BModel, EurusPRMStage1Model, EurusPRMStage2Model,
    INFORMModel, SkyworksGemmaModel,  QRMGemmaModel, LDLRewardGemmaModel,
    InternLM2RewardModel, InternLM2Reward7BModel, DecisionTreeRewardModel8B, 
    DecisionTreeRewardModel27B, Qwen72BPRMModel
)


class HFRewardModel:
    """
    Minimal RM wrapper. Returns a scalar reward per sample.
    - If logits dim=1, uses that as score.
    - If logits dim=2, uses the last logit as "good" score.
    """
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
        template: str = "{prompt}\n\n{response}",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="right", truncation_side="left")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.device = device
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.template = template
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self._needs_resize = True
        else:
            self._needs_resize = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def __call__(self, prompts: List[str], responses: List[str], targets: Optional[List[str]] = None) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts, 
            responses,
            truncation=True,
            max_length=4096,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        out = self.model(**inputs)
        logits = out.logits
        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits).item()
        else:
            if logits.shape[-1] == 2:
                if logits[0][0] > logits[0][1]:
                    scores = 0.0
                else:
                    scores = 1.0
            else:
                scores = logits[..., -1]  # assume last logit corresponds to "positive/good"
        return scores
