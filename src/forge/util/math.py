import torch


def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    context_length = logits.shape[1] - input_ids.shape[1]
    logits = logits[:, context_length - 1 : -1]
    logprobs = torch.log_softmax(logits / temperature, dim=-1).to(input_ids.device)
    logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    return logprobs
