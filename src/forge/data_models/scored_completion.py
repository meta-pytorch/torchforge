from dataclasses import dataclass

from forge.data_models.completion import Completion


@dataclass
class ScoredCompletion:
    """A completion with an associated score (from a reward model or human)."""

    completion: Completion
    score: float  # akin to reward

    # TODO: add more fields as needed.
