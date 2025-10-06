# Data Models

Base data models for common RL workflows.

## Completion

Outputs from vLLM. The {class}`forge.data_models.completion.Completion` represents a model-generated completion for a given prompt.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.data_models.completion.Completion
```

## Prompt

Input prompts for models. The {class}`forge.data_models.prompt.Prompt` encapsulates prompt data for language models.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.data_models.prompt.Prompt
```

## Episode

Training episodes for RL. The {class}`forge.data_models.episode.Episode` represents a complete interaction episode in reinforcement learning.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.data_models.episode.Episode
```

## ScoredCompletion

Completions with associated scores. The {class}`forge.data_models.scored_completion.ScoredCompletion` extends completions with scoring information.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   forge.data_models.scored_completion.ScoredCompletion
```
