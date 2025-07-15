# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset

from forge.protocols.tokenizer import TokenizerProtocol
from forge.rl.environments.base import Transform
from forge.rl.environments.chat import (
    ChatAction,
    ChatEnvironment,
    ChatObservation,
    ChatState,
    Message,
)
from torch.utils.data import DataLoader


@dataclass
class DatasetChatState(ChatState):
    """State for DatasetChatEnvironment that includes metadata.

    Extends ChatState to include metadata from the dataset.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnMapping:
    """Configuration for mapping dataset columns to chat roles or metadata.

    This class defines how to extract content from dataset columns and either:
    1. Assign to chat roles (for conversation messages)
    2. Store as metadata (for reference data like expected answers)
    """

    column: str
    role: Optional[str] = None  # None means store as metadata
    content_template: Optional[str] = None
    metadata_key: Optional[str] = None  # Key to store in metadata dict

    def __post_init__(self):
        """Validate that either role or metadata_key is provided."""
        if self.role is None and self.metadata_key is None:
            raise ValueError("Either 'role' or 'metadata_key' must be provided")
        if self.role is not None and self.metadata_key is not None:
            raise ValueError("Cannot specify both 'role' and 'metadata_key'")

        # If metadata_key not specified but role is None, use column name
        if self.role is None and self.metadata_key is None:
            self.metadata_key = self.column

    @property
    def is_metadata(self) -> bool:
        """Check if this mapping is for metadata storage."""
        return self.role is None

    def extract_content(self, row: Dict[str, Any]) -> str:
        """Extract content from a dataset row.

        Args:
            row: A single row from the dataset

        Returns:
            The extracted content as a string
        """
        content = row[self.column]
        if self.content_template:
            # Allow template formatting with row data
            return self.content_template.format(**row)
        return str(content)


class DatasetChatEnvironment(ChatEnvironment):
    """A ChatEnvironment that loads prompts from HuggingFace datasets.

    This environment follows the standard RL environment API:
    - reset() loads a new episode (conversation) from the dataset
    - step() adds a new turn to the current conversation

    Most datasets will be single-turn (reset() loads a prompt, done), but this
    design supports multi-turn conversations like TauBench where step() can
    extend the conversation.

    Example usage for single-turn datasets:
        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset_name="gsm8k",
            column_mappings=[
                ColumnMapping(column="question", role="user")
            ]
        )

        obs = env.reset()  # Loads new question from dataset
        # obs.messages = [{"role": "user", "content": "What is 2+2?"}]

        # Agent generates response
        action = ChatAction(role="assistant", content="4")
        obs = env.step(action)  # Adds assistant response

    Example usage for multi-turn datasets:
        env = DatasetChatEnvironment(
            tokenizer=tokenizer,
            dataset_name="taubench",
            column_mappings=[
                ColumnMapping(column="user_message", role="user"),
                ColumnMapping(column="assistant_message", role="assistant")
            ]
        )

        obs = env.reset()  # Loads initial conversation from dataset
        # Continue conversation with additional turns via step()
    """

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        dataset_name: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        column_mappings: Optional[List[ColumnMapping]] = None,
        system_prompt: Optional[str] = None,
        system_role: str = "system",
        split: str = "train",
        shuffle: bool = True,
        seed: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        transform: Optional[Transform] = None,
        **dataset_kwargs,
    ):
        """Initialize DatasetChatEnvironment.

        Args:
            tokenizer: Tokenizer for processing messages
            dataset_name: Name of HuggingFace dataset to load (e.g., "gsm8k")
            dataset: Pre-loaded dataset (alternative to dataset_name)
            column_mappings: List of ColumnMapping objects defining role assignments
            system_prompt: Optional system prompt
            system_role: Role for system messages
            split: Dataset split to use ("train", "test", "validation")
            shuffle: Whether to shuffle the dataset
            seed: Random seed for reproducibility
            filter_fn: Optional function to filter dataset rows
            transform: Optional transform function to apply to observations
            **dataset_kwargs: Additional arguments for load_dataset
        """
        # Initialize with a custom state class
        super().__init__(tokenizer, system_prompt, system_role, transform)

        # Replace the default state with our custom state
        self._state = DatasetChatState(
            history_messages=self._state.history_messages,
            history_tokens=self._state.history_tokens,
        )

        # Load dataset
        if dataset is not None:
            self.dataset = dataset
        elif dataset_name is not None:
            self.dataset = load_dataset(dataset_name, split=split, **dataset_kwargs)
        else:
            raise ValueError("Either dataset_name or dataset must be provided")

        # Apply filter if provided
        if filter_fn:
            self.dataset = self.dataset.filter(filter_fn)

        # Set up column mappings
        self.column_mappings = column_mappings or []
        if not self.column_mappings:
            raise ValueError(
                "column_mappings must be provided to define role assignments"
            )

        # Validate column names against dataset
        dataset_columns = set(self.dataset.column_names)
        for mapping in self.column_mappings:
            if mapping.column not in dataset_columns:
                raise ValueError(
                    f"Column '{mapping.column}' not found in dataset. Available columns: {', '.join(dataset_columns)}"
                )

        # Set up random seed for reproducibility
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.seed = seed

        # Set shuffle parameter
        self.shuffle = shuffle

        # Initialize dataloader and iterator
        self.reset_dataloader(seed)

    def _identity_collate_fn(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple collate function that returns the batch as-is."""
        return batch

    def _construct_conversation(self, row: Dict[str, Any]) -> List[Message]:
        """Construct a conversation from a dataset row using column mappings.

        Args:
            row: A single row from the dataset

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        conversation = []

        # Add messages based on column mappings (skip metadata mappings)
        for mapping in self.column_mappings:
            if not mapping.is_metadata:
                content = mapping.extract_content(row)
                if content.strip():  # Only add non-empty content
                    # Store the original content directly in the message
                    # This avoids the tokenization/detokenization process that loses role information
                    conversation.append({"role": mapping.role, "content": content})

        return conversation

    def _extract_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a dataset row using column mappings.

        Args:
            row: A single row from the dataset

        Returns:
            Dictionary of metadata key-value pairs
        """
        metadata = {}

        # Extract metadata based on column mappings
        for mapping in self.column_mappings:
            if mapping.is_metadata:
                content = mapping.extract_content(row)
                metadata_key = mapping.metadata_key or mapping.column
                metadata[metadata_key] = content

        return metadata

    def reset(self) -> ChatObservation:
        """Reset the environment by loading a new episode (conversation) from the dataset.

        This loads a new prompt/conversation from the dataset. For single-turn datasets,
        this typically loads just a user prompt. For multi-turn datasets, this may load
        an entire conversation.

        Returns:
            ChatObservation: Initial observation with the new conversation loaded
        """
        # Get the next row from the dataset
        row = self._get_next_row()

        # Reset the base environment state (clears history)
        super().reset()

        # Extract metadata and store in state
        self._state.metadata = self._extract_metadata(row)

        # Construct conversation from row
        conversation = self._construct_conversation(row)

        # Add all messages from the dataset to the conversation
        for message in conversation:
            action = self.message_to_action(message)
            super().step(action)

        # Return observation (which already includes metadata via _create_observation)
        return self._create_observation()

    def _get_next_row(self) -> Dict[str, Any]:
        """Get the next row from the dataset, handling dataset exhaustion.

        Returns:
            A single row from the dataset

        Note:
            If the dataset is exhausted, the dataloader is reset with the same seed
            to maintain reproducibility, unless shuffle=False in which case it's
            just reset to the beginning.
        """
        try:
            # Get next batch (always size 1)
            batch = next(self._data_iterator)
            return batch[0]
        except StopIteration:
            # If we've reached the end of the dataset, reset the dataloader
            # Use the same seed if not shuffling, or a new seed if shuffling
            new_seed = random.randint(0, 2**32 - 1) if self.shuffle else self.seed
            self.reset_dataloader(new_seed)

            try:
                batch = next(self._data_iterator)
                return batch[0]
            except StopIteration:
                # This should only happen if the dataset is empty
                raise ValueError("Dataset is empty - cannot load any examples")

    def step(self, action: ChatAction) -> ChatObservation:
        """Take a step by adding a new turn to the current conversation.

        This extends the current conversation loaded by reset() with a new message.
        Useful for multi-turn scenarios or when the agent needs to respond to
        the dataset prompt.

        Args:
            action: A ChatAction containing tokens to add

        Returns:
            ChatObservation: Updated observation with the new message added
        """
        # Use the parent class step method to add the action to the conversation
        super().step(action)

        # Create observation
        observation = self._create_observation()

        # Apply transform and return (with type casting)
        return self._apply_transform(observation)  # type: ignore

    def _create_observation(self) -> ChatObservation:
        """Override _create_observation to include metadata from state.

        Returns:
            ChatObservation: Observation with messages, tokens, and metadata
        """
        # Get the base observation from parent class
        observation = super()._create_observation()

        # Add metadata from state
        observation.metadata = self._state.metadata.copy()

        return observation

    def reset_dataloader(self, seed: Optional[int] = None):
        """Reset the dataloader to start from the beginning with optional reshuffling.

        This recreates the dataloader with either the original seed or a new one.

        Args:
            seed: Optional new random seed. If None, uses the original seed.

        Returns:
            self: The environment itself for method chaining
        """
        # Update seed if a new one is provided
        if seed is not None:
            self.seed = seed

        # Reset the generator with the current seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # Recreate the dataloader with the updated generator
        self.dataloader = DataLoader(
            self.dataset,  # type: ignore
            batch_size=1,  # Always batch size 1 for episode loading
            shuffle=self.shuffle,
            generator=self.generator if self.shuffle else None,
            collate_fn=self._identity_collate_fn,
        )

        # Initialize the iterator
        self._data_iterator = iter(self.dataloader)

        return self
