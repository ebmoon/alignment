from typing import TYPE_CHECKING, Iterable

import torch


if TYPE_CHECKING:
    pass

class MonitorState:
    """Abstract base class for all monitor states"""

class Monitor:
    """Abstract base class for all monitors that can be applied during monitor-guided generation."""

    def filter_vocab(self, input_ids: torch.LongTensor) -> Iterable[torch.LongTensor]:
        """
        Filter out next tokens for the current input that do not pass the monitor.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor`s of shape `(num_accepted_tokens)` containing indices of
            acceptable next tokens for each batch.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `filter_vocab`."
        )

    def update(self, next_tokens: torch.LongTensor) -> Iterable[MonitorState]:
        """
        Update the state of the monitor based on the selected next tokens.

        Args:
            next_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of selected next tokens in the vocabulary.

        Return:
            `MonitorState` after updating the state.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `update`."
        )

    def reset(self):
        """
        Reset the monitor state to the initial state
        """