from typing import Iterable, Tuple

import torch

class MonitorState:
    """Abstract base class for all monitor states"""

class Monitor:
    """Abstract base class for all monitors that can be applied during monitor-guided generation."""

    def filter_vocab(self, input_ids: torch.LongTensor):
        """
        Filter out next tokens for the current input that do not pass the monitor.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            A mask that indicates allowed tokens. The specific type and format may vary by implementation.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `filter_vocab`."
        )
    
    def get_tokens_from_mask(self, mask, input_ids: torch.LongTensor):
        """
        Convert a mask of valid tokens to a list of token IDs for each batch.
        
        Args:
            mask: The mask returned from filter_vocab that indicates allowed tokens.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                
        Return:
            `Iterable[torch.LongTensor]` containing indices of acceptable next tokens for each batch.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_tokens_from_mask`."
        )

    def mask_logits(self, logits: torch.Tensor, mask) -> None:
        """
        Apply the monitor's constraints directly to logits in-place.
        
        This is more efficient than filtering the vocabulary and then masking the logits separately,
        especially for large vocabulary sizes.

        Args:
            logits (`torch.Tensor` of shape `(batch_size, vocab_size)`):
                Logits to be masked in-place. Invalid tokens will be set to -inf.
            mask: The mask returned from filter_vocab that indicates allowed tokens.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `mask_logits`."
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
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `reset`."
        )