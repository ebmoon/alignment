from typing import TYPE_CHECKING

import torch

from ..monitor import Monitor, MonitorState

if TYPE_CHECKING:
    pass

class CFGMonitor(Monitor):
    """A monitor to guide LLM to generate output in a CFG"""
    
    def __init__(self, grammar_str: str):
        self.grammar_str = grammar_str

    def filter_vocab(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Filter out next tokens for the current input that do not pass the monitor.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, num_accepted_tokens)` containing indices of
            acceptable next tokens for each batch.
        """
        raise NotImplementedError


    def update(self, next_tokens: torch.LongTensor) -> MonitorState:
        """
        Update the state of the monitor based on the selected next tokens.

        Args:
            next_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of selected next tokens in the vocabulary.

        Return:
            `MonitorState` after updating the state.
        """
        raise NotImplementedError
