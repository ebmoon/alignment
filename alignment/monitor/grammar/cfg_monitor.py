from typing import TYPE_CHECKING, Dict, Iterable, Self

import torch
from transformers import PreTrainedTokenizer

from lark import Lark

from .partial_lexer import PartialLexerFST
from .partial_parser import TokenParsingTable
from ..monitor import Monitor, MonitorState

if TYPE_CHECKING:
    pass

class CFGMonitorState(MonitorState):
    """
    A state of CFG monitor that contains both lexer and parser state.
    """
    
    def __init__(
            self,
            lexer_state, stack,
            lexer: PartialLexerFST,
            parse_table: TokenParsingTable
        ):
        self.lexer_state = lexer_state
        self.stack = stack
        self.lexer = lexer
        self.parse_table = parse_table
        
        self._acceptance = None
        self._state_cache = {}

    @property
    def acceptance(self):
        """For caching of acceptance table"""
        if self._acceptance is None:
            self._acceptance = self.parse_table.acceptance(self.lexer_state, self.stack)
        return self._acceptance

    def feed_token(self, token_id: int) -> Self:
        """
        Feed token to the current state and return a new monitor state after transition.
        """
        if token_id not in self._state_cache:
            acceptance = self.acceptance
            lexer_state, stack = acceptance[token_id]
            monitor_state = CFGMonitorState(lexer_state, stack, self.lexer, self.parse_table)
            self._state_cache[token_id] = monitor_state

        return self._state_cache[token_id]

class CFGMonitor(Monitor):
    """A monitor to guide LLM to generate output in a CFG"""
    
    def __init__(
            self,
            grammar_str: str,
            vocabulary: Dict[str, int],
            eos_token_id: int,
            num_batch: int = 1
        ):
        self.grammar_str = grammar_str
        self.vocabulary = vocabulary
        self.eos_token_id = eos_token_id
        self.num_batch = num_batch

        self.initial_state = None
        self.state = None
        self._initialize_state()

    @classmethod
    def from_tokenizer(
            cls,
            grammar_str: str,
            tokenizer: PreTrainedTokenizer,
            num_batch: int = 1
        ) -> Self:
        """Build CFG monitor from grammar and tokenizer"""
        vocabulary = tokenizer.get_vocab()
        eos_token_id = tokenizer.eos_token_id

        return CFGMonitor(grammar_str, vocabulary, eos_token_id, num_batch)

    def _initialize_state(self):
        lark_parser = Lark(self.grammar_str, parser='lalr')
        terminal_parse_table = lark_parser.parser.parser._parse_table
        start = lark_parser.parser.parser_conf.start[0]

        lexer = PartialLexerFST(lark_parser.lexer_conf, self.vocabulary, self.eos_token_id)
        lexer_state = lexer.initial
        stack = [terminal_parse_table.start_states[start]]
        parse_table = TokenParsingTable.from_terminal_parse_table(
            lexer, terminal_parse_table, self.eos_token_id, start)

        self.initial_state = CFGMonitorState(lexer_state, stack, lexer, parse_table)
        self.state = [self.initial_state for _ in range(self.num_batch)]

    def filter_vocab(self, input_ids: torch.LongTensor) -> Iterable[torch.LongTensor]:
        """
        Filter out next tokens for the current input that do not pass the monitor.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, num_accepted_tokens)` containing indices of
            acceptable next tokens for each batch.
        """

        assert input_ids.shape[0] == self.num_batch
        acceptances = [torch.LongTensor(list(s.acceptance.keys())) for s in self.state]

        return acceptances


    def update(self, next_tokens: torch.LongTensor) -> Iterable[MonitorState]:
        """
        Update the state of the monitor based on the selected next tokens.

        Args:
            next_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of selected next tokens in the vocabulary.

        Return:
            `MonitorState`s after updating the state.
        """

        assert next_tokens.shape[0] == self.num_batch
        self.state = [s.feed_token(next_tokens[i].item()) for i, s in enumerate(self.state)]

        return self.state

    def reset(self):
        self.state = [self.initial_state for _ in range(self.num_batch)]

