from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional, Tuple

from lark.exceptions import UnexpectedCharacters
from lark.lexer import (
    BasicLexer,
    Lexer,
    LexerState
)

if TYPE_CHECKING:
    from lark.lexer import LexerConf, Token

class PartialLexerState(LexerState):
    """Represents the current state of the partial lexer"""

    def __init__(self, text: str, last_token: Optional["Token"]=None):
        super().__init__(text=text, line_ctr=None, last_token=last_token)

    def remainder(self) -> str:
        """
        Compute unlexed suffix of the text
        
        Return:
            `str`: unlexed suffix of the text
        """
        return self.text[self.line_ctr.char_pos:]

class PartialLexer(Lexer):
    """
    A base class for partial lexer,
    that splits a prefix into a sequence of lexed tokens and unlexed suffix.
    """

    @abstractmethod
    def partial_lex(
        self, lexer_state: PartialLexerState, parser_state: Any
    ) -> Tuple[Iterator["Token"], str]:
        """
        Split a prefix into a sequence of lexed tokens and unlexed suffix

        Args:
            lexer_state (`PartialLexerState`): the current state of partial lexer
            parser_state (`Any`): an optional parser state for contextual lexer

        Returns:
            `Iterator[ParserToken]`: a pair of sequence of lexed tokens
            `str`: unlexed suffix
        """

class PartialBasicLexer(BasicLexer, PartialLexer):
    """A non-contextual partial lexer"""

    def __init__(self, conf: "LexerConf"):
        super().__init__(conf)

    def partial_lex(
        self, lexer_state: PartialLexerState, parser_state: Any = None
    ) -> Tuple[Iterator["Token"], str]:
        """
        Split a prefix into a sequence of lexed tokens and unlexed suffix

        Args:
            lexer_state (`PartialLexerState`): the current state of partial lexer
            parser_state (`Any`): an optional parser state (not used, for inheritance)

        Return:
            `Iterator[ParserToken]`: a pair of sequence of lexed tokens
            `str`: unlexed suffix
        """
        lexer_tokens = []
        lexing_incomplete = False
        try:
            while lexer_state.line_ctr.char_pos < len(lexer_state.text):
                token = self.next_token(lexer_state)
                lexer_tokens.append(token)
        except UnexpectedCharacters:
            lexing_incomplete = True
        except EOFError:
            pass

        if lexing_incomplete:
            return lexer_tokens, lexer_state.remainder()
        else:
            remainder = lexer_tokens[-1] if len(lexer_tokens) > 0 else ""
            return lexer_tokens[:-1], remainder
