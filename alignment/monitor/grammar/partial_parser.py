from typing import TYPE_CHECKING, Iterator, Set

from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser_state import ParserState

from .partial_lexer import PartialLexerThread

if TYPE_CHECKING:
    from lark.lexer import Token

class PartialParser(InteractiveParser):
    """
    A base class for partial parser,
    that parses a sequence of lexed tokens and predicts available next tokens.
    """

    def __init__(self, parser, parser_state: ParserState, lexer_thread: PartialLexerThread):
        super().__init__(parser, parser_state, lexer_thread)

    def parse(self, tokens: Iterator["Token"]) -> Set["Token"]:
        """
        Parse a sequence of lexed tokens and computes acceptable next tokens

        Args:
            tokens (`Iterator[Token]): a sequence of lexed tokens

        Return:
            `Set[Token]`: a set of acceptable next tokens
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `parse`."
        )
