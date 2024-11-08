from typing import TYPE_CHECKING, Sequence, Tuple

from .partial_parser import ParserToken

if TYPE_CHECKING:
    pass

class PartialLexer:
    """
    A base class for partial lexer,
    that splits a prefix into a sequence of lexed tokens and unlexed suffix.
    """

    def lex(self, prefix: str) -> Tuple[Sequence[ParserToken], str]:
        """
        Split a prefix into a sequence of lexed tokens and unlexed suffix

        Args:
            prefix (`str`): an unlexed string

        Return:
            `Sequence[ParserToken]`: a pair of sequence of lexed tokens
            `str`: unlexed suffix
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `lex`."
        )