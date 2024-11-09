from typing import TYPE_CHECKING, Iterator, Set

if TYPE_CHECKING:
    from lark.lexer import Token

class PartialParser:
    """
    A base class for partial parser,
    that parses a sequence of lexed tokens and predicts available next tokens.
    """

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

