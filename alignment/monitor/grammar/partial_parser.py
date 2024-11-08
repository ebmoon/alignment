from typing import TYPE_CHECKING, Sequence, Set

if TYPE_CHECKING:
    pass


class ParserToken:
    """A base class for parser token"""

class PartialParser:
    """
    A base class for partial parser,
    that parses a sequence of lexed tokens and predicts available next tokens.
    """

    def parse(self, tokens: Sequence[ParserToken]) -> Set[ParserToken]:
        """
        Parse a sequence of lexed tokens and computes acceptable next tokens

        Args:
            tokens (`Sequence[ParserToken]): a sequence of lexed tokens

        Return:
            `Set[ParserToken]`: a set of acceptable next tokens
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `parse`."
        )

