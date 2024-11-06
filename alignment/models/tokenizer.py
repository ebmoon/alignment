from typing import TYPE_CHECKING, Dict, List, Union, Tuple

if TYPE_CHECKING:
    import torch


class Tokenizer:
    """
    An abstract class for all the tokenizers used by alignment models.
    """

    def __init__(
        self, eos_token_id: int, pad_token_id: int, vocabulary: Dict[str, int]
    ):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocabulary = vocabulary

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple["torch.LongTensor", "torch.LongTensor"]:
        """
        Tokenize input prompts into a pair of token ids and attention mask.

        Args:
            prompt (`str` or `List[str]]`):
                A string or a list of strings to be encoded.

        Returns:
            `(torch.LongTensor, torch.LongTensor)`: A pair of token ids
                and attention mask.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `encode`."
        )

    def decode(self, token_ids: "torch.LongTensor", **kwargs) -> List[str]:
        """
        Converts sequences of token ids into strings.

        Args:
            token_ids (`torch.LongTensor`):
                List of tokenized input ids.
                `torch.LongTensor` of shape `(batch_size, sequence_length)`.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `decode`."
        )
