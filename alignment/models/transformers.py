from typing import TYPE_CHECKING, List, Union, Tuple

from .tokenizer import Tokenizer

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer


class TransformerTokenizer(Tokenizer):
    """Tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: "PreTrainedTokenizer", **kwargs):
        super(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            vocabulary=tokenizer.get_vocab(),
        )

        self.tokenizer = tokenizer

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
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

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
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text
    
class Transformers:
    """A class for `transformers` models."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)
