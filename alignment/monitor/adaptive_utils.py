from typing import TYPE_CHECKING, Any, Dict, Optional, Self

import numpy as np
import torch
import json

if TYPE_CHECKING:
    pass


class AdaptiveMaskTrieNode:
    """
    Trie node containing approximated probability of successive generation.
    """

    def __init__(
        self,
        raw_likelihood: float,
        success_rate: float = 1,
        token_id: Optional[int] = None,
        is_end_of_sequence: bool = False,
    ):
        self.parent = None
        self.children = {}
        self.raw_likelihood = raw_likelihood
        self.success_rate = success_rate
        self.token_id = token_id
        self.is_end_of_sequence = is_end_of_sequence

    def _insert(self, token_id: int, child_node: Self):
        """
        Insert child node for the token id, update the node if a node already exists.

        Args:
            token_id (`torch.long`):
                Index of the token to be inserted in the vocabulary.
            child_node (`AdaptiveSampleTrieNode`):
                The child node containing raw likelihood and approximated success rate of the token.
        """
        self.children[token_id] = child_node
        child_node.parent = self

    def update(
        self, acceptance: torch.LongTensor, scores: torch.FloatTensor, eos_token_id: int
    ):
        """
        Update children from the list of accepted tokens and their scores.

        Args:
            acceptance (`torch.LongTensor` of shape `(batch_size, num_accepted_tokens)`):
                Indices of acceptable next tokens in the vocabulary.
            scores (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            eos_token_id (`torch.long`):
                The index of EOS token in the vocabulary.
        """
        likelihoods = torch.nn.functional.softmax(scores, dim=-1)

        for batch_index in range(acceptance.shape(0)):
            accepted_tokens = acceptance[batch_index].nonzero().squeeze(-1)

            for token_id in accepted_tokens:
                if token_id not in self.children:
                    token_id = token_id.item()
                    raw_likelihood = likelihoods[batch_index, token_id].item()
                    is_end_of_sequence = token_id == eos_token_id

                    child_node = AdaptiveMaskTrieNode(
                        raw_likelihood=raw_likelihood,
                        token_id=token_id,
                        is_end_of_sequence=is_end_of_sequence,
                    )

                    self._insert(token_id, child_node)

    def get_mask(self, batch_size: int, vocab_size: int) -> torch.FloatTensor:
        """
        Construct logit mask from approximated success rates of children.

        Args:
            batch_size (`int`):
                The size of batch.
            vocab_size (`int`):
                The number of tokens in the vocabulary.

        Return:
            `torch.FloatTensor` of shape `(batch_size, vocab_size)
            containing the log of approximated success rate for acceptable next tokens,
            and minus infinity for invalid next tokens.
        """

        mask = torch.ones([batch_size, vocab_size], dtype=torch.float)
        mask = -float("inf") * mask

        for token_id in self.children:
            mask[:, token_id] = np.log(self.children[token_id].success_rate)

        return mask

    def _update_success_rate(self):
        """
        Re-compute the success rate from the updated success rate of children
        """
        total_success_rate = sum(
            child.raw_likelihood * child.success_rate
            for child in self.children.values()
        )
        self.success_rate = total_success_rate

        # Back propagate the success rate
        if self.parent:
            self.parent.update_success_rate()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert a trie into a dictionary by removing the pointer to the parent.

        Return:
            `Dict[str, Any]` containing all informations about members but parent.
        """
        return {
            "raw_likelihood": self.raw_likelihood,
            "success_rate": self.success_rate,
            "token_id": self.token_id,
            "is_end_of_sequence": self.is_end_of_sequence,
            "children": [child.to_dict() for child in self.children.values()],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Self:
        """
        Recursively (re)construct trie from dictionary.

        Args:
            d (`Dict[str, Any]`):
                Dictionary containing information about the node.

        Return:
            `AdaptiveSampleTrieNode` constructed from the dictionary.
        """
        node = AdaptiveMaskTrieNode(
            raw_likelihood=d["raw_likelihood"],
            success_rate=d["success_rate"],
            token_id=d["token_id"],
            is_end_of_sequence=d["is_end_of_sequence"],
        )

        node.children = {
            child["token_id"]: AdaptiveMaskTrieNode.from_dict(child)
            for child in node.children
        }
        for child in node.children.values():
            child.parent = node

        return node


class AdaptiveMaskTrie:
    """
    Trie for adaptive masking in checker-guided generation.
    """

    def __init__(self):
        self.root = AdaptiveMaskTrieNode(raw_likelihood=1)

    def json(self) -> str:
        """
        Dump adaptive mask trie into a JSON string.

        Return:
            `str` a JSON string dump of the whole adaptive mask trie
        """

        return json.dumps(self.root.to_dict(), indent=2)

    @staticmethod
    def loads(js: str) -> Self:
        """
        Load adaptive mask trie from a JSON string.

        Args:
            js (`str`): a JSON string dump of the whole adaptive mask trie.

        Return:
            `AdaptiveMaskTrie` constructed from the JSON string.
        """
        trie = AdaptiveMaskTrie()
        trie.root = AdaptiveMaskTrieNode.from_dict(json.loads(js))

        return trie
