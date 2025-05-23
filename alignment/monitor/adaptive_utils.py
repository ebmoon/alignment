from typing import Any, Dict, Iterable, Optional, Self

import json
import numpy as np
import torch
from statistics import geometric_mean
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaptiveMaskState:
    """Abstract base class for adaptive mask states"""

class AdaptiveMask:
    """Abstract base class for adaptive mask monitor"""

    def mask(self, batch_size: int, vocab_size: int) -> torch.FloatTensor:
        """
        Construct logit mask from approximated success rates of children.

        Args:
            batch_size (`int`): The size of batch.
            vocab_size (`int`): The number of tokens in the vocabulary.

        Return:
            `torch.FloatTensor` of shape `(batch_size, vocab_size)
            containing the log of approximated success rate for acceptable next tokens,
            and minus infinity for invalid next tokens.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `mask`."
        )

    def reset(self):
        """
        Reset the adaptive mask state to the initial state
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `reset`."
        )

    def update_scores(
        self,
        acceptance: Iterable[torch.LongTensor],
        scores: torch.FloatTensor,
        eos_token_id: int,
        states: Optional[Iterable[AdaptiveMaskState]] = None
    ):
        """
        Update children from the list of accepted tokens and their scores.

        Args:
            acceptance (`torch.LongTensor`s of shape `(num_accepted_tokens)`):
                Indices of acceptable next tokens in the vocabulary.
            scores (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            eos_token_id (`torch.long`):
                The index of EOS token in the vocabulary.
            states (`Iterable[AdaptiveMaskState]`):
                Optional states to update scores. If not provided, apply update to the current states.

        Returns:
            `AdaptiveMaskState` after update
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `update`."
        )

    def feed_tokens(self, next_tokens: torch.LongTensor) -> Iterable[AdaptiveMaskState]:
        """
        Feed selected next tokens to update the current state of adaptive mask

        Args:
            next_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of selected next tokens in the vocabulary.

        Return:
            `AdaptiveMonitorState`s after updating the state.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `feed_tokens`."
        )

class AdaptiveMaskTrieNode(AdaptiveMaskState):
    """
    Trie node containing approximated probability of successive generation.
    """

    def __init__(
        self,
        raw_likelihood: float,
        success_rate: float = 1,
        token_id: Optional[int] = None,
        is_end_of_sequence: bool = False,
        is_updated: bool = False
    ):
        self.parent = None
        self.children = {}
        self.raw_likelihood = raw_likelihood
        self.success_rate = success_rate
        self.token_id = token_id
        self.is_end_of_sequence = is_end_of_sequence
        self.is_updated = is_updated

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
        self,
        acceptance: torch.LongTensor,
        scores: torch.FloatTensor,
        eos_token_id: int,
        delay_propagation: bool
    ):
        """
        Update children from the list of accepted tokens and their scores.

        Args:
            acceptance (`torch.LongTensor` of shape `(num_accepted_tokens)`):
                Indices of acceptable next tokens in the vocabulary.
            scores (`torch.FloatTensor` of shape `(vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            eos_token_id (`torch.long`):
                The index of EOS token in the vocabulary.
        """
        likelihoods = torch.nn.functional.softmax(scores, dim=-1)
        self.is_updated = True

        for token_id in acceptance:
            token_id = token_id.item()
            raw_likelihood = likelihoods[token_id].item()
            if token_id not in self.children:
                is_end_of_sequence = token_id == eos_token_id

                child_node = AdaptiveMaskTrieNode(
                    raw_likelihood=raw_likelihood,
                    token_id=token_id,
                    is_end_of_sequence=is_end_of_sequence,
                )

                self._insert(token_id, child_node)
            else:
                raw_likelihood = likelihoods[token_id].item()
                self.children[token_id].raw_likelihood = raw_likelihood
        
        if not delay_propagation:
            self.update_success_rate()

    def mask(self, vocab_size: int) -> torch.FloatTensor:
        """
        Construct logit mask from approximated success rates of children.

        Args:
            vocab_size (`int`):
                The number of tokens in the vocabulary.

        Return:
            `torch.FloatTensor` of shape `(vocab_size)
            containing the log of approximated success rate for acceptable next tokens,
            and minus infinity for invalid next tokens.
        """

        mask = torch.ones([vocab_size], dtype=torch.float)
        mask = -float("inf") * mask

        for token_id in self.children:
            # Ensure success_rate is positive before taking log
            mask[token_id] = np.log(self.children[token_id].success_rate)

        return mask

    def update_success_rate(self):
        """
        Re-compute the success rate from the updated success rate of children.
        For nodes that have never been explicitly updated, set their success rate
        to the average success rate of updated siblings under the same parent.
        """
        # Find the average success rate of updated children
        updated_children = [child for child in self.children.values() if child.is_updated]
        if updated_children:
            avg_success_rate = geometric_mean([child.success_rate for child in updated_children])
            bias = avg_success_rate * 1

            # Apply average success rate to never-updated children
            for child in self.children.values():
                if not child.is_updated:
                    child.success_rate = min(1, avg_success_rate + bias)
        
        # Calculate total success rate as before
        total_success_rate = sum(
            child.raw_likelihood * child.success_rate
            for child in self.children.values()
        )
        self.success_rate = max(1e-10, total_success_rate)

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
            "is_updated": self.is_updated,
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
            is_updated=d.get("is_updated", False),  # Handle older serialized data without this field
        )

        node.children = {
            child["token_id"]: AdaptiveMaskTrieNode.from_dict(child)
            for child in d["children"]
        }
        for child in node.children.values():
            child.parent = node

        return node


class AdaptiveMaskTrie(AdaptiveMask):
    """
    Trie for adaptive masking in checker-guided generation.
    """

    def __init__(self, batch_size: int = 1):
        self.root = AdaptiveMaskTrieNode(raw_likelihood=1)
        self.batch_size = batch_size
        self.states = [self.root for _ in range(batch_size)]

    def set_batch_size(self, batch_size: int):
        """
        Change batch size and reinitialize states
        
        Args:
            batch_size (`int`):
                The size of batch.
        """
        
        self.batch_size = batch_size
        self.reset()

    def mask(self, batch_size: int, vocab_size: int) -> torch.FloatTensor:
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
        if batch_size != self.batch_size:
            logger.warning(
                "The requested batch size %d is different to %d, resizing to %d.",
                batch_size, self.batch_size, batch_size
            )
            self.set_batch_size(batch_size)

        mask = torch.zeros([batch_size, vocab_size], dtype=torch.float)

        for i, state in enumerate(self.states):
            mask[i, :] = state.mask(vocab_size)

        return mask

    def update_scores(
        self,
        acceptance: Iterable[torch.LongTensor],
        scores: torch.FloatTensor,
        eos_token_id: int,
        states: Optional[Iterable[AdaptiveMaskTrieNode]] = None,
        delay_propagation: bool = False
    ) -> AdaptiveMaskState:
        """
        Update children from the list of accepted tokens and their scores.

        Args:
            acceptance (`torch.LongTensor`s of shape `(num_accepted_tokens)`):
                Indices of acceptable next tokens in the vocabulary.
            scores (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            eos_token_id (`torch.long`):
                The index of EOS token in the vocabulary.

        Returns:
            `AdaptiveMaskState` after update
        """

        if states is None:
            states = self.states

        for i, state in enumerate(states):
            state.update(acceptance[i], scores[i, :], eos_token_id, delay_propagation)

    def propagate_success_rate(self):
        """
        Re-compute the success rate from the updated success rate of children
        """
        for state in self.states:
            state.update_success_rate()
            
    def adjusted_probability(self, token_ids: torch.LongTensor) -> float:
        """
        Compute adjusted probability of a sentence as token-by-token product of 
        normalized raw_probability * success_rate over siblings.
        
        The formula for each step is: 
        P_adjusted(token | context) = (raw_prob(token) * success_rate(token)) / sum(raw_prob(sibling) * success_rate(sibling))
        
        Args:
            token_ids (`torch.LongTensor` of shape `(seq_length)`):
                Indices of tokens in the vocabulary representing the sentence.
                
        Return:
            `float`: The adjusted probability of the provided sentence.
        """
        # Start from root node
        current_node = self.root
        log_prob = 0.0
        
        for token_id in token_ids:
            token_id = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
            
            # Check if the token exists in current node's children
            if token_id not in current_node.children:
                return float('-inf')  # Impossible path
            
            # Get all siblings (including the token itself)
            siblings = current_node.children
            
            # Compute normalization factor: sum of (raw_prob * success_rate) for all siblings
            norm_factor = sum(
                sibling.raw_likelihood * sibling.success_rate
                for sibling in siblings.values()
            )
            
            # Get the current token's node
            token_node = siblings[token_id]
            
            # Compute adjusted probability for this step: (raw_prob * success_rate) / norm_factor
            if norm_factor > 0:
                step_prob = (token_node.raw_likelihood * token_node.success_rate) / norm_factor
            else:
                step_prob = 0.0
            
            # Add log probability
            if step_prob > 0:
                log_prob += np.log(step_prob)
            else:
                return float('-inf')  # Zero probability path
            
            # Move to the next node
            current_node = token_node
        
        return log_prob

    def feed_tokens(self, next_tokens: torch.LongTensor) -> Iterable[AdaptiveMaskState]:
        """
        Feed selected next tokens to update the current state of adaptive mask

        Args:
            next_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of selected next tokens in the vocabulary.

        Return:
            `AdaptiveMonitorState`s after updating the state.
        """

        self.states = [state.children[next_tokens[i].item()] \
                       for i, state in enumerate(self.states)]

        return self.states

    def reset(self):
        """
        Reset the monitor state to the initial state
        """

        self.states = [self.root for _ in range(self.batch_size)]

    def json(self) -> str:
        """
        Dump adaptive mask trie into a JSON string.

        Return:
            `str` a JSON string dump of the whole adaptive mask trie
        """

        return json.dumps(self.root.to_dict(), indent=2)

    @staticmethod
    def loads(js: str, num_batch: int = 1) -> Self:
        """
        Load adaptive mask trie from a JSON string.

        Args:
            js (`str`): a JSON string dump of the whole adaptive mask trie.

        Return:
            `AdaptiveMaskTrie` constructed from the JSON string.
        """
        trie = AdaptiveMaskTrie(num_batch)
        trie.root = AdaptiveMaskTrieNode.from_dict(json.loads(js))
        trie.states = [trie.root for _ in range(num_batch)]

        return trie
