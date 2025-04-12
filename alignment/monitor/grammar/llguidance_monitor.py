from typing import Dict, Iterable, Self
import torch
from transformers import PreTrainedTokenizer
from pathlib import Path

import llguidance.hf
from llguidance import JsonCompiler, LarkCompiler, LLInterpreter, RegexCompiler
from llguidance.torch import (
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
    fill_next_token_bitmask,
    get_bitmask_shape,
)

from ..monitor import Monitor, MonitorState


class LLGuidanceMonitorState(MonitorState):
    """
    A state of LLGuidance monitor that contains the interpreter state.
    """
    
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def feed_token(self, token_id: int) -> Self:
        """
        Feed token to the current state and return a new monitor state after transition.
        """
        self.interpreter.commit_token(token_id)

        return self


class LLGuidanceMonitor(Monitor):
    """A monitor to guide LLM to generate output using the llguidance library"""
    
    def __init__(
            self,
            grammar_path: str,
            tokenizer,
            num_batch: int = 1,
            enable_backtrack: bool = False,
            enable_ff_tokens: bool = False
        ):
        self.grammar_path = grammar_path
        self.tokenizer = tokenizer
        self.num_batch = num_batch
        self.enable_backtrack = enable_backtrack
        self.enable_ff_tokens = enable_ff_tokens
        
        self.compiled_grammar = self._compile_grammar(grammar_path)
        self.llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        
        self.initial_state = None
        self.state = None
        self._initialize_state()
        
        # For token filtering - properly handle batch size
        self.mask = allocate_token_bitmask(self.num_batch, self.llg_tokenizer.vocab_size)

    @classmethod
    def from_tokenizer(
            cls,
            grammar_path: str,
            tokenizer: PreTrainedTokenizer,
            num_batch: int = 1,
            enable_backtrack: bool = False,
            enable_ff_tokens: bool = False
        ) -> Self:
        """Build LLGuidance monitor from grammar and tokenizer"""
        return LLGuidanceMonitor(grammar_path, tokenizer, num_batch, enable_backtrack, enable_ff_tokens)

    def _compile_grammar(self, grammar_path: str):
        """Compile the grammar based on file extension"""
        path = Path(grammar_path)
        ext = path.suffix.lower()

        if ext == ".json":
            compiler = JsonCompiler()
        elif ext == ".lark":
            compiler = LarkCompiler()
        elif ext == ".regex":
            compiler = RegexCompiler()
        else:
            raise ValueError(f"Unsupported grammar file extension: {ext}")

        grammar_text = path.read_text()
        return compiler.compile(grammar_text)

    def _initialize_state(self):
        """Initialize the monitor state with interpreters"""
        interpreters = []
        for _ in range(self.num_batch):
            interpreter = LLInterpreter(
                self.llg_tokenizer,
                self.compiled_grammar,
                enable_backtrack=self.enable_backtrack,
                enable_ff_tokens=self.enable_ff_tokens,
            )
            interpreter.start_without_prompt()
            interpreters.append(interpreter)
            
        self.initial_state = LLGuidanceMonitorState(interpreters[0])
        self.state = [LLGuidanceMonitorState(interpreter) for interpreter in interpreters]

    def filter_vocab(self, input_ids: torch.LongTensor):
        """
        Filter out next tokens for the current input that do not pass the monitor.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Return:
            A mask tensor of shape `(batch_size, vocab_size/32, 32)` indicating allowed tokens.
            The mask is stored in a bit-packed format for efficiency.
        """
        assert input_ids.shape[0] == self.num_batch
        
        # Fill the mask for each batch item
        for idx, s in enumerate(self.state):
            fill_next_token_bitmask(s.interpreter, self.mask, index=idx)
        
        return self.mask
    
    def get_tokens_from_mask(self, mask, input_ids: torch.LongTensor):
        """
        Convert a mask of valid tokens to a list of token IDs for each batch.
        
        Args:
            mask: A bit-packed tensor mask of shape `(batch_size, vocab_size/32, 32)` 
                 indicating allowed tokens.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                
        Return:
            List of tensors containing acceptable token IDs for each batch item.
        """
        device = input_ids.device
        
        # Convert the bitmask to a list of valid token ids for each batch
        acceptances = []
        for idx in range(mask.shape[0]):
            # Get the mask for this batch item
            batch_mask = mask[idx]
            
            # Expand the mask to get individual bits
            mask_expanded = torch.repeat_interleave(batch_mask, 32, dim=0)
            bit_indices = torch.arange(32, device=mask.device, dtype=torch.int32).repeat(batch_mask.shape[0])
            
            # Extract each bit (1 means token is valid)
            bit_masks = (mask_expanded >> bit_indices) & 1
            
            # Get indices of valid tokens
            valid_token_indices = torch.nonzero(bit_masks, as_tuple=True)[0]
            
            # Trim to match vocab size if needed
            vocab_size = self.llg_tokenizer.vocab_size
            valid_token_indices = valid_token_indices[valid_token_indices < vocab_size]
            
            # Move to the correct device
            acceptances.append(valid_token_indices.to(device))
            
        return acceptances

    def mask_logits(self, logits: torch.Tensor, mask) -> None:
        """
        Apply the monitor's constraints directly to logits in-place.
        
        This is more efficient than filtering the vocabulary and then masking the logits separately,
        especially for large vocabulary sizes.

        Args:
            logits (`torch.Tensor` of shape `(batch_size, vocab_size)`):
                Logits to be masked in-place. Invalid tokens will be set to -inf.
            mask: A bit-packed tensor mask of shape `(batch_size, vocab_size/32, 32)` 
                 indicating allowed tokens.
        """
        assert logits.shape[0] == self.num_batch
        
        # Apply the mask directly to the logits
        apply_token_bitmask_inplace(logits, mask.to(logits.device))

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
        """Reset the monitor state"""
        interpreters = []
        for _ in range(self.num_batch):
            interpreter = LLInterpreter(
                self.llg_tokenizer,
                self.compiled_grammar,
                enable_backtrack=self.enable_backtrack,
                enable_ff_tokens=self.enable_ff_tokens,
            )
            interpreter.start_without_prompt()
            interpreters.append(interpreter)
            
        self.state = [LLGuidanceMonitorState(interpreter) for interpreter in interpreters]
