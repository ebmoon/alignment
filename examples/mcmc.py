import os
import json
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Callable, Any
from datasets import load_dataset

import torch
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer
from alignment.models.transformers import TransformersModel
from alignment.monitor.monitor import Monitor
from alignment.monitor.grammar.llguidance_monitor import LLGuidanceMonitor
from alignment.monitor.adaptive_utils import AdaptiveMask, AdaptiveMaskTrie

TMP_GRAMMAR_PATH = "tmp.lark"
DATASET = "ebmoon/GAD-dataset"
SPLIT = "SLIA"
ID = 'name-combine-2_short'

def is_valid_propose_style(propose_style):
    """
    Check if the propose style is valid
    Valid styles: "prefix", "priority", "restart", or "mix-{p}" where p is a float between 0 and 1
    """
    if propose_style in ["prefix", "priority", "restart"]:
        return True
    
    if propose_style.startswith("mix-"):
        try:
            mix, p = propose_style.split("-")
            p = float(p)
            if mix == "mix" and 0 <= p <= 1:
                return True
        except (ValueError, IndexError):
            pass
    
    return False


class MonitorModel:
    """
    A model wrapper that supports monitor-guided generation with MCMC sampling.
    This enables using grammar and other constraints via monitors rather than logit processors.
    """

    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        monitor: Monitor = None,
        jump_forward: bool = False,
        use_adaptive_mask: bool = False,
        adaptive_mask: AdaptiveMask = None,
    ):
        self.transformer_model = TransformersModel(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.config = model.config
        self.monitor = monitor
        self.jump_forward = jump_forward
        self.use_adaptive_mask = use_adaptive_mask
        self.adaptive_mask = adaptive_mask

    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt for chat models.
        """
        # If the model is a chat model, apply the chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted_prompt
        return prompt

    def set_monitor(self, monitor: Monitor):
        """
        Set or update the monitor for grammar-constrained generation.
        """
        self.monitor = monitor
    
    def set_adaptive_mask(self, adaptive_mask: AdaptiveMask = None):
        """
        Set or update the adaptive mask.
        """
        self.adaptive_mask = adaptive_mask
    
    def set_jump_forward(self, jump_forward: bool = True):
        """
        Set whether to use jump-forward optimization.
        """
        self.jump_forward = jump_forward
    
    @classmethod
    def from_grammar_file(
        cls,
        model_id: str,
        grammar_path: str,
        num_batch: int = 1,
        enable_backtrack: bool = False,
        enable_ff_tokens: bool = False,
        use_adaptive_mask: bool = False,
        cache_dir: str = "/trunk/model-hub",
        **model_kwargs
    ):
        """
        Create a MonitorModel from a grammar file.
        
        Args:
            model_id: The model identifier for loading from HuggingFace
            grammar_path: Path to grammar file (.lark, .json, or .regex)
            num_batch: Batch size for generation
            enable_backtrack: Whether to enable backtracking in the monitor
            enable_ff_tokens: Whether to enable fast-forward tokens
            adaptive_mask: Whether to use adaptive mask
            **model_kwargs: Additional arguments to pass to the model constructor
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, **model_kwargs)
        
        # Check file extension to determine monitor type
        grammar_path = Path(grammar_path)
        ext = grammar_path.suffix.lower()
        
        # Create the appropriate monitor based on file extension
        if ext in [".lark", ".json", ".regex"]:
            monitor = LLGuidanceMonitor.from_tokenizer(
                str(grammar_path),
                tokenizer,
                num_batch=num_batch,
                enable_backtrack=enable_backtrack,
                enable_ff_tokens=enable_ff_tokens
            )
        else:
            raise ValueError(f"Unsupported grammar file extension: {ext}")
        
        # Create adaptive mask if requested
        adaptive_mask_obj = None
        if use_adaptive_mask:
            adaptive_mask_obj = AdaptiveMaskTrie(num_batch)
            
        return cls(model, tokenizer, monitor, False, use_adaptive_mask, adaptive_mask_obj)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        return_scores: bool = False,
        **kwargs
    ) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.Tensor]]:
        """
        Generate text using monitor-guided decoding.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            return_scores: Whether to return logit scores
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated token IDs and optionally scores
        """
        from transformers import GenerationConfig
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
        )
        
        # Check if we have a monitor
        if self.monitor is None:
            raise ValueError("No monitor is set. Use set_monitor() to set a monitor before generation.")
        
        # Generate with monitor guidance
        output = self.transformer_model.generate(
            input_ids,
            generation_config=generation_config,
            monitor=self.monitor,
            jump_forward=self.jump_forward,
            adaptive_mask=self.adaptive_mask,
            **kwargs
        )

        # Extract sequences and scores
        sequences = output.sequences
        
        if return_scores:
            scores = None
            logits = None
            if hasattr(output, 'scores'):
                # Stack scores along sequence dimension
                scores = torch.stack(output.scores, dim=1)
            if hasattr(output, 'logits'):
                logits = torch.stack(output.logits, dim=1)
            return sequences, scores, logits
        return sequences
    
    def _get_seq_logprob_from_scores(self, scores: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of the sequences in `query_ids` given the `scores`.
        
        Args:
            scores: Tensor of shape (batch_size, seq_len, vocab_size)
            query_ids: Tensor of shape (batch_size, seq_len)
            
        Returns:
            Log probabilities of shape (batch_size,)
        """
        assert scores.shape[0] == query_ids.shape[0], "Batch sizes must match"
        assert scores.shape[1] == query_ids.shape[1], "Sequence lengths must match"
    
        # Apply log_softmax to get log-probabilities
        logprobs = torch.log_softmax(scores, dim=-1)
    
        batch_size, seq_len = query_ids.shape
    
        # Initialize result tensor
        result = torch.zeros(batch_size, device=scores.device)
    
        # Process each sequence in the batch
        for i in range(batch_size):
            # Get logprobs for this sequence's tokens
            seq_token_logprobs = logprobs[i, torch.arange(seq_len), query_ids[i]]

            # Find the first EOS token's position (if any)
            eos_mask = query_ids[i] == self.tokenizer.eos_token_id
            eos_positions = torch.nonzero(eos_mask)
            
            if eos_positions.shape[0] > 0:
                # Include up to and including the first EOS token
                first_eos_pos = eos_positions[0].item()
                result[i] = seq_token_logprobs[:first_eos_pos + 1].sum()
            else:
                # No EOS token, sum all logprobs
                result[i] = seq_token_logprobs.sum()
    
        return result
    
    def _resample_idx_distribution(
        self,
        propose_style: str,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the resampling distribution based on the proposal style.
        
        Args:
            propose_style: Style of proposal ("prefix", "priority", "restart")
            current_ids: Current token IDs
            current_scores: Current token scores
            
        Returns:
            Tensor with probability distribution over indices for resampling
        """
        if propose_style == "restart":
            # For resampling, we always resample from the beginning
            resample_distr = torch.zeros(len(current_ids[0]), dtype=torch.float32, device=current_ids.device)
            resample_distr[0] = 1.0
            resample_distr = torch.unsqueeze(resample_distr, 0)
        elif propose_style == "prefix":
            # For prefix sampling, the distribution is uniform
            resample_distr = torch.ones(len(current_ids[0]), device=current_ids.device) / len(current_ids[0])
            resample_distr = torch.unsqueeze(resample_distr, 0)
        elif propose_style == "priority":
            # For priority sampling, the distribution is proportional to the entropy
            current_logprobs = torch.log_softmax(current_scores, dim=-1)
            # Create a mask for non-inf logprobs to avoid NaN values
            mask = torch.isfinite(current_logprobs)
            probs = torch.exp(current_logprobs)
            # Zero out any non-finite values in the probs and calculate entropy properly
            masked_contribution = torch.where(mask, probs * current_logprobs, torch.zeros_like(probs))
            current_entropies = -torch.sum(masked_contribution, dim=-1)

            # Get a probability for each index that is proportional to the entropy
            resample_distr = torch.exp(current_entropies) - 1
            # Handle negative values by zeroing them out
            resample_distr = torch.where(resample_distr > 0, resample_distr, torch.zeros_like(resample_distr))
            # Normalize to sum to 1, handling zero-sum case
            sum_distr = torch.sum(resample_distr)
            if sum_distr > 0:
                resample_distr = resample_distr / sum_distr
            else:
                # Fallback to uniform if all entropies are 0
                resample_distr = torch.ones_like(resample_distr) / len(resample_distr)
        else:
            raise ValueError(f"Unknown proposal style: {propose_style}")
        
        assert resample_distr.shape == current_ids.shape
        assert torch.isclose(resample_distr.sum(), torch.tensor(1.0, device=resample_distr.device))
        return resample_distr
    
    def _propose_next_sequence_logprob(
        self,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
        next_ids: torch.Tensor,
        next_scores: torch.Tensor,
        propose_style: str,
    ) -> float:
        """
        Compute the log probability of proposing the next sequence given the current sequence.
        
        Args:
            current_ids: Current token IDs
            current_scores: Current token scores
            next_ids: Proposed next token IDs
            next_scores: Proposed next token scores
            propose_style: Style of proposal
            
        Returns:
            Log probability of the proposal
        """
        resample_idx_distr = self._resample_idx_distribution(
            propose_style, current_ids, current_scores
        )

        # Get the longest common prefix between the proposal and the current
        lcp_idx = 0
        for i, (p, c) in enumerate(zip(next_ids[0], current_ids[0])):
            if p == c:
                lcp_idx += 1
            else:
                break
        max_resample_idx = lcp_idx + 1
        max_resample_idx = min(max_resample_idx, len(current_ids[0]))

        # Compute the probability of the proposal
        proposal_logprob = -np.inf
        for i in range(max_resample_idx):
            # Get probability of selecting this index
            idx_resample_prob = resample_idx_distr[0][i].item()
            if idx_resample_prob == 0:
                continue
            idx_resample_logprob = np.log(idx_resample_prob)
            
            suffix_ids = next_ids[:, i:]
            suffix_scores = next_scores[:, i:]

            # Get log probability
            suffix_logprob = self._get_seq_logprob_from_scores(suffix_scores, suffix_ids)
            
            # Add to total probability using log-sum-exp
            proposal_logprob = np.logaddexp(proposal_logprob, idx_resample_logprob + suffix_logprob)

        return proposal_logprob
    
    def _propose_next_sequence(
        self,
        prompt_ids: torch.Tensor,
        current_ids: torch.Tensor,
        max_new_tokens: int,
        current_scores: Optional[torch.Tensor] = None,
        propose_style: str = "restart",
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Propose a new sequence by resampling from a chosen index.
        
        Args:
            prompt_ids: Prompt token IDs
            current_ids: Current token IDs
            max_new_tokens: Maximum number of new tokens to generate
            current_scores: Current token scores
            propose_style: Style of proposal
            
        Returns:
            Tuple of (next_ids, next_scores, proposal_logprob)
        """
        assert current_ids.shape[0] == 1
        
        # Get the scores if not provided
        if current_scores is None:
            _, current_scores, current_logits = self._generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                return_scores=True,
            )

        # Get distribution over indices for resampling
        resample_idx_distr = self._resample_idx_distribution(
            propose_style, current_ids, current_scores
        )

        # Sample an index based on the distribution
        resample_idx = torch.multinomial(
            resample_idx_distr[0], 
            num_samples=1
        ).item()
        
        print(f"Resample idx: {resample_idx}")

        # Get the corresponding prefix from current tokens
        prefix_ids = current_ids[:, :resample_idx]
        prefix_scores = current_scores[:, :resample_idx] if resample_idx > 0 else None

        # Combine prompt with prefix for new generation
        if prefix_ids.shape[1] > 0:
            start_ids = torch.cat([prompt_ids, prefix_ids], dim=1)
        else:
            start_ids = prompt_ids

        # Reset monitor state to start fresh
        self.monitor.reset()
        
        # Generate from the prefix
        next_ids, next_scores, next_logits = self._generate(
            start_ids,
            max_new_tokens=max_new_tokens - resample_idx,
            do_sample=True,
            return_scores=True,
        )
        
        # Extract just the newly generated part (excluding prompt)
        start_len = prompt_ids.shape[1]
        resample_ids = next_ids[:, start_len:]
        resample_scores = next_scores[:, :resample_ids.shape[1]]
        
        # If we had a prefix, combine it with the newly sampled part
        if prefix_scores is not None:
            next_scores = torch.cat([prefix_scores, resample_scores], dim=1)
        else:
            next_scores = resample_scores

        proposal_logprob = self._propose_next_sequence_logprob(
            current_ids=current_ids,
            current_scores=current_scores,
            next_ids=resample_ids,
            next_scores=next_scores,
            propose_style=propose_style,
        )

        return resample_ids, next_scores, proposal_logprob


class MCMC:
    """
    Markov Chain Monte Carlo sampling using monitor-guided generation.
    Uses a monitor to constrain the generation according to a grammar.
    """
    
    def __init__(
        self, 
        model: MonitorModel, 
        prompt: str, 
        propose_style: str,
        name_prefix: str,
        root_log_dir: str, 
    ):
        """
        Initialize the MCMC sampler.
        
        Args:
            model: A MonitorModel instance
            prompt: The prompt to start generation from
            propose_style: Style of proposal ("prefix", "priority", "restart", or "mix-{p}")
            name_prefix: Prefix for output file names
            root_log_dir: Directory to store output files
        """
        self.model = model
        prompt = model._format_prompt(prompt)
        self.prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        assert is_valid_propose_style(propose_style), f"Invalid propose style: {propose_style}"
        self.propose_style = propose_style
        self.root_log_dir = root_log_dir
        os.makedirs(root_log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"{root_log_dir}/{timestamp}-{name_prefix}-{propose_style}"
        os.makedirs(self.log_dir, exist_ok=True)

    def get_sample(self, n_steps: int, max_new_tokens: int):
        """
        Get a single sample using MCMC sampling.
        
        Args:
            n_steps: Number of MCMC steps
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated token IDs
        """
        # Clear cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get initial sample with monitor-guided generation
        current_ids, current_scores, current_logits = self.model._generate(
            self.prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_scores=True,
        )
        
        # Compute initial log probabilities
        current_raw_logprob = self.model._get_seq_logprob_from_scores(current_scores, current_ids)
        print(f"Initial: {[self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]}")

        steps = []
        timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
        sample_file = f"{self.log_dir}/{timestamp}-n{n_steps}.json"

        for i in range(n_steps):
            # Determine proposal style for this step
            step_propose_style = self.propose_style
            if step_propose_style.startswith("mix"):
                _, p = step_propose_style.split("-")
                p = float(p)
                step_propose_style = "restart" if np.random.rand() < p else "priority"
            print(f"Step {i} ({step_propose_style})")

            print(f"Current: {[self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]}")
            print(f"Current raw logprob: {current_raw_logprob.item()}")
            
            # Get proposal
            proposal_ids, proposal_scores, prop_logprob_cur_to_next = self.model._propose_next_sequence(
                prompt_ids=self.prompt_ids,
                current_ids=current_ids,
                max_new_tokens=max_new_tokens,
                current_scores=current_scores,
                propose_style=step_propose_style,
            )
            
            proposal_raw_logprob = self.model._get_seq_logprob_from_scores(proposal_scores, proposal_ids)
            print(f"Proposal: {[self.model.tokenizer.decode(token_id) for token_id in proposal_ids[0]]}")
            print(f"Proposal raw logprob: {proposal_raw_logprob.item()}")

            # Compute acceptance probability
            acceptance_prob = None
            if torch.equal(current_ids, proposal_ids):
                acceptance_prob = 1
            else:
                # Compute reverse proposal probability
                prop_logprob_next_to_cur = self.model._propose_next_sequence_logprob(
                    current_ids=proposal_ids,
                    current_scores=proposal_scores,
                    next_ids=current_ids,
                    next_scores=current_scores,
                    propose_style=step_propose_style,
                )

                # Metropolis-Hastings acceptance ratio
                log_acc_ratio = proposal_raw_logprob + prop_logprob_next_to_cur - \
                    current_raw_logprob - prop_logprob_cur_to_next

                acceptance_prob = min(1, np.exp(log_acc_ratio))
            
            print(f"Acceptance prob: {acceptance_prob}")
    
            # Decide whether to accept the proposal
            accepted = bool(np.random.rand() < acceptance_prob)

            # Save step information
            step = {
                "current": {
                    "tokens": [self.model.tokenizer.decode(token_id) for token_id in current_ids[0]],
                    "token_ids": [int(id) for id in current_ids[0]],
                    "raw_logprob": current_raw_logprob.item(),
                },
                "proposal": {
                    "tokens": [self.model.tokenizer.decode(token_id) for token_id in proposal_ids[0]],
                    "token_ids": [int(id) for id in proposal_ids[0]],
                    "raw_logprob": proposal_raw_logprob.item(),
                },
                "acceptance_prob": acceptance_prob,
                "accepted": accepted,
            }
            steps.append(step)
            steps_dump = {"steps": steps}
            with open(sample_file, "w") as f:
                json.dump(steps_dump, f, indent=4)

            # Update current sequence if accepted
            if accepted:
                current_ids = proposal_ids
                current_scores = proposal_scores
                current_raw_logprob = proposal_raw_logprob
                print(f"Accepted")
            
            print("\n\n")
            
        return current_ids

    def get_samples(self, n_samples: int, n_steps: int, max_new_tokens: int):
        """
        Get multiple samples using MCMC sampling.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Number of MCMC steps per sample
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of generated samples
        """
        samples = []
        for i in range(n_samples):
            print(f"Sample {i}")
            sample_start_time = time.time()
            sample = self.get_sample(n_steps, max_new_tokens)
            sample_end_time = time.time()
            sample_time = sample_end_time - sample_start_time
            print(f"Sample time: {sample_time:.2f} s")
            sample_str = self.model.tokenizer.decode(sample[0])
            print(f"Sample: {sample_str}")
            samples.append(sample_str)
        
        return samples

def load_dataset_and_grammar():
    """Load the dataset and grammar"""
    dataset = load_dataset(DATASET, split=SPLIT)
    
    prompt = None
    grammar_str = None
    
    for data in dataset:
        if ID == data['id']:
            prompt = data['prompt']
            grammar_str = data['grammar']
            break
    
    grammar_str = grammar_str.replace("ntInt", "ntint")
    grammar_str = grammar_str.replace("ntBool", "ntbool")
    grammar_str = grammar_str.replace("ntString", "ntstring")
    grammar_str = grammar_str.replace("root", "start")
    grammar_str = grammar_str.replace("Start", "ntstart")
    grammar_str = grammar_str.replace("::=", ":")
    
    print("Grammar loaded:")
    print(grammar_str)

    # Save grammar to temporary file
    with open(TMP_GRAMMAR_PATH, "w") as f:
        f.write(grammar_str)
    
    return prompt, grammar_str

def prepare_input(tokenizer, prompt):
    """Prepare the input for generation"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    decode_output = tokenizer(
        [formatted_prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )
    input_ids = decode_output["input_ids"]
    attention_mask = decode_output["attention_mask"]
    
    return input_ids, attention_mask

def run_mcmc_example():
    """
    Example of how to run MCMC with LLGuidance monitor.
    """

    # Model ID and parameters
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    torch_dtype = torch.bfloat16
    
    # Set up paths
    root_log_dir = "mcmc_runs"
    
    # Load dataset and grammar
    # prompt, grammar_str = load_dataset_and_grammar()
    # grammar_path = TMP_GRAMMAR_PATH

    with open("data/sygus/prompts/qm_max3.txt", "r") as f:
        prompt = f.read().strip()

    grammar_path = "data/sygus/grammar/qm_max3.lark"
    name_prefix = ID

    # Create model with monitor
    model = MonitorModel.from_grammar_file(
        model_id=model_id,
        grammar_path=grammar_path,
        device_map="auto", 
        torch_dtype=torch_dtype,
        use_adaptive_mask=True,
    )

    # Set up MCMC parameters
    propose_style = "restart"
    
    # Run MCMC
    mcmc = MCMC(model, prompt, propose_style, name_prefix, root_log_dir)
    samples = mcmc.get_samples(
        n_samples=200,
        n_steps=21,
        max_new_tokens=64
    )
    
    return samples


if __name__ == "__main__":
    run_mcmc_example()