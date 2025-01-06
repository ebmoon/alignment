from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union, Tuple

import copy
import inspect
import warnings
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (
    GenerationConfig,
    GenerateOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput
)
from transformers.generation.candidate_generator import (
    _prepare_attention_mask,
    _prepare_token_type_ids
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.utils import is_torchdynamo_compiling, logging

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .tokenizer import Tokenizer

from ..monitor.monitor import Monitor
from ..monitor.adaptive_utils import (
    AdaptiveMask
)

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


class TransformerTokenizer(Tokenizer):
    """Tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        self.tokenizer = tokenizer

        super().__init__(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            vocabulary=tokenizer.get_vocab(),
        )

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
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

    def decode(self, token_ids: torch.LongTensor, **kwargs) -> List[str]:
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


class TransformersModel:
    """A class for `transformers` models."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.config = model.config

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        monitor: Monitor = None,
        jump_forward: bool = True,
        adaptive_mask: AdaptiveMask = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self.model._validate_model_class()
        tokenizer = kwargs.pop(
            "tokenizer", None
        )  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop(
            "assistant_tokenizer", None
        )  # only used for assisted generation

        generation_config, model_kwargs = self.model._prepare_generation_config(
            generation_config, **kwargs
        )
        self.model._validate_model_kwargs(model_kwargs.copy())
        self.model._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )
        requires_attention_mask = ("encoder_outputs" not in model_kwargs
                                   or generation_config.cache_implementation == "static")
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self.model._prepare_special_tokens(
            generation_config, kwargs_has_attention_mask, device=device
        )

        # decoder-only models must use left-padding for batched generation.
        if not self.model.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(
                    inputs_tensor[:, -1] == generation_config._pad_token_tensor
                )
                > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if (
            not kwargs_has_attention_mask
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs["attention_mask"] = (
                self.model._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config, model_kwargs
                )
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if (
                model_input_name == "input_ids"
                and len(model_kwargs["attention_mask"].shape) > 2
            ):
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.model.config.is_encoder_decoder:
            input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = (
                inputs_tensor
                if model_input_name == "input_ids"
                else model_kwargs.pop("input_ids")
            )

        if generation_config.token_healing:
            input_ids = self.model.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and generation_config.min_length is not None
        )
        generation_config = self.model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if (
            self.model._supports_num_logits_to_keep()
            and "num_logits_to_keep" not in model_kwargs
        ):
            model_kwargs["num_logits_to_keep"] = 1

        self.model._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
        cache_name = (
            "past_key_values"
            if "mamba" not in self.__class__.__name__.lower()
            else "cache_params"
        )
        user_defined_cache = model_kwargs.get(cache_name)
        max_cache_length = generation_config.max_length
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.model.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self.model._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            assistant_model,
            batch_size,
            max_cache_length,
            device,
        )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.model.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.model.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.model.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if monitor:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing monitor guided generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if not model_kwargs["use_cache"]:
                raise ValueError("monitor guided generate requires `use_cache=True`")
            if self.model._is_stateful:
                # In monitor guided generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._monitor_guided_decoding(
                input_ids,
                monitor=monitor,
                jump_forward=jump_forward,
                adaptive_mask=adaptive_mask,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else:
            raise ValueError("Alignment only supports monitor-guided generation")

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache
            is not False  # Should check for `True` after v4.47
            and not is_torchdynamo_compiling()
            and hasattr(result, "past_key_values")
            and hasattr(result.past_key_values, "to_legacy_cache")
            and result.past_key_values.to_legacy_cache is not None
        ):
            # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
            should_convert_cache = generation_config.return_legacy_cache
            is_user_defined_cache = user_defined_cache is not None
            is_default_cache_type = (
                type(result.past_key_values) == DynamicCache # noqa E721
                or (
                    isinstance(result.past_key_values, EncoderDecoderCache)
                    and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
                    and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
                )
            )
            if not is_user_defined_cache and is_default_cache_type:
                logger.warning_once(
                    "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                    "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                    "keep returning the legacy format, please set `return_legacy_cache=True`."
                )
                should_convert_cache = True
            if should_convert_cache:
                result.past_key_values = result.past_key_values.to_legacy_cache()
        
        return result

    def _monitor_guided_decoding(
        self,
        input_ids: torch.LongTensor,
        monitor: Monitor,
        jump_forward: bool,
        adaptive_mask: AdaptiveMask,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), guided by external monitor. Monitor-guided generation is an extension
        of constrained-decoding. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            monitor (`Monitor`):
                A derived instance of [`Monitor`] that defines which sequences are invalid. For
                more information, the documentation of [`Monitor`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self.model._get_initial_cache_position(input_ids, model_kwargs)

        vocab_size = len(self.tokenizer.get_vocab())

        # initialize monitor state
        monitor.reset()

        if adaptive_mask:
            adaptive_mask.reset()
            temp_logits = torch.ones([batch_size, vocab_size], device=input_ids.device)

        this_peer_finished = False
        is_first_iteration = True # to preserve the same API in the output as other generation methods
        while self.model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            cur_len = input_ids.shape[-1]
            
            # 1. Check acceptable next tokens by monitor
            acceptance_batch = monitor.filter_vocab(input_ids)
            acceptance_batch_seq = [acceptance_batch]

            acceptance = torch.full((batch_size, 1, vocab_size), False)
            for i in range(batch_size):
                acceptance[i, 0, acceptance_batch[i]] = True
            acceptance_sequence = acceptance.clone()

            # 2. Jump-forward if only a single next token is acceptable
            jumped_len = 0
            jumped_states = []
            while jump_forward and input_ids.shape[-1] < max_length - 1 \
                and all(len(acceptance) == 1 for acceptance in acceptance_batch):

                ids = torch.tensor(
                    [acceptance[0] for acceptance in acceptance_batch],
                    dtype=torch.long,
                    device=input_ids.device)
                acceptance_tensor_col = ids.reshape(-1, 1)

                jumped_input_ids = torch.cat([input_ids, acceptance_tensor_col], dim=-1)
                is_done = stopping_criteria(jumped_input_ids, None)

                # 2.1. Do not apply jump-forward for the last token
                # to compute original logits for jump-forwarded tokens
                if not is_done:
                    input_ids = jumped_input_ids

                    monitor.update(ids)

                    # 2.2. If applying adaptive mask, set temporary logits for jumped tokens,
                    # but memorize those states for future update to correct logits
                    if adaptive_mask:
                        adaptive_mask.update_scores(
                            acceptance_batch,
                            temp_logits,
                            self.tokenizer.eos_token_id)

                        jumped_states.append(adaptive_mask.states)
                        adaptive_mask.feed_tokens(ids)

                    acceptance_batch = monitor.filter_vocab(input_ids)
                    acceptance_batch_seq.append(acceptance_batch)

                    acceptance = torch.full((batch_size, 1, vocab_size), False)
                    for i in range(batch_size):
                        acceptance[i, 0, acceptance_batch[i]] = True
                    acceptance_sequence = torch.concat([acceptance_sequence, acceptance], dim=1)

                    jumped_len += 1
                else:
                    break

            # 3. prepare model inputs
            candidate_kwargs = copy.copy(model_kwargs)
            candidate_kwargs = _prepare_attention_mask(
                candidate_kwargs, input_ids.shape[1], self.config.is_encoder_decoder
            )
            candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, input_ids.shape[1])

            # set cache position must be set to get multiple logits
            if "cache_position" in candidate_kwargs:
                candidate_kwargs["cache_position"] = torch.cat(
                    (
                        candidate_kwargs["cache_position"],
                        torch.arange(cur_len, cur_len + jumped_len, device=input_ids.device, dtype=torch.long),
                    ),
                    dim=0,
                )
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **candidate_kwargs)
            if "num_logits_to_keep" in model_inputs:
                model_inputs["num_logits_to_keep"] = jumped_len + 1

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            
            model_inputs.update({"position_ids": model_inputs["position_ids"].to(input_ids.device)})
            model_inputs.update({"attention_mask": model_inputs["attention_mask"].to(input_ids.device)})

            # forward pass to get next token
            outputs = self.model(**model_inputs)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                num_new_tokens=jumped_len + 1
            )
            if synced_gpus and this_peer_finished:
                continue

            # 3. Process the new logits
            # .float() is needed to retain precision for later logits manipulations
            new_logits = outputs.logits[:, -jumped_len - 1:].float()
            new_logits = new_logits.to(input_ids.device)
            next_token_logits = new_logits.clone()

            # 3.1. Apply binary mask to logits
            new_logits[~acceptance_sequence] = -float('inf')

            # 3.2. If mask is adaptive, apply adaptive mask for the last token
            if adaptive_mask:
                # 3.2.1. Update logit for the last token
                adaptive_mask.update_scores(
                    acceptance_batch,
                    next_token_logits[:, -1, :],
                    self.tokenizer.eos_token_id)

                # 3.2.2. Update jumped tokens to the correct logit
                for i, states in reversed(list(enumerate(jumped_states))):
                    adaptive_mask.update_scores(
                        acceptance_batch_seq[i],
                        next_token_logits[:, i, :],
                        self.tokenizer.eos_token_id,
                        states
                    )

                mask = adaptive_mask.mask(batch_size, vocab_size)
                mask = mask.to(input_ids.device)
                new_logits[:, -1, :] += mask

            # process logits by logits processors
            if len(logits_processor) > 0:
                for i in range(jumped_len + 1):
                    new_logits[:, i, :] = logits_processor(input_ids[:, : cur_len + i], new_logits[:, i, :])

            # token selection
            if do_sample:
                probs = nn.functional.softmax(new_logits[:, -1, :], dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(new_logits[:, -1, :], dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            new_cur_len = input_ids.shape[-1]

            # 4. Update monitor state by selected next tokens
            monitor.update(next_tokens)
            if adaptive_mask:
                adaptive_mask.feed_tokens(next_tokens)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                newly_added_length = jumped_len + 1
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(newly_added_length))
                if output_logits:
                    raw_logits += tuple(next_token_logits[:, i, :] for i in range(newly_added_length))

                newly_added_length = new_cur_len if is_first_iteration else newly_added_length
                if output_attentions:
                    if self.model.config.is_encoder_decoder:
                        cross_attentions = _split_model_outputs(
                            cross_attentions, outputs.cross_attentions, cur_len, newly_added_length
                        )
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.decoder_attentions,
                            cur_len,
                            newly_added_length,
                            is_decoder_attention=True,
                        )
                    # some (V)LLMs have hard requirement on SDPA and thus never return attn
                    elif outputs.attentions[0] is not None:
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.attentions,
                            cur_len,
                            newly_added_length,
                            is_decoder_attention=True,
                        )
                if output_hidden_states:
                    if self.model.config.is_encoder_decoder:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.decoder_hidden_states, cur_len, newly_added_length
                        )
                    else:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.hidden_states, cur_len, newly_added_length
                        )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            is_first_iteration = False

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate and self.model.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        elif return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
        
def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # The first iteration contains the prompt + 1 generated token, let's update the length variables accordingly
        cur_len += 1
        added_len -= cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i : i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs

