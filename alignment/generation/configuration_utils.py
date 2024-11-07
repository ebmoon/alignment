from typing import TYPE_CHECKING, Optional

from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ExplicitEnum

if TYPE_CHECKING:
    from transformers import PreTrainedModel

class AlignmentGenerationMode(ExplicitEnum):
    """
    Possible (additional) generation modes of alignments
    """

    # Non-beammethods
    MONITOR_GUIDED = "monitor_guided"
    SPECULATIVE_MONITOR_GUIDED = "speculative_monitor_guided"

class AlignmentGenerationConfig(GenerationConfig):
    """
    A subclass that holds a configuration for alignment generation task. 
    A `generate` call of alignment model supports the following generation methods.

        - *assisted guided decoding* if `assistant_model` is passed to `.generate()`.
        - *guided decoding* otherwise
    """
    def __init__(self, **kwargs):
        # Checker guided generation
        self.monitor = kwargs.pop("monitor", None)
        self.jump_forward = kwargs.pop("jump_forward", True)
        self.backtrack = kwargs.pop("backtrack", True)
        self.adaptive_mask = kwargs.pop("adaptive_mask", True)
        self.adaptive_trie = kwargs.pop("adaptive_trie", None)

        super().__init__(**kwargs)

    def get_alignment_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> AlignmentGenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `AlignmentGenerationMode`: The generation mode triggered by the instance.
        """
        if assistant_model is not None:
            generation_mode = AlignmentGenerationMode.SPECULATIVE_MONITOR_GUIDED
        elif self.monitor is not None:
            generation_mode = AlignmentGenerationMode.MONITOR_GUIDED
        else:
            raise ValueError('Guide must be provided for alignement generation')

        return generation_mode