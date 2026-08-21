"""Text-only loading path for natively-multimodal chat checkpoints (Gemma3,
Gemma4, Qwen3-VL, Mistral Small 3.1, etc.) that ship as *ForConditionalGeneration
architectures rather than plain AutoModelForCausalLM. The harness's own `hf`
backend only supports causal/seq2seq (huggingface.py's _get_backend), and its
`hf-multimodal` backend targets AutoModelForVision2Seq (removed in transformers
5.x, and not the class these newer checkpoints register under anyway). MILU is
pure text (no images) -- we don't need multimodal *input* handling, just a model
class that can load these checkpoints and run ordinary text loglikelihood scoring
through them, reusing HFLM's tokenizer/loglikelihood/generation code unmodified.
"""
import transformers

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("hf-causal-multimodal")
class HFCausalFromMultimodal(HFLM):
    """HFLM's internals branch on `self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM`
    in over a dozen places (tensor building, inplen calc, generation) with no case
    for any other class -- so we keep AUTO_MODEL_CLASS as AutoModelForCausalLM
    (correct: MILU's text-only loglikelihood scoring over these checkpoints *is*
    ordinary decoder-only behavior once no pixel_values are passed) and only swap
    in the real loading class for the single from_pretrained() call inside
    _create_model, which is the one place AUTO_MODEL_CLASS is used for loading
    rather than for causal-vs-seq2seq branching.
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def _create_model(self, *args, **kwargs):
        self.AUTO_MODEL_CLASS = transformers.AutoModelForImageTextToText
        try:
            super()._create_model(*args, **kwargs)
        finally:
            self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
