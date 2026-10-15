# FLUX 3 Image Klein support for Forge Neo
# The FLUX 3 text encoder: Qwen3-VL 4B hidden states of layers 4, 8, ... 32 under the plain instruct template,
# concatenated per token into 20480 features (upstream src/flux3/text_encoder.py, ComfyUI's Flux3TEModel).
#
# The sequence is right-padded with <|endoftext|> to a multiple of 80 tokens (at most 4096). Padding is masked
# out of the encoder's attention but its states are kept: the DiT attends to every text token, padding included.
# The padding sits at rope position 1, as transformers 4.57 (which upstream pins) puts it.

import math

import torch
from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import INF, SDClipModel, SDTokenizer

PAD = 151643
HIDDEN_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
PAD_MULTIPLE = 80
MAX_LENGTH = 4096


def pad_positions(attention_mask):
    """Rope positions for a right-padded mask: 0, 1, ... over the text, 1 on the padding."""
    return (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)


class PadPositions(torch.nn.Module):
    """Forge's Qwen3-VL with the padding's rope positions; Forge's SDClipModel never passes position ids."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = model.num_layers

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, x, attention_mask=None, **kwargs):
        if attention_mask is not None and kwargs.get("position_ids") is None:
            kwargs["position_ids"] = pad_positions(attention_mask)
        return self.model(x, attention_mask, **kwargs)


def padded_length(length: int) -> int:
    return min(math.ceil(length / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)


class Flux3TextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(PadPositions(text_encoder), layer=HIDDEN_LAYERS, layer_idx=None,
                                        special_tokens={"pad": PAD}, layer_norm_hidden_state=False,
                                        enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def token_ids(self, text: str) -> list[int]:
        # surrounding whitespace is dropped (as Forge's Krea 2 engine does): "", " " and a prompt left with only
        # LoRA tags are all upstream's empty prompt, the CFG unconditional of the base checkpoint
        return self.tokenizer.tokenizer.encode(TEMPLATE.format(text.strip()), add_special_tokens=False)[:MAX_LENGTH]

    def tokenize(self, texts: str | list[str]) -> int | list[int]:
        if isinstance(texts, str):
            return len(self.token_ids(texts))
        return [len(self.token_ids(t)) for t in texts]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                cache[line] = self.encode(line)
            zs.append(cache[line])
        return zs

    def encode(self, line: str) -> torch.Tensor:
        ids = self.token_ids(line)
        ids = ids + [PAD] * (padded_length(len(ids)) - len(ids))
        states = self.text_encoder.encode_token_weights([[(token, 1.0) for token in ids]])[0]
        # (1, layers, T, D) layers-major -> (T, layers * D), each token's layers side by side; no batch dimension,
        # as Forge's native engines return their conditioning (Forge stacks the batch)
        return states[0].transpose(0, 1).flatten(1)
