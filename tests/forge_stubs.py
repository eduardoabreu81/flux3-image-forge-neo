"""Minimal stand-ins for the Forge Neo modules the extension imports, for CPU tests without Forge.

Only the names the extension touches exist; anything else raises AttributeError, so a test notices when the
extension starts depending on more of Forge. Same shape as the Iris 3B extension's forge_stubs.
"""

import sys
import types
from contextlib import contextmanager


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class FakeHFTokenizer:
    """One token per character, ids above the special range; PAD 151643 never appears in text."""

    def encode(self, text, add_special_tokens=True):
        return [1000 + ord(c) for c in text]


class FakeSDTokenizer:
    def __init__(self, tokenizer_object=None, **kwargs):
        self.tokenizer = tokenizer_object


class FakeSDClipModel:
    """encode_token_weights as Forge's does with a layer list: states (B, L, T, D), layer l of token t = 100 l + t."""

    layers, dim = 8, 3

    def __init__(self, transformer=None, **kwargs):
        self.transformer = transformer
        self.kwargs = kwargs
        self.calls = []

    def encode_token_weights(self, token_weight_pairs):
        import torch

        tokens = [pair[0] for pair in token_weight_pairs[0]]
        self.calls.append(tokens)
        layer = torch.arange(self.layers, dtype=torch.float32)[:, None, None] * 100
        position = torch.arange(len(tokens), dtype=torch.float32)[None, :, None]
        states = (layer + position).expand(self.layers, len(tokens), self.dim)[None].clone()
        return states, None, {"attention_mask": torch.ones(1, len(tokens))}


def text_processing():
    """backend.args and backend.text_processing, enough to import forge_flux3.native.text_engine."""
    comfy = module("backend.text_processing._comfy", EMBEDDINGS=list, INF=float("inf"),
                   SDTokenizer=FakeSDTokenizer, SDClipModel=FakeSDClipModel)
    emphasis = module("backend.text_processing.emphasis", uses_emphasis=lambda text: False,
                      EmphasisNone=type("EmphasisNone", (), {}))
    package = module("backend.text_processing", emphasis=emphasis, _comfy=comfy)
    args = module("backend.args", dynamic_args=types.SimpleNamespace(last_extra_generation_params={},
                                                                     ref_latents=[], klein=True))
    return {"backend": module("backend", args=args, text_processing=package), "backend.args": args,
            "backend.text_processing": package, "backend.text_processing.emphasis": emphasis,
            "backend.text_processing._comfy": comfy}


@contextmanager
def installed(stubs):
    """The stubs in sys.modules for the duration, and the extension's modules re-imported against them."""
    saved = {name: sys.modules.get(name) for name in stubs}
    ours = {name: mod for name, mod in sys.modules.items() if name.startswith("forge_flux3")}
    sys.modules.update(stubs)
    for name in ours:
        del sys.modules[name]
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for name in [n for n in sys.modules if n.startswith("forge_flux3")]:
            del sys.modules[name]
        sys.modules.update(ours)
