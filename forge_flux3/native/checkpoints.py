# FLUX 3 Image Klein support for Forge Neo
# What the loader hook checks and cleans in the selected files before Forge builds the components. No Forge
# imports: the CPU tests run it on the real checkpoints' headers.

# a Qwen3-VL 4B language tower: 36 layers, hidden 2560. Forge renames the language keys inside its own branch, so
# the hook sees them as the file stores them: model.language_model.* (Comfy-Org's qwen3vl_4b_*) or model.*
TE_PREFIXES = ("model.language_model.", "model.")
TE_KEYS = ("layers.0.self_attn.q_norm.weight", "layers.35.self_attn.q_proj.weight")
TE_NORM = "layers.0.input_layernorm.weight"
TE_HIDDEN = 2560
MISSING_TE = ("FLUX 3 needs its Qwen3-VL 4B text encoder: select qwen3vl_4b_int8_convrot.safetensors (or "
              "qwen3vl_4b_bf16) under VAE / Text Encoder")
WRONG_TE = ("FLUX 3 needs the Qwen3-VL 4B text encoder, and the one selected under VAE / Text Encoder is another "
            "size: select qwen3vl_4b_int8_convrot.safetensors (or qwen3vl_4b_bf16)")
MISSING_VAE = "FLUX 3 needs the FLUX.2 autoencoder: select flux2-vae.safetensors under VAE / Text Encoder"

# per-block attention settings ComfyUI stores next to the INT8 weights ({"attention": "comfy_kitchen_int8"}): a
# ComfyUI kernel choice, not a weight
DROPPED_SUFFIXES = (".comfy_attention.config",)


def check_text_encoder(state_dict) -> None:
    """Raise the user-facing message when the selected text encoder cannot be the Qwen3-VL 4B language tower."""
    if not isinstance(state_dict, dict) or len(state_dict) <= 16:
        raise ValueError(MISSING_TE)
    prefix = next((p for p in TE_PREFIXES if all(f"{p}{k}" in state_dict for k in TE_KEYS)), None)
    if prefix is None:
        raise ValueError(MISSING_TE)
    norm = state_dict.get(f"{prefix}{TE_NORM}")
    shape = norm.get("shape") if isinstance(norm, dict) else getattr(norm, "shape", None)
    if shape is not None and shape[0] != TE_HIDDEN:
        raise ValueError(WRONG_TE)


def check_vae(state_dict) -> None:
    if not isinstance(state_dict, dict) or len(state_dict) <= 16:
        raise ValueError(MISSING_VAE)


def clean_transformer_state_dict(state_dict) -> None:
    for key in [k for k in state_dict if k.endswith(DROPPED_SUFFIXES)]:
        del state_dict[key]
