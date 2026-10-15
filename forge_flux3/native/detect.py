# FLUX 3 Image Klein support for Forge Neo
# Recognize a FLUX 3 image DiT and read its configuration from tensor shapes.
#
# Works on a loaded state dict or on a safetensors header: shapes are read through shape_of, so no tensor has to be
# materialized. Same identity keys as ComfyUI's model_detection (COMFYUI_COMMIT in native/__init__.py).

import re

IDENTITY_KEYS = ("content_mode_blocks.image.0.attn_out.weight", "single_stream_modulations.txt.lin.weight")
HEAD_DIM = 128
FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
# text features: 8 Qwen3-VL 4B layers of 2560 (text_engine.HIDDEN_LAYERS); read from the configuration, not the
# txt_in columns, which 4-bit builds pack two to a byte
TEXT_DIM = 20480


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_flux3(state_dict, key_prefix=""):
    return all(f"{key_prefix}{k}" in state_dict for k in IDENTITY_KEYS)


def detect(state_dict, key_prefix=""):
    """The Flux3DiT keyword arguments for this checkpoint, or None when it is not a FLUX 3 image DiT."""
    if not is_flux3(state_dict, key_prefix):
        return None

    def shape(key):
        return shape_of(state_dict[f"{key_prefix}{key}"])

    keys = list(state_dict.keys())
    # widths come from a weight's rows (its output features): NVFP4 packs two values per byte along the inputs
    hidden = shape("content_mode_blocks.image.0.attn_out.weight")[0]
    return {
        "image_model": "flux3",
        "hidden_size": hidden,
        "num_heads": hidden // HEAD_DIM,
        "depth": count_blocks(keys, f"{key_prefix}txt_mode_blocks."),
        "depth_single_blocks": count_blocks(keys, f"{key_prefix}single_blocks."),
        # mlp_in carries the SwiGLU gate and up halves
        "mlp_hidden": shape("single_blocks.0.mlp_in.weight")[0] // 2,
        "in_channels": shape("final_layer.image.linear.weight")[0],
        "text_dim": TEXT_DIM,
        # the INT8 ConvRot build fuses q, k, v into one projection (the NVFP4 one keeps them apart)
        "fused_qkv": f"{key_prefix}single_blocks.0.qkv.weight" in state_dict,
        # the distilled checkpoint (a flow map) embeds the time each step jumps to
        "flow_map": f"{key_prefix}{FLOW_MAP_KEY}" in state_dict,
    }
