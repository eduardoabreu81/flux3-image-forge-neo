# FLUX 3 Image Klein support for Forge Neo
# Runtime hooks into Forge Neo's loader and sampler. Every hook hands anything that is not FLUX 3 to the original
# function untouched; Forge Neo itself stays unchanged. Follows the Iris 3B extension's patches.py.

import logging

import torch
from backend import loader
from backend.nn import krea
from huggingface_guess import detection, model_list

from . import flowmap, model
from .checkpoints import check_text_encoder, check_vae, clean_transformer_state_dict
from .detect import TEXT_DIM
from .engine import Flux3Engine
from .transformer import Flux3DiT

logger = logging.getLogger("forge_flux3")

_applied = False


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_components() -> None:
    original = loader.load_huggingface_component

    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.Flux3):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "Flux3Transformer2DModel":
            if isinstance(state_dict, dict):
                clean_transformer_state_dict(state_dict)
            # the Krea 2 branch is Forge's generic DiT load: dtype, quantization (INT8 ConvRot, NVFP4) and device
            # handling; it imports backend.nn.krea.SingleStreamDiT inside the function, so the swap routes the
            # build to us
            original_krea = krea.SingleStreamDiT
            krea.SingleStreamDiT = Flux3DiT
            try:
                return original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            finally:
                krea.SingleStreamDiT = original_krea
        if cls_name == "Qwen3VLModel":
            # the Forge loads the 4B natively for Krea 2; the hook only explains a missing text encoder
            check_text_encoder(state_dict)
        if cls_name == "AutoencoderKLFlux2":
            check_vae(state_dict)

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_cond_concat() -> None:
    # Forge batches the positive and negative prompts together by repeating the shorter text up to a common length;
    # the FLUX 3 DiT attends to every text token, so a repeated prompt is a different prompt. Conditionings of
    # different lengths run separately instead (ComfyUI does the same). Only FLUX 3 text has 20480 features.
    from backend.sampling.condition import ConditionCrossAttn

    original = ConditionCrossAttn.can_concat

    def can_concat(self, other):
        a, b = self.cond.shape, other.cond.shape
        if a[-1] == TEXT_DIM and a != b:
            return False
        return original(self, other)

    ConditionCrossAttn.can_concat = can_concat


def _hook_sigmas() -> None:
    # the distilled checkpoint needs each step's next sigma (flowmap.py)
    from modules.sd_samplers_kdiffusion import KDiffusionSampler

    original = KDiffusionSampler.get_sigmas

    def get_sigmas(self, p, steps):
        sigmas = original(self, p, steps)
        flowmap.record(sigmas)
        return sigmas

    KDiffusionSampler.get_sigmas = get_sigmas


def _hook_references() -> None:
    # ImageStitch (Forge's multi-reference input) only runs for the models dynamic_args.can_reference names, keyed on
    # loader flags; FLUX 3 cannot borrow FLUX.2's "klein" flag, which also swaps the Automatic schedule for Flux2's
    from backend.args import dynamic_args

    original = dynamic_args.can_reference

    def can_reference(cls) -> bool:
        if original():
            return True
        from modules import shared

        return bool(getattr(shared.sd_model, "is_flux3", False))

    dynamic_args.can_reference = classmethod(can_reference)


def _register_scheduler() -> None:
    from modules import sd_schedulers, shared

    from . import scheduler as flux3_scheduler

    if flux3_scheduler.LABEL in sd_schedulers.schedulers_map:
        return
    entry = sd_schedulers.Scheduler(flux3_scheduler.NAME, flux3_scheduler.LABEL, flux3_scheduler.sigmas,
                                    need_inner_model=True, aliases=list(flux3_scheduler.ALIASES))
    sd_schedulers.all_schedulers.append(entry)
    if entry.label not in shared.opts.hide_schedulers:
        sd_schedulers.schedulers.append(entry)
    sd_schedulers.schedulers_map.update({entry.name: entry, entry.label: entry,
                                         **{alias: entry for alias in (entry.aliases or [])}})


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.Flux3 not in model_list.models:
        model_list.models.append(model.Flux3)
    if Flux3Engine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, Flux3Engine)
    _hook_components()
    _hook_cond_concat()
    _hook_sigmas()
    _hook_references()
    try:
        _register_scheduler()
    except Exception as e:
        # the model still works: Simple gives the same shifted grid, rounded to Forge's 1000 steps
        logger.warning(f"[FLUX 3] could not add the FLUX.3 scheduler: {e}")
    # Forge's built-in LoRA extension loads before this one; when it does not, the engine retries at model load
    from . import lora

    lora.install()
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works: Euler, FLUX.3, 16 steps, CFG 1 by hand
        logger.warning(f"[FLUX 3] could not add the flux3 UI preset: {e}")

    _applied = True
    print(f"[FLUX 3] native backend enabled (torch {torch.__version__})")
