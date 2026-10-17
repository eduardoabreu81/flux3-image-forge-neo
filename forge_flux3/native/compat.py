# FLUX 3 Image Klein support for Forge Neo
# Checks that the installed Forge Neo has every piece the FLUX 3 backend builds on, before anything is patched.
# Same mechanism as the Iris 3B extension's compat.py.

import importlib
import inspect

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.args": ("dynamic_args",),
    "backend.attention": ("attention_function",),
    "backend.memory_management": ("load_model_gpu", "cast_to"),
    "backend.loader": ("load_huggingface_component", "possible_models"),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.nn.vae": ("AutoencoderKLFlux2",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "backend.patcher.clip": ("CLIP",),
    "backend.patcher.lora": ("model_lora_keys_unet",),
    "backend.patcher.unet": ("UnetPatcher",),
    "backend.patcher.vae": ("VAE",),
    "backend.sampling.condition": ("ConditionCrossAttn",),
    "backend.text_processing._comfy": ("SDClipModel", "SDTokenizer", "INF"),
    "backend.text_processing.emphasis": ("EmphasisNone", "uses_emphasis"),
    "modules.sd_samplers_kdiffusion": ("KDiffusionSampler",),
    "modules.sd_schedulers": ("Scheduler", "all_schedulers", "schedulers", "schedulers_map"),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "register"),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
    "huggingface_guess.latent": ("Flux2",),
}

# the transformer goes through the Krea 2 branch of the loader, the autoencoder and text encoder through theirs
LOADER_BRANCHES = ("Krea2Transformer2DModel", "AutoencoderKLFlux2", "Qwen3VLModel")


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend.args import dynamic_args

        missing = [n for n in ("can_reference", "is_referencing", "ref_latents") if not hasattr(dynamic_args, n)]
        if missing:
            problems.append(f"backend.args.dynamic_args is missing {', '.join(missing)}")

        from backend import loader

        # the whole module, not load_huggingface_component: an extension that loaded first has already put its own wrapper
        # there, and a wrapper's source holds none of Forge's branches
        source = inspect.getsource(loader)
        missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
        if missing:
            problems.append(f"the model loader has no {', '.join(missing)} branch")

    return problems


def report(problems: list[str]) -> None:
    print("[FLUX 3] this Forge Neo is too old or too new for the extension, so it stays disabled "
          "(Forge Neo itself keeps working as usual):")
    for problem in problems:
        print(f"          {problem}")
