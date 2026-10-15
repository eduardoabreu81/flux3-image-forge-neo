# FLUX 3 Image Klein support for Forge Neo
# The FLUX 3 diffusion engine: text-to-image, image-to-image and editing with reference images.
#
# Follows Forge Neo's own FLUX.2 Klein engine (backend/diffusion_engine/flux2.py): the FLUX.2 autoencoder, and
# reference images through dynamic_args.ref_latents under the same "[Klein] Enable Reference" setting. With it on,
# the img2img input becomes the first reference; ImageStitch adds the others. The DiT reads them in forward.

import torch
from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from . import lora
from .model import Flux3
from .text_engine import PAD_MULTIPLE, Flux3TextEngine


def _references_enabled() -> bool:
    from modules.shared import opts

    return bool(getattr(opts, "klein_do_reference", True))


class Flux3Engine(ForgeDiffusionEngine):
    matched_guesses = [Flux3]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]},
                    tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.text_engine_flux3 = Flux3TextEngine(text_encoder=clip.cond_stage_model.qwen3vl_4b,
                                                 tokenizer=clip.tokenizer.qwen3vl_4b)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # not FLUX.2: its Klein flag would swap the Automatic schedule for the resolution-dependent Flux2 one
        dynamic_args.klein = False
        self.is_flux3 = True
        # Forge's LoRA extension is imported by now, whatever the extensions' load order was
        lora.ensure_installed()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            if not _references_enabled():
                dynamic_args.ref_latents.clear()
            else:
                references = [*self.ref_latents]
                if self.ini_latent is not None:
                    references.insert(0, self.ini_latent)
                    self.ini_latent = None
                dynamic_args.ref_latents = references.copy()

        return self.text_engine_flux3(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = self.text_engine_flux3.tokenize(prompt)
        return token_count, -(-token_count // PAD_MULTIPLE) * PAD_MULTIPLE

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        samples = super().encode_first_stage(x)

        if _references_enabled():
            sample = samples[0].detach().clone().unsqueeze(0).cpu()
            if dynamic_args.is_referencing:
                self.ref_latents.append(sample)
            else:
                self.ini_latent = sample

        return samples
