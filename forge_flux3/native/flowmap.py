# FLUX 3 Image Klein support for Forge Neo
# The distilled checkpoint is a flow map: each step also embeds the time it jumps to, the next sigma of the
# schedule (upstream sampling.denoise_flow_map). Forge's samplers hand the model only the current sigma, so the
# extension records every schedule Forge builds (KDiffusionSampler.get_sigmas, patched in patches.py) and the DiT
# looks the next sigma up here, as ComfyUI does from transformer_options["sample_sigmas"].

import logging

logger = logging.getLogger("forge_flux3")

_schedule: list[float] | None = None
_warned = False


def record(sigmas) -> None:
    global _schedule
    _schedule = [float(s) for s in sigmas]


def clear() -> None:
    global _schedule
    _schedule = None


def target(sigma: float) -> float:
    """The sigma the step at ``sigma`` lands on: the next entry of the recorded schedule (the last one is 0)."""
    global _warned
    if not _schedule:
        if not _warned:
            # every Forge k-diffusion sampler goes through get_sigmas; a sampler that does not still runs, as a
            # one-step jump to the clean image at every step
            logger.warning("[FLUX 3] no sigma schedule recorded for the distilled checkpoint; "
                           "each step jumps to sigma 0")
            _warned = True
        return 0.0
    i = min(range(len(_schedule)), key=lambda j: abs(_schedule[j] - sigma))
    return _schedule[min(i + 1, len(_schedule) - 1)]
