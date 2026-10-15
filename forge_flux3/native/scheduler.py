# FLUX 3 Image Klein support for Forge Neo
# The "FLUX.3" scheduler: upstream's schedule (src/flux3/sampling.timesteps), the time shift applied to
# linspace(1, 0, steps + 1), so the sigmas are exactly the ones upstream samples at. Forge's Simple scheduler gets
# close but rounds to its 1000-step grid (16 steps do not divide it).

import torch

# upstream's schedule: t' = alpha t / (1 + (alpha - 1) t) on linspace(1, 0) with alpha 6.93, which is Forge's
# discrete-flow time shift (ComfyUI writes it as the Flux exponential shift, mu = ln 6.93)
SHIFT = 6.93

NAME = "flux3"
LABEL = "FLUX.3"
ALIASES = ("Flux3", "flux.3")


def shift_of(inner_model) -> float:
    """The model's time shift, read back from its sigma grid (sigma(t) = a t / (1 + (a - 1) t) at t = (i + 1) / N)."""
    sigmas = getattr(inner_model, "sigmas", None)
    if sigmas is None or len(sigmas) < 4:
        return SHIFT
    i = len(sigmas) // 2 - 1
    t = (i + 1) / len(sigmas)
    s = float(sigmas[i])
    if not 0.0 < s < 1.0:
        return SHIFT
    return s * (1.0 - t) / (t * (1.0 - s))


def schedule(n: int, shift: float = SHIFT) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    return (shift * t / (1.0 + (shift - 1.0) * t)).float()


def sigmas(n, sigma_min=None, sigma_max=None, inner_model=None, device="cpu", **kwargs):
    return schedule(n, shift_of(inner_model)).to(device)
