# FLUX 3 Image Klein support for Forge Neo
# A "flux3" UI preset, added at runtime: PresetArch gains a member and the preset's options are registered through
# Forge Neo's own presets.register, so they match every other preset. Same mechanism as the Iris 3B extension.

from enum import Enum

from modules_forge import presets

PRESET = "flux3"

# the distilled checkpoint's defaults (upstream generate.py: 16 steps, no CFG); the base checkpoint wants
# 50 steps at CFG 3.5, set by hand
SAMPLER = "Euler"
SCHEDULER = "FLUX.3"
FALLBACK_SCHEDULER = "Simple"
STEPS = 16
CFG = 1.0


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return key.startswith(f"{PRESET}_") or key.endswith(f"_{PRESET}")


def register() -> None:
    arch = _add_enum_member(presets.PresetArch, PRESET)

    from modules import sd_schedulers

    presets.SAMPLERS[arch] = SAMPLER
    # without the FLUX.3 scheduler (it failed to register), Simple: the same shifted grid, rounded to 1000 steps
    presets.SCHEDULERS[arch] = SCHEDULER if SCHEDULER in sd_schedulers.schedulers_map else FALLBACK_SCHEDULER
    presets.STEPS[arch] = STEPS
    presets.CFG[arch] = CFG

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
