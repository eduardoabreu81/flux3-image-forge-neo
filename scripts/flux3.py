# FLUX 3 Image Klein support for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup, before the UI and before any model loads, which is
# when the model gets registered in the checkpoint list. Everything runs in the module scope; any failure leaves
# Forge Neo unchanged and says so in the console.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENABLED = False

try:
    from forge_flux3.native import compat

    _problems = compat.check()
    if _problems:
        compat.report(_problems)
    else:
        from forge_flux3.native import patches

        patches.apply()
        ENABLED = True
except Exception:
    print("[FLUX 3] failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()
