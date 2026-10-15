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


if ENABLED:
    from modules import scripts

    class Flux3LayoutScript(scripts.Script):
        """The FLUX 3 Layout panel: boxes drawn on a frame (javascript/flux3_layout.js) and written into the prompt.
        It only edits the prompt text, so it adds no generation arguments."""

        sorting_priority = 11
        create_group = False

        def title(self):
            return "FLUX 3 Layout"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            import gradio as gr

            tab = "img2img" if is_img2img else "txt2img"
            with gr.Accordion("FLUX 3 Layout", open=False, elem_id=f"flux3_layout_{tab}_accordion"):
                gr.HTML(f'<div class="flux3-layout" data-tab="{tab}"></div>')
            return []
