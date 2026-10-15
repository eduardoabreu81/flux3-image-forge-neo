# 🖼️ FLUX 3 Image Klein for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/flux3-image-forge-neo)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate with **[FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)** inside Forge Neo, using the checkpoint, preset and Generate button you already know - text-to-image, editing with up to 10 reference images and layout by bounding boxes, with the 16-step distilled model or the full base model, even on 8 GB GPUs with the INT8 files.

> [!Important]
> This extension requires an up-to-date **Forge Neo** (the `neo` branch). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Features](#-features)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.1.0 - First Release

- **FLUX 3 Image Klein in Forge Neo** - distilled and base checkpoints, in BF16, INT8 and NVFP4.
- **Reference images** - edit a picture, or combine up to 10 of them, as with FLUX.2 Klein.
- **flux3 UI preset** - sampler, schedule, steps and CFG ready to generate.
- **FLUX.3 schedule** - the exact steps of Black Forest Labs' own code.
- **Older GPUs** - works on cards without bf16 (RTX 20xx), tested on an 8 GB RTX 2070.

---

## 🎯 Features

### 🖼️ Native Generation

- Runs inside txt2img and img2img - no separate tab, no extra environment
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- Uses the same text encoder as Krea 2 and Iris 3B, and the same autoencoder as FLUX.2 - if you have them, you already have these files
- The distilled checkpoint makes an image in 16 steps without CFG; the base checkpoint, for LoRA training and research, uses 50 steps at CFG 3.5
- Loads the BF16, INT8 ConvRot (8 GB GPUs) and NVFP4 (RTX 50) files
- On an 8 GB RTX 2070 with the INT8 files: a 1024x1024 image in about 25 seconds with the distilled checkpoint, an edit with one reference in about a minute
- Works on GPUs without bf16 (RTX 20xx and older): the few values that outgrow float16 are kept in range, with no slowdown

### 🧩 Reference Images

- img2img with **Settings** → **[Klein] Enable Reference** on: the input picture becomes the reference, and the prompt describes the result
- More references with Forge Neo's **ImageStitch**, up to 10
- Write the prompt as a full description of the picture you want, with the change folded in - a short instruction such as "make it blue" gives a different framing

### 📐 Layout by Bounding Boxes

- Append the boxes as JSON to the prompt, exactly as in [Black Forest Labs' guide](https://docs.bfl.ai/flux_3/flux3_image_bounding_boxes) - each box `[top, left, bottom, right]` on a 0-1000 grid, tagged in the caption with `<id>`

### 🔌 LoRA

- FLUX 3 LoRAs through Forge's usual `<lora:name:weight>` syntax
- Works on the INT8 files too, where the q/k/v projections are fused into one

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Tells you which file is missing under **VAE / Text Encoder** instead of failing with a generic error
- Other models are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/flux3-image-forge-neo`
4. Click **Install** and restart the WebUI
5. Accept the license on [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) and place the files in the usual folders:

| Part | File | Folder |
| :--- | :--- | :--- |
| FLUX 3 checkpoint, distilled | `flux3_image_klein_sd_int8_convrot` (8 GB GPUs), `flux3_image_klein_sd_bf16` or `flux3_image_klein_sd_nvfp4` (RTX 50) | `models/Stable-diffusion` |
| FLUX 3 checkpoint, base (optional) | `flux3_image_klein_int8_convrot`, `flux3_image_klein_bf16` or `flux3_image_klein_nvfp4` | `models/Stable-diffusion` |
| Qwen3-VL 4B text encoder | `qwen3vl_4b_int8_convrot` or `qwen3vl_4b_bf16` | `models/text_encoder` |
| FLUX.2 autoencoder | `flux2-vae` | `models/VAE` |

Files with `_sd_` in the name are the distilled checkpoint; the others are the base checkpoint.

6. Pick the **flux3** UI preset, the checkpoint, and the text encoder and autoencoder under **VAE / Text Encoder**

---

## ⚙️ Recommended Settings

| Checkpoint | Sampler | Schedule type | Steps | CFG |
| :--- | :--- | :--- | :--- | :--- |
| **Distilled** (`_sd_`) | Euler | FLUX.3 | 16 | 1 |
| **Base** | Euler | FLUX.3 | 50 | 3.5 |

The **flux3** preset sets the distilled checkpoint's values. Leave the **negative prompt** empty, and use sides that are a multiple of 16.

---

## 💡 Tips

- Write full captions - framing, subject, background, lighting. Short prompts give generic pictures
- The distilled checkpoint is made for Euler and the FLUX.3 schedule; other samplers and schedules may give worse pictures
- With **[Klein] Enable Reference** on, every img2img input also becomes a reference - turn it off for plain img2img variations
- A batch of different prompts of different lengths fails in Forge Neo with this model, as with FLUX.2 Klein - generate them one at a time
- Before uninstalling, switch the UI preset to another one

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)** by Black Forest Labs - the model and its [reference code](https://github.com/black-forest-labs/flux-3-image-klein)
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous - the first implementation outside the reference code, and the INT8 file format
- **[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)** by the Qwen team - the text encoder

The FLUX 3 Image Klein weights are released under the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/LICENSE.md); this extension does not include them.

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/flux3-image-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/flux3-image-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
