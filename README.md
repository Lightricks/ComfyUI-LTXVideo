<div align="center">

# ComfyUI-LTXVideo

[![GitHub](https://img.shields.io/badge/LTXV-Repo-blue?logo=github)](https://github.com/Lightricks/LTX-Video)
[![Website](https://img.shields.io/badge/Website-LTXV-181717?logo=google-chrome)](https://www.lightricks.com/ltxv)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-Video)
[![LTXV Trainer](https://img.shields.io/badge/LTXV-Trainer%20Repo-9146FF)](https://github.com/Lightricks/LTX-Video-Trainer)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/ltx-video)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2501.00103)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/Mn8BRgUKKy)

</div>


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Workflows](#installation)
- [What's new](./docs/release-notes.md)


# Introduction

ComfyUI-LTXVideo is a collection of custom nodes for ComfyUI, designed to provide useful tools for working with the LTXV model.
The model itself is supported in the core ComfyUI [code](https://github.com/comfyanonymous/ComfyUI/tree/master/comfy/ldm/lightricks).
The main LTXVideo repository can be found [here](https://github.com/Lightricks/LTX-Video).


# Installation

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-LTXVideo` in the list of nodes and follow installation instructions.

### Manual installation

1. Install ComfyUI
2. Clone this repository to `custom-nodes` folder in your ComfyUI installation directory.
3. Install the required packages:

```bash
cd custom_nodes/ComfyUI-LTXVideo && pip install -r requirements.txt
```

For portable ComfyUI installations, run

```
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-LTXVideo\requirements.txt
```

### Models

1. Download [ltx-video-2b-v0.9.1.safetensors](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) from Hugging Face and place it under `models/checkpoints`.
2. Install one of the t5 text encoders, for example [google_t5-v1_1-xxl_encoderonly](https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/tree/main). You can install it using ComfyUI Model Manager.

# Workflows

Note that to run the example workflows, you need to have some additional custom nodes, like [ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite) and others, installed. You can do it by pressing "Install Missing Custom Nodes" button in ComfyUI Manager.

### Easy to use multi scale generation workflows

🧩 [Image to video mixed](example_workflows/ltxv13b-i2v-mixed-multiscale.json): mixed flow with full and distilled model for best quality and speed trade-off.<br>

### 13B model<br>
🧩 [Image to video](example_workflows/ltxv-13b-i2v-base.json)<br>
🧩 [Image to video with keyframes](example_workflows/ltxv-13b-i2v-keyframes.json)<br>
🧩 [Image to video with duration extension](example_workflows/ltxv-13b-i2v-extend.json)<br>
🧩 [Image to video 8b quantized](example_workflows/ltxv-13b-i2v-base-fp8.json)

### 13B distilled model<br>
🧩 [Image to video](example_workflows/13b-distilled/ltxv-13b-dist-i2v-base.json)<br>
🧩 [Image to video with keyframes](example_workflows/13b-distilled/ltxv-13b-dist-i2v-keyframes.json)<br>
🧩 [Image to video with duration extension](example_workflows/13b-distilled/ltxv-13b-dist-i2v-extend.json)<br>
🧩 [Image to video 8b quantized](example_workflows/13b-distilled/ltxv-13b-dist-i2v-base-fp8.json)

### ICLora
🧩 [Download workflow](example_workflows/ic_lora/ic-lora.json)

### Inversion

#### Flow Edit

🧩 [Download workflow](example_workflows/tricks/ltxvideo-flow-edit.json)<br>
![workflow](example_workflows/tricks/ltxvideo-flow-edit.png)

#### RF Edit

🧩 [Download workflow](example_workflows/tricks/ltxvideo-rf-edit.json)<br>
![workflow](example_workflows/tricks/ltxvideo-rf-edit.png)
