# ComfyUI-LTXVideo

[![GitHub](https://img.shields.io/badge/LTX-Repo-blue?logo=github)](https://github.com/Lightricks/LTX-2)
[![Website](https://img.shields.io/badge/Website-LTX-181717?logo=google-chrome)](https://ltx.io/model)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)
[![LTXV Trainer](https://img.shields.io/badge/LTX-Trainer%20Repo-9146FF)](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/ltx-2-playground/i2v)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv)](https://videos.ltx.io/LTX-2/grants/LTX_2_Technical_Report_compressed.pdf)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/ltxplatform)


A collection of powerful custom nodes that extend ComfyUI's capabilities for the LTX-2 video generation model.

LTX-2 is built into ComfyUI core ([see it here](https://github.com/comfyanonymous/ComfyUI/tree/master/comfy/ldm/lightricks)), making it readily accessible to all ComfyUI users. This repository hosts additional nodes and workflows to help you get the most out of LTX-2's advanced features.

**To learn more about LTX-2** See the [main LTX-2 repository](https://github.com/Lightricks/LTX-2) for model details and additional resources.


## Prerequisites
Before you begin using an LTX-2 workflow in ComfyUI, make sure you have:

* ComfyUI installed (Download here](https://www.comfy.org/download)
* CUDA-compatible GPU with 32GB+ VRAM
* 100GB+ free disk space for models and cache


## Quick Start 🚀

We recommend using the LTX-2 workflows available in Comfy Manager.

1. Open ComfyUI
2. Click the Manager button (or press Ctrl+M)
3. Select Install Custom Nodes
4. Search for “LTXVideo”
5. Click Install
6. Wait for installation to complete
7. Restart ComfyUI

The nodes will appear in your node menu under the “LTXVideo” category. Required models will be downloaded on first use.


## Example Workflows

The ComfyUI-LTXVideo installation includes several example workflows.
You can see them all at:
```
ComfyUI/custom_nodes/ComfyUI-LTXVideo/example_workflows/
```

LTX-2.3 Workflows:

* [`Text/image to video full/distilled model; single stage`](./example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json)
* [`Text/image to video distilled model; two stages (with upsampling)`](./example_workflows/2.3/LTX-2.3_T2V_I2V_Two_Stage_Distilled.json)
* [`IC-LoRA distilled model depth + human pose + edges`](./example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_Distilled.json)
* [`IC-LoRA distilled model I2V motion tracking`](./example_workflows/2.3/LTX-2.3_ICLoRA_Motion_Track_Distilled.json)
* [`IC-LoRA distilled model HDR`](./example_workflows/2.3/LTX-2.3_ICLoRA_HDR_Distilled.json)

Older Workflows (LTX-2.0):

* [`Text to video full model`](./example_workflows/2.0/LTX-2_T2V_Full_wLora.json)
* [`Text to video distilled model (Fast)`](./example_workflows/2.0/LTX-2_T2V_Distilled_wLora.json)
* [`Image to video full model`](./example_workflows/2.0/LTX-2_I2V_Full_wLora.json)
* [`Image to video distilled model (Fast)`](./example_workflows/2.0/LTX-2_I2V_Distilled_wLora.json)
* [`Video to video detailer`](./example_workflows/2.0/LTX-2_V2V_Detailer.json)
* [`IC-LoRA distilled model (depth + human pose + edges)`](./example_workflows/2.0/LTX-2_ICLoRA_All_Distilled.json)
* [`IC-LoRA distilled model with downscaled reference latents`](./example_workflows/2.0/LTX-2_ICLoRA_All_Distilled_ref0.5.json)

## Union IC-LoRA Model

We introduce a new **Union IC-LoRA** model that combines depth and edge (canny) control conditions into a single unified LoRA.

### Key Features

- **Unified Control**: A single LoRA that supports multiple control conditions (depth or edges).
- **Downsampled Latent Processing**: The union LoRA operates on a downsampled latent size, which reduces memory usage and significantly speeds up inference while maintaining quality.

### How It Works

The union LoRA is trained to understand and respond to both control signals (depth maps and edge maps) within a single model. The model learns to:

1. **Parse multiple conditions**: Identify which control signals are present in the input
2. **Process at reduced resolution**: Work on downsampled latents to improve efficiency

## HDR IC-LoRA

We provide an **HDR IC-LoRA** that generates linear HDR video encoded in ARRI LogC3, enabling workflows that output high-dynamic-range content suitable for grading and EXR export.

### Key Features

- **Linear HDR output**: The LoRA produces frames in LogC3-compressed space; the `LTXVHDRDecodePostprocess` node decodes these back to linear HDR values.
- **SDR preview + raw HDR**: The node outputs both a Reinhard-tonemapped SDR preview and the raw linear HDR tensor for downstream use.
- **EXR export**: Optionally writes the linear HDR frames as a 16/32-bit EXR image sequence. To enable EXR writing, set `OPENCV_IO_ENABLE_OPENEXR=1` in the environment before starting ComfyUI. The exported EXR sequence is best viewed in [DJV](https://github.com/grizzlypeak3d/DJV) (or [DJV for macOS](https://djv.en.uptodown.com/mac/download)).

## Required Models

Download the following models:

**LTX-2.3 Model Checkpoint** - Choose and download one of the models to `COMFYUI_ROOT_FOLDER/models/checkpoints` folder.
  * [`ltx-2.3-22b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors)
  * [`ltx-2.3-22b-distilled-1.1.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-1.1.safetensors)

**Spatial Upscaler** - Required for current two-stage pipeline implementations in this repository. Download to `COMFYUI_ROOT_FOLDER/models/latent_upscale_models` folder.
  * [`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors)
  * [`ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors)

**Temporal Upscaler** - Required for current two-stage pipeline implementations in this repository. Download to `COMFYUI_ROOT_FOLDER/models/latent_upscale_models` folder.
  * [`ltx-2.3-temporal-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors)

**Distilled LoRA** - Required for current two-stage pipeline implementations in this repository (except DistilledPipeline and ICLoraPipeline). Download to `COMFYUI_ROOT_FOLDER/models/loras` folder.
  * [`ltx-2.3-22b-distilled-lora-384-1.1.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors)

**Gemma Text Encoder** Download all files from the repository to `COMFYUI_ROOT_FOLDER/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized`.
  * [`Gemma 3`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)

**LoRAs** Choose and download to `COMFYUI_ROOT_FOLDER/models/loras` folder.
  * [`ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/blob/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors)
  * [`ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/blob/main/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors)
  * [`ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR)
  * [`ltx-2-19b-ic-lora-detailer.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/blob/main/ltx-2-19b-ic-lora-detailer.safetensors)
  * [`ltx-2-19b-ic-lora-pose-control.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/blob/main/ltx-2-19b-ic-lora-pose-control.safetensors)
  * [`ltx-2-19b-lora-camera-control-dolly-in.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/blob/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors)
  * [`ltx-2-19b-lora-camera-control-dolly-left.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/blob/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors)
  * [`ltx-2-19b-lora-camera-control-dolly-out.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/blob/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors)
  * [`ltx-2-19b-lora-camera-control-dolly-right.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/blob/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors)
  * [`ltx-2-19b-lora-camera-control-jib-down.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/blob/main/ltx-2-19b-lora-camera-control-jib-down.safetensors)
  * [`ltx-2-19b-lora-camera-control-jib-up.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/blob/main/ltx-2-19b-lora-camera-control-jib-up.safetensors)
  * [`ltx-2-19b-lora-camera-control-static.safetensors`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/blob/main/ltx-2-19b-lora-camera-control-static.safetensors)


## Advanced Techniques

### Low VRAM
* For systems with low VRAM you can use the model loader nodes from [low_vram_loaders.py](./low_vram_loaders.py). Those nodes ensure the correct order of execution and perform the model offloading such that generation fits in 32 GB VRAM.
* Use --reserve-vram ComfyUI parameter: `python -m main --reserve-vram 5` (or other number in GB).
* For complete information about using LTX-2 models, workflows, and nodes in ComfyUI, please visit our [Open Source documentation](https://docs.ltx.video/open-source-model/integration-tools/comfy-ui).
