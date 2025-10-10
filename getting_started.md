## Getting Started

Although sample nodes and a workflow can be found within ComfyUI, for best results we recommend the below process:

### Installation

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
2. Install a t5 text encoder. We recommend [google_t5-v1_1-xxl_encoderonly](https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/tree/main). You can install it using ComfyUI Model Manager.

## Example Workflows

Select a workflow from the examples and download it. 

Note that to run the example workflows below, you need to have some additional custom nodes, like [ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite) and others, installed. You can add them by pressing "Install Missing Custom Nodes" button in ComfyUI Manager.

### Long Video Generation
🧩 [Image to Video Long Video](example_workflows/ltxv-13b-i2v-long-multi-prompt.json): Long video generation with support for multiple prompts along the video duration.<br>
🧩 [Video to Video Long Video](example_workflows/ltxv-13b-v2v-long-depth.json): Long video-to-video generation. Given a guiding video—such as depth, pose, or edges—the flow generates a new video.

### Video Upscaling
🧩 [Video Upscaling](example_workflows/ltxv-13b-upscale.json): Upscales and adds fine details to any given video, increasing its spatial resolution by 2×.

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
