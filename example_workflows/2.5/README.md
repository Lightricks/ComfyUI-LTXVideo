# LTX-2.5 example workflows

These graphs generate video (and usually audio) with the **LTX-2.5 distilled** model. They share the same subgraph layout — load models, set inputs, preprocess, sample, decode — so you can switch files without relearning the graph.

All of them use the distilled transformer (`ltx-2.5-22b-distilled-transformer-bf16.safetensors`): a few sampling steps, meant for fast iteration. Two-stage graphs add a 2× spatial upscale and a short refine pass; single-stage graphs skip that.

Default values are tuned for lower-VRAM machines. If you have more memory, you can speed things up by increasing the **decode tile size** — see the Decode notes inside each graph (fewer, larger tiles run faster but need more VRAM).

## Getting started

This assumes ComfyUI is already installed. If not, see the [ComfyUI download page](https://www.comfy.org/download), the [ComfyUI setup guide](https://docs.ltx.io/open-source-model/integration-tools/comfy-ui), and the [system requirements](https://docs.ltx.io/open-source-model/getting-started/system-requirements).

1. Open ComfyUI.
2. Load a workflow from this folder (**Workflow** → **Open**, or drag the `.json` onto the canvas). The built-in ComfyUI **Templates** search for **LTX-2.5** is the same family of graphs if you prefer that entry point.
3. Open the **Workflow Overview** panel. On first use it lists **Missing Models**.
4. Click **Download all** to fetch the files into the right `ComfyUI/models/` folders. You only do this once — later runs reuse them.
5. Fill in the left-hand **Inputs** (prompt, optional image, duration, and anything the graph asks for such as audio or a reference video).
6. Click **Run**.

Weights also live on the [LTX-2.5 Hugging Face repo](https://huggingface.co/Lightricks/LTX-2.5) if you want to download them by hand. Each graph’s **Model Links** note lists the exact files it needs, including any IC-LoRA.

Prompting tips: [prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide). A beginner walkthrough of text-to-video: [Text-to-Video workflow](https://docs.ltx.io/open-source-model/usage-guides/text-to-video).

Frame count must be `1 + a multiple of 8`. Duration in the graphs is converted from fps × seconds and may land slightly off the number you typed.

## Workflows

### Generation

| Workflow | What it does |
| -------- | ------------ |
| [LTX-2.5_T2V_I2V_Two_Stage_Distilled.json](./LTX-2.5_T2V_I2V_Two_Stage_Distilled.json) | **Start here.** Text-to-video, optional first-frame image. Stage 1 at base resolution, 2× spatial upscale, then a 3-step refine. Generates video and audio together. |
| [LTX-2.5_T2V_I2V_Single_Stage_Distilled.json](./LTX-2.5_T2V_I2V_Single_Stage_Distilled.json) | Same text / image inputs, one distilled pass, no upscaler. Faster and lighter; lower spatial detail than two-stage. |
| [LTX-2.5_A2V_Two_Stage_Distilled.json](./LTX-2.5_A2V_Two_Stage_Distilled.json) | Audio-to-video. Encodes an input clip (with start + duration trim), **freezes** those tokens in both stages, and muxes the original waveform into the output (no audio decode). Optional first-frame image. |
| [LTX-2.5_T2A_Single_Stage_Distilled.json](./LTX-2.5_T2A_Single_Stage_Distilled.json) | Text-to-audio only. No video VAE or upscaler. |

### IC-LoRA (control and edit)

These keep the distilled 2.5 backbone and add an IC-LoRA so a guide (video, image sheet, tracks, or mask) steers generation.

| Workflow | What it does |
| -------- | ------------ |
| [LTX-2.5_ICLoRA_Union_Control_Distilled.json](./LTX-2.5_ICLoRA_Union_Control_Distilled.json) | Video-to-video from a **depth / canny / pose** annotator on a reference clip. Depth is wired by default; switch the annotator inside the graph. |
| [LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json](./LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json) | Video-to-video using the source frames as IC-LoRA guides. Ships with the Instant Shave LoRA as the example; swap the LoRA for other identity/edit LoRAs. Original audio is frozen through. |
| [LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json](./LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json) | Generate from a **reference sheet** (characters, props, wardrobe, location laid out in one image). Describe each element in the prompt by its place on the sheet. |
| [LTX-2.5_ICLoRA_Motion_Track_Distilled.json](./LTX-2.5_ICLoRA_Motion_Track_Distilled.json) | Image-to-video with **sparse motion tracks** you draw on the first frame. Optional still as the opening frame. |
| [LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json](./LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json) | Fill masked regions of a reference video (two-stage). Source audio can stay frozen. |
| [LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json](./LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json) | Extend the canvas of a reference video (two-stage). Same in/outpaint LoRA as inpaint. |

## Which workflow should I use?

```text
Do you only need audio (no picture)?
├─ YES → LTX-2.5_T2A_Single_Stage_Distilled.json
│
Do you have an audio file the video should follow?
├─ YES → LTX-2.5_A2V_Two_Stage_Distilled.json
│         (optional still as the first frame)
│
Do you have an existing video to edit or follow?
├─ YES → What kind of control?
│  ├─ Depth / edges / pose from the clip
│  │    → LTX-2.5_ICLoRA_Union_Control_Distilled.json
│  ├─ Keep the footage, change appearance (identity / style LoRA)
│  │    → LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json
│  ├─ Fill a masked region
│  │    → LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json
│  └─ Grow the frame / canvas
│       → LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json
│
Do you have stills rather than a video?
├─ YES → A sheet of characters / props / locations?
│  ├─ YES → LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json
│  └─ NO  → Draw motion paths on a first frame?
│     ├─ YES → LTX-2.5_ICLoRA_Motion_Track_Distilled.json
│     └─ NO  → Text-to-video with an optional opening still
│              (see below)
│
Text-to-video (optional image)?
├─ Want 2× spatial upscale and a refine pass?
│  └─ YES → LTX-2.5_T2V_I2V_Two_Stage_Distilled.json  (recommended)
└─ Want the fastest / lightest run?
   └─ YES → LTX-2.5_T2V_I2V_Single_Stage_Distilled.json
```

Two-stage is the better default when you care about spatial detail. Single-stage is for previews and tighter VRAM.

## Comparison

| Workflow | Stages | Upsample | Main conditioning | Best for |
| -------- | ------ | -------- | ----------------- | -------- |
| T2V / I2V two-stage | 2 | 2× spatial | Text + optional image | Default generation |
| T2V / I2V single-stage | 1 | — | Text + optional image | Fast previews |
| A2V two-stage | 2 | 2× spatial | Audio (frozen) + optional image | Video that must match a soundtrack |
| T2A single-stage | 1 | — | Text | Audio-only clips |
| Union Control | 1 | — | Reference video (depth / canny / pose) | Structure-following v2v |
| V2V IC-LoRA | 1 | — | Source video (+ frozen audio) | Appearance edits on existing footage |
| Ingredients | 1 | — | Reference sheet | Cast / props / location from one image |
| Motion Track | 1 | — | Image + drawn tracks | Directed motion from a still |
| Inpaint two-stage | 2 | 2× spatial | Video + mask (+ frozen audio) | Replacing a region |
| Outpaint two-stage | 2 | 2× spatial | Video + target size (+ frozen audio) | Extending the frame |

Python / pipeline equivalents of these ideas live in [pipeline-selection.md](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/pipeline-selection.md) in the LTX-2 repo.
