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
| [LTX-2.5_ICLoRA_SDR_to_HDR_Distilled.json](./HDR_workflows/LTX-2.5_ICLoRA_SDR_to_HDR_Distilled.json) | **SDR→HDR**: SDR MP4 → ACEScct IDT → HDR IC-LoRA → EXR + HLG. |
| [LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json](./LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json) | Generate from a **reference sheet** (characters, props, wardrobe, location laid out in one image). Describe each element in the prompt by its place on the sheet. |
| [LTX-2.5_ICLoRA_Motion_Track_Distilled.json](./LTX-2.5_ICLoRA_Motion_Track_Distilled.json) | Image-to-video with **sparse motion tracks** you draw on the first frame. Optional still as the opening frame. |
| [LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json](./LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json) | Fill masked regions of a reference video (two-stage). Source audio can stay frozen. |
| [LTX-2.5_ICLoRA_Inpaint_HDR_Two_Stage_Distilled.json](./HDR_workflows/LTX-2.5_ICLoRA_Inpaint_HDR_Two_Stage_Distilled.json) | Same inpaint, but **HDR→HDR**: EXR sequence + mask video → EXR + HLG out. |
| [LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json](./LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json) | Extend the canvas of a reference video (two-stage). Same in/outpaint LoRA as inpaint. |
| [LTX-2.5_I2V_HDR_Two_Stage_Distilled.json](./HDR_workflows/LTX-2.5_I2V_HDR_Two_Stage_Distilled.json) | **HDR still → HDR video**: EXR still → float32 VAE → EXR + HLG out. |

### Tiled Fusion (large-canvas IC-LoRA)

These keep one latent canvas and fuse overlapping spatial tiles after every denoise step (`LTXVTiledFusionSampler`). Use them when the canvas is larger than the LoRA window. They share the same subgraph layout as the other 2.5 graphs (Load Models, Inputs, Preprocess, Generate, Decode). **Get Tiling Sizes** and the canvas resize live inside **Preprocess**; the size combos (`tile_size` / `initial_canvas_size` = qHD / HD / FullHD, `output_size` = FullHD / 4K / 8K) are widgets on that subgraph. Source frames come from Load Video, or from Load EXR still on the native HDR→HDR graph. The node emits model-legal canvases (multiples of 32, `tile_frames` = 8n+1); a **Tiling sizes** preview sits below Preprocess. There is no first-frame still / `use image input` / I2V bypass. `use_tiled_encode` on the guide node stays **false**; `use_streaming` is on so each temporal window gets a fresh IC-LoRA encode.

Upscale prompts should describe **look and style** (lighting, palette, film stock, sharpness, grade), not named objects, people, or props — anything you name can stamp into every spatial tile after upscale. Native and HDR: the stage-1 prompt describes the **change** you want on the input video or HDR plate (plus enhance / API / [prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide)); the stage-2 prompt is look/style only, because only that pass uses spatial tiling. The negative prompt is used as written.

| Workflow | What it does |
| -------- | ------------ |
| [LTX-2.5_V2V_TiledFusion_Upscale.json](./LTX-2.5_V2V_TiledFusion_Upscale.json) | **V2V upscale.** Preprocess resizes the guide to `output_size` (FullHD / 4K / 8K), then 8 fused steps with the detail refiner. Tiles come from `tile_size` (HD). |
| [LTX-2.5_V2V_TiledFusion_Native_4K_8K.json](./LTX-2.5_V2V_TiledFusion_Native_4K_8K.json) | **V2V native / dynamic 4K/8K.** Stage 1 at `initial_canvas_size` FullHD (8 steps, Day-To-Night IC-LoRA). Latent ×2 (×4 when `output_size` is 8K), then 3 fused steps with a fresh prompt and the original clip resized to output. Optional second IC-LoRA on stage 2. |
| [LTX-2.5_V2V_TiledFusion_SDR_to_HDR.json](./LTX-2.5_V2V_TiledFusion_SDR_to_HDR.json) | Same recipe as Native, with the HDR IC-LoRA. SDR MP4 → ACEScct, then `LTXVHDRDecodePostprocess` (EXR + HLG). |
| [LTX-2.5_V2V_TiledFusion_HDR_to_HDR.json](./LTX-2.5_V2V_TiledFusion_HDR_to_HDR.json) | **Native HDR→HDR refine.** EXR still (ACEScct, float32 VAE, `img_compression` 0) with the detail refiner instead of the HDR IC-LoRA. Same 4K/8K tiled ladder; EXR + HLG out. |

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
│  ├─ Upgrade SDR footage to HDR (ACEScct IC-LoRA)
│  │    → HDR_workflows/LTX-2.5_ICLoRA_SDR_to_HDR_Distilled.json
│  ├─ Upscale an existing clip at full HD / 4K / 8K (tiled fusion)
│  │    → LTX-2.5_V2V_TiledFusion_Upscale.json
│  ├─ Viewport-style 4K/8K ladder (full HD, then ×2 / ×4)
│  │    → LTX-2.5_V2V_TiledFusion_Native_4K_8K.json
│  ├─ Upgrade SDR footage to HDR (ACEScct IC-LoRA)
│  │    → LTX-2.5_V2V_TiledFusion_SDR_to_HDR.json (tiled 4K/8K ladder)
│  │    → ../2.3/LTX-2.3_ICLoRA_HDR_Distilled.json (single-stage)
│  ├─ Refine HDR stills / EXR (native ACEScct I/O, refine LoRA)
│  │    → LTX-2.5_V2V_TiledFusion_HDR_to_HDR.json
│  ├─ Fill a masked region
│  │    → LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json (SDR)
│  │    → HDR_workflows/LTX-2.5_ICLoRA_Inpaint_HDR_Two_Stage_Distilled.json (HDR EXR)
│  └─ Grow the frame / canvas
│       → LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json
│
Do you have stills rather than a video?
├─ YES → HDR EXR still → video?
│  ├─ YES → HDR_workflows/LTX-2.5_I2V_HDR_Two_Stage_Distilled.json
│  └─ NO  → A sheet of characters / props / locations?
│     ├─ YES → LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json
│     └─ NO  → Draw motion paths on a first frame?
│        ├─ YES → LTX-2.5_ICLoRA_Motion_Track_Distilled.json
│        └─ NO  → Text-to-video with an optional opening still
│                 (see below)
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
| [SDR→HDR IC-LoRA](./HDR_workflows/LTX-2.5_ICLoRA_SDR_to_HDR_Distilled.json) | 1 | — | SDR video (+ frozen audio) | SDR→ACEScct HDR; EXR + HLG out |
| Ingredients | 1 | — | Reference sheet | Cast / props / location from one image |
| Motion Track | 1 | — | Image + drawn tracks | Directed motion from a still |
| Inpaint two-stage | 2 | 2× spatial | Video + mask (+ frozen audio) | Replacing a region |
| Outpaint two-stage | 2 | 2× spatial | Video + target size (+ frozen audio) | Extending the frame |
| [I2V HDR two-stage](./HDR_workflows/LTX-2.5_I2V_HDR_Two_Stage_Distilled.json) | 2 | 2× spatial | EXR still (ACEScct) | HDR still → HDR video; EXR + HLG out |
| [Inpaint HDR two-stage](./HDR_workflows/LTX-2.5_ICLoRA_Inpaint_HDR_Two_Stage_Distilled.json) | 2 | 2× spatial | EXR sequence + mask video | HDR inpaint; EXR + HLG out |
| Tiled Fusion upscale | 1 | — | Source video as IC-LoRA guide | Detail-refine at full HD / 4K / 8K |
| Tiled Fusion native 4K/8K | 2 | 2× or 4× latent | Stage 1 Day-To-Night + stage 2 original@x2 | Large-canvas V2V ladder |
| Tiled Fusion SDR→HDR | 2 | 2× or 4× latent | Same as native, HDR IC-LoRA | SDR→ACEScct 4K/8K; EXR + HLG |
| Tiled Fusion HDR→HDR | 2 | 2× or 4× latent | HDR EXR still + refine IC-LoRA | Native ACEScct refine; EXR + HLG |

Native HDR I/O: [LTX-2.5_V2V_TiledFusion_HDR_to_HDR.json](./LTX-2.5_V2V_TiledFusion_HDR_to_HDR.json) — `LTXVLoadEXRSequence` → ACEScct → `LTXVVAEForceFloat32` → refine IC-LoRA tiled fusion → `LTXVHDRDecodePostprocess` (`transfer=acescct`, optional EXR) → `LTXVSaveHLG` from `hdr_linear`. `img_compression = 0`. Same idea as pipelines `--hdr` on Distilled / IC-LoRA.

SDR→HDR IC-LoRA (separate from native EXR HDR): [LTX-2.3_ICLoRA_HDR_Distilled.json](../2.3/LTX-2.3_ICLoRA_HDR_Distilled.json) for a single-stage pass, or [LTX-2.5_V2V_TiledFusion_SDR_to_HDR.json](./LTX-2.5_V2V_TiledFusion_SDR_to_HDR.json) for the tiled 4K/8K ladder. `LoadVideo` → `LTXVSDRToHDRWorkingSpace` (`srgb_gamma`) → HDR IC-LoRA guide → float32 VAE decode → `LTXVHDRDecodePostprocess` (`transfer=acescct`, EXR=`acescct`) → HLG. Sampler **euler**; original audio frozen/muxed. Same idea as pipelines `hdr_ic_lora --text-embeddings`.

For native HDR, set **`img_compression = 0`** (Python `--hdr` EXR conditioning skips JPEG/CRF; CRF on ACEScct darkens highlights). Turn **prompt enhancer off** when matching goldens. I2V stage-2 distilled sigmas should be `0.909375, 0.725, 0.421875, 0.0` (not `0.85, …`). Inpaint keeps its own 2-step stage-2 schedule.

HDR graphs live under [`HDR_workflows/`](./HDR_workflows/).

Native HDR I/O: `LTXVLoadEXRSequence` → float32 VAE → `LTXVHDRDecodePostprocess` (`transfer=acescct`, optional EXR) → `LTXVSaveHLG` from `hdr_linear`. Same idea as pipelines `--hdr` on Distilled / IC-LoRA.

SDR→HDR IC-LoRA (separate from native EXR HDR): [LTX-2.5_ICLoRA_SDR_to_HDR_Distilled.json](./HDR_workflows/LTX-2.5_ICLoRA_SDR_to_HDR_Distilled.json) — `LoadVideo` → `LTXVSDRToHDRWorkingSpace` (`srgb_gamma`) → `Resize Image/Mask` (scale to multiple 32) → **fixed scene embeddings** (`LTXVLoadConditioning`, not a free-text prompt) → HDR IC-LoRA guide → float32 VAE decode → `LTXVHDRDecodePostprocess` (`transfer=acescct`, EXR=`acescg`) → HLG. Sampler **euler**; original audio frozen/muxed. Same idea as pipelines `hdr_ic_lora --text-embeddings`. Conditioning fps snaps to 60 when source fps is above 30 (playback fps unchanged for decode/export).

For native HDR, set **`img_compression = 0`** (Python `--hdr` EXR conditioning skips JPEG/CRF; CRF on ACEScct darkens highlights). Turn **prompt enhancer off** when matching goldens. I2V stage-2 distilled sigmas should be `0.909375, 0.725, 0.421875, 0.0` (not `0.85, …`). Inpaint keeps its own 2-step stage-2 schedule.

Python / pipeline equivalents of these ideas live in [pipeline-selection.md](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/pipeline-selection.md) in the LTX-2 repo.
