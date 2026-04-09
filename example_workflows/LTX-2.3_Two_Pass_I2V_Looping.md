# LTX-2.3 Two-Pass I2V Looping — Arbitrary-Length Video

## Overview

Two-pass image-to-video workflow for generating videos of any duration using
LTX-2.3 (22B) with `LTXVLoopingSampler`. A batch of soft guiding images at
tile boundaries maintains subject and scene continuity across temporal tiles.

**Stage 1** generates video+audio at base resolution (~544p) with temporal
tiling.  
**Stage 2** spatially upscales 2x and refines at high resolution with the
same prompts.

**Model:** `ltx-2.3-22b-dev.safetensors` + distilled LoRA (0.5 strength)  
**Text encoder:** Gemma 3 12B  
**No detailer LoRA required** (none exists for 2.3)

---

## Data Flow

```
LoadImage (reference)
    |
    +-- LTXVPreprocess --> Stage 1 I2V Cond --> Stage 1 AV Concat
    |                                               |
    +-- VAEEncode (negative_index_latents)     LTXVLoopingSampler (Stage 1)
    |                                               |
    +-- ResizeImage (for stage 2)          LTXVSeparateAVLatent
                                             |              |
                                     LTXVLatentUpsampler    |
                                             |              |
                                     Stage 2 I2V Cond      |
                                             |              |
                                     Stage 2 AV Concat------+
                                             |
                                     LTXVLoopingSampler (Stage 2)
                                          (AV refinement)
                                             |
                                     LTXVSeparateAVLatent
                                       |              |
                                  VAEDecodeTiled   AudioVAEDecode
                                       |              |
                                     CreateVideo------+
                                         |
                                      SaveVideo
```

**Audio refinement:** Both stages process AV latents jointly. Stage 1
generates video and audio from scratch. Stage 2 receives the upscaled video
+ stage 1 audio as an AV latent and refines both together — the looping
sampler initializes each tile's audio from the input audio data (not zeros),
so the model refines lipsync and audio-visual coherence at the higher
resolution.  This matches the behaviour of the standard two-stage workflow
using `SamplerCustomAdvanced`.

---

## Key Parameters

### Stage 1 — Generate

| Parameter | Value | Notes |
|---|---|---|
| Resolution | 960x544 | Base res; 2x upscale yields ~1920x1088 |
| temporal_tile_size | 128 | Pixel frames per tile |
| temporal_overlap | 24 | Overlap between tiles |
| temporal_overlap_cond_strength | 0.5 | How strongly previous tile conditions next |
| cond_image_strength | 1.0 | Guiding image influence |
| adain_factor | 0.15 | Prevents color drift across tiles |
| horizontal_tiles / vertical_tiles | 1 / 1 | No spatial tiling at 544p |
| Sigmas | `1.0, 0.994, 0.988, 0.981, 0.975, 0.909, 0.725, 0.422, 0.0` | Distilled schedule |
| Sampler | euler_ancestral_cfg_pp | Good for generation |
| CFG | 1 | With distilled LoRA |

### Stage 2 — Refine

| Parameter | Value | Notes |
|---|---|---|
| Resolution | ~1920x1088 | 2x from stage 1 |
| temporal_tile_size | 128 | Same as stage 1 |
| temporal_overlap | 24 | Same |
| horizontal_tiles / vertical_tiles | 2 / 1 | Spatial tiling for memory |
| adain_factor | 0.0 | Not needed for refinement |
| Sigmas | `0.85, 0.725, 0.422, 0.0` | Low — refinement only |
| Sampler | euler_cfg_pp | Deterministic for refinement |
| CFG | 1 | With distilled LoRA |

---

## Guiding Images

### Default: Reference image at frame 0

By default, `optional_cond_images` is connected to the preprocessed reference
image and `optional_cond_image_indices` is set to `"0"`. This provides soft
I2V conditioning at the first frame only.

For global subject anchoring across ALL tiles, `optional_negative_index_latents`
is connected to the VAE-encoded reference image. This attaches the reference
with negative positional embeddings to every tile, providing identity context
without pinning a specific frame position.

### Transition images at tile boundaries

To guide content at specific points in the video:

1. Batch multiple images using `ImageBatch` (or any node that produces an
   IMAGE batch).
2. Set `optional_cond_image_indices` to the pixel frame positions where each
   image should appear, e.g. `"0, 104, 208"`.
3. Connect the batch to `optional_cond_images`.

**The number of images must match the number of indices.**

Frame positions for tile boundaries with `tile_size=128, overlap=24`:

| Tiles | Total frames | Indices |
|---|---|---|
| 2 | 241 | `0, 104` |
| 3 | 345 | `0, 104, 208` |
| 4 | 449 | `0, 104, 208, 312` |
| N | 128 + (N-1)*104 + 1 | `0, 104, 208, ..., (N-1)*104` |

Frame indices must be divisible by 8 (except 0). The formula for new content
start per tile is: `tile_size - overlap = 128 - 24 = 104`.

### Per-tile prompts

Connect `LTXVMultiPromptProvider` to `optional_positive_conditionings` on the
Stage 1 looping sampler. Prompts are separated by `|`:

```
A woman walks through a meadow | She reaches a stream | She crosses a bridge
```

Each prompt maps to one temporal tile. If more tiles than prompts, the last
prompt repeats. Use the same provider for Stage 2 to keep prompts aligned.

---

## Duration and Tile Count

| Duration (24fps) | Pixel frames | Tiles (128/24) | Est. time (Strix Halo) |
|---|---|---|---|
| 5 sec | 121 | 1 | ~5 min |
| 10 sec | 241 | 3 | ~15 min |
| 30 sec | 721 | 7 | ~45 min |
| 1 min | 1441 | 14 | ~1.5 hr |
| 5 min | 7201 | 69 | ~7 hr |

Frame count must satisfy `8n+1` (e.g. 121, 241, 361...).  
Times are rough estimates for both stages combined on Strix Halo 128GB.

---

## Strix Halo / Unified Memory Notes

See `LTX-2_V2V_Detailer.md` for full Strix Halo tuning.

- Stage 1 at 544p with 1x1 spatial tiles fits easily.
- Stage 2 at ~1088p needs 2x1 spatial tiling (set in the workflow).
  Increase to 2x2 if OOM occurs.
- Add `LTXVChunkFeedForward` (from KJNodes) between the LoRA loader and
  the guiders if stage 2 still OOMs. Set `chunks=2, dim_threshold=4096`.
- VAE decode uses `LTXVSpatioTemporalTiledVAEDecode` with `spatial_tiles=6`.
  Increase to 8 if needed.

---

## Regenerating the Workflow

The workflow JSON is generated by the companion script:

```bash
cd custom_nodes/ComfyUI-LTXVideo/example_workflows
python generate_two_pass_i2v_looping.py
```

Edit the script to change default parameters, add nodes, or adjust layout.
