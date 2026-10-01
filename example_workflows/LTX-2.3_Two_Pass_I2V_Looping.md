# LTX-2.3 Two-Pass I2V Looping — Arbitrary-Length Video

## Overview

Two-pass image-to-video workflow for generating videos of any duration using
LTX-2.3 (22B) with `LTXVLoopingSampler`. The generated graph places a soft
reference image near the end of each temporal tile and pairs each late
reference with a per-tile prompt snippet.

**Stage 1** generates video+audio at base resolution (~544p) with temporal
tiling.  
**Stage 2** spatially upscales 2x and refines at high resolution with the
same computed reference positions and per-tile prompts.

**Model:** `ltx-2.3-22b-dev.safetensors` + distilled LoRA (0.5 strength)  
**Text encoder:** Gemma 3 12B  
**No detailer LoRA required** (none exists for 2.3)

The model path also includes an empty rgthree `Power Lora Loader` after the
distilled LoRA. Add optional extra LoRAs there so both sampling passes see
the same model changes.

Shared model, VAE, timing, reference, and prompt signals are routed with
KJNodes `Set`/`Get` buses. The graph keeps those Gets near their consumers so
long links do not cross the sampling lanes.

---

## Data Flow

```
LoadImage (reference)
    |
    +-- Image Size --> aspect-ratio width + final height --> Stage 1 Empty Latent
    |
    +-- Ref image batch --> Looping Reference Schedule --> both looping samplers
    |
    +-- LTXVPreprocess --> Stage 1 I2V Cond --> Stage 1 AV Concat
    |                                               |
    +-- VAEEncode (negative_index_latents)     LTXVLoopingSampler (Stage 1)
                                                    |
                                            LTXVSeparateAVLatent
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
| Resolution | ref-aspect width x half final height | Default final height 1088 gives 544 here |
| temporal_tile_size | 240 default | Derived from Basic Tile Duration |
| temporal_overlap | 80 default | Derived from Tile Overlap duration |
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
| Resolution | ref-aspect width x final height | Final height defaults to 1088 |
| temporal_tile_size | 240 default | Same schedule as stage 1 |
| temporal_overlap | 80 default | Same schedule as stage 1 |
| horizontal_tiles / vertical_tiles | 2 / 1 | Spatial tiling for memory |
| adain_factor | 0.0 | Not needed for refinement |
| Sigmas | `0.85, 0.725, 0.422, 0.0` | Low — refinement only |
| Sampler | euler_cfg_pp | Deterministic for refinement |
| CFG | 1 | With distilled LoRA |

---

## Resolution

Set `Final Height Target` in the workflow. It defaults to `1088`, is aligned
to a multiple of 64, then halved for Stage 1. The first reference image's
width/height ratio determines the aligned output width, so a 16:9 reference
at the default height yields the familiar Stage 1 `960x544` and Stage 2
`1920x1088`.

The initial I2V conditioning and looping reference paths resize internally.
Matching the late reference images to the first image's aspect ratio still
avoids center cropping when the image batch is assembled.

---

## Guiding Images

### Reference layout

`LTXVLoopingReferenceSchedule` computes `optional_cond_image_indices` from:

- its editable `total_duration` field
- the shared external `Frame Rate` control
- its editable `tile_duration` field
- its editable `overlap_duration` field
- its editable `reference_offset` field

The default four-tile graph uses `240`-frame tiles, `80` frames of overlap,
and a `16`-frame late-reference margin. That yields `713` total frames and
the image indices `0, 224, 384, 544, 704`.

Frame 0 uses the first `LoadImage` node as the I2V start frame. The explicit
late soft reference branches supply the next images. Late references are
intentionally near tile ends instead of tile boundaries, so each tile can
move toward its next anchor before the overlap is stitched.

The schedule node matches the reference-image batch to the computed index
list. If the clip becomes longer than the supplied references, it repeats the
last reference image through the remaining scheduled tiles. If the clip is
shorter, it trims extra supplied images.

For global subject anchoring across all Stage 1 tiles,
`optional_negative_index_latents` is connected to the VAE-encoded first
image. Stage 2 refines from the Stage 1 latent and uses the positioned image
batch without that extra negative-index anchor.

Frame indices must be divisible by 8 except frame 0. The schedule aligns late
indices and clips the final one to the last valid `8n` position.

### Prompt snippets

The generated graph exposes one global prompt node and one snippet node beside
each late reference image branch. Each tile prompt is built as:

```
global prompt + tile snippet
```

Those tile prompts are joined with `|` and fed to one
`LTXVMultiPromptProvider` shared by both looping samplers. The fallback
positive text encoder is wired to the global prompt as well. When there are
more generated temporal tiles than prompt snippets, `LTXVLoopingSampler`
reuses the last multi-prompt conditioning for the remaining tiles.

---

## Duration and Tile Count

| Temporal tiles | Pixel frames | Approx. duration at 24fps |
|---|---|---|
| 1 | 233 | 9.7 sec |
| 2 | 393 | 16.4 sec |
| 4 | 713 | 29.7 sec |
| 9 | 1513 | 63.0 sec |
| 45 | 7273 | 5 min 3 sec |

Frame count must satisfy `8n+1` (e.g. 121, 241, 361...). The schedule chooses
the largest valid frame count within `Total Clip Duration`:

```
frames = floor((duration_seconds * fps - 1) / 8) * 8 + 1
```

`tile_duration`, `overlap_duration`, and `reference_offset` are rounded to
8-frame schedule units. With the defaults at 24 fps, those become tile size
`240`, overlap `80`, and late reference offset `16`.

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

Edit the default constants at the top of the script to change the starting
workflow values or prompt snippets. Once loaded, `Final Height Target` and
`Frame Rate` remain external controls; duration, overlap, tile duration, and
late-reference offset live directly on `LTXVLoopingReferenceSchedule`.
