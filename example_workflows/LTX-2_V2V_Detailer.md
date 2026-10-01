# LTX-2 V2V Detailer — Tuning Notes

## Workflow Overview

Video-to-video detailer using LTX-2 19B with the IC-LoRA detailer. Upscales and refines
an input video by adding noise and denoising at the target resolution.

**Default upscale target:** 1920px max dimension (via `ImageScaleToMaxDimension`)
**Sampler:** Euler
**Text encoder:** Gemma 3 12B IT

---

## Known Issues at Large Upscale Ratios (e.g. 544 → 1920)

A 3.5× upscale in a single pass forces the model to invent ~12× more pixel area than
the source. This causes two symptoms at the default sigma settings:

- **Oversaturated colors** — model rebuilds rather than refines, drifting from source colors
- **Dithering/noise on fine textures** (hair, fabric) — hallucinated high-frequency detail

---

## Key Parameters & Recommended Values

### ManualSigmas — most impactful setting

Controls how aggressively the model re-generates the video. Lower = preserves original more.

| Scenario | Values |
|---|---|
| Default (too aggressive for large upscales) | `0.909375, 0.725, 0.421875, 0.0` |
| Recommended starting point | `0.5, 0.35, 0.2, 0.0` |
| Conservative (colors still drifting) | `0.35, 0.2, 0.1, 0.0` |

### LoRA Strength (LoraLoaderModelOnly)

The detailer LoRA at full strength over-sharpens fine structures.

| Default | Recommended |
|---|---|
| 1.0 | 0.65 – 0.75 |

### LTXVLoopingSampler

| Parameter | Default | Notes |
|---|---|---|
| guiding_strength | 1.0 | Keep at 1.0 — lowering causes drift from source |
| temporal_overlap_cond_strength | 0.5 | Leave as-is |
| horizontal_tiles / vertical_tiles | 1 / 1 | Single spatial tile at 1920px is fine |

### LTXVSpatioTemporalTiledVAEDecode

| Parameter | Default | Notes |
|---|---|---|
| spatial_tiles | 4 | Fine for 1920px |
| spatial_overlap | 4 | Fine as-is |
| temporal_tile_length | 16 | Fine as-is |

---

## Recommended Tuning Order

1. Set ManualSigmas to `0.5, 0.35, 0.2, 0.0` → run and compare
2. If hair/texture still dithers → reduce LoRA strength to 0.7
3. If colors still saturated → drop sigmas further to `0.35, 0.2, 0.1, 0.0`
4. If quality still insufficient → split into two upscale passes (see below)

---

## Two-Stage Upscaling (Best Quality for Large Ratios)

Rather than one 3.5× jump, run the workflow twice:

**Pass 1:** 544 → 1024, sigmas `0.5, 0.35, 0.2, 0.0`
**Pass 2:** 1024 → 1920, sigmas `0.25, 0.15, 0.05, 0.0`

Pass 2 needs very low sigmas — most detail is already correct, it is only sharpening.

---

## Handling Arbitrary-Length Videos

The workflow can process videos of any length. `LoadVideo` loads the full clip,
`ImageScaleToMaxDimension` rescales every frame, `VAEEncodeTiled` encodes the
full sequence into a latent, and `LTXVLoopingSampler` tiles along the temporal
axis with overlapping chunks.

For a video with N latent frames, the sampler produces tiles as:

```
Tile 0: frames [0, temporal_tile_size)
Tile 1: frames [temporal_tile_size - temporal_overlap, 2*temporal_tile_size - temporal_overlap)
Tile 2: ...
```

Each tile is denoised independently (conditioned on the overlap region from the
previous tile), then the results are stitched. There is no hard upper bound on
video length — the sampler simply produces more temporal tiles.

**Practical limits** are set by:

- **VAE encode/decode memory**: the full video must be encoded and decoded.
  `VAEEncodeTiled` and `LTXVSpatioTemporalTiledVAEDecode` tile spatially and
  temporally, so this scales to long clips. Increase `spatial_tiles` or reduce
  `temporal_tile_length` in the VAE decode node if the VAE step OOMs.
- **Latent tensor size**: the full video latent (shape `[1, 128, T, H, W]`)
  must fit in memory at once. At 1280px, each latent frame is ~0.26MB (128
  channels × 40 × 40 × bf16). A 10-minute clip at 24fps (14400 frames →
  ~1800 latent frames) is ~470MB — easily fits.
- **Wall-clock time**: each temporal tile requires a full sampling pass. On
  unified memory (~130GB/s bandwidth), a single tile at 1280px takes minutes.
  A 10-minute clip with `temporal_tile_size=32, temporal_overlap=16` produces
  ~113 tiles, which could take many hours.
- **Quality drift over many tiles**: temporal overlap conditioning keeps
  adjacent tiles coherent, but over very long sequences the style can drift
  gradually. `optional_normalizing_latents` and `adain_factor` can mitigate
  this by anchoring color/contrast statistics.

In practice, "infinite length" means you can process clips of any duration if
you have the patience. Memory is not the bottleneck — compute time is.

---

## Strix Halo 128GB Unified Memory — OOM Prevention

The default settings (1920px, single spatial tile, `temporal_tile_size=56`)
are tuned for discrete GPUs with fast HBM. On Strix Halo with ~120GB unified
memory allocated via TTM, the peak activation memory during sampling at 1920px
can exceed available GPU memory.

### Where the memory goes

| Component | Approximate size |
|---|---|
| LTX-2 19B (BF16) | ~38GB |
| Gemma 3 12B (Q4 quantized) | ~7GB |
| VAE | ~0.5GB |
| Activations during sampling (resolution-dependent) | 40–80GB+ at 1920px |

With `--highvram` keeping all models resident, ~46GB is consumed before any
activations are allocated.

### Recommended settings

| Parameter | Default | Recommended |
|---|---|---|
| `ImageScaleToMaxDimension` | 1920 | **1280** (or 1024) |
| `horizontal_tiles` | 1 | **2** (at 1920px) or **1** (at 1280px) |
| `vertical_tiles` | 1 | **2** (at 1920px) or **1** (at 1280px) |
| `temporal_tile_size` | 56 | **32** |
| `temporal_overlap` | 24 | **16** |
| `ManualSigmas` | `0.909, 0.725, 0.422, 0.0` | `0.5, 0.35, 0.2, 0.0` |
| `LoRA strength` | 1.0 | **0.7** |
| `LTXVSpatioTemporalTiledVAEDecode spatial_tiles` | 4 | **6–8** if VAE OOMs |

**Spatial tiling** (`horizontal_tiles × vertical_tiles`) is the most impactful
setting. It tiles the spatial dimension during sampling so that attention and
feedforward layers operate on a fraction of the full resolution. 2×2 at 1920px
reduces per-tile activation memory by roughly 4×.

**Temporal tile size** reduction also helps: fewer frames per tile means a
shorter sequence length for the transformer, reducing both attention (O(n²))
and feedforward memory.

### LTXV Chunk FeedForward (KJNodes)

The `LTXV Chunk FeedForward` node from comfyui-kjnodes can be added between
the model loader and the guider. It patches the feedforward layers in each
transformer block to process the token sequence in chunks rather than all at
once, reducing peak activation memory in the FFN (which expands hidden dim
by 4×).

| Parameter | Recommended |
|---|---|
| `chunks` | **2** (start here; increase to 3–4 if still tight) |
| `dim_threshold` | **4096** (default — only activates for large sequences) |

This is a secondary optimization — spatial tiling has more impact because it
reduces memory for both attention and FFN. Use Chunk FeedForward in addition
to spatial tiling, not instead of it. Note the node is marked experimental and
may cause minor numerical differences in output.

### If 1920px is required

Use the two-stage approach:

**Pass 1:** source → 1024, sigmas `0.5, 0.35, 0.2, 0.0`, 1×1 spatial tiles
**Pass 2:** 1024 → 1920, sigmas `0.25, 0.15, 0.05, 0.0`, 2×2 spatial tiles

Each pass individually fits in memory.
