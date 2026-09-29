# LTXVTiledFusionSampler

Per-step tiled latent fusion for IC-LoRA (and other LTX conditioning) at
resolutions larger than the model's trained window.

This is **not** `LTXVTiledSampler`. That node independently samples a grid of
tiles. This node keeps **one** latent canvas and **one** noise field, runs the
model on overlapping crops at **every** denoise step, and Gaussian-blends the
stepped tiles back onto the canvas. Tiles share a trajectory, so seams do not
form. Peak VRAM tracks **spatial tile size** and the **temporal window**
actually stepped (one content window plus one guide block when streaming),
not the full canvas. Cached tile latents/noise/masks follow `canvas_device`;
processed conds stay on the GPU. Streaming also does one full-res VAE encode
per temporal window.

## Graph shape

```
UNET + IC-LoRA
Gemma CLIP → CLIPTextEncode (pos / neg)
guide frames → LTX Add Video IC-LoRA Guide  (positive, negative, latents)
KSamplerSelect (euler / heun / …) → SAMPLER
sigmas (Manual Sigmas or any scheduler, descending, ending at 0)
        │
        ▼
LTXV Tiled Fusion Sampler → LATENT → VAE Decode (tiled)
```

The latent **must** come from the IC-LoRA guide node. It carries appended guide
frames and a noise mask. This node crops those extra frames itself, so
`LTXVCropGuides` is not needed.

Leave `use_streaming` **off** on the guide (the default) for a whole-clip encode.
For clips longer than the trained window, turn `use_streaming` **on** and set the
same `tile_frames` (typically 97) on both the guide and this sampler. That
re-encodes each temporal window as its own clip and writes
`ltxv_tiled_fusion_temporal_plan` onto the conditioning. The sampler disables
windowing if the plan is missing. Do not slice a whole-clip guide: mid-clip
causal latents are not fresh-clip-start latents.

## Sampler input

The `SAMPLER` socket is the same object `SamplerCustomAdvanced` takes. Fusion
owns the sigma loop and calls `sampler.sampler_function` once per tile with a
two-sigma slice. Use discrete step methods (`euler`, `heun`, `dpm_2`). History
methods (`dpmpp_2m`, `lms`) lose their multi-step memory. Adaptive samplers
cannot fuse. Ancestral methods add independent noise per tile and can seam.

## Settings that matter

| Input | Notes |
|---|---|
| `tile_width` / `tile_height` | The IC-LoRA's trained spatial window in pixels (multiples of 32). Smaller tiles see content at the wrong scale. |
| `overlap_frac` | Stay at **0.5** or above. Below that a periodic grid appears on structured content. |
| `blend_var` | Gaussian overlap variance. Default 0.05. |
| `grid_cycle` | 1 = fixed grid (cheapest), 4 = shift the grid across steps. Spatial only. |
| `tile_frames` | Temporal window in pixel frames (97 for LTX). 0 = one extent. >0 needs `use_streaming` on `LTXAddVideoICLoRAGuide` with the same value. |
| `vae` | Optional. Supplies temporal/spatial downscale (otherwise read from the diffusion model). |
| `use_tiled_encode` on the guide node | Must stay **false**. Spatially tiled guide encodes imprint a grid into the conditioning. |
| `use_streaming` on the guide node | Off by default. On = per-window fresh encode + plan. |

`overlap_frac`, `blend_var`, and `grid_cycle` are spatial fusion knobs. They do
not replace an untiled composition pass when the LoRA is generating a new look
at a canvas far above the trained window.

Take `SAMPLER` from `KSamplerSelect` (`euler`) and sigmas from
`Manual Sigmas` (descending, ending at 0). Keep `use_tiled_encode` **false** on
every `LTXAddVideoICLoRAGuide`. Turn `use_streaming` **on** and set the same
`tile_frames` on the guide and the sampler when the clip is longer than one
window.
