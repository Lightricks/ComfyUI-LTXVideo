"""Per-step tiled latent fusion for LTX video models.

Keeps one latent canvas and one noise field, runs the model once per overlapping
spatial tile at every denoise step (each tile conditioned on its own crop of the
IC-LoRA guide), and merges the stepped tiles back onto the canvas with Gaussian
weights. Tiles follow a single coupled trajectory, so seams do not form. Peak
VRAM is set by the spatial tile and by the temporal window actually stepped
(content window plus one guide block when streaming), not the full canvas
spatial size. Cached tile latents/noise/masks follow canvas_device; processed
conds stay on the GPU for the run.

This is distinct from LTXVTiledSampler, which independently samples a grid of
tiles. Fusion has to happen between denoise steps, so this node calls
``sampler.sampler_function`` once per tile with a two-sigma slice rather than
``sampler.sample()`` on the whole trajectory.

The latent input must be the one returned by an IC-LoRA guide node: it carries
guide frames appended along the frame axis plus the noise mask, and the
conditioning carries ``keyframe_idxs`` / ``guide_attention_entries``, all of
which this node crops per tile.
"""

import copy
import logging
import os

import comfy.model_management
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
import torch

from .nodes_registry import comfy_node
from .tiled_fusion_plan import (
    plan_grid,
    read_temporal_plan,
    shifted,
    snap_frames,
    temporal_plan,
)

logger = logging.getLogger(__name__)

# Per-step/per-window mean logging for drift forensics (set env before launch).
_DEBUG_STATS = os.environ.get("LTXV_TILED_FUSION_DEBUG", "") not in ("", "0")

# Temporal-window experiment knobs (env, not node inputs, while under study):
# LTXV_TILED_FUSION_T_OVERLAP   minimum temporal overlap fraction (default 0.5)
# LTXV_TILED_FUSION_SKIP_W0_LAT 0 = also merge the first latent of window w>0
#                               (that slot is a fresh-clip-start / causal frame
#                               and poisons a mid-clip canvas position).
#                               Default on: skip that latent.
_SKIP_FIRST_LAT = os.environ.get("LTXV_TILED_FUSION_SKIP_W0_LAT", "1") not in (
    "0",
    "",
)

_NODE = "LTXV Tiled Fusion Sampler"


def _scale_factors(model, vae=None):
    """Return (temporal, height, width) pixel-per-latent downscale factors."""
    if vae is not None:
        formula = getattr(vae, "downscale_index_formula", None)
        if formula is not None:
            st, sw, sh = formula
            return int(st), int(sh), int(sw)
    try:
        sf = model.model.latent_format
        sh = int(
            getattr(sf, "spatial_downscale", None)
            or getattr(sf, "spacial_downscale_ratio", None)
            or 32
        )
        sw = sh
        st = int(
            getattr(sf, "temporal_downscale", None)
            or getattr(sf, "temporal_downscale_ratio", None)
            or 8
        )
        return st, sh, sw
    except Exception:
        return 8, 32, 32


def gaussian_weight(tile_w, tile_h, var, device):
    """Separable Gaussian tile weight. Pure Gaussian, deliberately no floor."""

    def axis(n):
        x = torch.arange(n, dtype=torch.float32, device=device)
        mid = (n - 1) / 2
        return torch.exp(-((x - mid) ** 2) / (n * n * 2 * var))

    return axis(tile_h)[:, None] * axis(tile_w)[None, :]


def _crop_guide_metadata(
    conds, y0, x0, th, tw, canvas_h, canvas_w, sh, sw, entry_idx=None
):
    """Crop keyframe_idxs + guide_attention_entries to a spatial tile.

    y0/x0/th/tw are in latent units; sh/sw are pixels per latent step. Token
    order inside each guide's block is the patchifier's row-major (t, h, w),
    which is what makes the reshape below valid. Spatial positions are
    normalised to the tile so each tile is the same problem the LoRA was
    trained on.

    entry_idx selects a SINGLE guide entry: temporal windowing pairs window w
    with guide block w, each encoded FRESH from that window's pixel span, so
    its coords already describe a clip starting at 0 and are never rewritten
    here. None keeps every entry (whole-clip guide).
    Slicing a whole-clip guide's latents into windows does NOT work and no
    coordinate rewrite can save it: a mid-clip causal latent (each frame
    encodes 8 pixels given prior context) is not a fresh-clip latent (frame 0
    encodes exactly 1 pixel frame).
    """
    out = []
    py0, px0 = y0 * sh, x0 * sw
    for emb, d in conds:
        d = dict(d)
        entries = d.get("guide_attention_entries")
        kf = d.get("keyframe_idxs")
        if kf is not None and entries:
            pieces = []
            new_entries = []
            tok0 = 0
            for j, e in enumerate(entries):
                f, gh, gw = (int(v) for v in e["latent_shape"])
                n_tok = f * gh * gw
                blk = kf[:, :, tok0 : tok0 + n_tok, :]
                tok0 += n_tok
                if entry_idx is not None and j != entry_idx:
                    continue
                if (gh, gw) != (canvas_h, canvas_w):
                    raise ValueError(
                        f"{_NODE}: guide latent grid {gh}x{gw} does not match "
                        f"the canvas {canvas_h}x{canvas_w}. Downscaled "
                        f"(factor > 1) IC-LoRA guides are not supported; "
                        f"re-attach the guide at factor 1."
                    )
                b = blk.shape[0]
                blk = blk.reshape(b, 3, f, gh, gw, 2)
                blk = blk[:, :, :, y0 : y0 + th, x0 : x0 + tw, :].clone()
                blk[:, 1, ...] -= py0
                blk[:, 2, ...] -= px0
                pieces.append(blk.reshape(b, 3, f * th * tw, 2))
                ne = dict(e)
                ne["pre_filter_count"] = f * th * tw
                ne["latent_shape"] = [f, th, tw]
                pm = e.get("pixel_mask")
                if pm is not None:
                    ne["pixel_mask"] = pm[
                        :, :, :, py0 : py0 + th * sh, px0 : px0 + tw * sw
                    ]
                new_entries.append(ne)
            d["keyframe_idxs"] = torch.cat(pieces, dim=2)
            d["guide_attention_entries"] = new_entries
        out.append([emb, d])
    return out


def _crop_spatial(t, y0, x0, th, tw):
    """Crop the last two dims when they are real spatial dims (>1)."""
    if t is None:
        return None
    if t.shape[-2] > 1 or t.shape[-1] > 1:
        return t[..., y0 : y0 + th, x0 : x0 + tw]
    return t


class _TileGuider:
    """CFGGuider look-alike so KSamplerX0Inpaint / k-diffusion can call us.

    `inner_model` is the object `prepare_sampling` returned (has
    `scale_latent_inpaint`). `__call__` is the transformer+CFG pass.
    """

    def __init__(self, inner_model, conds, cfg):
        self.inner_model = inner_model
        self.conds = conds
        self.cfg = cfg

    def __call__(self, x, timestep, model_options={}, seed=None):
        return comfy.samplers.sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative"),
            self.conds.get("positive"),
            self.cfg,
            model_options=model_options,
            seed=seed,
        )


def _one_sampler_step(
    sampler,
    inner_model,
    conds,
    cfg,
    x,
    sig,
    sig_next,
    noise,
    latent_image,
    denoise_mask,
    model_options,
    seed,
    full_sigmas,
):
    """Run one k-diffusion step (sig -> sig_next) on a tile.

    Uses `sampler.sampler_function` (what KSamplerSelect builds), not
    `sampler.sample()`, which would re-scale noise and walk the whole
    schedule — fusion has to happen between steps.
    """
    fn = getattr(sampler, "sampler_function", None)
    if fn is None:
        raise ValueError(
            f"{_NODE} needs a SAMPLER from KSamplerSelect (or another node "
            "that exposes sampler_function). Adaptive samplers that run the "
            "whole trajectory in one go cannot fuse tiles between steps."
        )
    wrap = _TileGuider(inner_model, conds, cfg)
    model_k = comfy.samplers.KSamplerX0Inpaint(wrap, full_sigmas)
    model_k.latent_image = latent_image
    inpaint_opts = getattr(sampler, "inpaint_options", None) or {}
    if inpaint_opts.get("random", False):
        generator = torch.manual_seed(seed + 1)
        model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(
            device=noise.device, dtype=noise.dtype
        )
    else:
        model_k.noise = noise
    extra_args = {
        "model_options": model_options,
        "seed": seed,
        "denoise_mask": denoise_mask,
    }
    sigmas_pair = torch.tensor([sig, sig_next], device=x.device, dtype=torch.float32)
    extra_opts = dict(getattr(sampler, "extra_options", None) or {})
    return fn(
        model_k,
        x,
        sigmas_pair,
        extra_args=extra_args,
        callback=None,
        disable=True,
        **extra_opts,
    )


@comfy_node(
    name="LTXVTiledFusionSampler",
    description="Tiled Fusion Sampler",
)
class LTXVTiledFusionSampler:
    CATEGORY = "sampling"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("output",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model to use."},
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": "Positive conditioning from the IC-LoRA guide node. "
                        "Must carry keyframe_idxs / guide_attention_entries so each "
                        "tile can be cropped to the LoRA's trained window."
                    },
                ),
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Negative conditioning from the IC-LoRA guide node."},
                ),
                "latents": (
                    "LATENT",
                    {
                        "tooltip": "The latent RETURNED BY the IC-LoRA guide node "
                        "(it carries appended guide frames plus the noise mask). "
                        "A bare empty video latent will not work."
                    },
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "Denoise schedule. Must be descending and end at 0. "
                        "Fusion walks this list one step at a time."
                    },
                ),
                "sampler": (
                    "SAMPLER",
                    {
                        "tooltip": "From KSamplerSelect (same type as SamplerCustomAdvanced). "
                        "Discrete step samplers only: euler, euler_ancestral, heun, dpm_2, … "
                        "History methods (DPM++ 2M, LMS) lose their multi-step memory because "
                        "fusion owns the loop. Adaptive samplers cannot fuse. Ancestral methods "
                        "add independent noise per tile and can seam."
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed for the single shared noise field across all tiles.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale. Distilled models typically use 1.0.",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Tile width in pixels, multiple of 32. Set this to the "
                        "IC-LoRA's trained spatial window. Tiles smaller than the trained "
                        "window make every tile see content at the wrong scale.",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 576,
                        "min": 256,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Tile height in pixels, multiple of 32. Match tile_width to "
                        "the IC-LoRA's trained spatial window.",
                    },
                ),
                "overlap_frac": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.25,
                        "max": 0.75,
                        "step": 0.05,
                        "tooltip": "Fraction of each tile overlapping its neighbor. Stay at "
                        "0.5 or above: below that a periodic grid appears on structured content.",
                    },
                ),
                "blend_var": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.01,
                        "max": 0.2,
                        "step": 0.01,
                        "tooltip": "Variance of the Gaussian blend weights in the overlaps. "
                        "Leave at 0.05 unless you are measuring seams.",
                    },
                ),
                "grid_cycle": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Cycle shifted tile grids across steps. 1 = fixed grid "
                        "(cheapest), 4 = thorough. Spatial only: cycling the temporal grid "
                        "slides content in time.",
                    },
                ),
                "tile_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Temporal window in pixel frames (97 for LTX). 0 = one "
                        "extent. Other values are snapped to 8n+1. >0 needs "
                        "LTXAddVideoICLoRAGuide with use_streaming on and the SAME "
                        "tile_frames: that re-encodes a fresh guide per window and "
                        "writes the plan this sampler reads. A whole-clip guide "
                        "cannot be sliced, and windowing is disabled if the plan "
                        "is missing.",
                    },
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {
                        "tooltip": "Used to read the model's temporal and spatial downscale "
                        "factors. If omitted, they are taken from the diffusion model "
                        "(LTX defaults: 8 frames and 32 pixels per latent cell)."
                    },
                ),
                "canvas_device": (
                    ["auto", "gpu", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Where to keep the full-canvas tensors and cached tile "
                        "latents/noise/masks. auto = GPU if there is enough free "
                        "memory, otherwise CPU. Processed conds stay on the GPU.",
                    },
                ),
            },
        }

    @torch.inference_mode()
    def sample(
        self,
        model,
        positive,
        negative,
        latents,
        sigmas,
        sampler,
        seed,
        cfg,
        tile_width,
        tile_height,
        overlap_frac,
        blend_var,
        grid_cycle,
        tile_frames=0,
        vae=None,
        canvas_device="auto",
    ):
        samples = latents["samples"]
        noise_mask = latents.get("noise_mask")
        if samples.ndim != 5:
            raise ValueError(f"{_NODE} needs a 5D video latent.")
        b, C, F_total, H, W = samples.shape
        if b != 1:
            raise ValueError(f"{_NODE} supports batch size 1.")
        if noise_mask is None:
            raise ValueError(
                f"{_NODE} requires a noise_mask; the latent must be the one "
                "returned by the IC-LoRA guide node (it carries noise_mask)."
            )

        device = comfy.model_management.get_torch_device()
        st, sh, sw = _scale_factors(model, vae)

        tl_w, tl_h = tile_width // sw, tile_height // sh
        tl_w, tl_h = min(tl_w, W), min(tl_h, H)

        min_ov_x = max(2, int(tl_w * overlap_frac))
        min_ov_y = max(2, int(tl_h * overlap_frac))
        xs = plan_grid(W, tl_w, min_ov_x)
        ys = plan_grid(H, tl_h, min_ov_y)

        ov_x = tl_w - (xs[1] - xs[0]) if len(xs) > 1 else tl_w
        ov_y = tl_h - (ys[1] - ys[0]) if len(ys) > 1 else tl_h
        # Interior shift must leave at least min_ov overlap on cycled grids.
        max_shift = max(0, min(ov_x - min_ov_x, ov_y - min_ov_y))
        off = min(4, max_shift)
        if grid_cycle > 1 and off < 1:
            logger.warning(
                "grid_cycle=%d ignored: no interior shift can keep overlap_frac=%.2f",
                grid_cycle,
                overlap_frac,
            )
            offsets = [(0, 0)]
        else:
            offsets = [(0, 0), (off, off), (off, 0), (0, off)][: max(1, grid_cycle)]
        grids = [
            [(x, y) for y in shifted(ys, H, tl_h, oy) for x in shifted(xs, W, tl_w, ox)]
            for ox, oy in offsets
        ]
        all_origins = sorted({o for g in grids for o in g})
        logger.info(
            "tiled fusion: canvas %dx%d lat, tile %dx%d, grid %dx%d, "
            "cycle=%d, %d unique origins",
            W,
            H,
            tl_w,
            tl_h,
            len(xs),
            len(ys),
            len(grids),
            len(all_origins),
        )

        if len(all_origins) == 1:
            logger.info("canvas fits one tile; falling back to plain sampling")

        n_content = F_total
        if noise_mask is not None:
            fm = (
                noise_mask.to(dtype=torch.float32)
                .reshape(1, 1, F_total, -1)
                .amax(dim=-1)
            )
            for f in range(F_total - 1, -1, -1):
                if float(fm[0, 0, f]) < 0.999:
                    n_content = f
                else:
                    break
        n_guide = F_total - n_content

        plan = read_temporal_plan(positive)

        tile_frames = int(tile_frames)
        if tile_frames > 0:
            snapped = snap_frames(tile_frames)
            if snapped != tile_frames:
                logger.info("tile_frames %d snapped to %d (8n+1)", tile_frames, snapped)
            tile_frames = snapped

        tf_lat = 0
        if tile_frames > 0:
            tf_lat = max(1, (tile_frames - 1) // st + 1)
        use_temporal = 0 < tf_lat < n_content

        if plan is not None and int(plan["tile_frames"]) != int(tile_frames):
            raise ValueError(
                f"{_NODE}: the guide was encoded for tile_frames="
                f"{plan['tile_frames']} but the sampler got tile_frames="
                f"{tile_frames}; set both to the same value."
            )

        per_window_guides = False
        if use_temporal:
            if plan is not None:
                ts = [int(v) for v in plan["ts"]]
                if int(plan["n_content"]) != n_content or n_guide != len(ts) * tf_lat:
                    raise ValueError(
                        f"{_NODE}: temporal plan does not match the latent "
                        f"(plan: n_content={plan['n_content']}, "
                        f"{len(ts)} windows of {tf_lat}; latent: "
                        f"content={n_content}, guide={n_guide})."
                    )
                per_window_guides = True
            elif n_guide == 0:
                ts = temporal_plan(n_content, tf_lat)
            else:
                logger.warning(
                    "tile_frames=%d with a whole-clip guide: temporal windows "
                    "need a fresh IC-LoRA encode per window. Turn on "
                    "use_streaming on LTXAddVideoICLoRAGuide with the same "
                    "tile_frames; windowing DISABLED for this run.",
                    tile_frames,
                )
                use_temporal = False

        if use_temporal:
            logger.info(
                "temporal windows: %d x %d latent frames (%d content, " "%d guide%s)",
                len(ts),
                tf_lat,
                n_content,
                n_guide,
                ", per-window guides" if per_window_guides else "",
            )
            window_len = min(tf_lat, n_content)
            if per_window_guides:
                window_len += tf_lat
        else:
            ts = [0]
            tf_lat = F_total
            window_len = F_total
            if tile_frames > 0:
                logger.info(
                    "temporal windowing not active (clip fits one window, "
                    "or disabled above)"
                )

        if canvas_device == "cpu":
            cdev = torch.device("cpu")
        elif canvas_device == "gpu":
            cdev = device
        else:
            free = comfy.model_management.get_free_memory(device)
            need = samples.numel() * 4 * 4
            cdev = device if need < free * 0.25 else torch.device("cpu")
        logger.info("canvas tensors on %s", cdev)

        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(samples.shape, generator=g, dtype=torch.float32).to(cdev)

        tile_shape = [1, C, window_len, tl_h, tl_w]
        conds0 = {
            "positive": comfy.sampler_helpers.convert_cond(copy.deepcopy(positive)),
            "negative": comfy.sampler_helpers.convert_cond(copy.deepcopy(negative)),
        }
        inner_model, conds0, loaded = comfy.sampler_helpers.prepare_sampling(
            model, tile_shape, conds0, model.model_options
        )
        try:
            model_sampling = (
                getattr(inner_model, "model_sampling", None)
                or model.model.model_sampling
            )

            model.pre_run()

            latent_in = (
                model.model.process_latent_in(samples).to(torch.float32).to(cdev)
            )
            sigmas = sigmas.to(torch.float32)
            s0 = sigmas[0].item()

            canvas = model_sampling.noise_scaling(
                torch.tensor(s0, device=cdev), noise, latent_in, False
            )

            if noise_mask is not None:
                mask_full = noise_mask.to(torch.float32).to(cdev)
            else:
                mask_full = None

            def _window_slice(w, t0, tensor):
                """Take content[t0:t0+tf] plus window w's own guide block."""
                if not use_temporal:
                    return tensor
                parts = [tensor[:, :, t0 : t0 + tf_lat]]
                if per_window_guides:
                    g0 = n_content + w * tf_lat
                    parts.append(tensor[:, :, g0 : g0 + tf_lat])
                return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]

            tile_data = {}
            for w, t0_ in enumerate(ts):
                eidx = w if (use_temporal and per_window_guides) else None
                for x0_, y0_ in all_origins:
                    pos_t = _crop_guide_metadata(
                        copy.deepcopy(positive),
                        y0_,
                        x0_,
                        tl_h,
                        tl_w,
                        H,
                        W,
                        sh,
                        sw,
                        entry_idx=eidx,
                    )
                    neg_t = _crop_guide_metadata(
                        copy.deepcopy(negative),
                        y0_,
                        x0_,
                        tl_h,
                        tl_w,
                        H,
                        W,
                        sh,
                        sw,
                        entry_idx=eidx,
                    )
                    lat_t = _window_slice(
                        w,
                        t0_,
                        latent_in[:, :, :, y0_ : y0_ + tl_h, x0_ : x0_ + tl_w],
                    )
                    noise_t = _window_slice(
                        w,
                        t0_,
                        noise[:, :, :, y0_ : y0_ + tl_h, x0_ : x0_ + tl_w],
                    )
                    mask_t = None
                    if mask_full is not None:
                        mask_t = _crop_spatial(mask_full, y0_, x0_, tl_h, tl_w)
                        if mask_t.shape[2] == F_total:
                            mask_t = _window_slice(w, t0_, mask_t)
                        if mask_t.shape[2] == lat_t.shape[2]:
                            mask_t = mask_t.expand(1, 1, lat_t.shape[2], tl_h, tl_w)
                    # process_conds needs GPU tensors; lat/noise/mask stay on cdev
                    # until each step so a CPU canvas does not pin every tile.
                    conds_t = comfy.samplers.process_conds(
                        inner_model,
                        noise_t.to(device),
                        {
                            "positive": comfy.sampler_helpers.convert_cond(pos_t),
                            "negative": comfy.sampler_helpers.convert_cond(neg_t),
                        },
                        device,
                        latent_image=lat_t.to(device),
                        denoise_mask=None if mask_t is None else mask_t.to(device),
                        seed=seed,
                        latent_shapes=[lat_t.shape],
                    )
                    tile_data[(w, x0_, y0_)] = (conds_t, lat_t, noise_t, mask_t)

            w_tile = gaussian_weight(tl_w, tl_h, blend_var, cdev)

            w_len = tf_lat if use_temporal else n_content
            if use_temporal and len(ts) > 1:
                xs_t = torch.arange(w_len, dtype=torch.float32, device=cdev)
                mid_t = (w_len - 1) / 2
                w_time = torch.exp(
                    -((xs_t - mid_t) ** 2) / (w_len * w_len * 2 * blend_var)
                )
            else:
                w_time = torch.ones(w_len, dtype=torch.float32, device=cdev)

            def _w_skip(w):
                return int(
                    _SKIP_FIRST_LAT
                    and use_temporal
                    and w > 0
                    and ts[w] < ts[w - 1] + tf_lat
                )

            wsums = []
            for grid in grids:
                ws = torch.zeros((1, 1, n_content, H, W), device=cdev)
                for w, t0_ in enumerate(ts):
                    sk = _w_skip(w)
                    tw_ = w_time[sk : min(tf_lat, n_content - t0_)].reshape(
                        1, 1, -1, 1, 1
                    )
                    for x0_, y0_ in grid:
                        ws[
                            :,
                            :,
                            t0_ + sk : t0_ + sk + tw_.shape[2],
                            y0_ : y0_ + tl_h,
                            x0_ : x0_ + tl_w,
                        ] += (
                            w_tile * tw_
                        )
                wsums.append(ws.clamp_min(1e-12))

            model_options = model.model_options

            pbar = comfy.utils.ProgressBar(len(sigmas) - 1)
            for step in range(len(sigmas) - 1):
                sig = sigmas[step].item()
                sig_next = sigmas[step + 1].item()
                grid = grids[step % len(grids)]
                acc = torch.zeros(
                    (1, C, n_content, H, W), device=cdev, dtype=canvas.dtype
                )

                for w, t0_ in enumerate(ts):
                    for x0_, y0_ in grid:
                        conds_t, lat_t, noise_t, mask_t = tile_data[(w, x0_, y0_)]
                        x = _window_slice(
                            w,
                            t0_,
                            canvas[:, :, :, y0_ : y0_ + tl_h, x0_ : x0_ + tl_w],
                        ).to(device, torch.float32)

                        x_next = _one_sampler_step(
                            sampler,
                            inner_model,
                            conds_t,
                            cfg,
                            x,
                            sig,
                            sig_next,
                            noise_t.to(device, torch.float32),
                            lat_t.to(device, torch.float32),
                            None if mask_t is None else mask_t.to(device),
                            model_options,
                            seed,
                            sigmas,
                        )

                        nc = min(tf_lat, n_content - t0_) if use_temporal else n_content
                        sk = _w_skip(w)
                        tw_ = w_time[sk:nc].reshape(1, 1, -1, 1, 1)
                        acc[
                            :,
                            :,
                            t0_ + sk : t0_ + nc,
                            y0_ : y0_ + tl_h,
                            x0_ : x0_ + tl_w,
                        ] += (
                            x_next[:, :, sk:nc].to(cdev) * w_tile * tw_
                        )

                        if _DEBUG_STATS:
                            logger.info(
                                "dbg step %d w%d (t0=%d xy=%d,%d): x %.4f "
                                "x_next %.4f",
                                step,
                                w,
                                t0_,
                                x0_,
                                y0_,
                                float(x[:, :, :nc].mean()),
                                float(x_next[:, :, :nc].mean()),
                            )

                blended = acc / wsums[step % len(grids)]
                if _DEBUG_STATS:
                    logger.info(
                        "dbg step %d blended: mean %.4f std %.4f",
                        step,
                        float(blended.mean()),
                        float(blended.std()),
                    )
                if n_guide:
                    canvas = torch.cat([blended, canvas[:, :, n_content:]], dim=2)
                else:
                    canvas = blended
                pbar.update(1)

            out = canvas
            if mask_full is not None:
                frame_mask = mask_full.reshape(1, 1, F_total, -1).amax(dim=-1)
                keep = F_total
                for f in range(F_total - 1, -1, -1):
                    if frame_mask[0, 0, f] < 0.999:
                        keep = f
                    else:
                        break
                out = canvas[:, :, :keep]
                if keep < F_total:
                    logger.info("dropped %d appended guide frame(s)", F_total - keep)

            out = model.model.process_latent_out(out.to(torch.float32).cpu())
        finally:
            model.cleanup()
            comfy.sampler_helpers.cleanup_models(conds0, loaded)

        return ({"samples": out},)
