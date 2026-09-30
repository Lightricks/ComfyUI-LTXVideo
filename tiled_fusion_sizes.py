"""Canvas sizes and temporal window for tiled-fusion graphs.

Stage 1 aims at full HD in the source aspect without spatial tiling. Stage 2
is 2x that (~4K). 8K is another 2x. HD is the 720p step of the same ladder;
qHD is the 960 long-edge step.

Dimensions stay multiples of 32. ``tile_frames`` stays 8n+1, targeting 97,
stretching up to 121 only when that lets a slightly longer clip skip chunking.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

from .nodes_registry import comfy_node
from .tiled_fusion_plan import plan_grid, snap_frames, temporal_plan

SPATIAL_ALIGN = 32
FRAME_STEP = 8
SLACK_PX = 200
QHD_LONG_EDGE = 960
HD_LONG_EDGE = 1280
FULLHD_LONG_EDGE = 1920
TILE_FRAMES_TARGET = 97
TILE_FRAMES_MAX = 121
OVERLAP_FRAC = 0.5

TILE_CHOICES = ("qHD", "HD", "FullHD")
OUTPUT_CHOICES = ("FullHD", "4K", "8K")

_QHD_KEYS = frozenset({"qhd", "quarterhd", "quarter", "960"})
_HD_KEYS = frozenset({"hd", "720p", "720"})
_FULLHD_KEYS = frozenset({"fullhd", "fhd", "1080p", "1080"})
_4K_KEYS = frozenset({"4k", "uhd", "2160p", "2160"})
_8K_KEYS = frozenset({"8k", "eightk", "4320p", "4320"})


class TilingSizes(NamedTuple):
    qhd_width: int
    qhd_height: int
    hd_width: int
    hd_height: int
    fullhd_width: int
    fullhd_height: int
    overlap_frac: float
    tile_frames: int
    width_4k: int
    height_4k: int
    width_8k: int
    height_8k: int


class TilingPlan(NamedTuple):
    initial_canvas_width: int
    initial_canvas_height: int
    tile_width: int
    tile_height: int
    output_width: int
    output_height: int
    overlap_frac: float
    tile_frames: int
    spatial_tiles_x: int
    spatial_tiles_y: int
    output_tiles_x: int
    output_tiles_y: int
    temporal_chunks: int
    canvas_label: str
    tile_label: str
    output_label: str
    tile_clamped: bool
    is_8k: bool
    summary: str


def _norm_label(value: str) -> str:
    return str(value or "").lower().replace(" ", "").replace("_", "").replace("-", "")


def snap_spatial(value: float, align: int = SPATIAL_ALIGN) -> int:
    """Nearest multiple of ``align``, ties round up, never below ``align``."""
    value = max(float(align), float(value))
    return max(align, int(math.floor(value / align + 0.5) * align))


def tile_frames_for(frame_count: int) -> int:
    """Temporal window: 97 or the clip itself, stretching to 121 to skip chunks."""
    n = max(1, int(frame_count))
    if n <= TILE_FRAMES_TARGET:
        tf = snap_frames(n)
        return min(TILE_FRAMES_TARGET, tf)
    if n <= TILE_FRAMES_MAX:
        covering = ((n - 1 + FRAME_STEP - 1) // FRAME_STEP) * FRAME_STEP + 1
        return min(TILE_FRAMES_MAX, covering)
    return TILE_FRAMES_TARGET


def _fit_long_edge(width: float, height: float, long_edge: int) -> tuple[float, float]:
    long = max(width, height)
    scale = long_edge / long
    return width * scale, height * scale


def _keep_source_if_close(
    target_w: float, target_h: float, src_w: int, src_h: int, slack: int = SLACK_PX
) -> tuple[float, float]:
    if abs(src_w - target_w) <= slack and abs(src_h - target_h) <= slack:
        return float(src_w), float(src_h)
    return target_w, target_h


def _aligned_size(width: float, height: float) -> tuple[int, int]:
    return snap_spatial(width), snap_spatial(height)


def _ladder_step(src_w: int, src_h: int, long_edge: int) -> tuple[int, int]:
    tw, th = _fit_long_edge(src_w, src_h, long_edge)
    tw, th = _keep_source_if_close(tw, th, src_w, src_h)
    return _aligned_size(tw, th)


def compute_tiling_sizes(width: int, height: int, frame_count: int) -> TilingSizes:
    """Return the qHD / HD / full-HD / 4K / 8K ladder plus overlap and tile_frames."""
    src_w = max(1, int(width))
    src_h = max(1, int(height))

    qhd_w, qhd_h = _ladder_step(src_w, src_h, QHD_LONG_EDGE)
    hd_w, hd_h = _ladder_step(src_w, src_h, HD_LONG_EDGE)
    fhd_w, fhd_h = _ladder_step(src_w, src_h, FULLHD_LONG_EDGE)

    k4_w, k4_h = _keep_source_if_close(fhd_w * 2, fhd_h * 2, src_w, src_h)
    k4_w, k4_h = _aligned_size(k4_w, k4_h)

    k8_w, k8_h = _keep_source_if_close(k4_w * 2, k4_h * 2, src_w, src_h)
    k8_w, k8_h = _aligned_size(k8_w, k8_h)

    return TilingSizes(
        qhd_width=qhd_w,
        qhd_height=qhd_h,
        hd_width=hd_w,
        hd_height=hd_h,
        fullhd_width=fhd_w,
        fullhd_height=fhd_h,
        overlap_frac=OVERLAP_FRAC,
        tile_frames=tile_frames_for(frame_count),
        width_4k=k4_w,
        height_4k=k4_h,
        width_8k=k8_w,
        height_8k=k8_h,
    )


def _pick_tile_or_canvas(sizes: TilingSizes, choice: str) -> tuple[int, int, str]:
    key = _norm_label(choice)
    if key in _QHD_KEYS:
        return sizes.qhd_width, sizes.qhd_height, "qHD"
    if key in _HD_KEYS:
        return sizes.hd_width, sizes.hd_height, "HD"
    if key in _FULLHD_KEYS:
        return sizes.fullhd_width, sizes.fullhd_height, "FullHD"
    raise ValueError(
        f"Unknown tile/canvas size {choice!r}; expected qHD, HD, or FullHD."
    )


def _pick_output(sizes: TilingSizes, choice: str) -> tuple[int, int, str]:
    key = _norm_label(choice)
    if key in _8K_KEYS:
        return sizes.width_8k, sizes.height_8k, "8K"
    if key in _4K_KEYS:
        return sizes.width_4k, sizes.height_4k, "4K"
    if key in _FULLHD_KEYS:
        return sizes.fullhd_width, sizes.fullhd_height, "FullHD"
    raise ValueError(f"Unknown output size {choice!r}; expected FullHD, 4K, or 8K.")


def _latent_wh(width: int, height: int) -> tuple[int, int]:
    return max(1, int(width) // SPATIAL_ALIGN), max(1, int(height) // SPATIAL_ALIGN)


def spatial_tile_grid(
    canvas_w: int,
    canvas_h: int,
    tile_w: int,
    tile_h: int,
    overlap_frac: float = OVERLAP_FRAC,
) -> tuple[int, int]:
    """Number of spatial tiles covering the canvas, matching the sampler grid."""
    cw, ch = _latent_wh(canvas_w, canvas_h)
    tw, th = _latent_wh(tile_w, tile_h)
    tw, th = min(tw, cw), min(th, ch)
    min_ov_x = max(2, int(tw * overlap_frac))
    min_ov_y = max(2, int(th * overlap_frac))
    return (
        len(plan_grid(cw, tw, min_ov_x)),
        len(plan_grid(ch, th, min_ov_y)),
    )


def temporal_chunk_count(frame_count: int, tile_frames: int) -> int:
    """Number of temporal windows, matching the sampler / guide handshake."""
    n = max(1, int(frame_count))
    tf = max(1, int(tile_frames))
    n_lat = (n - 1) // FRAME_STEP + 1
    tf_lat = min(n_lat, (tf - 1) // FRAME_STEP + 1)
    return len(temporal_plan(n_lat, tf_lat))


def format_summary(
    *,
    src_w: int,
    src_h: int,
    frame_count: int,
    plan_fields: dict,
) -> str:
    sx, sy = plan_fields["spatial_tiles_x"], plan_fields["spatial_tiles_y"]
    ox, oy = plan_fields["output_tiles_x"], plan_fields["output_tiles_y"]
    chunks = plan_fields["temporal_chunks"]
    tile_note = " (clamped to canvas)" if plan_fields["tile_clamped"] else ""
    if (sx, sy) == (ox, oy):
        spatial = f"spatial     {sx}×{sy} tiles"
    else:
        spatial = f"spatial     {sx}×{sy} on canvas, {ox}×{oy} on output"
    chunk_word = "chunk" if chunks == 1 else "chunks"
    return "\n".join(
        (
            f"source      {src_w}×{src_h} × {frame_count}f",
            f"canvas      {plan_fields['initial_canvas_width']}×{plan_fields['initial_canvas_height']}"
            f"  {plan_fields['canvas_label']}",
            f"tile        {plan_fields['tile_width']}×{plan_fields['tile_height']}"
            f"  {plan_fields['tile_label']}{tile_note}",
            f"output      {plan_fields['output_width']}×{plan_fields['output_height']}"
            f"  {plan_fields['output_label']}",
            f"overlap     {plan_fields['overlap_frac']:g}",
            f"tile_frames {plan_fields['tile_frames']}",
            spatial,
            f"temporal    {chunks} {chunk_word}",
        )
    )


def resolve_tiling_plan(
    width: int,
    height: int,
    frame_count: int,
    tile_size: str = "FullHD",
    initial_canvas_size: str = "FullHD",
    output_size: str = "FullHD",
) -> TilingPlan:
    """Pick canvas / tile / output from the ladder and count tiles and chunks."""
    src_w = max(1, int(width))
    src_h = max(1, int(height))
    frames = max(1, int(frame_count))
    sizes = compute_tiling_sizes(src_w, src_h, frames)

    canvas_w, canvas_h, canvas_label = _pick_tile_or_canvas(sizes, initial_canvas_size)
    tile_w, tile_h, tile_label = _pick_tile_or_canvas(sizes, tile_size)
    out_w, out_h, out_label = _pick_output(sizes, output_size)

    clamped_w, clamped_h = min(tile_w, canvas_w), min(tile_h, canvas_h)
    tile_clamped = (clamped_w, clamped_h) != (tile_w, tile_h)
    tile_w, tile_h = clamped_w, clamped_h

    sx, sy = spatial_tile_grid(canvas_w, canvas_h, tile_w, tile_h, sizes.overlap_frac)
    ox, oy = spatial_tile_grid(out_w, out_h, tile_w, tile_h, sizes.overlap_frac)
    chunks = temporal_chunk_count(frames, sizes.tile_frames)

    fields = dict(
        initial_canvas_width=canvas_w,
        initial_canvas_height=canvas_h,
        tile_width=tile_w,
        tile_height=tile_h,
        output_width=out_w,
        output_height=out_h,
        overlap_frac=sizes.overlap_frac,
        tile_frames=sizes.tile_frames,
        spatial_tiles_x=sx,
        spatial_tiles_y=sy,
        output_tiles_x=ox,
        output_tiles_y=oy,
        temporal_chunks=chunks,
        canvas_label=canvas_label,
        tile_label=tile_label,
        output_label=out_label,
        tile_clamped=tile_clamped,
        is_8k=out_label == "8K",
    )
    summary = format_summary(
        src_w=src_w, src_h=src_h, frame_count=frames, plan_fields=fields
    )
    return TilingPlan(summary=summary, **fields)


def _dims_from_image(image) -> tuple[int, int, int]:
    """Comfy IMAGE is [frames, height, width, channels]."""
    frames, height, width = (
        int(image.shape[0]),
        int(image.shape[1]),
        int(image.shape[2]),
    )
    return width, height, frames


@comfy_node(
    name="LTXVGetTilingSizes",
    description="Get Tiling Sizes",
)
class LTXVGetTilingSizes:
    """Size calculator in the spirit of Comfy's Get Image Size, for tiled fusion."""

    CATEGORY = "ltxvideo"
    FUNCTION = "get_sizes"
    OUTPUT_NODE = True
    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "INT",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "initial_canvas_width",
        "initial_canvas_height",
        "tile_width",
        "tile_height",
        "output_width",
        "output_height",
        "overlap_frac",
        "tile_frames",
        "is_8k",
        "summary",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 1,
                        "max": 16384,
                        "tooltip": "Source width in pixels (from Load Video / Get Image Size).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 1,
                        "max": 16384,
                        "tooltip": "Source height in pixels.",
                    },
                ),
                "frame_count": (
                    "INT",
                    {
                        "default": 97,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Source frame count. tile_frames is 8n+1, 97 or less, "
                        "or up to 121 when that avoids temporal chunking.",
                    },
                ),
                "tile_size": (
                    list(TILE_CHOICES),
                    {
                        "default": "FullHD",
                        "tooltip": "Spatial tile long-edge. qHD=960, HD=1280, FullHD=1920. "
                        "Clamped to the initial canvas so a FullHD tile on a smaller "
                        "canvas does not force tiling.",
                    },
                ),
                "initial_canvas_size": (
                    list(TILE_CHOICES),
                    {
                        "default": "FullHD",
                        "tooltip": "Stage-1 canvas long-edge (same choices as tile size). "
                        "Source within 200px of FullHD is kept instead of rescaled.",
                    },
                ),
                "output_size": (
                    list(OUTPUT_CHOICES),
                    {
                        "default": "FullHD",
                        "tooltip": "Final canvas. FullHD is the 1920 long-edge step; "
                        "4K is 2× that; 8K is 4×. Single-stage graphs use this as the "
                        "canvas. Two-stage graphs use initial canvas for stage 1 and "
                        "this for stage 2.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional video/image batch. When connected, width, "
                        "height, and frame_count are taken from it (Get Image Size style)."
                    },
                ),
            },
        }

    def get_sizes(
        self,
        width,
        height,
        frame_count,
        tile_size="FullHD",
        initial_canvas_size="FullHD",
        output_size="FullHD",
        image: Optional[object] = None,
        **_kwargs,
    ):
        if image is not None:
            width, height, frame_count = _dims_from_image(image)
        plan = resolve_tiling_plan(
            width,
            height,
            frame_count,
            tile_size=tile_size,
            initial_canvas_size=initial_canvas_size,
            output_size=output_size,
        )
        return {
            "ui": {"text": [plan.summary]},
            "result": (
                plan.initial_canvas_width,
                plan.initial_canvas_height,
                plan.tile_width,
                plan.tile_height,
                plan.output_width,
                plan.output_height,
                plan.overlap_frac,
                plan.tile_frames,
                plan.is_8k,
                plan.summary,
            ),
        }
