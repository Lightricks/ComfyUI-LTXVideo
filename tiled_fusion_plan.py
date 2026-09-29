"""Shared temporal-window plan for tiled fusion.

The guide node encodes one fresh IC-LoRA clip per window and writes the plan
onto conditioning. The sampler reads it so window w always pairs with guide
block w. Both sides must use the same origins; slicing a whole-clip causal
guide cannot be repaired by rewriting coordinates.
"""

from __future__ import annotations

import os

# Minimum temporal overlap fraction. Keep in sync with the sampler env knobs.
_T_OV = float(os.environ.get("LTXV_TILED_FUSION_T_OVERLAP", "0.5"))

TEMPORAL_PLAN_KEY = "ltxv_tiled_fusion_temporal_plan"
GEAR_TEMPORAL_PLAN_KEY = "gear_temporal_plan"
FRAME_STEP = 8


def snap_frames(n: int) -> int:
    """Nearest 8n+1 length. Ties keep the shorter clip (no pad)."""
    n = max(1, int(n))
    below = ((n - 1) // FRAME_STEP) * FRAME_STEP + 1
    above = below + FRAME_STEP
    if above - n < n - below:
        return above
    return below


def plan_grid(canvas_lat, tile_lat, min_overlap):
    """Latent-aligned tile origins covering [0, canvas_lat) with >= min_overlap."""
    if canvas_lat <= tile_lat:
        return [0]
    tile_lat = max(1, int(tile_lat))
    canvas_lat = int(canvas_lat)
    # Overlap cannot reach tile_lat, so an unclamped request would never exit.
    min_overlap = min(max(0, min_overlap), tile_lat - 1)
    n = 2
    while n < canvas_lat and (n * tile_lat - canvas_lat) / (n - 1) < min_overlap:
        n += 1
    step = (canvas_lat - tile_lat) / (n - 1)
    return [round(i * step) for i in range(n)]


def shifted(base, canvas_lat, tile_lat, off):
    """Shift interior origins by `off`; endpoints stay pinned so coverage holds."""
    out = []
    for i, v in enumerate(base):
        if i in (0, len(base) - 1):
            out.append(v)
        else:
            out.append(max(1, min(canvas_lat - tile_lat - 1, v + off)))
    return sorted(set(out))


def temporal_plan(n_lat, tf_lat):
    """Shared window plan so a guide node and the sampler always agree."""
    return plan_grid(n_lat, tf_lat, max(0, int(tf_lat * _T_OV)))


def attach_temporal_plan(cond, plan):
    """Write the plan onto every conditioning entry."""
    out = []
    for emb, d in cond:
        d = dict(d)
        d[TEMPORAL_PLAN_KEY] = plan
        out.append([emb, d])
    return out


def read_temporal_plan(positive):
    """Return the plan from conditioning, including the Gear-era key."""
    for _, d in positive:
        if not isinstance(d, dict):
            continue
        plan = d.get(TEMPORAL_PLAN_KEY) or d.get(GEAR_TEMPORAL_PLAN_KEY)
        if plan:
            return plan
    return None
