# MODIFIED BY SHIFTLOOM (2026-11-21): this file was modified from the upstream
# Lightricks/ComfyUI-LTXVideo release (base 220e1d4) under the LTX-2 Community
# License. Change: anchor-aware temporal streaming. Adds the versioned anchor
# layout (read_still_anchors, select_anchor_ids, validate_anchor_layout,
# anchor_window_selection) so still LTXVAddGuide keyframes chained ahead of the
# streaming guide keep their exact frame in every covering window.
# Plans without anchors are unchanged.

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
ANCHOR_LAYOUT_KEY = "anchor_layout_version"
ANCHOR_LAYOUT_VERSION = 1


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


def _cond_value(cond, key):
    for _, d in cond:
        if isinstance(d, dict) and d.get(key) is not None:
            return d[key]
    return None


def read_still_anchors(positive, negative, samples_shape):
    """Return ([anchor records], reference_count) for stills appended to content.

    Core LTXVAddGuide appends each still after the content latents and records
    its exact pixel frame as the temporal start of its coordinate tokens. Each
    record is {frame_idx, entry_idx, token_start, token_count}. Raises
    ValueError if the references are not full-resolution single-image stills.
    """
    import comfy_extras.nodes_lt as nodes_lt

    kf_p, n_ref = nodes_lt.get_keyframe_idxs(positive, samples_shape)
    if n_ref == 0:
        return [], 0
    kf_n, n_ref_n = nodes_lt.get_keyframe_idxs(negative, samples_shape)
    hw = int(samples_shape[3]) * int(samples_shape[4])
    if kf_n is None or n_ref_n != n_ref or kf_n.shape[2] != kf_p.shape[2]:
        raise ValueError(
            "anchored streaming: positive and negative conditioning disagree "
            f"on appended references ({n_ref} vs {n_ref_n})."
        )
    entries = _cond_value(positive, "guide_attention_entries") or []
    if len(entries) != n_ref or kf_p.shape[2] != n_ref * hw:
        raise ValueError(
            "anchored streaming supports only full-resolution single-image "
            f"anchors before the video guide; found {len(entries)} attention "
            f"entries and {kf_p.shape[2]} coordinate tokens for {n_ref} "
            f"references of {hw} tokens each."
        )
    anchors = []
    for i, e in enumerate(entries):
        f, gh, gw = (int(v) for v in e["latent_shape"])
        if (f, gh, gw) != (1, int(samples_shape[3]), int(samples_shape[4])):
            raise ValueError(
                f"anchored streaming: reference {i} has latent shape "
                f"{[f, gh, gw]}, expected a single full-resolution frame."
            )
        tok = i * hw
        starts = kf_p[0, 0, tok : tok + hw, 0]
        if not bool((starts == starts[0]).all()):
            raise ValueError(
                f"anchored streaming: reference {i} coordinates do not share "
                "one start frame."
            )
        anchors.append(
            {
                "frame_idx": int(round(float(starts[0]))),
                "entry_idx": i,
                "token_start": tok,
                "token_count": hw,
            }
        )
    return anchors, n_ref


def select_anchor_ids(frames, t0, t_scale, span_px):
    """Indices (original order) of anchors whose frame lies in the window."""
    lo = int(t0) * int(t_scale)
    return [i for i, f in enumerate(frames) if lo <= int(f) < lo + int(span_px)]


def validate_anchor_layout(plan, positive, negative, samples_shape):
    """Validate a version-1 anchored plan against the latent and conditionings.

    Returns the plan or raises a descriptive ValueError before any denoising.
    """
    ver = plan.get(ANCHOR_LAYOUT_KEY)
    if ver != ANCHOR_LAYOUT_VERSION:
        raise ValueError(
            f"unsupported anchor_layout_version {ver!r}; this pack understands "
            f"version {ANCHOR_LAYOUT_VERSION}."
        )
    try:
        c = int(plan["n_content"])
        tf_lat = int(plan["tf_lat"])
        ts = [int(v) for v in plan["ts"]]
        anchors = plan["anchors"]
        windows = plan["windows"]
        int(plan["time_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"anchored temporal plan is malformed: {exc!r}") from exc
    a = len(anchors)
    n_w = len(ts)
    if len(windows) != n_w:
        raise ValueError(
            f"anchored plan lists {len(windows)} windows for {n_w} origins."
        )
    expected_time = c + a + n_w * tf_lat
    if int(samples_shape[2]) != expected_time:
        raise ValueError(
            f"anchored plan expects {expected_time} latent frames "
            f"({c} content + {a} anchors + {n_w}x{tf_lat} guide), "
            f"the latent has {samples_shape[2]}."
        )
    hw = int(samples_shape[3]) * int(samples_shape[4])
    for side, cond in (("positive", positive), ("negative", negative)):
        kf = _cond_value(cond, "keyframe_idxs")
        entries = _cond_value(cond, "guide_attention_entries")
        if kf is None or entries is None:
            raise ValueError(f"anchored plan: {side} conditioning lacks guide metadata.")
        if len(entries) != a + n_w:
            raise ValueError(
                f"anchored plan expects {a + n_w} attention entries on "
                f"{side}, found {len(entries)}."
            )
        offsets = []
        tok = 0
        for e in entries:
            f, gh, gw = (int(v) for v in e["latent_shape"])
            if (gh, gw) != (int(samples_shape[3]), int(samples_shape[4])):
                raise ValueError(
                    f"anchored plan: {side} entry grid {gh}x{gw} does not "
                    "match the latent."
                )
            offsets.append((tok, f))
            tok += f * gh * gw
        if tok != kf.shape[2]:
            raise ValueError(
                f"anchored plan: {side} entries cover {tok} tokens but "
                f"keyframe_idxs holds {kf.shape[2]}."
            )
        for i, rec in enumerate(anchors):
            t_off, f = offsets[i]
            if (
                int(rec["latent_start"]) != c + i
                or int(rec["latent_count"]) != 1
                or int(rec["entry_idx"]) != i
                or f != 1
                or int(rec["token_start"]) != t_off
                or int(rec["token_count"]) != hw
            ):
                raise ValueError(f"anchored plan: anchor {i} disagrees on {side}.")
            if int(round(float(kf[0, 0, t_off, 0]))) != int(rec["frame_idx"]):
                raise ValueError(
                    f"anchored plan: anchor {i} frame differs from {side} coordinates."
                )
        for w, rec in enumerate(windows):
            t_off, f = offsets[a + w]
            if (
                int(rec["video_start"]) != c + a + w * tf_lat
                or int(rec["video_count"]) != tf_lat
                or int(rec["video_entry"]) != a + w
                or f != tf_lat
                or int(rec["video_token_start"]) != t_off
                or int(rec["video_token_count"]) != tf_lat * hw
            ):
                raise ValueError(f"anchored plan: window {w} disagrees on {side}.")
            for i in rec["anchor_ids"]:
                if not 0 <= int(i) < a:
                    raise ValueError(f"anchored plan: window {w} names anchor {i}.")
    return plan


def anchor_window_selection(plan, w):
    """Ordered gather for window w of a validated anchored plan.

    Returns (ranges, entry_indices, temporal_shifts). ranges are half-open
    latent ranges in the physical latent: content first, then the selected
    stills, then the window's own video block. entry_indices and temporal_shifts
    pair with the stills then the video block; shifts are pixel-time additions
    (minus the window origin for stills, zero for the video).
    """
    t0 = int(plan["ts"][w])
    rec = plan["windows"][w]
    origin_px = t0 * int(plan["time_scale"])
    tf_lat = int(plan["tf_lat"])
    ranges = [(t0, t0 + tf_lat)]
    entries = []
    shifts = []
    for i in rec["anchor_ids"]:
        a = plan["anchors"][int(i)]
        s = int(a["latent_start"])
        ranges.append((s, s + int(a["latent_count"])))
        entries.append(int(a["entry_idx"]))
        shifts.append(-origin_px)
    vs = int(rec["video_start"])
    ranges.append((vs, vs + int(rec["video_count"])))
    entries.append(int(rec["video_entry"]))
    shifts.append(0)
    return ranges, entries, shifts
