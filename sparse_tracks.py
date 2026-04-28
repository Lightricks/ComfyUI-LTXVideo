import base64
import json
from io import BytesIO

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image

from .nodes_registry import comfy_node


def _catmull_rom(p0: dict, p1: dict, p2: dict, p3: dict, t: float) -> dict[str, float]:
    t2 = t * t
    t3 = t2 * t
    return {
        "x": 0.5
        * (
            2 * p1["x"]
            + (-p0["x"] + p2["x"]) * t
            + (2 * p0["x"] - 5 * p1["x"] + 4 * p2["x"] - p3["x"]) * t2
            + (-p0["x"] + 3 * p1["x"] - 3 * p2["x"] + p3["x"]) * t3
        ),
        "y": 0.5
        * (
            2 * p1["y"]
            + (-p0["y"] + p2["y"]) * t
            + (2 * p0["y"] - 5 * p1["y"] + 4 * p2["y"] - p3["y"]) * t2
            + (-p0["y"] + 3 * p1["y"] - 3 * p2["y"] + p3["y"]) * t3
        ),
    }


def _interpolate_spline(
    control_points: list[dict], num_samples: int
) -> list[dict[str, int]]:
    """Catmull-Rom spline interpolation matching the JS frontend logic."""
    if len(control_points) == 0:
        return []
    if len(control_points) == 1:
        p = control_points[0]
        return [{"x": round(p["x"]), "y": round(p["y"])} for _ in range(num_samples)]
    if len(control_points) == 2:
        a, b = control_points
        return [
            {
                "x": round(a["x"] + (b["x"] - a["x"]) * i / (num_samples - 1)),
                "y": round(a["y"] + (b["y"] - a["y"]) * i / (num_samples - 1)),
            }
            for i in range(num_samples)
        ]
    pts = [control_points[0], *control_points, control_points[-1]]
    n_seg = len(pts) - 3
    result = []
    for i in range(num_samples):
        g_t = (i / (num_samples - 1)) * n_seg
        seg = min(int(g_t), n_seg - 1)
        l_t = g_t - seg
        p = _catmull_rom(pts[seg], pts[seg + 1], pts[seg + 2], pts[seg + 3], l_t)
        result.append({"x": round(p["x"]), "y": round(p["y"])})
    return result


@comfy_node(name="LTXVSparseTrackEditor", description="LTX Sparse Track Editor")
class LTXVSparseTrackEditor(io.ComfyNode):
    """Interactive spline editor for drawing sparse motion tracks.

    Provides a canvas widget where users can draw and edit spline control
    points on top of a reference image. Outputs interpolated track
    coordinates compatible with LTXVDrawTracks.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVSparseTrackEditor",
            category="Lightricks/motion_tracking",
            description=(
                "Interactive spline editor for drawing sparse motion tracks "
                "on a reference image."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Reference image displayed as the editor canvas background.",
                ),
                io.String.Input(
                    "points_store",
                    default="[]",
                    tooltip="JSON array of spline control points managed by the editor widget.",
                ),
                io.String.Input(
                    "coordinates",
                    default="[]",
                    tooltip="JSON array of interpolated track coordinates produced by the editor.",
                ),
                io.Int.Input(
                    "points_to_sample",
                    default=121,
                    min=2,
                    max=10000,
                    tooltip="Number of points sampled along each spline curve.",
                ),
            ],
            outputs=[
                io.String.Output("tracks"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        image,
        points_store: str,
        coordinates: str,
        points_to_sample: int,
    ) -> io.NodeOutput:
        # Re-interpolate from control points so that changes to
        # points_to_sample are always respected, regardless of JS sync.
        try:
            splines = json.loads(points_store) if points_store else []
        except (json.JSONDecodeError, TypeError):
            splines = []

        if splines and isinstance(splines, list) and isinstance(splines[0], list):
            interpolated = [_interpolate_spline(sp, points_to_sample) for sp in splines]
            tracks = json.dumps(interpolated)
        elif coordinates and coordinates != "[]":
            tracks = coordinates
        else:
            tracks = "[]"

        img_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return io.NodeOutput(tracks, ui={"bg_image": [img_b64]})


def _parse_tracks(raw: str) -> list[list[dict]]:
    """Parse tracks from a JSON string, handling nested/wrapped formats."""
    parsed = json.loads(raw) if isinstance(raw, str) else raw

    if isinstance(parsed, list):
        unwrapped = []
        for item in parsed:
            unwrapped.append(json.loads(item) if isinstance(item, str) else item)
        parsed = unwrapped

    tracks: list[list[dict]] = []
    stack = [parsed]
    while stack:
        obj = stack.pop()
        if isinstance(obj, list) and len(obj) > 0:
            if isinstance(obj[0], dict) and "x" in obj[0] and "y" in obj[0]:
                tracks.append(obj)
            else:
                stack.extend(obj)
    return tracks


def _age_color_batch(ratios: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Vectorised age-ratio -> RGB [0..1] mapping on GPU.

    Gradient: blue -> green -> yellow -> red.
    """
    colors = torch.zeros(ratios.shape[0], 3, device=device)

    m1 = ratios <= 1 / 3
    tr1 = ratios[m1] * 3
    colors[m1, 1] = tr1
    colors[m1, 2] = 1 - tr1

    m2 = (ratios > 1 / 3) & (ratios <= 2 / 3)
    tr2 = (ratios[m2] - 1 / 3) * 3
    colors[m2, 0] = tr2
    colors[m2, 1] = 1

    m3 = ratios > 2 / 3
    tr3 = (ratios[m3] - 2 / 3) * 3
    colors[m3, 0] = 1
    colors[m3, 1] = 1 - tr3

    return colors


def _render_resolution(width: int, height: int, reference_short_side: int):
    """Compute the higher render resolution that preserves aspect ratio."""
    if height <= width:
        rw = int(width * reference_short_side / height)
        rh = reference_short_side
    else:
        rw = reference_short_side
        rh = int(height * reference_short_side / width)
    scale_x = rw / width
    scale_y = rh / height
    return rw, rh, scale_x, scale_y


_MIN_RADIUS = 2
_MAX_RADIUS = 8
_MAX_TRAIL = 50
_REF_SHORT_SIDE = 1080


@comfy_node(name="LTXVDrawTracks", description="LTX Draw Sparse Tracks")
class LTXVDrawTracks(io.ComfyNode):
    """GPU-accelerated sparse track renderer.

    Renders circles at a high reference resolution and downscales with
    bilinear interpolation so circle sizes match the CPU version.
    All work — rasterisation, compositing and resize — stays on GPU.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVDrawTracks",
            category="Lightricks/motion_tracking",
            description=(
                "GPU-accelerated sparse track renderer. Rasterises circles at "
                "high resolution and downscales with bilinear interpolation."
            ),
            inputs=[
                io.String.Input(
                    "tracks",
                    multiline=True,
                    tooltip="JSON string of track coordinates (list of point lists with x/y keys).",
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=8,
                    max=8192,
                    step=8,
                    tooltip="Output image width in pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=8,
                    max=8192,
                    step=8,
                    tooltip="Output image height in pixels.",
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, tracks: str, width: int, height: int) -> io.NodeOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parsed = _parse_tracks(tracks)
        if not parsed:
            blank = torch.zeros(1, height, width, 3, device=device, dtype=torch.half)
            return io.NodeOutput(blank)
        num_tracks = len(parsed)
        num_frames = max(len(t) for t in parsed)
        rw, rh, sx, sy = _render_resolution(width, height, _REF_SHORT_SIDE)

        point_xy = torch.zeros(num_tracks, num_frames, 2, device=device)
        vis = torch.zeros(num_tracks, num_frames, dtype=torch.bool, device=device)
        for i, trk in enumerate(parsed):
            coords = torch.tensor(
                [[p["x"] * sx, p["y"] * sy] for p in trk],
                dtype=torch.float32,
                device=device,
            )
            point_xy[i, : len(trk)] = coords
            vis[i, : len(trk)] = True

        max_d = 2 * _MAX_RADIUS + 3
        half_d = max_d // 2
        offsets = torch.arange(max_d, device=device) - half_d
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        template_dist_sq = oy.float().square() + ox.float().square()

        render_frames = torch.zeros(num_frames, rh, rw, 3, device=device)

        for t in range(num_frames):
            tau_min = max(0, t - _MAX_TRAIL)
            window = t - tau_min + 1

            active_xy = point_xy[:, tau_min : t + 1]
            active_vis = vis[:, tau_min : t + 1]

            ages = torch.arange(window - 1, -1, -1, device=device, dtype=torch.float32)
            ratios = 1.0 - ages / _MAX_TRAIL
            radii = _MIN_RADIUS + (_MAX_RADIUS - _MIN_RADIUS) * ratios
            colors = _age_color_batch(ratios, device)

            flat_xy = active_xy.reshape(-1, 2)
            flat_vis = active_vis.reshape(-1)
            flat_radii = radii.unsqueeze(0).expand(num_tracks, -1).reshape(-1)
            flat_colors = colors.unsqueeze(0).expand(num_tracks, -1, -1).reshape(-1, 3)

            idx = flat_vis.nonzero(as_tuple=True)[0]
            if idx.shape[0] == 0:
                continue

            pts = flat_xy[idx]
            r = flat_radii[idx]
            c = flat_colors[idx]

            flat_ages = ages.unsqueeze(0).expand(num_tracks, -1).reshape(-1)
            sort_order = flat_ages[idx].argsort(descending=True)
            pts = pts[sort_order]
            r = r[sort_order]
            c = c[sort_order]

            _rasterise_circles(
                render_frames[t], pts, r, c, template_dist_sq, half_d, max_d, rh, rw
            )

        out = torch.nn.functional.interpolate(
            render_frames.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        out = out[..., [2, 1, 0]]  # RGB -> BGR to match IC-LoRA training data format
        return io.NodeOutput(out.half())


def _rasterise_circles(
    frame: torch.Tensor,
    pts: torch.Tensor,
    radii: torch.Tensor,
    colors: torch.Tensor,
    template_dist_sq: torch.Tensor,
    half_d: int,
    max_d: int,
    H: int,
    W: int,
) -> None:
    """Stamp filled circles onto *frame* fully on-device.

    Uses ``scatter_reduce_`` with ``'amax'`` to resolve overlaps in
    painter's order (circles are expected oldest-first so the highest
    index = newest wins).
    """
    M = pts.shape[0]
    if M == 0:
        return
    device = pts.device

    # per-circle masks  [M, D, D]
    radii_sq = (radii * radii).view(M, 1, 1)
    circle_masks = template_dist_sq.unsqueeze(0) <= radii_sq

    # frame-space indices  [M, D, D]
    cx = pts[:, 0].round().long().view(M, 1, 1)
    cy = pts[:, 1].round().long().view(M, 1, 1)
    offsets_y = torch.arange(max_d, device=device).sub(half_d).view(1, max_d, 1)
    offsets_x = torch.arange(max_d, device=device).sub(half_d).view(1, 1, max_d)
    fy = (cy + offsets_y).expand(M, max_d, max_d)  # [M, D, D]
    fx = (cx + offsets_x).expand(M, max_d, max_d)  # [M, D, D]

    valid = circle_masks & (fy >= 0) & (fy < H) & (fx >= 0) & (fx < W)

    flat_fy = fy[valid]
    flat_fx = fx[valid]
    flat_lin = (flat_fy * W + flat_fx).long()

    # circle index per valid pixel (oldest=0 … newest=M-1)
    j_map = torch.arange(M, device=device, dtype=torch.float32).view(M, 1, 1)
    j_map = j_map.expand_as(valid)
    flat_j = j_map[valid]

    # priority map — highest index (newest) wins via 'amax' reduce
    priority = torch.full((H * W,), -1.0, device=device)
    priority.scatter_reduce_(0, flat_lin, flat_j, reduce="amax", include_self=False)
    priority = priority.view(H, W).long()

    has_circle = priority >= 0
    frame[has_circle] = colors[priority[has_circle]]
