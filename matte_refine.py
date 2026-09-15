"""User-facing matte cleanup after IC-LoRA alpha generation."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import InputImpl, io
from PIL import Image

from .alpha_composite import (
    join_rgba,
    match_length,
    match_mask_to_rgb,
    over_composite,
    mask_from_matte,
    resolve_plate,
    resize_rgb,
    rgb_frames,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, comfy_node

_LIVE_MAX_SIDE = 320
_LIVE_MAX_FRAMES = 240
_PLATE_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _live_b64(frame: torch.Tensor, jpeg: bool = False) -> str:
    t = frame.detach().float().cpu()
    if t.ndim == 2:
        img = Image.fromarray((t.clamp(0, 1) * 255).byte().numpy(), mode="L")
        fmt, kw = "PNG", {"compress_level": 6}
    else:
        img = Image.fromarray((t[..., :3].clamp(0, 1) * 255).byte().numpy(), mode="RGB")
        fmt, kw = ("JPEG", {"quality": 60}) if jpeg else ("PNG", {"compress_level": 6})
    buf = BytesIO()
    img.save(buf, format=fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _index_time(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(0, idx.to(device=tensor.device))


def _live_clip(
    source: torch.Tensor,
    rgb: torch.Tensor,
    plate: torch.Tensor | None = None,
) -> tuple[list[str], list[str], list[str], float, int]:
    n_source = int(source.shape[0])
    n, h, w = n_source, int(source.shape[-2]), int(source.shape[-1])
    if n > _LIVE_MAX_FRAMES:
        idx = torch.linspace(0, n - 1, _LIVE_MAX_FRAMES, device=source.device).round().long()
        source = _index_time(source, idx)
        rgb = _index_time(rgb, idx)
        if plate is not None:
            plate = _index_time(plate, idx)
        n = int(source.shape[0])
    scale = min(1.0, _LIVE_MAX_SIDE / float(max(h, w)))
    if scale < 1.0:
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        source = F.interpolate(
            source.unsqueeze(1), size=(nh, nw), mode="bilinear", align_corners=False
        ).squeeze(1)
        rgb = resize_rgb(rgb, nh, nw)
        if plate is not None:
            plate = resize_rgb(plate, nh, nw)
    source_cpu = source.detach().float().cpu()
    rgb_cpu = rgb.detach().float().cpu()
    masks = [_live_b64(source_cpu[i], jpeg=False) for i in range(n)]
    rgbs = [_live_b64(rgb_cpu[i], jpeg=True) for i in range(n)]
    bgs: list[str] = []
    if plate is not None:
        plate_cpu = plate.detach().float().cpu()
        bgs = [_live_b64(plate_cpu[i], jpeg=True) for i in range(n)]
    return masks, rgbs, bgs, float(scale), n_source


def _expand_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Square dilate (positive) or erode (negative) in pixel units.

    Replicate-pad before the pool so the border matches the live preview (which
    clamps coordinates) instead of max_pool2d's zero fill.
    """
    if pixels == 0:
        return mask
    x = mask.unsqueeze(1)
    pad = abs(int(pixels))
    k = pad * 2 + 1
    if pixels < 0:
        x = 1.0 - x
    x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
    x = F.max_pool2d(x, kernel_size=(1, k), stride=1)
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    x = F.max_pool2d(x, kernel_size=(k, 1), stride=1)
    if pixels < 0:
        x = 1.0 - x
    return x.squeeze(1)


def _gaussian_blur(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return mask
    radius = max(1, int(round(float(sigma) * 3.0)))
    k = radius * 2 + 1
    coords = torch.arange(k, device=mask.device, dtype=mask.dtype) - radius
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    x = mask.unsqueeze(1)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, k, 1))
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, 1, k))
    return x.squeeze(1)


def _gaussian_blur_rgb(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """NHWC RGB, same separable blur as `_gaussian_blur`."""
    if sigma <= 0:
        return image
    n, h, w, c = image.shape
    x = image.movedim(-1, 1).reshape(n * c, h, w)
    return _gaussian_blur(x, sigma).reshape(n, c, h, w).movedim(1, -1)


_UNMIX_FLOOR = 0.1
_UNMIX_GAIN_FLOOR = 0.25


def _uniform_clamp(rgb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Move `rgb` toward `target` as far as stays in gamut, scaling all channels
    together. Per-channel clamping would clip one channel first and shift hue."""
    delta = target - rgb
    big = torch.full_like(delta, 1e6)
    to_high = torch.where(delta > 0, (1.0 - rgb) / delta.clamp_min(1e-6), big)
    to_low = torch.where(delta < 0, rgb / (-delta).clamp_min(1e-6), big)
    t = torch.minimum(to_high, to_low).amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
    return rgb + t * delta


def _unmix_color(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    amount: float,
    radius: float,
) -> torch.Tensor:
    """Recover straight foreground color: C = αF + (1-α)B.

    Estimates B from low-alpha pixels, then F = (C - (1-α)B) / α, which is already
    identity at α=1. Two guards keep a bad α from turning into a colored fringe:
    the divisor is floored at `_UNMIX_GAIN_FLOOR` so background removal is never
    amplified more than ~3x, and the result is pulled back into gamut uniformly
    instead of per channel. Below `_UNMIX_FLOOR` the correction fades out, since
    α is then too small to solve for F at all. Amount 0 is identity.
    """
    if amount <= 0:
        return rgb
    a = alpha.clamp(0.0, 1.0)
    bg_w = (1.0 - a) * (1.0 - a)
    sigma = max(0.5, float(radius))
    num = _gaussian_blur_rgb(rgb * bg_w.unsqueeze(-1), sigma)
    den = _gaussian_blur(bg_w, sigma).unsqueeze(-1).clamp_min(1e-4)
    background = (num / den).clamp(0.0, 1.0)
    safe = a.unsqueeze(-1).clamp_min(_UNMIX_GAIN_FLOOR)
    foreground = _uniform_clamp(rgb, (rgb - (1.0 - safe) * background) / safe)
    gate = (a / _UNMIX_FLOOR).clamp(0.0, 1.0).unsqueeze(-1)
    mix = float(amount) * gate
    return (rgb * (1.0 - mix) + foreground * mix).clamp(0.0, 1.0)


def _clip_levels(mask: torch.Tensor, black_clip: float, white_clip: float) -> torch.Tensor:
    lo = float(black_clip)
    hi = 1.0 - float(white_clip)
    if lo <= 0.0 and hi >= 1.0:
        return mask
    if hi <= lo:
        hi = lo + 1e-4
    return ((mask - lo) / (hi - lo)).clamp(0.0, 1.0)


def _refine_mask(
    mask: torch.Tensor,
    invert: bool,
    expand: int,
    feather: float,
    black_clip: float,
    white_clip: float,
    mix: float,
) -> torch.Tensor:
    work = mask
    if invert:
        work = 1.0 - work
    raw = work
    mix = min(1.0, max(0.0, float(mix)))
    if mix <= 0.0:
        return raw.clamp(0.0, 1.0)
    work = _expand_mask(work, int(expand))
    work = _clip_levels(work, black_clip, white_clip)
    work = _gaussian_blur(work, float(feather))
    if mix < 1.0:
        work = torch.lerp(raw, work, mix)
    return work.clamp(0.0, 1.0)


def _load_plate_file(name: str) -> torch.Tensor:
    """Read a still or video from Comfy's input folder as NHWC RGB."""
    name = (name or "").strip().replace("\\", "/")
    parts = Path(name).parts
    if not name or Path(name).is_absolute() or ".." in parts:
        raise ValueError(f"Invalid plate file: {name!r}")
    exists = getattr(folder_paths, "exists_annotated_filepath", None)
    if callable(exists) and not exists(name):
        raise ValueError(f"Plate file not found: {name}")
    path = folder_paths.get_annotated_filepath(name)
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        mtime = 0
    key = (str(path), int(mtime))
    cached = _PLATE_CACHE.get(key)
    if cached is not None:
        return cached
    images = InputImpl.VideoFromFile(path).get_components().images
    if images is None or int(images.shape[0]) == 0:
        raise ValueError(f"Could not read plate file: {name}")
    frames = rgb_frames(images)
    _PLATE_CACHE.clear()
    _PLATE_CACHE[key] = frames
    return frames


@comfy_node(name="LTXRefineAlphaMatte")
class LTXRefineAlphaMatte(io.ComfyNode):
    """Post-process a generated alpha matte. Identity at default widget values."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXRefineAlphaMatte",
            display_name=NODES_DISPLAY_NAME_PREFIX + " Refine Alpha Matte",
            category="Lightricks/IC-LoRA",
            description=(
                "Cleanup for IC-LoRA alpha mattes, plus optional over-plate composite. "
                "Defaults leave the matte and RGB unchanged. Connect Decode via Get Video "
                "Components to `matte` (white = subject); `mask` still works. Expand is "
                "Shift Edge (negative = choke). Feather blurs the silhouette. Black/white "
                "clip is Resolve Clean Black / Clean White. Unmix pulls old-background "
                "color out of mixed edges. `rgba` is straight alpha. `over` is "
                "the opaque composite when background is image/video. Use **Load plate** "
                "on this node (no extra Load Image / Load Video nodes). Live preview uses "
                "the same plate as Queue Prompt."
            ),
            inputs=[
                io.Image.Input(
                    "matte",
                    optional=True,
                    tooltip="Matte frames (white = subject). Connect Get Video Components from Decode. Optional if `mask` is connected.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Source RGB for unmix, RGBA, and over. Gray placeholder if unconnected.",
                ),
                io.Boolean.Input(
                    "invert",
                    default=False,
                    label_on="inverted",
                    label_off="normal",
                    tooltip="Swap subject and background.",
                ),
                io.Int.Input(
                    "expand",
                    display_name="expand (px)",
                    default=0,
                    min=-128,
                    max=128,
                    tooltip="Shift Edge in pixels. Positive grows the subject, negative chokes it (kills halos).",
                ),
                io.Float.Input(
                    "feather",
                    default=0.0,
                    min=0.0,
                    max=64.0,
                    step=0.5,
                    tooltip="Gaussian blur of the silhouette in pixels. 2–8 is a typical AE feather.",
                ),
                io.Float.Input(
                    "black_clip",
                    display_name="clean black",
                    default=0.0,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip="Crush near-transparent gray specks in the background to 0.",
                ),
                io.Float.Input(
                    "white_clip",
                    display_name="clean white",
                    default=0.0,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip="Fill near-opaque gray holes in the subject to 1.",
                ),
                io.Float.Input(
                    "mix",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0 = raw matte (after invert), 1 = fully refined.",
                ),
                io.Float.Input(
                    "unmix",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Decontaminate edge color. 0 = original RGB. 1 = fully unmix mixed "
                        "pixels so they keep the subject color, not the old background."
                    ),
                ),
                io.Float.Input(
                    "unmix_radius",
                    display_name="unmix radius",
                    default=12.0,
                    min=1.0,
                    max=64.0,
                    step=0.5,
                    tooltip=(
                        "How far (pixels) to look for background color when unmixing. "
                        "Raise for long hair/fur; lower if nearby objects tint the edge."
                    ),
                ),
                io.Combo.Input(
                    "background",
                    options=["none", "image", "video"],
                    default="none",
                    tooltip=(
                        "none = original RGB on `over`. image / video = the file from "
                        "**Load plate** (or a wired background_image / background_video)."
                    ),
                ),
                io.Combo.Input(
                    "fit",
                    options=["cover", "contain", "stretch"],
                    default="cover",
                    tooltip="How the plate fills the frame: cover crops, contain letterboxes, stretch distorts.",
                ),
                io.Combo.Input(
                    "length",
                    options=["hold", "loop"],
                    default="hold",
                    tooltip="If the plate is shorter than the clip: hold last frame or loop.",
                ),
                io.String.Input(
                    "plate_file",
                    default="",
                    socketless=True,
                    tooltip="Set by Load plate. Filename in Comfy's input folder.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    advanced=True,
                    tooltip="Optional MASK. Used only if you do not connect `matte`.",
                ),
                io.Image.Input(
                    "background_image",
                    optional=True,
                    lazy=True,
                    advanced=True,
                    tooltip="Optional still from another node. Load plate does not need this.",
                ),
                io.Video.Input(
                    "background_video",
                    optional=True,
                    lazy=True,
                    advanced=True,
                    tooltip="Optional clip from another node. Load plate does not need this.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "over",
                    tooltip="Opaque composite over the plate (or original RGB when background=none). Width/height padded to even for H.264; rgba/mask/image keep the source size.",
                ),
                io.Image.Output(
                    "rgba",
                    tooltip="Straight RGBA from the refined matte (alpha 1 = keep). Use for WebM / EXR.",
                ),
                io.Mask.Output("mask", tooltip="Refined matte. White = keep."),
                io.Image.Output(
                    "image",
                    tooltip="Source RGB after color unmix. Same as input when unmix is 0.",
                ),
            ],
        )

    @classmethod
    def check_lazy_status(
        cls,
        matte=None,
        image=None,
        invert=False,
        expand=0,
        feather=0.0,
        black_clip=0.0,
        white_clip=0.0,
        mix=1.0,
        unmix=0.0,
        unmix_radius=12.0,
        background="none",
        fit="cover",
        length="hold",
        plate_file="",
        mask=None,
        background_image=None,
        background_video=None,
        **_kwargs,
    ):
        needed = []
        if background == "image" and background_image is None and not plate_file:
            needed.append("background_image")
        if background == "video" and background_video is None and not plate_file:
            needed.append("background_video")
        return needed

    @classmethod
    def execute(
        cls,
        matte: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        invert: bool = False,
        expand: int = 0,
        feather: float = 0.0,
        black_clip: float = 0.0,
        white_clip: float = 0.0,
        mix: float = 1.0,
        unmix: float = 0.0,
        unmix_radius: float = 12.0,
        background: str = "none",
        fit: str = "cover",
        length: str = "hold",
        plate_file: str = "",
        mask: torch.Tensor | None = None,
        background_image: torch.Tensor | None = None,
        background_video=None,
        **_kwargs,
    ) -> io.NodeOutput:
        source = mask_from_matte(matte, mask)
        work = _refine_mask(
            source,
            invert=invert,
            expand=int(expand),
            feather=float(feather),
            black_clip=float(black_clip),
            white_clip=float(white_clip),
            mix=float(mix),
        )

        n, h, w = work.shape
        if image is None:
            rgb = torch.full((n, h, w, 3), 0.5, device=work.device, dtype=work.dtype)
        else:
            rgb = resize_rgb(rgb_frames(image).to(device=work.device, dtype=work.dtype), h, w)
            if rgb.shape[0] != n:
                rgb = match_length(rgb, n, length)

        work = match_mask_to_rgb(work, n, h, w, length)
        source = match_mask_to_rgb(source, n, h, w, length)
        unmixed = _unmix_color(rgb, work, float(unmix), float(unmix_radius))
        if background != "none" and plate_file:
            loaded = None
            if background == "image" and background_image is None:
                loaded = _load_plate_file(plate_file)
                background_image = loaded
            elif background == "video" and background_video is None:
                loaded = _load_plate_file(plate_file)
                background_video = loaded
        plate = resolve_plate(
            background,
            h,
            w,
            n,
            fit,
            length,
            unmixed.device,
            unmixed.dtype,
            background_image=background_image,
            background_video=background_video,
        )
        rgba = join_rgba(unmixed, work)
        over = over_composite(unmixed, work, plate)
        live_mask, live_rgb, live_bg, live_scale, live_source = _live_clip(source, rgb, plate)
        return io.NodeOutput(
            over,
            rgba,
            work,
            unmixed.contiguous(),
            ui={
                "live_mask": live_mask,
                "live_rgb": live_rgb,
                "live_bg": live_bg,
                "live_scale": [live_scale],
                "live_source_frames": [live_source],
            },
        )
