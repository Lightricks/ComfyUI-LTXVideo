"""Shared matte → RGBA / over-plate helpers for Refine and Over Background."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def nchw_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask


def rgb_frames(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image[..., :3]


def frames_from_video(video: Any) -> torch.Tensor:
    if hasattr(video, "get_components"):
        images = video.get_components().images
    else:
        images = video
    return rgb_frames(images)


def luma_mask(image: torch.Tensor) -> torch.Tensor:
    """IMAGE matte → MASK. 4-channel uses alpha; otherwise the red channel
    (same as Comfy ImageToMask `red`, which matches a grayscale RGB matte)."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(dtype=torch.float32)
    if image.shape[-1] >= 4:
        return image[..., 3].clamp(0.0, 1.0)
    return image[..., 0].clamp(0.0, 1.0)


def mask_from_matte(
    matte: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if matte is not None:
        return luma_mask(matte)
    if mask is None:
        raise ValueError("Connect matte frames (IMAGE) or a MASK.")
    return nchw_mask(mask)


def resize_rgb(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if image.shape[1] == height and image.shape[2] == width:
        return image
    x = image.movedim(-1, 1)
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.movedim(1, -1)


def resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[-2:] == (height, width):
        return mask
    return F.interpolate(
        mask.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def match_length(plate: torch.Tensor, frames: int, length: str) -> torch.Tensor:
    count = plate.shape[0]
    if count == frames:
        return plate
    if count > frames:
        return plate[:frames]
    if length == "loop" and count > 1:
        reps = int(math.ceil(frames / count))
        return plate.repeat(reps, *([1] * (plate.ndim - 1)))[:frames]
    extra = frames - count
    return torch.cat([plate, plate[-1:].expand(extra, *([-1] * (plate.ndim - 1)))], dim=0)


def match_mask_to_rgb(
    alpha: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    length: str,
) -> torch.Tensor:
    if alpha.shape[-2:] != (height, width):
        alpha = resize_mask(alpha, height, width)
    if alpha.shape[0] == 1 and frames > 1:
        return alpha.expand(frames, -1, -1)
    if alpha.shape[0] != frames:
        return match_length(alpha.unsqueeze(-1), frames, length).squeeze(-1)
    return alpha


def fit_plate(plate: torch.Tensor, height: int, width: int, fit: str) -> torch.Tensor:
    plate = plate[..., :3]
    n, h, w, _ = plate.shape
    if h == height and w == width:
        return plate
    if fit == "stretch":
        return resize_rgb(plate, height, width)

    if fit == "contain":
        scale = min(height / h, width / w)
    else:
        scale = max(height / h, width / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if fit == "contain":
        nh = min(nh, height)
        nw = min(nw, width)
    else:
        nh = max(nh, height)
        nw = max(nw, width)
    x = resize_rgb(plate, nh, nw)
    if fit == "contain":
        canvas = torch.zeros((n, height, width, 3), device=x.device, dtype=x.dtype)
        top = (height - nh) // 2
        left = (width - nw) // 2
        canvas[:, top : top + nh, left : left + nw] = x
        return canvas
    top = max(0, (nh - height) // 2)
    left = max(0, (nw - width) // 2)
    cropped = x[:, top : top + height, left : left + width]
    if cropped.shape[1] != height or cropped.shape[2] != width:
        return resize_rgb(plate, height, width)
    return cropped


def pad_even(image: torch.Tensor) -> torch.Tensor:
    """H.264 yuv420p cannot encode odd width or height."""
    n, h, w, c = image.shape
    ph = h + (h % 2)
    pw = w + (w % 2)
    if ph == h and pw == w:
        return image
    out = image.new_zeros((n, ph, pw, c))
    out[:, :h, :w] = image
    return out


def resolve_plate(
    background: str,
    height: int,
    width: int,
    frames: int,
    fit: str,
    length: str,
    device: torch.device,
    dtype: torch.dtype,
    background_image: torch.Tensor | None = None,
    background_video: Any | None = None,
) -> torch.Tensor | None:
    if background == "none":
        return None
    if background == "image":
        if background_image is None:
            raise ValueError(
                "background=image but no still is loaded. Use Load plate on this node, "
                "or connect background_image."
            )
        plate = rgb_frames(background_image).to(device=device, dtype=dtype)
    else:
        if background_video is None:
            raise ValueError(
                "background=video but no clip is loaded. Use Load plate on this node, "
                "or connect background_video."
            )
        plate = frames_from_video(background_video).to(device=device, dtype=dtype)
    plate = fit_plate(plate, height, width, fit)
    return match_length(plate, frames, length)


def over_composite(
    fg: torch.Tensor,
    alpha: torch.Tensor,
    plate: torch.Tensor | None,
) -> torch.Tensor:
    fg = fg[..., :3].contiguous()
    if plate is None:
        return pad_even(fg)
    if fg.shape[0] == 1 and plate.shape[0] > 1:
        fg = fg.expand(plate.shape[0], -1, -1, -1)
        alpha = alpha.expand(plate.shape[0], -1, -1)
    a = alpha.unsqueeze(-1).to(device=fg.device, dtype=fg.dtype)
    out = fg * a + plate * (1.0 - a)
    return pad_even(out.clamp(0.0, 1.0).contiguous())


def join_rgba(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Straight RGBA. Alpha 1 = keep subject (does not invert like Join Image with Alpha)."""
    rgb = rgb[..., :3]
    a = alpha.unsqueeze(-1).to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
    return torch.cat([rgb, a], dim=-1).contiguous()
