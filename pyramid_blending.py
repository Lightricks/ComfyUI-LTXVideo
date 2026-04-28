import math

import comfy.model_management
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from kornia.geometry.transform.pyramid import (
    PyrUp,
    build_laplacian_pyramid,
    build_pyramid,
    find_next_powerof_two,
    is_powerof_two,
    pad,
)
from torch import Tensor

from .nodes_registry import comfy_node

_CHUNK_SIZE = 8
_MASK_LOW_RES_LONG_SIDE = 64


def _pad_for_laplacian(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad an image tensor so H and W are powers of two (kornia requirement)."""
    h, w = image.shape[2], image.shape[3]
    pad_right = 0
    pad_down = 0
    if not (is_powerof_two(h) and is_powerof_two(w)):
        pad_right = find_next_powerof_two(w) - w
        pad_down = find_next_powerof_two(h) - h
        image = pad(image, (0, pad_right, 0, pad_down), "reflect")
    return image, (pad_right, pad_down)


def _gaussian_pyramid(
    images: torch.Tensor,
    max_level: int,
    border_type: str = "reflect",
    align_corners: bool = False,
) -> list[Tensor]:
    h, w = images.shape[2], images.shape[3]
    if not (is_powerof_two(w) and is_powerof_two(h)):
        padding = (0, find_next_powerof_two(w) - w, 0, find_next_powerof_two(h) - h)
        images = pad(images, padding, border_type)
    return build_pyramid(images, max_level, border_type, align_corners)


def _resize_preserving_aspect_ratio(
    images: torch.Tensor, long_side: int, mode: str
) -> torch.Tensor:
    h, w = images.shape[-2:]
    current_long_side = max(h, w)
    if current_long_side == long_side:
        return images

    scale = long_side / current_long_side
    resized_h = max(1, int(round(h * scale)))
    resized_w = max(1, int(round(w * scale)))

    if mode == "nearest":
        return F.interpolate(images, size=(resized_h, resized_w), mode=mode)

    return F.interpolate(
        images,
        size=(resized_h, resized_w),
        mode=mode,
        align_corners=False,
    )


def _apply_low_res_mask_dilation(
    mask: torch.Tensor,
    spatial_radius: int,
    long_side: int = _MASK_LOW_RES_LONG_SIDE,
) -> torch.Tensor:
    if spatial_radius <= 0:
        return mask

    original_size = mask.shape[-2:]
    mask_low_res = _resize_preserving_aspect_ratio(
        mask.float(), long_side, mode="bilinear"
    )
    mask_low_res = F.max_pool2d(
        mask_low_res,
        kernel_size=spatial_radius * 2 + 1,
        stride=1,
        padding=spatial_radius,
    )
    return F.interpolate(
        mask_low_res,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )


def _pyramid_blend_chunk(
    image1: torch.Tensor,
    image2: torch.Tensor,
    mask: torch.Tensor,
    max_level: int = 7,
) -> torch.Tensor:
    """Blend a single chunk (already padded, already on device)."""
    pyramid1 = build_laplacian_pyramid(image1, max_level=max_level)
    pyramid2 = build_laplacian_pyramid(image2, max_level=max_level)
    pyramid_mask = _gaussian_pyramid(mask, max_level=max_level)
    pyr_up = PyrUp()

    output = pyramid1[-1] * pyramid_mask[-1] + pyramid2[-1] * (1 - pyramid_mask[-1])
    for i in range(len(pyramid1) - 2, -1, -1):
        residual = pyramid1[i] * pyramid_mask[i] + pyramid2[i] * (1 - pyramid_mask[i])
        output = pyr_up(output) + residual

    return output


def _pyramid_blend(
    image1: torch.Tensor,
    image2: torch.Tensor,
    mask: torch.Tensor,
    max_level: int = 7,
    device: torch.device | None = None,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    if image1.shape != image2.shape:
        raise ValueError(
            f"input images must have the same size, {image1.shape} != {image2.shape}"
        )
    if image1.shape[0] != mask.shape[0]:
        raise ValueError(
            "image_a, image_b, and mask must have the same frame count for blending"
        )
    if image1.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            "image_a, image_b, and mask must have the same spatial resolution for blending"
        )

    _, padding = _pad_for_laplacian(image1[:1])
    orig_h, orig_w = image1.shape[-2], image1.shape[-1]
    padded_min = min(orig_h + padding[1], orig_w + padding[0])
    max_level = min(max_level, int(math.log2(padded_min)))

    if any(padding):
        mask = torch.nn.functional.pad(
            mask, (0, padding[0], 0, padding[1]), mode="reflect"
        )

    B = image1.shape[0]
    results = []

    for start in range(0, B, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, B)

        img1_chunk, _ = _pad_for_laplacian(image1[start:end])
        img2_chunk, _ = _pad_for_laplacian(image2[start:end])
        mask_chunk = mask[start:end]

        if device is not None:
            img1_chunk = img1_chunk.to(device)
            img2_chunk = img2_chunk.to(device)
            mask_chunk = mask_chunk.to(device)

        out_chunk = _pyramid_blend_chunk(
            img1_chunk, img2_chunk, mask_chunk, max_level=max_level
        )
        cropped = out_chunk[..., :orig_h, :orig_w].clamp(0, 1)
        results.append(
            cropped.to(output_device) if output_device is not None else cropped
        )

    return torch.cat(results, dim=0)


@comfy_node(
    name="LTXVLaplacianPyramidBlend",
    description="LTX Laplacian Pyramid Blend",
)
class LTXVLaplacianPyramidBlend(io.ComfyNode):
    """Blend two images seamlessly using Laplacian pyramid blending with a mask."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVLaplacianPyramidBlend",
            category="Lightricks/utility",
            description="Blend two images seamlessly using Laplacian pyramid blending.",
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip="First source image.",
                ),
                io.Image.Input(
                    "image_b",
                    tooltip="Second source image.",
                ),
                io.Mask.Input(
                    "mask",
                    tooltip="Blend mask (white = image_a, black = image_b).",
                ),
                io.Boolean.Input(
                    "trim_to_shortest",
                    default=True,
                    tooltip="Trim image_a, image_b, and mask to the shortest sequence length before blending.",
                ),
                io.Int.Input(
                    "mask_low_res_dilation",
                    default=5,
                    min=0,
                    max=15,
                    tooltip="Downscale the mask to long side 64, dilate it spatially, then resize it back before blending.",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        mask: torch.Tensor,
        trim_to_shortest: bool,
        mask_low_res_dilation: int,
    ) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()

        if mask.ndim == 4:
            mask = mask[:, :, :, 0]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if image_a.shape[1:3] != image_b.shape[1:3]:
            raise ValueError(
                "image_a and image_b must have the same spatial resolution"
            )
        if image_a.shape[1:3] != mask.shape[1:3]:
            raise ValueError(
                "image_a, image_b, and mask must have the same spatial resolution"
            )

        if trim_to_shortest:
            shortest = min(image_a.shape[0], image_b.shape[0], mask.shape[0])
            image_a = image_a[:shortest]
            image_b = image_b[:shortest]
            mask = mask[:shortest]
        elif not (image_a.shape[0] == image_b.shape[0] == mask.shape[0]):
            raise ValueError(
                "image_a, image_b, and mask must have the same frame count unless trim_to_shortest is enabled"
            )

        original = image_a.permute(0, 3, 1, 2)
        target = image_b.permute(0, 3, 1, 2)
        mask_4d = mask.unsqueeze(1).float()
        mask_4d = _apply_low_res_mask_dilation(mask_4d, mask_low_res_dilation)

        result = _pyramid_blend(
            original,
            target,
            mask_4d,
            max_level=7,
            device=device,
            output_device=comfy.model_management.intermediate_device(),
        )

        return io.NodeOutput(result.permute(0, 2, 3, 1))
