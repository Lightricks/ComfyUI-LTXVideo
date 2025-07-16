from typing import Tuple

import comfy
import torch

from .nodes_registry import comfy_node


@comfy_node(
    name="LTXVFilmGrain",
)
class LTXVFilmGrain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "grain_intensity": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_film_grain"
    CATEGORY = "effects"
    DESCRIPTION = "Adds film grain to the image."

    def add_film_grain(
        self, images: torch.Tensor, grain_intensity: float, saturation: float
    ) -> Tuple[torch.Tensor]:
        if grain_intensity < 0 or grain_intensity > 1:
            raise ValueError("Grain intensity must be between 0 and 1.")
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        grain = torch.zeros(images[0:1].shape, device=device)

        # Process images in-place to reduce memory usage
        for i in range(images.shape[0]):
            # Generate grain for single image
            torch.randn(grain.shape, device=device, out=grain)
            grain[:, :, :, 0] *= 2
            grain[:, :, :, 2] *= 3
            grain = grain * saturation + grain[:, :, :, 1].unsqueeze(3).expand(
                -1, -1, -1, 3
            ) * (1 - saturation)

            # Blend the grain with the image in-place
            images[i : i + 1].add_(grain_intensity * grain)
            images[i : i + 1].clamp_(0, 1)

        images = images.to(comfy.model_management.intermediate_device())
        return (images,)
