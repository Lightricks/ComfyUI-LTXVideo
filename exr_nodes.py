"""
HDR exposure blending node for ComfyUI.

This module provides the ExposureBlendNode for blending multiple exposures
to create HDR images with preserved highlights and shadows.
"""

import torch

from .nodes_registry import comfy_node


@comfy_node(name="HDRExposureBlend")
class ExposureBlendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bright_exposure": (
                    "IMAGE",
                    {"tooltip": "Brighter exposure image - preserves shadows."},
                ),
                "dark_exposure": (
                    "IMAGE",
                    {"tooltip": "Darker exposure image - preserves highlights."},
                ),
                "transition_smoothness": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05},
                ),
                "exposure_offset_bright": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 6.0, "step": 0.5},
                ),
                "exposure_offset_dark": (
                    "FLOAT",
                    {"default": -2.0, "min": -6.0, "max": 0.0, "step": 0.5},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("blended_hdr", "blend_mask")
    FUNCTION = "blend_exposures"
    CATEGORY = "Lightricks/EXR"

    def _calculate_luminance(self, img: torch.Tensor) -> torch.Tensor:
        """Exact Rec.709 coefficients used in nodes_hdr."""
        return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    def blend_exposures(
        self,
        dark_exposure: torch.Tensor,
        bright_exposure: torch.Tensor,
        transition_smoothness: float = 0.3,
        exposure_offset_dark: float = -2.0,
        exposure_offset_bright: float = 2.0,
    ):
        # Ensure tensors are float32 for HDR processing
        dark_exposure = dark_exposure.float()
        bright_exposure = bright_exposure.float()

        batch_size = dark_exposure.shape[0]
        results = []
        masks = []

        for i in range(batch_size):
            low_exp = dark_exposure[i] * (2.0 ** (-exposure_offset_dark))
            high_exp = bright_exposure[i] * (2.0 ** (-exposure_offset_bright))

            lum_low = self._calculate_luminance(low_exp)

            # Sigmoid-based transition
            mid_point = 0.5
            # Prevent division by zero
            smoothness = max(transition_smoothness, 1e-5)
            x = (lum_low - mid_point) / smoothness

            low_weight = 1.0 / (1.0 + torch.exp(-x))
            high_weight = 1.0 - low_weight

            # Add channel dimension for broadcasting
            low_weight_expanded = low_weight.unsqueeze(-1)
            high_weight_expanded = high_weight.unsqueeze(-1)

            blended_frame = (
                low_exp * low_weight_expanded + high_exp * high_weight_expanded
            )

            blended_frame = torch.clamp(blended_frame, min=0.0)

            # Create mask with 3 channels
            mask_frame = low_weight_expanded.expand(-1, -1, 3)

            results.append(blended_frame)
            masks.append(mask_frame)

        result_batch = torch.stack(results, dim=0)
        mask_batch = torch.stack(masks, dim=0)

        return (result_batch, mask_batch)
