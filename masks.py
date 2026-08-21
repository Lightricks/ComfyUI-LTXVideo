from typing import Any

import torch
from comfy.model_management import is_device_cpu
from comfy_extras.nodes_mask import GrowMask

from .nodes_registry import comfy_node


@comfy_node(name="LTXVPreprocessMasks")
class LTXVPreprocessMasks:
    """
    Preprocesses masks for use with the LTXVideo model's latent masking.

    This node handles temporal mask processing by:
    1. Validating mask dimensions against VAE downscaling factors
    2. Optionally inverting masks
    3. Handling the first frame mask separately (can be ignored)
    4. Combining temporal masks using max pooling
    5. Growing/shrinking masks with morphological operations
    6. Clamping mask values to ensure proper opacity

    The output is a set of masks ready for latent-space masking operations.
    """

    @classmethod
    def INPUT_TYPES(cls: "LTXVPreprocessMasks") -> dict[str, Any]:  # noqa: N802
        return {
            "required": {
                "masks": ("MASK",),
                "vae": ("VAE",),
                "invert_input_masks": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the input masks before processing."
                        "Useful for masking vs unmasking operations.",
                    },
                ),
                "ignore_first_mask": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Zero out the first mask, typically used when"
                        "it corresponds to the conditioning frame.",
                    },
                ),
                "pooling_method": (
                    ["max", "mean", "min"],
                    {
                        "default": "max",
                        "tooltip": "Method to combine temporal masks. Max preserves strongest values, "
                        "mean averages them, min takes weakest values.",
                    },
                ),
                "grow_mask": (
                    "INT",
                    {
                        "default": 0,
                        "min": -16384,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Pixels to grow (positive) or shrink (negative) the mask."
                        "Uses morphological operations.",
                    },
                ),
                "tapered_corners": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use tapered corners in morphological operations for smoother mask edges.",
                    },
                ),
                "clamp_min": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum value to clamp masks to, ensuring minimum opacity.",
                    },
                ),
                "clamp_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Maximum value to clamp masks to.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "preprocess_masks"
    CATEGORY = "Lightricks/LTXV/Masking"
    DESCRIPTION = (
        "Preprocess masks to be used for masking latents in the LTXVideo model."
    )

    def preprocess_masks(
        self,
        masks: torch.Tensor,
        vae,  # noqa: ANN001
        pooling_method: str,
        grow_mask: int,
        tapered_corners: bool,
        clamp_min: float,
        clamp_max: float,
        ignore_first_mask: bool,
        invert_input_masks: bool,
    ) -> tuple[torch.Tensor]:
        """
        Preprocess masks for temporal latent masking.

        Args:
            masks: Input masks tensor of shape (batch_size, H, W) where batch_size = 1 + (num_frames - 1)
                   The first mask is for the conditioning frame, remaining masks are for generated frames
            vae: VAE model object containing downscale_index_formula attribute
            pooling_method: Method to combine temporal masks ('max', 'mean', 'min')
            grow_mask: Pixels to grow (positive) or shrink (negative) the mask
            tapered_corners: Whether to use tapered corners in morphological operations
            clamp_min: Minimum value to clamp masks to (ensures minimum opacity)
            clamp_max: Maximum value to clamp masks to
            ignore_first_mask: Whether to zero out the first mask (often used for conditioning)
            invert_input_masks: Whether to invert all input masks (1 - mask)

        Returns:
            Tuple containing the processed masks tensor

        Raises:
            ValueError: If mask dimensions are not compatible with VAE downscaling factors
        """

        if masks.ndim != 3:
            raise ValueError("Masks must be of shape (batch_size, H, W).")

        batch_size, H, W = masks.shape

        # Get VAE downscaling factors for temporal and spatial dimensions
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )

        # Validate that mask dimensions are compatible with VAE downscaling
        # The batch size minus 1 (for the first frame) must be divisible by time_scale_factor
        if (batch_size - 1) % time_scale_factor != 0:
            raise ValueError(
                f"Masks batch size must have a multiple of {time_scale_factor} masks + 1."
            )

        if H % height_scale_factor != 0:
            raise ValueError(
                f"Masks height must have a multiple of {height_scale_factor}."
            )

        if W % width_scale_factor != 0:
            raise ValueError(
                f"Masks width must have a multiple of {width_scale_factor}."
            )

        # Optionally invert all masks (useful for masking vs unmasking operations)
        if invert_input_masks:
            masks = 1.0 - masks

        # Handle first mask separately - often used as conditioning frame
        if ignore_first_mask:
            first_mask = masks[0:1, :, :] * 0  # Zero out the first mask
        else:
            first_mask = masks[0:1, :, :]

        # Process remaining masks by grouping them according to time_scale_factor
        # and taking the maximum value across each temporal group
        remaining = masks[1:].view(-1, time_scale_factor, H, W)

        # Apply selected pooling method
        pooling_methods = {
            "max": lambda x: x.max(dim=1)[0],
            "mean": lambda x: x.mean(dim=1),
            "min": lambda x: x.min(dim=1)[0],
        }

        combined = pooling_methods[pooling_method](remaining)

        # Apply morphological operations to grow or shrink masks
        if grow_mask != 0:
            grow_mask_node = GrowMask()

            # GrowMask node is not supported on GPU, so we need to move the combined tensor to CPU
            # if it is not already on CPU
            input_device = combined.device
            if not is_device_cpu(input_device):
                combined = combined.cpu()

            (combined,) = grow_mask_node.expand_mask(
                mask=combined, expand=grow_mask, tapered_corners=tapered_corners
            )

            # Move the combined tensor back to the original device
            if not is_device_cpu(input_device):
                combined = combined.to(input_device)

        # Combine first mask with processed remaining masks and clamp values
        # Clamping ensures masks have proper opacity values
        processed_masks = torch.cat(
            [first_mask, torch.clamp(combined, clamp_min, clamp_max)], dim=0
        )

        return (processed_masks,)
