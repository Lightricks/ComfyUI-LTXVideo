from typing import Optional

import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import numpy as np
import torch
from comfy.ldm.lightricks.vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR
from comfy.nested_tensor import NestedTensor

from .nodes_registry import comfy_node


@comfy_node(name="LTXVSelectLatents")
class LTXVSelectLatents:
    """
    Selects a range of frames from a video latent.

    Features:
    - Supports positive and negative indexing
    - Preserves batch processing capabilities
    - Handles noise masks if present
    - Maintains 5D tensor format
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "start_index": (
                    "INT",
                    {"default": 0, "min": -9999, "max": 9999, "step": 1},
                ),
                "end_index": (
                    "INT",
                    {"default": -1, "min": -9999, "max": 9999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "select_latents"
    CATEGORY = "latent/video"
    DESCRIPTION = (
        "Selects a range of frames from the video latent. "
        "start_index and end_index define a closed interval (inclusive of both endpoints)."
    )

    def select_latents(self, samples: dict, start_index: int, end_index: int) -> tuple:
        """
        Selects a range of frames from the video latent.

        Args:
            samples (dict): Video latent dictionary
            start_index (int): Starting frame index (supports negative indexing)
            end_index (int): Ending frame index (supports negative indexing)

        Returns:
            tuple: Contains modified latent dictionary with selected frames

        Raises:
            ValueError: If indices are invalid
        """
        try:
            s = samples.copy()
            video_latent = s["samples"]
            batch, channels, frames, height, width = video_latent.shape

            # Handle negative indices
            start_idx = frames + start_index if start_index < 0 else start_index
            end_idx = frames + end_index if end_index < 0 else end_index

            # Validate and clamp indices
            start_idx = max(0, min(start_idx, frames - 1))
            end_idx = max(0, min(end_idx, frames - 1))
            if start_idx > end_idx:
                start_idx = min(start_idx, end_idx)

            # Select frames while maintaining 5D format
            s["samples"] = video_latent[:, :, start_idx : end_idx + 1, :, :]

            # Handle noise mask if present
            if "noise_mask" in s and s["noise_mask"] is not None:
                s["noise_mask"] = s["noise_mask"][:, :, start_idx : end_idx + 1, :, :]

            return (s,)

        except Exception as e:
            print(f"[LTXVSelectLatents] Error: {str(e)}")
            raise


@comfy_node(name="LTXVAddLatents")
class LTXVAddLatents:
    """
    Concatenates two video latents along the frames dimension.

    Features:
    - Validates dimension compatibility
    - Handles device placement
    - Preserves noise masks with proper handling
    - Supports batch processing
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents1": ("LATENT",),
                "latents2": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_latents"
    CATEGORY = "latent/video"
    DESCRIPTION = (
        "Concatenates two video latents along the frames dimension. "
        "latents1 and latents2 must have the same dimensions except for the frames dimension."
    )

    def add_latents(
        self, latents1: torch.Tensor, latents2: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenates two video latents along the frames dimension.

        Args:
            latents1 (dict): First video latent dictionary
            latents2 (dict): Second video latent dictionary

        Returns:
            tuple: Contains concatenated latent dictionary

        Raises:
            ValueError: If latent dimensions don't match
            RuntimeError: If tensor operations fail
        """
        try:
            s = latents1.copy()
            video_latent1 = latents1["samples"]
            video_latent2 = latents2["samples"]

            # Ensure tensors are on the same device
            target_device = video_latent1.device
            video_latent2 = video_latent2.to(target_device)

            # Validate dimensions
            self._validate_dimensions(video_latent1, video_latent2)

            # Concatenate along frames dimension
            s["samples"] = torch.cat([video_latent1, video_latent2], dim=2)

            # Handle noise masks
            s["noise_mask"] = self._merge_noise_masks(
                latents1, latents2, video_latent1.shape[2], video_latent2.shape[2]
            )

            return (s,)

        except Exception as e:
            print(f"[LTXVAddLatents] Error: {str(e)}")
            raise

    def _validate_dimensions(self, latent1: torch.Tensor, latent2: torch.Tensor):
        """Validates that latent dimensions match except for frames."""
        # video latents
        if latent1.ndim == 5 and latent2.ndim == 5:
            b1, c1, f1, h1, w1 = latent1.shape
            b2, c2, f2, h2, w2 = latent2.shape

            if not (b1 == b2 and c1 == c2 and h1 == h2 and w1 == w2):
                raise ValueError(
                    f"Latent dimensions must match (except frames dimension).\n"
                    f"Got shapes {latent1.shape} and {latent2.shape}"
                )
        # audio latents
        elif latent1.ndim == 4 and latent2.ndim == 4:
            b1, c1, f1, s1 = latent1.shape
            b2, c2, f2, s2 = latent2.shape

            if not (b1 == b2 and c1 == c2 and s1 == s2):
                raise ValueError(
                    f"Latent dimensions must match (except frames dimension).\n"
                    f"Got shapes {latent1.shape} and {latent2.shape}"
                )
        else:
            raise ValueError(
                f"Latent dimensions must be 4 (audio) or 5 (video) for both inputs.\n"
                f"Got shapes {latent1.shape} and {latent2.shape}"
            )

    def _merge_noise_masks(
        self, latents1: torch.Tensor, latents2: torch.Tensor, frames1: int, frames2: int
    ) -> Optional[torch.Tensor]:
        """Merges noise masks from both latents with proper handling."""
        if "noise_mask" in latents1 and "noise_mask" in latents2:
            return torch.cat([latents1["noise_mask"], latents2["noise_mask"]], dim=2)
        elif "noise_mask" in latents1 and latents1["noise_mask"] is not None:
            zeros = torch.zeros_like(latents1["noise_mask"][:, :, :frames2, :, :])
            return torch.cat([latents1["noise_mask"], zeros], dim=2)
        elif "noise_mask" in latents2 and latents2["noise_mask"] is not None:
            zeros = torch.zeros_like(latents2["noise_mask"][:, :, :frames1, :, :])
            return torch.cat([zeros, latents2["noise_mask"]], dim=2)
        return None


@comfy_node(name="LTXVSetVideoLatentNoiseMasks")
class LTXVSetVideoLatentNoiseMasks:
    """
    Applies multiple masks to a video latent.

    Features:
    - Supports multiple input mask formats (2D, 3D, 4D)
    - Automatically handles fewer masks than frames by reusing the last mask
    - Resizes masks to match latent dimensions
    - Preserves batch processing capabilities

    Input Formats:
    - 2D mask: Single mask [H, W]
    - 3D mask: Multiple masks [M, H, W]
    - 4D mask: Multiple masks with channels [M, C, H, W]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"
    CATEGORY = "latent/video"
    DESCRIPTION = (
        "Applies multiple masks to a video latent. "
        "masks can be 2D, 3D, or 4D tensors. "
        "If there are fewer masks than frames, the last mask will be reused."
    )

    def set_mask(self, samples: dict, masks: torch.Tensor) -> tuple:
        """
        Applies masks to video latent frames.

        Args:
            samples (dict): Video latent dictionary containing 'samples' tensor
            masks (torch.Tensor): Mask tensor in various possible formats
                - 2D: [H, W] single mask
                - 3D: [M, H, W] multiple masks
                - 4D: [M, C, H, W] multiple masks with channels

        Returns:
            tuple: Contains modified latent dictionary with applied masks

        Raises:
            ValueError: If mask dimensions are unsupported
            RuntimeError: If tensor operations fail
        """
        try:
            s = samples.copy()
            video_latent = s["samples"]
            batch_size, channels, num_frames, height, width = video_latent.shape

            # Initialize noise_mask if not present or resize if dimensions don't match
            if "noise_mask" not in s:
                s["noise_mask"] = torch.zeros(
                    (batch_size, 1, num_frames, height, width),
                    dtype=video_latent.dtype,
                    device=video_latent.device,
                )
            else:
                existing_shape = s["noise_mask"].shape
                # Check if noise_mask has the right number of frames
                if existing_shape[2] != num_frames:
                    s["noise_mask"] = torch.zeros(
                        (batch_size, 1, num_frames, height, width),
                        dtype=video_latent.dtype,
                        device=video_latent.device,
                    )

            # Process masks
            masks_reshaped = self._reshape_masks(masks)
            M = masks_reshaped.shape[0]
            resized_masks = self._resize_masks(masks_reshaped, height, width)

            # Apply masks efficiently
            self._apply_masks(s["noise_mask"], resized_masks, num_frames, M)
            return (s,)

        except Exception as e:
            print(f"[LTXVSetVideoLatentNoiseMasks] Error: {str(e)}")
            raise

    def _reshape_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Reshapes input masks to consistent 4D format."""
        original_shape = tuple(masks.shape)
        ndims = masks.ndim

        if ndims == 2:
            return masks.unsqueeze(0).unsqueeze(0)
        elif ndims == 3:
            return masks.reshape(masks.shape[0], 1, masks.shape[1], masks.shape[2])
        elif ndims == 4:
            return masks.reshape(masks.shape[0], 1, masks.shape[2], masks.shape[3])
        else:
            raise ValueError(
                f"Unsupported 'masks' dimension: {original_shape}. "
                "Must be 2D (H,W), 3D (M,H,W), or 4D (M,C,H,W)."
            )

    def _resize_masks(
        self, masks: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Resizes all masks to match latent dimensions."""
        return torch.nn.functional.interpolate(
            masks, size=(height, width), mode="bilinear", align_corners=False
        )

    def _apply_masks(
        self,
        noise_mask: torch.Tensor,
        resized_masks: torch.Tensor,
        num_frames: int,
        M: int,
    ) -> None:
        """Applies resized masks to all frames."""
        for f in range(num_frames):
            mask_idx = min(f, M - 1)  # Reuse last mask if we run out
            noise_mask[:, :, f] = resized_masks[mask_idx]


@comfy_node(name="LTXVDilateLatent")
class LTXVDilateLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "horizontal_scale": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
                "vertical_scale": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "dilate_latent"
    CATEGORY = "latent/video"
    DESCRIPTION = "Dilates a latent by a grid size."

    def dilate_latent(
        self, latent: dict, horizontal_scale: int, vertical_scale: int
    ) -> tuple:
        if horizontal_scale == 1 and vertical_scale == 1:
            return (latent,)

        samples = latent["samples"]
        mask = latent.get("noise_mask", None)
        dilated_shape = samples.shape[:3] + (
            samples.shape[3] * vertical_scale,
            samples.shape[4] * horizontal_scale,
        )

        dilated_samples = torch.zeros(
            dilated_shape,
            device=samples.device,
            dtype=samples.dtype,
            requires_grad=False,
        )
        dilated_samples[..., ::vertical_scale, ::horizontal_scale] = samples

        dilated_mask_shape = (
            dilated_samples.shape[0],
            1,
            dilated_samples.shape[2],
            dilated_samples.shape[3],
            dilated_samples.shape[4],
        )
        dilated_mask = torch.full(
            dilated_mask_shape,
            -1.0,
            device=samples.device,
            dtype=samples.dtype,
            requires_grad=False,
        )
        dilated_mask[..., ::vertical_scale, ::horizontal_scale] = (
            mask if mask is not None else 1.0
        )
        latent = {"samples": dilated_samples, "noise_mask": dilated_mask}

        return (latent,)


@comfy_node(name="LTXVAddLatentGuide")
class LTXVAddLatentGuide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "guiding_latent": ("LATENT",),
                "latent_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Latent index to start the conditioning at. Can be negative to"
                        "indicate that the conditioning is on the frames before the latent.",
                    },
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "ltxtricks"
    FUNCTION = "generate"

    DESCRIPTION = "Adds a keyframe or a video segment at a specific frame index."

    def generate(
        self, vae, positive, negative, latent, guiding_latent, latent_idx, strength
    ):
        noise_mask = nodes_lt.get_noise_mask(latent)
        latent = latent["samples"]
        guide = guiding_latent["samples"]

        # Record original (pre-dilation) guide latent shape for spatial mask downsampling
        guide_orig_shape = list(guide.shape[2:])  # [F, H_small, W_small]

        assert (
            latent.shape[4] % guide.shape[4] == 0
            and latent.shape[3] % guide.shape[3] == 0
        ), "The ratio of the height and width of the latents and optional_guiding_latents must be an integer"

        guiding_latent = LTXVDilateLatent().dilate_latent(
            guiding_latent,
            horizontal_scale=latent.shape[4] // guide.shape[4],
            vertical_scale=latent.shape[3] // guide.shape[3],
        )[0]

        guide = guiding_latent["samples"]
        guide_mask = guiding_latent.get("noise_mask", None)

        # Pre-filter token count = product of dilated spatial dims
        # (before grid_mask filtering removes padding positions)
        iclora_tokens_added = guide.shape[2] * guide.shape[3] * guide.shape[4]

        scale_factors = vae.downscale_index_formula

        if latent_idx <= 0:
            frame_idx = latent_idx * scale_factors[0]
        else:
            frame_idx = 1 + (latent_idx - 1) * scale_factors[0]

        positive, negative, latent, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive=positive,
            negative=negative,
            frame_idx=frame_idx,
            latent_image=latent,
            noise_mask=noise_mask,
            guiding_latent=guide,
            strength=strength,
            scale_factors=scale_factors,
            guide_mask=guide_mask,
        )

        # Track this guide in guide_attention_entries for per-reference attention control.
        from .iclora_attention import append_guide_attention_entry

        positive = append_guide_attention_entry(
            positive, iclora_tokens_added, guide_orig_shape
        )
        negative = append_guide_attention_entry(
            negative, iclora_tokens_added, guide_orig_shape
        )

        return (
            positive,
            negative,
            {"samples": latent, "noise_mask": noise_mask},
        )


@comfy_node(name="LTXVImgToVideoConditionOnly")
class LTXVImgToVideoConditionOnly:
    """
    Applies image conditioning to the first frames of a video latent.

    Features:
    - Takes existing latent and applies image conditioning
    - Automatically resizes image to match latent dimensions
    - Creates noise mask for strength control
    - Supports bypass mode
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "bypass": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Bypass the conditioning."},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"
    DESCRIPTION = (
        "Applies image conditioning to the first frames of an existing latent. "
        "Creates a noise mask to control conditioning strength."
    )

    def encode_image(self, image, shape, vae):

        time_scale_factor, height_scale_factor, width_scale_factor = (
            vae.downscale_index_formula
        )
        batch, channels, frames, height, width = shape
        width *= width_scale_factor
        height *= height_scale_factor

        # Resize image to match latent dimensions
        if image.shape[1] != height or image.shape[2] != width:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        else:
            pixels = image

        # Encode image (only RGB channels)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        return t

    def generate(self, image, vae, latent, strength, bypass=False):
        if bypass:
            return (latent,)

        samples = latent["samples"]
        t = self.encode_image(image, samples.shape, vae)
        # Apply encoded image to first frames
        samples[:, :, : t.shape[2]] = t

        # Create noise mask for conditioning
        conditioning_latent_frames_mask = torch.ones(
            (1, 1, samples.shape[2], 1, 1),
            dtype=torch.float32,
            device=samples.device,
        )
        conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - strength

        return ({"samples": samples, "noise_mask": conditioning_latent_frames_mask},)


def get_video_latent_blend_coefficients(
    video_frame_index_start,
    video_frame_index_end,
    video_frame_count,
    slope_len=3,
):
    """
    Returns a blend coefficient list shaped such that:
    - It is 0.0 outside the range [video_frame_index_start, video_frame_index_end]
    - It ramps up from 0.0 to 1.0 starting at video_frame_index_start-slope_len to video_frame_index_start
    - It stays at 1.0 during [video_frame_index_start, video_frame_index_end]
    - It ramps down from 1.0 to 0.0 starting at video_frame_index_end to video_frame_index_end+slope_len
    - The 'slope_len' parameter controls the sharpness (in frames) of the slope.
    """

    coeffs = [0.0] * video_frame_count

    # Clamp arguments to safe range
    video_frame_index_start = max(
        0, min(video_frame_count - 1, video_frame_index_start)
    )
    video_frame_index_end = max(
        video_frame_index_start, min(video_frame_count - 1, video_frame_index_end)
    )
    slope_len = max(1, slope_len)

    # Ramp up
    rampl_start = max(0, video_frame_index_start - slope_len)
    for i in range(rampl_start, video_frame_index_start):
        # Linear ramp, but could be smoother (sigmoid/cosine)
        coeffs[i] = (i - rampl_start + 1) / slope_len

    # Plateau
    for i in range(video_frame_index_start, video_frame_index_end + 1):
        coeffs[i] = 1.0

    # Ramp down
    rampr_end = min(video_frame_count, video_frame_index_end + slope_len + 1)
    for i in range(video_frame_index_end + 1, rampr_end):
        coeffs[i] = 1.0 - ((i - (video_frame_index_end + 1) + 1) / slope_len)
        coeffs[i] = max(0.0, coeffs[i])

    import numpy as np

    num_coeffs = len(coeffs)
    pixel_frame_length = (num_coeffs - 1) * 8 + 1

    xp = np.array([0] + list(range(1, pixel_frame_length, 8)))
    fp = np.array(coeffs)

    # Calculate the upsampled coefficients using np.interp
    pixel_frame_positions = np.arange(pixel_frame_length)
    pixel_frame_coefficients = np.interp(pixel_frame_positions, xp, fp).tolist()

    return coeffs, pixel_frame_coefficients


@comfy_node(description="LTXV Set Audio Video Mask By Time")
class LTXVSetAudioVideoMaskByTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0}),
                "end_time": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 2000.0}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 500.0}),
                "mask_video": ("BOOLEAN", {"default": True}),
                "mask_audio": ("BOOLEAN", {"default": True}),
                "mask_init_value_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0},
                ),
                "mask_init_value_audio": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0},
                ),
                "slope_len": (
                    "INT",
                    {"default": 3, "min": 1, "max": 100, "step": 1},
                ),
            },
            "optional": {
                "spatial_mask": (
                    "MASK",
                    {"default": None, "tooltip": "Spatial mask."},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "positive",
        "negative",
        "av_latent",
        "video_latent_blend_coefficients",
        "video_pixel_blend_coefficients",
    )

    FUNCTION = "run"
    CATEGORY = "utility"
    DESCRIPTION = "Sets the audio and video mask by time."

    def run(
        self,
        av_latent,
        positive,
        negative,
        model,
        vae,
        audio_vae,
        start_time,
        end_time,
        video_fps,
        mask_video,
        mask_audio,
        mask_init_value_video,
        mask_init_value_audio,
        slope_len,
        spatial_mask=None,
    ):
        from comfy.ldm.lightricks.av_model import LTXAVModel

        if model.model.diffusion_model.__class__.__name__ != "LTXAVModel":
            raise ValueError()

        ltxav: LTXAVModel = model.model.diffusion_model

        # Extract configuration from the audio VAE
        sampling_rate = audio_vae.autoencoder.sampling_rate
        mel_hop_length = audio_vae.autoencoder.mel_hop_length
        audio_latents_per_second = (
            sampling_rate / mel_hop_length / LATENT_DOWNSAMPLE_FACTOR
        )

        time_scale_factor = vae.downscale_index_formula[0]
        video_latents_per_second = video_fps / time_scale_factor

        if not isinstance(av_latent["samples"], NestedTensor):
            raise ValueError("av_latent must be a NestedTensor")

        video_samples, audio_samples = ltxav.separate_audio_and_video_latents(
            av_latent["samples"].tensors,
            None,
        )
        video_mask = torch.full(
            (
                video_samples.shape[0],
                video_samples.shape[1],
                video_samples.shape[2],
                video_samples.shape[3],
                video_samples.shape[4],
            ),
            fill_value=mask_init_value_video,
        )
        audio_mask = torch.full(
            (
                audio_samples.shape[0],
                audio_samples.shape[1],
                audio_samples.shape[2],
                audio_samples.shape[3],
            ),
            fill_value=mask_init_value_audio,
        )
        if spatial_mask is not None:
            if spatial_mask.ndim == 3:
                spatial_mask = spatial_mask.unsqueeze(0)
            if spatial_mask.ndim == 2:
                spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)
            spatial_mask = torch.nn.functional.interpolate(
                spatial_mask,
                size=(video_samples.shape[3], video_samples.shape[4]),
                mode="bilinear",
                align_corners=False,
            )
        video_latent_frame_count = video_samples.shape[2]
        audio_latent_frame_count = audio_samples.shape[2]
        video_pixel_frame_count = (video_latent_frame_count - 1) * time_scale_factor + 1
        xp = np.array(
            [0]
            + list(
                range(1, video_pixel_frame_count + time_scale_factor, time_scale_factor)
            )
        )
        video_pixel_frame_start_raw = int(round(start_time * video_fps))
        # video_frame_index_start = index of the value in xp rounding up
        video_latent_frame_index_start = np.searchsorted(
            xp, video_pixel_frame_start_raw, side="left"
        )
        video_pixel_frame_end_raw = int(round(end_time * video_fps))
        # video_frame_index_end = index of the value in xp rounding down
        video_latent_frame_index_end = (
            np.searchsorted(xp, video_pixel_frame_end_raw, side="right") - 1
        )
        audio_latent_frame_index_start = int(
            round(start_time * audio_latents_per_second)
        )
        audio_latent_frame_index_end = (
            int(round(end_time * audio_latents_per_second)) + 1
        )
        # clamping
        video_latent_frame_index_start = max(0, video_latent_frame_index_start)
        video_latent_frame_index_end = min(
            video_latent_frame_index_end, video_latent_frame_count
        )
        audio_latent_frame_index_start = max(0, audio_latent_frame_index_start)
        audio_latent_frame_index_end = min(
            audio_latent_frame_index_end, audio_latent_frame_count
        )
        print(
            "noise mask start and end indices: video (%d %d), audio (%d %d), video fps: %f, video_latents_per_second: %f, audio_latents_per_second: %f, "
            "video_latent_frame_count: %d, video_pixel_frame_count: %d, video_pixel_frame_start_raw: %d, video_pixel_frame_end_raw: %d, start_time: %f, end_time: %f"
            % (
                video_latent_frame_index_start,
                video_latent_frame_index_end,
                audio_latent_frame_index_start,
                audio_latent_frame_index_end,
                video_fps,
                video_latents_per_second,
                audio_latents_per_second,
                video_latent_frame_count,
                video_pixel_frame_count,
                video_pixel_frame_start_raw,
                video_pixel_frame_end_raw,
                start_time,
                end_time,
            )
        )
        if mask_video:
            if spatial_mask is not None:
                video_mask[
                    :,
                    :,
                    video_latent_frame_index_start:video_latent_frame_index_end,
                    :,
                    :,
                ] = spatial_mask
            else:
                video_mask[
                    :, :, video_latent_frame_index_start:video_latent_frame_index_end
                ] = 1.0
        if mask_audio:
            audio_mask[
                :, :, audio_latent_frame_index_start:audio_latent_frame_index_end
            ] = 1.0

        if "noise_mask" in av_latent:
            base_mask = av_latent["noise_mask"].tensors[0].clone()
            if (
                base_mask.shape[0]
                == base_mask.shape[1]
                == 1
                == base_mask.shape[3]
                == base_mask.shape[4]
            ):
                for frame in range(base_mask.shape[2]):
                    video_mask[:, :, frame, :, :] *= base_mask[0, 0, frame, 0, 0]

        av_latent["noise_mask"] = NestedTensor(
            ltxav.recombine_audio_and_video_latents(video_mask, audio_mask)
        )

        video_latent_blend_coefficients, video_pixel_blend_coefficients = (
            get_video_latent_blend_coefficients(
                video_latent_frame_index_start,
                video_latent_frame_index_end,
                video_latent_frame_count,
                slope_len=slope_len,
            )
        )
        return (
            positive,
            negative,
            av_latent,
            video_latent_blend_coefficients,
            video_pixel_blend_coefficients,
        )
