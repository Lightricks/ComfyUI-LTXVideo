import math

import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import torch
from ltx_video.models.autoencoders.vae_encode import get_vae_size_scale_factor

from .img2vid import encode_media_conditioning
from .model import LTXVSampling
from .nodes_registry import comfy_node


def get_normal_shift(
    n_tokens: int,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    m = (max_shift - min_shift) / (max_tokens - min_tokens)
    b = min_shift - m * min_tokens
    return m * n_tokens + b


@comfy_node(name="LTXVModelConfigurator")
class LTXVModelConfigurator:
    @classmethod
    def INPUT_TYPES(s):
        PRESETS = [
            "Custom",
            "1216x704   | 41",
            "1088x704   | 49",
            "1056x640   | 57",
            "992x608    | 65",
            "896x608    | 73",
            "896x544    | 81",
            "832x544    | 89",
            "800x512    | 97",
            "768x512    | 97",
            "800x480    | 105",
            "736x480    | 113",
            "704x480    | 121",
            "704x448    | 129",
            "672x448    | 137",
            "640x416    | 153",
            "672x384    | 161",
            "640x384    | 169",
            "608x384    | 177",
            "576x384    | 185",
            "608x352    | 193",
            "576x352    | 201",
            "544x352    | 209",
            "512x352    | 225",
            "512x352    | 233",
            "544x320    | 241",
            "512x320    | 249",
            "512x320    | 257",
        ]
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "preset": (
                    PRESETS,
                    {
                        "default": "Custom",
                        "tooltip": "Preset resolution and frame count. Custom allows manual input.",
                    },
                ),
                "width": ("INT", {"default": 768, "min": 1, "max": 10000}),
                "height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "frames_number": (
                    "INT",
                    {
                        "default": 65,
                        "min": 9,
                        "max": 257,
                        "step": 8,
                        "tooltip": "Must be equal to N * 8 + 1",
                    },
                ),
                "frame_rate": ("INT", {"default": 25, "min": 1, "max": 60}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 60}),
                "mixed_precision": ("BOOLEAN", {"default": True}),
                "img_compression": (
                    "INT",
                    {
                        "default": 29,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Amount of compression to apply on conditioning image.",
                    },
                ),
            },
            "optional": {
                "conditioning": (
                    "IMAGE",
                    {"tooltip": "Optional conditioning image or video."},
                ),
                "initial_latent": (
                    "LATENT",
                    {
                        "tooltip": "initial latent that is combined with conditioning if given"
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "latent", "sigma_shift")
    FUNCTION = "configure_sizes"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Model Configurator"
    OUTPUT_NODE = False

    def latent_shape_and_frame_rate(
        self, vae, batch, height, width, frames_number, frame_rate
    ):
        video_scale_factor, vae_scale_factor, _ = get_vae_size_scale_factor(
            vae.first_stage_model
        )
        video_scale_factor = video_scale_factor if frames_number > 1 else 1

        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        latent_channels = vae.first_stage_model.config.latent_channels
        latent_num_frames = math.floor(frames_number / video_scale_factor) + 1
        latent_frame_rate = frame_rate / video_scale_factor

        latent_shape = [
            batch,
            latent_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        ]
        return latent_shape, latent_frame_rate

    def configure_sizes(
        self,
        model,
        vae,
        preset,
        width,
        height,
        frames_number,
        frame_rate,
        batch,
        mixed_precision,
        img_compression,
        conditioning=None,
        initial_latent=None,
    ):
        load_device = comfy.model_management.get_torch_device()
        if preset != "Custom":
            preset = preset.split("|")
            width, height = map(int, preset[0].strip().split("x"))
            frames_number = int(preset[1].strip())
        latent_shape, latent_frame_rate = self.latent_shape_and_frame_rate(
            vae, batch, height, width, frames_number, frame_rate
        )
        mask_shape = [
            latent_shape[0],
            1,
            latent_shape[2],
            latent_shape[3],
            latent_shape[4],
        ]
        conditioning_mask = torch.zeros(mask_shape, device=load_device)
        initial_latent = (
            None
            if initial_latent is None
            else initial_latent["samples"].to(load_device)
        )
        guiding_latent = None
        if conditioning is not None:
            latent = encode_media_conditioning(
                conditioning,
                vae,
                width,
                height,
                frames_number,
                image_compression=img_compression,
                initial_latent=initial_latent,
            )
            conditioning_mask[:, :, 0] = 1.0
            guiding_latent = latent[:, :, :1, ...]
        else:
            latent = torch.zeros(latent_shape, dtype=torch.float32, device=load_device)
            if initial_latent is not None:
                latent[:, :, : initial_latent.shape[2], ...] = initial_latent

        _, vae_scale_factor, _ = get_vae_size_scale_factor(vae.first_stage_model)

        patcher = model.clone()
        patcher.add_object_patch("diffusion_model.conditioning_mask", conditioning_mask)
        patcher.add_object_patch("diffusion_model.latent_frame_rate", latent_frame_rate)
        patcher.add_object_patch("diffusion_model.vae_scale_factor", vae_scale_factor)
        patcher.add_object_patch(
            "model_sampling", LTXVSampling(conditioning_mask, guiding_latent)
        )
        patcher.model_options.setdefault("transformer_options", {})[
            "mixed_precision"
        ] = mixed_precision

        num_latent_patches = latent_shape[2] * latent_shape[3] * latent_shape[4]
        return (patcher, {"samples": latent}, get_normal_shift(num_latent_patches))


@comfy_node(name="LTXVShiftSigmas")
class LTXVShiftSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "sigma_shift": ("FLOAT", {"default": 1.820833333}),
                "stretch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Stretch the sigmas to be in the range [terminal, 1].",
                    },
                ),
                "terminal": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.99,
                        "step": 0.01,
                        "tooltip": "The terminal value of the sigmas after stretching.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "lightricks/LTXV"

    FUNCTION = "shift_sigmas"
    DESCRIPTION = (
        "Transforms sigmas to values where the model can focus on denoising high noise."
    )

    def shift_sigmas(self, sigmas, sigma_shift, stretch, terminal):
        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return (sigmas,)
