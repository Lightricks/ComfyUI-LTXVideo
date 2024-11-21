from contextlib import nullcontext
import torch
import torch.nn as nn
import comfy.model_base
import comfy.model_management
import comfy.supported_models_base
import comfy.utils
import comfy.model_patcher
import comfy.sd
import comfy.latent_formats
import comfy.model_sampling
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
import math

from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.autoencoders.vae_encode import get_vae_size_scale_factor


class LTXVModelConfig:
    def __init__(self, latent_channels):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.LatentFormat()
        self.latent_format.latent_channels = latent_channels
        self.manual_cast_dtype = torch.float32
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 1
        # denoiser is handled by extension
        self.unet_config["disable_unet_model_creation"] = True


class LTXVSampling(torch.nn.Module, comfy.model_sampling.CONST):
    def __init__(self, condition_mask):
        super().__init__()
        self.condition_mask = condition_mask
        self.set_parameters(shift=1.0, multiplier=1)

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return 0.0

    @property
    def sigma_max(self):
        return 1.0

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return timestep

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        self.condition_mask = self.condition_mask.to(latent_image.device)
        scaled = latent_image * (1 - sigma) + noise * sigma
        result = latent_image * self.condition_mask + scaled * (1 - self.condition_mask)

        return result

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        result = model_input - model_output * sigma
        # In order to d * dT to be zero in euler step, we need to set result equal to input in first latent frame.
        result = result * (1 - self.condition_mask) + model_input * self.condition_mask
        return result


class LTXVModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_sampling = LTXVSampling(torch.zeros([1]))


class LTXVTransformer3D(nn.Module):
    def __init__(
        self, transformer: Transformer3DModel, patchifier: SymmetricPatchifier
    ):
        super().__init__()
        self.transformer = transformer
        self.dtype = transformer.dtype
        self.patchifier = patchifier

    def forward(
        self,
        latent,
        timesteps,
        context,
        indices_grid,
        img_hw=None,
        aspect_ratio=None,
        mixed_precision=True,
        **kwargs,
    ):
        # infer mask from context padding, assumes padding vectors are all zero.
        latent = latent.to(self.transformer.dtype)
        latent_patchified = self.patchifier.patchify(latent)
        context_mask = (context != 0).any(dim=2).float()

        l = latent_patchified
        if mixed_precision:
            context_manager = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            context_manager = nullcontext()
        with context_manager:
            noise_pred = self.transformer(
                l.to(self.transformer.dtype).to(self.transformer.device),
                indices_grid.to(self.transformer.device),
                encoder_hidden_states=context.to(self.transformer.device),
                encoder_attention_mask=context_mask.to(self.transformer.device).to(
                    torch.int64
                ),
                timestep=timesteps,
                return_dict=False,
            )[0]

        result = self.patchifier.unpatchify(
            latents=noise_pred,
            output_height=latent.shape[3],
            output_width=latent.shape[4],
            output_num_frames=latent.shape[2],
            out_channels=latent.shape[1] // math.prod(self.patchifier.patch_size),
        )
        return result


class LTXVTransformer3DWrapper(nn.Module):
    def __init__(
        self,
        transformer: LTXVTransformer3D,
        patchifier: SymmetricPatchifier,
        conditioning_mask,
        indices_grid,
    ):
        super().__init__()
        self.indices_grid = indices_grid
        self.dtype = transformer.dtype
        self.wrapped_transformer = transformer
        self.patchifier = patchifier
        self.conditioning_mask = conditioning_mask
        self.conditioning_mask_patchified = patchifier.patchify(
            self.conditioning_mask
        ).squeeze(-1)

    def forward(self, x, timesteps, context, img_hw=None, aspect_ratio=None, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        mixed_precision = transformer_options.get("mixed_precision", False)
        mask = self.conditioning_mask_patchified.to(x.device)
        ndim_mask = mask.ndimension()
        expanded_timesteps = timesteps.view(timesteps.size(0), *([1] * (ndim_mask - 1)))
        timesteps_masked = expanded_timesteps * (1 - mask)
        result = self.wrapped_transformer.forward(
            x,
            timesteps_masked,
            context,
            self.indices_grid,
            mixed_precision=mixed_precision,
        )
        return result
