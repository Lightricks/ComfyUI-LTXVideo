import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import torch
from diffusers.image_processor import VaeImageProcessor
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    vae_decode,
    vae_encode,
)


class LTXVVAE(comfy.sd.VAE):
    def __init__(self):
        self.device = comfy.model_management.vae_device()
        self.offload_device = comfy.model_management.vae_offload_device()

    @classmethod
    def from_pretrained(cls, vae_class, model_path, dtype=torch.bfloat16):
        instance = cls()
        model = vae_class.from_pretrained(
            pretrained_model_name_or_path=model_path,
            revision=None,
            torch_dtype=dtype,
            load_in_8bit=False,
        )
        instance._finalize_model(model)
        return instance

    @classmethod
    def from_config_and_state_dict(
        cls, vae_class, config, state_dict, dtype=torch.bfloat16
    ):
        instance = cls()
        model = vae_class.from_config(config)
        model.load_state_dict(state_dict)
        model.to(dtype)
        instance._finalize_model(model)
        return instance

    def _finalize_model(self, model):
        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            model
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.first_stage_model = model.eval().to(self.device)

    # Assumes that the input samples have dimensions in following order
    # (batch, channels, frames, height, width)
    def decode(self, samples_in):
        is_video = samples_in.shape[2] > 1
        result = vae_decode(
            samples_in.to(self.device),
            vae=self.first_stage_model,
            is_video=is_video,
            vae_per_channel_normalize=True,
        )
        result = self.image_processor.postprocess(
            result, output_type="pt", do_denormalize=[True]
        )
        return (
            result.squeeze(0).permute(1, 2, 3, 0).to(torch.float32)
        )  # .to(self.offload_device)

    # Underlying VAE expects b, c, n, h, w dimensions order and dtype specific dtype.
    # However in Comfy the convension is n, h, w, c.
    def encode(self, pixel_samples):
        preprocessed = self.image_processor.preprocess(
            pixel_samples.permute(3, 0, 1, 2)
        )
        input = preprocessed.unsqueeze(0).to(torch.bfloat16).to(self.device)
        latents = vae_encode(
            input, self.first_stage_model, vae_per_channel_normalize=True
        ).to(comfy.model_management.get_torch_device())
        return latents
