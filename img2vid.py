import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import torch
from ltx_video.models.autoencoders.vae_encode import get_vae_size_scale_factor


def pad_tensor(tensor, target_len):
    dim = 2
    repeat_factor = target_len - tensor.shape[dim]  # Ceiling division
    last_element = tensor.select(dim, -1).unsqueeze(dim)
    padding = last_element.repeat(1, 1, repeat_factor, 1, 1)
    return torch.cat([tensor, padding], dim=dim)


def encode_media_conditioning(init_media, vae, width, height, frames_number):
    pixels = comfy.utils.common_upscale(
        init_media.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    encode_pixels = pixels[:, :, :, :3]
    init_latents = vae.encode(encode_pixels).float()

    video_scale_factor, _, _ = get_vae_size_scale_factor(vae.first_stage_model)
    video_scale_factor = video_scale_factor if frames_number > 1 else 1
    target_len = (frames_number // video_scale_factor) + 1
    init_latents = init_latents[:, :, :target_len]

    init_image_frame_number = init_media.shape[0]
    if init_image_frame_number == 1:
        result = pad_tensor(init_latents, target_len)
    elif init_image_frame_number % 8 != 1:
        result = pad_tensor(init_latents, target_len)
    else:
        result = init_latents

    return result
