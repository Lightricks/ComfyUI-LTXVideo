from pathlib import Path
import safetensors.torch
import torch

import comfy
import comfy.model_management
import comfy.model_patcher
import folder_paths


from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import get_vae_size_scale_factor
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier

from .model import LTXVTransformer3D, LTXVModel, LTXVModelConfig
from .vae import LTXVVAE

from .nodes_registry import comfy_node


@comfy_node(name="LTXVLoader")
class LTXVLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Loader"
    OUTPUT_NODE = False

    def load(self, ckpt_name):
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        ckpt_path = Path(folder_paths.get_full_path("checkpoints", ckpt_name))

        weights = safetensors.torch.load_file(ckpt_path, device=str(load_device))

        vae = self._load_vae(weights)
        num_latent_channels = vae.first_stage_model.config.latent_channels

        model = self._load_unet(
            load_device, offload_device, weights, num_latent_channels
        )
        return (model, vae)

    def _load_vae(self, weights):
        vae_prefix = "vae."
        vae = LTXVVAE.from_config_and_state_dict(
            vae_class=CausalVideoAutoencoder,
            config={
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3,
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "blocks": [
                    ["res_x", 4],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x", 3],
                    ["res_x", 4],
                ],
                "scaling_factor": 1.0,
                "norm_layer": "pixel_norm",
                "patch_size": 4,
                "latent_log_var": "uniform",
                "use_quant_conv": False,
                "causal_decoder": False,
            },
            state_dict={
                key.removeprefix(vae_prefix): value
                for key, value in weights.items()
                if key.startswith(vae_prefix)
            },
        )
        return vae

    def _load_unet(self, load_device, offload_device, weights, num_latent_channels):
        config = {
            "_class_name": "Transformer3DModel",
            "_diffusers_version": "0.25.1",
            "_name_or_path": "PixArt-alpha/PixArt-XL-2-256x256",
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 64,
            "attention_type": "default",
            "caption_channels": 4096,
            "cross_attention_dim": 2048,
            "double_self_attention": False,
            "dropout": 0.0,
            "in_channels": 128,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-06,
            "norm_num_groups": 32,
            "num_attention_heads": 32,
            "num_embeds_ada_norm": 1000,
            "num_layers": 28,
            "num_vector_embeds": None,
            "only_cross_attention": False,
            "out_channels": 128,
            "project_to_2d_pos": True,
            "upcast_attention": False,
            "use_linear_projection": False,
            "qk_norm": "rms_norm",
            "standardization_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "positional_embedding_theta": 10000.0,
            "positional_embedding_max_pos": [20, 2048, 2048],
            "timestep_scale_multiplier": 1000,
        }

        transformer = Transformer3DModel.from_config(config)
        unet_prefix = "model.diffusion_model."
        transformer.load_state_dict(
            {
                key.removeprefix(unet_prefix): value
                for key, value in weights.items()
                if key.startswith(unet_prefix)
            }
        )
        transformer.to(torch.float32).to(load_device).eval()

        diffusion_model = LTXVTransformer3D(transformer, SymmetricPatchifier(1))
        model = LTXVModel(
            LTXVModelConfig(num_latent_channels),
            model_type=comfy.model_base.ModelType.FLOW,
            device=comfy.model_management.get_torch_device(),
        )
        model.diffusion_model = diffusion_model

        patcher = comfy.model_patcher.ModelPatcher(model, load_device, offload_device)

        return patcher
