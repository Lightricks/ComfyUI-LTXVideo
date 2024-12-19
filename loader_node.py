import json
from pathlib import Path

import comfy
import comfy.model_management
import comfy.model_patcher
import folder_paths
import safetensors.torch
import torch
try:
    import torch._dynamo
    DYNAMO = True
except ImportError:
    DYNAMO = False

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from safetensors import safe_open

from .model import LTXVModel, LTXVModelConfig, LTXVTransformer3D
from .nodes_registry import comfy_node
from .vae import LTXVVAE

@comfy_node(name="LTXTricksTorchCompileSettings")
class LTXTricksTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor", "cudagraphs"], {"default": "inductor", "tooltip": "Backend for torch.compile"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode in compilation"}),
                "mode": (
                    ["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"],
                    {"default": "default", "tooltip": "Compilation mode optimization level"}
                ),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic shapes in compilation"}),
                "dynamo_cache_size_limit": (
                    "INT",
                    {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "Set torch._dynamo.config.cache_size_limit"}
                ),
            },
        }

    RETURN_TYPES = ("COMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "compile_settings"
    CATEGORY = "lightricks/LTXV"
    DESCRIPTION = "torch compile settings for LTXV models. Requires Triton and torch 2.5.0+ recommended."

    def compile_settings(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit):
        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
        }

        return (compile_args, )

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
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "torch_compile_args": ("COMPILEARGS", {"tooltip": "Optional compile arguments for torch.compile"})
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Loader"
    OUTPUT_NODE = False

    def load(self, ckpt_name, dtype, torch_compile_args=None):
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        ckpt_path = Path(folder_paths.get_full_path("checkpoints", ckpt_name))

        vae_config = None
        unet_config = None
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata is not None:
                config_metadata = metadata.get("config", None)
                if config_metadata is not None:
                    config_metadata = json.loads(config_metadata)
                    vae_config = config_metadata.get("vae", None)
                    unet_config = config_metadata.get("transformer", None)

        weights = safetensors.torch.load_file(ckpt_path, device="cpu")

        vae = self._load_vae(weights, vae_config)
        num_latent_channels = vae.first_stage_model.config.latent_channels

        model = self._load_unet(
            load_device,
            offload_device,
            weights,
            num_latent_channels,
            dtype=dtype_map[dtype],
            config=unet_config,
        )

        # If compile arguments are provided, apply torch.compile
        if torch_compile_args is not None:
            backend = torch_compile_args.get("backend", "inductor")
            fullgraph = torch_compile_args.get("fullgraph", False)
            mode = torch_compile_args.get("mode", "default")
            dynamic = torch_compile_args.get("dynamic", False)
            dynamo_cache_size_limit = torch_compile_args.get("dynamo_cache_size_limit", 64)
            if DYNAMO:
                torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit

            # Compile the diffusion_model with provided arguments
            model.model.diffusion_model = torch.compile(
                model.model.diffusion_model,
                backend=backend,
                fullgraph=fullgraph,
                mode=mode,
                dynamic=dynamic,
            )

        return (model, vae)

    def _load_vae(self, weights, config=None):
        if config is None:
            config = {
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
            }
        vae_prefix = "vae."
        vae = LTXVVAE.from_config_and_state_dict(
            vae_class=CausalVideoAutoencoder,
            config=config,
            state_dict={
                key.removeprefix(vae_prefix): value
                for key, value in weights.items()
                if key.startswith(vae_prefix)
            },
        )
        return vae

    def _load_unet(
        self,
        load_device,
        offload_device,
        weights,
        num_latent_channels,
        dtype,
        config=None,
    ):
        if config is None:
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
        transformer.to(dtype).to(load_device).eval()
        patchifier = SymmetricPatchifier(1)
        diffusion_model = LTXVTransformer3D(transformer, patchifier, None, None, None)
        model = LTXVModel(
            LTXVModelConfig(num_latent_channels, dtype=dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=comfy.model_management.get_torch_device(),
        )
        model.diffusion_model = diffusion_model

        patcher = comfy.model_patcher.ModelPatcher(model, load_device, offload_device)

        return patcher
