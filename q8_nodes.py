import importlib.metadata
import logging

import comfy
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
import torch

try:
    from q8_kernels.functional.ops import hadamard_transform
    from q8_kernels.integration.patch_transformer import (
        patch_comfyui_native_avtransformer,
        patch_comfyui_native_transformer,
    )

    Q8_AVAILABLE = True
except ImportError:
    Q8_AVAILABLE = False

from .nodes_registry import comfy_node


def list_in_name(check_list, name):
    return any([x in name for x in check_list])


def check_q8_available():
    if not Q8_AVAILABLE:
        raise ImportError(
            "Q8 kernels are not available. To use this feature install the q8_kernels package from here:."
            "https://github.com/Lightricks/LTX-Video-Q8-Kernels"
        )


def check_deprecated():
    q8_version = tuple(
        int(x) for x in importlib.metadata.version("q8_kernels").split(".")[:3]
    )
    return q8_version >= (0, 1, 5)


@comfy_node(name="LTXQ8Patch")
class LTXVQ8Patch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "use_fp8_attention": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Use FP8 attention."},
                ),
                "quantization_preset": (
                    ["0.9.8", "ltxv2", "full_bf16", "custom"],
                    {"default": "0.9.8"},
                ),
                "quantize_self_attn": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Quantize Self Attention Layer"},
                ),
                "quantize_cross_attn": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Quantize Cross Attention Layer"},
                ),
                "quantize_ffn": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Quantize Feed Forward Layer"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Q8 Patcher"
    PRESETS = {
        "0.9.8": (True, True, True),
        "ltxv2": (True, False, True),
        "full_bf16": (False, False, False),
    }

    def patch(
        self,
        model,
        use_fp8_attention,
        quantization_preset,
        quantize_self_attn,
        quantize_cross_attn,
        quantize_ffn,
    ):
        check_q8_available()
        m = model.clone()
        transformer_key = "diffusion_model"
        transformer = m.get_model_object(transformer_key)

        is_av = transformer.__class__.__name__ == "LTXAVModel"
        if is_av:
            patcher = patch_comfyui_native_avtransformer
        else:
            patcher = patch_comfyui_native_transformer

        if check_deprecated():
            logging.info("This node is deprecated soon. Use new one.")
            quant_audio_path = False
            quant_type = "blockwise-fp8"
            patcher(transformer, use_fp8_attention, True, quant_type, quant_audio_path)
        else:
            quantize_self_attn, quantize_cross_attn, quantize_ffn = (
                LTXVQ8Patch.PRESETS.get(
                    quantization_preset,
                    (quantize_self_attn, quantize_cross_attn, quantize_ffn),
                )
            )
            if getattr(transformer, "quantization_config", None) is not None:
                if (quantize_self_attn, quantize_cross_attn, quantize_ffn) != getattr(
                    transformer, "quantization_config"
                ):
                    quantize_self_attn, quantize_cross_attn, quantize_ffn = getattr(
                        transformer, "quantization_config"
                    )
            patcher(
                transformer,
                use_fp8_attention,
                True,
                quantize_self_attn,
                quantize_cross_attn,
                quantize_ffn,
            )
            setattr(
                transformer,
                "quantization_config",
                (quantize_self_attn, quantize_cross_attn, quantize_ffn),
            )
        setattr(transformer, "is_q8_patched", True)
        setattr(transformer, "use_fp8_attention", use_fp8_attention)
        return (m,)


def idendity_quant_fn(x, t):
    return x.to(dtype=t)


@comfy_node(name="LTXVQ8LoraModelLoader")
class LTXVQ8LoraModelLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "lightricks/LTXV"
    FUNCTION = "load_lora_model_only"

    def load_lora(self, model, lora_name, strength_model):
        quant_fn = hadamard_transform
        transformer = model.get_model_object("diffusion_model")

        is_patched_transformer = getattr(transformer, "is_q8_patched", False)
        if not is_patched_transformer or not Q8_AVAILABLE:
            raise ValueError(
                "LTXV Q8 Patcher is not applied to the model. Please use LTXQ8Patch node before loading lora or install q8_kernels."
            )

        if strength_model == 0:
            return model
        quantize_self_attn, quantize_cross_attn, quantize_ffn = getattr(
            transformer, "quantization_config"
        )
        skip_list = []
        if not quantize_self_attn:
            skip_list += ["attn1"]
        if not quantize_cross_attn:
            skip_list += ["attn2"]
        if not quantize_ffn:
            skip_list += ["ff"]
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        new_lora = {}
        for k in lora:
            device = lora[k].device
            if lora[k].ndim == 2:
                if "lora_A" in k and not list_in_name(skip_list, k):
                    new_lora[k] = quant_fn(
                        lora[k].to(device="cuda", dtype=torch.bfloat16),
                        out_type=torch.bfloat16,
                    ).to(device)
                else:
                    new_lora[k] = lora[k]
            else:
                new_lora[k] = lora[k]
        self.loaded_lora = (lora_path, new_lora)

        model_lora, _ = comfy.sd.load_lora_for_models(
            model, None, new_lora, strength_model, 0
        )
        return model_lora

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, lora_name, strength_model),)
