try:
    from q8_kernels.integration.patch_transformer import (
        patch_comfyui_native_transformer,
        patch_comfyui_transformer,
    )

    Q8_AVAILABLE = True
except ImportError:
    Q8_AVAILABLE = False

from .nodes_registry import comfy_node


def check_q8_available():
    if not Q8_AVAILABLE:
        raise ImportError(
            "Q8 kernels are not available. Please install them to use this feature."
        )


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
                "transform_weights": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Transform weights to Q8 format."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Q8 Patcher"

    def patch(self, model, use_fp8_attention, transform_weights):
        check_q8_available()
        m = model.clone()
        diffusion_key = "diffusion_model"
        diffusion_model = m.get_model_object(diffusion_key)
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            transformer_key = "diffusion_model.transformer"
            patcher = patch_comfyui_transformer
        else:
            transformer_key = "diffusion_model"
            patcher = patch_comfyui_native_transformer
        transformer = m.get_model_object(transformer_key)
        patcher(transformer, use_fp8_attention, transform_weights)
        m.add_object_patch(transformer_key, transformer)
        return (m,)
