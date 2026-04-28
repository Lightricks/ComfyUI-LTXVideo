from comfy.model_management import dtype_size

from .nodes_registry import comfy_node


@comfy_node(name="LTXVPatcherVAE")
class LTXVPatcherVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV VAE Patcher"

    def patch(self, vae):
        from q8_kernels.integration.patch_vae import patch_vae

        vae_model = vae.first_stage_model
        vae.memory_used_decode = (
            lambda shape, dtype: shape[1]
            * shape[2]
            * shape[3]
            * shape[4]
            * 8
            * 8
            * 8
            * 3
            * dtype_size(dtype)
        )

        patch_vae(vae_model, patch_block=4)
        return (vae,)
