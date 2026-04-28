"""
Low VRAM loader nodes for sequential model loading.

These nodes add a 'dependencies' input to control execution order,
ensuring models load one after another instead of simultaneously.
This reduces peak VRAM usage in memory-constrained environments.
"""

import comfy.utils
import folder_paths
import nodes
from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
from comfy_api.latest import io
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader


class AnyType(str):
    """Custom type that matches any ComfyUI type for dependency inputs."""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class LowVRAMCheckpointLoader(nodes.CheckpointLoaderSimple):
    """Load Checkpoint with dependency input for sequential loading in low VRAM scenarios."""

    @classmethod
    def INPUT_TYPES(s) -> dict:
        base_types = super().INPUT_TYPES()
        base_types["optional"] = {
            "dependencies": (
                any_type,
                {
                    "tooltip": "Connect any output from a previous loader to ensure sequential loading."
                },
            ),
        }
        return base_types

    CATEGORY = "LTXV/loaders"
    DESCRIPTION = (
        "Loads a diffusion model checkpoint with dependency support. "
        "Connect 'dependencies' to a previous loader's output to ensure sequential loading and reduce peak VRAM usage."
    )
    FUNCTION = "load_checkpoint_sequentially"

    def load_checkpoint_sequentially(
        self,
        ckpt_name: str,
        dependencies: object = None,  # noqa: ARG002 - required for ComfyUI optional inputs
    ) -> tuple:
        return super().load_checkpoint(ckpt_name)


class LowVRAMAudioVAELoader:
    """Load LTXV Audio VAE with dependency input for sequential loading in low VRAM scenarios."""

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Audio VAE checkpoint to load."},
                ),
            },
            "optional": {
                "dependencies": (
                    any_type,
                    {
                        "tooltip": "Connect any output from a previous loader to ensure sequential loading."
                    },
                ),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("audio_vae",)
    CATEGORY = "LTXV/loaders"
    DESCRIPTION = (
        "Loads an LTXV Audio VAE checkpoint with dependency support. "
        "Connect 'dependencies' to a previous loader's output to ensure sequential loading and reduce peak VRAM usage."
    )
    FUNCTION = "load_audio_vae_sequentially"

    def load_audio_vae_sequentially(
        self,
        ckpt_name: str,
        dependencies: object = None,  # noqa: ARG002 - required for ComfyUI optional inputs
    ) -> tuple:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        audio_vae = AudioVAE(sd, metadata)
        return (audio_vae,)


class LowVRAMLatentUpscaleModelLoader(LatentUpscaleModelLoader):
    """Load latent upsampler with dependency input for sequential loading in low VRAM scenarios."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LowVRAMLatentUpscaleModelLoader",
            display_name="Low VRAMLoad Latent Upscale Model",
            category="LTXV/loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("latent_upscale_models"),
                ),
                io.AnyType.Input(
                    "dependencies",
                    optional=True,
                    tooltip="Connect any output from a previous loader to ensure sequential loading.",
                ),
            ],
            outputs=[
                io.LatentUpscaleModel.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name, dependencies=None) -> io.NodeOutput:
        return LatentUpscaleModelLoader.execute(model_name=model_name)
