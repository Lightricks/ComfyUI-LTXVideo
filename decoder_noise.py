from copy import copy

from .nodes_registry import comfy_node


@comfy_node(name="Set VAE Decoder Noise")
class DecoderNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "timestep": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "The timestep used for decoding the noise.",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 0.025,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "The scale of the noise added to the decoder.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            }
        }

    FUNCTION = "add_noise"
    RETURN_TYPES = ("VAE",)
    CATEGORY = "Lightricks/LTXV/Latent"

    def add_noise(self, vae, timestep, scale, seed):
        result = copy(vae)
        if hasattr(result, "first_stage_model"):
            result.first_stage_model.decode_timestep = timestep
            result.first_stage_model.decode_noise_scale = scale
        result._decode_timestep = timestep
        result.decode_noise_scale = scale
        result.seed = seed
        return (result,)
