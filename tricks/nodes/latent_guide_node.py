import comfy_extras.nodes_lt as nodes_lt


class AddLatentGuideNode(nodes_lt.LTXVAddGuide):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "image_latent": ("LATENT",),
                "index": ("INT", {"default": 0, "min": -1, "max": 9999, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")

    CATEGORY = "Lightricks/LTXV/Guidance"
    FUNCTION = "generate"

    def generate(self, model, latent, image_latent, index, strength):
        noise_mask = nodes_lt.get_noise_mask(latent)
        latent = latent["samples"]

        image_latent = image_latent["samples"]

        latent, noise_mask = self.replace_latent_frames(
            latent,
            noise_mask,
            image_latent,
            index,
            strength,
        )

        return (
            model,
            {"samples": latent, "noise_mask": noise_mask},
        )
