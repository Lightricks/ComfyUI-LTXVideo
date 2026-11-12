import copy

import torch

from .nodes_registry import comfy_node


@comfy_node(name="LTXVAdainLatent")
class LTXVAdainLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "reference": ("LATENT",),
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            },
            "optional": {
                "per_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_normalize"

    CATEGORY = "Lightricks/latents"

    def batch_normalize(self, latents, reference, factor, per_frame=False):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"]  #  B x C x F x H x W

        # Guard against division by a (near-)zero input std: a constant channel
        # or frame yields i_sd == 0 -> 0/0 == NaN. LTXVStatNormLatent guards the
        # same way (current_std > 1e-8).
        eps = 1e-8

        if per_frame:
            if reference["samples"].size(2) == 1:
                print("Reference has only one frame, using it for all frames")
                reference["samples"] = reference["samples"].repeat(
                    1, 1, t.size(2), 1, 1
                )

            elif t.size(2) > reference["samples"].size(2):
                raise ValueError("Latents have more frames than reference")

        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):
                if not per_frame:
                    r_sd, r_mean = torch.std_mean(
                        reference["samples"][i, c], dim=None
                    )  # index by original dim order
                    i_sd, i_mean = torch.std_mean(t[i, c], dim=None)

                    t[i, c] = ((t[i, c] - i_mean) / (i_sd + eps)) * r_sd + r_mean
                else:
                    for f in range(t.size(2)):
                        r_sd, r_mean = torch.std_mean(
                            reference["samples"][i, c, f], dim=None
                        )  # index by original dim order
                        i_sd, i_mean = torch.std_mean(t[i, c, f], dim=None)
                        t[i, c, f] = (
                            (t[i, c, f] - i_mean) / (i_sd + eps)
                        ) * r_sd + r_mean

        # torch.lerp(a, b, 0.0) is NOT a no-op when b holds NaN/Inf: 0 * NaN == NaN
        # leaks through. Short-circuit factor == 0, and sanitize otherwise so a
        # bad statistic can never propagate past the requested blend.
        if factor == 0:
            # t is latents_copy["samples"], mutated in place above; lerp(a, b, 0)
            # == a, so return the untouched original.
            latents_copy["samples"] = latents["samples"].clone()
            return (latents_copy,)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        latents_copy["samples"] = torch.lerp(latents["samples"], t, factor)
        return (latents_copy,)
