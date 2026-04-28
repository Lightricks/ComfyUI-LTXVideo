import copy
from typing import Callable

import torch
from comfy.model_patcher import ModelPatcher

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

                    t[i, c] = ((t[i, c] - i_mean) / i_sd) * r_sd + r_mean
                else:
                    for f in range(t.size(2)):
                        r_sd, r_mean = torch.std_mean(
                            reference["samples"][i, c, f], dim=None
                        )  # index by original dim order
                        i_sd, i_mean = torch.std_mean(t[i, c, f], dim=None)
                        t[i, c, f] = ((t[i, c, f] - i_mean) / i_sd) * r_sd + r_mean

        latents_copy["samples"] = torch.lerp(latents["samples"], t, factor)
        return (latents_copy,)


@comfy_node(name="LTXVStatNormLatent")
class LTXVStatNormLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "target_mean": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "target_std": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "percentile": (
                    "FLOAT",
                    {
                        "default": 95.0,
                        "min": 50.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.1,
                        "tooltip": "Percentile of distribution to use for statistics calculation",
                    },
                ),
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
                "clip_outliers": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "statistical_normalize"

    CATEGORY = "Lightricks/latents"

    def statistical_normalize(
        self, latents, target_mean, target_std, percentile, factor, clip_outliers
    ):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"]  # B x C x F x H x W

        # For 95% of distribution, we want to exclude 2.5% from each tail
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile

        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):  # channel
                channel_data = t[i, c]
                original_shape = channel_data.shape
                channel_flat = channel_data.flatten()

                # Calculate percentiles
                lower_bound = torch.quantile(channel_flat, lower_percentile / 100)
                upper_bound = torch.quantile(channel_flat, upper_percentile / 100)

                # Create mask for values within the percentile range
                mask = (channel_flat >= lower_bound) & (channel_flat <= upper_bound)

                # Calculate mean and std only on the masked values
                if mask.sum() > 0:
                    filtered_data = channel_flat[mask]
                    current_mean = filtered_data.mean()
                    current_std = filtered_data.std()

                    # Avoid division by zero
                    if current_std > 1e-8:
                        # Normalize all values
                        normalized_flat = (
                            (channel_flat - current_mean) / current_std
                        ) * target_std + target_mean

                        if clip_outliers:
                            # Calculate the normalized bounds
                            normalized_lower = (
                                (lower_bound - current_mean) / current_std
                            ) * target_std + target_mean
                            normalized_upper = (
                                (upper_bound - current_mean) / current_std
                            ) * target_std + target_mean

                            # Clip outliers to the normalized bounds
                            normalized_flat = torch.where(
                                channel_flat < lower_bound,
                                normalized_lower,
                                normalized_flat,
                            )
                            normalized_flat = torch.where(
                                channel_flat > upper_bound,
                                normalized_upper,
                                normalized_flat,
                            )

                        # Reshape back to original shape
                        t[i, c] = normalized_flat.reshape(original_shape)
                    else:
                        # If std is too small, just shift by mean
                        t[i, c] = channel_data - current_mean + target_mean

        latents_copy["samples"] = torch.lerp(latents["samples"], t, factor)
        return (latents_copy,)


class PerStepNormPatcher:
    """
    Base class for per-step normalization nodes.
    """

    @classmethod
    def required(s):
        return {
            "model": ("MODEL",),
            "factors": (
                "STRING",
                {
                    "default": "0.9, 0.75, 0.0",
                    "tooltip": "Comma-separated list of factors, each factor will be used for one step.",
                },
            ),
        }

    @classmethod
    def optional(s):
        return {}

    @staticmethod
    def patch_smooth_norm(
        model: ModelPatcher, factors: str, normalize_fn: Callable
    ) -> ModelPatcher:
        model = model.clone()
        factors = [float(x) for x in factors.split(",")]
        step = 0

        def norm_fn(args):
            nonlocal step
            latent = args["denoised"]
            factor = factors[min(step, len(factors) - 1)]
            step += 1
            out_latent = normalize_fn(latent={"samples": latent}, factor=factor)
            return out_latent[0]["samples"]

        model.set_model_sampler_post_cfg_function(norm_fn)
        return (model,)

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        return float("NaN")


@comfy_node(name="LTXVPerStepAdainPatcher")
class LTXVPerStepAdainPatcher(PerStepNormPatcher):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": super().required()
            | {
                "reference": ("LATENT",),
            },
            "optional": super().optional()
            | {
                "per_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"

    CATEGORY = "Lightricks/latents"

    def patch_model(self, model, factors, reference, per_frame=False):
        def cfg_adain(latent, factor):
            return LTXVAdainLatent().batch_normalize(
                latent, reference, factor, per_frame
            )

        return self.patch_smooth_norm(model, factors, cfg_adain)


@comfy_node(name="LTXVPerStepStatNormPatcher")
class LTXVPerStepStatNormPatcher(PerStepNormPatcher):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": super().required()
            | {
                "target_mean": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "target_std": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "percentile": (
                    "FLOAT",
                    {
                        "default": 95.0,
                        "min": 50.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.1,
                        "tooltip": "Percentile of distribution to use for statistics calculation",
                    },
                ),
                "clip_outliers": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"

    CATEGORY = "Lightricks/latents"

    def patch_model(
        self, model, factors, target_mean, target_std, percentile, clip_outliers
    ):
        def cfg_stat_norm(latent, factor):
            return LTXVStatNormLatent().statistical_normalize(
                latent, target_mean, target_std, percentile, factor, clip_outliers
            )

        return self.patch_smooth_norm(model, factors, cfg_stat_norm)
