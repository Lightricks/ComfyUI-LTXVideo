import comfy.samplers
import comfy.utils
import torch
from comfy.model_patcher import ModelPatcher
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from .nodes_registry import comfy_node


def stg(
    noise_pred_pos,
    noise_pred_neg,
    noise_pred_pertubed,
    cfg_scale,
    stg_scale,
    rescale_scale,
):
    noise_pred = (
        noise_pred_neg
        + cfg_scale * (noise_pred_pos - noise_pred_neg)
        + stg_scale * (noise_pred_pos - noise_pred_pertubed)
    )
    if rescale_scale != 0:
        factor = noise_pred_pos.std() / noise_pred.std()
        factor = rescale_scale * factor + (1 - rescale_scale)
        noise_pred = noise_pred * factor
    return noise_pred


@comfy_node(name="LTXVApplySTG")
class LTXVApplySTG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The model to apply the STG to."},
                ),
                "stg_mode": (["attention", "residual"],),
                "block_indices": (
                    "STRING",
                    {
                        "default": "14, 19",
                        "tooltip": "Comma-separated indices of the blocks to apply the STG to.",
                    },
                ),
            }
        }

    FUNCTION = "apply_stg"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "lightricks/LTXV"

    def apply_stg(self, model: ModelPatcher, stg_mode: str, block_indices: str):
        skip_block_list = [int(i.strip()) for i in block_indices.split(",")]
        stg_mode = (
            SkipLayerStrategy.Attention
            if stg_mode == "attention"
            else SkipLayerStrategy.Residual
        )
        new_model = model.clone()

        new_model.model_options["transformer_options"]["skip_layer_strategy"] = stg_mode
        if "skip_block_list" in new_model.model_options["transformer_options"]:
            skip_block_list.extend(
                new_model.model_options["transformer_options"]["skip_block_list"]
            )
        new_model.model_options["transformer_options"][
            "skip_block_list"
        ] = skip_block_list

        return (new_model,)


class STGGuider(comfy.samplers.CFGGuider):
    def set_conds(self, positive, negative):
        self.inner_set_conds(
            {"positive": positive, "negative": negative, "perturbed": positive}
        )

    def set_cfg(self, cfg, stg_scale, rescale_scale: float = None):
        self.cfg = cfg
        self.stg_scale = stg_scale
        self.rescale_scale = rescale_scale

    def predict_noise(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        model_options: dict = {},
        seed=None,
    ):
        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and perturbed, so we call calc_cond_batch([perturbed,pos,neg]) directly

        perturbed_cond = self.conds.get("perturbed", None)
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        noise_pred_neg = 0
        # no similar optimization for stg=0, use CFG guider instead.
        if self.cfg > 1:
            model_options["transformer_options"]["ptb_index"] = 2
            (noise_pred_perturbed, noise_pred_pos, noise_pred_neg) = (
                comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [perturbed_cond, positive_cond, negative_cond],
                    x,
                    timestep,
                    model_options,
                )
            )
        else:
            model_options["transformer_options"]["ptb_index"] = 1
            (noise_pred_perturbed, noise_pred_pos) = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [perturbed_cond, positive_cond],
                x,
                timestep,
                model_options,
            )
        stg_result = stg(
            noise_pred_pos,
            noise_pred_neg,
            noise_pred_perturbed,
            self.cfg,
            self.stg_scale,
            self.rescale_scale,
        )

        # normally this would be done in cfg_function, but we skipped
        # that for efficiency: we can compute the noise predictions in
        # a single call to calc_cond_batch() (rather than two)
        # so we replicate the hook here
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": stg_result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                # not in the original call in samplers.py:cfg_function, but made available for future hooks
                "perturbed_cond": positive_cond,
                "perturbed_cond_denoised": noise_pred_perturbed,
            }
            stg_result = fn(args)

        return stg_result


@comfy_node(name="STGGuider")
class STGGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "stg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "rescale": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "lightricks/LTXV"

    def get_guider(self, model, positive, negative, cfg, stg, rescale):
        guider = STGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg, stg, rescale)
        return (guider,)
