import contextlib
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import comfy.ldm.modules.attention
import comfy.samplers
import torch
from comfy.model_patcher import ModelPatcher

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
        noise_pred_pos
        + (cfg_scale - 1) * (noise_pred_pos - noise_pred_neg)
        + stg_scale * (noise_pred_pos - noise_pred_pertubed)
    )
    if rescale_scale != 0:
        factor = noise_pred_pos.std() / noise_pred.std()
        factor = rescale_scale * factor + (1 - rescale_scale)
        noise_pred = noise_pred * factor
    return noise_pred


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0: torch.Tensor, v1: torch.Tensor):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def apg(
    noise_pred_pos: torch.Tensor,
    noise_pred_neg: torch.Tensor,
    cfg_scale: float,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = noise_pred_pos - noise_pred_neg
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, noise_pred_pos)
    normalized_update = diff_orthogonal + eta * diff_parallel
    noise_pred = noise_pred_pos + (cfg_scale - 1) * normalized_update
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

    DESCRIPTION = "Defines the blocks to apply the STG to."

    def apply_stg(self, model: ModelPatcher, block_indices: str):
        skip_block_list = [int(i.strip()) for i in block_indices.split(",")]
        new_model = model.clone()

        if "skip_block_list" in new_model.model_options["transformer_options"]:
            skip_block_list.extend(
                new_model.model_options["transformer_options"]["skip_block_list"]
            )
        new_model.model_options["transformer_options"][
            "skip_block_list"
        ] = skip_block_list

        return (new_model,)


@dataclass
class STGFlag:
    do_skip: bool = False
    skip_layers: List[int] = None


# context manager that replaces the attention function in a transformer block
class PatchAttention(contextlib.AbstractContextManager):
    def __init__(self, attn_idx: Optional[Union[int, List[int]]] = None):
        self.current_idx = -1

        if isinstance(attn_idx, int):
            self.attn_idx = [attn_idx]
        elif attn_idx is None:
            self.attn_idx = [0]
        else:
            self.attn_idx = list(attn_idx)

    def __enter__(self):
        self.original_attention = comfy.ldm.modules.attention.optimized_attention
        self.original_attention_masked = (
            comfy.ldm.modules.attention.optimized_attention_masked
        )

        comfy.ldm.modules.attention.optimized_attention = self.stg_attention
        comfy.ldm.modules.attention.optimized_attention_masked = (
            self.stg_attention_masked
        )

    def __exit__(self, exc_type, exc_value, traceback):
        comfy.ldm.modules.attention.optimized_attention = self.original_attention
        comfy.ldm.modules.attention.optimized_attention_masked = (
            self.original_attention_masked
        )

        self.original_attention = None
        self.original_attention_masked = None

    def stg_attention(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx in self.attn_idx:
            return v
        else:
            return self.original_attention(q, k, v, heads, *args, **kwargs)

    def stg_attention_masked(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx in self.attn_idx:
            return v
        else:
            return self.original_attention_masked(q, k, v, heads, *args, **kwargs)


class STGBlockWrapper:
    """Wraps transformer blocks to be able to skip attention layers."""

    def __init__(self, block, stg_flag: STGFlag, idx: int):
        self.flag = stg_flag
        self.idx = idx
        self.block = block

    def __call__(self, args, extra_args):
        context_manager = contextlib.nullcontext()

        stg_indexes = args["transformer_options"].get("stg_indexes", [0])
        if self.flag.do_skip and self.idx in self.flag.skip_layers:
            context_manager = PatchAttention(stg_indexes)

        with context_manager:
            hidden_state = extra_args["original_block"](args)
        return hidden_state


class STGGuider(comfy.samplers.CFGGuider):
    def __init__(
        self, model: ModelPatcher, cfg, stg_scale, rescale_scale: float = None
    ):
        model = model.clone()
        super().__init__(model)

        self.stg_flag = STGFlag(
            do_skip=False,
            skip_layers=model.model_options["transformer_options"]["skip_block_list"],
        )

        self.patch_model(model, self.stg_flag)

        self.cfg = cfg
        self.stg_scale = stg_scale
        self.rescale_scale = rescale_scale

    @classmethod
    def patch_model(cls, model: ModelPatcher, stg_flag: STGFlag):
        transformer_blocks = cls.get_transformer_blocks(model)

        for i, block in enumerate(transformer_blocks):
            model.set_model_patch_replace(
                STGBlockWrapper(block, stg_flag, i), "dit", "double_block", i
            )

    @staticmethod
    def get_transformer_blocks(model: ModelPatcher):
        diffusion_model = model.get_model_object("diffusion_model")
        key = "diffusion_model.transformer_blocks"
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            key = "diffusion_model.transformer.transformer_blocks"
        return model.get_model_object(key)

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        model_options: dict = {},
        seed=None,
    ):
        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and perturbed, so we call calc_cond_batch([perturbed,pos,neg]) directly

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        if model_options.get("sigma_to_params_mapping", None) is not None:
            cfg_value, stg_scale, stg_layer_skip_layer_indices, stg_rescale = (
                model_options["sigma_to_params_mapping"](timestep)
            )
            self.stg_flag.skip_layers = stg_layer_skip_layer_indices
            self.patch_model(self.model_patcher, self.stg_flag)

        else:
            cfg_value = self.cfg
            stg_scale = self.stg_scale
            stg_rescale = self.rescale_scale

        noise_pred_pos = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [positive_cond],
            x,
            timestep,
            model_options,
        )[0]

        noise_pred_neg = 0
        noise_pred_perturbed = 0

        if not math.isclose(cfg_value, 1.0):
            noise_pred_neg = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [negative_cond],
                x,
                timestep,
                model_options,
            )[0]

        if not math.isclose(stg_scale, 0.0):
            try:
                model_options["transformer_options"]["ptb_index"] = 0
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [positive_cond],
                    x,
                    timestep,
                    model_options,
                )[0]
            finally:
                self.stg_flag.do_skip = False
                del model_options["transformer_options"]["ptb_index"]

        stg_result = stg(
            noise_pred_pos,
            noise_pred_neg,
            noise_pred_perturbed,
            cfg_value,
            stg_scale,
            stg_rescale,
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


class STGGuiderAdvanced(comfy.samplers.CFGGuider):
    def __init__(
        self,
        model: ModelPatcher,
        sigma_list,
        cfg_list,
        stg_scale_list,
        stg_rescale_list,
        stg_layers_indices_list,
        skip_steps_sigma_threshold,
        cfg_star_rescale,
        apply_apg,
        apg_cfg_scale,
        eta,
        norm_threshold,
    ):
        model = model.clone()
        super().__init__(model)

        self.stg_flag = STGFlag(
            do_skip=False,
            skip_layers=model.model_options["transformer_options"].get(
                "skip_block_list"
            ),
        )

        self.sigma_list = sigma_list
        self.cfg_list = cfg_list
        self.stg_scale_list = stg_scale_list
        self.stg_rescale_list = stg_rescale_list
        self.stg_layers_indices_list = stg_layers_indices_list
        self.skip_steps_sigma_threshold = skip_steps_sigma_threshold
        self.cfg_star_rescale = cfg_star_rescale
        self.apply_apg = apply_apg
        self.apg_cfg_scale = apg_cfg_scale
        self.eta = eta
        self.norm_threshold = norm_threshold
        STGGuider.patch_model(model, self.stg_flag)

    def sigma_to_params_mapping(self, sigma):
        # Find the closest higher sigma value and return corresponding cfg
        higher_sigmas = [s for s in self.sigma_list if s >= sigma]
        if not higher_sigmas:
            closest_idx = -1  # Return last cfg if no higher sigma exists
        else:
            closest_higher = min(higher_sigmas)
            closest_idx = self.sigma_list.index(closest_higher)
        return (
            self.cfg_list[closest_idx],
            self.stg_scale_list[closest_idx],
            self.stg_rescale_list[closest_idx],
            self.stg_layers_indices_list[closest_idx],
        )

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        model_options: dict = {},
        seed=None,
    ):
        # CFG zero init - skipping steps with timestep bigger than given threshold.
        if timestep > self.skip_steps_sigma_threshold:
            return torch.zeros_like(x)

        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and perturbed, so we call calc_cond_batch([perturbed,pos,neg]) directly
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        cfg_value, stg_scale, stg_rescale, stg_layer_skip_layer_indices = (
            self.sigma_to_params_mapping(timestep)
        )

        if stg_layer_skip_layer_indices is not None:
            self.stg_flag.skip_layers = stg_layer_skip_layer_indices
            STGGuider.patch_model(self.model_patcher, self.stg_flag)

        noise_pred_pos = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [positive_cond],
            x,
            timestep,
            model_options,
        )[0]

        noise_pred_neg = 0
        noise_pred_perturbed = 0

        if not math.isclose(cfg_value, 1.0) or (
            self.apply_apg and not math.isclose(self.apg_cfg_scale, 1.0)
        ):
            noise_pred_neg = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [negative_cond],
                x,
                timestep,
                model_options,
            )[0]

            if self.cfg_star_rescale:
                batch_size = noise_pred_pos.shape[0]

                positive_flat = noise_pred_pos.view(batch_size, -1)
                negative_flat = noise_pred_neg.view(batch_size, -1)
                dot_product = torch.sum(
                    positive_flat * negative_flat, dim=1, keepdim=True
                )
                squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                alpha = dot_product / squared_norm
                noise_pred_neg = alpha * noise_pred_neg

        if not math.isclose(stg_scale, 0.0):
            try:
                model_options["transformer_options"]["ptb_index"] = 0
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [positive_cond],
                    x,
                    timestep,
                    model_options,
                )[0]
            finally:
                self.stg_flag.do_skip = False
                del model_options["transformer_options"]["ptb_index"]

        stg_result = stg(
            noise_pred_pos,
            noise_pred_neg,
            noise_pred_perturbed,
            cfg_value,
            stg_scale,
            stg_rescale,
        )

        if self.apply_apg:
            stg_result = apg(
                stg_result,
                noise_pred_neg,
                cfg_scale=self.apg_cfg_scale,
                momentum_buffer=None,
                eta=self.eta,
                norm_threshold=self.norm_threshold,
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

    DESCRIPTION = (
        "Implements Spatiotemporal Skip Guidance (STG), a training-free method enhancing transformer-based "
        "video diffusion models by selectively skipping layers during sampling. This approach improves video "
        "quality without sacrificing diversity or motion fidelity."
        "Reference: https://arxiv.org/abs/2411.18664."
    )

    def get_guider(self, model, positive, negative, cfg, stg, rescale):
        guider = STGGuider(model, cfg, stg, rescale)
        guider.set_conds(positive, negative)
        return (guider,)


def load_stg_presets():
    preset_file_path = os.path.join(
        os.path.dirname(__file__), "presets", "stg_advanced_presets.json"
    )
    if os.path.exists(preset_file_path):
        presets = json.load(open(preset_file_path))
        preset_names = [preset["name"] for preset in presets]
    else:
        presets = []
        preset_names = ["Custom"]
    return presets, preset_names


STG_ADVANCED_PRESETS, STG_ADVANCED_PRESET_NAMES = load_stg_presets()


@comfy_node(name="STGGuiderAdvanced")
class STGGuiderAdvancedNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "skip_steps_sigma_threshold": (
                    "FLOAT",
                    {
                        "default": 0.998,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.001,
                        "tooltip": "Steps with sigma greater than this values will be skipped.",
                    },
                ),
                "cfg_star_rescale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, applies the cfg star rescale, scales the negative prediction according to dot product between positive and negative.",
                    },
                ),
                "sigmas": (
                    "STRING",
                    {
                        "default": "1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180",
                        "tooltip": "Comma-separated list sigmas, the actual parameters will be selected according to the closest sigma from this list to current timestep sigma.",
                    },
                ),
                "cfg_values": (
                    "STRING",
                    {
                        "default": "8, 6, 6, 4, 3, 1",
                        "tooltip": "Comma-separated list of cfg values. Should be same length as sigmas list.",
                    },
                ),
                "stg_scale_values": (
                    "STRING",
                    {
                        "default": "4, 4, 3, 2, 1, 0",
                        "tooltip": "Comma-separated list of stg scale values. Should be same length as sigmas list.",
                    },
                ),
                "stg_rescale_values": (
                    "STRING",
                    {
                        "default": "1, 1, 1, 1, 1, 1",
                        "tooltip": "Comma-separated list of stg rescale values. Should be same length as sigmas list.",
                    },
                ),
                "stg_layers_indices": (
                    "STRING",
                    {
                        "default": "[29], [29], [29], [29], [29], [29]",
                        "tooltip": "Comma-separated list of list of layer indices. Should be same length as sigmas list.",
                    },
                ),
            },
            "optional": {
                "preset": (
                    "STG_ADVANCED_PRESET",
                    {
                        "tooltip": "Preset resolution and frame count. Custom allows manual input.",
                    },
                ),
                "apply_apg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, applies the APG (Adaptive Projections Guidance) to the STG.",
                    },
                ),
                "apg_cfg_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "eta": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "norm_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "lightricks/LTXV"
    DESCRIPTION = """
    The Advanced STG Guider implements sophisticated techniques for controlling the denoising process:

    It creates a dynamic mapping from scheduler-defined sigma values to CFG and STG (Spatio-Temporal Skip Guidance [1]) parameters.
    This approach establishes distinct sigma value ranges that operate independently of step numbers, allowing precise control over:
    • CFG scale
    • STG scale and rescale factors
    • STG attention layer skipping patterns

    The guider also supports:
    • CFG-Zero* [2] rescaling, which dynamically adjusts negative predictions based on the dot product between positive and negative signals
    • Threshold-based noise prediction zeroing for steps with sigma values exceeding a specified threshold

    For example if the sigma ranges are defined as [1.0, 0.9, 0.85, 0.6] and the CFG values are defined as [4, 3, 2, 1] and STG scale values
    are defined as [2, 2, 2, 1] and STG rescale values are defined as [1, 1, 1, 1] and STG layers indices are defined as [[14, 17], [14, 16], [14], [14]], then the guider will:
    - use CFG=4, STG scale=2, STG rescale=1 and STG layers indices = [14, 17] for sigma in the range (0.9, 1.0]
    - use CFG=3, STG scale=2, STG rescale=1 and STG layers indices = [14, 16] for sigma in the range (0.85, 0.9]
    - use CFG=2, STG scale=2, STG rescale=1 and STG layers indices = [14] for sigma in the range (0.8, 0.85]
    - use CFG=1, STG scale=1, STG rescale=1 and STG layers indices = [14] for sigma in the range (0.6, 0.8]

    The guider will use the same parameters for the same sigma values, regardless of the step number.

    References:
    [1] https://arxiv.org/abs/2411.18664
    [2] https://arxiv.org/abs/2503.18886

    """

    @classmethod
    def parse_stg_layers_indices(cls, stg_layers_indices: str) -> List[List[int]]:
        # First split by "], " to separate the lists, but preserve the closing bracket
        lists = [s + "]" for s in stg_layers_indices.split("],")[:-1]]
        # Add the last list which already has its closing bracket
        if stg_layers_indices.strip():
            lists.append(stg_layers_indices.split("],")[-1])

        result = []
        for s in lists:
            s = s.strip()
            if s == "[]":  # Empty list case
                result.append([])
            else:
                # Remove brackets and whitespace
                s = s.strip("[]").strip()
                if not s:  # Handle case like "[]" after split
                    result.append([])
                else:
                    # Split by comma and convert to integers
                    numbers = [int(n.strip()) for n in s.split(",") if n.strip()]
                    result.append(numbers)

        return result

    def get_guider(
        self,
        model,
        positive,
        negative,
        skip_steps_sigma_threshold,
        cfg_star_rescale,
        sigmas,
        cfg_values,
        stg_scale_values,
        stg_rescale_values,
        stg_layers_indices,
        preset=None,
        apply_apg=False,
        apg_cfg_scale=1.0,
        eta=1.0,
        norm_threshold=0.0,
    ):
        if preset and preset != "Custom":
            preset_data = next(
                (item for item in STG_ADVANCED_PRESETS if item["name"] == preset), None
            )
            if preset_data:
                skip_steps_sigma_threshold = preset_data["skip_steps_sigma_threshold"]
                cfg_star_rescale = preset_data["cfg_star_rescale"]
                sigma_list = preset_data["sigmas"]
                cfg_list = preset_data["cfg_values"]
                stg_scale_list = preset_data["stg_scale_values"]
                stg_rescale_list = preset_data["stg_rescale_values"]
                stg_layers_indices_list = preset_data["stg_layers_indices"]
            else:
                raise ValueError(f"Preset {preset} not found in the presets list.")

        else:
            sigma_list = [float(s.strip()) for s in sigmas.split(",")]
            cfg_list = [float(c.strip()) for c in cfg_values.split(",")]
            stg_scale_list = [float(s.strip()) for s in stg_scale_values.split(",")]
            stg_rescale_list = [float(s.strip()) for s in stg_rescale_values.split(",")]
            stg_layers_indices_list = self.parse_stg_layers_indices(stg_layers_indices)
        print("Using preset: ", preset)
        print("Skip steps sigma threshold: ", skip_steps_sigma_threshold)
        print("Cfg star rescale: ", cfg_star_rescale)
        print("Sigma list: ", sigma_list)
        print("Cfg list: ", cfg_list)
        print("Stg scale list: ", stg_scale_list)
        print("Stg rescale list: ", stg_rescale_list)
        print("Stg layers indices list: ", stg_layers_indices_list)

        guider = STGGuiderAdvanced(
            model,
            sigma_list,
            cfg_list,
            stg_scale_list,
            stg_rescale_list,
            stg_layers_indices_list,
            skip_steps_sigma_threshold,
            cfg_star_rescale,
            apply_apg,
            apg_cfg_scale=apg_cfg_scale,
            eta=eta,
            norm_threshold=norm_threshold,
        )
        guider.set_conds(positive, negative)
        guider.raw_conds = (positive, negative)
        return (guider,)


@comfy_node(name="STGAdvancedPresets")
class STGAdvancedPresetsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (
                    STG_ADVANCED_PRESET_NAMES,
                    {
                        "default": "13b Balanced",
                        "tooltip": "Preset resolution and frame count. Custom allows manual input.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STG_ADVANCED_PRESET",)

    FUNCTION = "get_preset"
    CATEGORY = "lightricks/LTXV"

    def get_preset(self, preset=None):
        return (preset,)


class APGGuider(comfy.samplers.CFGGuider):
    def __init__(
        self,
        model: ModelPatcher,
        cfg_scale,
        eta,
        norm_threshold,
        momentum_coefficient: float = -0.9,
    ):
        self.model = model.clone()
        super().__init__(self.model)
        self.momentum_coefficient = momentum_coefficient
        self.momentum_buffer = MomentumBuffer(self.momentum_coefficient)
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.cfg_scale = cfg_scale
        self.previous_timestep = None

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        model_options: dict = {},
        seed=None,
    ):
        # CFG zero init - skipping steps with timestep bigger than given threshold.

        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and perturbed, so we call calc_cond_batch([perturbed,pos,neg]) directly
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        noise_pred_pos = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [positive_cond],
            x,
            timestep,
            model_options,
        )[0]

        if (
            self.previous_timestep is not None
            and timestep.item() > self.previous_timestep
        ):
            print("Resetting momentum buffer")
            self.momentum_buffer = MomentumBuffer(self.momentum_coefficient)

        noise_pred_neg = 0
        if not math.isclose(self.cfg_scale, 1.0):
            noise_pred_neg = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [negative_cond],
                x,
                timestep,
                model_options,
            )[0]

        apg_result = apg(
            noise_pred_pos,
            noise_pred_neg,
            self.cfg_scale,
            self.eta,
            self.norm_threshold,
        )

        # normally this would be done in cfg_function, but we skipped
        # that for efficiency: we can compute the noise predictions in
        # a single call to calc_cond_batch() (rather than two)
        # so we replicate the hook here
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": apg_result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
            }
            apg_result = fn(args)

        self.previous_timestep = timestep.item()
        return apg_result


@comfy_node(name="APGGuider")
class APGGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "eta": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "norm_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "momentum_coefficient": (
                    "FLOAT",
                    {"default": -0.9, "min": -3.0, "max": 3.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "lightricks/LTXV"

    DESCRIPTION = """
    The APG Guider implements Adaptive Projected Guidance (APG).
    Reference: https://arxiv.org/abs/2410.02416.
    """

    def get_guider(
        self,
        model,
        positive,
        negative,
        cfg_scale,
        eta,
        norm_threshold,
        momentum_coefficient,
    ):
        guider = APGGuider(
            model,
            cfg_scale,
            eta,
            norm_threshold,
            momentum_coefficient,
        )
        guider.set_conds(positive, negative)
        guider.raw_conds = (positive, negative)
        return (guider,)
