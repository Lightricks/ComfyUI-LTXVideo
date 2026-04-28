import math
from enum import Enum

from ..nodes_registry import comfy_node


class Modality(Enum):
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


class GuiderParameters:
    def __init__(
        self,
        cfg_scale: float = 1.0,
        stg_scale: float = 0.0,
        perturb_attn: bool = True,
        rescale_scale: float = 0.0,
        modality_scale: float = 1.0,
        skip_step: int = 0,
        cross_attn: bool = True,
        cfg_zero_star: bool = False,
        zero_init_sigma: float = 1.0,
    ):
        self.cfg_scale = cfg_scale
        self.stg_scale = stg_scale
        self.perturb_attn = perturb_attn
        self.rescale_scale = rescale_scale
        self.modality_scale = modality_scale
        self.skip_step = skip_step
        self.cross_attn = cross_attn
        self.cfg_zero_star = cfg_zero_star
        self.zero_init_sigma = zero_init_sigma

    def __str__(self):
        return f"cfg_scale: {self.cfg_scale}, stg_scale: {self.stg_scale}, rescale_scale: {self.rescale_scale}, modality_scale: {self.modality_scale}"

    def __repr__(self):
        return f"cfg_scale: {self.cfg_scale}, stg_scale: {self.stg_scale}, rescale_scale: {self.rescale_scale}, modality_scale: {self.modality_scale}"

    def calculate(
        self, noise_pred_pos, noise_pred_neg, noise_pred_pertubed, noise_pred_modality
    ):
        noise_pred = (
            noise_pred_pos
            + (self.cfg_scale - 1) * (noise_pred_pos - noise_pred_neg)
            + self.stg_scale * (noise_pred_pos - noise_pred_pertubed)
            + (self.modality_scale - 1) * (noise_pred_pos - noise_pred_modality)
        )

        if self.rescale_scale != 0:
            factor = noise_pred_pos.std() / noise_pred.std()
            factor = self.rescale_scale * factor + (1 - self.rescale_scale)
            noise_pred = noise_pred * factor

        return noise_pred

    def do_uncond(self):
        return not math.isclose(self.cfg_scale, 1.0)

    def do_perturbed(self):
        return not math.isclose(self.stg_scale, 0.0)

    def do_modality(self):
        return not math.isclose(self.modality_scale, 1.0)

    def do_skip(self, step: int) -> bool:
        if self.skip_step == 0:
            return False

        return step % (self.skip_step + 1) != 0

    def do_cross_attn(self, step: int) -> bool:
        return self.cross_attn and not self.do_skip(step)


@comfy_node(name="GuiderParameters")
class GuiderParametersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modality": (
                    [m.value for m in Modality],
                    {
                        "default": Modality.VIDEO.value,
                    },
                ),
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
                "perturb_attn": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "rescale": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "modality_scale": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "skip_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
                "cross_attn": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "parameters": (
                    "GUIDER_PARAMETERS",
                    {"default": None},
                ),
            },
        }

    RETURN_TYPES = ("GUIDER_PARAMETERS",)

    FUNCTION = "get_parameters"
    CATEGORY = "lightricks/LTXV"

    def get_parameters(
        self,
        modality,
        cfg,
        stg,
        perturb_attn,
        rescale,
        modality_scale,
        skip_step,
        cross_attn,
        parameters=None,
    ):
        parameters = parameters.copy() if parameters is not None else {}

        if modality in parameters:
            raise ValueError(f"Modality {modality} already exists in parameters")

        parameters.update(
            {
                modality: GuiderParameters(
                    cfg,
                    stg,
                    perturb_attn,
                    rescale,
                    modality_scale,
                    skip_step,
                    cross_attn,
                )
            }
        )

        return (parameters,)
