from typing import List

import comfy.samplers
import torch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP

from ..nodes_registry import comfy_node
from ..stg import STGBlockWrapper, STGFlag
from .parameters import GuiderParameters, Modality


class MultimodalGuider(comfy.samplers.CFGGuider):
    def __init__(
        self,
        model: ModelPatcher,
        parameters: dict[str, GuiderParameters],
        skip_blocks: List[int],
    ):
        model = model.clone()
        self.current_step = 0
        self.last_denoised_v = None
        self.last_denoised_a = None

        model.add_callback_with_key(
            CallbacksMP.ON_PRE_RUN, "mm_guider_on_pre_run", self.reset_current_step
        )
        super().__init__(model)

        self.stg_flag = STGFlag(
            do_skip=False,
            skip_layers=skip_blocks,
        )

        self.patch_model(model, self.stg_flag)
        self.parameters = parameters

    def reset_current_step(self, model_patcher=None):
        self.current_step = 0
        self.last_denoised_v = None
        self.last_denoised_a = None

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

    def calc_stg_indexes(
        self,
        run_vx: bool,
        run_ax: bool,
        audio_ptb: bool,
        video_ptb: bool,
    ):
        stg_indexes = set()
        num_self_attns = 0
        if run_vx:
            num_self_attns += 1
        if run_ax:
            num_self_attns += 1

        video_attn_idx = 0
        audio_attn_idx = 0 if num_self_attns == 1 else 2

        if video_ptb:
            stg_indexes.add(video_attn_idx)
        if audio_ptb:
            stg_indexes.add(audio_attn_idx)

        return list(stg_indexes)

    def unpack_latents(self, x: torch.Tensor):
        latent_shapes = (
            self.conds.get("positive", {})[0]
            .get("model_conds", {})
            .get("latent_shapes", None)
            .cond
        )
        vx, ax = comfy.utils.unpack_latents(x, latent_shapes)
        return vx, ax

    def pack_latents(self, vx: torch.Tensor, ax: torch.Tensor):
        x, latent_shapes = comfy.utils.pack_latents([vx, ax])
        return x, latent_shapes

    def predict_noise(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        model_options: dict = {},
        seed=None,
    ):
        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and perturbed, so we call calc_cond_batch([perturbed,pos,neg]) directly
        current_step = self.current_step
        self.current_step = current_step + 1

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        audio_params: GuiderParameters = self.parameters.get(
            Modality.AUDIO.value, GuiderParameters()
        )
        video_params: GuiderParameters = self.parameters.get(
            Modality.VIDEO.value, GuiderParameters()
        )

        run_vx = not video_params.do_skip(current_step)
        run_ax = not audio_params.do_skip(current_step)
        run_a2v = video_params.do_cross_attn(current_step)
        run_v2a = audio_params.do_cross_attn(current_step)

        vx, ax = self.unpack_latents(x)
        if not run_vx:
            vx = self.last_denoised_v
        if not run_ax:
            ax = self.last_denoised_a
        x, _ = self.pack_latents(vx, ax)

        if not run_vx and not run_ax:
            return x

        try:
            model_options["transformer_options"]["run_vx"] = run_vx
            model_options["transformer_options"]["run_ax"] = run_ax
            model_options["transformer_options"]["a2v_cross_attn"] = run_a2v
            model_options["transformer_options"]["v2a_cross_attn"] = run_v2a
            noise_pred_pos = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [positive_cond],
                x,
                timestep,
                model_options,
            )[0]
        finally:
            del model_options["transformer_options"]["run_vx"]
            del model_options["transformer_options"]["run_ax"]
            del model_options["transformer_options"]["a2v_cross_attn"]
            del model_options["transformer_options"]["v2a_cross_attn"]
        v_noise_pred_pos, a_noise_pred_pos = self.unpack_latents(noise_pred_pos)

        a_noise_pred_neg, v_noise_pred_neg = 0, 0
        a_noise_pred_perturbed, v_noise_pred_perturbed = 0, 0
        a_noise_pred_modality, v_noise_pred_modality = 0, 0

        if any(params.do_uncond() for params in [audio_params, video_params]):
            try:
                model_options["transformer_options"]["run_vx"] = run_vx
                model_options["transformer_options"]["run_ax"] = run_ax
                model_options["transformer_options"]["a2v_cross_attn"] = run_a2v
                model_options["transformer_options"]["v2a_cross_attn"] = run_v2a
                noise_pred_neg = comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [negative_cond],
                    x,
                    timestep,
                    model_options,
                )[0]
                v_noise_pred_neg, a_noise_pred_neg = self.unpack_latents(noise_pred_neg)
            finally:
                del model_options["transformer_options"]["run_vx"]
                del model_options["transformer_options"]["run_ax"]
                del model_options["transformer_options"]["a2v_cross_attn"]
                del model_options["transformer_options"]["v2a_cross_attn"]

        if any(params.do_perturbed() for params in [audio_params, video_params]):
            try:
                stg_indexes = self.calc_stg_indexes(
                    run_vx,
                    run_ax and ax.numel() > 0,
                    audio_params.perturb_attn,
                    video_params.perturb_attn,
                )
                model_options["transformer_options"]["run_vx"] = run_vx
                model_options["transformer_options"]["run_ax"] = run_ax
                model_options["transformer_options"]["a2v_cross_attn"] = run_a2v
                model_options["transformer_options"]["v2a_cross_attn"] = run_v2a
                model_options["transformer_options"]["ptb_index"] = 0
                model_options["transformer_options"]["stg_indexes"] = stg_indexes
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [positive_cond],
                    x,
                    timestep,
                    model_options,
                )[0]
                v_noise_pred_perturbed, a_noise_pred_perturbed = self.unpack_latents(
                    noise_pred_perturbed
                )
            finally:
                self.stg_flag.do_skip = False
                del model_options["transformer_options"]["ptb_index"]
                del model_options["transformer_options"]["run_vx"]
                del model_options["transformer_options"]["run_ax"]
                del model_options["transformer_options"]["a2v_cross_attn"]
                del model_options["transformer_options"]["v2a_cross_attn"]
                del model_options["transformer_options"]["stg_indexes"]

        if any(params.do_modality() for params in [audio_params, video_params]):
            try:
                model_options["transformer_options"]["run_vx"] = run_vx
                model_options["transformer_options"]["run_ax"] = run_ax
                model_options["transformer_options"]["a2v_cross_attn"] = False
                model_options["transformer_options"]["v2a_cross_attn"] = False
                noise_pred_modality = comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [positive_cond],
                    x,
                    timestep,
                    model_options,
                )[0]
                v_noise_pred_modality, a_noise_pred_modality = self.unpack_latents(
                    noise_pred_modality
                )
            finally:
                del model_options["transformer_options"]["a2v_cross_attn"]
                del model_options["transformer_options"]["v2a_cross_attn"]
                del model_options["transformer_options"]["run_vx"]
                del model_options["transformer_options"]["run_ax"]

        if run_vx:
            vx = video_params.calculate(
                v_noise_pred_pos,
                v_noise_pred_neg,
                v_noise_pred_perturbed,
                v_noise_pred_modality,
            )
        else:
            vx = self.last_denoised_v

        if run_ax:
            ax = audio_params.calculate(
                a_noise_pred_pos,
                a_noise_pred_neg,
                a_noise_pred_perturbed,
                a_noise_pred_modality,
            )
        else:
            ax = self.last_denoised_a

        x, _ = self.pack_latents(vx, ax)

        # normally this would be done in cfg_function, but we skipped
        # that for efficiency: we can compute the noise predictions in
        # a single call to calc_cond_batch() (rather than two)
        # so we replicate the hook here
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": x,
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
            x = fn(args)

        self.last_denoised_v, self.last_denoised_a = self.unpack_latents(x)
        return x


@comfy_node(name="MultimodalGuider")
class MultimodalGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "parameters": ("GUIDER_PARAMETERS",),
                "skip_blocks": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "lightricks/LTXV"

    def get_guider(self, model, positive, negative, parameters, skip_blocks):
        skip_blocks = [int(n.strip()) for n in skip_blocks.split(",") if n.strip()]
        guider = MultimodalGuider(model, parameters, skip_blocks)
        guider.set_conds(positive, negative)
        guider.raw_conds = (positive, negative)
        return (guider,)
