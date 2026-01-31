import math

import comfy.model_patcher
import comfy.samplers
import torch
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention
from einops import rearrange

DEFAULT_PAG_LTX = {"layers": set([14])}


def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class LTXPerturbedAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "rescale": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            },
            "optional": {
                "attn_override": ("ATTN_OVERRIDE",),
                # "attn_type": (["PAG", "SEG"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Lightricks/LTXV/Model Patches"

    def patch(
        self, model, scale, rescale, cfg, attn_override=DEFAULT_PAG_LTX, attn_type="PAG"
    ):
        m = model.clone()

        def pag_fn(q, k, v, heads, attn_precision=None, transformer_options=None):
            return v

        def seg_fn(q, k, v, heads, attn_precision=None, transformer_options=None):
            _, sequence_length, _ = q.shape
            b, c, f, h, w = transformer_options["original_shape"]

            q = rearrange(q, "b (f h w) d -> b (f d) w h", h=h, w=w)
            kernel_size = math.ceil(6 * scale) + 1 - math.ceil(6 * scale) % 2
            q = gaussian_blur_2d(q, kernel_size, scale)
            q = rearrange(q, "b (f d) w h -> b (f h w) d", f=f)
            return optimized_attention(q, k, v, heads, attn_precision=attn_precision)

        def post_cfg_function(args):
            model = args["model"]

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]

            len_conds = 1 if args.get("uncond", None) is None else 2

            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                if len_conds == 1:
                    return cond_pred
                return uncond_pred + (cond_pred - uncond_pred)

            attn_fn = pag_fn if attn_type == "PAG" else seg_fn
            for block_idx in attn_override["layers"]:
                model_options = comfy.model_patcher.set_model_options_patch_replace(
                    model_options, attn_fn, "layer", "self_attn", int(block_idx)
                )

            (perturbed,) = comfy.samplers.calc_cond_batch(
                model, [cond], x, sigma, model_options
            )

            # if len_conds == 1:
            #     output = cond_pred + scale * (cond_pred - pag)
            # else:
            #     output = cond_pred + (scale-1.0) * (cond_pred - uncond_pred) + scale * (cond_pred - pag)

            output = (
                uncond_pred
                + cfg * (cond_pred - uncond_pred)
                + scale * (cond_pred - perturbed)
            )
            if rescale > 0:
                factor = cond_pred.std() / output.std()
                factor = rescale * factor + (1 - rescale)
                output = output * factor

            return output

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)
