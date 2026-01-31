import torch
from comfy.samplers import KSAMPLER, CFGGuider, sampling_function
from tqdm import trange


class FlowEditGuider(CFGGuider):
    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self.cfgs = {}

    def set_conds(self, **kwargs):
        self.inner_set_conds(kwargs)

    def set_cfgs(self, **kwargs):
        self.cfgs = {**kwargs}

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        latent_type = model_options["transformer_options"]["latent_type"]
        positive = self.conds.get(f"{latent_type}_positive", None)
        negative = self.conds.get(f"{latent_type}_negative", None)
        cfg = self.cfgs.get(latent_type, self.cfg)
        return sampling_function(
            self.inner_model,
            x,
            timestep,
            negative,
            positive,
            cfg,
            model_options=model_options,
            seed=seed,
        )


class LTXFlowEditCFGGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "source_pos": ("CONDITIONING",),
                "source_neg": ("CONDITIONING",),
                "target_pos": ("CONDITIONING",),
                "target_neg": ("CONDITIONING",),
                "source_cfg": (
                    "FLOAT",
                    {"default": 2, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 0.01},
                ),
                "target_cfg": (
                    "FLOAT",
                    {"default": 4.5, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "Lightricks/LTXV/Guidance"

    def get_guider(
        self,
        model,
        source_pos,
        source_neg,
        target_pos,
        target_neg,
        source_cfg,
        target_cfg,
    ):
        guider = FlowEditGuider(model)
        guider.set_conds(
            source_positive=source_pos,
            source_negative=source_neg,
            target_positive=target_pos,
            target_negative=target_neg,
        )
        guider.set_cfgs(source=source_cfg, target=target_cfg)
        return (guider,)


def get_flowedit_sample(skip_steps, refine_steps, seed):
    generator = torch.manual_seed(seed)

    @torch.no_grad()
    def flowedit_sample(
        model, x_init, sigmas, extra_args=None, callback=None, disable=None
    ):
        extra_args = {} if extra_args is None else extra_args

        model_options = extra_args.get("model_options", {})
        transformer_options = model_options.get("transformer_options", {})
        transformer_options = {**transformer_options}
        model_options["transformer_options"] = transformer_options
        extra_args["model_options"] = model_options
        denoise_mask = extra_args.get("denoise_mask", 1)
        if denoise_mask is None:
            denoise_mask = 1
        else:
            extra_args["denoise_mask"] = torch.ones_like(denoise_mask)

        source_extra_args = {
            **extra_args,
            "model_options": {
                "transformer_options": {**transformer_options, "latent_type": "source"}
            },
        }

        sigmas = sigmas[skip_steps:]

        x_tgt = x_init.clone()
        N = len(sigmas) - 1
        s_in = x_init.new_ones([x_init.shape[0]])

        for i in trange(N, disable=disable):
            sigma = sigmas[i]
            noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)

            zt_src = (1 - sigma) * x_init + sigma * noise

            if i < N - refine_steps:
                zt_tgt = x_tgt + (zt_src - x_init) * denoise_mask
                transformer_options["latent_type"] = "source"
                source_extra_args["model_options"]["transformer_options"][
                    "latent_type"
                ] = "source"
                vt_src = model(zt_src, sigma * s_in, **source_extra_args)
            else:
                if i == N - refine_steps:
                    x_tgt = x_tgt + (zt_src - x_init) * denoise_mask
                zt_tgt = x_tgt
                vt_src = 0

            transformer_options["latent_type"] = "target"
            vt_tgt = model(zt_tgt, sigma * s_in, **extra_args)

            v_delta = (vt_tgt - vt_src) * denoise_mask
            x_tgt += (sigmas[i + 1] - sigmas[i]) * v_delta

            if callback is not None:
                callback(
                    {
                        "x": x_tgt,
                        "denoised": x_tgt,
                        "i": i + skip_steps,
                        "sigma": sigmas[i],
                        "sigma_hat": sigmas[i],
                    }
                )

        return x_tgt

    return flowedit_sample


class LTXFlowEditSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "skip_steps": (
                    "INT",
                    {"default": 4, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "refine_steps": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "Lightricks/LTXV/Sampling"

    def build(self, skip_steps, refine_steps, seed):
        sampler = KSAMPLER(get_flowedit_sample(skip_steps, refine_steps, seed))
        return (sampler,)
