import torch
from comfy.samplers import KSAMPLER
from tqdm import trange


def generate_trend_values(steps, start_time, end_time, eta, eta_trend):
    eta_values = [0] * steps

    if eta_trend == "constant":
        for i in range(start_time, end_time):
            eta_values[i] = eta
    elif eta_trend == "linear_increase":
        for i in range(start_time, end_time):
            progress = (i - start_time) / (end_time - start_time - 1)
            eta_values[i] = eta * progress
    elif eta_trend == "linear_decrease":
        for i in range(start_time, end_time):
            progress = 1 - (i - start_time) / (end_time - start_time - 1)
            eta_values[i] = eta * progress

    return eta_values


def get_sample_forward(
    gamma, start_step, end_step, gamma_trend, seed, attn_bank=None, order="first"
):
    # Controlled Forward ODE (Algorithm 1)
    generator = torch.Generator()
    generator.manual_seed(seed)

    @torch.no_grad()
    def sample_forward(model, y0, sigmas, extra_args=None, callback=None, disable=None):
        if attn_bank is not None:
            for block_idx in attn_bank["block_map"]:
                attn_bank["block_map"][block_idx].clear()

        extra_args = {} if extra_args is None else extra_args
        model_options = extra_args.get("model_options", {})
        model_options = {**model_options}
        transformer_options = model_options.get("transformer_options", {})
        transformer_options = {
            **transformer_options,
            "total_steps": len(sigmas) - 1,
            "sample_mode": "forward",
            "attn_bank": attn_bank,
        }
        model_options["transformer_options"] = transformer_options
        extra_args["model_options"] = model_options

        Y = y0.clone()
        y1 = torch.randn(Y.shape, generator=generator).to(y0.device)
        N = len(sigmas) - 1
        s_in = y0.new_ones([y0.shape[0]])
        gamma_values = generate_trend_values(
            N, start_step, end_step, gamma, gamma_trend
        )
        for i in trange(N, disable=disable):
            transformer_options["step"] = i
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            t_i = model.inner_model.inner_model.model_sampling.timestep(sigmas[i])

            conditional_vector_field = (y1 - Y) / (1 - t_i)

            transformer_options["pred_order"] = "first"
            pred = model(
                Y, s_in * sigmas[i], **extra_args
            )  # this implementation takes sigma instead of timestep

            if order == "second":
                transformer_options["pred_order"] = "second"
                img_mid = Y + (sigma_next - sigma) / 2 * pred
                sigma_mid = sigma + (sigma_next - sigma) / 2
                pred_mid = model(img_mid, s_in * sigma_mid, **extra_args)

                first_order = (pred_mid - pred) / ((sigma_next - sigma) / 2)
                pred = pred + gamma_values[i] * (conditional_vector_field - pred)
                # first_order = first_order + gamma_values[i] * (conditional_vector_field - first_order)
                Y = (
                    Y
                    + (sigma_next - sigma) * pred
                    + 0.5 * (sigma_next - sigma) ** 2 * first_order
                )
            else:
                pred = pred + gamma_values[i] * (conditional_vector_field - pred)
                Y = Y + pred * (sigma_next - sigma)

            if callback is not None:
                callback(
                    {"x": Y, "denoised": Y, "i": i, "sigma": sigma, "sigma_hat": sigma}
                )

        return Y

    return sample_forward


def get_sample_reverse(
    latent_image, eta, start_time, end_time, eta_trend, attn_bank=None, order="first"
):
    # Controlled Reverse ODE (Algorithm 2)
    @torch.no_grad()
    def sample_reverse(model, y1, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        model_options = extra_args.get("model_options", {})
        model_options = {**model_options}
        transformer_options = model_options.get("transformer_options", {})
        transformer_options = {
            **transformer_options,
            "total_steps": len(sigmas) - 1,
            "sample_mode": "reverse",
            "attn_bank": attn_bank,
        }
        model_options["transformer_options"] = transformer_options
        extra_args["model_options"] = model_options

        X = y1.clone()
        N = len(sigmas) - 1
        y0 = latent_image.clone().to(y1.device)
        s_in = y0.new_ones([y0.shape[0]])
        eta_values = generate_trend_values(N, start_time, end_time, eta, eta_trend)
        for i in trange(N, disable=disable):
            transformer_options["step"] = i
            t_i = 1 - model.inner_model.inner_model.model_sampling.timestep(sigmas[i])
            sigma = sigmas[i]
            sigma_prev = sigmas[i + 1]

            conditional_vector_field = (y0 - X) / (1 - t_i)

            transformer_options["pred_order"] = "first"
            pred = model(
                X, sigma * s_in, **extra_args
            )  # this implementation takes sigma instead of timestep

            if order == "second":
                transformer_options["pred_order"] = "second"
                img_mid = X + (sigma_prev - sigma) / 2 * pred
                sigma_mid = sigma + (sigma_prev - sigma) / 2
                pred_mid = model(img_mid, s_in * sigma_mid, **extra_args)

                first_order = (pred_mid - pred) / ((sigma_prev - sigma) / 2)
                pred = -pred + eta_values[i] * (conditional_vector_field + pred)

                first_order = -first_order + eta_values[i] * (
                    conditional_vector_field + first_order
                )
                X = (
                    X
                    + (sigma - sigma_prev) * pred
                    + 0.5 * (sigma - sigma_prev) ** 2 * first_order
                )
            else:
                controlled_vector_field = -pred + eta_values[i] * (
                    conditional_vector_field + pred
                )
                X = X + controlled_vector_field * (sigma - sigma_prev)

            if callback is not None:
                callback(
                    {
                        "x": X,
                        "denoised": X,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigmas[i],
                    }
                )

        return X

    return sample_reverse


class LTXRFForwardODESamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gamma": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "end_step": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                "gamma_trend": (["linear_decrease", "linear_increase", "constant"],),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "attn_bank": ("ATTN_BANK",),
                "order": (["first", "second"],),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "Lightricks/LTXV/Sampling"

    def build(
        self,
        gamma,
        start_step,
        end_step,
        gamma_trend,
        seed=0,
        attn_bank=None,
        order="first",
    ):
        sampler = KSAMPLER(
            get_sample_forward(
                gamma,
                start_step,
                end_step,
                gamma_trend,
                seed,
                attn_bank=attn_bank,
                order=order,
            )
        )

        return (sampler,)


class LTXRFReverseODESamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent_image": ("LATENT",),
                "eta": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "end_step": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1}),
            },
            "optional": {
                "eta_trend": (["linear_decrease", "linear_increase", "constant"],),
                "attn_inj": ("ATTN_INJ",),
                "order": (["first", "second"],),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "Lightricks/LTXV/Sampling"

    def build(
        self,
        model,
        latent_image,
        eta,
        start_step,
        end_step,
        eta_trend="constant",
        attn_inj=None,
        order="first",
    ):
        process_latent_in = model.get_model_object("process_latent_in")
        latent_image = process_latent_in(latent_image["samples"])
        sampler = KSAMPLER(
            get_sample_reverse(
                latent_image,
                eta,
                start_step,
                end_step,
                eta_trend,
                attn_bank=attn_inj,
                order=order,
            )
        )

        return (sampler,)
