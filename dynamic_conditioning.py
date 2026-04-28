import torch

from .nodes_registry import comfy_node


@comfy_node(name="DynamicConditioning")
class DynamicConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "power": ("FLOAT", {"default": 1.3, "min": 1, "max": 2, "step": 0.01}),
                "only_first_frame": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "lightricks/LTXV"
    INIT = False

    def apply(self, model, power, only_first_frame):
        self.only_first_frame = only_first_frame
        self.power = power
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def find_step(self, sigma: torch.Tensor, step_sigmas: torch.Tensor):
        for i, step_sigma in enumerate(step_sigmas):
            if step_sigma <= sigma:
                return i
        return len(step_sigmas) - 1

    def forward(
        self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict
    ):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        step = self.find_step(sigma, step_sigmas)
        # In order to apply power multiple times, this is the same as applying power number of times equal to step
        power = self.power**step
        denoise_mask = denoise_mask.clone()
        if self.only_first_frame:
            num_channels = model.model_patcher.model.diffusion_model.in_channels
            denoise_mask[:, :num_channels, :1] **= power
        else:
            denoise_mask **= power
        # make sure to update the denoise mask in the model, to get correct timestep values for all tokens
        for k in model.conds:
            if "positive" in k or "negative" in k:
                for cond in model.conds[k]:
                    if "model_conds" in cond and "denoise_mask" in cond["model_conds"]:
                        cond["model_conds"]["denoise_mask"].cond = denoise_mask
        # print(f"DynamicConditioning: power: {power}, step: {step}, sigma: {sigma}, step_sigmas: {step_sigmas}")
        return denoise_mask
