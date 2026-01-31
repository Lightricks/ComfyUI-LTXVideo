import comfy.latent_formats
import comfy.model_sampling
import comfy.sd


class InverseCONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_output

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class LTXForwardModelSamplingPredNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Lightricks/LTXV/Prediction"

    def patch(self, model):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = InverseCONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=1.15)
        m.add_object_patch("model_sampling", model_sampling)
        return (m,)


class ReverseCONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_output  # model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


class LTXReverseModelSamplingPredNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Lightricks/LTXV/Prediction"

    def patch(self, model):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = ReverseCONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=1.15)
        m.add_object_patch("model_sampling", model_sampling)
        return (m,)
