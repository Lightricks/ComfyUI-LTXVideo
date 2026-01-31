from ..modules.ltx_model import inject_model


class ModifyLTXModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)

    CATEGORY = "Lightricks/LTXV/Model Patches"
    FUNCTION = "modify"

    def modify(self, model):
        model.model.diffusion_model = inject_model(model.model.diffusion_model)
        return (model,)
