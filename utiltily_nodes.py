from .nodes_registry import comfy_node


@comfy_node(description="Image to CPU")
class ImageToCPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "utility"

    def run(self, image):
        return (image.cpu(),)
