def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


class LTXAttnOverrideNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("ATTN_OVERRIDE",)
    FUNCTION = "build"

    CATEGORY = "Lightricks/LTXV/Model Patches"

    def build(self, layers):
        layers_map = set([])
        for block in layers.split(","):
            block = block.strip()
            if is_integer(block):
                layers_map.add(int(block))

        return ({"layers": layers_map},)
