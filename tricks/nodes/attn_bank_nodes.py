from ..utils.attn_bank import AttentionBank


class LTXAttentionBankNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_steps": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "blocks": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("ATTN_BANK",)
    FUNCTION = "build"

    CATEGORY = "Lightricks/LTXV/Model Patches"

    def build(self, save_steps, blocks=""):
        block_map = {}
        block_list = blocks.split(",")
        for block in block_list:
            block_idx = int(block)
            block_map[block_idx] = {}

        bank = AttentionBank(save_steps, block_map)
        return (bank,)


class LTXPrepareAttnInjectionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "attn_bank": ("ATTN_BANK",),
                "query": ("BOOLEAN", {"default": False}),
                "key": ("BOOLEAN", {"default": False}),
                "value": ("BOOLEAN", {"default": False}),
                "inject_steps": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000, "step": 1},
                ),
            },
            "optional": {"blocks": ("LTX_BLOCKS",)},
        }

    RETURN_TYPES = ("LATENT", "ATTN_INJ")
    FUNCTION = "prepare"

    CATEGORY = "Lightricks/LTXV/Model Patches"

    def prepare(self, latent, attn_bank, query, key, value, inject_steps, blocks=None):
        if inject_steps > attn_bank["save_steps"]:
            raise ValueError("Can not inject more steps than were saved.")
        attn_bank = AttentionBank(
            attn_bank["save_steps"], attn_bank["block_map"], inject_steps
        )
        attn_bank["inject_settings"] = set([])
        if query:
            attn_bank["inject_settings"].add("q")
        if key:
            attn_bank["inject_settings"].add("k")
        if value:
            attn_bank["inject_settings"].add("v")

        if blocks is not None:
            attn_bank["block_map"] = {**attn_bank["block_map"]}
            for key in list(attn_bank["block_map"].keys()):
                if key not in blocks:
                    del attn_bank["block_map"][key]

        # Hack to force order of operations in ComfyUI graph
        return (latent, attn_bank)


class LTXAttentioOverrideNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"blocks": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("LTX_BLOCKS",)
    FUNCTION = "build"

    CATEGORY = "Lightricks/LTXV/Model Patches"

    def build(self, blocks=""):
        block_set = set(list(int(block) for block in blocks.split(",")))

        return (block_set,)
