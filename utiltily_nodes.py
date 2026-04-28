from .nodes_registry import comfy_node

# Internal keys used to store AV merge metadata inside model options
_AV_META_KEY = "ltxv_av_meta"
_AV_META_VIDEO_ONLY_KEYS = "video_only_keys"
_AV_META_AUDIO_ONLY_KEYS = "audio_only_keys"
_AV_META_SHARED_KEYS = "shared_keys"
_AV_META_AUDIO_SHARED_VALUES = "audio_shared_values"

# Common key constants
_KEY_SAMPLES = "samples"
_KEY_TYPE = "type"
_OPT_TRANSFORMER = "transformer_options"
_OPT_AUDIO_LENGTH = "audio_length"


def _build_av_meta(video_latent: dict, audio_latent: dict) -> dict:
    """Compute metadata that enables lossless split after AV concat.

    Notes:
    - We keep video values for shared keys in the merged latent, so we must
      persist the original audio values per shared key.
    """
    video_keys = set(video_latent.keys()) - {_KEY_SAMPLES}
    audio_keys = set(audio_latent.keys()) - {_KEY_SAMPLES, _KEY_TYPE}
    video_only_keys = list(video_keys - audio_keys)
    audio_only_keys = list(audio_keys - video_keys)
    shared_keys = list(video_keys & audio_keys)
    audio_shared_values = {k: audio_latent[k] for k in shared_keys}
    return {
        _AV_META_VIDEO_ONLY_KEYS: video_only_keys,
        _AV_META_AUDIO_ONLY_KEYS: audio_only_keys,
        _AV_META_SHARED_KEYS: shared_keys,
        _AV_META_AUDIO_SHARED_VALUES: audio_shared_values,
    }


@comfy_node(name="LTXFloatToInt", description="Float To Int")
class FloatToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("FLOAT", {"default": 0.0})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "op"
    CATEGORY = "math/conversion"

    def op(self, a: float) -> tuple[int]:
        return (round(a),)


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
