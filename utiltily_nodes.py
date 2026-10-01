import math

import torch

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


@comfy_node(description="Looping Reference Schedule")
class LTXVLoopingReferenceSchedule:
    TIME_SCALE = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01},
                ),
                "total_duration": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.1, "max": 3600.0, "step": 0.1},
                ),
                "tile_duration": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 3600.0, "step": 0.1},
                ),
                "overlap_duration": (
                    "FLOAT",
                    {"default": 80 / 24, "min": 0.1, "max": 3600.0, "step": 0.1},
                ),
                "reference_offset": (
                    "FLOAT",
                    {
                        "default": 16 / 24,
                        "min": 0.1,
                        "max": 3600.0,
                        "step": 0.1,
                        "tooltip": "Reference position as seconds before the end of each tile.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING", "INT")
    RETURN_NAMES = (
        "reference_images",
        "frame_count",
        "temporal_tile_size",
        "temporal_overlap",
        "reference_indices",
        "tile_count",
    )
    FUNCTION = "build"
    CATEGORY = "utility"

    @classmethod
    def _aligned_frames(cls, seconds, frame_rate, minimum):
        frames = round(seconds * frame_rate / cls.TIME_SCALE) * cls.TIME_SCALE
        return max(minimum, frames)

    def build(
        self,
        reference_images,
        frame_rate,
        total_duration,
        tile_duration,
        overlap_duration,
        reference_offset,
    ):
        frame_count = max(
            self.TIME_SCALE + 1,
            math.floor((total_duration * frame_rate - 1) / self.TIME_SCALE)
            * self.TIME_SCALE
            + 1,
        )
        tile_size = min(self._aligned_frames(tile_duration, frame_rate, 24), 1000)
        overlap = self._aligned_frames(overlap_duration, frame_rate, 16)
        overlap = min(overlap, 80, tile_size - self.TIME_SCALE)
        reference_margin = self._aligned_frames(
            reference_offset, frame_rate, self.TIME_SCALE
        )
        reference_margin = min(reference_margin, tile_size - self.TIME_SCALE)

        latent_frames = ((frame_count - 1) // self.TIME_SCALE) + 1
        latent_tile_size = tile_size // self.TIME_SCALE
        latent_overlap = overlap // self.TIME_SCALE
        latent_stride = latent_tile_size - latent_overlap
        tile_count = max(
            1, math.ceil((latent_frames - latent_overlap) / latent_stride)
        )

        final_index = ((frame_count - 1) // self.TIME_SCALE) * self.TIME_SCALE
        tile_stride = tile_size - overlap
        reference_indices = [0]
        for tile_index in range(tile_count):
            reference_index = min(
                tile_index * tile_stride + tile_size - reference_margin,
                final_index,
            )
            reference_index -= reference_index % self.TIME_SCALE
            if reference_index not in reference_indices:
                reference_indices.append(reference_index)

        target_count = len(reference_indices)
        source_count = reference_images.shape[0]
        if source_count < 1:
            raise ValueError("reference_images must contain at least one image")
        if source_count >= target_count:
            scheduled_images = reference_images[:target_count]
        else:
            repeated_last = reference_images[-1:].repeat(
                target_count - source_count, 1, 1, 1
            )
            scheduled_images = torch.cat((reference_images, repeated_last), dim=0)

        return (
            scheduled_images,
            frame_count,
            tile_size,
            overlap,
            ", ".join(str(index) for index in reference_indices),
            tile_count,
        )
