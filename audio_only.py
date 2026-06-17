"""Audio-only (text-to-audio) mode for the LTX-2 audio/video model.

The LTX-2 transformer in ComfyUI core (``comfy/ldm/lightricks/av_model.py``)
already supports an audio-only path: every block reads ``run_vx``,
``a2v_cross_attn`` and ``v2a_cross_attn`` from ``transformer_options`` and skips
the video self-attention/FFN and the audio<->video cross-attention when they are
``False``. These flags default to ``True``, so the model normally runs jointly.

``LTXVAudioOnlyModel`` flips those flags, which is the equivalent of the
reference pipeline's ``video=None``: the audio is denoised with no dependence on
the video latent, and the video stream is skipped for speed.

Note: the model splits its input positionally into ``[video, audio]``, so the
sampler still needs a (tiny) dummy video latent at index 0. Use
``LTXVAudioOnlyEmptyVideoLatent`` (a fixed 64x64 single-frame placeholder) joined
with the audio latent via ``LTXVConcatAVLatent``. With ``LTXVAudioOnlyModel``
active that dummy video is never attended to.
"""

import comfy.model_management
import torch
from comfy.model_patcher import ModelPatcher

from .nodes_registry import comfy_node


@comfy_node(name="LTXVAudioOnlyModel")
class LTXVAudioOnlyModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The LTX-2 audio/video model to run in audio-only mode."
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Lightricks/audio"
    DESCRIPTION = (
        "Run the LTX-2 audio/video model in audio-only mode for text-to-audio. "
        "Skips the video stream and disables the audio<->video cross-attention, so "
        "the generated audio does not depend on the video latent. The sampler still "
        "needs a minimal dummy video latent (EmptyLTXVLatentVideo at 64x64 length 1, "
        "joined with LTXVConcatAVLatent); it is never attended to. Remove or bypass "
        "this node to restore normal joint audio+video generation."
    )

    def patch(self, model: ModelPatcher):
        new_model = model.clone()
        transformer_options = new_model.model_options["transformer_options"]
        transformer_options["run_vx"] = False
        transformer_options["a2v_cross_attn"] = False
        transformer_options["v2a_cross_attn"] = False
        return (new_model,)


@comfy_node(name="LTXVAudioOnlyEmptyVideoLatent")
class LTXVAudioOnlyEmptyVideoLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "Lightricks/audio"
    DESCRIPTION = (
        "Fixed minimal placeholder video latent for audio-only (text-to-audio) "
        "generation - a single 64x64 frame. LTX-2 needs a video latent at input "
        "index 0, but with LTXV Audio Only Model it is never attended to, so the "
        "dimensions are fixed (and hidden) to avoid confusion. Join it with the "
        "audio latent via LTXVConcatAVLatent."
    )

    def generate(self):
        # LTX-2 video latent shape: (batch, 128 channels, frames, H/32, W/32).
        # Fixed to batch 1, a single frame, 64x64 -> (1, 128, 1, 2, 2).
        latent = torch.zeros(
            [1, 128, 1, 2, 2],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)
