from .audio_only import LTXVAudioOnlyEmptyVideoLatent, LTXVAudioOnlyModel
from .conditioning_loader import LTXVLoadConditioning
from .decoder_noise import DecoderNoise
from .dynamic_conditioning import DynamicConditioning
from .easy_samplers import (
    LinearOverlapLatentTransition,
    LTXVBaseSampler,
    LTXVExtendSampler,
    LTXVInContextSampler,
    LTXVNormalizingSampler,
)
from .gemma_api_conditioning import GemmaAPITextEncode
from .gemma_encoder import LTXVGemmaCLIPModelLoader, LTXVGemmaEnhancePrompt
from .guide import LTXVImgToVideoAdvanced
from .guiders import GuiderParametersNode, MultimodalGuiderNode
from .hdr_nodes import (
    LTXVHDRDecodePostprocess,
    LTXVLoadEXRSequence,
    LTXVVAEForceFloat32,
)
from .iclora import (
    LTXAddVideoICLoRAGuide,
    LTXAddVideoICLoRAGuideAdvanced,
    LTXICLoRALoaderModelOnly,
    LTXVSetAudioRefTokens,
)
from .latent_norm import LTXVAdainLatent
from .latents import (
    LTXVImgToVideoConditionOnly,
    LTXVSelectLatents,
    LTXVSetVideoLatentNoiseMasks,
)
from .looping_sampler import LTXVLoopingSampler, MultiPromptProvider
from .low_vram_loaders import (
    LowVRAMAudioVAELoader,
    LowVRAMCheckpointLoader,
    LowVRAMLatentUpscaleModelLoader,
)
from .nodes_registry import NODE_CLASS_MAPPINGS as RUNTIME_NODE_CLASS_MAPPINGS
from .nodes_registry import (
    NODE_DISPLAY_NAME_MAPPINGS as RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, camel_case_to_spaces
from .pyramid_blending import LTXVLaplacianPyramidBlend
from .q8_nodes import LTXVQ8LoraModelLoader, LTXVQ8Patch
from .sparse_tracks import LTXVDrawTracks, LTXVSparseTrackEditor
from .stg import (
    LTXVApplySTG,
    STGAdvancedPresetsNode,
    STGGuiderAdvancedNode,
    STGGuiderNode,
)
from .tiled_fusion_sampler import LTXVTiledFusionSampler
from .tiled_fusion_sizes import LTXVGetTilingSizes
from .tiled_sampler import LTXVTiledSampler
from .tiled_vae_decode import LTXVTiledVAEDecode
from .utiltily_nodes import FloatToInt, ImageToCPU
from .vanish_nodes import LTXVDilateVideoMask, LTXVInpaintPreprocess

# Static node mappings, required for ComfyUI-Manager mapping to work
NODE_CLASS_MAPPINGS = {
    "Set VAE Decoder Noise": DecoderNoise,
    "LTXVLinearOverlapLatentTransition": LinearOverlapLatentTransition,
    "LTXVImgToVideoAdvanced": LTXVImgToVideoAdvanced,
    "LTXVAdainLatent": LTXVAdainLatent,
    "LTXVImgToVideoConditionOnly": LTXVImgToVideoConditionOnly,
    "LTXVApplySTG": LTXVApplySTG,
    "LTXVBaseSampler": LTXVBaseSampler,
    "LTXVInContextSampler": LTXVInContextSampler,
    "LTXVExtendSampler": LTXVExtendSampler,
    "LTXVNormalizingSampler": LTXVNormalizingSampler,
    "LTXQ8Patch": LTXVQ8Patch,
    "LTXVQ8LoraModelLoader": LTXVQ8LoraModelLoader,
    "LTXVSelectLatents": LTXVSelectLatents,
    "LTXVSetVideoLatentNoiseMasks": LTXVSetVideoLatentNoiseMasks,
    "LTXVTiledSampler": LTXVTiledSampler,
    "LTXVTiledFusionSampler": LTXVTiledFusionSampler,
    "LTXVGetTilingSizes": LTXVGetTilingSizes,
    "LTXVLoopingSampler": LTXVLoopingSampler,
    "LTXVTiledVAEDecode": LTXVTiledVAEDecode,
    "MultimodalGuider": MultimodalGuiderNode,
    "GuiderParameters": GuiderParametersNode,
    "STGAdvancedPresets": STGAdvancedPresetsNode,
    "STGGuiderAdvanced": STGGuiderAdvancedNode,
    "STGGuiderNode": STGGuiderNode,
    "LTXVMultiPromptProvider": MultiPromptProvider,
    "ImageToCPU": ImageToCPU,
    "LTXFloatToInt": FloatToInt,
    "LTXVGemmaCLIPModelLoader": LTXVGemmaCLIPModelLoader,
    "LTXVGemmaEnhancePrompt": LTXVGemmaEnhancePrompt,
    "GemmaAPITextEncode": GemmaAPITextEncode,
    "DynamicConditioning": DynamicConditioning,
    "LowVRAMCheckpointLoader": LowVRAMCheckpointLoader,
    "LowVRAMAudioVAELoader": LowVRAMAudioVAELoader,
    "LowVRAMLatentUpscaleModelLoader": LowVRAMLatentUpscaleModelLoader,
    "LTXAddVideoICLoRAGuide": LTXAddVideoICLoRAGuide,
    "LTXAddVideoICLoRAGuideAdvanced": LTXAddVideoICLoRAGuideAdvanced,
    "LTXICLoRALoaderModelOnly": LTXICLoRALoaderModelOnly,
    "LTXVSetAudioRefTokens": LTXVSetAudioRefTokens,
    "LTXVLoadConditioning": LTXVLoadConditioning,
    "LTXVDrawTracks": LTXVDrawTracks,
    "LTXVSparseTrackEditor": LTXVSparseTrackEditor,
    "LTXVDilateVideoMask": LTXVDilateVideoMask,
    "LTXVInpaintPreprocess": LTXVInpaintPreprocess,
    "LTXVLaplacianPyramidBlend": LTXVLaplacianPyramidBlend,
    "LTXVHDRDecodePostprocess": LTXVHDRDecodePostprocess,
    "LTXVLoadEXRSequence": LTXVLoadEXRSequence,
    "LTXVVAEForceFloat32": LTXVVAEForceFloat32,
    "LTXVAudioOnlyModel": LTXVAudioOnlyModel,
    "LTXVAudioOnlyEmptyVideoLatent": LTXVAudioOnlyEmptyVideoLatent,
}

# Consistent display names between static and dynamic node mappings in nodes_registry.py,
# to prevent ComfyUI initializing them with default display names.
NODE_DISPLAY_NAME_MAPPINGS = {
    name: f"{NODES_DISPLAY_NAME_PREFIX} {camel_case_to_spaces(name)}"
    for name in NODE_CLASS_MAPPINGS.keys()
}

# Update with runtime mappings (these will override static mappings if there are any differences)
NODE_CLASS_MAPPINGS.update(RUNTIME_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(RUNTIME_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

# Export so that ComfyUI can pick them up.
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
