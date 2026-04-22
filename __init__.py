from .conditioning_loader import LTXVLoadConditioning
from .conditioning_saver import LTXVSaveConditioning
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
from .guide import LTXVAddGuideAdvanced, LTXVAddGuideAdvancedAttention
from .guiders import GuiderParametersNode, MultimodalGuiderNode
from .iclora import (
    LTXAddVideoICLoRAGuide,
    LTXAddVideoICLoRAGuideAdvanced,
    LTXICLoRALoaderModelOnly,
)
from .latent_norm import (
    LTXVAdainLatent,
    LTXVPerStepAdainPatcher,
    LTXVPerStepStatNormPatcher,
    LTXVStatNormLatent,
)
from .latents import (
    LTXVAddLatentGuide,
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
from .masks import LTXVPreprocessMasks
from .nodes_registry import NODE_CLASS_MAPPINGS as RUNTIME_NODE_CLASS_MAPPINGS
from .nodes_registry import (
    NODE_DISPLAY_NAME_MAPPINGS as RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, camel_case_to_spaces
from .prompt_enhancer_nodes import LTXVPromptEnhancer, LTXVPromptEnhancerLoader
from .pyramid_blending import LTXVLaplacianPyramidBlend
from .q8_nodes import LTXVQ8LoraModelLoader, LTXVQ8Patch
from .sparse_tracks import LTXVDrawTracks, LTXVSparseTrackEditor
from .stg import (
    LTXVApplySTG,
    STGAdvancedPresetsNode,
    STGGuiderAdvancedNode,
    STGGuiderNode,
)
from .tiled_sampler import LTXVTiledSampler
from .tiled_vae_decode import LTXVTiledVAEDecode
from .tricks import NODE_CLASS_MAPPINGS as TRICKS_NODE_CLASS_MAPPINGS
from .tricks import NODE_DISPLAY_NAME_MAPPINGS as TRICKS_NODE_DISPLAY_NAME_MAPPINGS
from .utiltily_nodes import FloatToInt, ImageToCPU
from .vae_patcher import LTXVPatcherVAE
from .vanish_nodes import LTXVDilateVideoMask, LTXVInpaintPreprocess

# Static node mappings, required for ComfyUI-Manager mapping to work
NODE_CLASS_MAPPINGS = {
    "Set VAE Decoder Noise": DecoderNoise,
    "LTXVLinearOverlapLatentTransition": LinearOverlapLatentTransition,
    "LTXVAddGuideAdvanced": LTXVAddGuideAdvanced,
    "LTXVAddGuideAdvancedAttention": LTXVAddGuideAdvancedAttention,
    "LTXVAddLatentGuide": LTXVAddLatentGuide,
    "LTXVAdainLatent": LTXVAdainLatent,
    "LTXVImgToVideoConditionOnly": LTXVImgToVideoConditionOnly,
    "LTXVPerStepAdainPatcher": LTXVPerStepAdainPatcher,
    "LTXVApplySTG": LTXVApplySTG,
    "LTXVBaseSampler": LTXVBaseSampler,
    "LTXVInContextSampler": LTXVInContextSampler,
    "LTXVExtendSampler": LTXVExtendSampler,
    "LTXVNormalizingSampler": LTXVNormalizingSampler,
    "LTXVPreprocessMasks": LTXVPreprocessMasks,
    "LTXVPatcherVAE": LTXVPatcherVAE,
    "LTXVPromptEnhancer": LTXVPromptEnhancer,
    "LTXVPromptEnhancerLoader": LTXVPromptEnhancerLoader,
    "LTXQ8Patch": LTXVQ8Patch,
    "LTXVQ8LoraModelLoader": LTXVQ8LoraModelLoader,
    "LTXVSelectLatents": LTXVSelectLatents,
    "LTXVSetVideoLatentNoiseMasks": LTXVSetVideoLatentNoiseMasks,
    "LTXVTiledSampler": LTXVTiledSampler,
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
    "LTXVStatNormLatent": LTXVStatNormLatent,
    "LTXVPerStepStatNormPatcher": LTXVPerStepStatNormPatcher,
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
    "LTXVLoadConditioning": LTXVLoadConditioning,
    "LTXVSaveConditioning": LTXVSaveConditioning,
    "LTXVDrawTracks": LTXVDrawTracks,
    "LTXVSparseTrackEditor": LTXVSparseTrackEditor,
    "LTXVDilateVideoMask": LTXVDilateVideoMask,
    "LTXVInpaintPreprocess": LTXVInpaintPreprocess,
    "LTXVLaplacianPyramidBlend": LTXVLaplacianPyramidBlend,
}

# Consistent display names between static and dynamic node mappings in nodes_registry.py,
# to prevent ComfyUI initializing them with default display names.
NODE_DISPLAY_NAME_MAPPINGS = {
    name: f"{NODES_DISPLAY_NAME_PREFIX} {camel_case_to_spaces(name)}"
    for name in NODE_CLASS_MAPPINGS.keys()
}

# Merge the node mappings from tricks into the main mappings
NODE_CLASS_MAPPINGS.update(TRICKS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TRICKS_NODE_DISPLAY_NAME_MAPPINGS)

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
