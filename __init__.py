from .decoder_noise import DecoderNoise
from .easy_samplers import (
    LinearOverlapLatentTransition,
    LTXVBaseSampler,
    LTXVExtendSampler,
    LTXVInContextSampler,
)
from .film_grain import LTXVFilmGrain
from .guide import LTXVAddGuideAdvanced
from .latent_adain import LTXVAdainLatent
from .latent_upsampler import LTXVLatentUpsampler
from .latents import LTXVSelectLatents, LTXVSetVideoLatentNoiseMasks
from .looping_sampler import LTXVLoopingSampler, MultiPromptProvider
from .masks import LTXVPreprocessMasks
from .nodes_registry import NODE_CLASS_MAPPINGS as RUNTIME_NODE_CLASS_MAPPINGS
from .nodes_registry import (
    NODE_DISPLAY_NAME_MAPPINGS as RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, camel_case_to_spaces
from .prompt_enhancer_nodes import LTXVPromptEnhancer, LTXVPromptEnhancerLoader
from .q8_nodes import LTXVQ8LoraModelLoader, LTXVQ8Patch
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
from .utiltily_nodes import ImageToCPU
from .vae_patcher.vae_patcher import LTXVPatcherVAE

# Static node mappings, required for ComfyUI-Manager mapping to work
NODE_CLASS_MAPPINGS = {
    "Set VAE Decoder Noise": DecoderNoise,
    "LTXVLinearOverlapLatentTransition": LinearOverlapLatentTransition,
    "LTXVAddGuideAdvanced": LTXVAddGuideAdvanced,
    "LTXVAdainLatent": LTXVAdainLatent,
    "LTXVApplySTG": LTXVApplySTG,
    "LTXVBaseSampler": LTXVBaseSampler,
    "LTXVInContextSampler": LTXVInContextSampler,
    "LTXVExtendSampler": LTXVExtendSampler,
    "LTXVFilmGrain": LTXVFilmGrain,
    "LTXVPreprocessMasks": LTXVPreprocessMasks,
    "LTXVLatentUpsampler": LTXVLatentUpsampler,
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
    "STGAdvancedPresets": STGAdvancedPresetsNode,
    "STGGuiderAdvanced": STGGuiderAdvancedNode,
    "STGGuiderNode": STGGuiderNode,
    "LTXVMultiPromptProvider": MultiPromptProvider,
    "ImageToCPU": ImageToCPU,
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

# Export so that ComfyUI can pick them up.
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
