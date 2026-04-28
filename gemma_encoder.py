import logging
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import comfy.model_management
import comfy.sd
import comfy.supported_models_base
import folder_paths
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

from .nodes_registry import comfy_node
from .text_embeddings_connectors import load_text_embeddings_pipeline

logger = logging.getLogger(__name__)


def _load_system_prompt(filename: str) -> str:
    """Load system prompt from file at module level."""
    try:
        prompt_path = Path(__file__).parent / "system_prompts" / filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning(f"Could not load {filename}: {e}")
    return ""


DEFAULT_T2V_SYSTEM_PROMPT = _load_system_prompt("gemma_t2v_system_prompt.txt")
DEFAULT_I2V_SYSTEM_PROMPT = _load_system_prompt("gemma_i2v_system_prompt.txt")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    numpy_image = (tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(numpy_image)


class LTXVGemmaTokenizer:
    def __init__(self, tokenizer_path: str, max_length: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, model_max_length=max_length
        )
        # Gemma expects left padding for chat-style prompts; for plain text it doesn't matter much.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False):
        text = text.strip()
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        tuples = [
            (token_id, attn, i)
            for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0]))
        ]
        out = {"gemma": tuples}

        if not return_word_ids:
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out


class LTXVGemmaTextEncoderModel(torch.nn.Module):
    def __init__(
        self,
        model: Gemma3ForConditionalGeneration,
        feature_extractor,  # FeatureExtractorV1/V2
        embeddings_processor,  # VideoEmbeddingsProcessor or AVEmbeddingsProcessor
        processor: Gemma3Processor | None = None,
        dtype=torch.bfloat16,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.feature_extractor = feature_extractor.to(dtype=dtype)
        self.embeddings_processor = embeddings_processor.to(dtype=dtype)
        self.dtypes = set([dtype])
        # Cache an estimate of memory required to load/keep the model on device
        # weights size + small overhead
        self._model_memory_required = (
            comfy.model_management.module_size(self.model) + 256 * 1024 * 1024
        )

    def set_clip_options(self, options):
        pass

    def reset_clip_options(self):
        pass

    def forward(self, input_ids, attention_mask, padding_side="right"):
        # Block 1: Run Gemma model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_layer_hiddens = torch.stack(outputs.hidden_states, dim=-1)  # [B, T, D, L]

        # Block 2: Feature extraction
        features = self.feature_extractor(
            all_layer_hiddens, attention_mask, padding_side
        )
        return features  # dict with "video" and optionally "audio"

    def encode_token_weights(self, token_weight_pairs):
        token_pairs = token_weight_pairs["gemma"]
        input_ids = torch.tensor(
            [[t[0] for t in token_pairs]], device=self.model.device
        )
        attention_mask = torch.tensor(
            [[w[1] for w in token_pairs]], device=self.model.device
        )

        self.to(self.model.device)

        features = self(input_ids, attention_mask, padding_side="left")

        # Convert binary mask -> additive mask for processor
        encoded_input_dtype = next(iter(features.values())).dtype
        connector_attention_mask = (attention_mask - 1).to(encoded_input_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(encoded_input_dtype).max

        # Block 3: Embeddings processor
        encoded, mask = self.embeddings_processor.create_embeddings(
            features, connector_attention_mask
        )

        return encoded, None, {"attention_mask": mask}

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def memory_required(self, input_shape):
        # Return a conservative estimate in bytesed(input_shape)
        return self._model_memory_required


def ltxv_gemma_tokenizer(tokenizer_path, max_length=256):
    class _LTXVGemmaTokenizer(LTXVGemmaTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(tokenizer_path, max_length=max_length)

    return _LTXVGemmaTokenizer


def ltxv_gemma_clip(encoder_path, ltxv_path, processor=None, dtype=None):
    class _LTXVGemmaTextEncoderModel(LTXVGemmaTextEncoderModel):
        def __init__(self, device="cpu", dtype=dtype, model_options={}):
            dtype = torch.bfloat16  # TODO: make this configurable

            gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                encoder_path,
                local_files_only=True,
                torch_dtype=dtype,
            )

            feature_extractor, embeddings_processor = load_text_embeddings_pipeline(
                ltxv_path,
                dtype=dtype,
                fallback_proj_path=encoder_path / "proj_linear.safetensors",
            )

            super().__init__(
                model=gemma_model,
                feature_extractor=feature_extractor,
                embeddings_processor=embeddings_processor,
                processor=processor,
                dtype=dtype,
                device=device,
            )

    return _LTXVGemmaTextEncoderModel


def find_matching_dir(root_path: str, pattern: str) -> str:
    """
    Recursively search for files matching a glob pattern and return the parent directory of the first match.
    """
    matches = [
        Path(p)
        for p in glob(f"{root_path}/**", recursive=True)
        if Path(p).match(pattern)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found under {root_path}"
        )
    return str(matches[0].parent)


@comfy_node(name="LTXVGemmaCLIPModelLoader", description="Gemma 3 Model Loader")
class LTXVGemmaCLIPModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gemma_path": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "The name of the text encoder model to load."},
                ),
                "ltxv_path": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the ltxv model to load."},
                ),
                "max_length": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 131072, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_model"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Gemma CLIP Loader"
    OUTPUT_NODE = False

    def load_model(self, gemma_path: str, ltxv_path: str, max_length: int):
        path = Path(folder_paths.get_full_path("text_encoders", gemma_path))
        model_root = path.parents[1]
        tokenizer_path = Path(find_matching_dir(model_root, "tokenizer.model"))
        gemma_model_path = Path(find_matching_dir(model_root, "model*.safetensors"))
        processor_path = Path(find_matching_dir(model_root, "preprocessor_config.json"))
        tokenizer_class = ltxv_gemma_tokenizer(tokenizer_path, max_length=max_length)

        processor = None
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                str(processor_path),
                local_files_only=True,
            )
            processor = Gemma3Processor(
                image_processor=image_processor,
                tokenizer=tokenizer_class().tokenizer,
            )
            logger.info(f"Loaded processor from {model_root} - enhancement enabled")
        except Exception as e:
            logger.warning(f"Could not load processor from {model_root}: {e}")

        clip_dtype = torch.bfloat16
        ltxv_full_path = folder_paths.get_full_path("checkpoints", ltxv_path)
        clip_target = comfy.supported_models_base.ClipTarget(
            tokenizer=tokenizer_class,
            clip=ltxv_gemma_clip(
                gemma_model_path, ltxv_full_path, processor=processor, dtype=clip_dtype
            ),
        )

        return (comfy.sd.CLIP(clip_target),)


_UNICODE_REPLACEMENTS = str.maketrans(
    "\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-"
)


def clean_response(text):
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


@comfy_node(name="LTXVGemmaEnhancePrompt", description="Gemma 3 Prompt Enhancer")
class LTXVGemmaEnhancePrompt:
    """Enhance prompts using Gemma 3 model. Supports T2V and I2V modes."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_T2V_SYSTEM_PROMPT,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 32, "max": 1024, "step": 16},
                ),
                "bypass_i2v": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0xFFFFFFFF},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Gemma Enhance Prompt"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Enhance text prompts using Gemma 3 VLLM for improved video generation."
    )

    def enhance(
        self,
        clip,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        bypass_i2v: bool,
        image: Optional[torch.Tensor] = None,
        seed: int = 42,
    ):
        if not isinstance(seed, int):
            seed = 42

        clip.load_model()
        encoder = clip.cond_stage_model

        if not hasattr(encoder, "processor") or encoder.processor is None:
            if hasattr(encoder, "gemma3_12b"):
                model, processor = transformers_gemma3_from_encoder(encoder)
            else:
                raise ValueError(
                    "Processor not loaded - enhancement not available. "
                    "Ensure your model directory has chat_template.json, processor_config.json, "
                    "and preprocessor_config.json files."
                )
        elif isinstance(encoder, LTXVGemmaTextEncoderModel):
            model = encoder.model
            processor = encoder.processor
        # Determine mode: use I2V if image is provided and not bypassed
        use_i2v = image is not None and not bypass_i2v

        # Auto-select the appropriate system prompt if user is using default T2V prompt
        if use_i2v and system_prompt.strip() == DEFAULT_T2V_SYSTEM_PROMPT.strip():
            system_prompt = DEFAULT_I2V_SYSTEM_PROMPT
            logger.info("Auto-selected I2V system prompt for image-to-video mode")

        if not system_prompt or not system_prompt.strip():
            raise ValueError(
                "system_prompt is required and cannot be empty or whitespace-only"
            )

        if use_i2v:
            pil_image = tensor_to_pil(image)
            enhanced_prompt = enhance_i2v(
                processor=processor,
                model=model,
                prompt=prompt,
                image=pil_image,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                seed=seed,
            )
        else:
            enhanced_prompt = enhance_t2v(
                processor=processor,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                seed=seed,
            )

        enhanced_prompt = clean_response(enhanced_prompt)

        return (enhanced_prompt,)


def _enhance(
    processor: Gemma3Processor,
    model: Gemma3ForConditionalGeneration,
    messages: list,
    image: Optional[Image.Image] = None,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> str:
    """Common enhancement logic for both T2V and I2V modes."""
    if processor is None:
        raise ValueError("Processor not loaded - enhancement not available")

    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    pad_token_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else 0
    )
    model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id)

    with (
        torch.inference_mode(),
        torch.random.fork_rng(devices=[model.device]),
        torch.autocast(device_type=model.device.type, dtype=model.dtype),
    ):
        torch.manual_seed(seed)
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
        generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
        enhanced_prompt = processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

    return enhanced_prompt


def enhance_t2v(
    processor: Gemma3Processor,
    model: Gemma3ForConditionalGeneration,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> str:
    """Enhance a text prompt for T2V generation."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Raw Input Prompt: {prompt}."},
    ]
    return _enhance(
        processor, model, messages, max_new_tokens=max_new_tokens, seed=seed
    )


def enhance_i2v(
    processor: Gemma3Processor,
    model: Gemma3ForConditionalGeneration,
    prompt: str,
    image: Image.Image,
    system_prompt: str,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> str:
    """Enhance a text prompt for I2V generation using a reference image."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
            ],
        },
    ]
    return _enhance(
        processor,
        model,
        messages,
        image=image,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(model_inputs, pad_token_id, alignment: int = 8):
    """Pad sequence length to multiple of alignment for Flash Attention compatibility.

    Flash Attention within SDPA requires sequence lengths aligned to 8 bytes.
    This pads input_ids, attention_mask, and token_type_ids (if present) to prevent
    'p.attn_bias_ptr is not correctly aligned' errors.
    """
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(
            model_inputs.input_ids, padding_length, pad_token_id
        )

        model_inputs["attention_mask"] = _cat_with_padding(
            model_inputs.attention_mask, padding_length, 0
        )

        if (
            "token_type_ids" in model_inputs
            and model_inputs["token_type_ids"] is not None
        ):
            model_inputs["token_type_ids"] = _cat_with_padding(
                model_inputs["token_type_ids"], padding_length, 0
            )

    return model_inputs


def _locate_model_within_model(super_model, model_name):
    class_name = MODEL_MAPPING_NAMES.get(model_name, None)
    if class_name is None:
        return None
    for module in super_model.modules():
        if module.__class__.__name__ == class_name:
            return module
    return None


def _locate_unique_parameter_owner_by_leaf(
    root: torch.nn.Module,
    leaf_param_name: str,
    must_have_in_path: Optional[str] = None,
) -> Optional[Tuple[torch.nn.Module, str, torch.nn.Parameter, str]]:

    modules = dict(root.named_modules())

    candidates: List[Tuple[torch.nn.Module, str, torch.nn.Parameter, str]] = []
    for full_name, p in root.named_parameters(recurse=True):
        parts = full_name.split(".")
        leaf = parts[-1]
        if leaf != leaf_param_name:
            continue
        if must_have_in_path is not None and must_have_in_path not in parts:
            continue

        owner_path = ".".join(parts[:-1])
        owner = modules.get(owner_path, root if owner_path == "" else None)
        if owner is None:
            continue
        candidates.append((owner, leaf, p, full_name))

    if not candidates:
        return None
    return candidates[0]


def transformers_gemma3_from_encoder(encoder):
    jsons_path = Path(__file__).parent / "gemma_configs"
    config = Gemma3Config.from_json_file(jsons_path / "gemma3cfg.json")
    with torch.device("meta"):
        metamodel = Gemma3ForConditionalGeneration(config)
    t_model_name = config.text_config.model_type
    t_model = _locate_model_within_model(metamodel, t_model_name)
    if t_model is None:
        raise ValueError(
            "Can't locate text model while converting text encoder to Gemma3ForConditionalGeneration"
        )
    t_model.load_state_dict(
        encoder.gemma3_12b.transformer.model.state_dict(), assign=True, strict=False
    )
    v_tower_name = config.vision_config.model_type
    v_tower = _locate_model_within_model(metamodel, v_tower_name)
    if v_tower is None:
        raise ValueError(
            "Can't locate vision model while converting text encoder to Gemma3ForConditionalGeneration"
        )
    v_model = v_tower.vision_model
    v_model.load_state_dict(
        encoder.gemma3_12b.transformer.vision_model.state_dict(),
        assign=True,
        strict=False,
    )
    metamodel.multi_modal_projector.load_state_dict(
        encoder.gemma3_12b.transformer.multi_modal_projector.state_dict(),
        assign=True,
        strict=False,
    )
    config = config.text_config
    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    base = config.rope_local_base_freq

    device = encoder.device
    positions_length = len(v_model.embeddings.position_ids[0])
    position_ids = torch.arange(
        positions_length, dtype=torch.long, device="cpu"
    ).unsqueeze(0)
    v_model.embeddings.register_buffer("position_ids", position_ids)
    embed_scale = torch.tensor(config.hidden_size**0.5, device=device)
    t_model.embed_tokens.register_buffer("embed_scale", embed_scale)
    local_rope_freqs = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    t_model.rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)
    rope_freqs, _ = ROPE_INIT_FUNCTIONS[config.rope_scaling["rope_type"]](
        config, device
    )
    t_model.rotary_emb.register_buffer("inv_freq", rope_freqs)
    lm_head_requires_grad = False
    loc_result = _locate_unique_parameter_owner_by_leaf(
        metamodel, leaf_param_name="weight", must_have_in_path="lm_head"
    )
    if loc_result is None:
        raise ValueError(
            "Can't locate lm_head while converting text encoder to Gemma3ForConditionalGeneration"
        )
    lm_head_owner, lm_head_attr, _, _ = loc_result
    real_w = t_model.embed_tokens.weight
    setattr(
        lm_head_owner,
        lm_head_attr,
        torch.nn.Parameter(real_w, requires_grad=lm_head_requires_grad),
    )
    metamodel.to(device)

    tokenizer_class = ltxv_gemma_tokenizer(jsons_path, max_length=1024)
    image_processor = AutoImageProcessor.from_pretrained(
        str(jsons_path),
        local_files_only=True,
    )
    processor = Gemma3Processor(
        image_processor=image_processor,
        tokenizer=tokenizer_class().tokenizer,
    )
    return metamodel, processor
