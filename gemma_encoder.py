import json
import logging
import gc
from glob import glob
from pathlib import Path
from typing import Optional

import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.supported_models_base
import folder_paths
import safetensors
import torch
from comfy.ldm.lightricks.model import LTXFrequenciesPrecision, LTXRopeType
from comfy.utils import load_torch_file
from einops import rearrange
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
)

from .embeddings_connector import Embeddings1DConnector
from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)

PREFIX_BASE = "model.diffusion_model."


def _get_device_list():
    """Returns a list of available devices for the dropdown."""
    devices = ["auto", "cpu"]
    # Strictly list only detected CUDA devices
    count = torch.cuda.device_count()
    for i in range(count):
        devices.append(f"cuda:{i}")
    return devices


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


def load_video_embeddings_connector(ltxv_path, dtype=torch.bfloat16):
    sd, metadata = load_torch_file(str(ltxv_path), return_metadata=True)
    config = json.loads(metadata.get("config", "{}"))
    transformer_config = config.get("transformer", {})
    rope_type = LTXRopeType.from_dict(transformer_config)
    frequencies_precision = LTXFrequenciesPrecision.from_dict(transformer_config)
    pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])
    sd_keys = list(sd.keys())

    video_only_connector_prefix = f"{PREFIX_BASE}embeddings_connector."
    av_connector_prefix = f"{PREFIX_BASE}video_embeddings_connector."
    prefix = (
        av_connector_prefix
        if f"{PREFIX_BASE}audio_adaln_single.linear.weight" in sd_keys
        else video_only_connector_prefix
    )
    return load_embeddings_connector(
        sd, prefix, dtype, rope_type, frequencies_precision, pe_max_pos
    )


def load_audio_embeddings_connector(ltxv_path, dtype=torch.bfloat16):
    sd, metadata = load_torch_file(str(ltxv_path), return_metadata=True)
    config = json.loads(metadata.get("config", "{}"))
    transformer_config = config.get("transformer", {})
    rope_type = LTXRopeType.from_dict(transformer_config)
    frequencies_precision = LTXFrequenciesPrecision.from_dict(transformer_config)
    pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])
    return load_embeddings_connector(
        sd,
        f"{PREFIX_BASE}audio_embeddings_connector.",
        dtype,
        rope_type,
        frequencies_precision,
        pe_max_pos,
    )


def load_embeddings_connector(
    sd,
    connector_prefix,
    dtype=torch.bfloat16,
    rope_type=LTXRopeType.INTERLEAVED,
    frequencies_precision=LTXFrequenciesPrecision.FLOAT32,
    pe_max_pos=None,
):
    sd_connector = {
        k[len(connector_prefix) :]: v
        for k, v in sd.items()
        if k.startswith(connector_prefix)
    }

    if len(sd_connector) == 0:
        return None

    operations = comfy.ops.pick_operations(dtype, dtype, disable_fast_fp8=True)
    connector = Embeddings1DConnector(
        dtype=dtype,
        operations=operations,
        positional_embedding_max_pos=pe_max_pos if pe_max_pos is not None else [1],
        split_rope=rope_type == LTXRopeType.SPLIT,
        double_precision_rope=frequencies_precision == LTXFrequenciesPrecision.FLOAT64,
    )
    connector.load_state_dict(sd_connector)
    return connector


def load_proj_matrix_from_ltxv(ltxv_path: Path, prefix=""):
    with safetensors.safe_open(ltxv_path, framework="pt", device="cpu") as f:
        keys = filter(lambda key: key.startswith(prefix), f.keys())
        sd = {k.removeprefix(prefix): f.get_tensor(k) for k in keys if k in f.keys()}
    if not sd:
        return None
    model = GemmaFeaturesExtractorProjLinear()
    model.load_state_dict(sd)
    return model


def load_proj_matrix_from_checkpoint(checkpoint_path: Path):
    """
    Load model weights from a checkpoint file.
    :param checkpoint_path: Path to the checkpoint file.
    """
    model = GemmaFeaturesExtractorProjLinear()
    loaded_state_dict = load_torch_file(str(checkpoint_path), return_metadata=False)
    if "aggregate_embed.weight" not in loaded_state_dict:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'aggregate_embed.weight'."
        )
    model.load_state_dict(loaded_state_dict)
    return model


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


class GemmaFeaturesExtractorProjLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)

    def forward(self, x):
        return self.aggregate_embed(x)


class LTXVGemmaTextEncoderModel(torch.nn.Module):
    def __init__(
        self,
        model: Gemma3ForConditionalGeneration,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector | None = None,
        processor: Gemma3Processor | None = None,
        dtype=torch.bfloat16,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.dtypes = set([dtype])

        # Ensure all sub-modules are initialized on the target device immediately
        target_device = self.model.device

        self.feature_extractor_linear = feature_extractor_linear.to(
            dtype=dtype, device=target_device
        )
        self.embeddings_connector = embeddings_connector.to(
            dtype=dtype, device=target_device
        )

        self.audio_embeddings_connector = (
            audio_embeddings_connector.to(dtype=dtype, device=target_device)
            if audio_embeddings_connector is not None
            else None
        )

        # Cache an estimate of memory required to load/keep the model on device
        self._model_memory_required = (
            comfy.model_management.module_size(self.model) + 256 * 1024 * 1024
        )

    def set_clip_options(self, options):
        pass

    def reset_clip_options(self):
        pass

    def to(self, *args, **kwargs):
        """Override .to() to ensure all sub-components move with the main model."""
        super().to(*args, **kwargs)
        
        # Explicitly parse args/kwargs to find device/dtype
        device = None
        dtype = None
        
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
        
        if "device" in kwargs:
            device = kwargs["device"]
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
            
        if device is not None:
            self.model.to(device)
            self.feature_extractor_linear.to(device)
            self.embeddings_connector.to(device)
            if self.audio_embeddings_connector:
                self.audio_embeddings_connector.to(device)
                
        if dtype is not None:
             self.model.to(dtype=dtype)
             self.feature_extractor_linear.to(dtype=dtype)
             self.embeddings_connector.to(dtype=dtype)
             if self.audio_embeddings_connector:
                self.audio_embeddings_connector.to(dtype=dtype)

        return self

    @staticmethod
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

    @staticmethod
    def norm_and_concat_padded_batch(
        encoded_text: torch.Tensor,
        sequence_lengths: torch.Tensor,
        padding_side: str = "right",
    ) -> torch.Tensor:
        """
        Normalize a 4D tensor [B, T, D, L] per sample and per layer, using sequence_lengths to mask.
        Returns [B, T,  D * L] tensor with original padding preserved.
        """
        B, T, D, L = encoded_text.shape
        device = encoded_text.device

        # Build mask: [B, T, 1, 1]
        token_indices = torch.arange(T, device=device)[None, :]  # [1, T]

        if padding_side == "right":
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask = token_indices < sequence_lengths[:, None]  # [B, T]
        elif padding_side == "left":
            # For left padding, valid tokens are from (T - sequence_length) to T-1
            start_indices = T - sequence_lengths[:, None]  # [B, 1]
            mask = token_indices >= start_indices  # [B, T]
        else:
            raise ValueError(
                f"padding_side must be 'left' or 'right', got {padding_side}"
            )

        mask = rearrange(mask, "b t 1 1")

        # Compute masked mean: [B, 1, 1, L]
        masked = encoded_text.masked_fill(~mask, 0.0)
        denom = (sequence_lengths * D).view(B, 1, 1, 1)
        mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + 1e-6)

        # Compute masked min/max: [B, 1, 1, L]
        x_min = encoded_text.masked_fill(~mask, float("inf")).amin(
            dim=(1, 2), keepdim=True
        )
        x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(
            dim=(1, 2), keepdim=True
        )
        range_ = x_max - x_min

        # Normalize only the valid tokens
        normed = 8 * (encoded_text - mean) / (range_ + 1e-6)

        # concat to be [Batch, T,  D * L] - this preserves the original structure
        normed = normed.reshape(B, T, -1)  # [B, T, D * L]

        # Apply mask to preserve original padding (set padded positions to 0)
        mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, D * L)
        normed = normed.masked_fill(~mask_flattened, 0.0)

        return normed

    def forward(self, input_ids, attention_mask, padding_side="right"):
        # Runtime device alignment
        device = self.model.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        # Ensure submodules are aligned
        if self.feature_extractor_linear.aggregate_embed.weight.device != device:
            self.feature_extractor_linear.to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        encoded_text_features = torch.stack(outputs.hidden_states, dim=-1)
        encoded_text_features_dtype = encoded_text_features.dtype

        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = (
            LTXVGemmaTextEncoderModel.norm_and_concat_padded_batch(
                encoded_text_features, sequence_lengths, padding_side=padding_side
            )
        )

        projected = self.feature_extractor_linear(
            normed_concated_encoded_text_features.to(encoded_text_features_dtype)
        )

        return projected

    def encode_token_weights(self, token_weight_pairs):
        token_pairs = token_weight_pairs["gemma"]

        # Use the device the model is currently on
        device = self.model.device

        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=device)

        # Ensure submodules are on the right device
        if self.embeddings_connector.positional_embedding.device != device:
            self.embeddings_connector.to(device)

        encoded_input = self(input_ids, attention_mask, padding_side="left")

        # convert attention mask to format embeddings connector expects.
        connector_attention_mask = (attention_mask - 1).to(encoded_input.dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(encoded_input.dtype).max

        encoded, encoded_connector_attention_mask = self.embeddings_connector(
            encoded_input,
            connector_attention_mask,
        )

        # restore the mask values to int64
        attention_mask = (encoded_connector_attention_mask < 0.000001).to(torch.int64)

        attention_mask = attention_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * attention_mask

        if self.audio_embeddings_connector is not None:
            # Ensure audio connector alignment
            if self.audio_embeddings_connector.positional_embedding.device != device:
                self.audio_embeddings_connector.to(device)

            encoded_for_audio, _ = self.audio_embeddings_connector(
                encoded_input, connector_attention_mask
            )
            encoded = torch.concat(
                [encoded, encoded_for_audio], dim=len(encoded.shape) - 1
            )

        return encoded, None, {"attention_mask": attention_mask.squeeze(-1)}

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def memory_required(self, input_shape):
        # Return a conservative estimate in bytesed(input_shape)
        return self._model_memory_required

    def _pad_inputs_for_attention_alignment(self, model_inputs, alignment: int = 8):
        """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
        seq_len = model_inputs.input_ids.shape[1]
        padded_len = ((seq_len + alignment - 1) // alignment) * alignment
        padding_length = padded_len - seq_len

        if padding_length > 0:
            pad_token_id = (
                self.processor.tokenizer.pad_token_id
                if self.processor.tokenizer.pad_token_id is not None
                else 0
            )

            model_inputs["input_ids"] = self._cat_with_padding(
                model_inputs.input_ids, padding_length, pad_token_id
            )

            model_inputs["attention_mask"] = self._cat_with_padding(
                model_inputs.attention_mask, padding_length, 0
            )

            if (
                "token_type_ids" in model_inputs
                and model_inputs["token_type_ids"] is not None
            ):
                model_inputs["token_type_ids"] = self._cat_with_padding(
                    model_inputs["token_type_ids"], padding_length, 0
                )

        return model_inputs

    def _enhance(
        self,
        messages: list,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 512,
        seed: int = 42,
        forced_target_device: Optional[torch.device] = None,
    ) -> str:
        """Common enhancement logic for both T2V and I2V modes."""
        
        if self.processor is None:
            raise ValueError("Processor not loaded - enhancement not available")

        # 1. Determine target device
        if forced_target_device is not None:
            device = forced_target_device
            logger.info(f"Moving Gemma model to {device}...")
            self.model.to(device)
            self.to(device)
        else:
            device = self.model.device

        # 2. Strict Device Context
        # This ensures internal buffers are created on the correct GPU
        if device.type == "cuda":
            # Explicitly set the current device index to prevent fallback to cuda:0
            torch.cuda.set_device(device)
            cuda_ctx = torch.cuda.device(device)
        else:
            cuda_ctx = torch.no_grad() # Dummy context for CPU
        
        enhanced_prompt = ""
        model_inputs = None
        outputs = None

        try:
            with cuda_ctx:
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                model_inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt",
                )
                
                # Move inputs to device explicitly
                model_inputs = model_inputs.to(device)
                
                # Pad inputs
                model_inputs = self._pad_inputs_for_attention_alignment(model_inputs)

                with torch.inference_mode(), torch.random.fork_rng(devices=[device]):
                    torch.manual_seed(seed)
                    outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                    )
                    generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
                    enhanced_prompt = self.processor.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
        except Exception as e:
            logger.error(f"Error during enhancement: {e}")
            raise e
        finally:
            # 3. Cleanup to prevent fragmentation
            del model_inputs
            del outputs
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int = 512,
        seed: int = 42,
        target_device: Optional[torch.device] = None,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Raw Input Prompt: {prompt}."},
        ]
        return self._enhance(
            messages, 
            max_new_tokens=max_new_tokens, 
            seed=seed, 
            forced_target_device=target_device
        )

    def enhance_i2v(
        self,
        prompt: str,
        image: Image.Image,
        system_prompt: str,
        max_new_tokens: int = 512,
        seed: int = 42,
        target_device: Optional[torch.device] = None,
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
        return self._enhance(
            messages, 
            image=image, 
            max_new_tokens=max_new_tokens, 
            seed=seed,
            forced_target_device=target_device
        )


def ltxv_gemma_tokenizer(tokenizer_path, max_length=256):
    class _LTXVGemmaTokenizer(LTXVGemmaTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(tokenizer_path, max_length=max_length)

    return _LTXVGemmaTokenizer


def ltxv_gemma_clip(
    encoder_path, ltxv_path, processor=None, dtype=None, load_device_str="cpu"
):
    class _LTXVGemmaTextEncoderModel(LTXVGemmaTextEncoderModel):
        def __init__(self, device="cpu", dtype=dtype, model_options={}):
            # Force context to target device if CUDA to prevent cuda:0 leakage
            ctx = torch.no_grad()
            if load_device_str.startswith("cuda:"):
                try:
                    target_idx = int(load_device_str.split(":")[-1])
                    ctx = torch.cuda.device(target_idx)
                except ValueError:
                    pass

            with ctx:
                dtype = torch.bfloat16 

                # CRITICAL: Disable device_map to prevent Accelerate from "pinning" weights
                gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                    encoder_path,
                    local_files_only=True,
                    torch_dtype=dtype,
                    device_map=None, 
                    low_cpu_mem_usage=True,
                )

                feature_extractor_linear = load_proj_matrix_from_ltxv(
                    ltxv_path,
                    "text_embedding_projection.",
                )
                if feature_extractor_linear is None:
                    feature_extractor_linear = load_proj_matrix_from_checkpoint(
                        encoder_path / "proj_linear.safetensors"
                    )

                embeddings_connector = load_video_embeddings_connector(ltxv_path)
                audio_embeddings_connector = load_audio_embeddings_connector(ltxv_path)

                # Move to target device immediately inside the context
                if load_device_str != "cpu":
                    target_device = torch.device(load_device_str)
                    gemma_model = gemma_model.to(target_device)
                
                super().__init__(
                    model=gemma_model,
                    feature_extractor_linear=feature_extractor_linear,
                    embeddings_connector=embeddings_connector,
                    audio_embeddings_connector=audio_embeddings_connector,
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
                "device": (_get_device_list(), {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_model"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Gemma CLIP Loader"
    OUTPUT_NODE = False

    def load_model(
        self,
        gemma_path: str,
        ltxv_path: str,
        max_length: int,
        device: str = "auto",
    ):
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

        # Determine loading device string
        if device.startswith("cuda:"):
            load_device_str = device
        elif device == "cpu":
            load_device_str = "cpu"
        else:
            load_device_str = "cpu"  # default for auto

        clip_dtype = torch.bfloat16
        ltxv_full_path = folder_paths.get_full_path("checkpoints", ltxv_path)
        clip_target = comfy.supported_models_base.ClipTarget(
            tokenizer=tokenizer_class,
            clip=ltxv_gemma_clip(
                gemma_model_path,
                ltxv_full_path,
                processor=processor,
                dtype=clip_dtype,
                load_device_str=load_device_str,
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
                "device": (_get_device_list(), {"default": "auto"}),
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
        device: str = "auto",
        image: Optional[torch.Tensor] = None,
        seed: int = 42,
    ):
        if not isinstance(seed, int):
            seed = 42

        # Device Handling Strategy
        target_device = None
        bypass_comfy_load = False

        if device.startswith("cuda:"):
            target_device = torch.device(device)
            bypass_comfy_load = True
        elif device == "cpu":
            target_device = torch.device("cpu")
            bypass_comfy_load = True
        
        # If specific device is requested, BYPASS Comfy's load_model()
        if not bypass_comfy_load:
            clip.load_model()
            encoder = clip.cond_stage_model
        else:
            encoder = clip.cond_stage_model
        
        if encoder.processor is None:
            raise ValueError(
                "Processor not loaded - enhancement not available. "
                "Ensure your model directory has chat_template.json, processor_config.json, "
                "and preprocessor_config.json files."
            )

        # Determine mode: use I2V if image is provided and not bypassed
        use_i2v = image is not None and not bypass_i2v

        # Auto-select the appropriate system prompt
        if use_i2v and system_prompt.strip() == DEFAULT_T2V_SYSTEM_PROMPT.strip():
            system_prompt = DEFAULT_I2V_SYSTEM_PROMPT
            logger.info("Auto-selected I2V system prompt for image-to-video mode")

        if not system_prompt or not system_prompt.strip():
            raise ValueError(
                "system_prompt is required and cannot be empty or whitespace-only"
            )

        try:
            if use_i2v:
                pil_image = tensor_to_pil(image)
                enhanced_prompt = encoder.enhance_i2v(
                    prompt=prompt,
                    image=pil_image,
                    system_prompt=system_prompt,
                    max_new_tokens=max_tokens,
                    seed=seed,
                    target_device=target_device,
                )
            else:
                enhanced_prompt = encoder.enhance_t2v(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_tokens,
                    seed=seed,
                    target_device=target_device,
                )
        finally:
            # MEMORY UNLOAD: If we moved it to GPU manually, move it back to CPU
            if target_device is not None and target_device.type == "cuda":
                logger.info("Unloading Gemma model to CPU...")
                encoder.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()

        enhanced_prompt = clean_response(enhanced_prompt)

        return (enhanced_prompt,)
