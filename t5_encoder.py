from pathlib import Path

import comfy.clip_model
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.sd1_clip
import comfy.supported_models_base
import comfy.utils
import folder_paths
import torch
from transformers import T5EncoderModel, T5Tokenizer

from .nodes_registry import comfy_node


class LTXVTokenizer(comfy.sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path: str):
        self.tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False):
        """
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        """
        text = text.lower().strip()
        text_inputs = self.tokenizer(
            text,
            padding="max_length",  # do_not_pad, longest, max_length
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask

        out = {
            "t5xxl": [
                (token, weight, i)
                for i, (token, weight) in enumerate(
                    zip(text_input_ids[0], prompt_attention_mask[0])
                )
            ]
        }

        if not return_word_ids:
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out


class LTXVTextEncoderModel(torch.nn.Module):
    def __init__(
        self, encoder_path, dtype_t5=None, device="cpu", dtype=None, model_options={}
    ):
        super().__init__()
        dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        self.t5xxl = (
            T5EncoderModel.from_pretrained(encoder_path, local_files_only=True)
            .to(dtype_t5)
            .to(device)
        )
        self.dtypes = set([dtype, dtype_t5])

    def set_clip_options(self, options):
        pass

    def reset_clip_options(self):
        pass

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]
        text_input_ids = torch.tensor(
            [[t[0] for t in token_weight_pairs_t5]],
            device=self.t5xxl.device,
        )
        prompt_attention_mask = torch.tensor(
            [[w[1] for w in token_weight_pairs_t5]],
            device=self.t5xxl.device,
        )
        self.to(self.t5xxl.device)  # comfyui skips loading some weights to gpu
        out = self.t5xxl(text_input_ids, attention_mask=prompt_attention_mask)[0]
        out = out * prompt_attention_mask.unsqueeze(2)
        return out, None, {"attention_mask": prompt_attention_mask}

    def load_sd(self, sd):
        return self.t5xxl.load_state_dict(sd, strict=False)


def ltxv_clip(encoder_path, dtype_t5=None):
    class LTXVTextEncoderModel_(LTXVTextEncoderModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            super().__init__(
                encoder_path=encoder_path,
                dtype_t5=dtype_t5,
                device=device,
                dtype=dtype,
                model_options=model_options,
            )

    return LTXVTextEncoderModel_


def ltxv_tokenizer(tokenizer_path):
    class LTXVTokenizer_(LTXVTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(tokenizer_path)

    return LTXVTokenizer_


@comfy_node(name="LTXVCLIPModelLoader", description="LTXV CLIP Model Loader")
class LTXVCLIPModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_path": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "The name of the text encoder model to load."},
                )
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_model"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Model Loader"
    OUTPUT_NODE = False

    def load_model(self, clip_path):
        path = Path(folder_paths.get_full_path("text_encoders", clip_path))
        tokenizer_path = path.parents[1] / "tokenizer"
        encoder_path = path.parents[1] / "text_encoder"

        clip_target = comfy.supported_models_base.ClipTarget(
            tokenizer=ltxv_tokenizer(tokenizer_path),
            clip=ltxv_clip(encoder_path, dtype_t5=torch.bfloat16),
        )

        return (comfy.sd.CLIP(clip_target),)
