import os
import shutil

import comfy.model_management
import comfy.model_patcher
import folder_paths
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from .nodes_registry import comfy_node
from .prompt_enhancer_utils import generate_cinematic_prompt

LLM_NAME = ["unsloth/Llama-3.2-3B-Instruct"]

IMAGE_CAPTIONER = ["MiaoshouAI/Florence-2-large-PromptGen-v2.0"]

MODELS_PATH_KEY = "LLM"


class PromptEnhancer(torch.nn.Module):
    def __init__(
        self,
        image_caption_processor: AutoProcessor,
        image_caption_model: AutoModelForCausalLM,
        llm_model: AutoModelForCausalLM,
        llm_tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self.image_caption_processor = image_caption_processor
        self.image_caption_model = image_caption_model
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.device = image_caption_model.device
        # model parameters and buffer sizes plus some extra 1GB.
        self.model_size = (
            self.get_model_size(self.image_caption_model)
            + self.get_model_size(self.llm_model)
            + 1073741824
        )

    def forward(self, prompt, image_conditioning, max_resulting_tokens, seed=-1):
        enhanced_prompt = generate_cinematic_prompt(
            self.image_caption_model,
            self.image_caption_processor,
            self.llm_model,
            self.llm_tokenizer,
            prompt,
            image_conditioning,
            max_new_tokens=max_resulting_tokens,
            seed=seed
        )

        return enhanced_prompt

    @staticmethod
    def get_model_size(model):
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        total_size += sum(b.numel() * b.element_size() for b in model.buffers())
        return total_size

    def memory_required(self, input_shape):
        return self.model_size


@comfy_node(name="LTXVPromptEnhancerLoader")
class LTXVPromptEnhancerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_name": (
                    "STRING",
                    {
                        "default": LLM_NAME,
                        "tooltip": "The hugging face name of the llm model to load.",
                    },
                ),
                "image_captioner_name": (
                    "STRING",
                    {
                        "default": IMAGE_CAPTIONER,
                        "tooltip": "The hugging face name of the image captioning model to load.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LTXV_PROMPT_ENHANCER",)
    RETURN_NAMES = ("prompt_enhancer",)
    FUNCTION = "load"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Prompt Enhancer (Down)Loader"
    OUTPUT_NODE = False
    DESCRIPTION = "Downloads and initializes LLM and image captioning models from Hugging Face to enhance text prompts for image generation."

    def model_path_download_if_needed(self, model_name):
        model_directory = os.path.join(folder_paths.models_dir, MODELS_PATH_KEY)
        os.makedirs(model_directory, exist_ok=True)

        model_name_ = model_name.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name_)

        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download

            try:
                snapshot_download(
                    repo_id=model_name,
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                )
            except Exception:
                shutil.rmtree(model_path, ignore_errors=True)
                raise
        return model_path

    def down_load_llm_model(self, llm_name, load_device):
        model_path = self.model_path_download_if_needed(llm_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )

        return llm_model, llm_tokenizer

    def down_load_image_captioner(self, image_captioner, load_device):
        model_path = self.model_path_download_if_needed(image_captioner)
        image_caption_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

        image_caption_processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        return image_caption_model, image_caption_processor

    def load(self, llm_name, image_captioner_name):
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.vae_offload_device()
        llm_model, llm_tokenizer = self.down_load_llm_model(llm_name, load_device)
        image_caption_model, image_caption_processor = self.down_load_image_captioner(
            image_captioner_name, load_device
        )

        enhancer = PromptEnhancer(
            image_caption_processor, image_caption_model, llm_model, llm_tokenizer
        )
        patcher = comfy.model_patcher.ModelPatcher(
            enhancer,
            load_device,
            offload_device,
        )
        return (patcher,)


@comfy_node(name="LTXVPromptEnhancer")
class LTXVPromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING",),
                "prompt_enhancer": ("LTXV_PROMPT_ENHANCER",),
                "max_resulting_tokens": (
                    "INT",
                    {"default": 256, "min": 32, "max": 512},
                ),
            },
            "optional": {
                "image_prompt": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "enhance"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Prompt Enhancer"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Enhances text prompts for image generation using LLMs. "
        "Optionally incorporates reference images to create more contextually relevant descriptions."
    )

    def enhance(
        self,
        prompt,
        prompt_enhancer: comfy.model_patcher.ModelPatcher,
        image_prompt: torch.Tensor = None,
        max_resulting_tokens=256,
        seed=-1,
    ):
        comfy.model_management.free_memory(
            prompt_enhancer.memory_required([]),
            comfy.model_management.get_torch_device(),
        )
        comfy.model_management.load_model_gpu(prompt_enhancer)
        model = prompt_enhancer.model
        image_conditioning = None
        if image_prompt is not None:
            permuted_image = image_prompt.permute(3, 0, 1, 2)[None, :]
            image_conditioning = [(permuted_image, 0, 1.0)]

        enhanced_prompt = model(prompt, image_conditioning, max_resulting_tokens, seed)
        return (enhanced_prompt[0],)
