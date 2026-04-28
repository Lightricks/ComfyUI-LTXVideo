import logging

import comfy
import comfy.sd
import comfy_extras.nodes_lt as nodes_lt
import folder_paths
import torch
from comfy_api.latest import io

from .latents import LTXVDilateLatent
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, comfy_node


@comfy_node(name="LTXAddVideoICLoRAGuide")
class LTXAddVideoICLoRAGuide(io.ComfyNode):
    PATCHIFIER = nodes_lt.SymmetricPatchifier(1, start_end=True)

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAddVideoICLoRAGuide",
            display_name=NODES_DISPLAY_NAME_PREFIX + " Add Video IC-LoRA Guide",
            category="Lightricks/IC-LoRA",
            description=(
                "Adds one or more conditioning frames starting at the specified frame index. "
                "Supports both single images and multi-frame videos. "
                "The latent_downscale_factor resizes input to a fraction of the target size "
                "(1 = original, 2 = half, 3 = third, etc.) for IC-LoRA on small grids."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input(
                    "latent",
                    tooltip="Video-only latent to condition. Must be a 5D video latent (batch, channels, frames, height, width).",
                ),
                io.Image.Input("image"),
                io.Int.Input(
                    "frame_idx",
                    default=0,
                    min=-9999,
                    max=9999,
                    tooltip=(
                        "Frame index to start the conditioning at. "
                        "For single-frame videos, any frame_idx value is acceptable. "
                        "For videos, frame_idx must be 1 modulo 8, otherwise it will be rounded "
                        "down to the nearest 1 modulo 8. Negative values are counted from the end of the video."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "latent_downscale_factor",
                    default=1.0,
                    min=1.0,
                    max=10.0,
                    step=1.0,
                    tooltip="For IC-LoRA on small grid. 1 means original size, 2 means half size, 3 means third, etc.",
                ),
                io.Combo.Input(
                    "crop",
                    options=["disabled", "center"],
                    default="disabled",
                    tooltip="Crop mode when resizing. 'center' crops to fit, 'disabled' stretches to fit.",
                ),
                io.Boolean.Input(
                    "use_tiled_encode",
                    default=False,
                    tooltip="Enable tiled VAE encoding for large resolutions/long videos to reduce memory usage.",
                ),
                io.Int.Input(
                    "tile_size",
                    default=256,
                    min=64,
                    max=512,
                    step=32,
                    tooltip="Spatial tile size for tiled encoding. Only used when use_tiled_encode is enabled.",
                ),
                io.Int.Input(
                    "tile_overlap",
                    default=64,
                    min=16,
                    max=256,
                    step=16,
                    tooltip="Overlap between tiles for tiled encoding. Only used when use_tiled_encode is enabled.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def encode(
        cls,
        vae,
        latent_width,
        latent_height,
        images,
        scale_factors,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
    ):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        num_frames_to_keep = (
            (images.shape[0] - 1) // time_scale_factor
        ) * time_scale_factor + 1
        images = images[:num_frames_to_keep]
        # Divide target size by latent_downscale_factor
        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(
            latent_height * height_scale_factor / latent_downscale_factor
        )
        pixels = comfy.utils.common_upscale(
            images.movedim(-1, 1),
            target_width,
            target_height,
            "bilinear",
            crop=crop,
        ).movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if use_tiled_encode:
            guide_latent = vae.encode_tiled(
                encode_pixels,
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=tile_overlap,
            )
        else:
            guide_latent = vae.encode(encode_pixels)
        return encode_pixels, guide_latent

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        image,
        frame_idx,
        strength,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
    ) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        time_scale_factor = scale_factors[0]
        num_frames_to_keep = (
            (image.shape[0] - 1) // time_scale_factor
        ) * time_scale_factor + 1
        causal_fix = frame_idx == 0 or num_frames_to_keep == 1
        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        image, guide_latent = cls.encode(
            vae,
            latent_width,
            latent_height,
            image,
            scale_factors,
            latent_downscale_factor,
            crop,
            use_tiled_encode,
            tile_size,
            tile_overlap,
        )

        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            image = image[1:]

        # Record original (pre-dilation) guide latent shape for spatial mask downsampling
        guide_orig_shape = list(guide_latent.shape[2:])  # [F, H_small, W_small]

        guide_mask = None

        # Dilate the latent if latent_downscale_factor > 1
        if latent_downscale_factor > 1:
            if (
                latent_width % latent_downscale_factor != 0
                or latent_height % latent_downscale_factor != 0
            ):
                raise ValueError(
                    f"Latent spatial size {latent_width}x{latent_height} must be divisible by latent_downscale_factor {latent_downscale_factor}"
                )

            dilated = LTXVDilateLatent().dilate_latent(
                {"samples": guide_latent},
                horizontal_scale=int(latent_downscale_factor),
                vertical_scale=int(latent_downscale_factor),
            )[0]
            guide_mask = dilated["noise_mask"]
            guide_latent = dilated["samples"]

        # Pre-filter token count = product of (potentially dilated) spatial dims
        iclora_tokens_added = (
            guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        )

        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
            positive, latent_length, len(image), frame_idx, scale_factors
        )
        assert (
            latent_idx + guide_latent.shape[2] <= latent_length
        ), "Conditioning frames exceed the length of the latent sequence."

        positive, negative, latent_image, noise_mask = (
            nodes_lt.LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                guide_latent,
                strength,
                scale_factors,
                guide_mask=guide_mask,
                latent_downscale_factor=latent_downscale_factor,
                causal_fix=causal_fix,
            )
        )

        # Track this guide in guide_attention_entries for per-reference attention control.
        from .iclora_attention import append_guide_attention_entry

        positive = append_guide_attention_entry(
            positive, iclora_tokens_added, guide_orig_shape
        )
        negative = append_guide_attention_entry(
            negative, iclora_tokens_added, guide_orig_shape
        )

        return io.NodeOutput(
            positive, negative, {"samples": latent_image, "noise_mask": noise_mask}
        )


@comfy_node(name="LTXAddVideoICLoRAGuideAdvanced")
class LTXAddVideoICLoRAGuideAdvanced(LTXAddVideoICLoRAGuide):
    """Extended IC-LoRA guide node with per-guide attention strength control.

    Same as LTXAddVideoICLoRAGuide, but adds attention_strength and
    attention_mask inputs to control how strongly this guide's conditioning
    influences generation via self-attention.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAddVideoICLoRAGuideAdvanced",
            display_name=NODES_DISPLAY_NAME_PREFIX
            + " Add Video IC-LoRA Guide Advanced",
            category="Lightricks/IC-LoRA",
            description=(
                "Adds IC-LoRA guide conditioning with per-guide attention strength control. "
                "Same as LTXAddVideoICLoRAGuide, but allows controlling how strongly this "
                "guide influences generation via self-attention, optionally with a spatial mask."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input(
                    "latent",
                    tooltip="Video-only latent to condition. Must be a 5D video latent.",
                ),
                io.Image.Input("image"),
                io.Int.Input(
                    "frame_idx",
                    default=0,
                    min=-9999,
                    max=9999,
                    tooltip=(
                        "Frame index to start the conditioning at. The value is rounded to the "
                        "nearest frame and wrapped modulo the number of video frames. Negative "
                        "values are counted from the end of the video before wrapping."
                    ),
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "latent_downscale_factor",
                    default=1.0,
                    min=1.0,
                    max=10.0,
                    step=1.0,
                    tooltip="For IC-LoRA on small grid. 1 = original size, 2 = half, etc.",
                ),
                io.Combo.Input(
                    "crop",
                    options=["disabled", "center"],
                    default="disabled",
                ),
                io.Boolean.Input("use_tiled_encode", default=False),
                io.Int.Input("tile_size", default=256, min=64, max=512, step=32),
                io.Int.Input("tile_overlap", default=64, min=16, max=256, step=16),
                io.Float.Input(
                    "attention_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Controls how strongly this guide influences generation via "
                        "self-attention. 1.0 = full conditioning (default), 0.0 = ignore. "
                        "When an attention_mask is also provided, this multiplies the mask values."
                    ),
                ),
                io.Mask.Input(
                    "attention_mask",
                    optional=True,
                    tooltip=(
                        "Optional pixel-space spatial mask. Shape (F, H, W) or (H, W). "
                        "Values in [0, 1]. Controls per-region conditioning influence. "
                        "Multiplied by attention_strength."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        image,
        frame_idx,
        strength,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
        attention_strength=1.0,
        attention_mask=None,
    ) -> io.NodeOutput:
        from .iclora_attention import normalize_mask
        from .latents import LTXVDilateLatent

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        time_scale_factor = scale_factors[0]
        num_frames_to_keep = (
            (image.shape[0] - 1) // time_scale_factor
        ) * time_scale_factor + 1
        causal_fix = frame_idx == 0 or num_frames_to_keep == 1
        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        image, guide_latent = cls.encode(
            vae,
            latent_width,
            latent_height,
            image,
            scale_factors,
            latent_downscale_factor,
            crop,
            use_tiled_encode,
            tile_size,
            tile_overlap,
        )

        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            image = image[1:]

        guide_orig_shape = list(guide_latent.shape[2:])
        guide_mask = None

        if latent_downscale_factor > 1:
            if (
                latent_width % latent_downscale_factor != 0
                or latent_height % latent_downscale_factor != 0
            ):
                raise ValueError(
                    f"Latent spatial size {latent_width}x{latent_height} must be "
                    f"divisible by latent_downscale_factor {latent_downscale_factor}"
                )
            dilated = LTXVDilateLatent().dilate_latent(
                {"samples": guide_latent},
                horizontal_scale=int(latent_downscale_factor),
                vertical_scale=int(latent_downscale_factor),
            )[0]
            guide_mask = dilated["noise_mask"]
            guide_latent = dilated["samples"]

        iclora_tokens_added = (
            guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        )

        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
            positive, latent_length, len(image), frame_idx, scale_factors
        )
        assert (
            latent_idx + guide_latent.shape[2] <= latent_length
        ), "Conditioning frames exceed the length of the latent sequence."

        positive, negative, latent_image, noise_mask = (
            nodes_lt.LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                guide_latent,
                strength,
                scale_factors,
                guide_mask=guide_mask,
                latent_downscale_factor=latent_downscale_factor,
                causal_fix=causal_fix,
            )
        )

        from .iclora_attention import append_guide_attention_entry

        norm_mask = normalize_mask(attention_mask)
        positive = append_guide_attention_entry(
            positive,
            iclora_tokens_added,
            guide_orig_shape,
            attention_strength=attention_strength,
            attention_mask=norm_mask,
        )
        negative = append_guide_attention_entry(
            negative,
            iclora_tokens_added,
            guide_orig_shape,
            attention_strength=attention_strength,
            attention_mask=norm_mask,
        )

        return io.NodeOutput(
            positive, negative, {"samples": latent_image, "noise_mask": noise_mask}
        )


@comfy_node(name="LTXICLoRALoaderModelOnly")
class LTXICLoRALoaderModelOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXICLoRALoaderModelOnly",
            display_name=NODES_DISPLAY_NAME_PREFIX + " IC-LoRA Loader Model Only",
            category="Lightricks/IC-LoRA",
            description="Loads a LoRA model and extracts the latent_downscale_factor from the safetensors metadata.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                ),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Float.Output("latent_downscale_factor"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        # Load lora and extract metadata
        lora, metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )

        # Extract latent_downscale_factor from metadata
        try:
            latent_downscale_factor = float(metadata["reference_downscale_factor"])
        except (KeyError, ValueError, TypeError):
            latent_downscale_factor = 1.0
            logging.warning(
                "Failed to extract reference_downscale_factor from metadata for %s, using 1.0",
                lora_path,
            )

        if strength_model == 0:
            return io.NodeOutput(model, latent_downscale_factor)

        model_lora, _ = comfy.sd.load_lora_for_models(
            model, None, lora, strength_model, 0
        )
        return io.NodeOutput(model_lora, latent_downscale_factor)
