import logging

import comfy
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import folder_paths
import node_helpers
import torch
from comfy_api.latest import io

from .latents import LTXVDilateLatent
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, comfy_node
from .tiled_fusion_plan import snap_frames

FRAME_IDX_TOOLTIP = (
    "RoPE position of the reference, in pixel frames.\n"
    "0 aligns the reference with the generated clip. Use for video references.\n"
    ">0 places the reference at a specific pixel-frame position in the clip.\n"
    "-1 places the reference outside the generated frame range, while still using it "
    "for attention conditioning. Use for single-image references.\n"
    "For multi-frame references, >0 values are rounded down to 1 mod 8."
)


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
                "(1 = original, 2 = half, 3 = third, etc.) for IC-LoRA on small grids. "
                "For LTXVTiledFusionSampler temporal windows, turn on use_streaming and "
                "set tile_frames to the same 8n+1 value as the sampler (97 for LTX)."
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
                    tooltip=FRAME_IDX_TOOLTIP,
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
                io.Boolean.Input(
                    "use_streaming",
                    optional=True,
                    default=False,
                    tooltip=(
                        "Off (default): encode the whole clip as one IC-LoRA guide. "
                        "On: re-encode one fresh guide per LTXVTiledFusionSampler "
                        "temporal window and write the plan the sampler reads. "
                        "Set tile_frames on this node and the sampler to the same "
                        "value (97 for LTX)."
                    ),
                ),
                io.Int.Input(
                    "tile_frames",
                    optional=True,
                    default=97,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip=(
                        "Window length in pixel frames, used only when use_streaming "
                        "is on. Snapped to 8n+1. Must match LTXVTiledFusionSampler. "
                        "Ignored when use_streaming is off."
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
    def resolve_frame_idx(
        cls,
        positive,
        frame_idx,
        latent_length,
        image_length,
        guide_latent,
        scale_factors,
    ):
        """Map frame_idx to the position append_keyframe should use.

        A negative value is a RoPE position before the clip, not an index counted back
        from its end, so it bypasses get_latent_index. That is what lets a one-frame
        reference sit at -1: adjacent to the first generated frame without landing in
        its slot, which is the only position a one-frame guide can occupy that makes it
        indistinguishable from the frame being generated. The length assert is
        meaningless there -- the guide is outside the clip by construction.
        """
        if frame_idx < 0:
            return frame_idx
        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
            positive, latent_length, image_length, frame_idx, scale_factors
        )
        assert (
            latent_idx + guide_latent.shape[2] <= latent_length
        ), "Conditioning frames exceed the length of the latent sequence."
        return frame_idx

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
    def encode_windowed_guides(
        cls,
        positive,
        negative,
        vae,
        latent_image,
        noise_mask,
        image,
        latent_length,
        latent_width,
        latent_height,
        scale_factors,
        strength,
        crop,
        tile_frames,
        attention_strength=1.0,
        attention_mask=None,
        frame_idx=0,
        use_tiled_encode=False,
    ):
        """Re-encode each fusion window as its own clip, or return None.

        Mid-clip causal latents cannot be sliced into windows, so each window's
        pixels are encoded as a clip starting at 0. The shared plan is written
        onto the conditioning so LTXVTiledFusionSampler pairs window w with
        guide block w. Returns None when the clip fits one window.
        """
        from .iclora_attention import append_guide_attention_entry
        from .tiled_fusion_plan import attach_temporal_plan, temporal_plan

        t_scale = scale_factors[0]
        tf_lat = max(1, (tile_frames - 1) // t_scale + 1)
        if not (0 < tf_lat < latent_length):
            return None
        if int(frame_idx) != 0:
            raise ValueError(
                "use_streaming encodes each window as a fresh clip at frame 0; "
                f"frame_idx={frame_idx} is not supported. Use the whole-clip path "
                "(use_streaming off) for stills or mid-clip placement."
            )
        if use_tiled_encode:
            raise ValueError(
                "use_streaming cannot use tiled VAE encode; tiled guide encodes "
                "imprint a grid into the conditioning. Leave use_tiled_encode off."
            )
        keep = ((image.shape[0] - 1) // t_scale) * t_scale + 1
        image = image[:keep]
        ts = temporal_plan(latent_length, tf_lat)
        span_px = (tf_lat - 1) * t_scale + 1
        need = ts[-1] * t_scale + span_px
        if image.shape[0] < need:
            raise ValueError(
                f"windowed guide needs {need} frames for {latent_length} "
                f"content latents, found {image.shape[0]}"
            )
        for w, t0 in enumerate(ts):
            span = image[t0 * t_scale : t0 * t_scale + span_px]
            _, guide_latent = cls.encode(
                vae,
                latent_width,
                latent_height,
                span,
                scale_factors,
                1.0,
                crop,
                False,
                256,
                64,
            )
            shape = list(guide_latent.shape[2:])
            toks = shape[0] * shape[1] * shape[2]
            positive, negative, latent_image, noise_mask = (
                nodes_lt.LTXVAddGuide.append_keyframe(
                    positive,
                    negative,
                    0,
                    latent_image,
                    noise_mask,
                    guide_latent,
                    strength,
                    scale_factors,
                    guide_mask=None,
                    latent_downscale_factor=1.0,
                    causal_fix=True,
                )
            )
            # Pixel mask is (1, 1, F, H, W) after normalize_mask. F==1 broadcasts;
            # otherwise it must match this window's encoded span, not the full clip.
            window_mask = attention_mask
            if window_mask is not None and window_mask.shape[2] != 1:
                start = t0 * t_scale
                window_mask = window_mask[:, :, start : start + span_px]
            positive = append_guide_attention_entry(
                positive,
                toks,
                shape,
                attention_strength=attention_strength,
                attention_mask=window_mask,
            )
            negative = append_guide_attention_entry(
                negative,
                toks,
                shape,
                attention_strength=attention_strength,
                attention_mask=window_mask,
            )
            logging.info(
                "window %d/%d: guide frames %d-%d -> %s",
                w + 1,
                len(ts),
                t0 * t_scale,
                t0 * t_scale + span_px - 1,
                tuple(shape),
            )
        plan = {
            "tile_frames": int(tile_frames),
            "tf_lat": int(tf_lat),
            "n_content": int(latent_length),
            "ts": [int(v) for v in ts],
        }
        positive = attach_temporal_plan(positive, plan)
        negative = attach_temporal_plan(negative, plan)
        logging.info("%d per-window guides (t0 = %s)", len(ts), plan["ts"])
        return io.NodeOutput(
            positive, negative, {"samples": latent_image, "noise_mask": noise_mask}
        )

    @classmethod
    def maybe_encode_windowed(
        cls,
        positive,
        negative,
        vae,
        latent_image,
        noise_mask,
        image,
        latent_length,
        latent_width,
        latent_height,
        scale_factors,
        strength,
        crop,
        use_streaming,
        tile_frames,
        latent_downscale_factor=1.0,
        attention_strength=1.0,
        attention_mask=None,
        frame_idx=0,
        use_tiled_encode=False,
    ):
        """Run the windowed encode only when use_streaming is on. Else None."""
        if not use_streaming:
            return None
        if latent_downscale_factor != 1:
            raise ValueError(
                "use_streaming requires latent_downscale_factor=1; "
                "windowed fusion guides cannot be dilated."
            )
        tf = snap_frames(int(tile_frames) if tile_frames and tile_frames > 0 else 97)
        return cls.encode_windowed_guides(
            positive,
            negative,
            vae,
            latent_image,
            noise_mask,
            image,
            latent_length,
            latent_width,
            latent_height,
            scale_factors,
            strength,
            crop,
            tf,
            attention_strength=attention_strength,
            attention_mask=attention_mask,
            frame_idx=frame_idx,
            use_tiled_encode=use_tiled_encode,
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
        use_streaming=False,
        tile_frames=0,
    ) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        windowed = cls.maybe_encode_windowed(
            positive,
            negative,
            vae,
            latent_image,
            noise_mask,
            image,
            latent_length,
            latent_width,
            latent_height,
            scale_factors,
            strength,
            crop,
            use_streaming,
            tile_frames,
            latent_downscale_factor=latent_downscale_factor,
            frame_idx=frame_idx,
            use_tiled_encode=use_tiled_encode,
        )
        if windowed is not None:
            return windowed

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

        frame_idx = cls.resolve_frame_idx(
            positive, frame_idx, latent_length, len(image), guide_latent, scale_factors
        )

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
                "guide influences generation via self-attention, optionally with a spatial mask. "
                "use_streaming + matching tile_frames re-encodes one guide per Tiled Fusion window."
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
                    tooltip=FRAME_IDX_TOOLTIP,
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
                io.Boolean.Input(
                    "use_streaming",
                    optional=True,
                    default=False,
                    tooltip=(
                        "Off (default): whole-clip guide. On: one fresh encode per "
                        "fusion temporal window. Must match the sampler's tile_frames."
                    ),
                ),
                io.Int.Input(
                    "tile_frames",
                    optional=True,
                    default=97,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip=(
                        "Used only when use_streaming is on. Snapped to 8n+1. "
                        "97 for LTX. Must match the sampler."
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
        use_streaming=False,
        tile_frames=0,
    ) -> io.NodeOutput:
        from .iclora_attention import normalize_mask
        from .latents import LTXVDilateLatent

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        windowed = cls.maybe_encode_windowed(
            positive,
            negative,
            vae,
            latent_image,
            noise_mask,
            image,
            latent_length,
            latent_width,
            latent_height,
            scale_factors,
            strength,
            crop,
            use_streaming,
            tile_frames,
            latent_downscale_factor=latent_downscale_factor,
            attention_strength=attention_strength,
            attention_mask=normalize_mask(attention_mask),
            frame_idx=frame_idx,
            use_tiled_encode=use_tiled_encode,
        )
        if windowed is not None:
            return windowed

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

        frame_idx = cls.resolve_frame_idx(
            positive, frame_idx, latent_length, len(image), guide_latent, scale_factors
        )

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


def _patchify_audio_latent(latent):
    """Reshape audio latent (b, c, t, f) → ref_audio token dict (b, t, c*f)."""
    b, c, t, f = latent.shape
    ref_tokens = latent.permute(0, 2, 1, 3).reshape(b, t, c * f)
    return {"tokens": ref_tokens}


@comfy_node(name="LTXVSetAudioRefTokens")
class LTXVSetAudioRefTokens(io.ComfyNode):
    """Provide speaker identity context for audio generation.

    Patchifies the audio latent and attaches it as reference tokens on both
    positive and negative conditioning. The model prepends these tokens with
    negative temporal positions so they serve as identity context without
    being part of the generated output.

    Also outputs a frozen copy of the audio latent (noise_mask=0) for
    direct use in stage 2 without re-encoding.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVSetAudioRefTokens",
            display_name=NODES_DISPLAY_NAME_PREFIX + " Set Audio Ref Tokens",
            category="Lightricks/IC-LoRA",
            description=(
                "Provides speaker identity context for audio generation by attaching "
                "reference audio tokens to the conditioning. The tokens are prepended "
                "with negative temporal positions so the model treats them as context "
                "rather than generation targets."
            ),
            inputs=[
                io.Conditioning.Input(
                    "positive",
                    tooltip="Positive conditioning to attach the reference audio tokens to.",
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip="Negative conditioning to attach the reference audio tokens to.",
                ),
                io.Latent.Input(
                    "audio_latent",
                    tooltip="Encoded audio latent from LTXV Audio VAE Encode.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="Positive conditioning with reference audio tokens attached.",
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="Negative conditioning with reference audio tokens attached.",
                ),
                io.Latent.Output(
                    display_name="frozen_audio",
                    tooltip="Audio latent with noise_mask=0, fully frozen during denoising.",
                ),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, audio_latent) -> io.NodeOutput:
        latent = audio_latent["samples"]
        ref_audio = _patchify_audio_latent(latent)
        positive = node_helpers.conditioning_set_values(
            positive, {"ref_audio": ref_audio}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"ref_audio": ref_audio}
        )

        frozen = audio_latent.copy()
        b, c, t, f = latent.shape
        frozen["noise_mask"] = torch.zeros((b, 1, t, 1), dtype=torch.float32)

        return io.NodeOutput(positive, negative, frozen)
