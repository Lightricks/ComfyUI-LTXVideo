import comfy
import comfy_extras.nodes_lt as nodes_lt
import comfy_extras.nodes_post_processing as post_processing
import nodes

from .nodes_registry import comfy_node


def blur_internal(image, blur_radius):
    if blur_radius > 0:
        # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel
        # sigma = 0.3 * blur_radius + 0.5 is what is recommended in the OpenCV doc for the
        # relationship between sigma and kernel size 2*blur_radius + 1, however we want somewhat weaker
        # blurring, so we use 0.3 * blur_radius instead, reducing the sigma value by 0.5
        sigma = 0.3 * blur_radius
        image = post_processing.Blur.execute(image, blur_radius, sigma)[0]
    return image


@comfy_node(name="LTXVAddGuideAdvanced")
class LTXVAddGuideAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "frame_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "tooltip": "Frame index to start the conditioning at. For single-frame images or "
                        "videos with 1-8 frames, any frame_idx value is acceptable. For videos with 9+ "
                        "frames, frame_idx must be divisible by 8, otherwise it will be rounded down to "
                        "the nearest multiple of 8. Negative values are counted from the end of the video.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Strength of the conditioning. Higher values will make the conditioning more exact.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 29,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "CRF value for the video. Higher values mean more motion, lower values mean higher quality.",
                    },
                ),
                "blur_radius": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 7,
                        "step": 1,
                        "tooltip": "Blur kernel radius size. Higher values mean more motion, lower values mean higher quality.",
                    },
                ),
                "interpolation": (
                    [
                        "lanczos",
                        "bislerp",
                        "nearest",
                        "bilinear",
                        "bicubic",
                        "area",
                        "nearest-exact",
                    ],
                    {"default": "lanczos"},
                ),
                "crop": (["center", "disabled"], {"default": "disabled"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"

    DESCRIPTION = (
        "Adds a conditioning frame or a video at a specific frame index. "
        "This node is used to add a keyframe or a video segment which should appear in the "
        "generated video at a specified index. It resizes the image to the correct size and "
        "applies preprocessing to it."
    )

    def generate(
        self,
        positive,
        negative,
        vae,
        latent,
        image,
        frame_idx,
        strength,
        crf,
        blur_radius,
        interpolation,
        crop,
    ):
        _, width_scale_factor, height_scale_factor = vae.downscale_index_formula
        width, height = (
            latent["samples"].shape[4] * width_scale_factor,
            latent["samples"].shape[3] * height_scale_factor,
        )
        image = (
            comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, interpolation, crop=crop
            )
            .movedim(1, -1)
            .clamp(0, 1)
        )
        image = nodes_lt.LTXVPreprocess().execute(image, crf)[0]
        image = blur_internal(image, blur_radius)
        return nodes_lt.LTXVAddGuide().execute(
            positive=positive,
            negative=negative,
            vae=vae,
            latent=latent,
            image=image,
            frame_idx=frame_idx,
            strength=strength,
        )


@comfy_node(name="LTXVImgToVideoAdvanced")
class LTXVImgToVideoAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 768,
                        "min": 64,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 32,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 32,
                    },
                ),
                "length": (
                    "INT",
                    {"default": 97, "min": 9, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "crf": (
                    "INT",
                    {
                        "default": 29,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "CRF value for the video. Higher values mean more motion, lower values mean higher quality.",
                    },
                ),
                "blur_radius": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 7,
                        "step": 1,
                        "tooltip": "Blur kernel radius size. Higher values mean more motion, lower values mean higher quality.",
                    },
                ),
                "interpolation": (
                    [
                        "lanczos",
                        "bislerp",
                        "nearest",
                        "bilinear",
                        "bicubic",
                        "area",
                        "nearest-exact",
                    ],
                    {"default": "lanczos"},
                ),
                "crop": (["center", "disabled"], {"default": "disabled"}),
                "strength": ("FLOAT", {"default": 0.9, "min": 0, "max": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"

    DESCRIPTION = (
        "Adds a conditioning frame or a video at index 0. "
        "This node is used to add a keyframe or a video segment which should appear in the "
        "generated video at index 0. It resizes the image to the correct size "
        "and applies preprocessing to it."
    )

    def generate(
        self,
        positive,
        negative,
        vae,
        image,
        width,
        height,
        length,
        batch_size,
        crf,
        blur_radius,
        interpolation,
        crop,
        strength,
    ):
        image = comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, interpolation, crop=crop
        ).movedim(1, -1)
        image = nodes_lt.LTXVPreprocess().execute(image, crf)[0]
        image = blur_internal(image, blur_radius)
        return nodes_lt.LTXVImgToVideo().execute(
            positive=positive,
            negative=negative,
            vae=vae,
            image=image,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            strength=strength,
        )


@comfy_node(name="LTXVAddGuideAdvancedAttention")
class LTXVAddGuideAdvancedAttention:
    """Extended keyframe guide node with per-guide attention strength control.

    Same preprocessing as LTXVAddGuideAdvanced (CRF, blur, interpolation, crop),
    plus attention_strength and attention_mask inputs to control how strongly
    this guide's conditioning influences generation via self-attention.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "frame_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "tooltip": (
                            "Frame index to start the conditioning at. "
                            "Negative values are counted from the end of the video."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Strength of the conditioning. Higher values make it more exact.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 29,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "CRF value. Higher = more motion, lower = higher quality.",
                    },
                ),
                "blur_radius": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 7,
                        "step": 1,
                        "tooltip": "Blur kernel radius. Higher = more motion.",
                    },
                ),
                "interpolation": (
                    [
                        "lanczos",
                        "bislerp",
                        "nearest",
                        "bilinear",
                        "bicubic",
                        "area",
                        "nearest-exact",
                    ],
                    {"default": "lanczos"},
                ),
                "crop": (["center", "disabled"], {"default": "disabled"}),
                "attention_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Controls how strongly this guide influences generation via "
                            "self-attention. 1.0 = full conditioning, 0.0 = ignore."
                        ),
                    },
                ),
            },
            "optional": {
                "attention_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional pixel-space spatial mask. Shape (F, H, W) or (H, W). "
                            "Values in [0, 1]. Controls per-region conditioning influence. "
                            "Multiplied by attention_strength."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"

    DESCRIPTION = (
        "Adds a conditioning frame/video at a specific frame index with per-guide "
        "attention strength control. Same preprocessing as LTXVAddGuideAdvanced, "
        "plus attention_strength and optional spatial attention_mask."
    )

    def generate(
        self,
        positive,
        negative,
        vae,
        latent,
        image,
        frame_idx,
        strength,
        crf,
        blur_radius,
        interpolation,
        crop,
        attention_strength=1.0,
        attention_mask=None,
    ):
        from .iclora_attention import append_guide_attention_entry, normalize_mask

        # Preprocessing: resize, CRF, blur (same as LTXVAddGuideAdvanced)
        scale_factors = vae.downscale_index_formula
        _, width_scale_factor, height_scale_factor = scale_factors
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)
        _, _, latent_length, latent_height, latent_width = latent_image.shape

        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor
        image = (
            comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, interpolation, crop=crop
            )
            .movedim(1, -1)
            .clamp(0, 1)
        )
        image = nodes_lt.LTXVPreprocess().execute(image, crf)[0]
        image = blur_internal(image, blur_radius)

        # Encode
        _, t = nodes_lt.LTXVAddGuide.encode(
            vae, latent_width, latent_height, image, scale_factors
        )

        # Compute latent index
        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
            positive, latent_length, len(image), frame_idx, scale_factors
        )
        assert (
            latent_idx + t.shape[2] <= latent_length
        ), "Conditioning frames exceed the length of the latent sequence."

        # Append keyframe
        positive, negative, latent_image, noise_mask = (
            nodes_lt.LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
            )
        )

        # Track with custom attention strength/mask
        pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
        guide_latent_shape = list(t.shape[2:])
        norm_mask = normalize_mask(attention_mask)
        positive = append_guide_attention_entry(
            positive,
            pre_filter_count,
            guide_latent_shape,
            attention_strength=attention_strength,
            attention_mask=norm_mask,
        )
        negative = append_guide_attention_entry(
            negative,
            pre_filter_count,
            guide_latent_shape,
            attention_strength=attention_strength,
            attention_mask=norm_mask,
        )

        return (
            positive,
            negative,
            {"samples": latent_image, "noise_mask": noise_mask},
        )
