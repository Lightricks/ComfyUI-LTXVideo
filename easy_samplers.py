import copy

import comfy
import comfy_extras
import nodes
import torch
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVAddGuide, LTXVCropGuides
from nodes import VAEEncode
import json

from .guide import blur_internal
from .latent_adain import LTXVAdainLatent
from .latents import LTXVAddLatentGuide, LTXVSelectLatents
from .nodes_registry import comfy_node


@comfy_node(
    name="LTXVBaseSampler",
)
class LTXVBaseSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to use."}),
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
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
                "num_frames": (
                    "INT",
                    {"default": 97, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "guider": (
                    "GUIDER",
                    {"tooltip": "The guider to use, must be a STGGuiderAdvanced."},
                ),
                "sampler": ("SAMPLER", {"tooltip": "The sampler to use."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigmas to use."}),
                "noise": ("NOISE", {"tooltip": "The noise to use for the sampling."}),
            },
            "optional": {
                "optional_cond_images": (
                    "IMAGE",
                    {"tooltip": "The images to use for conditioning the sampling."},
                ),
                "optional_cond_indices": (
                    "STRING",
                    {
                        "tooltip": "The indices of the images to use for conditioning the sampling."
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0,
                        "max": 1,
                        "tooltip": "The strength of the conditioning on the images.",
                    },
                ),
                "crop": (
                    ["center", "disabled"],
                    {
                        "default": "disabled",
                        "tooltip": "The crop mode to use for the images.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 35,
                        "min": 0,
                        "max": 100,
                        "tooltip": "The CRF value to use for preprocessing the images.",
                    },
                ),
                "blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "tooltip": "The blur value to use for preprocessing the images.",
                    },
                ),
                "optional_cond_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "This allows specifying the strength of each image in the conditional using an index based approach of comma separated values, note this will override strength"
                    },
                ),
                "optional_cond_use_latent_guide": (
                    "STRING",
                    {
                        "default": "f",
                        "tooltip": "Comma separated value of f and t, if f it will use LTXVAddGuide which allows for specific frame forceful guiding, if t it will use LTXVAddLatentGuide acting more like a negative index reference; false is usually the better"
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("denoised_output", "positive", "negative")
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        vae,
        width,
        height,
        num_frames,
        guider,
        sampler,
        sigmas,
        noise,
        optional_cond_images=None,
        optional_cond_indices=None,
        strength=0.9,
        crop="disabled",
        crf=35,
        blur=0,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        optional_initialization_latents=None,
        optional_cond_strength=None,
        optional_cond_use_latent_guide=None,
    ):
        if optional_cond_images is not None:
            assert optional_cond_indices is not None and bool(optional_cond_indices.strip()), "You must specify optional_cond_indices if optional_cond_images specified"
            assert (optional_cond_strength is not None and bool(optional_cond_strength.strip())) or strength is not None, "You must specify optional_cond_strength or strength if optional_cond_images specified"

            optional_cond_images = (
                comfy.utils.common_upscale(
                    optional_cond_images.movedim(-1, 1),
                    width,
                    height,
                    "bilinear",
                    crop=crop,
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )
            optional_cond_images = comfy_extras.nodes_lt.LTXVPreprocess().preprocess(
                optional_cond_images, crf
            )[0]
            for i in range(optional_cond_images.shape[0]):
                optional_cond_images[i] = blur_internal(
                    optional_cond_images[i].unsqueeze(0), blur
                )

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(
                optional_cond_images
            ), "Number of optional cond images must match number of optional cond indices"

        if optional_cond_strength is not None and bool(optional_cond_strength.strip()) and optional_cond_images is not None:
            optional_cond_strength = optional_cond_strength.split(",")
            optional_cond_strength = [float(i) for i in optional_cond_strength]
            assert len(optional_cond_strength ) == len(
                optional_cond_images
            ), "Number of optional cond images must match number of optional cond strength"
            # New assertion to check if all strengths are between 0 and 1
            assert all(0.0 <= s <= 1.0 for s in optional_cond_strength), "All optional cond strengths must be floats between 0 and 1"
        else:
            optional_cond_strength = None

        if optional_cond_use_latent_guide is not None and bool(optional_cond_use_latent_guide.strip()) and optional_cond_images is not None:
            optional_cond_use_latent_guide = optional_cond_use_latent_guide.split(",")
            assert len(optional_cond_use_latent_guide) == len(
                optional_cond_images
            ), "Number of optional cond use latent guide must match number of optional cond strength"
            # New assertion to check if all strengths are between 0 and 1
            assert all(s == "t" or s == "f" for s in optional_cond_use_latent_guide), "All optional cond use latent guide must be t or f"
        else:
            optional_cond_use_latent_guide = None

        try:
            positive, negative = guider.raw_conds
        except AttributeError:
            raise ValueError(
                "Guider does not have raw conds, cannot use it as a guider. "
                "Please use STGGuiderAdvanced."
            )

        if optional_initialization_latents is None:
            (latents,) = EmptyLTXVLatentVideo().generate(width, height, num_frames, 1)
        else:
            latents = optional_initialization_latents

        if (
            optional_cond_images is not None
            and optional_cond_images.shape[0] == 1
            and optional_cond_indices[0] == 0 and
            (
                optional_cond_use_latent_guide is None or
                optional_cond_use_latent_guide[0] == "f"
            )
        ):
            pixels = comfy.utils.common_upscale(
                optional_cond_images[0].unsqueeze(0).movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            encode_pixels = pixels[:, :, :, :3]
            t = vae.encode(encode_pixels)
            latents["samples"][:, :, : t.shape[2]] = t

            conditioning_latent_frames_mask = torch.ones(
                (1, 1, latents["samples"].shape[2], 1, 1),
                dtype=torch.float32,
                device=latents["samples"].device,
            )
            if optional_cond_strength is None:
                conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - strength
            else:
                conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - optional_cond_strength[0]
            latents["noise_mask"] = conditioning_latent_frames_mask

        elif optional_cond_images is not None:
            cond_strengths_list = optional_cond_strength if optional_cond_strength is not None else [strength] * len(optional_cond_images)
            cond_use_latent_guide_list = optional_cond_use_latent_guide if optional_cond_use_latent_guide is not None else ["f"] * len(optional_cond_images)

            for cond_image, cond_idx, cond_strength, cond_use_latent_guide in zip(
                optional_cond_images, optional_cond_indices, cond_strengths_list, cond_use_latent_guide_list
            ):
                if cond_use_latent_guide == "f":
                    (
                        positive,
                        negative,
                        latents,
                   ) = LTXVAddGuide().generate(
                        positive=positive,
                        negative=negative,
                        vae=vae,
                        latent=latents,
                        image=cond_image.unsqueeze(0),
                        frame_idx=cond_idx,
                        strength=cond_strength,
                    )
                else:
                    time_scale_factor, _, _ = (
                        vae.downscale_index_formula
                    )
                    latent_idx = int((cond_idx + 7) // time_scale_factor)
                    (cond_image_latent,) = VAEEncode().encode(vae, cond_image.unsqueeze(0))
                    (
                        positive,
                        negative,
                        latents,
                    ) = LTXVAddLatentGuide().generate(
                        vae=vae,
                        positive=positive,
                        negative=negative,
                        latent=latents,
                        guiding_latent=cond_image_latent,
                        latent_idx=latent_idx,
                        strength=cond_strength,
                    )

        if optional_negative_index_latents is not None:
            (
                positive,
                negative,
                latents,
            ) = LTXVAddLatentGuide().generate(
                vae=vae,
                positive=positive,
                negative=negative,
                latent=latents,
                guiding_latent=optional_negative_index_latents,
                latent_idx=optional_negative_index,
                strength=optional_negative_index_strength,
            )

        guider = copy.copy(guider)
        guider.set_conds(positive, negative)

        # Denoise the latent video
        (output_latents, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latents,
        )

        # Clean up guides if image conditioning was used
        positive, negative, denoised_output_latents = LTXVCropGuides().crop(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )

        return (denoised_output_latents, positive, negative)


@comfy_node(
    name="LTXVExtendSampler",
)
class LTXVExtendSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to use."}),
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
                "latents": (
                    "LATENT",
                    {"tooltip": "The latents of the video to extend."},
                ),
                "num_new_frames": (
                    "INT",
                    {
                        "default": 80,
                        "min": -1,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": "If -1, the number of frames will be based on the number of frames in the optional_guiding_latents.",
                    },
                ),
                "frame_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 16,
                        "max": 128,
                        "step": 8,
                        "tooltip": "The overlap region to use for conditioning the new frames on the end of the provided latents.",
                    },
                ),
                "guider": (
                    "GUIDER",
                    {"tooltip": "The guider to use, must be a STGGuiderAdvanced."},
                ),
                "sampler": ("SAMPLER", {"tooltip": "The sampler to use."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigmas to use."}),
                "noise": ("NOISE", {"tooltip": "The noise to use for the sampling."}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "The strength of the conditioning on the overlapping latents, when using optional_guiding_latents.",
                    },
                ),
            },
            "optional": {
                "optional_guiding_latents": (
                    "LATENT",
                    {"tooltip": "Optional latents to guide the sampling."},
                ),
                "optional_cond_images": (
                    "IMAGE",
                    {"tooltip": "The images to use for conditioning the sampling."},
                ),
                "optional_cond_indices": (
                    "STRING",
                    {
                        "tooltip": "The indices of the images to use for conditioning the sampling."
                    },
                ),
                "optional_cond_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "This allows specifying the strength of each image in the conditional using an index based approach of comma separated values"
                    },
                ),
                "crop": (
                    ["center", "disabled"],
                    {
                        "default": "disabled",
                        "tooltip": "The crop mode to use for the images.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 35,
                        "min": 0,
                        "max": 100,
                        "tooltip": "The CRF value to use for preprocessing the images.",
                    },
                ),
                "blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "tooltip": "The blur value to use for preprocessing the images.",
                    },
                ),
                "optional_cond_use_latent_guide": (
                    "STRING",
                    {
                        "default": "f",
                        "tooltip": "Comma separated value of f and t, if f it will use LTXVAddGuide which allows for specific frame forceful guiding, if t it will use LTXVAddLatentGuide acting more like a negative index reference; false is usually the better"
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("denoised_output", "positive", "negative")
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        vae,
        latents,
        num_new_frames,
        frame_overlap,
        guider,
        sampler,
        sigmas,
        noise,
        strength=0.5,
        guiding_strength=1.0,
        optional_guiding_latents=None,
        optional_reference_latents=None,
        optional_initialization_latents=None,
        adain_factor=0.0,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        crop="disabled",
        crf=35,
        blur=0,
        optional_cond_images=None,
        optional_cond_indices=None,
        optional_cond_strength=None,
        optional_cond_use_latent_guide=None,

        guiding_latents_already_cropped=False,
    ):
        try:
            positive, negative = guider.raw_conds
        except AttributeError:
            raise ValueError(
                "Guider does not have raw conds, cannot use it as a guider. "
                "Please use STGGuiderAdvanced."
            )

        samples = latents["samples"]
        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        width = width * width_scale_factor
        height = height * height_scale_factor

        overlap = frame_overlap // time_scale_factor

        if optional_cond_images is not None:
            assert optional_cond_indices is not None and bool(optional_cond_indices.strip()), "You must specify optional_cond_indices if optional_cond_images specified"
            assert optional_cond_strength is not None and bool(optional_cond_strength.strip()), "You must specify optional_cond_strength if optional_cond_images specified"

            optional_cond_images = (
                comfy.utils.common_upscale(
                    optional_cond_images.movedim(-1, 1),
                    width,
                    height,
                    "bilinear",
                    crop=crop,
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )
            optional_cond_images = comfy_extras.nodes_lt.LTXVPreprocess().preprocess(
                optional_cond_images, crf
            )[0]
            for i in range(optional_cond_images.shape[0]):
                optional_cond_images[i] = blur_internal(
                    optional_cond_images[i].unsqueeze(0), blur
                )

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(
                optional_cond_images
            ), "Number of optional cond images must match number of optional cond indices"

        if optional_cond_strength is not None and bool(optional_cond_strength.strip()) and optional_cond_images is not None:
            optional_cond_strength = optional_cond_strength.split(",")
            optional_cond_strength = [float(i) for i in optional_cond_strength]
            assert len(optional_cond_strength ) == len(
                optional_cond_images
            ), "Number of optional cond images must match number of optional cond strength"
            # New assertion to check if all strengths are between 0 and 1
            assert all(0.0 <= s <= 1.0 for s in optional_cond_strength), "All optional cond strengths must be floats between 0 and 1"
        else:
            optional_cond_strength = None

        if optional_cond_use_latent_guide is not None and bool(optional_cond_use_latent_guide.strip()) and optional_cond_images is not None:
            optional_cond_use_latent_guide = optional_cond_use_latent_guide.split(",")
            assert len(optional_cond_use_latent_guide) == len(
                optional_cond_images
            ), "Number of optional cond use latent guide must match number of optional cond strength"
            # New assertion to check if all strengths are between 0 and 1
            assert all(s == "t" or s == "f" for s in optional_cond_use_latent_guide), "All optional cond use latent guide must be t or f"
        else:
            optional_cond_use_latent_guide = None

        if num_new_frames == -1 and optional_guiding_latents is not None:
            num_new_frames = (
                optional_guiding_latents["samples"].shape[2]
            ) * time_scale_factor

        (last_overlap_latents,) = LTXVSelectLatents().select_latents(
            latents, -overlap, -1
        )

        if optional_initialization_latents is None:
            new_latents = EmptyLTXVLatentVideo().generate(
                width=width,
                height=height,
                length=(overlap * time_scale_factor) + num_new_frames,
                batch_size=1,
            )[0]
        else:
            new_latents = optional_initialization_latents

        last_overlap_latents["samples"] = last_overlap_latents["samples"].to(
            new_latents["samples"].device
        )

        (
            positive,
            negative,
            new_latents,
        ) = LTXVAddLatentGuide().generate(
            vae=vae,
            positive=positive,
            negative=negative,
            latent=new_latents,
            guiding_latent=last_overlap_latents,
            latent_idx=0,
            strength=strength,
        )

        if optional_cond_images is not None:
            cond_strengths_list = optional_cond_strength if optional_cond_strength is not None else [1] * len(optional_cond_images)
            cond_use_latent_guide_list = optional_cond_use_latent_guide if optional_cond_use_latent_guide is not None else ["f"] * len(optional_cond_images)
            for cond_image, cond_idx, cond_strength, cond_use_latent_guide in zip(
                optional_cond_images, optional_cond_indices, cond_strengths_list, cond_use_latent_guide_list
            ):
                if cond_use_latent_guide == "f":
                    (
                        positive,
                        negative,
                        new_latents,
                    ) = LTXVAddGuide().generate(
                        positive=positive,
                        negative=negative,
                        vae=vae,
                        latent=new_latents,
                        image=cond_image.unsqueeze(0),
                        frame_idx=cond_idx + frame_overlap,
                        strength=cond_strength,
                    )
                else:
                    time_scale_factor, _, _ = (
                        vae.downscale_index_formula
                    )
                    latent_idx = int((cond_idx + frame_overlap) // time_scale_factor)
                    (cond_image_latent,) = VAEEncode().encode(vae, cond_image.unsqueeze(0))
                    (
                        positive,
                        negative,
                        new_latents,
                    ) = LTXVAddLatentGuide().generate(
                        vae=vae,
                        positive=positive,
                        negative=negative,
                        latent=new_latents,
                        guiding_latent=cond_image_latent,
                        latent_idx=latent_idx,
                        strength=cond_strength,
                    )

        if optional_guiding_latents is not None:
            # this seems to be wrong, it's chopping my guiding latent start frames
            # and then setting them to the end, instead it should go as it is, ensuring that
            # any first guided action happens even if end actions get cropped (which means you must increase num_frames)
            # if num_frames was specified

            # has to stay because it would otherwise not make work the looping sampler because they send the movement information
            # as well regarding the overlapping frames and have to crop it
            if not guiding_latents_already_cropped:
                optional_guiding_latents = LTXVSelectLatents().select_latents(
                    optional_guiding_latents, overlap, -1
                )[0]

            #now the guiding latents start where it ended, even if the end is chopped off
            (
                positive,
                negative,
                new_latents,
            ) = LTXVAddLatentGuide().generate(
                vae=vae,
                positive=positive,
                negative=negative,
                latent=new_latents,
                guiding_latent=optional_guiding_latents,
                latent_idx=last_overlap_latents["samples"].shape[2],
                strength=guiding_strength,
            )
        if optional_negative_index_latents is not None:
            (
                positive,
                negative,
                new_latents,
            ) = LTXVAddLatentGuide().generate(
                vae=vae,
                positive=positive,
                negative=negative,
                latent=new_latents,
                guiding_latent=optional_negative_index_latents,
                latent_idx=optional_negative_index,
                strength=optional_negative_index_strength,
            )

        guider = copy.copy(guider)
        guider.set_conds(positive, negative)

        # Denoise the latent video
        (output_latents, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=new_latents,
        )

        # Clean up guides if image conditioning was used
        positive, negative, denoised_output_latents = LTXVCropGuides().crop(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )

        # drop first output latent as it's a reinterpreted 8-frame latent understood as a 1-frame latent
        truncated_denoised_output_latents = LTXVSelectLatents().select_latents(
            denoised_output_latents, 1, -1
        )[0]

        if optional_reference_latents is not None:
            truncated_denoised_output_latents = LTXVAdainLatent().batch_normalize(
                latents=truncated_denoised_output_latents,
                reference=optional_reference_latents,
                factor=adain_factor,
            )[0]

        # Fuse new frames with old ones by calling LinearOverlapLatentTransition
        (latents,) = LinearOverlapLatentTransition().process(
            latents, truncated_denoised_output_latents, overlap - 1, axis=2
        )
        return (latents, positive, negative)


@comfy_node(
    name="LTXVInContextSampler",
)
class LTXVInContextSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
                "guider": (
                    "GUIDER",
                    {"tooltip": "The guider to use, must be a STGGuiderAdvanced."},
                ),
                "sampler": ("SAMPLER", {"tooltip": "The sampler to use."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigmas to use."}),
                "noise": ("NOISE", {"tooltip": "The noise to use for the sampling."}),
                "guiding_latents": (
                    "LATENT",
                    {
                        "tooltip": "The latents to use for guiding the sampling, typically with an IC-LoRA."
                    },
                ),
            },
            "optional": {
                "optional_cond_image": (
                    "IMAGE",
                    {
                        "tooltip": "The images to use for conditioning the sampling, if not provided, the sampling will be unconditioned (t2v setup). The image will be resized to the size of the first frame."
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "If -1, the number of frames will be based on the number of frames in the guiding_latents.",
                    },
                ),
                "optional_cond_image_indices": (
                    "STRING",
                    {
                        "tooltip": "A comma separated value for the image indices for use in the video generation to add a latent guide"
                    },
                ),
                "optional_cond_image_strength": (
                    "STRING",
                    {
                        "tooltip": "A comma separated value for the image strengths for use in the video generation to add a latent guide"
                    },
                ),
                "crop": (
                    ["center", "disabled"],
                    {
                        "default": "disabled",
                        "tooltip": "The crop mode to use for the images.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 35,
                        "min": 0,
                        "max": 100,
                        "tooltip": "The CRF value to use for preprocessing the images.",
                    },
                ),
                "blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "tooltip": "The blur value to use for preprocessing the images.",
                    },
                ),
                "optional_guiding_strength": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 1,
                        "tooltip": "A the strength to use for the latent guiding"
                    },
                ),
                "optional_cond_use_latent_guide": (
                    "STRING",
                    {
                        "default": "t",
                        "tooltip": "Comma separated value of f and t, if f it will use LTXVAddGuide which allows for specific frame forceful guiding, if t it will use LTXVAddLatentGuide acting more like a negative index reference; false is more accurate in the case of this sampler true was the default as it creates a weak reference"
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("denoised_output", "positive", "negative")
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        vae,
        guider,
        sampler,
        sigmas,
        noise,
        guiding_latents,
        optional_cond_image=None,
        num_frames=0,
        optional_initialization_latents=None,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        optional_cond_strength=1.0,
        optional_guiding_strength=1.0,
        optional_cond_image_strength=None,
        optional_cond_image_indices="0",
        optional_cond_use_latent_guide=None,
        crop="disabled",
        crf=35,
        blur=0,
    ):
        try:
            positive, negative = guider.raw_conds
        except AttributeError:
            raise ValueError(
                "Guider does not have raw conds, cannot use it as a guider. "
                "Please use STGGuiderAdvanced."
            )

        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )

        batch, channels, frames, height, width = guiding_latents["samples"].shape
        if num_frames != -1:
            frames = (num_frames - 1) // time_scale_factor + 1

        if optional_initialization_latents is not None:
            new_latents = optional_initialization_latents
        else:
            new_latents = EmptyLTXVLatentVideo().generate(
                width=width * width_scale_factor,
                height=height * height_scale_factor,
                length=(frames - 1) * time_scale_factor + 1,
                batch_size=1,
            )[0]

        if optional_cond_image is not None:
            assert (optional_cond_image_strength is not None and bool(optional_cond_image_strength.strip())) or optional_cond_strength is not None, "You must specify optional_cond_image_strength or optional_cond_strength if optional_cond_image specified"

            optional_cond_image = (
                comfy.utils.common_upscale(
                    optional_cond_image.movedim(-1, 1),
                    width * width_scale_factor,
                    height * height_scale_factor,
                    "bilinear",
                    crop=crop,
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )
            optional_cond_image = comfy_extras.nodes_lt.LTXVPreprocess().preprocess(
                optional_cond_image, crf
            )[0]
            for i in range(optional_cond_image.shape[0]):
                optional_cond_image[i] = blur_internal(
                    optional_cond_image[i].unsqueeze(0), blur
                )

        if optional_cond_image_indices is None:
            optional_cond_image_indices = "0"

        if optional_cond_image is not None:
            optional_cond_image_indices = optional_cond_image_indices.split(",")
            optional_cond_image_indices = [int(i) for i in optional_cond_image_indices]
            assert len(optional_cond_image_indices) == len(
                optional_cond_image
            ), "Number of optional cond images must match number of optional cond indices"

        if optional_cond_image_strength is not None and bool(optional_cond_image_strength.strip()) and optional_cond_image is not None:
            optional_cond_image_strength = optional_cond_image_strength.split(",")
            optional_cond_image_strength = [float(i) for i in optional_cond_image_strength]
            assert len(optional_cond_image_strength) == len(
                optional_cond_image
            ), "Number of optional cond images must match number of optional cond strength"
            # New assertion to check if all strengths are between 0 and 1
            assert all(0.0 <= s <= 1.0 for s in optional_cond_image_strength), "All optional cond image strengths must be floats between 0 and 1"
        else:
            optional_cond_strength = None

        if optional_cond_use_latent_guide is not None and bool(optional_cond_use_latent_guide.strip()) and optional_cond_image is not None:
            optional_cond_use_latent_guide = optional_cond_use_latent_guide.split(",")
            assert len(optional_cond_use_latent_guide) == len(
                optional_cond_image
            ), "Number of optional cond use latent guide must match number of optional cond strength"
            # New assertion to check if all strengths are between 0 and 1
            assert all(s == "t" or s == "f" for s in optional_cond_use_latent_guide), "All optional cond use latent guide must be t or f"
        else:
            optional_cond_use_latent_guide = None

        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )

        found_zero_index = False
        zero_index_matters = False
        zero_index_has_power_1 = False
        zero_index_has_t = False

        if optional_cond_image is not None:

            zero_index_matters = True

            cond_strengths_list = optional_cond_image_strength if optional_cond_image_strength is not None else [optional_cond_strength] * len(optional_cond_image)
            cond_use_latent_guide_list = optional_cond_use_latent_guide if optional_cond_use_latent_guide is not None else ["f"] * len(optional_cond_image)
            if (optional_cond_use_latent_guide is None):
                cond_use_latent_guide_list[0] = "t"

            for cond_image, cond_idx, cond_strength, cond_use_latent_guide in zip(
                optional_cond_image, optional_cond_image_indices, cond_strengths_list, cond_use_latent_guide_list
            ):
                if cond_idx == 0:
                    found_zero_index = True

                if cond_strength == 1 and cond_idx == 0:
                    zero_index_has_power_1 = True

                if cond_use_latent_guide == "t" and cond_idx == 0:
                    zero_index_has_t = True

                if (cond_use_latent_guide == "f"):
                    (
                        positive,
                        negative,
                        new_latents,
                    ) = LTXVAddGuide().generate(
                        positive=positive,
                        negative=negative,
                        vae=vae,
                        latent=new_latents,
                        image=cond_image.unsqueeze(0),
                        frame_idx=cond_idx,
                        strength=cond_strength,
                    )
                else:
                    latent_idx = int(cond_idx // time_scale_factor)
                    (cond_image_latent,) = VAEEncode().encode(vae, cond_image.unsqueeze(0))
                    (
                        positive,
                        negative,
                        new_latents,
                    ) = LTXVAddLatentGuide().generate(
                        vae=vae,
                        positive=positive,
                        negative=negative,
                        latent=new_latents,
                        guiding_latent=cond_image_latent,
                        latent_idx=latent_idx,
                        strength=cond_strength,
                    )

        if zero_index_matters:
            if not found_zero_index:
                print("### The initial image was not found this will result in arbitrary video")
            elif not zero_index_has_power_1:
                print("### The initial image has a strength lower than 1, this will result in video with less adherence to the initial image")
            elif not zero_index_has_t:
                print("### The initial image was used as a image guide rather than a latent guide, the default LTXV Behaviour is using it as a latent guide")

        if optional_cond_image is not None:
            guiding_latents = LTXVSelectLatents().select_latents(
                guiding_latents, 1, -1
            )[0]

        (
            positive,
            negative,
            new_latents,
        ) = LTXVAddLatentGuide().generate(
            vae=vae,
            positive=positive,
            negative=negative,
            latent=new_latents,
            guiding_latent=guiding_latents,
            latent_idx=1 if optional_cond_image is not None else 0,
            strength=optional_guiding_strength,
        )
        if optional_negative_index_latents is not None:
            (
                positive,
                negative,
                new_latents,
            ) = LTXVAddLatentGuide().generate(
                vae=vae,
                positive=positive,
                negative=negative,
                latent=new_latents,
                guiding_latent=optional_negative_index_latents,
                latent_idx=optional_negative_index,
                strength=optional_negative_index_strength,
            )

        guider = copy.copy(guider)
        guider.set_conds(positive, negative)

        # Denoise the latent video
        (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=new_latents,
        )

        # Clean up guides if image conditioning was used
        positive, negative, denoised_output_latents = LTXVCropGuides().crop(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )

        return (denoised_output_latents, positive, negative)
    
def get_negative_indices(positive_indices, num_elements):
    """
    Converts a list of positive indices to their corresponding negative indices.

    Args:
        positive_indices: A list of 0-based positive indices.
        num_elements: The total number of elements in the list.

    Returns:
        A list of negative indices.
    """
    negative_indices = []
    for index in positive_indices:
        # Check if the index is within the valid range.
        if 0 <= index < num_elements:
            # Convert to negative index.
            negative_index = index - num_elements
            negative_indices.append(negative_index)
        else:
            # Handle invalid index.
            print(f"Warning: Index {index} is out of bounds for a list of size {num_elements}. Skipping.")
    return negative_indices

@comfy_node(
    name="LTXVHybridSampler",
)
class LTXVHybridSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to use."}),
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
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
                "num_frames": (
                    "INT",
                    {"default": 97, "min": 1, "max": nodes.MAX_RESOLUTION, "tooltip": "The number of frames to generate, if initial_video is specified the number of frames " +
                     "to extend with; if initial_video is specified frames should be divisible by 8 if not (frames - 1) should be divisible by 8"},
                ),
                "guider": (
                    "GUIDER",
                    {"tooltip": "The guider to use, must be a STGGuiderAdvanced."},
                ),
                "sampler": ("SAMPLER", {"tooltip": "The sampler to use."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigmas to use."}),
                "noise": ("NOISE", {"tooltip": "The noise to use for the sampling."}),
            },
            "optional": {
                "optional_cond_images": (
                    "IMAGE",
                    {"tooltip": "The images to use for conditioning the sampling."},
                ),
                "optional_cond_indices": (
                    "STRING",
                    {
                        "tooltip": "The indices of the images to use for conditioning the sampling."
                    },
                ),
                "optional_cond_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "This allows specifying the strength of each image in the conditional using an index based approach of comma separated values"
                    },
                ),
                "crop": (
                    ["center", "disabled"],
                    {
                        "default": "center",
                        "tooltip": "The crop mode to use for the images.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "The CRF value to use for preprocessing the images.",
                    },
                ),
                "blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "tooltip": "The blur value to use for preprocessing the images.",
                    },
                ),
                "frame_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 16,
                        "max": 128,
                        "step": 8,
                        "tooltip": "The overlap region to use for conditioning the new frames on the end of the provided initial_video latents",
                    },
                ),
                "initial_video": (
                    "LATENT",
                    {"tooltip": "The video to extend, use the Load Video and then VAE Encode it, otherwise load plain latents that you have saved, it is recommended to save the latents because the process of VAE Encode and VAE Decode and export as video causes losses"},
                ),
                "initial_video_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0,
                        "max": 1,
                        "tooltip": "The strength of the conditioning on the initial video",
                    },
                ),
                "optional_cond_use_latent_guide": (
                    "STRING",
                    {
                        "default": "f",
                        "tooltip": "Comma separated value of f and t, if f it will use LTXVAddGuide which allows for specific frame forceful guiding, if t it will use LTXVAddLatentGuide acting more like a negative index reference; false is usually the better when no guiding_video used, otherwise true is good on a first frame when using a guiding video reference, eg. 'f,f,f' for base generation with only an image, 't,f,f' for base generation with a guiding video and 'f,f,f' again when initial_video specified since you should not use frame zero when extending"
                    },
                ),
                "guiding_video": (
                    "LATENT",
                    {
                        "tooltip": "The guiding video to use to guide the generation, use with the guiding mechanism, remember to use either depth, pose or canny in the workflow",
                    },
                ),
                "guiding_video_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0,
                        "max": 1,
                        "tooltip": "The strength of the conditioning on the guiding video",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "LATENT",
        "CONDITIONING",
        "CONDITIONING",
        "STRING",
        "STRING",
        "LATENT",
        "STRING",
    )
    RETURN_NAMES = (
        "denoised_output",
        "positive",
        "negative",
        "generated_frames_range",
        "reference_frames_only_generated_idx",
        "denoised_output_only_generated",
        "chunk_info",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        vae,
        width,
        height,
        num_frames,
        guider,
        sampler,
        sigmas,
        noise,
        initial_video=None,
        initial_video_strength=0.5,
        frame_overlap=16,
        optional_cond_images=None,
        optional_cond_indices=None,
        optional_cond_strength=None,
        optional_cond_use_latent_guide=None,
        crop="disabled",
        crf=35,
        blur=0,
        guiding_video=None,
        guiding_video_strength=1.0,
    ):
        if initial_video is None:
            assert (num_frames - 1) % 8 == 0, "The number of frames minus 1 should be divisible by 8 when no initial video is specified"
        else:
            assert num_frames % 8 == 0, "The number of frames should be divisible by 8 when an initial video is specified"

        if initial_video is None and guiding_video is None:
            generated_frames_range = ":" + str(-num_frames)
            reference_frames_only_generated_idx = "" if optional_cond_indices is None else optional_cond_indices
            latents, positive, negative = LTXVBaseSampler().sample(
                model,
                vae,
                width,
                height,
                num_frames,
                guider,
                sampler,
                sigmas,
                noise,
                optional_cond_images=optional_cond_images,
                optional_cond_indices=optional_cond_indices,
                optional_cond_strength=optional_cond_strength,
                optional_cond_use_latent_guide=optional_cond_use_latent_guide,
                strength=None,
                crop=crop,
                crf=crf,
                blur=blur,
            )

            info_dict = {
                "cond_indices": None if optional_cond_indices is None else optional_cond_indices,
                "cond_strengths": None if optional_cond_strength is None else optional_cond_strength,
                "cond_use_latent_guide": None if optional_cond_use_latent_guide is None else optional_cond_use_latent_guide,
                "num_frames_requested": num_frames,
                "frames_generated": num_frames,
                "images_used": 0 if optional_cond_images is None else len(optional_cond_images),
                "crf": crf,
                "blur": blur,
                "crop": crop,
                "frame_overlap": frame_overlap,
                "extra_latents_only_generated_provided": 0,
                "extra_frames_only_generated_provided": 0,
            }
            return (latents, positive, negative, generated_frames_range, reference_frames_only_generated_idx, latents, json.dumps(info_dict))

        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )

        gv_px_height = None
        gv_px_width = None
        gv_frames = None

        if guiding_video is not None:
            g_samples = guiding_video["samples"]
            _, _, g_frames, gv_height, gv_width = g_samples.shape
            
            gv_px_height = gv_height * height_scale_factor
            gv_px_width = gv_width * width_scale_factor
            gv_frames = (g_frames * time_scale_factor) - 7

            assert gv_px_width == width, "The width of the guiding video and the width of the settings do not match, provided " + str(width) + " but the guiding video is " + str(gv_px_width)
            assert gv_px_height == height, "The height of the guiding video and the height of the settings do not match, provided " + str(height) + " but the guiding video is " + str(gv_px_height)
            assert gv_frames == num_frames, "The number of frames of the guiding video and the number of frames of the settings differs  " + str(num_frames) + " but the guiding video is " + str(gv_frames) + " set your frames to " + str(gv_frames)

            if initial_video is None:
                generated_frames_range = ":" + str(-num_frames)
                reference_frames_only_generated_idx = "" if optional_cond_indices is None else optional_cond_indices

                latents, positive, negative = LTXVInContextSampler().sample(
                    vae,
                    guider,
                    sampler,
                    sigmas,
                    noise,
                    guiding_video,
                    optional_cond_image=optional_cond_images,
                    num_frames=-1,
                    optional_cond_image_indices=optional_cond_indices,
                    optional_cond_image_strength=optional_cond_strength,
                    optional_cond_use_latent_guide=optional_cond_use_latent_guide,
                    optional_guiding_strength=guiding_video_strength,
                    crop=crop,
                    crf=crf,
                    blur=blur,
                )

                info_dict = {
                    "cond_indices": None if optional_cond_indices is None else optional_cond_indices,
                    "cond_strengths": None if optional_cond_strength is None else optional_cond_strength,
                    "cond_use_latent_guide": None if optional_cond_use_latent_guide is None else optional_cond_use_latent_guide,
                    "num_frames_requested": num_frames,
                    "frames_generated": num_frames,
                    "images_used": 0 if optional_cond_images is None else len(optional_cond_images),
                    "crf": crf,
                    "blur": blur,
                    "crop": crop,
                    "frame_overlap": frame_overlap,
                    "extra_latents_only_generated_provided": 0,
                    "extra_frames_only_generated_provided": 0,
                }

                return (latents, positive, negative, generated_frames_range, reference_frames_only_generated_idx, latents, json.dumps(info_dict))

        samples = initial_video["samples"]
        batch, channels, frames, v_height, v_width = samples.shape
        v_px_height = v_height * height_scale_factor
        v_px_width = v_width * width_scale_factor

        assert v_px_width == width, "The width of the provided video and the width of the settings do not match, provided " + str(width) + " but the video is " + str(v_px_width)
        assert v_px_height == height, "The height of the provided video and the height of the settings do not match, provided " + str(height) + " but the video is " + str(v_px_height)

        latents, positive, negative = LTXVExtendSampler().sample(
            model,
            vae,
            initial_video,
            num_frames,
            frame_overlap,
            guider,
            sampler,
            sigmas,
            noise,
            strength=initial_video_strength,
            guiding_strength=guiding_video_strength,
            crop=crop,
            crf=crf,
            blur=blur,
            optional_guiding_latents=guiding_video,
            optional_cond_images=optional_cond_images,
            optional_cond_indices=optional_cond_indices,
            optional_cond_strength=optional_cond_strength,
            optional_cond_use_latent_guide=optional_cond_use_latent_guide,
            guiding_latents_already_cropped=True,
        )

        generated_frames_range = ":" + str(-num_frames)

        # and then we want to shift our conditional indices that way, first since the diff frames are removed at the start, and well
        # it is a negative number we add that, and then add the whole video frames that shift the whole thing
        reference_frames_only_generated_idx_parsed = [] if optional_cond_indices is None else [(float(i) - 7) for i in optional_cond_indices.split(",")]

        for i in range(0, len(reference_frames_only_generated_idx_parsed)):
            if (reference_frames_only_generated_idx_parsed[i] < 0):
                reference_frames_only_generated_idx_parsed[i] = 0

        (reference_createdonly_latents,) = LTXVSelectLatents().select_latents(
            latents, int(-(num_frames // 8)) - 1, -1
        )

        info_dict = {
            "cond_indices": None if optional_cond_indices is None else optional_cond_indices,
            "cond_strengths": None if optional_cond_strength is None else optional_cond_strength,
            "cond_use_latent_guide": None if optional_cond_use_latent_guide is None else optional_cond_use_latent_guide,
            "num_frames_requested": num_frames,
            "frames_generated": num_frames,
            "images_used": 0 if optional_cond_images is None else len(optional_cond_images),
            "crf": crf,
            "blur": blur,
            "crop": crop,
            "frame_overlap": frame_overlap,
            "extra_latents_only_generated_provided": 1,
            "extra_frames_only_generated_provided": 1,
        }

        return (latents, positive, negative, generated_frames_range, ",".join([str(i) for i in reference_frames_only_generated_idx_parsed]), reference_createdonly_latents, json.dumps(info_dict))


@comfy_node(description="Linear transition with overlap")
class LinearOverlapLatentTransition:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples1": ("LATENT",),
                "samples2": ("LATENT",),
                "overlap": ("INT", {"default": 1, "min": 1, "max": 256}),
            },
            "optional": {
                "axis": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"

    CATEGORY = "Lightricks/latent"

    def get_subbatch(self, samples):
        s = samples.copy()
        samples = s["samples"]
        return samples

    def process(self, samples1, samples2, overlap, axis=0):
        samples1 = self.get_subbatch(samples1)
        samples2 = self.get_subbatch(samples2)

        # Create transition coefficients
        alpha = torch.linspace(1, 0, overlap + 2)[1:-1].to(samples1.device)

        # Create shape for broadcasting based on the axis
        shape = [1] * samples1.dim()
        shape[axis] = alpha.size(0)
        alpha = alpha.reshape(shape)

        # Create slices for the overlap regions
        slice_all = [slice(None)] * samples1.dim()
        slice_overlap1 = slice_all.copy()
        slice_overlap1[axis] = slice(-overlap, None)
        slice_overlap2 = slice_all.copy()
        slice_overlap2[axis] = slice(0, overlap)
        slice_rest1 = slice_all.copy()
        slice_rest1[axis] = slice(None, -overlap)
        slice_rest2 = slice_all.copy()
        slice_rest2[axis] = slice(overlap, None)

        # Combine samples
        parts = [
            samples1[tuple(slice_rest1)],
            alpha * samples1[tuple(slice_overlap1)]
            + (1 - alpha) * samples2[tuple(slice_overlap2)],
            samples2[tuple(slice_rest2)],
        ]

        combined_samples = torch.cat(parts, dim=axis)
        combined_batch_index = torch.arange(0, combined_samples.shape[0])

        return (
            {
                "samples": combined_samples,
                "batch_index": combined_batch_index,
            },
        )
