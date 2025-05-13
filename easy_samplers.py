import copy

import comfy
import comfy_extras
import nodes
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVAddGuide,
    LTXVCropGuides,
    LTXVImgToVideo,
)

from .guide import blur_internal
from .latents import LTXVAddLatentGuide, LTXVSelectLatents
from .nodes_registry import comfy_node
from .recurrent_sampler import LinearOverlapLatentTransition


@comfy_node(
    name="LTXVBaseSampler",
)
class LTXVBaseSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
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
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "noise": ("NOISE",),
            },
            "optional": {
                "optional_cond_images": ("IMAGE",),
                "optional_cond_indices": ("STRING",),
                "strength": ("FLOAT", {"default": 0.9, "min": 0, "max": 1}),
                "crop": (["center", "disabled"], {"default": "disabled"}),
                "crf": ("INT", {"default": 35, "min": 0, "max": 100}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 10}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("denoised_output",)
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
    ):

        if optional_cond_images is not None:
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
            print("optional_cond_images shape", optional_cond_images.shape)
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

        try:
            positive, negative = guider.raw_conds
        except AttributeError:
            raise ValueError(
                "Guider does not have raw conds, cannot use it as a guider. "
                "Please use STGGuiderAdvanced."
            )

        if optional_cond_images is None:
            (latents,) = EmptyLTXVLatentVideo().generate(width, height, num_frames, 1)
        elif optional_cond_images.shape[0] == 1 and optional_cond_indices[0] == 0:
            (
                positive,
                negative,
                latents,
            ) = LTXVImgToVideo().generate(
                positive=positive,
                negative=negative,
                vae=vae,
                image=optional_cond_images[0].unsqueeze(0),
                width=width,
                height=height,
                length=num_frames,
                batch_size=1,
                strength=strength,
            )
        else:
            (latents,) = EmptyLTXVLatentVideo().generate(width, height, num_frames, 1)
            for cond_image, cond_idx in zip(
                optional_cond_images, optional_cond_indices
            ):
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
                    strength=strength,
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
        print("before guide crop", denoised_output_latents["samples"].shape)
        positive, negative, denoised_output_latents = LTXVCropGuides().crop(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )
        print("after guide crop", denoised_output_latents["samples"].shape)

        return (denoised_output_latents,)


@comfy_node(
    name="LTXVExtendSampler",
)
class LTXVExtendSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "num_new_frames": (
                    "INT",
                    {"default": 80, "min": 8, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "frame_overlap": (
                    "INT",
                    {"default": 16, "min": 16, "max": 128, "step": 8},
                ),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "noise": ("NOISE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("denoised_output",)
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
        overlap = frame_overlap // time_scale_factor

        (last_overlap_latents,) = LTXVSelectLatents().select_latents(
            latents, -overlap, -1
        )

        new_latents = EmptyLTXVLatentVideo().generate(
            width=width * width_scale_factor,
            height=height * height_scale_factor,
            length=overlap * time_scale_factor + num_new_frames,
            batch_size=1,
        )[0]
        print("new_latents shape: ", new_latents["samples"].shape)
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
            strength=1.0,
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
        print("before guide crop", denoised_output_latents["samples"].shape)
        positive, negative, denoised_output_latents = LTXVCropGuides().crop(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )
        print("after guide crop", denoised_output_latents["samples"].shape)

        # drop first output latent as it's a reinterpreted 8-frame latent understood as a 1-frame latent
        truncated_denoised_output_latents = LTXVSelectLatents().select_latents(
            denoised_output_latents, 1, -1
        )[0]
        # Fuse new frames with old ones by calling LinearOverlapLatentTransition
        (latents,) = LinearOverlapLatentTransition().process(
            latents, truncated_denoised_output_latents, overlap - 1, axis=2
        )
        print("latents shape after linear overlap blend: ", latents["samples"].shape)
        return (latents,)
