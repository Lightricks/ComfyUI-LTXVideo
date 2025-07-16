import copy

import comfy
import comfy_extras
import nodes
import torch
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVAddGuide, LTXVCropGuides
from nodes import VAEEncode

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

        if optional_initialization_latents is None:
            (latents,) = EmptyLTXVLatentVideo().generate(width, height, num_frames, 1)
        else:
            latents = optional_initialization_latents

        if (
            optional_cond_images is not None
            and optional_cond_images.shape[0] == 1
            and optional_cond_indices[0] == 0
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
            conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - strength
            latents["noise_mask"] = conditioning_latent_frames_mask

        elif optional_cond_images is not None:
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

        if num_new_frames == -1 and optional_guiding_latents is not None:
            num_new_frames = (
                optional_guiding_latents["samples"].shape[2] - overlap
            ) * time_scale_factor

        (last_overlap_latents,) = LTXVSelectLatents().select_latents(
            latents, -overlap, -1
        )

        if optional_initialization_latents is None:
            new_latents = EmptyLTXVLatentVideo().generate(
                width=width * width_scale_factor,
                height=height * height_scale_factor,
                length=overlap * time_scale_factor + num_new_frames,
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

        if optional_guiding_latents is not None:
            optional_guiding_latents = LTXVSelectLatents().select_latents(
                optional_guiding_latents, overlap, -1
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
                        "tooltip": "The image to use for conditioning the sampling, if not provided, the sampling will be unconditioned (t2v setup). The image will be resized to the size of the first frame."
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
            optional_cond_image = (
                comfy.utils.common_upscale(
                    optional_cond_image.movedim(-1, 1),
                    width * width_scale_factor,
                    height * height_scale_factor,
                    "bilinear",
                    crop="disabled",
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )
            (cond_image_latent,) = VAEEncode().encode(vae, optional_cond_image)
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
                latent_idx=0,
                strength=optional_cond_strength,
            )

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
