import copy

import comfy
import comfy_extras
import nodes
import torch
from comfy.ldm.lightricks.av_model import LTXAVModel
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SplitSigmas
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVAddGuide, LTXVCropGuides

from .guide import blur_internal
from .latent_norm import LTXVAdainLatent
from .latents import LTXVAddLatentGuide, LTXVSelectLatents
from .nodes_registry import comfy_node


def _get_raw_conds_from_guider(guider):
    if not hasattr(guider, "raw_conds"):
        if "negative" not in guider.original_conds:
            # for BasicGuider
            raise ValueError(
                "Guider does not have negative conds, cannot use it as a guider."
            )
        raw_pos = guider.original_conds["positive"]
        positive = [[raw_pos[0]["cross_attn"], copy.deepcopy(raw_pos[0])]]
        raw_neg = guider.original_conds["negative"]
        negative = [[raw_neg[0]["cross_attn"], copy.deepcopy(raw_neg[0])]]
        guider.raw_conds = (positive, negative)
    return guider.raw_conds


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
    RETURN_NAMES = ("denoised", "positive", "negative")
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
        guiding_start_step=0,
        guiding_end_step=1000,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)
        positive, negative = _get_raw_conds_from_guider(guider)

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
            optional_cond_images = comfy_extras.nodes_lt.LTXVPreprocess.execute(
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

        if optional_initialization_latents is None:
            (latents,) = EmptyLTXVLatentVideo().execute(width, height, num_frames, 1)
        else:
            latents = optional_initialization_latents

        if optional_cond_images is not None and 0 in optional_cond_indices:
            # apply classical i2v conditioning on the first frame
            idx_0 = optional_cond_indices.index(0)
            encode_pixels = optional_cond_images[idx_0 : idx_0 + 1, :, :, :3]
            t = vae.encode(encode_pixels)
            latents["samples"][:, :, : t.shape[2]] = t

            if "noise_mask" not in latents:
                conditioning_latent_frames_mask = torch.ones(
                    (1, 1, latents["samples"].shape[2], 1, 1),
                    dtype=torch.float32,
                    device=latents["samples"].device,
                )
                conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - strength
                latents["noise_mask"] = conditioning_latent_frames_mask
            else:
                latents["noise_mask"][:, :, : t.shape[2]] = 1.0 - strength
                conditioning_latent_frames_mask = latents["noise_mask"]
        else:
            conditioning_latent_frames_mask = None

        high_sigmas, rest_sigmas = SplitSigmas().get_sigmas(sigmas, guiding_start_step)
        middle_sigmas, low_sigmas = SplitSigmas().get_sigmas(
            rest_sigmas, guiding_end_step - guiding_start_step
        )

        if len(high_sigmas) > 1:
            print("Denoising with no conditioning on sigmas: ", high_sigmas)
            (_, new_latents) = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=high_sigmas,
                latent_image=latents,
            )

        if optional_cond_images is not None:
            # add conditioning on keyframes with index > 0
            for cond_image, cond_idx in zip(
                optional_cond_images, optional_cond_indices
            ):
                if cond_idx == 0:
                    # 0 is handled by classical i2v conditioning
                    continue
                (
                    positive,
                    negative,
                    latents,
                ) = LTXVAddGuide.execute(
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

        guider.set_conds(positive, negative)

        # Denoise the latent video
        print("Denoising with conditioning on sigmas: ", middle_sigmas)
        (output_latents, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=middle_sigmas,
            latent_image=latents,
        )

        # Clean up guides if image conditioning was used
        positive, negative, denoised_output_latents = LTXVCropGuides.execute(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )

        denoised_output_latents["noise_mask"] = conditioning_latent_frames_mask

        if len(low_sigmas) > 1:
            print(
                "Denoising with no conditioning but with classical i2v noise mask on sigmas: ",
                low_sigmas,
            )
            (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=low_sigmas,
                latent_image=denoised_output_latents,
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
                "optional_cond_indices": ("STRING",),
                "cond_image_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "The strength of the conditioning on the images.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("denoised_video", "positive", "negative")
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
        cond_image_strength=1.0,
        optional_guiding_latents=None,
        optional_cond_images=None,
        optional_cond_indices=None,
        optional_reference_latents=None,
        optional_initialization_latents=None,
        adain_factor=0.0,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        guiding_start_step=0,
        guiding_end_step=1000,
        normalize_per_frame=False,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(
                optional_cond_images
            ), "Number of optional cond images must match number of optional cond indices"

        positive, negative = _get_raw_conds_from_guider(guider)

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
            new_latents = EmptyLTXVLatentVideo.execute(
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

        high_sigmas, rest_sigmas = SplitSigmas().get_sigmas(sigmas, guiding_start_step)
        middle_sigmas, low_sigmas = SplitSigmas().get_sigmas(
            rest_sigmas, guiding_end_step - guiding_start_step
        )

        if optional_cond_images is not None:
            print("Adding conditioning on keyframes")
            for cond_image, cond_idx in zip(
                optional_cond_images, optional_cond_indices
            ):
                if optional_guiding_latents is not None and cond_idx % 8 == 1:
                    raise ValueError(
                        f"Conditioning image index {cond_idx} (relative to the in the temporal chunk) is a multiple of 8 + 1,"
                        "and guiding latents are used. Please provide other conditioning image indices"
                    )
                (
                    positive,
                    negative,
                    new_latents,
                ) = LTXVAddGuide.execute(
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    latent=new_latents,
                    image=cond_image.unsqueeze(0),
                    frame_idx=cond_idx,
                    strength=cond_image_strength,
                )

        if len(high_sigmas) > 1:
            guider.set_conds(positive, negative)
            print("Denoising with overlap conditioning only on sigmas: ", high_sigmas)
            (_, new_latents) = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=high_sigmas,
                latent_image=new_latents,
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

        guider.set_conds(positive, negative)

        # Denoise the latent video
        print("Denoising with full conditioning on sigmas: ", middle_sigmas)
        (output_latents, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=middle_sigmas,
            latent_image=new_latents,
        )

        positive, negative, denoised_output_latents = LTXVCropGuides.execute(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )

        if len(low_sigmas) > 1:
            (
                positive,
                negative,
                denoised_output_latents,
            ) = LTXVAddLatentGuide().generate(
                vae=vae,
                positive=positive,
                negative=negative,
                latent=denoised_output_latents,
                guiding_latent=last_overlap_latents,
                latent_idx=0,
                strength=strength,
            )

            if optional_cond_images is not None:
                print("Adding conditioning on keyframes")
                for cond_image, cond_idx in zip(
                    optional_cond_images, optional_cond_indices
                ):
                    if optional_guiding_latents is not None and cond_idx % 8 == 1:
                        raise ValueError(
                            f"Conditioning image index {cond_idx} (relative to the in the temporal chunk) is a multiple of 8 + 1,"
                            "and guiding latents are used. Please provide other conditioning image indices"
                        )
                    (
                        positive,
                        negative,
                        denoised_output_latents,
                    ) = LTXVAddGuide.execute(
                        positive=positive,
                        negative=negative,
                        vae=vae,
                        latent=denoised_output_latents,
                        image=cond_image.unsqueeze(0),
                        frame_idx=cond_idx,
                        strength=cond_image_strength,
                    )

            guider.set_conds(positive, negative)
            print(
                "Denoising with overlap + keyframes conditioning only on sigmas: ",
                low_sigmas,
            )
            (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=low_sigmas,
                latent_image=denoised_output_latents,
            )
            positive, negative, denoised_output_latents = LTXVCropGuides.execute(
                positive=positive,
                negative=negative,
                latent=denoised_output_latents,
            )

        if optional_reference_latents is not None:
            denoised_output_latents = LTXVAdainLatent().batch_normalize(
                latents=denoised_output_latents,
                reference=optional_reference_latents,
                factor=adain_factor,
                per_frame=normalize_per_frame,
            )[0]

        # drop first output latent as it's a reinterpreted 8-frame latent understood as a 1-frame latent
        truncated_denoised_output_latents = LTXVSelectLatents().select_latents(
            denoised_output_latents, 1, -1
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
                "optional_cond_images": (
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
    RETURN_NAMES = ("denoised_video", "positive", "negative")
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
        optional_cond_images=None,
        optional_cond_indices=None,
        num_frames=0,
        optional_initialization_latents=None,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        cond_image_strength=1.0,
        guiding_strength=1.0,
        guiding_start_step=0,
        guiding_end_step=1000,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)
        if optional_cond_images is None:
            optional_cond_indices = None

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(
                optional_cond_images
            ), "Number of optional cond images must match number of optional cond indices"

        positive, negative = _get_raw_conds_from_guider(guider)

        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )

        batch, channels, frames, height, width = guiding_latents["samples"].shape
        if num_frames != -1:
            frames = (num_frames - 1) // time_scale_factor + 1

        if optional_initialization_latents is not None:
            new_latents = optional_initialization_latents
        else:
            new_latents = EmptyLTXVLatentVideo.execute(
                width=width * width_scale_factor,
                height=height * height_scale_factor,
                length=(frames - 1) * time_scale_factor + 1,
                batch_size=1,
            )[0]

        high_sigmas, rest_sigmas = SplitSigmas().get_sigmas(sigmas, guiding_start_step)
        middle_sigmas, low_sigmas = SplitSigmas().get_sigmas(
            rest_sigmas, guiding_end_step - guiding_start_step
        )

        if len(high_sigmas) > 1:
            print(
                "Denoising with keyframes only [if available] on sigmas: ",
                high_sigmas,
            )
            (_, new_latents) = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=high_sigmas,
                latent_image=new_latents,
            )

        if optional_cond_indices is not None and 0 in optional_cond_indices:
            guiding_latents = LTXVSelectLatents().select_latents(
                guiding_latents, 1, -1
            )[0]
            skip_one_guiding_latent = True
        else:
            skip_one_guiding_latent = False

        print("Adding conditioning on guiding latents")
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
            latent_idx=1 if skip_one_guiding_latent else 0,
            strength=guiding_strength,
        )
        if optional_cond_images is not None:
            print("Adding conditioning on keyframes", optional_cond_indices)
            for cond_image, cond_idx in zip(
                optional_cond_images, optional_cond_indices
            ):
                if cond_idx % 8 == 1:
                    raise ValueError(
                        f"Conditioning image index {cond_idx} is a multiple of 8 + 1 and guiding latents are used. Please provide other cond image indices"
                    )
                (
                    positive,
                    negative,
                    new_latents,
                ) = LTXVAddGuide.execute(
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    latent=new_latents,
                    image=cond_image.unsqueeze(0),
                    frame_idx=cond_idx,
                    strength=cond_image_strength,
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

        guider.set_conds(positive, negative)

        # Denoise the latent video
        print("Denoising with full conditioning on sigmas: ", middle_sigmas)
        (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=middle_sigmas,
            latent_image=new_latents,
        )

        # Clean up guides if image conditioning was used
        positive, negative, denoised_output_latents = LTXVCropGuides.execute(
            positive=positive,
            negative=negative,
            latent=denoised_output_latents,
        )

        if len(low_sigmas) > 1:
            guider.set_conds(positive, negative)
            print(
                "Denoising with keyframes only [if available] conditioning on sigmas: ",
                low_sigmas,
            )
            (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=low_sigmas,
                latent_image=denoised_output_latents,
            )
            positive, negative, denoised_output_latents = LTXVCropGuides.execute(
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
        combined_batch_index = torch.arange(0, combined_samples.shape[0]).to(
            dtype=torch.float32
        )

        return (
            {
                "samples": combined_samples,
                "batch_index": combined_batch_index,
            },
        )


@comfy_node(description="LTXV Normalizing Sampler")
class LTXVNormalizingSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVNormalizingSampler",
            category="utility",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.String.Input(
                    "video_normalization_factors", default="1,1,1,1,1,1,1,1"
                ),
                io.String.Input(
                    "audio_normalization_factors", default="1,1,0.25,1,1,0.25,1,1"
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="denoised_output"),
            ],
        )

    @classmethod
    def execute(
        cls,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        video_normalization_factors,
        audio_normalization_factors,
    ) -> io.NodeOutput:

        if (
            guider.model_patcher.model.diffusion_model.__class__.__name__
            != "LTXAVModel"
        ):
            raise ValueError()

        ltxav: LTXAVModel = guider.model_patcher.model.diffusion_model

        video_normalization_factors = video_normalization_factors.split(",")
        audio_normalization_factors = audio_normalization_factors.split(",")
        video_normalization_factors = [
            float(factor) for factor in video_normalization_factors
        ]
        audio_normalization_factors = [
            float(factor) for factor in audio_normalization_factors
        ]

        # Extend normalization factors to match the length of sigmas
        sigmas_len = len(sigmas) - 1
        if len(video_normalization_factors) < sigmas_len:
            if len(video_normalization_factors) > 0:
                video_normalization_factors.extend(
                    [video_normalization_factors[-1]]
                    * (sigmas_len - len(video_normalization_factors))
                )
        if len(audio_normalization_factors) < sigmas_len:
            if len(audio_normalization_factors) > 0:
                audio_normalization_factors.extend(
                    [audio_normalization_factors[-1]]
                    * (sigmas_len - len(audio_normalization_factors))
                )

        # Calculate indices where both normalization factors are not 1.0
        sampling_split_indices = [
            i + 1
            for i, (v, a) in enumerate(
                zip(video_normalization_factors, audio_normalization_factors)
            )
            if v != 1.0 or a != 1.0
        ]
        print("Sampling split indices: %s" % sampling_split_indices, flush=True)

        # Split sigmas according to sampling_split_indices
        def split_by_indices(arr, indices):
            """
            Splits arr into chunks according to indices (split points).
            Indices are treated as starting a new chunk at each index in the list.
            """
            if not indices:
                return [arr]
            split_points = sorted(set(indices))
            chunks = []
            prev = 0
            for idx in split_points:
                if prev < idx:
                    chunks.append(arr[prev : idx + 1])
                prev = idx
            if prev < len(arr):
                chunks.append(arr[prev:])
            return chunks

        sigmas_chunks = split_by_indices(sigmas, sampling_split_indices)
        print("Sigmas chunks: %s" % sigmas_chunks, flush=True)

        i = 0
        for sigmas_chunk in sigmas_chunks:
            i += len(sigmas_chunk) - 1
            print("Sampling with sigmas %s" % (sigmas_chunk), flush=True)
            (_, latent_image) = SamplerCustomAdvanced().execute(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas_chunk,
                latent_image=latent_image,
            )
            video_samples, audio_samples = ltxav.separate_audio_and_video_latents(
                latent_image["samples"].tensors,
                None,
            )
            if i - 1 < len(video_normalization_factors) and i - 1 < len(
                audio_normalization_factors
            ):
                video_samples = video_samples * video_normalization_factors[i - 1]
                audio_samples = audio_samples * audio_normalization_factors[i - 1]
                latent_image["samples"] = NestedTensor(
                    ltxav.recombine_audio_and_video_latents(
                        video_samples, audio_samples
                    )
                )
                print(
                    "After %d steps, the latent image was normalized by %f and %f"
                    % (
                        i,
                        video_normalization_factors[i - 1],
                        audio_normalization_factors[i - 1],
                    ),
                    flush=True,
                )

        return io.NodeOutput(latent_image)
