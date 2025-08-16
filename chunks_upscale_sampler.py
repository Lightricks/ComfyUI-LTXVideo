from dataclasses import dataclass
import copy
import json
import torch
import comfy

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from nodes import VAEDecode, VAEEncode

from comfy_extras.nodes_post_processing import Blend
from comfy_extras.nodes_lt import LTXVAddGuide, LTXVCropGuides, LTXVPreprocess, EmptyLTXVLatentVideo

from .latents import LTXVSelectLatents, LTXVAddLatentGuide
from .nodes_registry import comfy_node

from .nodes_registry import comfy_node
from .easy_samplers import LTXVExtendSampler, LTXVInContextSampler, LinearOverlapLatentTransition

from .guide import blur_internal

def smallest_valid_part(n):
    if n < 48:
        return n  # Return as is if below minimum

    for part in range(48, n + 1, 8):  # Only multiples of 8 starting from 48
        if n % part == 0:
            return part  # Found the smallest valid part

    return None

@comfy_node(
    name="LTXVChunksUpscaleSampler",
)
class LTXVChunksUpscaleSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to use."}),
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
                "noise": ("NOISE", {"tooltip": "The noise to use."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler to use."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigmas to use."}),
                "use_single_pass": (
                    "BOOLEAN",
                    {"tooltip": "Sample using a single sampler pass, the chunk_conditionings will be ignored and it will use a different traditional method to resample, this is useful for videos under 300 frames, " +
                     "the advantage is slighly higher image quality and clarity, disadvantage is out of memory errors with large videos", "default": False}
                ),
                "guider": (
                    "GUIDER",
                    {"tooltip": "The guider to use, must be a STGGuiderAdvanced."},
                ),
                "latents": (
                    "LATENT",
                    {
                        "tooltip": "The latents that will be upscaled, they should have already been processed by the upscale model",
                    },
                ),
                "chunks": (
                    "STRING",
                    {
                        "tooltip": "Each chunk as it was generated at the time",
                    },
                ),
                "chunk_conditionings": (
                    "CONDITIONING",
                    {
                        "tooltip": "Multi conditionings for each chunk"
                    },
                )
            },
            "optional": {
                "optional_cond_images": (
                    "IMAGE",
                    {
                        "tooltip": "The images that were used for conditioning during the generation, in order, each chunk will consume images as specified by the size of the image tensor that was used"
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("upscaled_latents",)

    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        use_single_pass,
        guider,
        latents,
        chunks,
        chunk_conditionings,
        optional_cond_images=None,
    ):
        #first we need to get our chunk information from the chunk data
        chunkssplitted = chunks.split("|")

        batch, channels, frames, height, width = latents["samples"].shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        width = width * width_scale_factor
        height = height * height_scale_factor

        assert len(chunkssplitted) == len(chunk_conditionings), "The positive conditionings must match the size of the chunks splitted by a pipe"

        chunkssplitted = [json.loads(i) for i in chunkssplitted]

        start_index = 0
        for chunkindex in range(0, len(chunkssplitted)):
            chunk = chunkssplitted[chunkindex]
            images_to_consume = chunk.get('images_used', 0)
            #prev_chunk = None if chunkindex == 0 else chunkssplitted[chunkindex - 1]
            #next_chunk = None if chunkindex == len(chunkssplitted) - 1 else chunkssplitted[chunkindex + 1]
            
            if images_to_consume != 0:
                end_index = start_index + images_to_consume

                # Check for over-consumption
                if optional_cond_images is None or end_index > len(optional_cond_images):
                    raise ValueError("Error: Not enough images in the optional_cond_images tensor to satisfy a chunk.")
                
                # Slice the tensor
                consumed_images = optional_cond_images[start_index:end_index]

                # We preprocess the images if we are to use a single pass
                if use_single_pass:
                    consumed_images = (
                        comfy.utils.common_upscale(
                            consumed_images.movedim(-1, 1),
                            width,
                            height,
                            "bilinear",
                            crop=chunk.get("crop", "disabled"),
                        )
                        .movedim(1, -1)
                        .clamp(0, 1)
                    )
                    consumed_images = LTXVPreprocess().preprocess(
                        consumed_images, 0
                    )[0]
                    for i in range(consumed_images.shape[0]):
                        consumed_images[i] = blur_internal(
                            consumed_images[i].unsqueeze(0), 0
                        )

                # Add the consumed images to the chunk and append to results
                chunk['cond_images'] = consumed_images
                image_indexes = [int(i) for i in chunk.get('cond_indices', "").split(",")]
                chunk["cond_indices"] = image_indexes
                image_strengths = [float(i) for i in chunk.get('cond_strengths', "").split(",")]
                chunk["cond_strengths"] = image_strengths
                chunk["cond_use_latent_guide"] = chunk.get('cond_use_latent_guide', "").split(",")
                # Update the starting index for the next iteration
                start_index = end_index

        if optional_cond_images is not None and start_index < len(optional_cond_images):
            raise ValueError("Error: Not all images were consumed from the tensor.")

        # now time to process the latents
        if not use_single_pass:
            # we do simple procesisng if not single pass to get our latents
            current_latent_idx_processed = 0
            for chunkindex in range(0, len(chunkssplitted)):
                chunk = chunkssplitted[chunkindex]
                num_frames = chunk.get("frames_generated", 0)
                frames_in_latent_space = int((num_frames - 1) // 8) + 1
                from_frame = current_latent_idx_processed
                until_frame = current_latent_idx_processed + frames_in_latent_space
                conditioning = chunk_conditionings[chunkindex]
                num_frames = chunk.get("frames_generated", 0)
                chunk["latents"] = LTXVSelectLatents().select_latents(
                    latents, from_frame, until_frame - 1
                )[0]
                chunk["conditioning"] = conditioning
                current_latent_idx_processed = current_latent_idx_processed + frames_in_latent_space
        else:
            # otherwise instead we are going to modify the latent by adding the images
            frames_accumulated = 0
            added_images = False
            for chunkindex in range(0, len(chunkssplitted)):
                chunk = chunkssplitted[chunkindex]
                num_frames = chunk.get("frames_generated", 0)

                for cond_image, cond_raw_idx, cond_strength, cond_use_latent_guide in zip(
                    chunk['cond_images'], chunk["cond_indices"], chunk["cond_strengths"], chunk["cond_use_latent_guide"]
                ):
                    cond_idx = cond_raw_idx + frames_accumulated
                    added_images = True
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

                frames_accumulated = frames_accumulated + num_frames

            guider = copy.copy(guider)
            guider.set_conds(positive, negative)

            # Denoise the tile
            denoised_output_simple = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latents,
            )[0]

            if added_images:
                _, _, denoised_output_simple = LTXVCropGuides().crop(
                    positive=positive,
                    negative=negative,
                    latent=denoised_output_simple,
                )

            return (denoised_output_simple, )

        # now we will split too large chunks in half after the second chunk forwards
        # or at least about that much, we are going to split these chunks so they are not as large
        realchunks = []
        for chunkindex in range(0, len(chunkssplitted)):
            # grab the actual chunk
            chunk = chunkssplitted[chunkindex]
            # get the number of frames
            num_frames = chunk.get("frames_generated", 0)
            # get the smallest valid divisible chunk amount
            expected_fragment_size = smallest_valid_part(num_frames) if chunkindex != 0 else smallest_valid_part(num_frames+7)
            # find the amounts
            expected_framgent_amounts = num_frames // expected_fragment_size if chunkindex != 0 else (num_frames+7) // expected_fragment_size

            # now we can start looping to divide this chunk by those amounts
            current_chunk_start = 0
            for i in range(0, expected_framgent_amounts):
                start_at = current_chunk_start
                end_at = current_chunk_start + expected_fragment_size

                if chunkindex == 0 and i == 0:
                    # remove these lost frames of the start to make the first even frame
                    end_at = end_at - 7
                
                minichunk = chunk.copy()
                minichunk['cond_images'] = []
                minichunk['cond_indices'] = []
                minichunk['cond_strengths'] = []

                #extra_latents_to_add = 1 if chunkindex == 0 and i == 0 else 0
                extra_latents_to_add = 0
                minichunk['latents'] = LTXVSelectLatents().select_latents(
                    chunk['latents'], start_at // 8, ((current_chunk_start + expected_fragment_size) // 8) - 1 + extra_latents_to_add
                )[0]
                
                real_start_at_with_n_index = start_at
                if chunkindex == 0 and i != 0:
                    real_start_at_with_n_index = real_start_at_with_n_index - 7

                for index_in_array, image_frame_index_in_video in enumerate(chunk["cond_indices"]):
                    if image_frame_index_in_video < end_at and image_frame_index_in_video >= real_start_at_with_n_index:
                        minichunk['cond_images'].append(chunk['cond_images'][index_in_array].unsqueeze(0))
                        minichunk['cond_indices'].append(image_frame_index_in_video - real_start_at_with_n_index)
                        minichunk['cond_strengths'].append(chunk['cond_strengths'][index_in_array])

                minichunk['cond_images'] = None if len(minichunk['cond_images']) == 0 else torch.cat(minichunk['cond_images'], dim=0)
                realchunks.append(minichunk)

                current_chunk_start = current_chunk_start + expected_fragment_size
        
        # Commented out this code checks that the shape of the tensor is correct
        #real_tensor_idx = 0
        #for i in range(0, len(realchunks)):
        #    chunk = realchunks[i]
        #    latents_to_check = chunk["latents"]["samples"]

        #    for j in range(0, latents_to_check.shape[2]):
        #        latent_a = latents_to_check[:, :, j : j + 1, :, :]
        #        latent_b = latents["samples"][:, :, real_tensor_idx : real_tensor_idx + 1, :, :]
        #        if torch.equal(latent_a, latent_b):
        #            print(real_tensor_idx, "TENSOR IS EQUAL AT CHUNK", i, j)
        #        else:
        #            print(real_tensor_idx, "TENSOR IS NOT EQUAL AT CHUNK", i, j)

                    #find which one it is
        #            for tensor_special_idx in range(0, latents["samples"].shape[2]):
        #                compared_latent_tensor = latents["samples"][:, :, tensor_special_idx : tensor_special_idx + 1, :, :]

        #                if torch.equal(latent_a, compared_latent_tensor):
        #                    raise ValueError("TENSOR IS NOT EQUAL, FOUND AT: " + str(tensor_special_idx))

        #            raise ValueError("TENSOR IS NOT EQUAL")
                
        #        real_tensor_idx = real_tensor_idx + 1


        final_latents = None
        for i in range(0, len(realchunks)):
            chunk = realchunks[i]

            new_guider = guider
            new_guider = copy.copy(guider)
            positive, negative = guider.raw_conds
            new_guider.set_conds(
                chunk["conditioning"],
                negative,
            )
            new_guider.raw_conds = (
                chunk["conditioning"],
                negative,
            )

            # avoid modifying other latents that share the same slice
            latents_to_render = chunk["latents"]
            # get the latent from the previous latent info

            if final_latents is None:
                final_latents = LTXVInContextSampler().sample(
                    vae,
                    new_guider,
                    sampler,
                    sigmas,
                    noise,
                    latents_to_render,
                    optional_cond_image=chunk['cond_images'],
                    num_frames=-1,
                    optional_cond_strength=1.0,
                    optional_guiding_strength=0.2,
                    optional_cond_image_strength=",".join(str(i) for i in chunk['cond_strengths']),
                    optional_cond_image_indices=",".join(str(i) for i in chunk['cond_indices']),
                    optional_cond_use_latent_guide=None,
                    crop=chunk.get("crop", "disabled"),
                    crf=chunk.get("crf", 0),
                    blur=chunk.get("blur", 0),
                )
            else:
                return LTXVExtendSampler().sample(
                    model,
                    vae,
                    final_latents,
                    -1,
                    16,
                    new_guider,
                    sampler,
                    sigmas,
                    noise,
                    strength=0.5,
                    guiding_strength=0.2,
                    optional_guiding_latents=latents_to_render,
                    crop=chunk.get("crop", "disabled"),
                    crf=chunk.get("crf", 0),
                    blur=chunk.get("blur", 0),
                    optional_cond_images=chunk['cond_images'],
                    optional_cond_indices=",".join(str(i) for i in chunk['cond_indices']),
                    optional_cond_strength=",".join(str(i) for i in chunk['cond_strengths']),
                    optional_cond_use_latent_guide=None,

                    guiding_latents_already_cropped=True,
                )

        return (final_latents,)
