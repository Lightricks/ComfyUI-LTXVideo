import copy

import comfy
import torch
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_lt import LTXVAddGuide, LTXVCropGuides

from .latents import LTXVAddLatentGuide, LTXVSelectLatents
from .nodes_registry import comfy_node


@comfy_node(
    name="LTXVTiledSampler",
)
class LTXVTiledSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "guider": ("GUIDER",),
                "latents": ("LATENT",),
                "horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 6}),
                "vertical_tiles": ("INT", {"default": 1, "min": 1, "max": 6}),
                "overlap": ("INT", {"default": 1, "min": 1, "max": 8}),
                "latents_cond_strength": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "boost_latent_similarity": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "crop": (["center", "disabled"], {"default": "disabled"}),
            },
            "optional": {
                "optional_cond_images": ("IMAGE",),
                "optional_cond_indices": ("STRING", {"default": "0"}),
                "images_cond_strengths": ("STRING", {"default": "0.9"}),
            },
        }

    RETURN_TYPES = (
        "LATENT",
        "LATENT",
    )
    RETURN_NAMES = (
        "output",
        "denoised_output",
    )

    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        guider,
        latents,
        horizontal_tiles,
        vertical_tiles,
        overlap,
        latents_cond_strength,
        boost_latent_similarity,
        crop="disabled",
        optional_cond_images=None,
        optional_cond_indices="0",
        images_cond_strengths="0.9",
    ):

        # Get the latent samples
        samples = latents["samples"]

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        # Validate image dimensions if provided
        if optional_cond_images is not None:
            img_height = height * height_scale_factor
            img_width = width * width_scale_factor
            cond_images = comfy.utils.common_upscale(
                optional_cond_images.movedim(-1, 1),
                img_width,
                img_height,
                "bicubic",
                crop=crop,
            ).movedim(1, -1)
            img_batch, img_height, img_width, img_channels = cond_images.shape
        else:
            cond_images = None

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(
                optional_cond_images
            ), "Number of optional cond images must match number of optional cond indices"

        images_cond_strengths = [float(i) for i in images_cond_strengths.split(",")]
        if optional_cond_images is not None and len(images_cond_strengths) < len(
            optional_cond_images
        ):
            # Repeat the last value to match the length of optional_cond_images
            images_cond_strengths = images_cond_strengths + [
                images_cond_strengths[-1]
            ] * (len(optional_cond_images) - len(images_cond_strengths))

        # Calculate tile sizes with overlap
        base_tile_height = (height + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_width = (width + (horizontal_tiles - 1) * overlap) // horizontal_tiles

        # Initialize output tensor and weight tensor
        output = torch.zeros_like(samples)
        denoised_output = torch.zeros_like(samples)
        weights = torch.zeros_like(samples)

        # Get positive and negative conditioning
        try:
            positive, negative = guider.raw_conds
        except AttributeError:
            raise ValueError(
                "Guider does not have raw conds, cannot use it as a guider. "
                "Please use STGGuiderAdvanced."
            )

        # Process each tile
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                # Calculate tile boundaries
                h_start = h * (base_tile_width - overlap)
                v_start = v * (base_tile_height - overlap)

                # Adjust end positions for edge tiles
                h_end = (
                    min(h_start + base_tile_width, width)
                    if h < horizontal_tiles - 1
                    else width
                )
                v_end = (
                    min(v_start + base_tile_height, height)
                    if v < vertical_tiles - 1
                    else height
                )

                # Calculate actual tile dimensions
                tile_height = v_end - v_start
                tile_width = h_end - h_start

                print(f"Processing tile at row {v}, col {h}:")
                print(f"  Position: ({v_start}:{v_end}, {h_start}:{h_end})")
                print(f"  Size: {tile_height}x{tile_width}")

                # Extract tile
                tile = samples[:, :, :, v_start:v_end, h_start:h_end]

                # Create tile latents dict
                tile_latents = {"samples": tile}
                unconditioned_tile_latents = tile_latents.copy()

                # Handle image conditioning if provided
                if cond_images is not None:
                    # Scale coordinates for image
                    img_h_start = v_start * height_scale_factor
                    img_h_end = v_end * height_scale_factor
                    img_w_start = h_start * width_scale_factor
                    img_w_end = h_end * width_scale_factor

                    # Create copies of conditioning for this tile
                    tile_positive = positive.copy()
                    tile_negative = negative.copy()

                    for i_cond_image, (
                        cond_image,
                        cond_image_idx,
                        cond_image_strength,
                    ) in enumerate(
                        zip(cond_images, optional_cond_indices, images_cond_strengths)
                    ):
                        # Extract image tile
                        img_tile = cond_image[
                            img_h_start:img_h_end, img_w_start:img_w_end, :
                        ].unsqueeze(0)

                        print(
                            f"Applying image conditioning on cond image {i_cond_image} for tile at row {v}, col {h} with strength {cond_image_strength} at position {cond_image_idx}:"
                        )
                        print(
                            f"  Image tile position: ({img_h_start}:{img_h_end}, {img_w_start}:{img_w_end})"
                        )
                        print(f"  Image tile size: {img_tile.shape}")

                        # Add guide from image tile
                        (
                            tile_positive,
                            tile_negative,
                            tile_latents,
                        ) = LTXVAddGuide().generate(
                            positive=tile_positive,
                            negative=tile_negative,
                            vae=vae,
                            latent=tile_latents,
                            image=img_tile,
                            frame_idx=cond_image_idx,
                            strength=cond_image_strength,
                        )
                    if boost_latent_similarity:
                        middle_latent_idx = (frames - 1) // 2
                        middle_index_latent = LTXVSelectLatents().select_latents(
                            samples=unconditioned_tile_latents,
                            start_index=middle_latent_idx,
                            end_index=middle_latent_idx,
                        )[0]
                        last_index_latent = LTXVSelectLatents().select_latents(
                            samples=unconditioned_tile_latents,
                            start_index=-1,
                            end_index=-1,
                        )[0]
                        print(
                            f"using LTXVAddLatentGuide on tiled latent with latent index {middle_latent_idx} and strength {latents_cond_strength}"
                        )
                        (
                            tile_positive,
                            tile_negative,
                            tile_latents,
                        ) = LTXVAddLatentGuide().generate(
                            vae=vae,
                            positive=tile_positive,
                            negative=tile_negative,
                            latent=tile_latents,
                            guiding_latent=middle_index_latent,
                            latent_idx=middle_latent_idx,
                            strength=latents_cond_strength,
                        )
                        print(
                            f"using LTXVAddLatentGuide on tiled latent with latent index {frames-1} and strength {latents_cond_strength}"
                        )
                        (
                            tile_positive,
                            tile_negative,
                            tile_latents,
                        ) = LTXVAddLatentGuide().generate(
                            vae=vae,
                            positive=tile_positive,
                            negative=tile_negative,
                            latent=tile_latents,
                            guiding_latent=last_index_latent,
                            latent_idx=frames - 1,
                            strength=latents_cond_strength,
                        )

                    guider = copy.copy(guider)
                    guider.set_conds(tile_positive, tile_negative)

                # Denoise the tile
                denoised_tile = SamplerCustomAdvanced().sample(
                    noise=noise,
                    guider=guider,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent_image=tile_latents,
                )[0]

                # Clean up guides if image conditioning was used
                if cond_images is not None:
                    tile_positive, tile_negative, denoised_tile = LTXVCropGuides().crop(
                        positive=tile_positive,
                        negative=tile_negative,
                        latent=denoised_tile,
                    )

                # Create weight mask for this tile
                tile_weights = torch.ones_like(tile)

                # Apply horizontal blending weights
                if h > 0:  # Left overlap
                    h_blend = torch.linspace(0, 1, overlap, device=tile.device)
                    tile_weights[:, :, :, :, :overlap] *= h_blend.view(1, 1, 1, 1, -1)
                if h < horizontal_tiles - 1:  # Right overlap
                    h_blend = torch.linspace(1, 0, overlap, device=tile.device)
                    tile_weights[:, :, :, :, -overlap:] *= h_blend.view(1, 1, 1, 1, -1)

                # Apply vertical blending weights
                if v > 0:  # Top overlap
                    v_blend = torch.linspace(0, 1, overlap, device=tile.device)
                    tile_weights[:, :, :, :overlap, :] *= v_blend.view(1, 1, 1, -1, 1)
                if v < vertical_tiles - 1:  # Bottom overlap
                    v_blend = torch.linspace(1, 0, overlap, device=tile.device)
                    tile_weights[:, :, :, -overlap:, :] *= v_blend.view(1, 1, 1, -1, 1)

                # Add weighted tile to output
                output[:, :, :, v_start:v_end, h_start:h_end] += (
                    denoised_tile["samples"] * tile_weights
                )
                denoised_output[:, :, :, v_start:v_end, h_start:h_end] += (
                    denoised_tile["samples"] * tile_weights
                )

                # Add weights to weight tensor
                weights[:, :, :, v_start:v_end, h_start:h_end] += tile_weights

        # Normalize by weights
        output = output / (weights + 1e-8)
        denoised_output = denoised_output / (weights + 1e-8)

        return {"samples": output}, {"samples": denoised_output}
