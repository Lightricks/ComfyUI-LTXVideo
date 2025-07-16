import copy
from dataclasses import dataclass

import comfy
import torch

from .easy_samplers import LTXVBaseSampler, LTXVExtendSampler, LTXVInContextSampler
from .latents import LTXVSelectLatents
from .nodes_registry import comfy_node


@dataclass
class TileConfig:
    """Configuration for spatial tile processing."""

    tile_latents: dict
    tile_guiding_latents: dict
    tile_negative_index_latents: dict
    tile_cond_image: torch.Tensor
    tile_height: int
    tile_width: int
    v: int
    h: int
    vertical_tiles: int
    horizontal_tiles: int
    first_seed: int


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""

    temporal_tile_size: int
    temporal_overlap: int
    temporal_overlap_cond_strength: float
    guiding_strength: float
    adain_factor: float
    optional_negative_index: int
    optional_negative_index_strength: float
    optional_positive_conditionings: list
    time_scale_factor: int
    width_scale_factor: int
    height_scale_factor: int
    per_tile_seed_offsets: list
    per_tile_use_negative_latents: list


@dataclass
class ModelConfig:
    """Configuration for model components."""

    model: object
    vae: object
    noise: object
    sampler: object
    sigmas: object
    guider: object


@comfy_node(
    name="LTXVLoopingSampler",
)
class LTXVLoopingSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to use."}),
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
                "noise": ("NOISE", {"tooltip": "The noise to use."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler to use."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigmas to use."}),
                "guider": (
                    "GUIDER",
                    {"tooltip": "The guider to use, must be a STGGuiderAdvanced."},
                ),
                "latents": (
                    "LATENT",
                    {
                        "tooltip": "The latents to use for creating the long video, they can be guiding latents or empty latents when no guidance is used."
                    },
                ),
                "guiding_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the conditioning on guiding latents, when optional_guiding_latents are provided.",
                    },
                ),
                "temporal_tile_size": (
                    "INT",
                    {
                        "default": 80,
                        "min": 24,
                        "max": 1000,
                        "step": 8,
                        "tooltip": "The size of the temporal tile to use for the sampling, in pixel frames, in addition to the overlapping region.",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 24,
                        "min": 16,
                        "max": 80,
                        "step": 8,
                        "tooltip": "The overlap between the temporal tiles, in pixel frames.",
                    },
                ),
                "temporal_overlap_cond_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the conditioning on the latents from the previous temporal tile.",
                    },
                ),
                "horizontal_tiles": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "Number of horizontal spatial tiles.",
                    },
                ),
                "vertical_tiles": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "Number of vertical spatial tiles.",
                    },
                ),
                "spatial_overlap": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "tooltip": "Overlap between spatial tiles.",
                    },
                ),
            },
            "optional": {
                "optional_cond_image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to use for conditioning the first frame in the video (i2v setup). If not provided, the first frame will be unconditioned (t2v setup). The image will be resized to the size of the first frame."
                    },
                ),
                "optional_guiding_latents": (
                    "LATENT",
                    {
                        "tooltip": "The latents to use for guiding the sampling, typically with an IC-LoRA."
                    },
                ),
                "adain_factor": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the AdaIn operation used to fix the statistics of each new generated temporal tile, to prevent accumulated oversaturation.",
                    },
                ),
                "optional_positive_conditionings": (
                    "CONDITIONING",
                    {
                        "tooltip": "Optional way to provide changing positive prompts, one per temporal tile, using the MultiPromptProvider node."
                    },
                ),
                "optional_negative_index_latents": (
                    "LATENT",
                    {
                        "tooltip": "Special optional latents to condition on a negative index before each new temporal tile as a way to provide long term context during video generation."
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("denoised_output",)

    FUNCTION = "sample"
    CATEGORY = "sampling"

    def _extract_latent_spatial_tile(self, latent_dict, v_start, v_end, h_start, h_end):
        """Extract spatial tile from a latent dictionary."""
        if latent_dict is None:
            return None
        tile_samples = latent_dict["samples"][:, :, :, v_start:v_end, h_start:h_end]
        return {"samples": tile_samples}

    def _extract_spatial_tile(
        self,
        samples,
        optional_guiding_latents,
        optional_negative_index_latents,
        optional_cond_image,
        v_start,
        v_end,
        h_start,
        h_end,
        height_scale_factor,
        width_scale_factor,
    ):
        """Extract spatial tiles from all inputs for a given spatial region."""
        # Extract spatial tile from latents
        tile_latents = self._extract_latent_spatial_tile(
            {"samples": samples}, v_start, v_end, h_start, h_end
        )

        # Extract spatial tile from guiding latents if provided
        tile_guiding_latents = self._extract_latent_spatial_tile(
            optional_guiding_latents, v_start, v_end, h_start, h_end
        )

        # Extract spatial tile from negative index latents if provided
        tile_negative_index_latents = self._extract_latent_spatial_tile(
            optional_negative_index_latents, v_start, v_end, h_start, h_end
        )

        # Extract spatial tile from conditioning image if provided
        tile_cond_image = None
        if optional_cond_image is not None:
            # Scale coordinates for image
            img_h_start = v_start * height_scale_factor
            img_h_end = v_end * height_scale_factor
            img_w_start = h_start * width_scale_factor
            img_w_end = h_end * width_scale_factor

            # Extract image tile
            tile_cond_image = optional_cond_image[
                :, img_h_start:img_h_end, img_w_start:img_w_end, :
            ]

        return (
            tile_latents,
            tile_guiding_latents,
            tile_negative_index_latents,
            tile_cond_image,
        )

    def _process_temporal_chunks(
        self,
        tile_config: TileConfig,
        sampling_config: SamplingConfig,
        model_config: ModelConfig,
    ):
        """Process all temporal chunks for a single spatial tile."""
        chunk_index = 0
        tile_out_latents = None
        first_tile_out_latents = None

        for i_temporal_tile, (start_index, end_index) in enumerate(
            zip(
                range(
                    0,
                    tile_config.tile_latents["samples"].shape[2]
                    + sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                    sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                ),
                range(
                    sampling_config.temporal_tile_size,
                    tile_config.tile_latents["samples"].shape[2]
                    + sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                    sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                ),
            )
        ):
            if tile_config.tile_guiding_latents is not None:
                guiding_latent_chunk = LTXVSelectLatents().select_latents(
                    tile_config.tile_guiding_latents,
                    start_index,
                    min(
                        end_index - 1,
                        tile_config.tile_guiding_latents["samples"].shape[2] - 1,
                    ),
                )[0]
            else:
                guiding_latent_chunk = None

            latent_chunk = LTXVSelectLatents().select_latents(
                tile_config.tile_latents,
                start_index,
                min(end_index - 1, tile_config.tile_latents["samples"].shape[2] - 1),
            )[0]

            print(
                "Processing temporal chunk at index",
                start_index,
                "to",
                min(end_index - 1, tile_config.tile_latents["samples"].shape[2] - 1),
            )

            seed_offset = self._get_per_tile_value(
                sampling_config.per_tile_seed_offsets, i_temporal_tile
            )
            use_negative_latents = self._get_per_tile_value(
                sampling_config.per_tile_use_negative_latents, i_temporal_tile
            )

            model_config.noise.seed = self._calculate_tile_seed(
                tile_config.first_seed,
                start_index,
                tile_config.vertical_tiles,
                tile_config.horizontal_tiles,
                tile_config.v,
                tile_config.h,
                seed_offset,
            )

            # Handle optional positive conditionings
            new_guider = self._prepare_guider_for_chunk(
                model_config.guider,
                sampling_config.optional_positive_conditionings,
                chunk_index,
            )

            if start_index == 0:
                if tile_config.tile_guiding_latents is not None:
                    tile_out_latents = LTXVInContextSampler().sample(
                        vae=model_config.vae,
                        guider=new_guider,
                        sampler=model_config.sampler,
                        sigmas=model_config.sigmas,
                        noise=model_config.noise,
                        guiding_latents=guiding_latent_chunk,
                        optional_cond_image=tile_config.tile_cond_image,
                        num_frames=-1,
                        optional_negative_index_latents=(
                            tile_config.tile_negative_index_latents
                            if use_negative_latents
                            else None
                        ),
                        optional_negative_index=sampling_config.optional_negative_index,
                        optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                        optional_initialization_latents=latent_chunk,
                        optional_cond_strength=sampling_config.temporal_overlap_cond_strength,
                        optional_guiding_strength=sampling_config.guiding_strength,
                    )[0]
                else:
                    tile_out_latents = LTXVBaseSampler().sample(
                        model=model_config.model,
                        vae=model_config.vae,
                        noise=model_config.noise,
                        sampler=model_config.sampler,
                        sigmas=model_config.sigmas,
                        guider=new_guider,
                        num_frames=(
                            min(
                                sampling_config.temporal_tile_size,
                                tile_config.tile_latents["samples"].shape[2],
                            )
                            - 1
                        )
                        * sampling_config.time_scale_factor
                        + 1,
                        width=tile_config.tile_width
                        * sampling_config.width_scale_factor,
                        height=tile_config.tile_height
                        * sampling_config.height_scale_factor,
                        optional_cond_images=tile_config.tile_cond_image,
                        optional_cond_indices="0",
                        crop="center",
                        crf=30,
                        strength=sampling_config.temporal_overlap_cond_strength,
                        optional_negative_index_latents=(
                            tile_config.tile_negative_index_latents
                            if use_negative_latents
                            else None
                        ),
                        optional_negative_index=sampling_config.optional_negative_index,
                        optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                        optional_initialization_latents=latent_chunk,
                    )[0]
                first_tile_out_latents = copy.deepcopy(tile_out_latents)
            else:
                tile_out_latents = LTXVExtendSampler().sample(
                    model=model_config.model,
                    vae=model_config.vae,
                    sampler=model_config.sampler,
                    sigmas=model_config.sigmas,
                    noise=model_config.noise,
                    latents=tile_out_latents,
                    num_new_frames=(
                        latent_chunk["samples"].shape[2]
                        - sampling_config.temporal_overlap
                    )
                    * sampling_config.time_scale_factor,
                    frame_overlap=sampling_config.temporal_overlap
                    * sampling_config.time_scale_factor,
                    guider=new_guider,
                    strength=sampling_config.temporal_overlap_cond_strength,
                    guiding_strength=sampling_config.guiding_strength,
                    optional_guiding_latents=guiding_latent_chunk,
                    optional_reference_latents=first_tile_out_latents,
                    adain_factor=sampling_config.adain_factor,
                    optional_negative_index_latents=(
                        tile_config.tile_negative_index_latents
                        if use_negative_latents
                        else None
                    ),
                    optional_negative_index=sampling_config.optional_negative_index,
                    optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                    optional_initialization_latents=latent_chunk,
                )[0]

            chunk_index += 1

        return tile_out_latents

    def _create_spatial_weights(
        self, tile_samples, v, h, horizontal_tiles, vertical_tiles, spatial_overlap
    ):
        """Create blending weights for spatial tiles."""
        tile_weights = torch.ones_like(tile_samples, device=tile_samples.device)

        # Apply horizontal blending weights
        if h > 0:  # Left overlap
            h_blend = torch.linspace(0, 1, spatial_overlap, device=tile_samples.device)
            tile_weights[:, :, :, :, :spatial_overlap] *= h_blend.view(1, 1, 1, 1, -1)
        if h < horizontal_tiles - 1:  # Right overlap
            h_blend = torch.linspace(1, 0, spatial_overlap, device=tile_samples.device)
            tile_weights[:, :, :, :, -spatial_overlap:] *= h_blend.view(1, 1, 1, 1, -1)

        # Apply vertical blending weights
        if v > 0:  # Top overlap
            v_blend = torch.linspace(0, 1, spatial_overlap, device=tile_samples.device)
            tile_weights[:, :, :, :spatial_overlap, :] *= v_blend.view(1, 1, 1, -1, 1)
        if v < vertical_tiles - 1:  # Bottom overlap
            v_blend = torch.linspace(1, 0, spatial_overlap, device=tile_samples.device)
            tile_weights[:, :, :, -spatial_overlap:, :] *= v_blend.view(1, 1, 1, -1, 1)

        return tile_weights

    def _calculate_tile_seed(
        self,
        first_seed,
        start_index,
        vertical_tiles,
        horizontal_tiles,
        v,
        h,
        seed_offset,
    ):
        """Calculate the seed value for a specific temporal and spatial tile."""
        return (
            first_seed
            + start_index * (vertical_tiles * horizontal_tiles)
            + v * horizontal_tiles
            + h
            + seed_offset
        )

    def _get_per_tile_value(self, value_list, tile_index):
        """Get a value from a per-tile configuration list, using the last value if the list is shorter."""
        return value_list[min(tile_index, len(value_list) - 1)]

    def _parse_per_tile_config(self, config_string, default_value, converter_func):
        """Parse a comma-separated per-tile configuration string into a list with type conversion."""
        if config_string == "":
            config_string = default_value
        return [converter_func(item.strip()) for item in config_string.split(",")]

    def _prepare_guider_for_chunk(
        self, guider, optional_positive_conditionings, chunk_index
    ):
        """Prepare the guider for a specific chunk, handling optional positive conditionings."""
        if optional_positive_conditionings is not None:
            new_guider = copy.copy(guider)
            positive, negative = guider.raw_conds
            # Use the conditioning at chunk_index, or the last one if we've run out
            conditioning_index = min(
                chunk_index, len(optional_positive_conditionings) - 1
            )
            new_guider.set_conds(
                optional_positive_conditionings[conditioning_index],
                negative,
            )
            new_guider.raw_conds = (
                optional_positive_conditionings[conditioning_index],
                negative,
            )
            return new_guider
        else:
            return guider

    def sample(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        guider,
        latents,
        guiding_strength,
        adain_factor,
        temporal_tile_size,
        temporal_overlap,
        temporal_overlap_cond_strength,
        horizontal_tiles,
        vertical_tiles,
        spatial_overlap,
        optional_cond_image=None,
        optional_guiding_latents=None,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        optional_positive_conditionings=None,
        per_tile_seed_offsets="0",
        per_tile_use_negative_latents="1",
    ):

        # Get dimensions and prepare for spatial tiling
        samples = latents["samples"]
        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        temporal_tile_size = temporal_tile_size // time_scale_factor
        temporal_overlap = temporal_overlap // time_scale_factor
        first_seed = noise.seed

        per_tile_seed_offsets = self._parse_per_tile_config(
            per_tile_seed_offsets, "0", int
        )
        per_tile_use_negative_latents = self._parse_per_tile_config(
            per_tile_use_negative_latents, "1", lambda x: bool(int(x))
        )

        if optional_guiding_latents is not None:
            assert (
                latents["samples"].shape[2]
                == optional_guiding_latents["samples"].shape[2]
            ), "The number of frames in the latents and optional_guiding_latents must be the same"

        # Calculate tile sizes with overlap
        base_tile_height = (
            height + (vertical_tiles - 1) * spatial_overlap
        ) // vertical_tiles
        base_tile_width = (
            width + (horizontal_tiles - 1) * spatial_overlap
        ) // horizontal_tiles

        # Initialize output tensor and weight tensor
        final_output = torch.zeros_like(samples, device=samples.device)
        weights = torch.zeros_like(samples, device=samples.device)

        if optional_cond_image is not None:
            img_height = height * height_scale_factor
            img_width = width * width_scale_factor
            optional_cond_image = comfy.utils.common_upscale(
                optional_cond_image.movedim(-1, 1),
                img_width,
                img_height,
                "bicubic",
                crop="center",
            ).movedim(1, -1)
            img_batch, img_height, img_width, img_channels = optional_cond_image.shape

        # Process each spatial tile
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                # Calculate tile boundaries
                h_start = h * (base_tile_width - spatial_overlap)
                v_start = v * (base_tile_height - spatial_overlap)

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

                print(f"Processing spatial tile at row {v}, col {h}:")
                print(f"  Position: ({v_start}:{v_end}, {h_start}:{h_end})")
                print(f"  Size: {tile_height}x{tile_width}")

                # Extract spatial tiles from all inputs
                (
                    tile_latents,
                    tile_guiding_latents,
                    tile_negative_index_latents,
                    tile_cond_image,
                ) = self._extract_spatial_tile(
                    samples,
                    optional_guiding_latents,
                    optional_negative_index_latents,
                    optional_cond_image,
                    v_start,
                    v_end,
                    h_start,
                    h_end,
                    height_scale_factor,
                    width_scale_factor,
                )

                # Process all temporal chunks for this spatial tile
                tile_config = TileConfig(
                    tile_latents=tile_latents,
                    tile_guiding_latents=tile_guiding_latents,
                    tile_negative_index_latents=tile_negative_index_latents,
                    tile_cond_image=tile_cond_image,
                    tile_height=tile_height,
                    tile_width=tile_width,
                    v=v,
                    h=h,
                    vertical_tiles=vertical_tiles,
                    horizontal_tiles=horizontal_tiles,
                    first_seed=first_seed,
                )

                sampling_config = SamplingConfig(
                    temporal_tile_size=temporal_tile_size,
                    temporal_overlap=temporal_overlap,
                    temporal_overlap_cond_strength=temporal_overlap_cond_strength,
                    guiding_strength=guiding_strength,
                    adain_factor=adain_factor,
                    optional_negative_index=optional_negative_index,
                    optional_negative_index_strength=optional_negative_index_strength,
                    optional_positive_conditionings=optional_positive_conditionings,
                    time_scale_factor=time_scale_factor,
                    width_scale_factor=width_scale_factor,
                    height_scale_factor=height_scale_factor,
                    per_tile_seed_offsets=per_tile_seed_offsets,
                    per_tile_use_negative_latents=per_tile_use_negative_latents,
                )

                model_config = ModelConfig(
                    model=model,
                    vae=vae,
                    noise=noise,
                    sampler=sampler,
                    sigmas=sigmas,
                    guider=guider,
                )

                tile_out_latents = self._process_temporal_chunks(
                    tile_config,
                    sampling_config,
                    model_config,
                )

                # Create weight mask for this spatial tile
                tile_weights = self._create_spatial_weights(
                    tile_latents["samples"],
                    v,
                    h,
                    horizontal_tiles,
                    vertical_tiles,
                    spatial_overlap,
                )

                # Add weighted tile to final output
                final_output[:, :, :, v_start:v_end, h_start:h_end] += (
                    tile_out_latents["samples"].to(final_output.device) * tile_weights
                )
                weights[:, :, :, v_start:v_end, h_start:h_end] += tile_weights

        # Normalize by weights
        final_output = final_output / (weights + 1e-8)
        out_latents = {"samples": final_output}

        noise.seed = first_seed
        return (out_latents,)


@comfy_node(
    name="MultiPromptProvider",
)
class MultiPromptProvider:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Prompts to encode, one per line. Each prompt will be encoded separately. Each prompt will be used in one temporal_tile in LTXVLoopingSampler.",
                    },
                ),
                "clip": ("CLIP", {"tooltip": "CLIP model to encode the prompts."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditionings",)

    FUNCTION = "get_prompt_list"
    CATEGORY = "prompt"

    def get_prompt_list(self, prompts, clip):
        prompt_list = prompts.split("|")
        prompt_list = [prompt.strip() for prompt in prompt_list]
        encoded_prompt_list = [
            clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            for prompt in prompt_list
        ]
        return (encoded_prompt_list,)
