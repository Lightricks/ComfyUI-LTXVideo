import logging
import math
import os
import shutil
import tempfile
import weakref

import torch

import folder_paths

from .nodes_registry import comfy_node

# Same RAM-safety shape as DaSiWa's RTX Upscaler fix: decoding a long/high-res
# batch to a single torch.empty(...) output tensor can require tens of GB in
# one shot. Below this, allocate normally (fast); above it, back the tensor
# with a memory-mapped temp file so the OS pages it in/out instead of
# committing it all to RAM at once; above the hard cap, refuse with a clear
# error instead of risking an OOM freeze.
_MAX_IN_MEMORY_OUTPUT_BYTES = 8 * 1024 * 1024 * 1024
_MAX_DISK_BACKED_OUTPUT_BYTES = 64 * 1024 * 1024 * 1024
_TEMP_DISK_RESERVE_BYTES = 1024 * 1024 * 1024


def _remove_temp_file(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _allocate_decode_output(shape, dtype, device):
    required_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()

    if device.type != "cpu" or required_bytes <= _MAX_IN_MEMORY_OUTPUT_BYTES:
        return torch.empty(shape, device=device, dtype=dtype)

    if required_bytes > _MAX_DISK_BACKED_OUTPUT_BYTES:
        raise RuntimeError(
            f"LTXV Tiled VAE Decode output requires {required_bytes / 1024 ** 3:.2f} GiB, "
            f"exceeding the {_MAX_DISK_BACKED_OUTPUT_BYTES / 1024 ** 3:.0f} GiB disk-backed "
            "safety limit. Reduce frame count or resolution."
        )

    directory = folder_paths.get_temp_directory()
    os.makedirs(directory, exist_ok=True)
    if shutil.disk_usage(directory).free < required_bytes + _TEMP_DISK_RESERVE_BYTES:
        raise RuntimeError(
            f"ComfyUI temporary directory '{directory}' lacks space for the projected "
            f"{required_bytes / 1024 ** 3:.2f} GiB decode output plus a 1 GiB reserve."
        )

    descriptor, path = tempfile.mkstemp(prefix="ltxv_tiled_decode_", suffix=".mmap", dir=directory)
    os.close(descriptor)
    try:
        output = torch.from_file(path, shared=True, size=math.prod(shape), dtype=dtype).reshape(shape)
    except Exception:
        _remove_temp_file(path)
        raise
    weakref.finalize(output, _remove_temp_file, path)
    logging.info(
        f"[LTXV Tiled VAE Decode] Disk-backed output ({required_bytes / 1024 ** 3:.2f} GiB): {path}"
    )
    return output


@comfy_node(
    name="LTXVTiledVAEDecode",
)
class LTXVTiledVAEDecode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 6}),
                "vertical_tiles": ("INT", {"default": 1, "min": 1, "max": 6}),
                "overlap": ("INT", {"default": 1, "min": 1, "max": 8}),
                "last_frame_fix": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "working_device": (["cpu", "auto"], {"default": "auto"}),
                "working_dtype": (["float16", "float32", "auto"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(
        self,
        vae,
        latents,
        horizontal_tiles,
        vertical_tiles,
        overlap,
        last_frame_fix,
        working_device="auto",
        working_dtype="auto",
    ):
        # Get the latent samples
        samples = latents["samples"]

        if last_frame_fix:
            # Repeat the last frame along dimension 2 (frames)
            # samples: [batch, channels, frames, height, width]
            last_frame = samples[
                :, :, -1:, :, :
            ]  # shape: [batch, channels, 1, height, width]
            samples = torch.cat([samples, last_frame], dim=2)

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        image_frames = 1 + (frames - 1) * time_scale_factor

        # Calculate output image dimensions
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        # Calculate tile sizes with overlap
        base_tile_height = (height + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_width = (width + (horizontal_tiles - 1) * overlap) // horizontal_tiles

        # Initialize output tensor and weight tensor
        # VAE decode returns images in format [batch, height, width, channels]
        output = None
        weights = None

        target_device = samples.device if working_device == "auto" else working_device
        if working_dtype == "auto":
            target_dtype = samples.dtype
        elif working_dtype == "float16":
            target_dtype = torch.float16
        elif working_dtype == "float32":
            target_dtype = torch.float32

        output = _allocate_decode_output(
            (batch, image_frames, output_height, output_width, 3),
            target_dtype,
            torch.device(target_device) if not isinstance(target_device, torch.device) else target_device,
        )
        output.zero_()
        weights = _allocate_decode_output(
            (batch, image_frames, output_height, output_width, 1),
            target_dtype,
            torch.device(target_device) if not isinstance(target_device, torch.device) else target_device,
        )
        weights.zero_()

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

                logging.info(f"Processing VAE decode tile at row {v}, col {h}:")
                logging.info(f"  Position: ({v_start}:{v_end}, {h_start}:{h_end})")
                logging.info(f"  Size: {tile_height}x{tile_width}")

                # Extract tile
                tile = samples[:, :, :, v_start:v_end, h_start:h_end]

                # Create tile latents dict
                tile_latents = {"samples": tile}

                # Decode the tile
                decoded_tile = vae.decode(tile_latents["samples"])

                # Calculate output tile boundaries
                out_h_start = v_start * height_scale_factor
                out_h_end = v_end * height_scale_factor
                out_w_start = h_start * width_scale_factor
                out_w_end = h_end * width_scale_factor

                # Create weight mask for this tile
                tile_out_height = out_h_end - out_h_start
                tile_out_width = out_w_end - out_w_start
                tile_weights = torch.ones(
                    (batch, image_frames, tile_out_height, tile_out_width, 1),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )

                # Calculate overlap regions in output space
                overlap_out_h = overlap * height_scale_factor
                overlap_out_w = overlap * width_scale_factor

                # Apply horizontal blending weights
                if h > 0:  # Left overlap
                    h_blend = torch.linspace(
                        0, 1, overlap_out_w, device=decoded_tile.device
                    )
                    tile_weights[:, :, :, :overlap_out_w, :] *= h_blend.view(
                        1, 1, 1, -1, 1
                    )
                if h < horizontal_tiles - 1:  # Right overlap
                    h_blend = torch.linspace(
                        1, 0, overlap_out_w, device=decoded_tile.device
                    )
                    tile_weights[:, :, :, -overlap_out_w:, :] *= h_blend.view(
                        1, 1, 1, -1, 1
                    )

                # Apply vertical blending weights
                if v > 0:  # Top overlap
                    v_blend = torch.linspace(
                        0, 1, overlap_out_h, device=decoded_tile.device
                    )
                    tile_weights[:, :, :overlap_out_h, :, :] *= v_blend.view(
                        1, 1, -1, 1, 1
                    )
                if v < vertical_tiles - 1:  # Bottom overlap
                    v_blend = torch.linspace(
                        1, 0, overlap_out_h, device=decoded_tile.device
                    )
                    tile_weights[:, :, -overlap_out_h:, :, :] *= v_blend.view(
                        1, 1, -1, 1, 1
                    )

                # Add weighted tile to output
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += (
                    decoded_tile * tile_weights
                ).to(target_device, target_dtype)

                # Add weights to weight tensor
                weights[
                    :, :, out_h_start:out_h_end, out_w_start:out_w_end, :
                ] += tile_weights.to(target_device, target_dtype)

        # Normalize by weights. `weights + 1e-8` looks harmless but is NOT
        # in-place: it allocates a brand-new full-size tensor before the
        # division even starts, bypassing the disk-backed allocation used
        # for `output`/`weights` above. For a large batch that's a second
        # multi-GB buffer via the plain RAM allocator, right at the tail end
        # of the whole tiled decode - exactly the kind of spike this file's
        # disk-backed allocation was meant to prevent.
        weights.add_(1e-8)
        output /= weights

        # Reshape output to match expected format [batch * frames, height, width, channels]
        output = output.view(
            batch * image_frames, output_height, output_width, output.shape[-1]
        )

        if last_frame_fix:
            output = output[:-time_scale_factor, :, :]

        return (output,)


def compute_chunk_boundaries(
    chunk_start: int,
    temporal_tile_length: int,
    temporal_overlap: int,
    total_latent_frames: int,
):
    """Compute chunk boundaries for temporal tiling.

    Args:
        chunk_start: Starting frame index for the current chunk
        temporal_tile_length: Length of each temporal tile
        temporal_overlap: Number of frames to overlap between chunks
        total_latent_frames: Total number of latent frames

    Returns:
        Tuple of (overlap_start, chunk_end)
    """
    if chunk_start == 0:
        # First chunk: no overlap needed
        chunk_end = min(chunk_start + temporal_tile_length, total_latent_frames)
        overlap_start = chunk_start
    else:
        # Subsequent chunks: include overlap from previous chunk
        # -1 because we need one extra frame to overlap, which is decoded to a single frame
        # never overlap with the first latent frame
        overlap_start = max(1, chunk_start - temporal_overlap - 1)
        extra_frames = chunk_start - overlap_start
        chunk_end = min(
            chunk_start + temporal_tile_length - extra_frames,
            total_latent_frames,
        )

    return overlap_start, chunk_end


def calculate_temporal_output_boundaries(
    overlap_start: int, time_scale_factor: int, tile_out_frames: int
):
    """Calculate temporal output boundaries for the decoded tile.

    Args:
        overlap_start: Starting frame index including overlap
        time_scale_factor: Time scaling factor from VAE
        tile_out_frames: Number of frames in the decoded tile

    Returns:
        Tuple of (out_t_start, out_t_end)
    """
    # +1 for the first frame
    out_t_start = 1 + overlap_start * time_scale_factor

    # Calculate actual output temporal dimensions
    out_t_end = out_t_start + tile_out_frames

    return out_t_start, out_t_end


@comfy_node(
    name="LTXVSpatioTemporalTiledVAEDecode",
)
class LTXVSpatioTemporalTiledVAEDecode(LTXVTiledVAEDecode):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
                "latents": ("LATENT", {"tooltip": "The latent samples to decode."}),
                "spatial_tiles": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "tooltip": "The number of spatial tiles to use, horizontal and vertical.",
                    },
                ),
                "spatial_overlap": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "tooltip": "The overlap between the spatial tiles. (in latent frames)",
                    },
                ),
                "temporal_tile_length": (
                    "INT",
                    {
                        "default": 16,
                        "min": 2,
                        "max": 1000,
                        "tooltip": "The length of the temporal tile to use for the sampling, in latent frames, including the overlapping region.",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "tooltip": "The overlap between the temporal tiles, in latent frames.",
                    },
                ),
                "last_frame_fix": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, the last frame will be repeated and discarded after the decoding.",
                    },
                ),
                "working_device": (
                    ["cpu", "auto"],
                    {
                        "default": "auto",
                        "tooltip": "The device to use for the decoding. auto->same as the latents.",
                    },
                ),
                "working_dtype": (
                    ["float16", "float32", "auto"],
                    {
                        "default": "auto",
                        "tooltip": "The data type to use for the decoding. auto->same as the latents.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "decode_spatial_temporal"

    CATEGORY = "latent"

    def decode_spatial_temporal(
        self,
        vae,
        latents,
        spatial_tiles=4,
        spatial_overlap=1,
        temporal_tile_length=16,
        temporal_overlap=1,
        last_frame_fix=False,
        working_device="auto",
        working_dtype="auto",
    ):
        if temporal_tile_length < temporal_overlap + 1:
            raise ValueError(
                "Temporal tile length must be greater than temporal overlap + 1"
            )

        # Get the latent samples
        samples = latents["samples"]

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        image_frames = 1 + (frames - 1) * time_scale_factor

        # Calculate output image dimensions
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        target_device = samples.device if working_device == "auto" else working_device
        if working_dtype == "auto":
            target_dtype = samples.dtype
        elif working_dtype == "float16":
            target_dtype = torch.float16
        elif working_dtype == "float32":
            target_dtype = torch.float32

        # Initialize output tensor and weight tensor
        output = _allocate_decode_output(
            (batch, image_frames, output_height, output_width, 3),
            target_dtype,
            torch.device(target_device) if not isinstance(target_device, torch.device) else target_device,
        )

        # Process temporal chunks similar to reference function
        total_latent_frames = frames
        chunk_start = 0

        while chunk_start < total_latent_frames:
            # Calculate chunk boundaries
            overlap_start, chunk_end = compute_chunk_boundaries(
                chunk_start, temporal_tile_length, temporal_overlap, total_latent_frames
            )

            # units are latent frames
            chunk_frames = chunk_end - overlap_start
            logging.info(
                f"Processing temporal chunk: {overlap_start}:{chunk_end} ({chunk_frames} latent frames)"
            )

            # Extract tile
            tile = samples[:, :, overlap_start:chunk_end]

            # Create tile latents dict
            tile_latents = {"samples": tile}

            # Decode the tile
            decoded_tile = self.decode(
                vae=vae,
                latents=tile_latents,
                vertical_tiles=spatial_tiles,
                horizontal_tiles=spatial_tiles,
                overlap=spatial_overlap,
                last_frame_fix=last_frame_fix,
                working_device=working_device,
                working_dtype=working_dtype,
            )[0][None]

            if chunk_start == 0:
                output[:, : decoded_tile.shape[1]] = decoded_tile

            # Drop first frame if needed (overlap)
            else:
                if decoded_tile.shape[1] == 1:
                    raise ValueError("Dropping first frame but tile has only 1 frame")
                decoded_tile = decoded_tile[:, 1:]  # Drop first frame

                # Calculate temporal output boundaries
                out_t_start, out_t_end = calculate_temporal_output_boundaries(
                    overlap_start, time_scale_factor, decoded_tile.shape[1]
                )

                # Create weight mask for this tile
                overlap_frames = temporal_overlap * time_scale_factor
                frame_weights = torch.linspace(
                    0,
                    1,
                    overlap_frames + 2,
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )[1:-1]
                tile_weights = frame_weights.view(1, -1, 1, 1, 1)
                after_overlap_frames_start = out_t_start + overlap_frames
                # Add weighted tile to output
                overlap_output = decoded_tile[:, :overlap_frames]
                output[:, out_t_start:after_overlap_frames_start] *= 1 - tile_weights
                output[:, out_t_start:after_overlap_frames_start] += (
                    tile_weights * overlap_output
                )
                output[:, after_overlap_frames_start:out_t_end] = decoded_tile[
                    :, overlap_frames:
                ]

            # Move to next chunk
            chunk_start = chunk_end

        # Reshape output to match expected format [batch * frames, height, width, channels]
        output = output.view(
            batch * image_frames, output_height, output_width, output.shape[-1]
        )

        return (output,)
