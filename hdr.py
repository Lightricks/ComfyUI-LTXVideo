"""HDR utilities and ComfyUI node for HDR IC-LoRA inference.

Provides HDR compression/decompression transforms and a single ComfyUI node:

- LTXVHDRDecodePostprocess: Decompress VAE output, tonemap for preview,
  and optionally save raw linear HDR frames as EXR.
"""

import logging
import os

import torch
from torch import Tensor

from .nodes_registry import comfy_node

logger = logging.getLogger("LTXVideo.hdr")


# ---------------------------------------------------------------------------
# HDR Transform Classes (ported from ltx_core/hdr.py)
# ---------------------------------------------------------------------------


class LogC3:
    """ARRI LogC3 (EI 800) HDR compression.

    Maps linear [0, inf) -> [0, 1] via the camera log curve, then scales to
    [-1, 1] for VAE input.
    """

    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591

    def compress(self, hdr: Tensor) -> Tensor:
        x = torch.clamp(hdr, min=0.0)
        log_part = self.C * torch.log10(self.A * x + self.B) + self.D
        lin_part = self.E * x + self.F
        logc = torch.where(x >= self.CUT, log_part, lin_part)
        logc = torch.clamp(logc, 0.0, 1.0)
        return logc * 2.0 - 1.0

    def decompress(self, z: Tensor) -> Tensor:
        logc = torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        lin_from_log = (torch.pow(10.0, (logc - self.D) / self.C) - self.B) / self.A
        lin_from_lin = (logc - self.F) / self.E
        return torch.where(logc >= cut_log, lin_from_log, lin_from_lin)


# ---------------------------------------------------------------------------
# Transform registry — singleton (stateless)
# ---------------------------------------------------------------------------

_LOGC3 = LogC3()


def _hdr_decompress(decoded_01: Tensor) -> Tensor:
    """Decompress VAE-decoded image from [0,1] to linear HDR [0, inf).

    ComfyUI's VAE decode returns images in [0, 1] via ``(raw + 1) / 2``.
    This function reverses that and applies LogC3 HDR decompression.
    """
    raw = decoded_01.float() * 2.0 - 1.0
    return _LOGC3.decompress(raw)


def _linear_to_srgb(x: Tensor) -> Tensor:
    """Convert linear [0, 1] to sRGB [0, 1]."""
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(x.clamp(min=0.0031308), 1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------


@comfy_node(name="LTXVHDRDecodePostprocess")
class LTXVHDRDecodePostprocess:
    """Decompress HDR from VAE output, tonemap for preview, optionally save EXR.

    Place after VAE Decode in an HDR IC-LoRA workflow. Recovers linear HDR
    values from the compressed latent space and tonemaps them to SDR for
    display. When ``save_exr`` is enabled, also writes the raw linear HDR
    frames as an EXR image sequence.
    make sure to set OPENCV_IO_ENABLE_OPENEXR=1 environment in the command line  # Must be set before cv2 import


    Outputs:
        tonemapped: SDR preview [0, 1] after Reinhard tonemap + sRGB gamma.
        hdr_linear: Raw linear HDR values [0, inf) for further processing.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": (
                            "Exposure in stops (EV). 0 = no change, "
                            "+1 = 2x brighter, -1 = half brightness."
                        ),
                    },
                ),
                "save_exr": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Save raw linear HDR frames as EXR sequence.",
                    },
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "output/hdr_exr",
                        "tooltip": (
                            "Directory for EXR frames (relative to ComfyUI "
                            "output directory, or absolute path)."
                        ),
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "frame"},
                ),
                "half_precision": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Save EXR as float16 (half). Smaller files, "
                            "negligible quality loss for most workflows."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("tonemapped", "hdr_linear")
    OUTPUT_NODE = True
    FUNCTION = "postprocess"
    CATEGORY = "Lightricks/HDR"
    DESCRIPTION = (
        "Decompresses VAE-decoded output from HDR IC-LoRA (LogC3) and applies "
        "Reinhard tonemapping. Place after VAE Decode. 'tonemapped' is the "
        "SDR preview; 'hdr_linear' is raw linear HDR for downstream use. "
        "Enable 'save_exr' to write an EXR image sequence."
        "if save_exr is enabled, make sure to set OPENCV_IO_ENABLE_OPENEXR=1 environment in the command line"
    )

    def postprocess(
        self,
        image: torch.Tensor,
        exposure: float = 0.0,
        save_exr: bool = False,
        output_dir: str = "output/hdr_exr",
        filename_prefix: str = "frame",
        half_precision: bool = True,
    ) -> tuple:

        hdr = _hdr_decompress(image)
        hdr = torch.clamp(hdr, min=0.0, max=1e4)

        # Reinhard tonemap with exposure (stops → linear multiplier)
        exposure_mult = 2.0**exposure
        hdr_exposed = hdr * exposure_mult
        tonemapped_linear = (hdr_exposed / (1.0 + hdr_exposed)).clamp(0.0, 1.0)
        tonemapped = _linear_to_srgb(tonemapped_linear)

        if save_exr:
            assert os.environ.get("OPENCV_IO_ENABLE_OPENEXR") == "1", (
                "EXR output is enabled (save_exr = TRUE), but OpenCV does not support EXR by default. "
                "To enable it, set the environment variable OPENCV_IO_ENABLE_OPENEXR=1 before starting ComfyUI, then restart. "
                "Alternatively, disable EXR output or switch to PNG/JPG."
            )
            self._save_exr_frames(hdr, output_dir, filename_prefix, half_precision)

        return (tonemapped, hdr)

    @staticmethod
    def _save_exr_frames(
        hdr_image: torch.Tensor,
        output_dir: str,
        filename_prefix: str,
        half_precision: bool,
    ) -> None:
        try:
            import cv2
        except ImportError:
            logger.error(
                "opencv-python is required for EXR export. "
                "Install with: pip install opencv-python"
            )
            return

        import folder_paths
        import numpy as np

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(folder_paths.get_output_directory(), output_dir)

        os.makedirs(output_dir, exist_ok=True)

        frames = hdr_image.cpu().numpy()
        exr_type = (
            cv2.IMWRITE_EXR_TYPE_HALF if half_precision else cv2.IMWRITE_EXR_TYPE_FLOAT
        )
        params = [
            cv2.IMWRITE_EXR_TYPE,
            exr_type,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_ZIP,
        ]

        for i in range(frames.shape[0]):
            frame_bgr = frames[i][:, :, ::-1].astype(np.float32).copy()
            path = os.path.join(output_dir, f"{filename_prefix}_{i:05d}.exr")
            cv2.imwrite(path, frame_bgr, params)

        logger.info(
            "Saved %d EXR frame(s) to %s (%s)",
            frames.shape[0],
            output_dir,
            "float16" if half_precision else "float32",
        )
