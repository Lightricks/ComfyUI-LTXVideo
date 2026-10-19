"""HDR file I/O helpers: EXR read/write, path utilities.

Shared by Comfy HDR nodes (ingest + EXR delivery). Color math lives in ``hdr_color``;
node classes live in ``hdr_nodes``.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import folder_paths
import numpy as np
import torch
from torch import Tensor

from .hdr_color import Primaries

logger = logging.getLogger("LTXVideo.hdr_io")


def _resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() and p.exists():
        return p
    # Prefer Comfy input dir for relative paths.
    candidate = Path(folder_paths.get_input_directory()) / path
    if candidate.exists():
        return candidate
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(
        f"HDR path not found: {path!r} (tried {candidate} and {p.resolve()})"
    )


def _natural_sort_key(name: str) -> tuple:
    """Sort key so ``frame_9.exr`` precedes ``frame_10.exr`` (not lexicographic)."""
    parts = re.split(r"(\d+)", name)
    key: list = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold()))
    return tuple(key)


def _list_exr_files(path: Path) -> list[Path]:
    """Resolve a still ``.exr`` or an EXR folder to a sorted frame list."""
    if path.is_file():
        if path.suffix.lower() != ".exr":
            raise ValueError(f"Expected a .exr file; got {path}")
        return [path]
    if path.is_dir():
        # Case-insensitive match (Linux is case-sensitive; macOS usually is not).
        files = sorted(
            {p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".exr"},
            key=lambda p: _natural_sort_key(p.name),
        )
        if not files:
            raise RuntimeError(f"No .exr frames in {path}")
        return files
    raise ValueError(f"Path must be a .exr file or a directory of EXRs; got {path}")


def _silent_audio(num_frames: int, frame_rate: float, sample_rate: int = 44100) -> dict:
    """Stereo silence long enough for the LTX audio VAE mel path.

    EXR plates have no soundtrack; V2V still freezes an audio latent. A 0-length
    waveform crashes ``torchaudio.functional.resample`` / ``VAEEncodeAudio``.
    """
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be > 0; got {frame_rate}")
    duration_s = max(float(num_frames) / float(frame_rate), 1.0)
    # Keep a comfortable floor for mel/STFT (typical n_fft is 1024–2048).
    n_samples = max(int(math.ceil(duration_s * sample_rate)), sample_rate, 8192)
    return {
        "waveform": torch.zeros(1, 2, n_samples, dtype=torch.float32),
        "sample_rate": int(sample_rate),
    }


def _read_exr(path: Path) -> Tensor:
    """Read one EXR as float32 ``[H,W,3]`` (unbounded linear or log codes as stored).

    Requires OpenImageIO (same as ``ltx_pipelines`` media I/O). No OpenCV fallback.
    """
    try:
        import OpenImageIO as oiio
    except ImportError as e:
        raise RuntimeError(
            "Reading EXR requires OpenImageIO. Install with: pip install openimageio"
        ) from e

    buf = oiio.ImageBuf(str(path))
    pixels = buf.get_pixels(oiio.FLOAT)
    if pixels is None or buf.has_error:
        raise RuntimeError(f"OpenImageIO failed on '{path}': {buf.geterror()}")
    arr = np.ascontiguousarray(np.asarray(pixels, dtype=np.float32))

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(arr[..., :3]))


def _save_exr_frame(
    frame_hwc: Tensor,
    path: Path,
    *,
    half_precision: bool,
    primaries: Primaries | None,
    color_space_tag: str | None,
) -> None:
    """Write one EXR via OpenImageIO (same backend as ``ltx_pipelines``).

    Optional ``primaries`` / ``color_space_tag`` set EXR header attributes when
    provided; ``exr_color_space=linear`` may omit them for legacy untagged plates.
    """
    try:
        import OpenImageIO as oiio
    except ImportError as e:
        raise RuntimeError(
            "Saving EXR requires OpenImageIO. Install with: pip install openimageio"
        ) from e

    rgb = frame_hwc.detach().float().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgb.shape
    fmt = oiio.HALF if half_precision else oiio.FLOAT
    spec = oiio.ImageSpec(w, h, 3, fmt)
    spec.channelnames = ("R", "G", "B")
    spec.attribute("compression", "zip")
    if primaries is not None:
        spec.attribute(
            "chromaticities",
            "float[8]",
            primaries.exr_chromaticities,
        )
    if color_space_tag is not None:
        spec.attribute("colorSpace", color_space_tag)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(
            f"OpenImageIO cannot create writer for {path}. "
            "Ensure OpenImageIO is built with OpenEXR support."
        )
    try:
        if not out.open(str(path), spec):
            raise RuntimeError(
                f"Failed to access EXR output file at '{path}': {out.geterror()}"
            )
        # OIIO converts to HALF from float32 buffer when spec is HALF.
        if not out.write_image(np.ascontiguousarray(rgb, dtype=np.float32)):
            raise RuntimeError(f"Failed to write EXR '{path}': {out.geterror()}")
    finally:
        out.close()


def _trim_to_8k1(images: Tensor) -> Tensor:
    """Keep longest legal ``8k+1`` prefix (VAE temporal scale 8)."""
    n = images.shape[0]
    legal = ((n - 1) // 8) * 8 + 1
    if legal < 1:
        raise ValueError(f"Need at least 1 frame; got {n}")
    if legal != n:
        logger.warning(
            "Trimmed EXR sequence from %d to %d frames (must be 8k+1)", n, legal
        )
    return images[:legal]
