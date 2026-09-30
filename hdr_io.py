"""HDR file I/O helpers: EXR read/write, HLG encode, path utilities.

Shared by Comfy HDR nodes (ingest + delivery). Color math lives in ``hdr_color``;
node classes live in ``hdr_nodes``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path

import folder_paths
import numpy as np
import torch
from torch import Tensor

from .hdr_color import Primaries

logger = logging.getLogger("LTXVideo.hdr_io")

# BT.2020 / HLG (ARIB STD-B67) encode constants — from ltx_core.color.hlg
_HLG_A, _HLG_B, _HLG_C = 0.17883277, 0.28466892, 0.55991073
_AV_COLOR_PRIMARIES_BT2020 = 9
_AV_COLOR_TRC_ARIB_STD_B67 = 18
_AV_COLORSPACE_BT2020_NCL = 9
_AV_COLOR_RANGE_MPEG = 1
_KR_2020, _KG_2020, _KB_2020 = 0.2627, 0.6780, 0.0593
_BT2020_RGB_TO_YUV = torch.tensor(
    [
        [_KR_2020, _KG_2020, _KB_2020],
        [-_KR_2020 / 1.8814, -_KG_2020 / 1.8814, 0.5],
        [0.5, -_KG_2020 / 1.4746, -_KB_2020 / 1.4746],
    ],
    dtype=torch.float32,
)


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


def _hlg_inverse_oetf_scalar(v: float) -> float:
    return (
        (v * v) / 3.0 if v <= 0.5 else (math.exp((v - _HLG_C) / _HLG_A) + _HLG_B) / 12.0
    )


def _linear_to_hlg_signal(
    rgb_fhwc: Tensor,
    primaries: Primaries = Primaries.REC709,
    white_signal: float = 0.75,
) -> Tensor:
    """Scene-linear RGB ``[F,H,W,3]`` in ``primaries`` → HLG signal in BT.2020.

    Maps with ``primaries.matrix_to_rec2020`` then clamps (same as
    ``ltx_core.color.hlg.HlgGpuConverter``). Prefer ACEScg→Rec.2020 directly
    over ACEScg→Rec.709→Rec.2020 so OOG negatives are not crushed early.
    """
    white_x = _hlg_inverse_oetf_scalar(white_signal)
    roll_k = white_x / (1.0 - white_x)
    mat = primaries.matrix_to_rec2020.to(device=rgb_fhwc.device, dtype=rgb_fhwc.dtype)
    lin = torch.nan_to_num((rgb_fhwc @ mat.T).clamp(min=0.0), nan=0.0, neginf=0.0)
    x = torch.where(
        lin <= 1.0,
        lin * white_x,
        1.0 - (1.0 - white_x) * torch.exp(-roll_k * (lin - 1.0)),
    )
    sig = torch.where(
        x <= 1.0 / 12.0,
        torch.sqrt((3.0 * x).clamp(min=0.0)),
        _HLG_A * torch.log((12.0 * x - _HLG_B).clamp(min=1e-12)) + _HLG_C,
    )
    return sig.clamp(0.0, 1.0)


def _rgb_hlg_to_yuv420p10(
    rgb_fhwc: Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HLG RGB ``[F,H,W,3]`` → planar uint16 Y,U,V (limited / BT.2020)."""
    if rgb_fhwc.shape[-2] % 2 or rgb_fhwc.shape[-3] % 2:
        raise ValueError(f"HLG encode needs even H and W; got {tuple(rgb_fhwc.shape)}")
    rgb = rgb_fhwc.movedim(-1, -3).contiguous()  # F,3,H,W
    mat = _BT2020_RGB_TO_YUV.to(device=rgb.device, dtype=rgb.dtype)
    pixels = rgb.movedim(-3, -1)
    yuv = pixels.flatten(-3, -2) @ mat.T
    yuv = yuv.unflatten(-2, rgb.shape[-2:]).movedim(-1, -3)
    y = yuv[:, :1]
    uv = torch.nn.functional.avg_pool2d(
        yuv[:, 1:3].contiguous(), kernel_size=2, stride=2
    )
    # MPEG limited 10-bit
    y = y.mul(219 * 4).add_(16 * 4)
    uv = uv.mul(224 * 4).add_(128 * 4)
    y16 = y[:, 0].round_().clamp_(0, 1023).to(torch.uint16).cpu().numpy()
    u16 = uv[:, 0].round_().clamp_(0, 1023).to(torch.uint16).cpu().numpy()
    v16 = uv[:, 1].round_().clamp_(0, 1023).to(torch.uint16).cpu().numpy()
    return y16, u16, v16


def _encode_hlg_mp4(
    frames_linear: Tensor,
    out_path: Path,
    fps: float,
    audio: dict | None = None,
    *,
    primaries: Primaries = Primaries.REC709,
) -> None:
    """Encode scene-linear ``[F,H,W,3]`` (in ``primaries``) to BT.2020/HLG/10-bit HEVC.

    Optional Comfy ``AUDIO`` dict ``{waveform, sample_rate}`` is muxed as AAC.
    The AAC stream is registered *before* video packets (PyAV requires this).
    """
    from fractions import Fraction

    import av

    hlg = _linear_to_hlg_signal(frames_linear.float(), primaries=primaries)
    y, u, v = _rgb_hlg_to_yuv420p10(hlg)
    f, height, width = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(out_path), mode="w", options={"movflags": "+faststart"})
    audio_stream = None
    try:
        stream = container.add_stream(
            "libx265", rate=Fraction(fps).limit_denominator(1000)
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p10le"
        stream.codec_tag = "hvc1"
        threads = max(1, min(os.cpu_count() or 8, 16))
        stream.options = {
            "crf": "12",
            "preset": "medium",
            "x265-params": (
                "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:"
                f"repeat-headers=1:info=0:pools={threads}:frame-threads=4"
            ),
        }
        ctx = stream.codec_context
        ctx.thread_count = threads
        ctx.color_primaries = _AV_COLOR_PRIMARIES_BT2020
        ctx.color_trc = _AV_COLOR_TRC_ARIB_STD_B67
        ctx.colorspace = _AV_COLORSPACE_BT2020_NCL
        ctx.color_range = _AV_COLOR_RANGE_MPEG

        if audio is not None:
            audio_stream = _prepare_aac_stream(container, audio)

        for i in range(f):
            frame = av.VideoFrame(width, height, "yuv420p10le")
            for plane, src in zip(frame.planes, (y[i], u[i], v[i]), strict=True):
                dest = np.frombuffer(plane, dtype=np.uint16).reshape(
                    plane.height, plane.line_size // 2
                )
                dest[:, : src.shape[1]] = src
            frame.colorspace = _AV_COLORSPACE_BT2020_NCL
            frame.color_range = _AV_COLOR_RANGE_MPEG
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

        if audio is not None and audio_stream is not None:
            _mux_comfy_audio_aac(container, audio_stream, audio)
    finally:
        container.close()


def _comfy_audio_stereo_n2(audio: dict) -> tuple[Tensor, int]:
    """Return stereo int16-ready float samples ``[N, 2]`` and sample rate."""
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if waveform is None or sample_rate <= 0:
        raise ValueError("AUDIO must include waveform and sample_rate")

    samples = waveform.detach().float().cpu()
    if samples.ndim == 3:
        samples = samples[0]
    if samples.ndim != 2:
        raise ValueError(
            f"Expected AUDIO waveform [C,N] or [B,C,N]; got {tuple(waveform.shape)}"
        )
    if samples.shape[0] == 2:
        samples_cn = samples
    elif samples.shape[1] == 2:
        samples_cn = samples.T
    elif samples.shape[0] == 1:
        samples_cn = samples.repeat(2, 1)
    else:
        raise ValueError(
            f"AUDIO must be mono or stereo; got shape {tuple(samples.shape)}"
        )
    return samples_cn.T.contiguous(), sample_rate


def _prepare_aac_stream(container, audio: dict):
    """Register a stereo AAC stream before any packets are muxed."""
    from fractions import Fraction

    _, sample_rate = _comfy_audio_stereo_n2(audio)
    audio_stream = container.add_stream("aac", rate=sample_rate)
    audio_stream.codec_context.sample_rate = sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, sample_rate)
    return audio_stream


def _mux_comfy_audio_aac(container, audio_stream, audio: dict) -> None:
    """Mux Comfy AUDIO onto an already-added AAC stream (after video flush)."""
    import av

    samples_n2, sample_rate = _comfy_audio_stereo_n2(audio)
    samples_i16 = (samples_n2.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples_i16.reshape(1, -1).numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = sample_rate

    cc = audio_stream.codec_context
    resampler = av.audio.resampler.AudioResampler(
        format=cc.format or "fltp",
        layout=cc.layout or "stereo",
        rate=cc.sample_rate or sample_rate,
    )
    next_pts = 0
    for rframe in resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = next_pts
        next_pts += rframe.samples
        for packet in audio_stream.encode(rframe):
            container.mux(packet)
    for rframe in resampler.resample(None):
        if rframe.pts is None:
            rframe.pts = next_pts
        next_pts += rframe.samples
        for packet in audio_stream.encode(rframe):
            container.mux(packet)
    for packet in audio_stream.encode(None):
        container.mux(packet)


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
