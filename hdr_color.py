"""ACEScct working-space + primaries math for HDR Comfy paths.

Kept in lockstep with ``ltx_core.hdr`` / ``ltx_core.color.primaries`` on
ltx-2-internal: ACEScct curve constants match ``ltx_core.hdr``, and primaries
matrices / EXR chromaticities come from ``colour-science`` with Bradford CAT
(same source OpenColorIO-Config-ACES uses for its utility CLFs).

``colour-science`` is required (same as ``ltx_core.color.primaries``).

Used by HDR Comfy nodes in ``hdr_nodes`` (EXR ↔ VAE, SDR→HDR IC-LoRA ingest
via ``SDRColorSpace`` / sRGB EOTF → ACEScct). File I/O helpers live in ``hdr_io``.
"""

from __future__ import annotations

import enum
import logging

import colour
import torch
from torch import Tensor

logger = logging.getLogger("LTXVideo.hdr_color")

# ACEScct (AMPAS S-2016-001) — same constants as ``ltx_core.hdr``
_ACESCCT_A_LIN = 10.5402377416545
_ACESCCT_B_LIN = 0.0729055341958355
_ACESCCT_X_BRK = 0.0078125
_ACESCCT_Y_BRK = 0.155251141552511
_ACESCCT_LOG_M = 17.52
_ACESCCT_LOG_B = 9.72

_CAT = "Bradford"
_CS_ACESCG = colour.RGB_COLOURSPACES["ACEScg"]
_CS_REC709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]
_CS_REC2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]


def _rgb_to_rgb(src: colour.RGB_Colourspace, dst: colour.RGB_Colourspace) -> Tensor:
    """Linear RGB→RGB matrix from colour-science, as float32."""
    return torch.as_tensor(
        colour.matrix_RGB_to_RGB(src, dst, chromatic_adaptation_transform=_CAT),
        dtype=torch.float32,
    )


def _exr_chroma(cs: colour.RGB_Colourspace) -> tuple[float, ...]:
    """EXR ``chromaticities`` order: Rxy, Gxy, Bxy, Wxy."""
    r, g, b = cs.primaries
    w = cs.whitepoint
    return (
        float(r[0]),
        float(r[1]),
        float(g[0]),
        float(g[1]),
        float(b[0]),
        float(b[1]),
        float(w[0]),
        float(w[1]),
    )


_ACESCG_TO_SRGB = _rgb_to_rgb(_CS_ACESCG, _CS_REC709)
_SRGB_TO_ACESCG = torch.linalg.inv(_ACESCG_TO_SRGB.double()).to(torch.float32)
_REC709_TO_2020 = _rgb_to_rgb(_CS_REC709, _CS_REC2020)
_ACESCG_TO_2020 = _rgb_to_rgb(_CS_ACESCG, _CS_REC2020)
_CHROMA_REC709 = _exr_chroma(_CS_REC709)
_CHROMA_ACESCG = _exr_chroma(_CS_ACESCG)
_MATRICES_SOURCE = "colour-science"

# Public aliases (parity with ``ltx_core.color.primaries`` exports).
ACESCG_TO_SRGB = _ACESCG_TO_SRGB
SRGB_TO_ACESCG = _SRGB_TO_ACESCG
REC709_TO_2020 = _REC709_TO_2020
ACESCG_TO_2020 = _ACESCG_TO_2020

_CHROMATICITIES = {
    "rec709": _CHROMA_REC709,
    "acescg": _CHROMA_ACESCG,
}


class Primaries(enum.Enum):
    """Authoring / working-space colour primaries (linear light)."""

    REC709 = "rec709"
    ACESCG = "acescg"

    @property
    def exr_chromaticities(self) -> tuple[float, ...]:
        """EXR ``chromaticities`` header (R/G/B/W x,y)."""
        return _CHROMATICITIES[self.value]

    @property
    def matrix_to_rec2020(self) -> Tensor:
        """3x3 linear map from this basis to Rec.2020 (for HLG)."""
        return _REC709_TO_2020 if self is Primaries.REC709 else _ACESCG_TO_2020

    def to(self, target: Primaries, video: Tensor) -> Tensor:
        """Convert linear light from this basis into ``target``. Identity when same."""
        if self is target:
            return video
        if (self, target) == (Primaries.REC709, Primaries.ACESCG):
            return _apply_matrix(video, _SRGB_TO_ACESCG)
        if (self, target) == (Primaries.ACESCG, Primaries.REC709):
            return _apply_matrix(video, _ACESCG_TO_SRGB)
        raise ValueError(f"No primaries conversion from {self!r} to {target!r}.")

    def to_rec2020(self, video: Tensor) -> Tensor:
        """Linear light in this basis → Rec.2020 (HLG master space)."""
        return _apply_matrix(video, self.matrix_to_rec2020)


class HDRColorSpace(enum.Enum):
    """Source / EXR authoring colour space (matches pipelines ``--hdr``)."""

    SRGB_LINEAR = "srgb_linear"
    ACESCG = "acescg"
    ACESCCT = "acescct"

    @property
    def is_log_working(self) -> bool:
        return self is HDRColorSpace.ACESCCT

    @property
    def source_primaries(self) -> Primaries:
        if self in (HDRColorSpace.ACESCG, HDRColorSpace.ACESCCT):
            return Primaries.ACESCG
        return Primaries.REC709

    def exr_output_tags(self) -> tuple[Primaries, str]:
        """``(primaries, colorSpace tag)`` for EXR headers."""
        if self is HDRColorSpace.ACESCCT:
            return Primaries.ACESCG, "ACEScct"
        if self is HDRColorSpace.ACESCG:
            return Primaries.ACESCG, "ACEScg"
        return Primaries.REC709, "sRGB"


def _apply_matrix(video: Tensor, mat: Tensor) -> Tensor:
    """Apply 3x3 on channel-last ``[..., H, W, 3]`` or channel-first ``[..., 3, H, W]``.

    Comfy IMAGE tensors are channel-last; ``ltx_core`` often uses channel-first.
    """
    mat = mat.to(device=video.device, dtype=video.dtype)
    if video.shape[-1] == 3:
        return video @ mat.T
    if video.shape[-3] == 3:
        return torch.einsum("...chw,dc->...dhw", video, mat)
    raise ValueError(f"Expected RGB in last or -3 dim; got shape {tuple(video.shape)}")


# sRGB EOTF (IEC 61966-2-1) — same constants as ``ltx_core.hdr``.
_SRGB_A = 0.055
_SRGB_LINEAR_THRESHOLD = 0.04045
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_GAMMA = 2.4


def compress_acescct(hdr: Tensor) -> Tensor:
    """Linear HDR [0, ∞) → ACEScct [0, 1]."""
    x = torch.clamp(hdr, min=0.0)
    log_part = (torch.log2(torch.clamp(x, min=1e-12)) + _ACESCCT_LOG_B) / _ACESCCT_LOG_M
    lin_part = _ACESCCT_A_LIN * x + _ACESCCT_B_LIN
    return torch.clamp(torch.where(x > _ACESCCT_X_BRK, log_part, lin_part), 0.0, 1.0)


def decompress_acescct(ct: Tensor) -> Tensor:
    """ACEScct [0, 1] → linear ACEScg [0, ∞)."""
    ct = torch.clamp(ct, 0.0, 1.0)
    lin_from_log = torch.pow(2.0, ct * _ACESCCT_LOG_M - _ACESCCT_LOG_B)
    lin_from_lin = (ct - _ACESCCT_B_LIN) / _ACESCCT_A_LIN
    return torch.where(ct > _ACESCCT_Y_BRK, lin_from_log, lin_from_lin)


def srgb_eotf_to_linear(srgb_01: Tensor) -> Tensor:
    """sRGB-encoded [0, 1] → display-linear Rec.709/sRGB (IEC 61966-2-1 EOTF)."""
    x = torch.clamp(srgb_01.float(), 0.0, 1.0)
    return torch.where(
        x <= _SRGB_LINEAR_THRESHOLD,
        x / _SRGB_LINEAR_SLOPE,
        torch.pow((x + _SRGB_A) / (1.0 + _SRGB_A), _SRGB_GAMMA),
    )


def to_acescct_working_space(linear_rgb: Tensor, source_primaries: Primaries) -> Tensor:
    """Scene-linear RGB → ACEScct [0, 1] (channel-last or channel-first)."""
    native = source_primaries.to(Primaries.ACESCG, linear_rgb.float())
    return compress_acescct(native.clamp(min=0.0))


def acescct_to_linear(
    working: Tensor, out_primaries: Primaries = Primaries.REC709
) -> Tensor:
    """ACEScct [0, 1] → scene-linear in ``out_primaries``."""
    linear_acescg = decompress_acescct(working.float())
    return Primaries.ACESCG.to(out_primaries, linear_acescg).clamp(min=0.0)


class SDRColorSpace(enum.Enum):
    """SDR ingest colour space for mapping float RGB into ACEScct.

    Matches pipelines ``SDRColorSpace`` / ``hdr_ic_lora`` IDT:

    * ``srgb_gamma`` — display-referred sRGB (apply EOTF, then ACEScct).
    * ``srgb`` — scene-linear Rec.709 / sRGB primaries (no EOTF).
    * ``acescg`` — scene-linear ACEScg → ACEScct.
    * ``acescct`` — already ACEScct log codes; clamp passthrough.
    """

    SRGB = "srgb"
    SRGB_GAMMA = "srgb_gamma"
    ACESCG = "acescg"
    ACESCCT = "acescct"

    def to_working_space(self, video: Tensor) -> Tensor:
        """Map SDR float RGB into ACEScct ``[0, 1]`` (channel-last or channel-first)."""
        if not video.is_floating_point():
            raise ValueError(
                f"SDRColorSpace.to_working_space expects float RGB; got dtype={video.dtype}. "
                "Normalize integer pixels before calling."
            )
        video = video.float()
        if self is SDRColorSpace.ACESCCT:
            return video.clamp(0.0, 1.0)
        if self is SDRColorSpace.SRGB_GAMMA:
            return to_acescct_working_space(
                srgb_eotf_to_linear(video), Primaries.REC709
            )
        if self is SDRColorSpace.SRGB:
            return to_acescct_working_space(video, Primaries.REC709)
        if self is SDRColorSpace.ACESCG:
            return to_acescct_working_space(video, Primaries.ACESCG)
        raise ValueError(f"Unsupported SDR color space: {self!r}")


def images_to_acescct(images: Tensor, color_space: HDRColorSpace) -> Tensor:
    """Convert loaded EXR frames ``[F,H,W,3]`` into ACEScct ``[0,1]`` for the VAE."""
    images = images.float()
    if color_space.is_log_working:
        return images.clamp(0.0, 1.0)
    return to_acescct_working_space(images, color_space.source_primaries)


def sdr_images_to_acescct(images: Tensor, color_space: SDRColorSpace) -> Tensor:
    """Convert SDR frames ``[F,H,W,C]`` into ACEScct ``[0,1]`` for HDR IC-LoRA."""
    rgb = images[..., :3]
    working = color_space.to_working_space(rgb)
    if images.shape[-1] > 3:
        return torch.cat([working, images[..., 3:].float()], dim=-1)
    return working
