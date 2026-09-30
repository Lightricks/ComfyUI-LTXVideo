"""ComfyUI HDR nodes: ingest, decode postprocess, and delivery.

Stage-oriented layout (shared ACEScct working space):

- Ingest: ``LTXVSDRToHDRWorkingSpace``, ``LTXVLoadEXRSequence``
- Runtime: ``LTXVVAEForceFloat32``
- Delivery: ``LTXVHDRDecodePostprocess``, ``LTXVSaveHLG``

Color math: ``hdr_color``. File I/O helpers: ``hdr_io``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import folder_paths
import torch
from torch import Tensor

from .hdr_color import (
    HDRColorSpace,
    Primaries,
    SDRColorSpace,
    compress_acescct,
    images_to_acescct,
    sdr_images_to_acescct,
)
from .hdr_io import (
    _encode_hlg_mp4,
    _list_exr_files,
    _read_exr,
    _resolve_path,
    _save_exr_frame,
    _silent_audio,
    _trim_to_8k1,
)
from .nodes_registry import comfy_node

logger = logging.getLogger("LTXVideo.hdr_nodes")

_EXR_COLOR_SPACE_CHOICES = ["linear", "acescct", "srgb_linear", "acescg"]
_SDR_COLOR_SPACE_CHOICES = [c.value for c in SDRColorSpace]
_COLOR_SPACE_CHOICES = [c.value for c in HDRColorSpace]


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


def _hdr_decompress(decoded_01: Tensor, transfer: str = "acescct") -> Tensor:
    """Decompress VAE-decoded ``[0,1]`` image to linear HDR ``[0, inf)``.

    ComfyUI's VAE decode returns images in ``[0, 1]`` via ``(raw + 1) / 2``.

    * ``acescct`` — ACEScct working codes (SDR→HDR IC-LoRA + native HDR).
    * ``logc3`` — legacy LogC3 SDR→HDR LoRA only.
    """
    transfer = (transfer or "acescct").lower()
    if transfer == "acescct":
        from .hdr_color import decompress_acescct

        # Decoded IMAGE is already ACEScct working-space [0, 1].
        return decompress_acescct(decoded_01.float())
    if transfer != "logc3":
        raise ValueError(
            f"Unsupported HDR transfer {transfer!r}; use 'logc3' or 'acescct'."
        )
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
# ComfyUI Nodes
# ---------------------------------------------------------------------------


@comfy_node(name="LTXVSDRToHDRWorkingSpace")
class LTXVSDRToHDRWorkingSpace:
    """Map SDR float RGB into ACEScct ``[0, 1]`` for HDR IC-LoRA conditioning.
    Place **before** resize when the source is display sRGB so
    the EOTF is not applied to bilinear-filtered codes.

    Default ``srgb_gamma`` matches typical ``LoadVideo`` / MP4 decode.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_space": (
                    _SDR_COLOR_SPACE_CHOICES,
                    {
                        "default": "srgb_gamma",
                        "tooltip": (
                            "SDR ingest IDT. srgb_gamma = display sRGB (EOTF then "
                            "ACEScct); srgb = scene-linear Rec.709; acescg / acescct "
                            "for float EXR-style plates."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("acescct",)
    FUNCTION = "convert"
    CATEGORY = "Lightricks/HDR"
    DESCRIPTION = (
        "SDR → ACEScct working space for HDR IC-LoRA (pipelines SDRColorSpace). "
        "Wire LoadVideo images through this before resize / IC-LoRA guide."
    )

    def convert(self, image: torch.Tensor, color_space: str = "srgb_gamma") -> tuple:
        space = SDRColorSpace(color_space)
        return (sdr_images_to_acescct(image, space),)


def _exr_payload_for_color_space(
    image: Tensor,
    hdr_linear: Tensor,
    transfer: str,
    exr_color_space: str,
) -> tuple[Tensor, Primaries | None, str | None]:
    """Build EXR RGB frames and optional primaries / colorSpace tags.

    ``linear`` keeps the legacy 2.3 behaviour (raw decompressed buffer, no tags).
    Other values match pipelines ``--hdr`` EXR authoring.
    """
    space = (exr_color_space or "linear").lower()
    transfer = (transfer or "acescct").lower()
    rgb_in = image[..., :3].float()
    hdr = hdr_linear[..., :3].float()

    if space == "linear":
        return hdr, None, None

    if space == "acescct":
        if transfer == "acescct":
            frames = rgb_in.clamp(0.0, 1.0)
        else:
            # LogC3 linear is treated as Rec.709 scene-linear for ACEScct authoring.
            frames = compress_acescct(
                Primaries.REC709.to(Primaries.ACESCG, hdr).clamp(min=0.0)
            )
        return frames, Primaries.ACESCG, "ACEScct"

    if space == "acescg":
        if transfer == "acescct":
            frames = hdr  # decompress_acescct → ACEScg linear
        else:
            frames = Primaries.REC709.to(Primaries.ACESCG, hdr).clamp(min=0.0)
        return frames, Primaries.ACESCG, "ACEScg"

    if space == "srgb_linear":
        if transfer == "acescct":
            frames = Primaries.ACESCG.to(Primaries.REC709, hdr).clamp(min=0.0)
        else:
            frames = hdr.clamp(min=0.0)
        return frames, Primaries.REC709, "sRGB"

    raise ValueError(
        f"Unsupported exr_color_space {exr_color_space!r}; "
        f"use one of {_EXR_COLOR_SPACE_CHOICES}."
    )


@comfy_node(name="LTXVHDRDecodePostprocess")
class LTXVHDRDecodePostprocess:
    """Decompress HDR from VAE output, tonemap for preview, optionally save EXR.

    Place after VAE Decode. Recovers linear HDR from the VAE ``[0,1]`` image and
    tonemaps to SDR for display. When ``save_exr`` is enabled, writes an EXR
    sequence (legacy ``linear``, or tagged ``acescct`` / ``srgb_linear`` /
    ``acescg``).

    For BT.2020/HLG masters, wire ``hdr_linear`` into ``LTXVSaveHLG``.

    Outputs:
        tonemapped: SDR preview [0, 1] after Reinhard tonemap + sRGB gamma.
        hdr_linear: Raw linear HDR values [0, inf) for further processing.
            With ``transfer=acescct`` this is ACEScg scene-linear; with
            ``logc3`` it is LogC3/Rec.709-style scene-linear.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "transfer": (
                    ["acescct", "logc3"],
                    {
                        "default": "acescct",
                        "tooltip": (
                            "Working-space curve of the VAE decode. "
                            "Use acescct for SDR→HDR IC-LoRA (after "
                            "LTXVSDRToHDRWorkingSpace) and for native HDR EXR; "
                            "logc3 only for legacy LogC3 HDR LoRA checkpoints."
                        ),
                    },
                ),
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
                            "+1 = 2x brighter, -1 = half brightness. "
                            "Affects the tonemapped preview only."
                        ),
                    },
                ),
                "save_exr": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Save HDR frames as an EXR sequence.",
                    },
                ),
                "exr_color_space": (
                    _EXR_COLOR_SPACE_CHOICES,
                    {
                        "default": "acescct",
                        "tooltip": (
                            "EXR contents when save_exr is on. "
                            "Default 'acescct' matches pipelines hdr_ic_lora "
                            "ACEScct log masters. 'acescg' / 'srgb_linear' are "
                            "tagged linear; 'linear' is legacy raw decompress "
                            "(untagged). All paths use OpenImageIO."
                        ),
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
        "Decompresses VAE-decoded HDR (ACEScct or legacy LogC3), tonemaps for "
        "preview, and optionally writes EXR (linear / ACEScct / sRGB linear / "
        "ACEScg). Wire hdr_linear into LTXVSaveHLG for a BT.2020/HLG master."
    )

    def postprocess(
        self,
        image: torch.Tensor,
        transfer: str = "acescct",
        exposure: float = 0.0,
        save_exr: bool = False,
        exr_color_space: str = "acescct",
        output_dir: str = "output/hdr_exr",
        filename_prefix: str = "frame",
        half_precision: bool = True,
    ) -> tuple:

        hdr = _hdr_decompress(image, transfer=transfer)
        hdr = torch.clamp(hdr, min=0.0, max=1e4)

        # Reinhard tonemap with exposure (stops → linear multiplier)
        exposure_mult = 2.0**exposure
        hdr_exposed = hdr * exposure_mult
        tonemapped_linear = (hdr_exposed / (1.0 + hdr_exposed)).clamp(0.0, 1.0)
        tonemapped = _linear_to_srgb(tonemapped_linear)

        if save_exr:
            frames, primaries, tag = _exr_payload_for_color_space(
                image, hdr, transfer, exr_color_space
            )
            self._save_exr_frames_tagged(
                frames,
                output_dir,
                filename_prefix,
                half_precision,
                primaries=primaries,
                color_space_tag=tag,
            )

        return (tonemapped, hdr)

    @staticmethod
    def _resolve_exr_dir(output_dir: str) -> str:
        import folder_paths

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(folder_paths.get_output_directory(), output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @classmethod
    def _save_exr_frames_tagged(
        cls,
        frames: torch.Tensor,
        output_dir: str,
        filename_prefix: str,
        half_precision: bool,
        *,
        primaries: Primaries | None,
        color_space_tag: str | None,
    ) -> None:
        output_dir = cls._resolve_exr_dir(output_dir)
        for i, frame in enumerate(frames):
            _save_exr_frame(
                frame,
                Path(output_dir) / f"{filename_prefix}_{i:05d}.exr",
                half_precision=half_precision,
                primaries=primaries,
                color_space_tag=color_space_tag,
            )
        logger.info(
            "Saved %d EXR frame(s) to %s (%s%s)",
            frames.shape[0],
            output_dir,
            "float16" if half_precision else "float32",
            f", {color_space_tag}" if color_space_tag else ", untagged linear",
        )


@comfy_node(name="LTXVLoadEXRSequence")
class LTXVLoadEXRSequence:
    """Load an EXR still or folder and convert to ACEScct ``[0,1]`` IMAGE for the VAE.

    Declares the source colour space the same way as pipelines ``--hdr``:
    ``srgb_linear`` / ``acescg`` (compress to ACEScct) or ``acescct`` (passthrough).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to a .exr still or a directory of *.exr frames. "
                            "I2V: single still is fine. V2V: use a folder with 8k+1 frames. "
                            "Relative paths are resolved under ComfyUI's input folder."
                        ),
                    },
                ),
                "color_space": (_COLOR_SPACE_CHOICES, {"default": "srgb_linear"}),
                "frame_rate": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": "Required for EXR folders (no container fps).",
                    },
                ),
            },
            "optional": {
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "frame_cap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "tooltip": "Max frames to load (0 = all). Still trimmed to 8k+1.",
                    },
                ),
                "trim_to_8k1": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Trim to longest 8k+1 prefix required by the video VAE.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("images", "frame_rate", "width", "height", "frame_count", "audio")
    FUNCTION = "load"
    CATEGORY = "Lightricks/HDR"
    DESCRIPTION = (
        "Native HDR input: load EXR → ACEScct [0,1] frames for IC-LoRA / VAE encode. "
        "Also emits silent AUDIO matching duration so V2V graphs that freeze audio still wire. "
        "Native EXR path only (not SDR→HDR IC-LoRA ingest). "
        "Downstream: set LTXVPreprocess img_compression=0 — Python --hdr EXR conditioning "
        "skips JPEG/CRF; compressing ACEScct codes darkens HDR highlights."
    )

    def load(
        self,
        path: str,
        color_space: str,
        frame_rate: float,
        frame_start: int = 0,
        frame_cap: int = 0,
        trim_to_8k1: bool = True,
    ):
        root = _resolve_path(path)
        files = _list_exr_files(root)
        files = files[frame_start:]
        if frame_cap > 0:
            files = files[:frame_cap]
        if not files:
            raise ValueError("No EXR frames left after frame_start / frame_cap.")

        cs = HDRColorSpace(color_space)
        frames = [_read_exr(f) for f in files]
        shapes = {tuple(f.shape) for f in frames}
        if len(shapes) != 1:
            raise ValueError(
                "EXR frames must share the same H×W×C; got "
                + ", ".join(str(s) for s in sorted(shapes))
            )
        images = torch.stack(frames, dim=0)
        images = images_to_acescct(images, cs)
        if trim_to_8k1:
            images = _trim_to_8k1(images)

        h, w = int(images.shape[1]), int(images.shape[2])
        n = int(images.shape[0])
        if n < 1:
            raise ValueError(f"No EXR frames loaded from {root}")
        audio = _silent_audio(n, float(frame_rate))
        logger.info(
            "Loaded %d EXR frame(s) from %s as %s → ACEScct (%dx%d @ %.3f fps); "
            "silent audio %d samples @ %d Hz",
            n,
            root if root.is_dir() else (root.parent if root.is_file() else root),
            cs.value,
            w,
            h,
            frame_rate,
            int(audio["waveform"].shape[-1]),
            int(audio["sample_rate"]),
        )
        return (images, float(frame_rate), w, h, n, audio)


@comfy_node(name="LTXVVAEForceFloat32")
class LTXVVAEForceFloat32:
    """Force the video VAE to encode/decode in float32 (native HDR requirement).

    This node does **not** clone the VAE into a separate float32 copy for the
    output. It mutates the same object that was passed in (``vae_dtype`` and
    ``first_stage_model`` weights). The output socket is that same instance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "Lightricks/HDR"
    DESCRIPTION = (
        "Forces float32 on the video VAE (vae_dtype + first_stage_model) for "
        "native HDR. IMPORTANT: this does NOT clone a float32 VAE for the "
        "output — it mutates the input object in place and returns the same "
        "instance. If you also wire that loader's VAE (or this output) into "
        "another branch, those nodes share the mutated model: they may see "
        "float16 or float32 depending on execution order (a race with the "
        "original dtype). Prefer a single chain VAELoader → this node → all "
        "encode/decode / IC-LoRA guide uses."
    )

    def patch(self, vae):
        # In-place: Comfy Module.to(dtype) mutates weights; a wrapper copy would
        # still share first_stage_model and waste peak memory for no isolation.
        vae.vae_dtype = torch.float32
        model = getattr(vae, "first_stage_model", None)
        if model is not None:
            try:
                model.to(dtype=torch.float32)
            except Exception as exc:
                raise RuntimeError(
                    "LTXVVAEForceFloat32 could not cast VAE weights to float32 "
                    f"({type(exc).__name__}: {exc}). Native HDR requires float32."
                ) from exc
        logger.info("VAE dtype forced to float32 for native HDR")
        return (vae,)


@comfy_node(name="LTXVSaveHLG")
class LTXVSaveHLG:
    """Encode scene-linear ``hdr_linear`` frames to a BT.2020 / HLG 10-bit HEVC master.

    Wire the ``hdr_linear`` output of ``LTXVHDRDecodePostprocess`` here (not the
    tonemapped preview). Matches pipelines ``encode_video`` HLG delivery.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_linear": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Scene-linear HDR from LTXVHDRDecodePostprocess. "
                            "With transfer=acescct this is ACEScg linear — leave "
                            "linear_primaries=acescg. For logc3, use rec709."
                        ),
                    },
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001},
                ),
                "filename_prefix": ("STRING", {"default": "hdr/ltxv_hlg"}),
            },
            "optional": {
                "linear_primaries": (
                    ["acescg", "rec709"],
                    {
                        "default": "acescg",
                        "tooltip": (
                            "Primaries of hdr_linear. ACEScct decode "
                            "(SDR→HDR IC-LoRA or native HDR) → acescg. "
                            "Legacy LogC3 → rec709."
                        ),
                    },
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional track to mux into the HLG MP4 (e.g. Decode AUDIO). "
                            "Omit for video-only."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Lightricks/HDR"
    DESCRIPTION = (
        "Writes a BT.2020/HLG 10-bit HEVC master from scene-linear hdr_linear "
        "(from LTXVHDRDecodePostprocess). Maps linear_primaries → Rec.2020 "
        "directly (no Rec.709 hop), then HLG. Optional AUDIO is muxed as AAC."
    )

    def save(
        self,
        hdr_linear: Tensor,
        frame_rate: float,
        filename_prefix: str,
        linear_primaries: str = "acescg",
        audio: dict | None = None,
    ):
        if hdr_linear.ndim != 4 or hdr_linear.shape[-1] < 3:
            raise ValueError(f"Expected IMAGE [F,H,W,C]; got {tuple(hdr_linear.shape)}")

        # ACEScg/Rec.709 → Rec.2020 inside HLG encode (Primaries.to_rec2020 /
        # matrix_to_rec2020). No Rec.709 intermediate that would crush OOG early.
        linear = hdr_linear[..., :3].float()
        prim = (linear_primaries or "acescg").lower()
        if prim == "acescg":
            src_primaries = Primaries.ACESCG
        elif prim == "rec709":
            src_primaries = Primaries.REC709
        else:
            raise ValueError(
                f"Unsupported linear_primaries {linear_primaries!r}; use acescg or rec709."
            )

        # Even dims for 4:2:0
        h, w = linear.shape[1], linear.shape[2]
        if h % 2 or w % 2:
            linear = linear[:, : h - (h % 2), : w - (w % 2), :]

        full_output_folder, filename, counter, _subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                linear.shape[2],
                linear.shape[1],
            )
        )
        hlg_path = Path(full_output_folder) / f"{filename}_{counter:05d}_hlg.mp4"
        _encode_hlg_mp4(
            linear,
            hlg_path,
            fps=float(frame_rate),
            audio=audio,
            primaries=src_primaries,
        )
        logger.info(
            "Wrote HLG 10-bit master to %s%s",
            hlg_path,
            " (with audio)" if audio is not None else "",
        )
        return {}
