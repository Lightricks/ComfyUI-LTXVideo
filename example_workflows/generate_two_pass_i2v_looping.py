#!/usr/bin/env python3
"""Generate a two-pass I2V arbitrary-length workflow for LTX-2.3.

Stage 1: LTXVLoopingSampler at base resolution (~544p) with soft guiding
         images at tile boundaries for subject/scene continuity.
Stage 2: Spatial upscale (2x) → LTXVLoopingSampler refinement at high
         resolution with spatial tiling.

Run:  python generate_two_pass_i2v_looping.py
Out:  LTX-2.3_Two_Pass_I2V_Looping.json  (importable ComfyUI workflow)
"""

import json
import uuid

# ─── Workflow builder ────────────────────────────────────────────────

_link_counter = 0
_nodes: list[dict] = []
_links: list[list] = []


def _next_link_id():
    global _link_counter
    _link_counter += 1
    return _link_counter


def node(
    nid: int,
    ntype: str,
    pos: tuple[int, int],
    widgets: list | None = None,
    size: tuple[int, int] = (300, 200),
    title: str | None = None,
    color: str | None = None,
    bgcolor: str | None = None,
):
    """Register a node and return its id for wiring."""
    n = {
        "id": nid,
        "type": ntype,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": nid,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets if widgets is not None else [],
    }
    if title:
        n["title"] = title
    if color:
        n["color"] = color
    if bgcolor:
        n["bgcolor"] = bgcolor
    _nodes.append(n)
    return nid


def inp(nid: int, name: str, typ: str):
    """Declare an input slot on a node (call in slot order)."""
    for n in _nodes:
        if n["id"] == nid:
            n["inputs"].append({"name": name, "type": typ, "link": None})
            return
    raise ValueError(f"node {nid} not found")


def out(nid: int, name: str, typ: str):
    """Declare an output slot on a node (call in slot order)."""
    for n in _nodes:
        if n["id"] == nid:
            n["outputs"].append(
                {
                    "name": name,
                    "type": typ,
                    "links": [],
                    "slot_index": len(n["outputs"]),
                }
            )
            return
    raise ValueError(f"node {nid} not found")


def link(from_id: int, from_slot: int, to_id: int, to_slot: int, typ: str):
    """Wire from_id:from_slot → to_id:to_slot."""
    lid = _next_link_id()
    _links.append([lid, from_id, from_slot, to_id, to_slot, typ])
    # Update node bookkeeping
    for n in _nodes:
        if n["id"] == from_id and from_slot < len(n["outputs"]):
            n["outputs"][from_slot]["links"].append(lid)
        if n["id"] == to_id and to_slot < len(n["inputs"]):
            n["inputs"][to_slot]["link"] = lid


def build():
    return {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": max(n["id"] for n in _nodes),
        "last_link_id": _link_counter,
        "nodes": _nodes,
        "links": _links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.6, "offset": [0, 0]},
            "info": {
                "name": "LTX-2.3 Two-Pass I2V Looping",
                "description": (
                    "Two-pass I2V workflow for arbitrary-length video. "
                    "Stage 1 generates at base resolution with temporal tiling. "
                    "Stage 2 spatially upscales and refines. "
                    "Soft guiding images at tile boundaries maintain subject continuity."
                ),
            },
        },
        "version": 0.4,
    }


# ─── Layout constants ───────────────────────────────────────────────

COL_INPUT = 0
COL_MODELS = 450
COL_TEXT = 900
COL_S1_PREP = 1400
COL_S1_SAMPLE = 1950
COL_MID = 2500
COL_S2_SAMPLE = 3050
COL_OUTPUT = 3600

ROW_TOP = 0
ROW_MID = 400
ROW_BOT = 800
ROW_DEEP = 1200

# Group colors
S1_COLOR = "#335533"
S1_BG = "#223322"
S2_COLOR = "#333355"
S2_BG = "#222233"

# ─── Nodes ───────────────────────────────────────────────────────────

# ── Shared primitives ──

node(1, "LoadImage", (COL_INPUT, ROW_TOP), ["reference_image.png", "image"], (300, 300))
out(1, "IMAGE", "IMAGE")
out(1, "MASK", "MASK")

node(2, "LTXVPreprocess", (COL_INPUT, ROW_TOP + 340), [18])
inp(2, "image", "IMAGE")
out(2, "output_image", "IMAGE")
link(1, 0, 2, 0, "IMAGE")  # LoadImage → Preprocess

node(3, "PrimitiveInt", (COL_INPUT, ROW_BOT), [241, "fixed"], (200, 100),
     title="Frame Count")
out(3, "INT", "INT")

node(4, "PrimitiveFloat", (COL_INPUT, ROW_BOT + 130), [24], (200, 100),
     title="Frame Rate")
out(4, "FLOAT", "FLOAT")

node(5, "PrimitiveBoolean", (COL_INPUT, ROW_BOT + 260), [True], (200, 80),
     title="I2V Enable")
out(5, "BOOLEAN", "BOOLEAN")

# Guiding image indices — comma-separated pixel frame positions.
# Default "0" = reference at first frame only.
# For 3 tiles (241 frames, tile_size=128, overlap=24):
#   "0, 104, 208" places the guiding image at each tile boundary.
# The number of indices must match the number of guiding images.
# With a single image and "0", only frame 0 gets soft conditioning.
# Use optional_negative_index_latents for global subject anchoring.
node(6, "Note", (COL_INPUT, ROW_DEEP), [
    "## Guiding Image Indices\n\n"
    "Set `optional_cond_image_indices` on the Stage 1 Looping Sampler.\n"
    "Default: \"0\" (reference image at first frame only).\n\n"
    "For multi-tile conditioning, set indices at tile boundaries.\n"
    "With tile_size=128, overlap=24, new content starts every 104 frames:\n"
    "  \"0, 104, 208\" for a 241-frame (3-tile) clip.\n\n"
    "The number of images in the guiding batch must match the indices.\n"
    "Use LatentBatch or ImageBatch to provide multiple images.\n"
    "By default, a single reference image at index 0 is used."
], (400, 280))

# ── Model loading ──

node(10, "CheckpointLoaderSimple", (COL_MODELS, ROW_TOP),
     ["ltx-2.3-22b-dev.safetensors"], (350, 150))
out(10, "MODEL", "MODEL")
out(10, "CLIP", "CLIP")
out(10, "VAE", "VAE")

node(11, "LTXAVTextEncoderLoader", (COL_MODELS, ROW_TOP + 180),
     ["comfy_gemma_3_12B_it.safetensors", "ltx-2.3-22b-dev.safetensors", "default"],
     (380, 130))
out(11, "CLIP", "CLIP")

node(12, "LTXVAudioVAELoader", (COL_MODELS, ROW_MID),
     ["ltx-2.3-22b-dev.safetensors"], (350, 100))
out(12, "Audio VAE", "VAE")

node(13, "LoraLoaderModelOnly", (COL_MODELS, ROW_MID + 130),
     ["ltx-2.3-22b-distilled-lora-384.safetensors", 0.5], (380, 100),
     title="Distilled LoRA (both stages)")
inp(13, "model", "MODEL")
out(13, "MODEL", "MODEL")
link(10, 0, 13, 0, "MODEL")  # Checkpoint → LoRA

node(14, "LatentUpscaleModelLoader", (COL_MODELS, ROW_MID + 260),
     ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"], (380, 100))
out(14, "LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")

# ── Text encoding ──

node(20, "CLIPTextEncode", (COL_TEXT, ROW_TOP),
     ["A woman walks through a sunlit meadow. Warm breeze rustles the tall grass. "
      "Birds sing in the distance. She pauses to admire wildflowers."],
     (400, 180), title="Positive Prompt")
inp(20, "clip", "CLIP")
out(20, "CONDITIONING", "CONDITIONING")
link(11, 0, 20, 0, "CLIP")

node(21, "CLIPTextEncode", (COL_TEXT, ROW_TOP + 220),
     ["pc game, console game, video game, cartoon, childish, ugly, blurry"],
     (400, 120), title="Negative Prompt")
inp(21, "clip", "CLIP")
out(21, "CONDITIONING", "CONDITIONING")
link(11, 0, 21, 0, "CLIP")

node(22, "LTXVConditioning", (COL_TEXT, ROW_MID), [24], (300, 120))
inp(22, "positive", "CONDITIONING")
inp(22, "negative", "CONDITIONING")
inp(22, "frame_rate", "FLOAT")
out(22, "positive", "CONDITIONING")
out(22, "negative", "CONDITIONING")
link(20, 0, 22, 0, "CONDITIONING")
link(21, 0, 22, 1, "CONDITIONING")
link(4, 0, 22, 2, "FLOAT")

# ── Resize reference image (for both stages) ──

node(23, "ResizeImageMaskNode", (COL_INPUT + 340, ROW_TOP + 340),
     ["scale longer dimension", 1536, "lanczos"], (300, 120),
     title="Resize Reference")
inp(23, "input", "IMAGE,MASK")
out(23, "resized", "IMAGE")
link(1, 0, 23, 0, "IMAGE")  # Original image → resize

# ── Stage 1 prep ──

node(30, "EmptyLTXVLatentVideo", (COL_S1_PREP, ROW_TOP), [960, 544, 241, 1],
     (250, 150), title="Stage 1 Empty Latent")
inp(30, "length", "INT")
out(30, "LATENT", "LATENT")
link(3, 0, 30, 0, "INT")  # Frame count

node(31, "LTXVEmptyLatentAudio", (COL_S1_PREP, ROW_TOP + 180), [97, 25, 1],
     (250, 130))
inp(31, "audio_vae", "VAE")
inp(31, "frames_number", "INT")
inp(31, "frame_rate", "INT")
out(31, "Latent", "LATENT")
link(12, 0, 31, 0, "VAE")   # Audio VAE
link(3, 0, 31, 1, "INT")    # Frame count

node(34, "CM_FloatToInt", (COL_S1_PREP - 100, ROW_MID + 60), [0], (150, 80),
     title="FPS→Int")
inp(34, "a", "FLOAT")
out(34, "INT", "INT")
link(4, 0, 34, 0, "FLOAT")
link(34, 0, 31, 2, "INT")   # Frame rate int → audio

node(32, "LTXVImgToVideoConditionOnly", (COL_S1_PREP, ROW_MID),
     [0.7, False], (300, 130), title="Stage 1 I2V Cond",
     color=S1_COLOR, bgcolor=S1_BG)
inp(32, "vae", "VAE")
inp(32, "image", "IMAGE")
inp(32, "latent", "LATENT")
inp(32, "bypass", "BOOLEAN")
out(32, "latent", "LATENT")
link(10, 2, 32, 0, "VAE")   # Checkpoint VAE
link(2, 0, 32, 1, "IMAGE")  # Preprocessed reference
link(30, 0, 32, 2, "LATENT")  # Empty latent
link(5, 0, 32, 3, "BOOLEAN")  # I2V enable

node(33, "LTXVConcatAVLatent", (COL_S1_PREP, ROW_BOT), [],
     (250, 100), title="Stage 1 AV Concat",
     color=S1_COLOR, bgcolor=S1_BG)
inp(33, "video_latent", "LATENT")
inp(33, "audio_latent", "LATENT")
out(33, "latent", "LATENT")
link(32, 0, 33, 0, "LATENT")  # Conditioned video latent
link(31, 0, 33, 1, "LATENT")  # Empty audio latent

# VAE-encode reference for negative_index_latents (global subject anchor)
node(35, "VAEEncode", (COL_S1_PREP, ROW_DEEP), [], (250, 100),
     title="Encode Reference Latent")
inp(35, "pixels", "IMAGE")
inp(35, "vae", "VAE")
out(35, "LATENT", "LATENT")
link(2, 0, 35, 0, "IMAGE")   # Preprocessed reference
link(10, 2, 35, 1, "VAE")    # Checkpoint VAE

# ── Stage 1 sampling ──

node(40, "RandomNoise", (COL_S1_SAMPLE, ROW_TOP - 80), [42, "fixed"],
     (200, 100), title="Stage 1 Noise")
out(40, "NOISE", "NOISE")

node(41, "KSamplerSelect", (COL_S1_SAMPLE, ROW_TOP + 40),
     ["euler_ancestral_cfg_pp"], (250, 80), title="Stage 1 Sampler")
out(41, "SAMPLER", "SAMPLER")

node(42, "ManualSigmas", (COL_S1_SAMPLE, ROW_TOP + 140),
     ["1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"],
     (350, 80), title="Stage 1 Sigmas")
out(42, "SIGMAS", "SIGMAS")

node(43, "CFGGuider", (COL_S1_SAMPLE, ROW_TOP + 240), [1], (250, 130),
     title="Stage 1 Guider", color=S1_COLOR, bgcolor=S1_BG)
inp(43, "model", "MODEL")
inp(43, "positive", "CONDITIONING")
inp(43, "negative", "CONDITIONING")
out(43, "GUIDER", "GUIDER")
link(13, 0, 43, 0, "MODEL")        # Model with distilled LoRA
link(22, 0, 43, 1, "CONDITIONING")  # Positive
link(22, 1, 43, 2, "CONDITIONING")  # Negative

# LTXVLoopingSampler — Stage 1
# Widgets: temporal_tile_size, temporal_overlap, guiding_strength,
#          temporal_overlap_cond_strength, cond_image_strength,
#          horizontal_tiles, vertical_tiles, spatial_overlap,
#          adain_factor, guiding_start_step, guiding_end_step,
#          optional_cond_image_indices
node(44, "LTXVLoopingSampler", (COL_S1_SAMPLE, ROW_MID),
     [128, 24, 1.0, 0.5, 1.0, 1, 1, 1, 0.15, 0, 1000, "0"],
     (400, 580), title="Stage 1 — Generate",
     color=S1_COLOR, bgcolor=S1_BG)
# Required inputs (slots 0-6)
inp(44, "model", "MODEL")
inp(44, "vae", "VAE")
inp(44, "noise", "NOISE")
inp(44, "sampler", "SAMPLER")
inp(44, "sigmas", "SIGMAS")
inp(44, "guider", "GUIDER")
inp(44, "latents", "LATENT")
# Optional inputs (slots 7-11)
inp(44, "optional_cond_images", "IMAGE")
inp(44, "optional_guiding_latents", "LATENT")
inp(44, "optional_positive_conditionings", "CONDITIONING")
inp(44, "optional_negative_index_latents", "LATENT")
inp(44, "optional_normalizing_latents", "LATENT")
out(44, "denoised_output", "LATENT")

link(13, 0, 44, 0, "MODEL")   # Model with distilled LoRA
link(10, 2, 44, 1, "VAE")     # Checkpoint VAE
link(40, 0, 44, 2, "NOISE")   # Noise
link(41, 0, 44, 3, "SAMPLER") # Sampler
link(42, 0, 44, 4, "SIGMAS")  # Sigmas
link(43, 0, 44, 5, "GUIDER")  # Guider
link(33, 0, 44, 6, "LATENT")  # AV latent (video + audio)
link(2, 0, 44, 7, "IMAGE")    # Guiding images (preprocessed reference)
# slot 8: optional_guiding_latents — not connected (no IC-LoRA guide)
# slot 9: optional_positive_conditionings — not connected (single prompt)
link(35, 0, 44, 10, "LATENT") # Negative index latents (global subject anchor)
# slot 11: optional_normalizing_latents — not connected

# ── Between stages ──
# Single split of stage 1 AV output:
#   video → upscaler → stage 2 looping sampler
#   audio → directly to final audio decode (bypasses stage 2)

node(50, "LTXVSeparateAVLatent", (COL_MID, ROW_MID), [], (250, 100),
     title="Split Stage 1 AV")
inp(50, "av_latent", "LATENT")
out(50, "video_latent", "LATENT")
out(50, "audio_latent", "LATENT")
link(44, 0, 50, 0, "LATENT")  # Stage 1 output → split

node(51, "LTXVLatentUpsampler", (COL_MID, ROW_MID + 130), [], (300, 100),
     title="Spatial Upscale 2x")
inp(51, "samples", "LATENT")
inp(51, "upscale_model", "LATENT_UPSCALE_MODEL")
inp(51, "vae", "VAE")
out(51, "LATENT", "LATENT")
link(50, 0, 51, 0, "LATENT")              # Video latent only
link(14, 0, 51, 1, "LATENT_UPSCALE_MODEL")  # Upscale model
link(10, 2, 51, 2, "VAE")                   # VAE

node(52, "LTXVImgToVideoConditionOnly", (COL_MID, ROW_MID + 260),
     [1.0, False], (300, 130), title="Stage 2 I2V Cond",
     color=S2_COLOR, bgcolor=S2_BG)
inp(52, "vae", "VAE")
inp(52, "image", "IMAGE")
inp(52, "latent", "LATENT")
inp(52, "bypass", "BOOLEAN")
out(52, "latent", "LATENT")
link(10, 2, 52, 0, "VAE")     # VAE
link(23, 0, 52, 1, "IMAGE")   # Resized reference (full res for stage 2)
link(51, 0, 52, 2, "LATENT")  # Upscaled video latent
link(5, 0, 52, 3, "BOOLEAN")  # I2V enable

# Stage 2 receives AV latent (upscaled video + stage 1 audio).
# The looping sampler preserves input audio data for refinement:
# base tile uses the corresponding input audio slice, extend tiles
# pass source audio for new-frame initialization via _audio_new_init.
node(53, "LTXVConcatAVLatent", (COL_MID, ROW_BOT + 200), [], (250, 100),
     title="Stage 2 AV Concat", color=S2_COLOR, bgcolor=S2_BG)
inp(53, "video_latent", "LATENT")
inp(53, "audio_latent", "LATENT")
out(53, "latent", "LATENT")
link(52, 0, 53, 0, "LATENT")  # Conditioned upscaled video
link(50, 1, 53, 1, "LATENT")  # Audio from stage 1

# ── Stage 2 sampling ──

node(60, "RandomNoise", (COL_S2_SAMPLE, ROW_TOP - 80), [43, "fixed"],
     (200, 100), title="Stage 2 Noise")
out(60, "NOISE", "NOISE")

node(61, "KSamplerSelect", (COL_S2_SAMPLE, ROW_TOP + 40),
     ["euler_cfg_pp"], (250, 80), title="Stage 2 Sampler")
out(61, "SAMPLER", "SAMPLER")

node(62, "ManualSigmas", (COL_S2_SAMPLE, ROW_TOP + 140),
     ["0.85, 0.7250, 0.4219, 0.0"], (300, 80), title="Stage 2 Sigmas")
out(62, "SIGMAS", "SIGMAS")

node(63, "CFGGuider", (COL_S2_SAMPLE, ROW_TOP + 240), [1], (250, 130),
     title="Stage 2 Guider", color=S2_COLOR, bgcolor=S2_BG)
inp(63, "model", "MODEL")
inp(63, "positive", "CONDITIONING")
inp(63, "negative", "CONDITIONING")
out(63, "GUIDER", "GUIDER")
link(13, 0, 63, 0, "MODEL")        # Same model with distilled LoRA
link(22, 0, 63, 1, "CONDITIONING")  # Same positive
link(22, 1, 63, 2, "CONDITIONING")  # Same negative

# LTXVLoopingSampler — Stage 2
# spatial tiling 2x1 for upscaled resolution
node(64, "LTXVLoopingSampler", (COL_S2_SAMPLE, ROW_MID),
     [128, 24, 1.0, 0.5, 1.0, 2, 1, 1, 0.0, 0, 1000, "0"],
     (400, 580), title="Stage 2 — Refine",
     color=S2_COLOR, bgcolor=S2_BG)
inp(64, "model", "MODEL")
inp(64, "vae", "VAE")
inp(64, "noise", "NOISE")
inp(64, "sampler", "SAMPLER")
inp(64, "sigmas", "SIGMAS")
inp(64, "guider", "GUIDER")
inp(64, "latents", "LATENT")
inp(64, "optional_cond_images", "IMAGE")
inp(64, "optional_guiding_latents", "LATENT")
inp(64, "optional_positive_conditionings", "CONDITIONING")
inp(64, "optional_negative_index_latents", "LATENT")
inp(64, "optional_normalizing_latents", "LATENT")
out(64, "denoised_output", "LATENT")

link(13, 0, 64, 0, "MODEL")   # Model with distilled LoRA
link(10, 2, 64, 1, "VAE")     # VAE
link(60, 0, 64, 2, "NOISE")   # Noise
link(61, 0, 64, 3, "SAMPLER") # Sampler
link(62, 0, 64, 4, "SIGMAS")  # Sigmas
link(63, 0, 64, 5, "GUIDER")  # Guider
link(53, 0, 64, 6, "LATENT")  # Stage 2 AV latent (upscaled video + stage 1 audio)
link(23, 0, 64, 7, "IMAGE")   # Guiding images (resized reference)
# slot 8-11: not connected for stage 2

# ── Output ──
# Both video and audio from stage 2 (refined jointly).

node(70, "LTXVSeparateAVLatent", (COL_OUTPUT, ROW_MID), [], (250, 100),
     title="Split Final AV")
inp(70, "av_latent", "LATENT")
out(70, "video_latent", "LATENT")
out(70, "audio_latent", "LATENT")
link(64, 0, 70, 0, "LATENT")  # Stage 2 AV output

node(71, "LTXVSpatioTemporalTiledVAEDecode", (COL_OUTPUT, ROW_MID + 130),
     [6, 4, 16, 4, False, "auto", "auto"], (350, 200),
     title="Decode Video (Tiled)")
inp(71, "samples", "LATENT")
inp(71, "vae", "VAE")
out(71, "IMAGE", "IMAGE")
link(70, 0, 71, 0, "LATENT")  # Refined video
link(10, 2, 71, 1, "VAE")     # VAE

node(72, "LTXVAudioVAEDecode", (COL_OUTPUT, ROW_MID + 360), [], (250, 100))
inp(72, "samples", "LATENT")
inp(72, "audio_vae", "VAE")
out(72, "Audio", "AUDIO")
link(70, 1, 72, 0, "LATENT")  # Refined audio
link(12, 0, 72, 1, "VAE")     # Audio VAE

node(73, "CreateVideo", (COL_OUTPUT, ROW_BOT + 200), [30], (250, 100))
inp(73, "images", "IMAGE")
inp(73, "audio", "AUDIO")
inp(73, "fps", "FLOAT")
out(73, "VIDEO", "VIDEO")
link(71, 0, 73, 0, "IMAGE")
link(72, 0, 73, 1, "AUDIO")
link(4, 0, 73, 2, "FLOAT")   # Frame rate

node(74, "SaveVideo", (COL_OUTPUT, ROW_DEEP), ["LTX-2.3/Looping", "auto", "auto"],
     (250, 100))
inp(74, "video", "VIDEO")
link(73, 0, 74, 0, "VIDEO")


# ─── Generate ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    wf = build()
    out_path = os.path.join(os.path.dirname(__file__), "LTX-2.3_Two_Pass_I2V_Looping.json")
    with open(out_path, "w") as f:
        json.dump(wf, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  {len(_nodes)} nodes, {len(_links)} links")
