#!/usr/bin/env python3
"""Generate a two-pass AV I2V looping workflow for LTX-2.3.

Stage 1: LTXVLoopingSampler at base resolution (~544p) with soft guiding
         images near tile ends for subject/scene continuity.
Stage 2: Spatial upscale (2x) → LTXVLoopingSampler refinement at high
         resolution with spatial tiling.

Run:  python generate_two_pass_i2v_looping.py
Out:  LTX-2.3_Two_Pass_I2V_Looping.json  (importable ComfyUI workflow)
"""

import math
import json
import uuid

# These defaults seed the editable workflow math nodes. The generated graph
# derives frame count, tile size, and late-reference indices at runtime.
TIME_SCALE = 8
DEFAULT_FRAME_RATE = 24
DEFAULT_TOTAL_DURATION = 30.0
DEFAULT_TILE_DURATION = 10.0
DEFAULT_OVERLAP_DURATION = 80 / DEFAULT_FRAME_RATE
DEFAULT_LATE_REFERENCE_OFFSET = 16 / DEFAULT_FRAME_RATE
DEFAULT_FINAL_HEIGHT = 1088

GLOBAL_PROMPT = (
    "A cinematic live-action scene with the same subject, wardrobe, lighting, "
    "and location throughout. Natural motion, stable anatomy, coherent audio."
)
TILE_SNIPPETS = [
    "The subject enters the shot and begins the action.",
    "The action continues with a small camera move.",
    "The subject completes the central beat.",
    "The motion settles into the ending pose.",
]


def aligned_frames(seconds: float, frame_rate: float) -> int:
    """Return a positive frame count rounded to the nearest 8-frame block."""
    return max(TIME_SCALE, round(seconds * frame_rate / TIME_SCALE) * TIME_SCALE)


def frame_count_for_duration(seconds: float, frame_rate: float) -> int:
    """Return the largest valid 8n+1 clip length within the duration."""
    return max(
        TIME_SCALE + 1,
        math.floor((seconds * frame_rate - 1) / TIME_SCALE) * TIME_SCALE + 1,
    )


def temporal_tile_starts(frame_count: int, tile_size: int, overlap: int) -> list[int]:
    """Return temporal tile starts in pixel-frame units."""
    if frame_count % TIME_SCALE != 1:
        raise ValueError("frame_count must satisfy 8n+1")
    if tile_size <= overlap:
        raise ValueError("tile_size must be greater than overlap")
    if tile_size % TIME_SCALE or overlap % TIME_SCALE:
        raise ValueError("tile_size and overlap must be multiples of 8")

    latent_frames = ((frame_count - 1) // TIME_SCALE) + 1
    latent_tile_size = tile_size // TIME_SCALE
    latent_overlap = overlap // TIME_SCALE
    latent_stride = latent_tile_size - latent_overlap
    tile_count = math.ceil((latent_frames - latent_overlap) / latent_stride)
    return [tile_index * (tile_size - overlap) for tile_index in range(tile_count)]


def late_reference_indices(
    frame_count: int,
    tile_size: int,
    overlap: int,
    margin: int,
) -> list[int]:
    """Return frame 0 plus one aligned late reference index per temporal tile."""
    if margin < TIME_SCALE or margin >= tile_size or margin % TIME_SCALE:
        raise ValueError("late reference margin must be an 8-aligned tile offset")

    final_aligned_index = ((frame_count - 1) // TIME_SCALE) * TIME_SCALE
    indices = [0]
    for tile_start in temporal_tile_starts(frame_count, tile_size, overlap):
        late_index = min(tile_start + tile_size - margin, final_aligned_index)
        late_index -= late_index % TIME_SCALE
        if late_index not in indices:
            indices.append(late_index)
    return indices


FRAME_COUNT = frame_count_for_duration(DEFAULT_TOTAL_DURATION, DEFAULT_FRAME_RATE)
TEMPORAL_TILE_SIZE = aligned_frames(DEFAULT_TILE_DURATION, DEFAULT_FRAME_RATE)
TEMPORAL_OVERLAP = aligned_frames(DEFAULT_OVERLAP_DURATION, DEFAULT_FRAME_RATE)
LATE_REFERENCE_MARGIN = aligned_frames(
    DEFAULT_LATE_REFERENCE_OFFSET, DEFAULT_FRAME_RATE
)
TILE_STARTS = temporal_tile_starts(FRAME_COUNT, TEMPORAL_TILE_SIZE, TEMPORAL_OVERLAP)
COND_IMAGE_INDICES = late_reference_indices(
    FRAME_COUNT,
    TEMPORAL_TILE_SIZE,
    TEMPORAL_OVERLAP,
    LATE_REFERENCE_MARGIN,
)
COND_IMAGE_INDICES_TEXT = ", ".join(str(index) for index in COND_IMAGE_INDICES)

# ─── Workflow builder ────────────────────────────────────────────────

_link_counter = 0
_nodes: list[dict] = []
_links: list[list] = []
_groups: list[dict] = []
_bus_node_id = 110


def _next_link_id():
    global _link_counter
    _link_counter += 1
    return _link_counter


def next_bus_id():
    """Reserve IDs for Set/Get bus nodes outside hand-written graph IDs."""
    global _bus_node_id
    nid = _bus_node_id
    _bus_node_id += 1
    return nid


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


def inp(nid: int, name: str, typ: str, widget: bool = False):
    """Declare an input slot on a node (call in slot order)."""
    for n in _nodes:
        if n["id"] == nid:
            node_input = {"name": name, "type": typ, "link": None}
            if widget:
                node_input["widget"] = {"name": name}
            n["inputs"].append(node_input)
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


def group(
    gid: int,
    title: str,
    bounding: tuple[int, int, int, int],
    color: str = "#3f789e",
):
    """Add a LiteGraph group frame."""
    _groups.append(
        {
            "id": gid,
            "title": title,
            "bounding": list(bounding),
            "color": color,
            "font_size": 24,
            "flags": {},
        }
    )


def set_bus(
    nid: int,
    pos: tuple[int, int],
    name: str,
    source_id: int,
    source_slot: int,
    typ: str,
):
    """Publish a typed KJNodes Set bus beside a long-lived source."""
    node(nid, "SetNode", pos, [name], (190, 60), title=f"Set_{name}")
    inp(nid, typ, typ)
    for n in _nodes:
        if n["id"] == nid:
            n["flags"]["collapsed"] = True
            n["outputs"] = [{"name": "*", "type": "*", "links": None}]
            n["properties"] = {
                "Node name for S&R": "SetNode",
                "aux_id": "kijai/ComfyUI-KJNodes",
                "previousName": name,
            }
            break
    link(source_id, source_slot, nid, 0, typ)
    return nid


def get_bus(
    nid: int,
    pos: tuple[int, int],
    name: str,
    typ: str,
):
    """Read a typed KJNodes Set bus close to its consumer."""
    node(nid, "GetNode", pos, [name], (190, 58), title=f"Get_{name}")
    out(nid, typ, typ)
    for n in _nodes:
        if n["id"] == nid:
            n["flags"]["collapsed"] = True
            n["properties"] = {
                "Node name for S&R": "GetNode",
                "aux_id": "kijai/ComfyUI-KJNodes",
            }
            break
    return nid


def primitive_string(nid: int, pos: tuple[int, int], title: str, value: str):
    node(nid, "PrimitiveStringMultiline", pos, [value], (500, 180), title=title)
    inp(nid, "value", "STRING", widget=True)
    out(nid, "STRING", "STRING")
    for n in _nodes:
        if n["id"] == nid:
            n["properties"]["Run widget replace on values"] = False
            break
    return nid


def math_expression(
    nid: int,
    pos: tuple[int, int],
    title: str,
    expression: str,
    values: list[tuple[str, int, int, str]],
):
    node(
        nid,
        "ComfyMathExpression",
        pos,
        [expression],
        (290, 150),
        title=title,
    )
    for input_index, (name, source_id, source_slot, source_type) in enumerate(values):
        inp(nid, f"values.{name}", "FLOAT,INT")
        link(source_id, source_slot, nid, input_index, source_type)
    out(nid, "FLOAT", "FLOAT")
    out(nid, "INT", "INT")
    return nid


def concatenate_text(
    nid: int,
    pos: tuple[int, int],
    title: str,
    string_a: int,
    string_b: int,
    delimiter: str,
):
    node(nid, "StringConcatenate", pos, ["", "", delimiter], (240, 166), title=title)
    inp(nid, "string_a", "STRING", widget=True)
    inp(nid, "string_b", "STRING", widget=True)
    inp(nid, "delimiter", "STRING", widget=True)
    out(nid, "STRING", "STRING")
    link(string_a, 0, nid, 0, "STRING")
    link(string_b, 0, nid, 1, "STRING")
    return nid


def image_batch(
    nid: int,
    pos: tuple[int, int],
    title: str,
    image_a: int,
    image_b: int,
):
    node(nid, "ImageBatch", pos, [], (220, 46), title=title)
    inp(nid, "image1", "IMAGE")
    inp(nid, "image2", "IMAGE")
    out(nid, "IMAGE", "IMAGE")
    link(image_a, 0, nid, 0, "IMAGE")
    link(image_b, 0, nid, 1, "IMAGE")
    return nid


def multi_prompt_provider(nid: int, pos: tuple[int, int], clip_id: int):
    node(
        nid,
        "MultiPromptProvider",
        pos,
        [""],
        (400, 220),
        title="Per-Tile Prompts From Global + Snippets",
    )
    inp(nid, "prompts", "STRING", widget=True)
    inp(nid, "clip", "CLIP")
    out(nid, "conditionings", "CONDITIONING")
    link(clip_id, 0, nid, 1, "CLIP")
    return nid


def build():
    return {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": max(n["id"] for n in _nodes),
        "last_link_id": _link_counter,
        "nodes": _nodes,
        "links": _links,
        "groups": _groups,
        "config": {},
        "extra": {
            "ds": {"scale": 0.6, "offset": [0, 0]},
            "info": {
                "name": "LTX-2.3 Two-Pass AV I2V Looping Late Refs",
                "description": (
                    "Two-pass AV I2V workflow for long video. "
                    "Stage 1 generates at base resolution with temporal tiling. "
                    "Stage 2 spatially upscales and refines. "
                    "Late soft reference images and per-tile prompt snippets "
                    "maintain continuity across temporal tiles."
                ),
            },
        },
        "version": 0.4,
    }


# ─── Layout constants ───────────────────────────────────────────────

COL_INPUT = 0
COL_MODELS = 600
COL_TEXT = 1150
COL_S1_PREP = 1950
COL_S1_SAMPLE = 2800
COL_MID = 3650
COL_S2_SAMPLE = 4500
COL_OUTPUT = 5350

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
set_bus(next_bus_id(), (COL_INPUT + 320, ROW_TOP + 20), "start_image", 1, 0, "IMAGE")

node(2, "LTXVPreprocess", (COL_INPUT, ROW_TOP + 340), [18])
inp(2, "image", "IMAGE")
out(2, "output_image", "IMAGE")
link(1, 0, 2, 0, "IMAGE")  # LoadImage → Preprocess
set_bus(
    next_bus_id(),
    (COL_INPUT + 320, ROW_TOP + 400),
    "preprocessed_start_image",
    2,
    0,
    "IMAGE",
)

node(4, "PrimitiveFloat", (COL_INPUT, ROW_BOT), [DEFAULT_FRAME_RATE],
     (200, 100),
     title="Frame Rate")
out(4, "FLOAT", "FLOAT")
set_bus(next_bus_id(), (COL_INPUT + 220, ROW_BOT + 110), "fps", 4, 0, "FLOAT")

node(5, "PrimitiveBoolean", (COL_INPUT, ROW_BOT + 130), [True], (200, 80),
     title="I2V Enable")
out(5, "BOOLEAN", "BOOLEAN")
set_bus(next_bus_id(), (COL_INPUT + 220, ROW_BOT + 180), "i2v_enable", 5, 0, "BOOLEAN")

node(7, "PrimitiveInt", (COL_INPUT + 220, ROW_BOT), [DEFAULT_FINAL_HEIGHT, "fixed"],
     (210, 100), title="Final Height Target")
out(7, "INT", "INT")

node(16, "GetImageSize", (COL_INPUT, ROW_TOP + 580), [], (300, 100),
     title="Reference Image Size")
inp(16, "image", "IMAGE")
out(16, "width", "INT")
out(16, "height", "INT")
out(16, "batch_size", "INT")
link(1, 0, 16, 0, "IMAGE")

math_expression(
    17,
    (COL_INPUT + 470, ROW_BOT + 200),
    "Align Final Height x64",
    "max(64, round(a / 64) * 64)",
    [("a", 7, 0, "INT")],
)
math_expression(
    18,
    (COL_INPUT + 780, ROW_BOT + 200),
    "Final Width From Ref Aspect",
    "max(64, round((a * b / max(1, c)) / 64) * 64)",
    [("a", 17, 1, "INT"), ("b", 16, 0, "INT"), ("c", 16, 1, "INT")],
)
math_expression(
    19,
    (COL_INPUT + 1090, ROW_BOT + 200),
    "Stage 1 Width",
    "max(32, int(a / 2))",
    [("a", 18, 1, "INT")],
)
set_bus(
    next_bus_id(),
    (COL_INPUT + 1100, ROW_BOT + 380),
    "stage_1_width",
    19,
    1,
    "INT",
)
math_expression(
    23,
    (COL_INPUT + 470, ROW_BOT + 380),
    "Stage 1 Height",
    "max(32, int(a / 2))",
    [("a", 17, 1, "INT")],
)
set_bus(
    next_bus_id(),
    (COL_INPUT + 780, ROW_BOT + 430),
    "stage_1_height",
    23,
    1,
    "INT",
)

node(24, "LTXVLoopingReferenceSchedule", (COL_INPUT, ROW_BOT + 260),
     [
         DEFAULT_FRAME_RATE,
         DEFAULT_TOTAL_DURATION,
         DEFAULT_TILE_DURATION,
         DEFAULT_OVERLAP_DURATION,
         DEFAULT_LATE_REFERENCE_OFFSET,
     ],
     (430, 310), title="Looping Timing + Reference Schedule")
inp(24, "reference_images", "IMAGE")
inp(24, "frame_rate", "FLOAT", widget=True)
inp(24, "total_duration", "FLOAT", widget=True)
inp(24, "tile_duration", "FLOAT", widget=True)
inp(24, "overlap_duration", "FLOAT", widget=True)
inp(24, "reference_offset", "FLOAT", widget=True)
out(24, "reference_images", "IMAGE")
out(24, "frame_count", "INT")
out(24, "temporal_tile_size", "INT")
out(24, "temporal_overlap", "INT")
out(24, "reference_indices", "STRING")
out(24, "tile_count", "INT")
link(4, 0, 24, 1, "FLOAT")
set_bus(
    next_bus_id(),
    (COL_INPUT + 450, ROW_DEEP + 180),
    "scheduled_reference_images",
    24,
    0,
    "IMAGE",
)
set_bus(
    next_bus_id(),
    (COL_INPUT + 450, ROW_DEEP + 250),
    "frame_count",
    24,
    1,
    "INT",
)
set_bus(
    next_bus_id(),
    (COL_INPUT + 650, ROW_DEEP + 180),
    "temporal_tile_size",
    24,
    2,
    "INT",
)
set_bus(
    next_bus_id(),
    (COL_INPUT + 650, ROW_DEEP + 250),
    "temporal_overlap",
    24,
    3,
    "INT",
)
set_bus(
    next_bus_id(),
    (COL_INPUT + 850, ROW_DEEP + 180),
    "reference_indices",
    24,
    4,
    "STRING",
)

node(6, "Note", (COL_INPUT, ROW_DEEP + 210), [
    "## Late Reference Tile Layout\n\n"
    "The Looping Timing + Reference Schedule node calculates clip frames, "
    "sampler tile size, overlap, and late reference indices.\n\n"
    f"Default frame count: {FRAME_COUNT} (`8n+1`).\n"
    f"Default tile size: {TEMPORAL_TILE_SIZE}. Overlap: {TEMPORAL_OVERLAP}. "
    f"Stride: {TEMPORAL_TILE_SIZE - TEMPORAL_OVERLAP}.\n"
    f"Default tile starts: {', '.join(str(start) for start in TILE_STARTS)}.\n\n"
    "The current image/snippet branches match these default indices:\n"
    f"  `{COND_IMAGE_INDICES_TEXT}`\n"
    "If duration adds tiles after the supplied refs, the schedule repeats the "
    "last image. It truncates extra supplied refs for shorter clips.\n"
    "The looping sampler already repeats the last tile prompt after the "
    "snippet list ends.\n\n"
    "Edit duration, tile duration, overlap, and late-reference offset "
    "inside the schedule node. Edit the Global Positive Prompt once and "
    "each Tile Prompt Snippet "
    "beside its late reference branch. The graph concatenates "
    "`global + snippet` for each tile and joins those prompts with `|` for "
    "the multi-prompt node."
], (440, 340))

# ── Model loading ──

node(10, "CheckpointLoaderSimple", (COL_MODELS, ROW_TOP),
     ["ltx-2.3-22b-dev.safetensors"], (350, 150))
out(10, "MODEL", "MODEL")
out(10, "CLIP", "CLIP")
out(10, "VAE", "VAE")
set_bus(next_bus_id(), (COL_MODELS + 360, ROW_TOP + 70), "video_vae", 10, 2, "VAE")

node(11, "LTXAVTextEncoderLoader", (COL_MODELS, ROW_TOP + 180),
     ["comfy_gemma_3_12B_it.safetensors", "ltx-2.3-22b-dev.safetensors", "default"],
     (380, 130))
out(11, "CLIP", "CLIP")

node(12, "LTXVAudioVAELoader", (COL_MODELS, ROW_MID),
     ["ltx-2.3-22b-dev.safetensors"], (350, 100))
out(12, "Audio VAE", "VAE")
set_bus(next_bus_id(), (COL_MODELS + 360, ROW_MID + 20), "audio_vae", 12, 0, "VAE")

node(13, "LoraLoaderModelOnly", (COL_MODELS, ROW_MID + 130),
     ["ltx-2.3-22b-distilled-lora-384.safetensors", 0.5], (380, 100),
     title="Distilled LoRA (both stages)")
inp(13, "model", "MODEL")
out(13, "MODEL", "MODEL")
link(10, 0, 13, 0, "MODEL")  # Checkpoint → LoRA

node(25, "Power Lora Loader (rgthree)", (COL_MODELS, ROW_MID + 260),
     [], (400, 190), title="Extra LoRAs (rgthree)")
inp(25, "model", "MODEL")
inp(25, "clip", "CLIP")
out(25, "MODEL", "MODEL")
out(25, "CLIP", "CLIP")
link(13, 0, 25, 0, "MODEL")
for n in _nodes:
    if n["id"] == 25:
        n["properties"].update(
            {
                "cnr_id": "rgthree-comfy",
                "aux_id": "rgthree/rgthree-comfy",
                "Show Strengths": "Single Strength",
                "Match": "",
            }
        )
        break
set_bus(next_bus_id(), (COL_MODELS + 410, ROW_BOT + 60), "model", 25, 0, "MODEL")

node(14, "LatentUpscaleModelLoader", (COL_MODELS, ROW_BOT + 80),
     ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"], (380, 100))
out(14, "LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")
set_bus(
    next_bus_id(),
    (COL_MODELS + 410, ROW_BOT + 450),
    "latent_upscale_model",
    14,
    0,
    "LATENT_UPSCALE_MODEL",
)

# ── Text encoding ──

node(20, "CLIPTextEncode", (COL_TEXT, ROW_TOP),
     [""], (400, 180), title="Global Prompt Fallback Encode")
inp(20, "clip", "CLIP")
inp(20, "text", "STRING", widget=True)
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
conditioning_fps_id = get_bus(next_bus_id(), (COL_TEXT, ROW_MID - 58), "fps", "FLOAT")
link(conditioning_fps_id, 0, 22, 2, "FLOAT")
set_bus(next_bus_id(), (COL_TEXT + 320, ROW_MID), "positive_conditioning", 22, 0, "CONDITIONING")
set_bus(next_bus_id(), (COL_TEXT + 320, ROW_MID + 70), "negative_conditioning", 22, 1, "CONDITIONING")

# ── Stage 1 prep ──

node(30, "EmptyLTXVLatentVideo", (COL_S1_PREP, ROW_TOP), [960, 544, FRAME_COUNT, 1],
     (250, 150), title="Stage 1 Empty Latent")
inp(30, "width", "INT", widget=True)
inp(30, "height", "INT", widget=True)
inp(30, "length", "INT", widget=True)
out(30, "LATENT", "LATENT")
stage1_width_id = get_bus(next_bus_id(), (COL_S1_PREP - 210, ROW_TOP), "stage_1_width", "INT")
stage1_height_id = get_bus(next_bus_id(), (COL_S1_PREP - 210, ROW_TOP + 70), "stage_1_height", "INT")
stage1_frames_id = get_bus(next_bus_id(), (COL_S1_PREP - 210, ROW_TOP + 140), "frame_count", "INT")
link(stage1_width_id, 0, 30, 0, "INT")   # Aspect-ratio width at stage 1
link(stage1_height_id, 0, 30, 1, "INT")  # Half final height at stage 1
link(stage1_frames_id, 0, 30, 2, "INT")  # Valid 8n+1 frame count

node(31, "LTXVEmptyLatentAudio", (COL_S1_PREP, ROW_TOP + 180), [FRAME_COUNT, 25, 1],
     (250, 130))
inp(31, "audio_vae", "VAE")
inp(31, "frames_number", "INT")
inp(31, "frame_rate", "INT")
out(31, "Latent", "LATENT")
stage1_audio_vae_id = get_bus(next_bus_id(), (COL_S1_PREP - 210, ROW_TOP + 220), "audio_vae", "VAE")
link(stage1_audio_vae_id, 0, 31, 0, "VAE")  # Audio VAE
link(stage1_frames_id, 0, 31, 1, "INT")     # Frame count

node(34, "CM_FloatToInt", (COL_S1_PREP - 180, ROW_TOP + 320), [0], (150, 80),
     title="FPS→Int")
inp(34, "a", "FLOAT")
out(34, "INT", "INT")
stage1_fps_id = get_bus(next_bus_id(), (COL_S1_PREP - 390, ROW_TOP + 330), "fps", "FLOAT")
link(stage1_fps_id, 0, 34, 0, "FLOAT")
link(34, 0, 31, 2, "INT")   # Frame rate int → audio

node(32, "LTXVImgToVideoConditionOnly", (COL_S1_PREP, ROW_MID),
     [0.7, False], (300, 130), title="Stage 1 I2V Cond",
     color=S1_COLOR, bgcolor=S1_BG)
inp(32, "vae", "VAE")
inp(32, "image", "IMAGE")
inp(32, "latent", "LATENT")
inp(32, "bypass", "BOOLEAN")
out(32, "latent", "LATENT")
stage1_video_vae_id = get_bus(next_bus_id(), (COL_S1_PREP - 210, ROW_MID + 20), "video_vae", "VAE")
stage1_preprocessed_id = get_bus(
    next_bus_id(),
    (COL_S1_PREP - 210, ROW_MID + 90),
    "preprocessed_start_image",
    "IMAGE",
)
stage1_i2v_id = get_bus(next_bus_id(), (COL_S1_PREP - 210, ROW_MID + 160), "i2v_enable", "BOOLEAN")
link(stage1_video_vae_id, 0, 32, 0, "VAE")      # Checkpoint VAE
link(stage1_preprocessed_id, 0, 32, 1, "IMAGE")  # Preprocessed reference
link(30, 0, 32, 2, "LATENT")  # Empty latent
link(stage1_i2v_id, 0, 32, 3, "BOOLEAN")        # I2V enable

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
stage1_anchor_image_id = get_bus(
    next_bus_id(),
    (COL_S1_PREP - 210, ROW_DEEP),
    "preprocessed_start_image",
    "IMAGE",
)
stage1_anchor_vae_id = get_bus(next_bus_id(), (COL_S1_PREP - 210, ROW_DEEP + 70), "video_vae", "VAE")
link(stage1_anchor_image_id, 0, 35, 0, "IMAGE")  # Preprocessed reference
link(stage1_anchor_vae_id, 0, 35, 1, "VAE")      # Checkpoint VAE
set_bus(
    next_bus_id(),
    (COL_S1_PREP + 270, ROW_DEEP + 20),
    "stage_1_anchor_latent",
    35,
    0,
    "LATENT",
)

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
stage1_guider_model_id = get_bus(next_bus_id(), (COL_S1_SAMPLE - 210, ROW_TOP + 220), "model", "MODEL")
stage1_guider_positive_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_TOP + 285),
    "positive_conditioning",
    "CONDITIONING",
)
stage1_guider_negative_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_TOP + 350),
    "negative_conditioning",
    "CONDITIONING",
)
link(stage1_guider_model_id, 0, 43, 0, "MODEL")              # Model with all LoRAs
link(stage1_guider_positive_id, 0, 43, 1, "CONDITIONING")    # Positive
link(stage1_guider_negative_id, 0, 43, 2, "CONDITIONING")    # Negative

# LTXVLoopingSampler — Stage 1
# Widgets: temporal_tile_size, temporal_overlap, guiding_strength,
#          temporal_overlap_cond_strength, cond_image_strength,
#          horizontal_tiles, vertical_tiles, spatial_overlap,
#          adain_factor, guiding_start_step, guiding_end_step,
#          optional_cond_image_indices
node(44, "LTXVLoopingSampler", (COL_S1_SAMPLE, ROW_MID),
     [
         TEMPORAL_TILE_SIZE,
         TEMPORAL_OVERLAP,
         1.0,
         0.5,
         1.0,
         1,
         1,
         1,
         0.15,
         0,
         1000,
         COND_IMAGE_INDICES_TEXT,
     ],
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
inp(44, "temporal_tile_size", "INT", widget=True)
inp(44, "temporal_overlap", "INT", widget=True)
inp(44, "optional_cond_image_indices", "STRING", widget=True)
out(44, "denoised_output", "LATENT")

stage1_model_id = get_bus(next_bus_id(), (COL_S1_SAMPLE - 210, ROW_MID + 10), "model", "MODEL")
stage1_sampler_vae_id = get_bus(next_bus_id(), (COL_S1_SAMPLE - 210, ROW_MID + 75), "video_vae", "VAE")
stage1_references_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_MID + 140),
    "scheduled_reference_images",
    "IMAGE",
)
stage1_prompts_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_MID + 205),
    "tile_prompt_conditioning",
    "CONDITIONING",
)
stage1_tile_size_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_MID + 270),
    "temporal_tile_size",
    "INT",
)
stage1_overlap_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_MID + 335),
    "temporal_overlap",
    "INT",
)
stage1_ref_indices_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_MID + 400),
    "reference_indices",
    "STRING",
)
stage1_anchor_id = get_bus(
    next_bus_id(),
    (COL_S1_SAMPLE - 210, ROW_MID + 465),
    "stage_1_anchor_latent",
    "LATENT",
)
link(stage1_model_id, 0, 44, 0, "MODEL")       # Model with all LoRAs
link(stage1_sampler_vae_id, 0, 44, 1, "VAE")    # Checkpoint VAE
link(40, 0, 44, 2, "NOISE")   # Noise
link(41, 0, 44, 3, "SAMPLER") # Sampler
link(42, 0, 44, 4, "SIGMAS")  # Sigmas
link(43, 0, 44, 5, "GUIDER")  # Guider
link(33, 0, 44, 6, "LATENT")  # AV latent (video + audio)
link(stage1_references_id, 0, 44, 7, "IMAGE")  # Scheduled and repeated references
# slot 8: optional_guiding_latents — not connected (no IC-LoRA guide)
link(stage1_prompts_id, 0, 44, 9, "CONDITIONING")  # Per-tile prompt conditioning
link(stage1_anchor_id, 0, 44, 10, "LATENT")  # Negative index latents anchor
# slot 11: optional_normalizing_latents — not connected
link(stage1_tile_size_id, 0, 44, 12, "INT")        # Tile duration in aligned frames
link(stage1_overlap_id, 0, 44, 13, "INT")          # Temporal overlap in aligned frames
link(stage1_ref_indices_id, 0, 44, 14, "STRING")   # Reference frame indices

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
set_bus(next_bus_id(), (COL_MID + 270, ROW_MID + 40), "stage_1_audio", 50, 1, "LATENT")

node(51, "LTXVLatentUpsampler", (COL_MID, ROW_MID + 130), [], (300, 100),
     title="Spatial Upscale 2x")
inp(51, "samples", "LATENT")
inp(51, "upscale_model", "LATENT_UPSCALE_MODEL")
inp(51, "vae", "VAE")
out(51, "LATENT", "LATENT")
stage2_upscale_model_id = get_bus(
    next_bus_id(),
    (COL_MID - 210, ROW_MID + 150),
    "latent_upscale_model",
    "LATENT_UPSCALE_MODEL",
)
stage2_upscale_vae_id = get_bus(next_bus_id(), (COL_MID - 210, ROW_MID + 215), "video_vae", "VAE")
link(50, 0, 51, 0, "LATENT")              # Video latent only
link(stage2_upscale_model_id, 0, 51, 1, "LATENT_UPSCALE_MODEL")  # Upscale model
link(stage2_upscale_vae_id, 0, 51, 2, "VAE")                     # VAE

node(52, "LTXVImgToVideoConditionOnly", (COL_MID, ROW_MID + 260),
     [1.0, False], (300, 130), title="Stage 2 I2V Cond",
     color=S2_COLOR, bgcolor=S2_BG)
inp(52, "vae", "VAE")
inp(52, "image", "IMAGE")
inp(52, "latent", "LATENT")
inp(52, "bypass", "BOOLEAN")
out(52, "latent", "LATENT")
stage2_i2v_vae_id = get_bus(next_bus_id(), (COL_MID - 210, ROW_MID + 290), "video_vae", "VAE")
stage2_start_image_id = get_bus(next_bus_id(), (COL_MID - 210, ROW_MID + 355), "start_image", "IMAGE")
stage2_i2v_enable_id = get_bus(next_bus_id(), (COL_MID - 210, ROW_MID + 420), "i2v_enable", "BOOLEAN")
link(stage2_i2v_vae_id, 0, 52, 0, "VAE")       # VAE
link(stage2_start_image_id, 0, 52, 1, "IMAGE")  # Reference; conditioner resizes internally
link(51, 0, 52, 2, "LATENT")  # Upscaled video latent
link(stage2_i2v_enable_id, 0, 52, 3, "BOOLEAN")  # I2V enable

# Stage 2 receives AV latent (upscaled video + stage 1 audio).
# The looping sampler preserves input audio data for refinement:
# base tile uses the corresponding input audio slice, extend tiles
# pass source audio for new-frame initialization via _audio_new_init.
node(53, "LTXVConcatAVLatent", (COL_MID, ROW_BOT + 200), [], (250, 100),
     title="Stage 2 AV Concat", color=S2_COLOR, bgcolor=S2_BG)
inp(53, "video_latent", "LATENT")
inp(53, "audio_latent", "LATENT")
out(53, "latent", "LATENT")
stage2_audio_id = get_bus(next_bus_id(), (COL_MID - 210, ROW_BOT + 230), "stage_1_audio", "LATENT")
link(52, 0, 53, 0, "LATENT")  # Conditioned upscaled video
link(stage2_audio_id, 0, 53, 1, "LATENT")  # Audio from stage 1

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
stage2_guider_model_id = get_bus(next_bus_id(), (COL_S2_SAMPLE - 210, ROW_TOP + 220), "model", "MODEL")
stage2_guider_positive_id = get_bus(
    next_bus_id(),
    (COL_S2_SAMPLE - 210, ROW_TOP + 285),
    "positive_conditioning",
    "CONDITIONING",
)
stage2_guider_negative_id = get_bus(
    next_bus_id(),
    (COL_S2_SAMPLE - 210, ROW_TOP + 350),
    "negative_conditioning",
    "CONDITIONING",
)
link(stage2_guider_model_id, 0, 63, 0, "MODEL")             # Same model with all LoRAs
link(stage2_guider_positive_id, 0, 63, 1, "CONDITIONING")   # Same positive
link(stage2_guider_negative_id, 0, 63, 2, "CONDITIONING")   # Same negative

# LTXVLoopingSampler — Stage 2
# spatial tiling 2x1 for upscaled resolution
node(64, "LTXVLoopingSampler", (COL_S2_SAMPLE, ROW_MID),
     [
         TEMPORAL_TILE_SIZE,
         TEMPORAL_OVERLAP,
         1.0,
         0.5,
         1.0,
         2,
         1,
         1,
         0.0,
         0,
         1000,
         COND_IMAGE_INDICES_TEXT,
     ],
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
inp(64, "temporal_tile_size", "INT", widget=True)
inp(64, "temporal_overlap", "INT", widget=True)
inp(64, "optional_cond_image_indices", "STRING", widget=True)
out(64, "denoised_output", "LATENT")

stage2_model_id = get_bus(next_bus_id(), (COL_S2_SAMPLE - 210, ROW_MID + 10), "model", "MODEL")
stage2_sampler_vae_id = get_bus(next_bus_id(), (COL_S2_SAMPLE - 210, ROW_MID + 75), "video_vae", "VAE")
stage2_references_id = get_bus(
    next_bus_id(),
    (COL_S2_SAMPLE - 210, ROW_MID + 140),
    "scheduled_reference_images",
    "IMAGE",
)
stage2_prompts_id = get_bus(
    next_bus_id(),
    (COL_S2_SAMPLE - 210, ROW_MID + 205),
    "tile_prompt_conditioning",
    "CONDITIONING",
)
stage2_tile_size_id = get_bus(next_bus_id(), (COL_S2_SAMPLE - 210, ROW_MID + 270), "temporal_tile_size", "INT")
stage2_overlap_id = get_bus(next_bus_id(), (COL_S2_SAMPLE - 210, ROW_MID + 335), "temporal_overlap", "INT")
stage2_ref_indices_id = get_bus(
    next_bus_id(),
    (COL_S2_SAMPLE - 210, ROW_MID + 400),
    "reference_indices",
    "STRING",
)
link(stage2_model_id, 0, 64, 0, "MODEL")       # Model with all LoRAs
link(stage2_sampler_vae_id, 0, 64, 1, "VAE")    # VAE
link(60, 0, 64, 2, "NOISE")   # Noise
link(61, 0, 64, 3, "SAMPLER") # Sampler
link(62, 0, 64, 4, "SIGMAS")  # Sigmas
link(63, 0, 64, 5, "GUIDER")  # Guider
link(53, 0, 64, 6, "LATENT")  # Stage 2 AV latent (upscaled video + stage 1 audio)
link(stage2_references_id, 0, 64, 7, "IMAGE")  # Scheduled and repeated references
# slot 8: optional_guiding_latents — not connected
link(stage2_prompts_id, 0, 64, 9, "CONDITIONING")  # Per-tile prompt conditioning
# slot 10-11: not connected for stage 2
link(stage2_tile_size_id, 0, 64, 12, "INT")       # Tile duration in aligned frames
link(stage2_overlap_id, 0, 64, 13, "INT")         # Temporal overlap in aligned frames
link(stage2_ref_indices_id, 0, 64, 14, "STRING")  # Reference frame indices

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
output_video_vae_id = get_bus(next_bus_id(), (COL_OUTPUT - 210, ROW_MID + 165), "video_vae", "VAE")
link(70, 0, 71, 0, "LATENT")  # Refined video
link(output_video_vae_id, 0, 71, 1, "VAE")  # VAE

node(72, "LTXVAudioVAEDecode", (COL_OUTPUT, ROW_MID + 360), [], (250, 100))
inp(72, "samples", "LATENT")
inp(72, "audio_vae", "VAE")
out(72, "Audio", "AUDIO")
output_audio_vae_id = get_bus(next_bus_id(), (COL_OUTPUT - 210, ROW_MID + 395), "audio_vae", "VAE")
link(70, 1, 72, 0, "LATENT")  # Refined audio
link(output_audio_vae_id, 0, 72, 1, "VAE")  # Audio VAE

node(73, "CreateVideo", (COL_OUTPUT, ROW_BOT + 200), [30], (250, 100))
inp(73, "images", "IMAGE")
inp(73, "audio", "AUDIO")
inp(73, "fps", "FLOAT")
out(73, "VIDEO", "VIDEO")
output_fps_id = get_bus(next_bus_id(), (COL_OUTPUT - 210, ROW_BOT + 230), "fps", "FLOAT")
link(71, 0, 73, 0, "IMAGE")
link(72, 0, 73, 1, "AUDIO")
link(output_fps_id, 0, 73, 2, "FLOAT")   # Frame rate

node(74, "SaveVideo", (COL_OUTPUT, ROW_DEEP), ["LTX-2.3/Looping", "auto", "auto"],
     (250, 100))
inp(74, "video", "VIDEO")
link(73, 0, 74, 0, "VIDEO")

# ── Per-tile reference and prompt branches ──

_dynamic_node_id = 80


def next_dynamic_id():
    global _dynamic_node_id
    nid = _dynamic_node_id
    _dynamic_node_id += 1
    return nid


global_prompt_id = primitive_string(
    next_dynamic_id(),
    (COL_TEXT + 250, ROW_BOT + 50),
    "Global Positive Prompt",
    GLOBAL_PROMPT,
)
set_bus(
    next_bus_id(),
    (COL_TEXT + 250, ROW_BOT + 240),
    "global_prompt",
    global_prompt_id,
    0,
    "STRING",
)
fallback_global_prompt_id = get_bus(
    next_bus_id(),
    (COL_TEXT, ROW_TOP - 70),
    "global_prompt",
    "STRING",
)
link(fallback_global_prompt_id, 0, 20, 1, "STRING")

multi_prompt_id = multi_prompt_provider(
    next_dynamic_id(),
    (COL_TEXT, ROW_MID + 140),
    11,
)
set_bus(
    next_bus_id(),
    (COL_TEXT + 400, ROW_MID + 170),
    "tile_prompt_conditioning",
    multi_prompt_id,
    0,
    "CONDITIONING",
)

late_load_ids = []
full_prompt_ids = []
late_ref_x = COL_INPUT - 400
for tile_index, late_index in enumerate(COND_IMAGE_INDICES[1:]):
    tile_y = ROW_DEEP + 620 + tile_index * 330

    late_load_id = next_dynamic_id()
    node(
        late_load_id,
        "LoadImage",
        (late_ref_x, tile_y),
        [f"reference_tile_{tile_index}_late.png", "image"],
        (300, 300),
        title=f"Late Ref Tile {tile_index} - Frame {late_index}",
    )
    out(late_load_id, "IMAGE", "IMAGE")
    out(late_load_id, "MASK", "MASK")
    late_load_ids.append(late_load_id)

    snippet = TILE_SNIPPETS[tile_index] if tile_index < len(TILE_SNIPPETS) else ""
    snippet_id = primitive_string(
        next_dynamic_id(),
        (late_ref_x + 340, tile_y),
        f"Tile {tile_index} Prompt Snippet",
        snippet,
    )
    tile_global_prompt_id = get_bus(
        next_bus_id(),
        (late_ref_x + 860, tile_y + 180),
        "global_prompt",
        "STRING",
    )
    full_prompt_id = concatenate_text(
        next_dynamic_id(),
        (late_ref_x + 860, tile_y),
        f"Global + Tile {tile_index} Snippet",
        tile_global_prompt_id,
        snippet_id,
        " ",
    )
    full_prompt_ids.append(full_prompt_id)

reference_batch_id = get_bus(
    next_bus_id(),
    (late_ref_x + 340, ROW_DEEP + 620 + 270),
    "start_image",
    "IMAGE",
)
for batch_index, load_id in enumerate(late_load_ids):
    tile_y = ROW_DEEP + 620 + batch_index * 330
    reference_batch_id = image_batch(
        next_dynamic_id(),
        (late_ref_x + 340, tile_y + 220),
        f"Ref Batch {batch_index + 1}",
        reference_batch_id,
        load_id,
    )

set_bus(
    next_bus_id(),
    (late_ref_x + 580, ROW_DEEP + 620 + (len(late_load_ids) - 1) * 330 + 220),
    "reference_image_batch",
    reference_batch_id,
    0,
    "IMAGE",
)
schedule_reference_batch_id = get_bus(
    next_bus_id(),
    (COL_INPUT - 210, ROW_BOT + 330),
    "reference_image_batch",
    "IMAGE",
)
link(schedule_reference_batch_id, 0, 24, 0, "IMAGE")

joined_prompt_id = full_prompt_ids[0]
for join_index, full_prompt_id in enumerate(full_prompt_ids[1:]):
    tile_y = ROW_DEEP + 620 + (join_index + 1) * 330
    joined_prompt_id = concatenate_text(
        next_dynamic_id(),
        (late_ref_x + 1140, tile_y),
        f"Join Tile Prompts {join_index + 2}",
        joined_prompt_id,
        full_prompt_id,
        " | ",
    )
set_bus(
    next_bus_id(),
    (late_ref_x + 1400, ROW_DEEP + 620 + (len(full_prompt_ids) - 1) * 330 + 40),
    "joined_tile_prompts",
    joined_prompt_id,
    0,
    "STRING",
)
joined_tile_prompts_id = get_bus(
    next_bus_id(),
    (COL_TEXT, ROW_MID + 370),
    "joined_tile_prompts",
    "STRING",
)
link(joined_tile_prompts_id, 0, multi_prompt_id, 0, "STRING")


# ── Functional groups ──

group(1, "Inputs + Timing", (-240, -90, 790, 1870), "#6a8b80")
group(2, "Models + LoRAs", (560, -90, 670, 1430), "#8a6d3b")
group(3, "Prompt Conditioning", (1110, -110, 820, 1260), "#76518a")
group(4, "Dimensions + Timing Buses", (440, 960, 980, 590), "#5b7e9c")
group(5, "Stage 1 Base AV", (1710, -120, 1540, 1490), "#51724d")
group(6, "Upscale + Stage 2", (3410, -120, 1540, 1270), "#555d96")
group(7, "Final Output", (5110, 250, 650, 1120), "#9b6c4b")
group(8, "Late References + Tile Snippets", (-440, 1740, 1670, 1450), "#3f789e")


# ─── Generate ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    wf = build()
    out_path = os.path.join(os.path.dirname(__file__), "LTX-2.3_Two_Pass_I2V_Looping.json")
    with open(out_path, "w") as f:
        json.dump(wf, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  {len(_nodes)} nodes, {len(_links)} links")
