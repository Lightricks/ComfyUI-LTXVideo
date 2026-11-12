"""Node replacements for nodes removed from this pack.

ComfyUI offers to upgrade saved workflows that still contain these nodes.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("LTXVideo.node_replacements")

_pending: set[asyncio.Task] = set()


def _replacements(io) -> list:
    return [
        # The old node defaulted to srgb_gamma; workflows that picked another
        # colour space need the new node's source re-checked after the upgrade.
        io.NodeReplace(
            new_node_id="ImageColorSpace",
            old_node_id="LTXVSDRToHDRWorkingSpace",
            old_widget_ids=["color_space"],
            input_mapping=[
                {"new_id": "image", "old_id": "image"},
                {"new_id": "source", "set_value": "sRGB"},
                {"new_id": "destination", "set_value": "HDR ACEScct"},
            ],
            output_mapping=[{"new_idx": 0, "old_idx": 0}],
        ),
        # Only applies once LTXVVAEForceFloat32 is removed; while the deprecated
        # node is registered, ComfyUI keeps loading it as-is.
        io.NodeReplace(
            new_node_id="VAEComputeDtype",
            old_node_id="LTXVVAEForceFloat32",
            input_mapping=[
                {"new_id": "vae", "old_id": "vae"},
                {"new_id": "dtype", "set_value": "fp32"},
            ],
            output_mapping=[{"new_idx": 0, "old_idx": 0}],
        ),
    ]


async def _register_all(api, replacements: list) -> None:
    for replacement in replacements:
        await api.node_replacement.register(replacement)


def _on_done(task: asyncio.Task) -> None:
    _pending.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Could not register node replacements: %s", task.exception())


def register_node_replacements() -> None:
    """Schedule registration on ComfyUI's event loop, which runs while custom nodes import."""
    try:
        from comfy_api.latest import ComfyAPI, io

        loop = asyncio.get_running_loop()
    except (ImportError, RuntimeError):
        return
    api = ComfyAPI()
    if not hasattr(api, "node_replacement") or not hasattr(io, "NodeReplace"):
        return
    task = loop.create_task(_register_all(api, _replacements(io)))
    _pending.add(task)
    task.add_done_callback(_on_done)
