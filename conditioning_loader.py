import hashlib
from pathlib import Path
from typing import Any

import folder_paths
import safetensors
import torch
from comfy_api.latest import io

from .nodes_registry import comfy_node


@comfy_node(name="LTXVLoadConditioning")
class LTXVLoadConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        files = folder_paths.get_filename_list("embeddings")
        if not files:
            files = [""]
        return io.Schema(
            node_id="LTXVLoadConditioning",
            display_name="🅛🅣🅧 LTXV Load Conditioning",
            category="lightricks/LTXV",
            inputs=[
                io.Combo.Input("file_name", options=sorted(files)),
                io.Combo.Input("device", options=["cpu", "gpu"]),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, file_name: str, device: str) -> io.NodeOutput:
        file_path = folder_paths.get_full_path("embeddings", file_name)
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Conditioning file not found: {file_path}")

        target_device = "cpu"
        if device == "gpu":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        conditioning: list[list[Any]] = []

        with safetensors.safe_open(
            file_path, framework="pt", device=target_device
        ) as f:
            tensor_keys = [k for k in f.keys() if k.startswith("conditioning_data_")]

            for tensor_key in sorted(tensor_keys):
                idx = tensor_key.replace("conditioning_data_", "")
                tensor = f.get_tensor(tensor_key)

                options: dict[str, Any] = {}
                mask_key = f"attention_mask_{idx}"
                if mask_key in f.keys():
                    options["attention_mask"] = f.get_tensor(mask_key)

                conditioning.append([tensor, options])

        if not conditioning:
            raise ValueError(f"No conditioning data found in file: {file_name}")

        return io.NodeOutput(conditioning)

    @classmethod
    def fingerprint_inputs(cls, file_name: str, device: str) -> str:
        file_path = folder_paths.get_full_path("embeddings", file_name)
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def validate_inputs(cls, file_name: str, device: str) -> bool | str:
        if not file_name:
            return "No files found. Please save a conditioning first."
        try:
            file_path = folder_paths.get_full_path("embeddings", file_name)
            if not Path(file_path).exists():
                return f"File not found: {file_name}"
        except Exception:
            return f"Invalid file: {file_name}"
        return True
