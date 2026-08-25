import json
from datetime import datetime
from pathlib import Path

import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io, ui

from .nodes_registry import comfy_node


@comfy_node(name="LTXVSaveConditioning")
class LTXVSaveConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVSaveConditioning",
            display_name="🅛🅣🅧 LTXV Save Conditioning",
            category="lightricks/LTXV",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("filename", default="conditioning"),
                io.Combo.Input("dtype", options=["bfloat16", "float16"]),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, conditioning: list, filename: str, dtype: str) -> io.NodeOutput:
        if not conditioning or len(conditioning) == 0:
            raise ValueError("Conditioning is empty")

        embeddings_folder = Path(folder_paths.get_folder_paths("embeddings")[0])
        embeddings_folder.mkdir(parents=True, exist_ok=True)

        sanitized_filename = "".join(
            c for c in filename if c.isalnum() or c in ("_", "-", ".")
        )
        if not sanitized_filename:
            sanitized_filename = "conditioning"

        output_path = embeddings_folder / f"{sanitized_filename}.safetensors"

        target_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        tensors_to_save: dict[str, torch.Tensor] = {}
        # Non-tensor options decide how the model treats the embeddings, so they
        # have to survive the round-trip alongside the tensors.
        non_tensor_options: dict[str, object] = {}

        for idx, (cond_tensor, cond_options) in enumerate(conditioning):
            tensor_converted = cond_tensor.to(dtype=target_dtype).contiguous()
            tensors_to_save[f"conditioning_data_{idx}"] = tensor_converted

            for key, value in cond_options.items():
                if key == "attention_mask":
                    tensors_to_save[f"attention_mask_{idx}"] = value.contiguous()
                elif torch.is_tensor(value):
                    tensors_to_save[f"option_{idx}_{key}"] = value.contiguous()
                else:
                    try:
                        json.dumps(value)
                    except (TypeError, ValueError):
                        continue
                    non_tensor_options[f"{idx}:{key}"] = value

        metadata = {
            "num_conditionings": str(len(conditioning)),
            "dtype": dtype,
            "created_at": str(datetime.now()),
            "non_tensor_options": json.dumps(non_tensor_options),
        }

        comfy.utils.save_torch_file(
            tensors_to_save, str(output_path), metadata=metadata
        )

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        return io.NodeOutput(
            ui=ui.PreviewText(f"Saved: {output_path.name} ({file_size_mb:.2f} MB)")
        )
