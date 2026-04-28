import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .nodes_registry import comfy_node


@comfy_node(name="LTXVDilateVideoMask")
class LTXVDilateVideoMask(io.ComfyNode):
    """Dilates a video mask spatially and/or temporally using max-pooling and thresholds the result."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVDilateVideoMask",
            category="Lightricks/mask_operations",
            description=(
                "Dilates a video mask spatially and/or temporally using "
                "separable max-pooling and thresholds the result."
            ),
            inputs=[
                io.Int.Input(
                    "spatial_radius",
                    default=1,
                    min=0,
                    max=30,
                    tooltip="Half-size of the spatial dilation kernel. Kernel = 2*radius+1.",
                ),
                io.Int.Input(
                    "temporal_radius",
                    default=0,
                    min=0,
                    max=10,
                    tooltip="Half-size of the temporal dilation kernel. Kernel = 2*radius+1.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Video mask to dilate. Either this or image_as_mask must be provided.",
                ),
                io.Image.Input(
                    "image_as_mask",
                    optional=True,
                    tooltip="Image to use as mask (channel-averaged). Either this or mask must be provided.",
                ),
            ],
            outputs=[
                io.Mask.Output("mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        spatial_radius: int,
        temporal_radius: int,
        mask: torch.Tensor | None = None,
        image_as_mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        if mask is None and image_as_mask is None:
            raise ValueError("Either 'mask' or 'image_as_mask' must be provided.")

        if mask is None:
            mask = image_as_mask.mean(dim=-1)

        if mask.ndim == 4:
            mask = mask[:, :, :, 0]

        s_kernel = spatial_radius * 2 + 1
        t_kernel = temporal_radius * 2 + 1

        # Separable dilation: 2D spatial + 1D temporal (much faster than 3D pooling)
        if s_kernel > 1:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
            mask = F.max_pool2d(
                mask, kernel_size=s_kernel, stride=1, padding=spatial_radius
            )
            mask = mask.squeeze(1)

        if t_kernel > 1:
            B, H, W = mask.shape
            mask = mask.permute(1, 2, 0).reshape(H * W, 1, B)
            mask = F.max_pool1d(
                mask, kernel_size=t_kernel, stride=1, padding=temporal_radius
            )
            mask = mask.reshape(H, W, B).permute(2, 0, 1)

        mask = (mask > 0.5).float()
        return io.NodeOutput(mask)


_BG_COLOR_RGB = (102, 255, 0)


@comfy_node(name="LTXVInpaintPreprocess")
class LTXVInpaintPreprocess(io.ComfyNode):
    """Composites images with a green (#66FF00) background where mask is active.

    If the mask has a single frame it is broadcast to match the video length.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVInpaintPreprocess",
            category="Lightricks/image_processing",
            description=(
                "Composites images with a green background where mask is "
                "active, for inpainting conditioning."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Video frames to composite onto the green background.",
                ),
                io.Mask.Input(
                    "mask",
                    tooltip="Mask indicating regions to replace with green. Single-frame masks are broadcast.",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> io.NodeOutput:
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]

        if mask.shape[0] == 1 and images.shape[0] > 1:
            mask = mask.expand(images.shape[0], -1, -1)

        min_frames = min(mask.shape[0], images.shape[0])
        mask = mask[:min_frames]
        images = images[:min_frames]

        mask_4d = mask.unsqueeze(-1)  # (B, H, W, 1) for broadcasting
        bg_color = torch.tensor(_BG_COLOR_RGB).float().to(images.device) / 255

        result = images * (1 - mask_4d) + bg_color.view(1, 1, 1, 3) * mask_4d
        return io.NodeOutput(result)
