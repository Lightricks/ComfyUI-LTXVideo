import importlib
import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent
comfy_root = repo_root.parent.parent
if str(comfy_root) not in sys.path:
    sys.path.insert(0, str(comfy_root))

pb = importlib.import_module("custom_nodes.ComfyUI-LTXVideo.pyramid_blending")


def test_imports():
    assert pb.LTXVLaplacianPyramidBlend is not None


def test_pad_for_laplacian():
    x = torch.zeros(1, 3, 100, 100)
    padded, (pad_r, pad_d) = pb._pad_for_laplacian(x)
    assert pad_r == 28
    assert pad_d == 28
    assert padded.shape == (1, 3, 128, 128)


def test_gaussian_pyramid():
    x = torch.zeros(1, 3, 100, 100)
    pyr = pb._gaussian_pyramid(x, max_level=3)
    assert len(pyr) == 3


def test_pyramid_blend():
    img1 = torch.ones(1, 3, 64, 64)
    img2 = torch.zeros(1, 3, 64, 64)
    mask = torch.full((1, 1, 64, 64), 0.5)
    blended = pb._pyramid_blend(img1, img2, mask, max_level=3)
    assert blended.shape == (1, 3, 64, 64)
