"""
Pytest configuration: sets up ComfyUI stubs so nodes can be imported
without a running ComfyUI environment, and handles the hyphenated package
name by creating a proper package alias.
"""

import importlib.util
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
PKG_NAME = "ComfyUI_LTXVideo"  # Python-safe alias for the repo folder

# ---------------------------------------------------------------------------
# Stub heavy ComfyUI / ML dependencies
# ---------------------------------------------------------------------------

comfy_pkg = types.ModuleType("comfy")
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mp = types.ModuleType("comfy.model_patcher")
comfy_pkg.model_management = comfy_mm
comfy_pkg.model_patcher = comfy_mp
sys.modules.setdefault("comfy", comfy_pkg)
sys.modules.setdefault("comfy.model_management", comfy_mm)
sys.modules.setdefault("comfy.model_patcher", comfy_mp)

fp = types.ModuleType("folder_paths")
fp.models_dir = "/tmp/models"
fp.get_filename_list = lambda _: []
fp.get_full_path_or_raise = lambda *a: "/tmp/fake"
sys.modules.setdefault("folder_paths", fp)

torch_stub = types.ModuleType("torch")
torch_stub.Tensor = object  # minimal stub
sys.modules.setdefault("torch", torch_stub)

PIL_stub = types.ModuleType("PIL")
PIL_Image_stub = types.ModuleType("PIL.Image")
PIL_Image_stub.Image = object  # minimal stub
PIL_stub.Image = PIL_Image_stub
sys.modules.setdefault("PIL", PIL_stub)
sys.modules.setdefault("PIL.Image", PIL_Image_stub)

for _mod in [
    "transformers",
    "safetensors",
    "safetensors.torch",
    "einops",
    "kornia",
    "kornia.geometry",
    "huggingface_hub",
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# ---------------------------------------------------------------------------
# Register the repo root as a Python package so relative imports work
# ---------------------------------------------------------------------------

pkg = types.ModuleType(PKG_NAME)
# Explicitly do NOT set __path__ or __file__ to prevent pytest from
# traversing the repo root and trying to import its __init__.py
pkg.__package__ = PKG_NAME
pkg.__spec__ = None
sys.modules[PKG_NAME] = pkg

# With --import-mode=importlib, pytest falls back to module_name_from_path()
# which produces "__init__" as the module name for the repo's __init__.py.
# Pre-register it so pytest finds it in sys.modules and skips exec'ing it.
_init_stub = types.ModuleType("__init__")
_init_stub.__package__ = PKG_NAME
sys.modules["__init__"] = _init_stub


def _load_submodule(name: str) -> types.ModuleType:
    """Load a .py file from REPO_ROOT as PKG_NAME.name."""
    full_name = f"{PKG_NAME}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    path = os.path.join(REPO_ROOT, f"{name}.py")
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = PKG_NAME
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    setattr(pkg, name, mod)
    return mod


# Pre-load the modules that tests depend on
_load_submodule("nodes_registry")
_load_submodule("prompt_enhancer_utils")
_load_submodule("minimax_prompt_enhancer")
