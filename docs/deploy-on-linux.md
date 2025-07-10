# 📦 ComfyUI-LTXVideo Provisioning Summary

For easily deploy with Vast.ai, check out [this template](https://cloud.vast.ai/?ref_id=276779&creator_id=276779&name=ComfyUI%20%2B%20LTX%20Video%20Lite).

[The template script](https://gist.githubusercontent.com/ElishaKay/f92e86c2d43be9de20088991b89b0228/raw/419f67dcb0c393232d2745f63795624f6dfb0fea/ltx-video-lite.sh) provisions a full **ComfyUI environment with LTXVideo support** by downloading required nodes, models, workflows, and inputs.

Please note: the downloading stage of the relevant assets (listed below) can take upwards of an hour and may prevent the ComfyUI frontend from displaying during this time. To view the status of your downloads and machine setup, visit: `<your-ip:port>/#/logs`.

---

## 📦 1. Custom Nodes (Plugins)

These GitHub repositories are cloned into:

```
${COMFYUI_DIR}/custom_nodes/
```

| Node Name | Source Repo | Purpose |
|-----------|-------------|---------|
| **LTXVideo** | [`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo) | Core video generation node (I2V / T2V) |
| **KJNodes** | [`kijai/ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes) | Utility and general-purpose nodes |
| **LogicUtils** | [`aria1th/ComfyUI-LogicUtils`](https://github.com/aria1th/ComfyUI-LogicUtils) | Logic and control flow nodes |
| **Mattabyte Registry** | [`Mattabyte/ComfyUI-LTXVideo-Registry_Mattabyte`](https://github.com/Mattabyte/ComfyUI-LTXVideo-Registry_Mattabyte) | Additional registry support |
| **VideoHelperSuite** | [`Kosinkadink/ComfyUI-VideoHelperSuite`](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | Frame/video utility nodes |

---

## 🧠 2. Model Files

### ✅ Checkpoint Models

Downloaded into:

```
${COMFYUI_DIR}/models/checkpoints/
```

| File | Source |
|------|--------|
| `ltx-video-2b-v0.9.5.safetensors` | [HuggingFace - Lightricks](https://huggingface.co/Lightricks/LTX-Video) |

---

### ✅ CLIP Models

Downloaded into:

```
${COMFYUI_DIR}/models/clip/
```

| File | Source |
|------|--------|
| `t5xxl_fp16.safetensors` | [HuggingFace - comfyanonymous](https://huggingface.co/comfyanonymous/flux_text_encoders) |

---

## 🖼️ 3. Input Files

Downloaded into:

```
${COMFYUI_DIR}/input/
```

| File | Source |
|------|--------|
| `island.jpg` | [ComfyUI Example](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/island.jpg) |

---

## 🧰 4. Workflows

Downloaded into:

```
${COMFYUI_DIR}/user/default/workflows/
```

| Workflow | Description |
|----------|-------------|
| `ltx-video-i2v-simple.json` | Image-to-Video example workflow |
| `ltx-video-t2v-simple.json` | Text-to-Video example workflow |

---

## 🚫 5. Empty/Unused Model Categories

These categories are defined in the script but **no URLs were provided**, so they are skipped during provisioning:

- `UNET_MODELS`
- `LORA_MODELS`
- `VAE_MODELS`
- `ESRGAN_MODELS`
- `CONTROLNET_MODELS`

---

## ✅ Summary of What’s Pulled

| Component Type | Count | Destination |
|----------------|-------|-------------|
| Custom Nodes | 5 | `custom_nodes/` |
| Checkpoint Models | 1 | `models/checkpoints/` |
| CLIP Models | 1 | `models/clip/` |
| Workflows | 2 | `user/default/workflows/` |
| Input Images | 1 | `input/` |

---
