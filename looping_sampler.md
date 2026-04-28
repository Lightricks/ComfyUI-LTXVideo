# LTXVLoopingSampler Documentation

## Overview

The **LTXVLoopingSampler** is a unified ComfyUI node designed for generating long, high-resolution videos through versatile guidance and conditioningi techniques, as well as temporal and spatial tiling. It overcomes memory and computational limitations by breaking video generation into manageable chunks, enabling the creation of videos that would otherwise be impossible to generate in a single model run.
It provides a multitude of features (autoregressive long vidoes, keyframes, guidance using IC-LoRA video modalities, normalization, 'long memory' with conditioning on negative positional encodings, and more ) all in one place.

The **MultiPromptProvider** utility node for providing dynamic prompts per temporal tile.

---

## LTXVLoopingSampler

### Core Concept

The LTXVLoopingSampler addresses two fundamental challenges in video generation:

1. **Temporal Length**: Generating very long videos by dividing them into overlapping temporal segments
2. **Spatial Resolution**: Creating high-resolution frames by splitting them into overlapping spatial regions

Each "tile" is processed independently and then seamlessly blended with its neighbors to create the final coherent video output.

### Key Features

- **🎬 Temporal Tiling**: Generates long videos by processing overlapping time segments
- **🖼️ Spatial Tiling**: Creates high-resolution output by dividing frames into spatial regions
- **🎯 Multiple Conditioning Modes**: Supports image conditioning, guiding latents, and negative index conditioning
- **⚡ Memory Efficient**: Processes one spatial tile at a time to minimize memory usage
- **🔄 Seamless Blending**: Uses weighted blending for smooth transitions between tiles
- **🎲 Advanced Seeding**: Configurable per-tile seeding for reproducible results

### Interface Reference

#### Required Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **model** | MODEL | - | - | The diffusion model for video generation |
| **vae** | VAE | - | - | VAE for encoding/decoding between pixel and latent space |
| **noise** | NOISE | - | - | Noise generator for the sampling process |
| **sampler** | SAMPLER | - | - | Sampling algorithm (e.g., Euler, DPM++) |
| **sigmas** | SIGMAS | - | - | Noise schedule defining the denoising steps |
| **guider** | GUIDER | - | - | Conditioning guider (must be STGGuiderAdvanced to work with LTXVLoopingSampler) |
| **latents** | LATENT | - | - | Input latent tensor defining video dimensions and length. Non empty latents can be passed here to enable a partial denoising use case. |

#### Temporal Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **temporal_tile_size** | INT | 80 | 24-1000 (step: 8) | Frames per temporal tile (in pixel space). The first tile will be one frame longer than this number, and all new tiles will add `temporal_tile_size-temporal_overlap` new frames each. |
| **temporal_overlap** | INT | 24 | 16-80 (step: 8) | Overlapping frames between consecutive tiles |
| **temporal_overlap_cond_strength** | FLOAT | 0.5 | 0.0-1.0 | How strongly the final frames of the previous tile condition the new tile generation |

#### Spatial Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **horizontal_tiles** | INT | 1 | 1-6 | Number of horizontal spatial divisions |
| **vertical_tiles** | INT | 1 | 1-6 | Number of vertical spatial divisions |
| **spatial_overlap** | INT | 1 | 1-8 | Overlapping latent 'pixels' between spatial tiles (latent space) |

#### Conditioning Controls

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **guiding_strength** | FLOAT | 1.0 | 0.0-1.0 | Influence of guiding latents on generation, if provided.  |
| **cond_image_strength** | FLOAT | 1.0 | 0.0-1.0 | Influence of conditioning images on first image in classical i2v setup, and of all keyframes. |

#### Optional Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **optional_cond_images** | IMAGE | None | Images for I2V conditioning (resized to match output using center crop operation) |
| **optional_guiding_latents** | LATENT | None | Latents for guided generation (usually used with IC-LoRA loaded) |
| **adain_factor** | FLOAT | 0.0 (0.0-1.0) | AdaIn normalization strength to prevent oversaturation. AdaIn is applied on each new temporal tile, with respect to the latent statistics of the first temporal tile. |
| **optional_positive_conditionings** | CONDITIONING | None | Per-tile prompts (from MultiPromptProvider) |
| **optional_negative_index_latents** | LATENT | None | Context latents for long-term coherence |
| **guiding_start_step** | INT | 0 (0-1000) | Step to begin applying guiding latents, if provided. |
| **guiding_end_step** | INT | 1000 (0-1000) | Step to stop applying guiding latents, if provided. |
| **optional_cond_image_indices** | STRING | "0" | Keyframe indices for conditioning (comma-separated list of indices) |
| **optional_normalizing_latents** | LATENT | None | Reference latents for output normalization. If provided, used in the AdaIn operation, each spatio-temporal tile with its counterpart, instead of using the first temporal tile as a normalization refernece.  |

### Output

| Output | Type | Description |
|--------|------|-------------|
| **denoised_output** | LATENT | Complete video latents after tiling and blending |

### Generation Modes

#### 🎨 Text-to-Video (T2V)
Pure text-driven video generation starting from noise.

**Setup:**
- Don't connect `optional_cond_images`
- Provide text conditioning through the guider
- Video generates from pure noise

#### 🖼️ Image-to-Video (I2V)
Video generation conditioned on input images.

**Setup:**
- Connect a single image to `optional_cond_images` and set `optional_cond_image_indices` to "0".
- Images are automatically resized to match output dimensions

#### 🗝️ Keyframe-Based Generation

Keyframe-based generation allows you to guide the video at specific frames using conditioning images (keyframes). This is useful for tasks like animating a sequence between important visual states or ensuring the video matches certain reference images at chosen points.

- Use `optional_cond_images` to provide one or more keyframe images.
- Use `optional_cond_image_indices` to specify which frames should be conditioned on these images. This should be a comma-separated list of frame indices (e.g., `"0, 32, 64"`).
- Each image in `optional_cond_images` will be applied to the corresponding frame index in `optional_cond_image_indices`.
- The node will automatically resize and center-crop the images to match the output dimensions.

**Example:**
- To condition the first and last frames of a 64-frame video, set:
  - `optional_cond_images`: [start_image, end_image]
  - `optional_cond_image_indices`: `"0, 63"`

This will ensure the generated video starts and ends with the provided keyframes, smoothly interpolating between them.

**Tip:** You can combine keyframe conditioning with text prompts and guiding latents for even more control over the video generation process.

#### 🎯 Guided Generation
Advanced control using guiding latents (often with IC-LoRA).

**Setup:**
- Connect guiding latents to `optional_guiding_latents`
- Adjust `guiding_strength` (0.0-1.0)
- Set `guiding_start_step` and `guiding_end_step` for control over the effect of the guidance during the denoising steps.

#### 📏 High-Resolution Generation
Create videos larger than base model resolution using spatial tiling.

**Configuration Example:**
```
horizontal_tiles: 2
vertical_tiles: 2
spatial_overlap: 2
```
This creates a 2x2 grid of spatial tiles with 2-pixel overlap for seamless blending.

#### ⏱️ Long Video Generation
Generate extended sequences using temporal tiling.

**How It Works:**
- Each temporal tile is conditioned on the **end frames** of the previous tile
- This creates seamless motion continuity across tile boundaries
- The `temporal_overlap` defines how many frames from the previous tile are used for conditioning

**Best Practices:**
- Use `temporal_overlap` = 1/3 of `temporal_tile_size`
- Higher `temporal_overlap_cond_strength` = stronger temporal consistency
- Set `adain_factor` = 0.1-0.3 for long sequences to prevent oversaturation
- Consider using different prompts per tile via MultiPromptProvider

#### 🧩 Negative Index Latents for Enhanced Coherence

You can provide `optional_negative_index_latents` to further improve long video consistency. This feature allows each temporal chunk to attend to an additional latent (or set of latents), typically derived from another image or video, to guide the generation process and maintain global coherence.

- **How it works:**
  Each temporal chunk is conditioned on the provided negative index latent(s) using *negative positional embeddings*. This mechanism helps the model reference global context or style cues throughout the video, reducing drift and improving overall consistency, especially in long sequences.

- **Usage:**
  - Connect your reference latent(s) to the `optional_negative_index_latents` input.

- **Typical applications:**
  - Maintaining a consistent subject or style across long videos
  - Referencing a global template or anchor image
  - Enforcing scene or character coherence

**Tip:** You can combine negative index latents with keyframes, guiding latents, and text prompts for maximum control.

### Technical Implementation

#### Temporal Tiling Algorithm

The node processes video in overlapping temporal chunks, with each tile conditioned on the preceding tile's output:

1. **First Tile (0 → temporal_tile_size)**
   - Uses `LTXVInContextSampler` (with guidance) or `LTXVBaseSampler` (without)
   - Establishes the foundation for the video sequence
   - Generated from pure noise (T2V) or conditioning images (I2V / keyframes)

2. **Subsequent Tiles**
   - Uses `LTXVExtendSampler` to extend from previous tile's results
   - **Conditioning Mechanism**: Each new tile is conditioned on the **final frames** (overlap region) of the preceding tile
   - The overlap region from the previous tile provides temporal continuity and motion consistency
   - `temporal_overlap_cond_strength` controls how strongly the previous tile's end frames influence the new generation
   - Each tile advances by `(temporal_tile_size - temporal_overlap)` frames, creating seamless temporal progression
   - Guiding latents, if provided, are used outside of the overlap region.
   - Conditioning images (`optional_cond_images`), if provided, are used to condition the generation as well.

#### Spatial Tiling Algorithm

For high-resolution generation, frames are divided spatially:

1. **Tile Calculation**: Frame divided into overlapping regions
2. **Independent Processing**: Each spatial region processed separately
3. **Weight Generation**: Linear blending weights created for overlap regions
4. **Final Compositing**: Weighted accumulation produces seamless result

#### Seed Management

Each tile receives a unique, deterministic seed:
```
tile_seed = base_seed + temporal_offset + spatial_offset + custom_offset (hidedn interface)
```

This ensures:
- Reproducible results across runs
- Consistent randomness per tile
- Ability to re-generate specific tiles

#### Memory Optimization Strategies

- **Sequential Spatial Processing**: Only one spatial tile in memory at a time
- **Temporal Chunking**: Long videos processed in manageable segments
- **Weighted Accumulation**: Final output built incrementally
- **Garbage Collection**: Intermediate results freed after use

### Configuration Guidelines

#### Temporal Parameters
- **Short Videos (< 5 seconds)**: `temporal_tile_size: 40-80`
- **Medium Videos (5-15 seconds)**: `temporal_tile_size: 80-120`
- **Long Videos (> 15 seconds)**: `temporal_tile_size: 120-200`
- **Overlap Rule**: Use 25-30% of tile size for smooth transitions

#### Spatial Parameters
- **Standard Resolution**: `horizontal_tiles: 1, vertical_tiles: 1`
- **2K Generation**: `horizontal_tiles: 2, vertical_tiles: 1-2`
- **4K Generation**: `horizontal_tiles: 2-3, vertical_tiles: 2-3`
- **Overlap Minimum**: At least 1 pixel, increase for better blending

#### Quality Settings
- **High Quality**: Lower `adain_factor` (0.0-0.1), higher overlap
- **Fast Generation**: Higher `adain_factor` (0.2-0.5), lower overlap
- **Temporal Consistency**: Higher `temporal_overlap_cond_strength`

### Common Use Cases

#### Creating Cinematic Sequences
```
temporal_tile_size: 80
temporal_overlap: 24
horizontal_tiles: 2
vertical_tiles: 1
adain_factor: 0.1
```

#### Long-Form Content Generation
```
temporal_tile_size: 120
temporal_overlap: 40
adain_factor: 0.2
+ MultiPromptProvider for narrative progression
```

#### High-Resolution Showcase Videos
```
horizontal_tiles: 3
vertical_tiles: 2
spatial_overlap: 3
temporal_tile_size: 60
```

### Troubleshooting

#### Common Issues
- **Visible Seams**: Increase spatial/temporal overlap
- **Oversaturation**: Increase `adain_factor`
- **Memory Errors**: Reduce tile sizes or use fewer spatial tiles
- **Inconsistent Motion**: Increase `temporal_overlap` and `temporal_overlap_cond_strength`
- **Temporal Discontinuities**: The end frames of each tile condition the next tile - ensure sufficient overlap
- **Color Shifts**: Use `optional_normalizing_latents` for reference and/or increase `adain_factor`

#### Performance Tips
- Process spatial tiles sequentially to minimize memory usage
- Use appropriate tile sizes for your hardware capabilities
- Consider temporal overlap vs. generation speed trade-offs

---

## MultiPromptProvider

A companion utility node that enables dynamic prompt changes across temporal tiles.
- Connect multiple pipe-separated prompts (`"prompt1|prompt2|prompt3"`) to create evolving narratives throughout video generation.
- Each prompt is used in one temporal_tile in `LTXVLoopingSampler`.
- If not enough prompts are provided, the last one is repeated.
- If too many prompts provided, the last ones are ignored.

---
