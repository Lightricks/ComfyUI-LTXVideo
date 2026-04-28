"""Helpers for per-guide attention strength tracking in conditioning metadata."""


def _get_guide_attention_entries(conditioning):
    """Read the current guide_attention_entries list from conditioning."""
    for t in conditioning:
        entries = t[1].get("guide_attention_entries", None)
        if entries is not None:
            return entries
    return []


def _set_guide_attention_entries(conditioning, entries):
    """Write guide_attention_entries into conditioning (immutable update)."""
    import node_helpers

    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": entries}
    )


def append_guide_attention_entry(
    conditioning,
    pre_filter_count,
    latent_shape,
    attention_strength=1.0,
    attention_mask=None,
):
    """Append a new guide attention entry to conditioning metadata.

    Called by guide-adding nodes after appending tokens.

    Args:
        conditioning: ComfyUI conditioning list.
        pre_filter_count: Token count for this guide (before grid_mask filtering).
        latent_shape: [F, H, W] of the pre-dilation guide latent.
        attention_strength: Scalar in [0, 1]. 1.0 = full attention (default).
        attention_mask: Optional pixel-space mask tensor, shape (1, 1, F, H, W).

    Returns:
        Updated conditioning.
    """
    existing_entries = _get_guide_attention_entries(conditioning)
    entries = [*existing_entries]
    entries.append(
        {
            "pre_filter_count": pre_filter_count,
            "strength": attention_strength,
            "pixel_mask": attention_mask,
            "latent_shape": latent_shape,
        }
    )
    return _set_guide_attention_entries(conditioning, entries)


def normalize_mask(mask):
    """Normalize a ComfyUI MASK to (1, 1, F, H, W) for downstream processing.

    ComfyUI MASK type is typically (B, H, W) or (H, W).
    """
    if mask is None:
        return None
    if mask.dim() == 2:  # (H, W) → single frame
        return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:  # (F, H, W) → video mask
        return mask.unsqueeze(0).unsqueeze(0)
    return mask
