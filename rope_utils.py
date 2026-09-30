"""RoPE frequency helpers used by the embeddings connector.

These were previously imported from ``comfy.ldm.lightricks.model``, but upstream
ComfyUI does not ship them (it folds the equivalent logic into
``freqs_cis_matrix``). They are kept here, local to this node, so the connector
keeps its exact original behavior without relying on fork-only additions to core
ComfyUI.
"""

import torch


def interleaved_freqs_cis(freqs, pad_size):
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def split_freqs_cis(freqs, pad_size, num_attention_heads):
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    B, T, half_HD = cos_freq.shape

    cos_freq = cos_freq.reshape(
        B, T, num_attention_heads, half_HD // num_attention_heads
    )
    sin_freq = sin_freq.reshape(
        B, T, num_attention_heads, half_HD // num_attention_heads
    )

    cos_freq = torch.swapaxes(cos_freq, 1, 2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)
    return cos_freq, sin_freq
