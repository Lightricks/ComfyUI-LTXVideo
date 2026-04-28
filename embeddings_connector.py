import math
from typing import Optional

import comfy.ldm.common_dit
import comfy.ops
import torch
from comfy.ldm.lightricks.model import (
    CrossAttention,
    FeedForward,
    LTXFrequenciesPrecision,
    LTXRopeType,
    generate_freq_grid_np,
    interleaved_freqs_cis,
    split_freqs_cis,
)
from torch import nn

from .pos_embedding_exp_values import POS_EMBEDDING_EXP_VALUES


class BasicTransformerBlock1D(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:

        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        standardization_norm (`str`, *optional*, defaults to `"layer_norm"`): The type of pre-normalization to use. Can be `"layer_norm"` or `"rms_norm"`.
        norm_eps (`float`, *optional*, defaults to 1e-5): Epsilon value for normalization layers.
        qk_norm (`str`, *optional*, defaults to None):
            Set to 'layer_norm' or `rms_norm` to perform query and key normalization.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*): Dimension of the inner feed-forward layer. If not provided, defaults to `dim * 4`.
        ff_bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the feed-forward layer.
        attention_out_bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the attention output layer.
        use_rope (`bool`, *optional*, defaults to `False`): Whether to use Rotary Position Embeddings (RoPE).
        ffn_dim_mult (`int`, *optional*, defaults to 4): Multiplier for the inner dimension of the feed-forward layer.
    """

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        context_dim=None,
        attn_precision=None,
        apply_gated_attention=False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            context_dim=None,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dim_out=dim,
            glu=True,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(self, hidden_states, attention_mask=None, pe=None) -> torch.FloatTensor:

        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = comfy.ldm.common_dit.rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = comfy.ldm.common_dit.rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels=128,
        cross_attention_dim=2048,
        attention_head_dim=128,
        num_attention_heads=30,
        num_layers=2,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[1],
        causal_temporal_positioning=False,
        num_learnable_registers: Optional[int] = 128,
        apply_gated_attention=False,
        dtype=None,
        device=None,
        operations=None,
        split_rope=False,
        double_precision_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.out_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.split_rope = split_rope
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = nn.ModuleList(
            [
                BasicTransformerBlock1D(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    context_dim=cross_attention_dim,
                    apply_gated_attention=apply_gated_attention,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(num_layers)
            ]
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = nn.Parameter(
                torch.rand(
                    self.num_learnable_registers, inner_dim, dtype=dtype, device=device
                )
                * 2.0
                - 1.0
            )

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(1)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs(self, indices_grid, spacing):
        source_dtype = indices_grid.dtype
        dtype = (
            torch.float32
            if source_dtype in (torch.bfloat16, torch.float16)
            else source_dtype
        )

        fractional_positions = self.get_fractional_positions(indices_grid)
        indices = (
            generate_freq_grid_np(
                self.positional_embedding_theta,
                indices_grid.shape[1],
                self.inner_dim,
            )
            if self.double_precision_rope
            else self.generate_freq_grid(spacing, dtype, fractional_positions.device)
        ).to(device=fractional_positions.device)

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )
        return freqs

    def generate_freq_grid(self, spacing, dtype, device):
        dim = self.inner_dim
        theta = self.positional_embedding_theta
        n_pos_dims = 1
        n_elem = 2 * n_pos_dims  # 2 for cos and sin e.g. x 3 = 6
        start = 1
        end = theta

        if spacing == "exp":
            # Workaround float precision issues when calculating positional embeddings exponents to ensure compatibility with JAX.
            indices = torch.tensor(POS_EMBEDDING_EXP_VALUES, dtype=dtype, device=device)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, n_elem, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(
                start, end, dim // n_elem, device=device, dtype=dtype
            )
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // n_elem, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        return indices

    def precompute_freqs_cis(self, indices_grid, spacing="exp"):
        dim = self.inner_dim
        n_elem = 2  # 2 because of cos and sin
        freqs = self.precompute_freqs(indices_grid, spacing)
        if self.split_rope:
            expected_freqs = dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq, sin_freq = split_freqs_cis(
                freqs, pad_size, self.num_attention_heads
            )
        else:
            cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
        return cos_freq.to(self.dtype), sin_freq.to(self.dtype), self.split_rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            indices_grid (`torch.LongTensor` of shape `(batch size, 3, num latent pixels)`):
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input

        if self.num_learnable_registers:
            # replace all padded tokens with learnable registers
            assert (
                hidden_states.shape[1] % self.num_learnable_registers == 0
            ), f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by num_learnable_registers {self.num_learnable_registers}."

            num_registers_duplications = (
                hidden_states.shape[1] // self.num_learnable_registers
            )
            learnable_registers = torch.tile(
                self.learnable_registers, (num_registers_duplications, 1)
            ).to(hidden_states.device)

            attention_mask_binary = (
                attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0
            ).int()

            non_zero_hidden_states = hidden_states[
                :, attention_mask_binary.squeeze().bool(), :
            ]
            non_zero_nums = non_zero_hidden_states.shape[1]
            pad_length = hidden_states.shape[1] - non_zero_nums
            adjusted_hidden_states = torch.nn.functional.pad(
                non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0
            )
            flipped_mask = torch.flip(attention_mask_binary, dims=[1])
            hidden_states = (
                flipped_mask * adjusted_hidden_states
                + (1 - flipped_mask) * learnable_registers
            )

            attention_mask = torch.full_like(
                attention_mask,
                0.0,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        indices_grid = torch.arange(
            hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device
        )
        indices_grid = indices_grid[None, None, :]
        freqs_cis = self.precompute_freqs_cis(indices_grid)

        # 2. Blocks
        for block_idx, block in enumerate(self.transformer_1d_blocks):
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, pe=freqs_cis
            )

        # 3. Output
        # if self.output_scale is not None:
        #     hidden_states = hidden_states / self.output_scale

        hidden_states = comfy.ldm.common_dit.rms_norm(hidden_states)

        return hidden_states, attention_mask


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_PREFIX_BASE = "model.diffusion_model."


def load_embeddings_connector(
    sd,
    connector_prefix,
    connector_config,
    dtype=torch.bfloat16,
    rope_type=LTXRopeType.INTERLEAVED,
    frequencies_precision=LTXFrequenciesPrecision.FLOAT32,
    pe_max_pos=None,
):
    sd_connector = {
        k[len(connector_prefix) :]: v
        for k, v in sd.items()
        if k.startswith(connector_prefix)
    }

    if len(sd_connector) == 0:
        return None

    operations = comfy.ops.pick_operations(dtype, dtype, disable_fast_fp8=True)
    connector = Embeddings1DConnector(
        num_attention_heads=connector_config["num_attention_heads"],
        attention_head_dim=connector_config["attention_head_dim"],
        num_layers=connector_config["num_layers"],
        apply_gated_attention=connector_config.get("apply_gated_attention", False),
        dtype=dtype,
        operations=operations,
        positional_embedding_max_pos=pe_max_pos if pe_max_pos is not None else [1],
        split_rope=rope_type == LTXRopeType.SPLIT,
        double_precision_rope=frequencies_precision == LTXFrequenciesPrecision.FLOAT64,
    )
    connector.load_state_dict(sd_connector)
    return connector


def load_video_embeddings_connector(sd, transformer_config, dtype=torch.bfloat16):
    rope_type = LTXRopeType.from_dict(transformer_config)
    frequencies_precision = LTXFrequenciesPrecision.from_dict(transformer_config)
    pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

    video_only_connector_prefix = f"{_PREFIX_BASE}embeddings_connector."
    av_connector_prefix = f"{_PREFIX_BASE}video_embeddings_connector."
    prefix = (
        av_connector_prefix
        if f"{_PREFIX_BASE}audio_adaln_single.linear.weight" in sd
        else video_only_connector_prefix
    )

    connector_config = {
        "num_attention_heads": transformer_config.get(
            "connector_num_attention_heads", 30
        ),
        "attention_head_dim": transformer_config.get(
            "connector_attention_head_dim", 128
        ),
        "num_layers": transformer_config.get("connector_num_layers", 2),
        "apply_gated_attention": transformer_config.get(
            "connector_apply_gated_attention", False
        ),
    }

    return load_embeddings_connector(
        sd,
        prefix,
        connector_config,
        dtype,
        rope_type,
        frequencies_precision,
        pe_max_pos,
    )


def load_audio_embeddings_connector(sd, transformer_config, dtype=torch.bfloat16):
    rope_type = LTXRopeType.from_dict(transformer_config)
    frequencies_precision = LTXFrequenciesPrecision.from_dict(transformer_config)
    pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

    connector_config = {
        "num_attention_heads": transformer_config.get(
            "audio_connector_num_attention_heads",
            transformer_config.get("connector_num_attention_heads", 30),
        ),
        "attention_head_dim": transformer_config.get(
            "audio_connector_attention_head_dim",
            transformer_config.get("connector_attention_head_dim", 128),
        ),
        "num_layers": transformer_config.get(
            "audio_connector_num_layers",
            transformer_config.get("connector_num_layers", 2),
        ),
        "apply_gated_attention": transformer_config.get(
            "connector_apply_gated_attention", False
        ),
    }

    return load_embeddings_connector(
        sd,
        f"{_PREFIX_BASE}audio_embeddings_connector.",
        connector_config,
        dtype,
        rope_type,
        frequencies_precision,
        pe_max_pos,
    )
