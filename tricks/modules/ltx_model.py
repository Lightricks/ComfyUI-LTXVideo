import math

import comfy.ldm.common_dit
import comfy.ldm.modules.attention
import torch
from comfy.ldm.lightricks.model import (
    BasicTransformerBlock,
    LTXVModel,
    apply_rotary_emb,
)
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from torch import nn

from ..utils.feta_enhance_utils import get_feta_scores


class LTXModifiedCrossAttention(nn.Module):
    def forward(self, x, context=None, mask=None, pe=None, transformer_options={}):
        context = x if context is None else context
        context_v = x if context is None else context

        step = transformer_options.get("step", -1)
        total_steps = transformer_options.get("total_steps", 0)
        attn_bank = transformer_options.get("attn_bank", None)
        sample_mode = transformer_options.get("sample_mode", None)
        if attn_bank is not None and self.idx in attn_bank["block_map"]:
            len_conds = len(transformer_options["cond_or_uncond"])
            pred_order = transformer_options["pred_order"]
            if (
                sample_mode == "forward"
                and total_steps - step - 1 < attn_bank["save_steps"]
            ):
                step_idx = f"{pred_order}_{total_steps-step-1}"
                attn_bank["block_map"][self.idx][step_idx] = x.cpu()
            elif sample_mode == "reverse" and step < attn_bank["inject_steps"]:
                step_idx = f"{pred_order}_{step}"
                inject_settings = attn_bank.get("inject_settings", {})
                if len(inject_settings) > 0:
                    inj = (
                        attn_bank["block_map"][self.idx][step_idx]
                        .to(x.device)
                        .repeat(len_conds, 1, 1)
                    )
                if "q" in inject_settings:
                    x = inj
                if "k" in inject_settings:
                    context = inj
                if "v" in inject_settings:
                    context_v = inj

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context_v)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, pe)

        feta_score = None
        if (
            transformer_options.get("feta_weight", 0) > 0
            and self.idx in transformer_options["feta_layers"]["layers"]
        ):
            feta_score = get_feta_scores(q, k, self.heads, transformer_options)

        alt_attn_fn = (
            transformer_options.get("patches_replace", {})
            .get("layer", {})
            .get(("self_attn", self.idx), None)
        )
        if alt_attn_fn is not None:
            out = alt_attn_fn(
                q,
                k,
                v,
                self.heads,
                attn_precision=self.attn_precision,
                transformer_options=transformer_options,
            )
        elif mask is None:
            out = comfy.ldm.modules.attention.optimized_attention(
                q, k, v, self.heads, attn_precision=self.attn_precision
            )
        else:
            out = comfy.ldm.modules.attention.optimized_attention_masked(
                q, k, v, self.heads, mask, attn_precision=self.attn_precision
            )

        if feta_score is not None:
            out *= feta_score

        return self.to_out(out)


class LTXModifiedBasicTransformerBlock(BasicTransformerBlock):
    def forward(
        self,
        x,
        context=None,
        attention_mask=None,
        timestep=None,
        pe=None,
        transformer_options={},
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + timestep.reshape(
                x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1
            )
        ).unbind(dim=2)
        x += (
            self.attn1(
                comfy.ldm.common_dit.rms_norm(x) * (1 + scale_msa) + shift_msa,
                pe=pe,
                transformer_options=transformer_options,
            )
            * gate_msa
        )

        x += self.attn2(x, context=context, mask=attention_mask)

        y = comfy.ldm.common_dit.rms_norm(x) * (1 + scale_mlp) + shift_mlp
        x += self.ff(y) * gate_mlp

        return x


class LTXVModelModified(LTXVModel):

    def forward(
        self,
        x,
        timestep,
        context,
        attention_mask,
        frame_rate=25,
        transformer_options={},
        keyframe_idxs=None,
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})

        orig_shape = list(x.shape)

        x, latent_coords = self.patchifier.patchify(x)
        pixel_coords = latent_to_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.vae_scale_factors,
            causal_fix=self.causal_temporal_positioning,
        )

        if keyframe_idxs is not None:
            pixel_coords[:, :, -keyframe_idxs.shape[2] :] = keyframe_idxs

        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

        x = self.patchify_proj(x)
        timestep = timestep * 1000.0

        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ) * torch.finfo(x.dtype).max

        pe = self._precompute_freqs_cis(
            fractional_coords, dim=self.inner_dim, out_dtype=x.dtype
        )

        batch_size = x.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=x.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = x.shape[0]
            context = self.caption_projection(context)
            context = context.view(batch_size, -1, x.shape[-1])

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"] = block(
                        args["img"],
                        context=args["txt"],
                        attention_mask=args["attention_mask"],
                        timestep=args["vec"],
                        pe=args["pe"],
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {
                        "img": x,
                        "txt": context,
                        "attention_mask": attention_mask,
                        "vec": timestep,
                        "pe": pe,
                    },
                    {"original_block": block_wrap},
                )
                x = out["img"]
            else:
                x = block(
                    x,
                    context=context,
                    attention_mask=attention_mask,
                    timestep=timestep,
                    pe=pe,
                    transformer_options=transformer_options,
                )

        # 3. Output
        scale_shift_values = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = self.norm_out(x)
        # Modulation
        x = x * (1 + scale) + shift
        x = self.proj_out(x)

        x = self.patchifier.unpatchify(
            latents=x,
            output_height=orig_shape[3],
            output_width=orig_shape[4],
            output_num_frames=orig_shape[2],
            out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
        )

        return x


def inject_model(diffusion_model):
    diffusion_model.__class__ = LTXVModelModified
    for idx, transformer_block in enumerate(diffusion_model.transformer_blocks):
        transformer_block.__class__ = LTXModifiedBasicTransformerBlock
        transformer_block.idx = idx
        transformer_block.attn1.__class__ = LTXModifiedCrossAttention
        transformer_block.attn1.idx = idx
    return diffusion_model
