"""Faster, exactly equivalent timestep encoding for the Exeggcute members (labelling only).

MetamonGroupedTstepEncoderV2 encodes the seven Pokemon of a decision with ONE perceiver call over all seven at once, using block-diagonal
attention masks so that each Pokemon only sees its own tokens. That is the same arithmetic as encoding each Pokemon separately with
the shared weights ("semantically identical", per its docstring), but the masked call computes a 7x larger attention than needed. Two
changes, neither of which alters the result beyond float rounding:

  1. Encode each Pokemon on its own (batch = number of Pokemon, no masks).
  2. A Pokemon's embedding is a function of its tokens and numbers alone, and between two steps of a game most of them do not change
     (the five reserves, most turns). Keep the last embedding of every (game, slot) and encode only the ones whose input changed.

`install(policy)` patches the members' encoders; `reset(policy)` must be called when a new batch of games starts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from amago.nets.utils import symlog

import os

COMPILE = bool(int(os.environ.get("FAST_TSTEP_COMPILE", "0")))  # compile the part after the Pokemon embeddings (global + fusion perceivers)


def _encode_independent(enc, text: torch.Tensor, nums: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """What `_encode_pokemon` computes for each of M Pokemon (text [M, T], nums [M, N], slot ids [M]); returns [M, d_fusion]."""
    tok_emb = enc.pokemon_token_emb(text).unsqueeze(1)
    seq = enc.pokemon_fuse(tok_emb, nums.unsqueeze(1)).squeeze(1)  # [M, seq_len, d]
    seq = seq + enc.pokemon_pos(enc._pokemon_pos_ids)
    if enc._pokemon_role_emb is not None:  # the role signal is added to each Pokemon's first token
        role = enc._pokemon_role_emb(enc._pokemon_role_ids)[slots]  # [M, d]
        seq = torch.cat([seq[:, :1] + role[:, None], seq[:, 1:]], dim=1)
    perceiver = enc.pokemon_perceiver
    latents = perceiver.latents.unsqueeze(0).expand(seq.shape[0], -1, -1)
    for layer in perceiver.layers:  # one group: every mask would be all-true
        latents = layer(latents, seq)
    latents = enc.pokemon_out_norm(latents).flatten(1)
    return enc.pokemon_proj(latents)


def _patch(enc) -> None:
    cache = {"text": None, "nums": None, "emb": None}
    enc._pokemon_cache = cache
    slots_all = torch.arange(enc.NUM_POKEMON)

    def pokemon_embeddings(text: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        B, P = text.shape[:2]
        same = None
        if cache["text"] is not None and cache["text"].shape == text.shape and cache["nums"].shape == nums.shape and cache["text"].device == text.device:
            same = (cache["text"] == text).all(-1) & (cache["nums"] == nums).all(-1)  # [B, P]
        changed = torch.ones(B, P, dtype=torch.bool, device=text.device) if same is None else ~same
        idx = changed.nonzero(as_tuple=True)
        fresh = _encode_independent(enc, text[idx], nums[idx], slots_all.to(text.device)[idx[1]])
        if same is None:
            emb = fresh.new_empty(B, P, fresh.shape[-1])
        else:
            emb = cache["emb"].to(fresh.dtype).clone()
        emb[idx] = fresh
        cache["text"], cache["nums"], cache["emb"] = text.clone(), nums.clone(), emb
        return emb

    def rest(pokemon_embs, rl2s_flat, global_nums_flat, global_text_flat):
        extras = F.leaky_relu(enc.extra_emb(symlog(rl2s_flat)))
        global_emb = enc._encode_global(global_text_flat, torch.cat([global_nums_flat, extras], dim=-1), None)
        all_embs = torch.cat([pokemon_embs, global_emb.unsqueeze(1)], dim=1) + enc.entity_type_emb(enc._entity_type_ids)
        emb = enc.fusion_out_norm(enc.fusion(all_embs, flatten=False))
        return emb.flatten(1)

    compiled = {"fn": torch.compile(rest, dynamic=True) if COMPILE else rest}

    def impl(pokemon_text, pokemon_nums, rl2s_flat, global_nums_flat, global_text_flat, log_dict=None):
        pokemon_embs = pokemon_embeddings(pokemon_text, pokemon_nums)
        try:
            return compiled["fn"](pokemon_embs, rl2s_flat, global_nums_flat, global_text_flat)
        except Exception:  # no working compiler on this machine: the same function, uncompiled
            if compiled["fn"] is rest:
                raise
            compiled["fn"] = rest
            return rest(pokemon_embs, rl2s_flat, global_nums_flat, global_text_flat)

    enc._fast_impl = impl
    enc._inner_forward_impl = impl


def install(policy) -> None:
    for member in policy.members:
        _patch(member.policy.tstep_encoder)


def enable(policy, on: bool) -> None:
    """Switches every member between this module's encoder and its own (compiled) one."""
    for member in policy.members:
        enc = member.policy.tstep_encoder
        if on:
            enc._inner_forward_impl = enc._fast_impl
        else:
            enc.__dict__.pop("_inner_forward_impl", None)


def reset(policy) -> None:
    for member in policy.members:
        cache = getattr(member.policy.tstep_encoder, "_pokemon_cache", None)
        if cache is not None:
            cache["text"] = cache["nums"] = cache["emb"] = None


def freeze_sigma_reparam(policy, half=None) -> int:
    """SigmaReparam linears (every projection of the trajectory transformers) rebuild their effective weight, `gamma / sigma * W`, on
    every call, which for a one-token step is a pass over the whole matrix that costs about as much as the matmul itself, and under
    autocast the matrix is then cast to half precision again on every call. The weights do not change during labelling, so compute
    the effective weight once, with sigma evaluated under the same autocast the labeler runs in (so it carries the same rounding as
    the original layer), and keep a half-precision copy for the autocast calls. Returns the number of layers frozen."""
    from amago.nets.transformer import SigmaReparam
    from einops import einsum

    frozen = 0
    for member in policy.members:
        kind = member.device.type
        autocast = dict(device_type=kind, dtype=half, enabled=half is not None and kind in ("cuda", "cpu"))
        for module in member.policy.modules():
            if isinstance(module, SigmaReparam) and not module.training:
                with torch.no_grad(), torch.autocast(**autocast):
                    sigma = einsum(module.u.detach(), module.weight, module.v.detach(), "d, d c , c->")
                    w_hat = (module.gamma / sigma * module.weight).detach().contiguous()
                bias = None if module.bias is None else module.bias.detach()
                module.forward = _frozen_linear(w_hat, bias)
                frozen += 1
    return frozen


def _frozen_linear(w_hat: torch.Tensor, bias):
    low: dict = {}

    def forward(x):
        kind = x.device.type
        if torch.is_autocast_enabled(kind):
            dtype = torch.get_autocast_dtype(kind)
            if dtype not in low:  # what autocast would compute from these tensors on every call, done once
                low[dtype] = (w_hat.to(dtype), None if bias is None else bias.to(dtype))
            weight, b = low[dtype]
            return F.linear(x, weight, b)
        return F.linear(x, w_hat, bias)

    return forward
