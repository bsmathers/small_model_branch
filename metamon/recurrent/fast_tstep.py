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

    def impl(pokemon_text, pokemon_nums, rl2s_flat, global_nums_flat, global_text_flat, log_dict=None):
        pokemon_embs = pokemon_embeddings(pokemon_text, pokemon_nums)
        extras = F.leaky_relu(enc.extra_emb(symlog(rl2s_flat)))
        global_emb = enc._encode_global(global_text_flat, torch.cat([global_nums_flat, extras], dim=-1), log_dict)
        all_embs = torch.cat([pokemon_embs, global_emb.unsqueeze(1)], dim=1) + enc.entity_type_emb(enc._entity_type_ids)
        emb = enc.fusion_out_norm(enc.fusion(all_embs, flatten=False))
        return emb.flatten(1)

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
