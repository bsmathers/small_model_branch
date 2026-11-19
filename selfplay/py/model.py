"""The self-play model (~5.2M parameters, SELFPLAY_DESIGN.md): one recurrent step per decision of one player.

Inputs (one environment row, selfplay/zig/src/env.zig): the board (ids / floats: the 1.3M model's input layout), the log
events since the player's previous decision (events [32, 12], n_events) and the carried state h_prev [d].

    log encoder   each event's 12 fields -> 32-wide embeddings, concatenated (384) -> MLP 384 -> 512 -> d -> LayerNorm; one
                  transformer layer over the events (position embeddings, padding masked) with a summary token
    core          16 tokens: carried state, log summary, 6 own Pokemon, 6 opponent Pokemon, field, value token; L pre-LayerNorm
                  transformer blocks (width d, 32-wide heads, feed-forward ff); the state token's output is the next state
    heads         value: 3 logits (this player wins / loses / ties) from the value token; policy: a pointer head over the 9
                  slots, score = query(state) . key(option) / sqrt(d), keys: moves = MLP(move embedding, PP, the active
                  Pokemon's output), switches = a projection of that reserve Pokemon's output; illegal slots masked
The board's "last turn" fields are not read: the log replaces them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# the environment's row layout (env.zig `flatten`: rmodel.Input order)
ID_SPLITS = (("own_species", 6), ("own_status", 6), ("own_moves", 24), ("opp_species", 6), ("opp_status", 6), ("opp_moves", 24), ("last", 6))
FLOAT_SPLITS = (("own", 60), ("opp", 42), ("last", 34), ("field", 39))
N_EVENT_FIELDS = 12
MAX_EVENTS = 32
# event fields (events.zig F): which embedding table each uses
EVENT_TABLES = ("kind", "side", "species", "side", "species", "move", "move", "status", "reason", "hp", "num", "flags")
EVENT_VOCAB = {"kind": 96, "side": 3, "species": 152, "move": 256, "status": 8, "reason": 256, "hp": 101, "num": 256, "flags": 256}


@dataclass
class Config:
    d: int = 256
    layers: int = 6
    ff: int = 768
    log_layers: int = 1
    event_emb: int = 32
    log_hidden: int = 512
    n_species: int = 152
    n_moves: int = 168
    n_status: int = 8
    emb: int = 16
    status_emb: int = 4

    @property
    def heads(self) -> int:
        return self.d // 32


def split_ids(ids: torch.Tensor) -> dict:
    out, i = {}, 0
    for name, n in ID_SPLITS:
        out[name] = ids[..., i:i + n].long()
        i += n
    out["own_moves"] = out["own_moves"].unflatten(-1, (6, 4))
    out["opp_moves"] = out["opp_moves"].unflatten(-1, (6, 4))
    return out


def split_floats(x: torch.Tensor) -> dict:
    out, i = {}, 0
    for name, n in FLOAT_SPLITS:
        out[name] = x[..., i:i + n]
        i += n
    out["own"] = out["own"].unflatten(-1, (6, 10))
    out["opp"] = out["opp"].unflatten(-1, (6, 7))
    return out


class Block(nn.Module):
    """Pre-LayerNorm transformer block; `mask` [B, N] marks padding (True = ignore)."""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn_mask = None if mask is None else ~mask[:, None, None, :]  # (True = attend)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class LogEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tables = nn.ModuleDict({name: nn.Embedding(n, cfg.event_emb) for name, n in EVENT_VOCAB.items()})
        self.mlp = nn.Sequential(nn.Linear(N_EVENT_FIELDS * cfg.event_emb, cfg.log_hidden), nn.GELU(), nn.Linear(cfg.log_hidden, cfg.d), nn.LayerNorm(cfg.d))
        self.pos = nn.Parameter(torch.randn(MAX_EVENTS + 1, cfg.d) * 0.02)
        self.summary = nn.Parameter(torch.randn(cfg.d) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ff) for _ in range(cfg.log_layers))
        self.norm = nn.LayerNorm(cfg.d)

    def forward(self, events: torch.Tensor, n_events: torch.Tensor) -> torch.Tensor:
        """events [B, E, 12] (uint8), n_events [B] -> [B, d]."""
        b, e, _ = events.shape
        ev = events.long()
        fields = [self.tables[name](ev[..., i].clamp_max(EVENT_VOCAB[name] - 1)) for i, name in enumerate(EVENT_TABLES)]
        rows = self.mlp(torch.cat(fields, dim=-1))  # [B, E, d]
        x = torch.cat([self.summary.expand(b, 1, -1), rows], dim=1) + self.pos[: e + 1]
        pad = torch.arange(e, device=events.device)[None, :] >= n_events.long()[:, None]
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=events.device), pad], dim=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x[:, 0])


class SelfPlayModel(nn.Module):
    N_TOKENS = 16

    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or Config()
        d = cfg.d
        self.species = nn.Embedding(cfg.n_species, cfg.emb, padding_idx=0)
        self.moves = nn.Embedding(cfg.n_moves, cfg.emb, padding_idx=0)
        self.status = nn.Embedding(cfg.n_status, cfg.status_emb)
        own_w = cfg.emb + 1 + cfg.status_emb + 1 + 4 + 4 * (cfg.emb + 1)
        opp_w = 1 + cfg.emb + 1 + cfg.status_emb + 1 + 4 + 4 * cfg.emb
        self.state_in = nn.Linear(d, d)
        self.own_in = nn.Linear(own_w, d)
        self.opp_in = nn.Linear(opp_w, d)
        self.field_in = nn.Linear(39, d)
        self.value_token = nn.Parameter(torch.randn(d) * 0.02)
        self.role = nn.Parameter(torch.randn(self.N_TOKENS, d) * 0.02)
        self.log = LogEncoder(cfg)
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.ff) for _ in range(cfg.layers))
        self.final_norm = nn.LayerNorm(d)
        self.value_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 3))
        self.query = nn.Linear(d, d)
        self.move_key = nn.Sequential(nn.Linear(cfg.emb + 1 + d, d), nn.GELU(), nn.Linear(d, d))
        self.switch_key = nn.Linear(d, d)

    def board_tokens(self, ids: dict, floats: dict) -> torch.Tensor:
        sp, mv, st = self.species, self.moves, self.status
        own = torch.cat([
            sp(ids["own_species"]), floats["own"][..., 0:1], st(ids["own_status"]), floats["own"][..., 1:2], floats["own"][..., 2:6],
            torch.cat([mv(ids["own_moves"]), floats["own"][..., 6:10, None]], dim=-1).flatten(-2),
        ], dim=-1)
        opp = torch.cat([
            floats["opp"][..., 0:1], sp(ids["opp_species"]), floats["opp"][..., 1:2], st(ids["opp_status"]), floats["opp"][..., 2:3],
            floats["opp"][..., 3:7], mv(ids["opp_moves"]).flatten(-2),
        ], dim=-1)
        return torch.cat([self.own_in(own), self.opp_in(opp), self.field_in(floats["field"])[..., None, :]], dim=-2)  # [B, 13, d]

    def step(self, ids: torch.Tensor, floats: torch.Tensor, events: torch.Tensor, n_events: torch.Tensor, legal: torch.Tensor,
             h_prev: torch.Tensor):
        """One decision for a batch of rows. Returns (policy logits [B, 9] with illegal slots at -inf, value logits [B, 3],
        next state [B, d])."""
        i, f = split_ids(ids), split_floats(floats)
        b = ids.shape[0]
        board = self.board_tokens(i, f)
        update = self.log(events, n_events)
        x = torch.cat([self.state_in(h_prev)[:, None], update[:, None], board, self.value_token.expand(b, 1, -1)], dim=1) + self.role
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        h = x[:, 0]
        value = self.value_head(x[:, -1])
        # pointer head: moves 0-3 (the active Pokemon, own token 0), switches 4-8 (own tokens 1-5)
        active = x[:, 2]
        move_in = torch.cat([self.moves(i["own_moves"][:, 0]), f["own"][:, 0, 6:10, None], active[:, None].expand(-1, 4, -1)], dim=-1)
        keys = torch.cat([self.move_key(move_in), self.switch_key(x[:, 3:8])], dim=1)  # [B, 9, d]
        logits = torch.einsum("bd,bsd->bs", self.query(h), keys) / math.sqrt(self.cfg.d)
        logits = logits.float().masked_fill(legal == 0, float("-inf"))
        return logits, value.float(), h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
