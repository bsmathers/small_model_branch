"""The recurrent transformer policy/value model (pkmn-search/docs/model_spec.md).

One step reads 15 tokens: the carried state `h_prev`, the last turn, six own slots, six opponent slots and
the field. The state token's output is the next `h` and the input of both heads:

    policy  Linear(192, 9) on the state token (logits; the caller masks illegal slots)
    win     LayerNorm -> Linear(192, 64) -> GELU -> Linear(64, 1) (a logit; sigmoid is P(win))

Training runs a sequence by stepping the recurrence over time. Everything that does not depend on `h`
(the 14 observation tokens) is computed for all steps at once; only the 15-token transformer is stepped.

Input batch (all tensors [B, T, ...]; the feature encoder produces one step of this):
    ids    own_species [6], own_status [6], own_moves [6,4], opp_species [6], opp_status [6],
           opp_moves [6,4], last_{own,opp}_{move,species,status} [1]       (int64)
    floats own [6,10], opp [6,7], last [34], field [39]                    (float32)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ID_KEYS = (
    "own_species", "own_status", "own_moves", "opp_species", "opp_status", "opp_moves",
    "last_own_move", "last_opp_move", "last_own_species", "last_opp_species", "last_own_status", "last_opp_status",
)
FLOAT_KEYS = ("own", "opp", "last", "field")
N_TOKENS = 15


@dataclass
class Config:
    d: int = 192
    layers: int = 4
    heads: int = 6
    ff: int = 384
    n_species: int = 152
    n_moves: int = 168
    n_status: int = 8
    emb: int = 16
    status_emb: int = 4
    own_width: int = 94
    opp_width: int = 91
    last_width: int = 106
    field_width: int = 39
    n_actions: int = 9


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = cfg.heads
        self.norm1 = nn.LayerNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d)
        self.proj = nn.Linear(cfg.d, cfg.d)
        self.norm2 = nn.LayerNorm(cfg.d)
        self.ff1 = nn.Linear(cfg.d, cfg.ff)
        self.ff2 = nn.Linear(cfg.ff, cfg.d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)  # unmasked: all 15 tokens attend to each other
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class PolicyValueModel(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or Config()
        self.species = nn.Embedding(cfg.n_species, cfg.emb, padding_idx=0)
        self.moves = nn.Embedding(cfg.n_moves, cfg.emb, padding_idx=0)
        self.status = nn.Embedding(cfg.n_status, cfg.status_emb)
        self.state_in = nn.Linear(cfg.d, cfg.d)
        self.last_in = nn.Linear(cfg.last_width, cfg.d)
        self.own_in = nn.Linear(cfg.own_width, cfg.d)
        self.opp_in = nn.Linear(cfg.opp_width, cfg.d)
        self.field_in = nn.Linear(cfg.field_width, cfg.d)
        self.role = nn.Parameter(torch.zeros(N_TOKENS, cfg.d))
        nn.init.normal_(self.role, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_norm = nn.LayerNorm(cfg.d)
        self.policy = nn.Linear(cfg.d, cfg.n_actions)
        self.win = nn.Sequential(nn.LayerNorm(cfg.d), nn.Linear(cfg.d, 64), nn.GELU(), nn.Linear(64, 1))

    # ---- observation tokens (independent of h; computed for all steps at once) ----

    def observation_tokens(self, ids: dict, floats: dict) -> torch.Tensor:
        """[B, T, 14, d]: last turn, 6 own slots, 6 opponent slots, field (role embeddings added)."""
        b, t = floats["field"].shape[:2]
        sp, mv, st = self.species, self.moves, self.status

        own = torch.cat(
            [
                sp(ids["own_species"]),  # [B,T,6,16]
                floats["own"][..., 0:1],  # hp
                st(ids["own_status"]),
                floats["own"][..., 1:2],  # active
                floats["own"][..., 2:6],  # boosts
                torch.cat([mv(ids["own_moves"]), floats["own"][..., 6:10, None]], dim=-1).flatten(-2),
            ],
            dim=-1,
        )
        opp = torch.cat(
            [
                floats["opp"][..., 0:1],  # seen
                sp(ids["opp_species"]),
                floats["opp"][..., 1:2],  # hp
                st(ids["opp_status"]),
                floats["opp"][..., 2:3],  # active
                floats["opp"][..., 3:7],  # boosts
                mv(ids["opp_moves"]).flatten(-2),
            ],
            dim=-1,
        )
        last_f = floats["last"].view(b, t, 2, -1)
        last = torch.cat(
            [
                torch.cat(
                    [
                        last_f[:, :, side],
                        mv(ids[f"last_{who}_move"][:, :, 0]),
                        sp(ids[f"last_{who}_species"][:, :, 0]),
                        st(ids[f"last_{who}_status"][:, :, 0]),
                    ],
                    dim=-1,
                )
                for side, who in enumerate(("own", "opp"))
            ],
            dim=-1,
        )
        tokens = torch.cat(
            [
                self.last_in(last)[:, :, None],
                self.own_in(own),
                self.opp_in(opp),
                self.field_in(floats["field"])[:, :, None],
            ],
            dim=2,
        )
        return tokens + self.role[1:]

    # ---- one recurrent step ----

    def step(self, obs_tokens: torch.Tensor, h_prev: torch.Tensor):
        """obs_tokens [B, 14, d], h_prev [B, d] -> (policy logits [B, 9], win logit [B], h [B, d])."""
        x = torch.cat([(self.state_in(h_prev) + self.role[0])[:, None], obs_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        h = self.final_norm(x[:, 0])
        return self.policy(h), self.win(h).squeeze(-1), h

    def forward(self, ids: dict, floats: dict, h0: torch.Tensor | None = None, burn_in: int = 0):
        """Runs a sequence. The first `burn_in` steps only build the recurrent state (no gradient, no
        outputs); outputs cover the remaining steps. Returns (logits [B,T',9], win logits [B,T'], h_last)."""
        b, t = floats["field"].shape[:2]
        h = h0 if h0 is not None else floats["field"].new_zeros(b, self.cfg.d)
        tokens = self.observation_tokens(ids, floats)
        if burn_in:
            with torch.no_grad():
                for i in range(burn_in):
                    _, _, h = self.step(tokens[:, i], h)
            h = h.detach()
        logits, wins = [], []
        for i in range(burn_in, t):
            lg, w, h = self.step(tokens[:, i], h)
            logits.append(lg)
            wins.append(w)
        return torch.stack(logits, 1), torch.stack(wins, 1), h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
