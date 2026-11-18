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
HEAD_WIDTH = 32  # attention head width of every size (the loaders infer the head count as d / 32)


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
    # optional heads (off in every model trained before them; from_state turns them on when a weights file has them)
    deep_actor: bool = False  # a 3-layer actor head (192 -> 64 -> 64 -> 9) added to the linear one (zero-initialised output: starts as the linear head)
    actor_width: int = 64
    opponent_head: bool = False  # predicts the opponent's next action: a move (by move id) or a switch (n_moves + species id)
    head_hidden: int = 128

    @property
    def n_opponent(self) -> int:
        return self.n_moves + self.n_species

    @classmethod
    def sized(cls, d: int, layers: int, ff: int = 0, **kw) -> "Config":
        """A model of width d (a multiple of 32: 32-wide heads), `layers` blocks and feed-forward width ff (default 2 d). The inputs,
        embeddings and heads keep their sizes. 192 / 4 / 384 is the original 1.31M model; 96 / 2 / 192 has 205,066 parameters, 256 / 6 / 512 3,342,762."""
        if d % HEAD_WIDTH or d <= 0 or layers <= 0:
            raise ValueError(f"width {d} must be a positive multiple of {HEAD_WIDTH}, layers positive")
        return cls(d=d, layers=layers, heads=d // HEAD_WIDTH, ff=ff or 2 * d, **kw)


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


class DeepActor(nn.Module):
    """The deeper actor head: LayerNorm, then 3 linear layers on the state token (192 -> 64 -> 64 -> 9, GELU between), added to
    the linear head's logits. The last layer starts at zero, so a model that gets this head plays exactly as before until trained."""

    def __init__(self, cfg: "Config"):
        super().__init__()
        w = cfg.actor_width
        self.net = nn.Sequential(nn.LayerNorm(cfg.d), nn.Linear(cfg.d, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, cfg.n_actions))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


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
        # how much of the win loss's gradient reaches the shared network through the state token (training only; the forward
        # value is unchanged): 1 = all of it, 0 = none (the win head learns on the policy's features)
        self.value_grad_scale = 1.0
        if cfg.deep_actor:
            self.policy_deep = DeepActor(cfg)
        if cfg.opponent_head:
            self.opponent = nn.Sequential(nn.LayerNorm(cfg.d), nn.Linear(cfg.d, cfg.head_hidden), nn.GELU(), nn.Linear(cfg.head_hidden, cfg.n_opponent))

    @staticmethod
    def size_of(state: dict) -> dict:
        """The size of the model these weights are for: width d (state_in), layers (blocks.N), feed-forward width (ff1) and
        heads (d / 32: every model is built with 32-wide attention heads, see Config.sized)."""
        d = state["state_in.weight"].shape[0]
        layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
        return {"d": d, "layers": layers, "ff": state["blocks.0.ff1.weight"].shape[0], "heads": max(1, d // HEAD_WIDTH)}

    @classmethod
    def from_state(cls, state: dict, deep_actor: bool = False, opponent_head: bool = False) -> "PolicyValueModel":
        """A model for these weights (a state dict, or a checkpoint with "model"), of their size: the optional heads the weights have,
        plus the ones asked for, which start fresh (the deep actor's output at zero, so the policy is unchanged). Any other missing or
        extra key is an error."""
        state = state["model"] if "model" in state else state
        has_deep = any(k.startswith("policy_deep.") for k in state)
        has_opp = any(k.startswith("opponent.") for k in state)
        model = cls(Config(deep_actor=deep_actor or has_deep, opponent_head=opponent_head or has_opp, **cls.size_of(state)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        fresh = ("policy_deep." if not has_deep else "\0", "opponent." if not has_opp else "\0")
        bad = [k for k in missing if not k.startswith(fresh)] + list(unexpected)
        if bad:
            raise RuntimeError(f"weights do not fit the model: {bad[:5]}")
        return model

    def policy_logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.policy(h)
        if self.cfg.deep_actor:
            logits = logits + self.policy_deep(h)
        return logits

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
        s = self.value_grad_scale
        hv = h if s == 1.0 else h * s + h.detach() * (1.0 - s)  # (the same value, a scaled gradient)
        return self.policy_logits(h), self.win(hv).squeeze(-1), h

    def forward(self, ids: dict, floats: dict, h0: torch.Tensor | None = None, burn_in: int = 0, states: bool = False):
        """Runs a sequence. The first `burn_in` steps only build the recurrent state (no gradient, no
        outputs); outputs cover the remaining steps. Returns (logits [B,T',9], win logits [B,T'], h_last), and with
        states=True also the state token of every output step [B,T',d] (the opponent head reads it)."""
        b, t = floats["field"].shape[:2]
        h = h0 if h0 is not None else floats["field"].new_zeros(b, self.cfg.d)
        tokens = self.observation_tokens(ids, floats)
        if burn_in:
            with torch.no_grad():
                for i in range(burn_in):
                    _, _, h = self.step(tokens[:, i], h)
            h = h.detach()
        logits, wins, hs = [], [], []
        for i in range(burn_in, t):
            lg, w, h = self.step(tokens[:, i], h)
            logits.append(lg)
            wins.append(w)
            hs.append(h)
        if states:
            return torch.stack(logits, 1), torch.stack(wins, 1), h, torch.stack(hs, 1)
        return torch.stack(logits, 1), torch.stack(wins, 1), h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
