"""Policy blending: several trained models played as one policy.

Each member keeps its own recurrent state and reads the same observation; at a decision their move distributions over the legal
options are combined:

    mix    sum_i w_i p_i                (default: the mixed strategy that plays member i with probability w_i, decided afresh
                                         at every decision)
    logit  proportional to prod_i p_i^w_i   (a geometric blend: sharper where the members agree)

A spec names the members: "a.pt" (one model, the same as playing it alone), or "a.pt@0.5,b.pt@0.3,c.pt@0.2" (weights are
normalised; a member without "@w" gets weight 1).
"""

from __future__ import annotations

import hashlib
import os

import torch

from metamon.recurrent.model import PolicyValueModel


def parse_spec(spec: str) -> list[tuple[str, float]]:
    members = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        path, _, weight = part.rpartition("@") if "@" in part else (part, "", "1")
        members.append((os.path.expanduser(path), float(weight)))
    if not members or any(w < 0 for _, w in members) or sum(w for _, w in members) <= 0:
        raise ValueError(f"bad blend spec {spec!r}")
    return members


def describe(spec: str) -> list[dict]:
    """Each member's path, weight and sha256 (for the tournament's result file)."""
    out = []
    for path, weight in parse_spec(spec):
        with open(path, "rb") as f:
            out.append({"weights": path, "weight": weight, "sha256": hashlib.sha256(f.read()).hexdigest()})
    return out


class PolicyBlend:
    def __init__(self, models: list[PolicyValueModel], weights: list[float], mode: str = "mix"):
        if mode not in ("mix", "logit"):
            raise ValueError(f"blend mode {mode!r}: mix or logit")
        total = sum(weights)
        self.models = models
        self.weights = [w / total for w in weights]
        self.mode = mode

    @classmethod
    def load(cls, spec: str, mode: str = "mix") -> "PolicyBlend":
        members = parse_spec(spec)
        models = []
        for path, _ in members:
            model = PolicyValueModel.from_state(torch.load(path, map_location="cpu"))
            models.append(model.eval())
        return cls(models, [w for _, w in members], mode)

    @property
    def size(self) -> int:
        return len(self.models)

    def initial_state(self) -> list[torch.Tensor]:
        return [torch.zeros(1, m.cfg.d) for m in self.models]

    @torch.no_grad()
    def decide(self, ids: dict, floats: dict, h: list[torch.Tensor], legal_slots: list[int]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One decision ([1, 1, ...] inputs): (log-probabilities [1, 9] of the blended policy, -inf off the legal slots; the members'
        next states). With no legal slot, the first member's raw logits (the caller then plays its fallback)."""
        new_h, logps, raw = [], [], None
        legal = torch.zeros(1, self.models[0].cfg.n_actions, dtype=torch.bool)
        legal[0, legal_slots] = True
        for model, state in zip(self.models, h):
            tokens = model.observation_tokens(ids, floats)
            logits, _, nxt = model.step(tokens[:, 0], state)
            new_h.append(nxt)
            raw = logits if raw is None else raw
            logps.append(torch.log_softmax(logits.float().masked_fill(~legal, float("-inf")), dim=-1))
        if not legal_slots:
            return raw, new_h
        if len(logps) == 1:
            return logps[0], new_h
        stacked = torch.stack(logps)  # [members, 1, 9]
        w = torch.tensor(self.weights)[:, None, None]
        if self.mode == "mix":
            blended = torch.logsumexp(stacked.masked_fill(~legal, float("-inf")) + w.log(), dim=0)
        else:
            blended = torch.log_softmax((stacked.masked_fill(~legal, 0.0) * w).sum(0).masked_fill(~legal, float("-inf")), dim=-1)
        return blended, new_h
