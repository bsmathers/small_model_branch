"""Ataraxos-style self-play for the 5M recurrent model (SELFPLAY_DESIGN.md), on one GPU.

Each iteration: every game plays --turns turns with the current network sampling both sides (temperature 1); then one
epoch over the iteration's decisions in chunks of --chunk consecutive decisions of one player, each chunk starting from
the recurrent state recorded while playing (the state runs through the whole game; the gradient is cut at chunk starts).

Targets (per player and game, gamma 1, the reward the game's outcome: +1 / -1 / 0): advantages by GAE with lambda 0.5 on the
scalar value P(win) - P(loss); outcome targets by the categorical lambda-return with lambda 0.8. A game still running at the
iteration's end is bootstrapped from the network's prediction at that player's next decision.

Update (Ataraxos S3.4 / Table S7): keep the decisions whose |advantage| is at least the --filter-quantile of the iteration's
(0.5: the top half) and at least 0.01; on them: PPO-clipped surrogate (clip 0.2) + 0.1 x KL(policy || behavior) +
magnet x KL(policy || uniform over the legal slots), magnet = 0.05 / i^0.3, + 1 x cross-entropy of the outcome prediction.
Gradient-norm clip 0.267; Adam; EMA of the weights 0.999 (for evaluation). Learning rate: Ataraxos's clip(0.5 / i^1.1, 5e-6,
1e-4) with i = 1 + decisions seen / 5e6 (their iteration's size), scaled by sqrt(decisions per step / 24576) (their batch),
times an adaptive multiplier (<= 1) that keeps the policy's change per iteration near --target-kl.

    python train_selfplay.py OUT --games 4096 --turns 64 --threads 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Config, SelfPlayModel, count_parameters  # noqa: E402
from pkrl import Env  # noqa: E402

NEG = -1e4  # (log-probability of an illegal slot, finite for fp16 storage)


def log(out: Path, text: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- acting


class Collector:
    """Plays the games; keeps each row's recurrent state across iterations."""

    def __init__(self, env: Env, device: torch.device, d: int, turns: int):
        self.env, self.device, self.turns = env, device, turns
        R, G = env.rows, env.n_games
        self.R, self.G = R, G
        self.obs = env.reset()
        self.h = torch.zeros(R, d, device=device)
        T = turns
        dev = device
        self.buf = {
            "ids": torch.zeros(T, R, env.IDS, dtype=torch.int16, device=dev),
            "floats": torch.zeros(T, R, env.FLOATS, dtype=torch.float16, device=dev),
            "events": torch.zeros(T, R, env.MAX_EVENTS, env.FIELDS, dtype=torch.uint8, device=dev),
            "n_events": torch.zeros(T, R, dtype=torch.uint8, device=dev),
            "legal": torch.zeros(T, R, env.SLOTS, dtype=torch.uint8, device=dev),
            "valid": torch.zeros(T, R, dtype=torch.bool, device=dev),
            "forced": torch.zeros(T, R, dtype=torch.bool, device=dev),
            "first": torch.zeros(T, R, dtype=torch.bool, device=dev),
            "h": torch.zeros(T, R, d, dtype=torch.float16, device=dev),
            "action": torch.zeros(T, R, dtype=torch.long, device=dev),
            "logp": torch.zeros(T, R, env.SLOTS, dtype=torch.float16, device=dev),
            "probs": torch.zeros(T, R, 3, dtype=torch.float16, device=dev),
            "ended": torch.zeros(T, G, dtype=torch.bool, device=dev),
            "outcome": torch.zeros(T, G, dtype=torch.int8, device=dev),
        }
        self.boot = {"valid": None, "first": None, "probs": None}

    def _forward(self, model, obs: dict, amp: bool):
        o = {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}
        valid = o["valid"].bool()
        first = o["first"].bool()
        h_prev = torch.where(first[:, None], torch.zeros_like(self.h), self.h)
        legal = o["legal"].clone()
        legal[~valid, 0] = 1  # (invalid rows: a dummy legal slot, so their outputs stay finite)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=amp):
            logits, value, h_new = model.step(o["ids"], o["floats"], o["events"], o["n_events"], legal, h_prev)
        return o, valid, first, h_prev, logits.float(), value.float(), h_new.float()

    @torch.no_grad()
    def run(self, model, amp: bool, generator: torch.Generator) -> dict:
        model.eval()
        b = self.buf
        for t in range(self.turns):
            o, valid, first, h_prev, logits, value, h_new = self._forward(model, self.obs, amp)
            logp = torch.log_softmax(logits, -1)
            action = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            b["ids"][t] = o["ids"]
            b["floats"][t] = o["floats"].half()
            b["events"][t] = o["events"]
            b["n_events"][t] = o["n_events"]
            b["legal"][t] = o["legal"]
            b["valid"][t] = valid
            b["forced"][t] = o["forced"].bool()
            b["first"][t] = first
            b["h"][t] = h_prev.half()
            b["action"][t] = action
            b["logp"][t] = logp.clamp_min(NEG).half()
            b["probs"][t] = torch.softmax(value, -1).half()
            self.h = torch.where(valid[:, None], h_new, h_prev)
            self.obs = self.env.step(action.to(torch.uint8).cpu())
            b["ended"][t] = self.obs["ended"].to(self.device).bool()
            b["outcome"][t] = self.obs["outcome"].to(self.device)
        # the next decisions' predictions, to bootstrap the games still running
        o, valid, first, _, _, value, _ = self._forward(model, self.obs, amp)
        self.boot = {"valid": valid, "first": first, "probs": torch.softmax(value, -1)}
        return b


# ---------------------------------------------------------------- targets


@torch.no_grad()
def targets(b: dict, boot: dict, lam_adv: float, lam_val: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages [T, R] and outcome targets [T, R, 3] (from each row's player's side: win, loss, tie), going backward
    through each row's decisions; an `ended` game's outcome is the next step for its players' last decisions."""
    T, R = b["valid"].shape
    dev = b["valid"].device
    probs = b["probs"].float()
    v = probs[..., 0] - probs[..., 1]
    player = torch.arange(R, device=dev) % 2
    game = torch.arange(R, device=dev) // 2
    adv = torch.zeros(T, R, device=dev)
    tgt = torch.zeros(T, R, 3, device=dev)
    # the successor of each row's latest decision: the bootstrap (a running game), or nothing (own value: no signal)
    has = boot["valid"] & ~boot["first"]
    next_v = torch.where(has, boot["probs"][:, 0] - boot["probs"][:, 1], torch.full((R,), float("nan"), device=dev))
    next_p = torch.where(has[:, None], boot["probs"], torch.full((R, 3), float("nan"), device=dev))
    next_g = next_p.clone()
    next_a = torch.zeros(R, device=dev)
    for t in range(T - 1, -1, -1):
        ended = b["ended"][t][game]
        if ended.any():
            o = b["outcome"][t][game].long()  # P1's side: 1, -1, 0
            mine = torch.where(player == 0, o, -o)
            onehot = torch.stack([(mine == 1), (mine == -1), (mine == 0)], -1).float()
            next_v = torch.where(ended, mine.float(), next_v)
            next_p = torch.where(ended[:, None], onehot, next_p)
            next_g = torch.where(ended[:, None], onehot, next_g)
            next_a = torch.where(ended, torch.zeros_like(next_a), next_a)
        valid = b["valid"][t]
        vt = v[t]
        nv = torch.where(torch.isnan(next_v), vt, next_v)
        a_t = (nv - vt) + lam_adv * next_a
        np_ = torch.where(torch.isnan(next_p), probs[t], next_p)
        ng = torch.where(torch.isnan(next_g), probs[t], next_g)
        g_t = (1 - lam_val) * np_ + lam_val * ng
        adv[t] = torch.where(valid, a_t, 0.0)
        tgt[t] = torch.where(valid[:, None], g_t, 0.0)
        next_v = torch.where(valid, vt, next_v)
        next_p = torch.where(valid[:, None], probs[t], next_p)
        next_g = torch.where(valid[:, None], g_t, next_g)
        next_a = torch.where(valid, a_t, next_a)
    return adv, tgt


def make_chunks(valid: np.ndarray, first: np.ndarray, length: int) -> np.ndarray:
    """[N, length, 2] (t, r) of `length` consecutive decisions of one row in one game; -1 pads."""
    T, R = valid.shape
    chunks = []
    for r in range(R):
        ts = np.nonzero(valid[:, r])[0]
        if len(ts) == 0:
            continue
        starts = np.nonzero(first[ts, r])[0]
        bounds = sorted(set([0, *starts.tolist(), len(ts)]))
        for a, z in zip(bounds[:-1], bounds[1:]):
            seg = ts[a:z]
            for i in range(0, len(seg), length):
                c = np.full((length, 2), -1, dtype=np.int64)
                part = seg[i:i + length]
                c[: len(part), 0] = part
                c[: len(part), 1] = r
                chunks.append(c)
    return np.stack(chunks) if chunks else np.zeros((0, length, 2), dtype=np.int64)


# ---------------------------------------------------------------- the update


def chunk_loss(model, b: dict, idx: torch.Tensor, adv: torch.Tensor, tgt: torch.Tensor, keep: torch.Tensor, args, magnet: float, amp: bool):
    """idx [B, L, 2]; returns (loss, stats). The recurrence starts from the recorded state at each chunk's first decision."""
    B, L, _ = idx.shape
    t, r = idx[..., 0], idx[..., 1]
    real = t >= 0
    tt, rr = t.clamp_min(0), r.clamp_min(0)
    h = b["h"][tt[:, 0], rr[:, 0]].float()
    pol_terms, val_terms, n_pol, n_val = [], [], 0, 0
    stats = {"clip_frac": 0.0, "kl_behavior": 0.0, "entropy": 0.0}
    for k in range(L):
        tk, rk, m = tt[:, k], rr[:, k], real[:, k]
        first = b["first"][tk, rk] & m
        h = torch.where(first[:, None], torch.zeros_like(h), h)
        with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=amp):
            logits, value, h_new = model.step(b["ids"][tk, rk], b["floats"][tk, rk].float(), b["events"][tk, rk], b["n_events"][tk, rk],
                                              b["legal"][tk, rk], h)
        h = torch.where(m[:, None], h_new.float(), h)
        logits = logits.float()
        legal = b["legal"][tk, rk] > 0
        logp = torch.log_softmax(logits.masked_fill(~legal, NEG), -1)
        p = logp.exp()
        sel = keep[tk, rk] & m
        if sel.any():
            a = b["action"][tk, rk]
            logp_b = b["logp"][tk, rk].float()
            ratio = torch.exp(logp.gather(-1, a[:, None]).squeeze(-1) - logp_b.gather(-1, a[:, None]).squeeze(-1))
            A = adv[tk, rk]
            surr = -torch.minimum(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A)
            kl_b = (p * (logp - logp_b)).masked_fill(~legal, 0).sum(-1)
            n_legal = legal.sum(-1).clamp_min(1).float()
            kl_u = (p * logp).masked_fill(~legal, 0).sum(-1) + torch.log(n_legal)
            term = surr + args.kl_behavior * kl_b + magnet * kl_u
            pol_terms.append(term[sel].sum())
            n_pol += int(sel.sum())
            vt = -(tgt[tk, rk] * torch.log_softmax(value.float(), -1)).sum(-1)
            val_terms.append(vt[sel if not args.value_all else m].sum())
            n_val += int((sel if not args.value_all else m).sum())
            stats["clip_frac"] += float(((ratio - 1).abs() > args.clip)[sel].float().sum().detach())
            stats["kl_behavior"] += float(kl_b[sel].sum().detach())
            stats["entropy"] += float(-(p * logp).masked_fill(~legal, 0).sum(-1)[sel].sum().detach())
        elif args.value_all and m.any():
            vt = -(tgt[tk, rk] * torch.log_softmax(value.float(), -1)).sum(-1)
            val_terms.append(vt[m].sum())
            n_val += int(m.sum())
    if n_pol == 0 and n_val == 0:
        return None, stats, 0
    loss = (torch.stack(pol_terms).sum() / max(n_pol, 1) if pol_terms else 0.0)
    loss = loss + args.value_weight * (torch.stack(val_terms).sum() / max(n_val, 1) if val_terms else 0.0)
    for k in stats:
        stats[k] /= max(n_pol, 1)
    return loss, stats, n_pol


def lr_schedule(seen: float, decisions_per_step: int) -> tuple[float, float]:
    i = 1.0 + seen / 5e6
    scale = math.sqrt(decisions_per_step / 24576)
    lr = min(max(0.5 / i ** 1.1, 5e-6), 1e-4) * scale
    magnet = 0.05 / i ** 0.3
    return lr, magnet


def probe_chunks(model, collector, args, device, amp: bool) -> int:
    """The largest number of chunks per step whose forward + backward fits in ~85% of GPU memory (CPU: --batch-chunks)."""
    if device.type != "cuda":
        return args.batch_chunks
    total = torch.cuda.get_device_properties(device).total_memory
    b = collector.buf
    T, R = b["valid"].shape
    for n in (4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 128, 64):
        idx = torch.stack([torch.randint(0, T, (n, args.chunk)), torch.randint(0, R, (n, args.chunk))], -1).to(device)
        adv = torch.ones(T, R, device=device)
        tgt = torch.full((T, R, 3), 1 / 3, device=device)
        keep = torch.ones(T, R, dtype=torch.bool, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            model.train()
            loss, _, _ = chunk_loss(model, b, idx, adv, tgt, keep, args, 0.01, amp)
            loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device)
            if peak < 0.85 * total:
                return n
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return 64


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--tables", default=str(root / "selfplay" / "data" / "tables"), help="an export folder with tables/ (export_recurrent.py)")
    ap.add_argument("--teams-a", default=str(root / "duel" / "teamset"))
    ap.add_argument("--teams-b", default=str(root / "selfplay" / "data" / "teams" / "hl_05_26" / "gen1ou"))
    ap.add_argument("--share-a", type=float, default=0.5)
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--turns", type=int, default=64, help="turns per game per iteration")
    ap.add_argument("--threads", type=int, default=0, help="environment threads (0: the CPUs)")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--batch-chunks", type=int, default=0, help="chunks per gradient step (0: as many as fit; CPU default 64)")
    ap.add_argument("--iterations", type=int, default=1_000_000)
    ap.add_argument("--lambda-adv", type=float, default=0.5)
    ap.add_argument("--lambda-value", type=float, default=0.8)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=0.267)
    ap.add_argument("--kl-behavior", type=float, default=0.1)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--value-all", action="store_true", help="value loss on every decision (Ataraxos: only on the kept ones)")
    ap.add_argument("--filter-quantile", type=float, default=0.5, help="keep |advantage| >= this quantile (Ataraxos 0.75)")
    ap.add_argument("--filter-min", type=float, default=0.01)
    ap.add_argument("--target-kl", type=float, default=0.01, help="the policy's change per iteration the adaptive learning rate aims at")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--max-turns", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--ff", type=int, default=768)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    threads = args.threads or os.cpu_count() or 1
    env = Env(args.games, threads, args.seed, args.tables, (args.teams_a, args.teams_b), share_a=args.share_a, max_turns=args.max_turns)
    model = SelfPlayModel(Config(d=args.width, layers=args.layers, ff=args.ff)).to(device)
    ema = SelfPlayModel(model.cfg).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    state = {"iteration": 0, "seen": 0, "lr_mult": 1.0}
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        log(out, f"resumed at iteration {state['iteration']} ({state['seen']:,} decisions)")
    log(out, f"model {count_parameters(model):,} parameters; {args.games} games x {args.turns} turns per iteration on {threads} threads; "
             f"teams {env.pool_sizes()}; device {device}")
    collector = Collector(env, device, model.cfg.d, args.turns)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    chunks_per_step = args.batch_chunks or (64 if device.type != "cuda" else 0)
    while state["iteration"] < args.iterations:
        it = state["iteration"] + 1
        t0 = time.time()
        stats0 = env.stats()
        b = collector.run(model, amp, gen)
        t_act = time.time() - t0
        if not chunks_per_step:
            chunks_per_step = probe_chunks(model, collector, args, device, amp)
            log(out, f"batch: {chunks_per_step} chunks of {args.chunk} decisions per step (measured)")
        adv, tgt = targets(b, collector.boot, args.lambda_adv, args.lambda_value)
        eligible = b["valid"] & ~b["forced"]
        absadv = adv.abs()[eligible]
        thr = max(float(torch.quantile(absadv.float()[torch.randperm(absadv.numel(), device=device)[:1_000_000]], args.filter_quantile)) if absadv.numel() else 0.0, args.filter_min)
        keep = eligible & (adv.abs() >= thr)
        chunks = torch.from_numpy(make_chunks(b["valid"].cpu().numpy(), b["first"].cpu().numpy(), args.chunk)).to(device)
        chunks = chunks[torch.randperm(len(chunks), device=device)]
        n_dec = int(b["valid"].sum())
        lr, magnet = lr_schedule(state["seen"], chunks_per_step * args.chunk)
        lr *= state["lr_mult"]
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        t1 = time.time()
        sums = {"loss": 0.0, "clip_frac": 0.0, "kl_behavior": 0.0, "entropy": 0.0}
        steps = 0
        for s in range(0, len(chunks), chunks_per_step):
            idx = chunks[s:s + chunks_per_step]
            loss, st, n_pol = chunk_loss(model, b, idx, adv, tgt, keep, args, magnet, amp)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
            sums["loss"] += float(loss.detach())
            for k in st:
                sums[k] += st[k]
            steps += 1
        t_train = time.time() - t1
        # the policy's change this iteration, on a sample of its decisions: drives the adaptive learning rate
        with torch.no_grad():
            sample = chunks[: min(len(chunks), max(64, chunks_per_step // 4))]
            _, st_after, _ = chunk_loss(model, b, sample, adv, tgt, eligible, args, 0.0, amp)
        kl_iter = st_after["kl_behavior"]
        if kl_iter > 2 * args.target_kl:
            state["lr_mult"] = max(0.05, state["lr_mult"] / 1.5)
        elif kl_iter < 0.5 * args.target_kl:
            state["lr_mult"] = min(1.0, state["lr_mult"] * 1.5)
        state["iteration"] = it
        state["seen"] += n_dec
        s1 = env.stats()
        games = s1["games"] - stats0["games"]
        p1 = (s1["p1_wins"] - stats0["p1_wins"] + 0.5 * (s1["ties"] - stats0["ties"])) / max(games, 1)
        mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        log(out, f"iter {it} decisions {n_dec:,} (kept {int(keep.sum()):,}, |A| >= {thr:.4f}) games {games} P1 {p1:.3f} | "
                 f"loss {sums['loss'] / max(steps, 1):.4f} entropy {sums['entropy'] / max(steps, 1):.3f} clip {sums['clip_frac'] / max(steps, 1):.3f} "
                 f"KL(step) {sums['kl_behavior'] / max(steps, 1):.4f} KL(iter) {kl_iter:.4f} | lr {lr:.2e} (x{state['lr_mult']:.2f}) magnet {magnet:.4f} "
                 f"steps {steps} | act {t_act:.1f}s train {t_train:.1f}s ({n_dec / (t_act + t_train):,.0f} decisions/s) mem {mem:.1f} GiB")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "state": state, "args": vars(args)}, latest.with_suffix(".tmp"))
        os.replace(latest.with_suffix(".tmp"), latest)
        if it % 100 == 0:
            torch.save(ema.state_dict(), out / f"ema_{it}.pt")


if __name__ == "__main__":
    main()
