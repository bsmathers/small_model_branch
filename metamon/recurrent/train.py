"""Trains the recurrent policy/value model from built games (build.py).

    python -m metamon.recurrent.train OUT_DIR \
        --source human=/data/built/human:0.4 --source strong=/data/built/strong:0.6 \
        --steps 20000 --batch auto --device cuda

Loss per scored step:  alpha * KL(target || policy over the legal slots) + bc_weight * BC + BCE(win logit, result)
with target = the proposer blend by default (a duel of 250 games found the blend player level with the router);
`--target judge|mix` selects the judge softmax or a mixture. Decisions where the observer could not act (action -1)
carry no distillation term. BC is the cross-entropy of the action the observer actually played, KL(one-hot ||
policy), on games played by the imitated bots (build.py --bc-players); it is added to, not in place of, the
distillation term. latest.pt (model, optimizer, step) is rewritten every iteration by a background thread
(--ckpt-every), numbered weights_<step>.pt every --save-every steps. Progress and validation are plain text in
OUT_DIR/log.txt.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metamon.recurrent.dataset import Windows, collate, scan
from metamon.recurrent.model import FLOAT_KEYS, ID_KEYS, Config, PolicyValueModel, count_parameters

BIG_NEGATIVE = -1e9


def policy_target(batch: dict, mode: str, mix: float) -> torch.Tensor:
    blend, judge = batch["blend"], batch["judge"]
    if mode == "blend":
        return blend
    if mode == "judge":
        return torch.where(judge.sum(-1, keepdim=True) > 0, judge, blend)
    mixed = (1 - mix) * judge + mix * blend
    return torch.where(judge.sum(-1, keepdim=True) > 0, mixed, blend)


def compute(model_out, batch: dict, mode: str, mix: float):
    """Per-step quantities, [B, T] each, over the post-burn-in outputs."""
    logits, win_logit = model_out
    legal = batch["legal"] > 0
    logp = F.log_softmax(logits.float().masked_fill(~legal, BIG_NEGATIVE), dim=-1)
    target = policy_target(batch, mode, mix)
    target = target * legal
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    kl = (target * (target.clamp_min(1e-8).log() - logp)).sum(-1)
    top1 = (logp.argmax(-1) == target.argmax(-1)).float()
    win = batch["win"][:, None].expand_as(win_logit)
    bce = F.binary_cross_entropy_with_logits(win_logit.float(), win, reduction="none")
    p = torch.sigmoid(win_logit.float())
    # KL(one-hot of the played action || policy). Code 9 is "some move" (which one is unknown): the policy's
    # total probability on the moves.
    played = batch["played"]
    fight = played == 9
    moves_mass = logp[..., :4].exp().sum(-1).clamp_min(1e-8)
    bc = torch.where(fight, -moves_mass.log(), -logp.gather(-1, played.clamp(0, 8).unsqueeze(-1)).squeeze(-1))
    best = logp.argmax(-1)
    bc_top1 = torch.where(fight, best < 4, best == played).float()
    return {"kl": kl, "top1": top1, "bce": bce, "brier": (p - win) ** 2, "p": p, "bc": bc, "bc_top1": bc_top1}


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1)


def run_batch(model, batch: dict, args, ref=None):
    ids = {k: batch[f"ids_{k}"] for k in ID_KEYS}
    floats = {k: batch[f"floats_{k}"] for k in FLOAT_KEYS}
    if model.cfg.opponent_head:
        logits, win_logit, _, states = model(ids, floats, states=True)
    else:
        logits, win_logit, _ = model(ids, floats)
    out = compute((logits, win_logit), batch, args.target, args.mix)
    if model.cfg.opponent_head and "opp_action" in batch:
        target = batch["opp_action"]
        opp_logits = model.opponent(states).float()
        out["opp"] = F.cross_entropy(opp_logits.transpose(1, 2), target.clamp_min(0), reduction="none").masked_fill(target < 0, 0.0)
        out["opp_top1"] = (opp_logits.argmax(-1) == target).float()
    if ref is not None:
        out["anchor"] = anchor_kl(logits, ref, ids, floats, batch["legal"] > 0)
    return out


def anchor_kl(logits: torch.Tensor, ref, ids: dict, floats: dict, legal: torch.Tensor) -> torch.Tensor:
    """KL(reference policy || current policy) over the legal slots, [B, T]. The reference is the frozen first-stage model, so this
    measures how far the second stage has moved the policy (and with it the backbone that feeds the policy head)."""
    with torch.no_grad():
        ref_logits, _, _ = ref(ids, floats)
    legal = legal | ~legal.any(-1, keepdim=True)  # (a row with no legal slot contributes nothing: both sides are uniform there)
    logp = F.log_softmax(logits.float().masked_fill(~legal, BIG_NEGATIVE), dim=-1)
    ref_logp = F.log_softmax(ref_logits.float().masked_fill(~legal, BIG_NEGATIVE), dim=-1)
    return (ref_logp.exp() * (ref_logp - logp)).masked_fill(~legal, 0.0).sum(-1)


@torch.no_grad()
def validate(model, loader, args, device, ref=None) -> dict:
    model.eval()
    sums = {k: 0.0 for k in ("kl", "top1", "bce", "brier", "bc", "bc_top1", "anchor", "kl_nonbot", "opp", "opp_top1")}
    count_p = count_w = count_b = count_n = count_o = 0.0
    bins = np.zeros((10, 3))  # count, sum of predictions, sum of results
    by_source: dict[int, list[float]] = {}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype(args), enabled=amp_enabled(args, device)):
            out = run_batch(model, batch, args, ref)
        pol = (batch["policy_ok"] & batch["scored"]).float()
        sc = batch["scored"].float()
        if ref is not None:
            sums["anchor"] += (out["anchor"] * sc).sum().item()
        bcm = (batch["bc_ok"] & batch["scored"]).float()
        for k in ("bc", "bc_top1"):
            sums[k] += (out[k] * bcm).sum().item()
        count_b += bcm.sum().item()
        for k in ("kl", "top1"):
            sums[k] += (out[k] * pol).sum().item()
        nonbot = pol * (~batch["bc_game"]).float()
        sums["kl_nonbot"] += (out["kl"] * nonbot).sum().item()
        count_n += nonbot.sum().item()
        if "opp" in out:
            om = ((batch["opp_action"] >= 0) & batch["scored"]).float()
            sums["opp"] += (out["opp"] * om).sum().item()
            sums["opp_top1"] += (out["opp_top1"] * om).sum().item()
            count_o += om.sum().item()
        for k in ("bce", "brier"):
            sums[k] += (out[k] * sc).sum().item()
        count_p += pol.sum().item()
        count_w += sc.sum().item()
        idx = (out["p"].clamp(0, 0.9999) * 10).long()
        for b in range(10):
            m = (idx == b).float() * sc
            bins[b] += [m.sum().item(), (out["p"] * m).sum().item(), (batch["win"][:, None].expand_as(out["p"]) * m).sum().item()]
        for s in batch["source"].unique().tolist():
            m = (batch["source"] == s).float()[:, None]
            r = by_source.setdefault(s, [0.0, 0.0, 0.0, 0.0])
            r[0] += (out["kl"] * pol * m).sum().item()
            r[1] += (pol * m).sum().item()
            r[2] += (out["bce"] * sc * m).sum().item()
            r[3] += (sc * m).sum().item()
    model.train()
    result = {
        "kl": sums["kl"] / max(count_p, 1),
        "top1": sums["top1"] / max(count_p, 1),
        "bce": sums["bce"] / max(count_w, 1),
        "brier": sums["brier"] / max(count_w, 1),
        "bc": sums["bc"] / max(count_b, 1),
        "bc_top1": sums["bc_top1"] / max(count_b, 1),
        "anchor": sums["anchor"] / max(count_w, 1),
        "kl_nonbot": sums["kl_nonbot"] / max(count_n, 1),
        "opp": sums["opp"] / max(count_o, 1),
        "opp_top1": sums["opp_top1"] / max(count_o, 1),
        "opp_decisions": count_o,
        "bc_decisions": count_b,
        "decisions": count_w,
    }
    result["calibration"] = [(int(c), s / c if c else 0.0, w / c if c else 0.0) for c, s, w in bins]
    result["by_source"] = {s: (r[0] / max(r[1], 1), r[2] / max(r[3], 1)) for s, r in by_source.items()}
    return result


def amp_dtype(args):
    return torch.float16 if args.precision == "fp16" else torch.bfloat16


def amp_enabled(args, device) -> bool:
    return args.precision != "fp32" and device.type == "cuda"


def to_cpu(obj):
    """A detached CPU copy of a nested structure of tensors (model or optimizer state)."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


class Checkpointer:
    """Writes the checkpoint file from a background thread so that saving every iteration does not stall
    training. submit() copies the state to CPU (a few milliseconds for this model) and hands it over; if the
    disk is still busy with the previous one, the older pending state is replaced by the newer, so the file
    is never more than a write behind. Each write goes to a temporary file that atomically replaces the old
    checkpoint, so a crash or full disk leaves the previous complete checkpoint in place."""

    def __init__(self, path: str):
        self.path = path
        self.pending = None
        self.error = None
        self.closed = False
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, state: dict) -> None:
        if self.error:
            raise RuntimeError(f"checkpoint writer failed: {self.error}")
        with self.cv:
            self.pending = state
            self.cv.notify()

    def _run(self) -> None:
        while True:
            with self.cv:
                while self.pending is None and not self.closed:
                    self.cv.wait()
                state, self.pending = self.pending, None
                if state is None:
                    return
            try:
                tmp = self.path + ".tmp"
                torch.save(state, tmp)
                os.replace(tmp, self.path)
            except Exception as exc:  # reported at the next submit
                self.error = exc

    def close(self) -> None:
        """Waits for the last submitted state to reach the disk."""
        with self.cv:
            self.closed = True
            self.cv.notify()
        self.thread.join()
        if self.error:
            raise RuntimeError(f"checkpoint writer failed: {self.error}")


def save_atomic(obj, path: str) -> None:
    """torch.save to a temporary name, then rename: a crash mid-write leaves the previous file (or none), never a truncated one
    that a restart would fail to load."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Backup:
    """Copies of the run on separate storage (a vast.ai local volume, say), for when the container disk is lost:
    weights_<step>.pt (model only) at every backup, latest.pt (model, optimizer, step) overwritten at every backup,
    and copies of weights_best.pt, weights_final.pt and log.txt. Every file is written to a temporary name and
    renamed, so an interruption leaves the previous copy whole. A failed write is logged and training goes on
    (the run on the container disk is unaffected); a nearly full volume skips the numbered weights first and
    keeps latest.pt going."""

    RESERVE = 200 * 2**20  # bytes kept free on the volume beyond a latest.pt's size

    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        self.log = print
        self.latest_size = 0
        self.warned_full = False

    def free_gib(self) -> float:
        return shutil.disk_usage(self.dir).free / 2**30

    def _write(self, name: str, write) -> bool:
        path = os.path.join(self.dir, name)
        tmp = path + ".tmp"
        try:
            write(tmp)
            os.replace(tmp, path)
            return True
        except Exception as exc:  # noqa: BLE001
            self.log(f"BACKUP FAILED: {name} to {self.dir}: {type(exc).__name__}: {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False

    def copy(self, src: str) -> None:
        if os.path.exists(src):
            self._write(os.path.basename(src), lambda tmp: shutil.copyfile(src, tmp))

    def save_step(self, step: int, state: dict) -> None:
        free = shutil.disk_usage(self.dir).free
        if free > 2 * self.latest_size + self.RESERVE:
            self._write(f"weights_{step}.pt", lambda tmp: torch.save(state["model"], tmp))
        elif not self.warned_full:
            self.warned_full = True
            self.log(f"backup volume {self.dir} nearly full ({free / 2**20:.0f} MiB free): numbered weights are no longer backed up; latest.pt still is")
        if self._write("latest.pt", lambda tmp: torch.save(state, tmp)):
            self.latest_size = os.path.getsize(os.path.join(self.dir, "latest.pt"))

    def restore(self, out: str) -> str | None:
        """Copies latest.pt, weights_best.pt, weights_final.pt and log.txt from the backup into OUT (which has no latest.pt)."""
        if not os.path.exists(os.path.join(self.dir, "latest.pt")):
            return None
        names = [n for n in ("latest.pt", "weights_best.pt", "weights_final.pt", "log.txt")
                 if os.path.exists(os.path.join(self.dir, n)) and (n == "latest.pt" or not os.path.exists(os.path.join(out, n)))]
        for n in names:
            shutil.copyfile(os.path.join(self.dir, n), os.path.join(out, n))
        return f"restored {', '.join(names)} from the backup {self.dir} (no latest.pt in {out})"


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    frac = (step - args.warmup) / max(args.steps - args.warmup, 1)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


def synthetic_batch(batch: int, size: int, device) -> dict:
    """A full-size random batch with every key the loss reads, for measuring memory and speed."""
    g = torch.Generator(device="cpu").manual_seed(0)
    out = {}
    for k in ID_KEYS:
        shape = {"own_species": (6,), "own_status": (6,), "own_moves": (6, 4), "opp_species": (6,), "opp_status": (6,), "opp_moves": (6, 4)}.get(k, (1,))
        out[f"ids_{k}"] = torch.randint(0, 8, (batch, size, *shape), generator=g)
    for k, width in (("own", (6, 10)), ("opp", (6, 7)), ("last", (34,)), ("field", (39,))):
        out[f"floats_{k}"] = torch.rand(batch, size, *width, generator=g)
    out["legal"] = torch.ones(batch, size, 9)
    out["blend"] = torch.full((batch, size, 9), 1 / 9)
    out["judge"] = torch.full((batch, size, 9), 1 / 9)
    out["win"] = torch.randint(0, 2, (batch,), generator=g).float()
    out["policy_ok"] = torch.ones(batch, size, dtype=torch.bool)
    out["scored"] = torch.ones(batch, size, dtype=torch.bool)
    out["played"] = torch.zeros(batch, size, dtype=torch.long)
    out["bc_ok"] = torch.ones(batch, size, dtype=torch.bool)
    out["bc_game"] = torch.zeros(batch, size, dtype=torch.bool)
    out["opp_action"] = torch.randint(0, 320, (batch, size), generator=g)
    return {k: v.to(device) for k, v in out.items()}


def probe_batch(model, args, device, candidates=(256, 192, 128, 96, 64, 48, 32), headroom: float = 0.85) -> int:
    """The largest candidate batch whose training steps fit in GPU memory (peak below `headroom` of the
    total), measured on a copy of the model with random data. CPU runs use a small fixed batch."""
    import copy

    if device.type != "cuda":
        return 32
    total = torch.cuda.get_device_properties(device).total_memory
    size = args.chunk + args.burn_in
    for b in candidates:
        probe = copy.deepcopy(model)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch = synthetic_batch(b, size, device)
            start = None
            for i in range(4):
                if i == 1:
                    torch.cuda.synchronize(device)
                    start = time.time()
                with torch.autocast(device.type, dtype=amp_dtype(args), enabled=amp_enabled(args, device)):
                    out = run_batch(probe, batch, args)
                one = torch.ones_like(out["kl"])
                loss = masked_mean(out["kl"], one) + masked_mean(out["bc"], one) + masked_mean(out["bce"], one)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize(device)
            rate = 3 / (time.time() - start)
            peak = torch.cuda.max_memory_allocated(device)
        except torch.cuda.OutOfMemoryError:
            del probe, opt
            torch.cuda.empty_cache()
            continue
        del probe, opt, batch
        torch.cuda.empty_cache()
        print(f"probe: batch {b} peak {peak / 2**30:.1f} of {total / 2**30:.1f} GiB, {rate:.2f} steps/s", flush=True)
        if peak <= headroom * total:
            return b
    return candidates[-1]


def rebalanced_sources(items: list[str], spec: str, validation: bool, fraction: int, seed: int):
    """--rebalance bots=W1,human=W2,otherbots=W3 over --source human=DIR and strong=DIR: every strong game seen from an imitated
    bot's side ("bots"), and fixed random subsets of the human games and of the other strong games sized so that the game counts
    are in the ratio of the weights; each is sampled with its weight."""
    from metamon.recurrent.dataset import select_games

    dirs = {name: rest.rsplit(":", 1)[0] for name, rest in (item.split("=", 1) for item in items)}
    weights = {k: float(v) for k, v in (kv.split("=") for kv in spec.split(","))}
    if set(weights) != {"bots", "human", "otherbots"} or weights["bots"] <= 0 or not {"human", "strong"} <= set(dirs):
        raise SystemExit("--rebalance needs bots=,human=,otherbots= weights and --source human=... and strong=...")
    bots = select_games(dirs["strong"], imitated=True)
    if not bots:
        raise SystemExit(f"--rebalance: no games by the imitated bots in {dirs['strong']}")
    per = len(bots) / weights["bots"]
    sources = [scan("bots", dirs["strong"], weights["bots"], validation, fraction, bots)]
    # (a weight of 0 leaves that source out: bots=1,human=0,otherbots=0 trains on the imitated bots' games alone)
    if weights["human"] > 0:
        human = select_games(dirs["human"], count=round(per * weights["human"]), seed=seed)
        sources.append(scan("human", dirs["human"], weights["human"], validation, fraction, human))
    if weights["otherbots"] > 0:
        other = select_games(dirs["strong"], imitated=False, count=round(per * weights["otherbots"]), seed=seed)
        sources.append(scan("otherbots", dirs["strong"], weights["otherbots"], validation, fraction, other))
    return sources


def parse_sources(items: list[str], validation: bool, fraction: int):
    sources = []
    for item in items:
        name, rest = item.split("=", 1)
        directory, weight = rest.rsplit(":", 1)
        sources.append(scan(name, directory, float(weight), validation, fraction))
    return sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--source", action="append", required=True, help="name=directory:weight")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", default="128", help="a number, or auto to take the largest that fits in GPU memory")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--burn-in", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per step (gradient accumulation: the same step in less GPU memory)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the KL term against the win BCE")
    ap.add_argument("--bc-weight", type=float, default=1.0, help="weight of the imitation term on the imitated bots' games")
    ap.add_argument("--stage", choices=("joint", "policy", "value"), default="joint",
                    help="policy: the win head gets no gradient (distillation and imitation only). value: starts from --init, trains the win head and lets the "
                         "backbone move only as far as --anchor-weight allows, with no policy loss of its own. joint: everything at once")
    ap.add_argument("--init", default=None, help="weights (weights_<step>.pt or a checkpoint) to start from; required for --stage value, and the frozen reference of its anchor")
    ap.add_argument("--value-bc-weight", type=float, default=0.0,
                    help="(--stage value) weight of the imitation term on the imitated bots' games: the policy then moves toward their played "
                         "actions there, held near --init everywhere by the anchor (default 0: no policy loss of its own)")
    ap.add_argument("--rebalance", default=None,
                    help="bots=W1,human=W2,otherbots=W3: train on all games seen from the imitated bots' side plus random subsets of the human "
                         "and other bot games, game counts in the ratio of the weights (see rebalanced_sources)")
    ap.add_argument("--freeze", choices=("none", "trunk", "actor"), default="none",
                    help="trunk: train only the win head MLP (embeddings, the recurrent transformer and the actor head stay as --init has them, "
                         "so the policy does not change). actor: train only the actor head (its linear layer and, with --deep-actor, the deep head) "
                         "on the frozen trunk")
    ap.add_argument("--deep-actor", action="store_true",
                    help="add the deeper actor head (model.DeepActor: 192 -> 64 -> 64 -> 9, added to the linear head, output "
                         "zero-initialised so the policy starts as --init's); kept by later runs")
    ap.add_argument("--opponent-head", action="store_true", help="add the opponent-action head (predicts the opponent's next move or switch)")
    ap.add_argument("--opp-weight", type=float, default=0.0, help="weight of the opponent head's cross-entropy (labels: the next decision's record of the opponent's action)")
    ap.add_argument("--kl-scope", choices=("all", "nonbot"), default="all",
                    help="where the KL to Exeggcute applies: every decision, or only the games not seen from an imitated bot's side "
                         "(nonbot: the bots' games are imitated, the others distilled)")
    ap.add_argument("--monitor", choices=("auto", "kl", "bce", "loss"), default="auto",
                    help="the validation number early stopping and weights_best.pt follow: auto = bce in the value stage, kl otherwise; "
                         "loss = the stage's own weighted loss (alpha * kl + bc_weight * bc + anchor_weight * anchor + value_weight * bce)")
    ap.add_argument("--keep-value-head", action="store_true", help="(--stage value) keep the win head of --init instead of re-initialising it")
    ap.add_argument("--value-weight", type=float, default=None, help="weight of the win BCE (default: 0 in the policy stage, else 1)")
    ap.add_argument("--anchor-weight", type=float, default=None, help="weight of KL(first-stage policy || current policy) (default: 5 in the value stage, else 0)")
    ap.add_argument("--target", choices=("mix", "blend", "judge"), default="blend")
    ap.add_argument("--mix", type=float, default=0.5, help="share of the proposer blend in the mixed target")
    ap.add_argument("--val-fraction", type=int, default=20, help="hold out 1 game in N")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=0, help="stop after this many validations in a row without an improvement of --min-delta (0: run all --steps)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="the smallest drop of the monitored validation loss that counts as progress (KL; the win BCE in the value stage)")
    ap.add_argument("--val-batches", type=int, default=0, help="limit validation to N batches (0 = all)")
    ap.add_argument("--save-every", type=int, default=5000, help="steps between numbered weights_<step>.pt snapshots")
    ap.add_argument("--ckpt-every", type=int, default=1, help="steps between writes of latest.pt (model, optimizer, step); 1 = every iteration")
    ap.add_argument("--backup-dir", default=None,
                    help="a second folder, on separate storage (e.g. a vast.ai local volume), that keeps weights_<step>.pt every --backup-every steps, "
                         "latest.pt (overwritten each time), weights_best.pt, weights_final.pt and log.txt. With --resume and no latest.pt in OUT, "
                         "the run restarts from the backup's")
    ap.add_argument("--backup-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16", help="autocast on CUDA; fp16 uses loss scaling")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "value":
        if not args.init:
            ap.error("--stage value needs --init (the first stage's weights)")
        args.alpha = 0.0  # (no distillation: the policy is held by the anchor)
        args.bc_weight = args.value_bc_weight  # (imitation of the bots' played actions, only when asked for)
    if args.value_weight is None:
        args.value_weight = 0.0 if args.stage == "policy" else 1.0
    if args.anchor_weight is None:
        args.anchor_weight = 5.0 if args.stage == "value" else 0.0
    if args.anchor_weight > 0 and not args.init:
        ap.error("--anchor-weight needs --init (the reference policy)")

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    backup = Backup(args.backup_dir) if args.backup_dir else None
    restored = None
    if backup and args.resume and not os.path.exists(os.path.join(args.out, "latest.pt")):
        restored = backup.restore(args.out)  # (before log.txt is opened, so the restored log continues)
    log_file = open(os.path.join(args.out, "log.txt"), "a")

    def log(text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    if backup:
        backup.log = log

    if args.rebalance:
        train_sources = rebalanced_sources(args.source, args.rebalance, False, args.val_fraction, args.seed)
        val_sources = rebalanced_sources(args.source, args.rebalance, True, args.val_fraction, args.seed)
    else:
        train_sources = parse_sources(args.source, False, args.val_fraction)
        val_sources = parse_sources(args.source, True, args.val_fraction)
    names = [s.name for s in train_sources]
    log(f"sources {' '.join(f'{s.name}:{len(s.paths)}games/{int(s.lengths.sum())}decisions/weight{s.weight}' for s in train_sources)}")
    log(f"validation {' '.join(f'{s.name}:{len(s.paths)}games' for s in val_sources)}")

    latest = os.path.join(args.out, "latest.pt")
    if backup:
        log(restored or f"backups every {args.backup_every} steps to {args.backup_dir} ({backup.free_gib():.1f} GiB free)")
    resume_state =torch.load(latest, map_location="cpu") if args.resume and os.path.exists(latest) else None
    start_step = resume_state["step"] if resume_state else 0

    heads = {"deep_actor": args.deep_actor, "opponent_head": args.opponent_head}
    if resume_state:
        model = PolicyValueModel.from_state(resume_state["model"], **heads)  # (the heads of the run being resumed)
    elif args.init:
        model = PolicyValueModel.from_state(torch.load(args.init, map_location="cpu"), **heads)
    else:
        model = PolicyValueModel(Config(**heads))
    model = model.to(device)
    if model.cfg.deep_actor or model.cfg.opponent_head:
        log(f"heads: deep actor {model.cfg.deep_actor}, opponent head {model.cfg.opponent_head}")
    if args.opp_weight > 0 and not model.cfg.opponent_head:
        raise SystemExit("--opp-weight needs --opponent-head (or --init weights that have one)")
    ref = None
    if args.init and not resume_state:
        log(f"initialised from {args.init}")
        if args.stage == "value" and not args.keep_value_head:
            for module in model.win.modules():  # (fresh random weights: whatever the first stage's win head learned is discarded)
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            log("win head re-initialised")
    if args.anchor_weight > 0:
        init = torch.load(args.init, map_location="cpu")
        ref = PolicyValueModel.from_state(init).to(device)
        ref.eval().requires_grad_(False)
        del init
        log(f"anchor: KL to the policy of {args.init}, weight {args.anchor_weight}")
    log(f"stage {args.stage}: alpha {args.alpha} bc {args.bc_weight} value {args.value_weight} anchor {args.anchor_weight}")
    log(f"model {count_parameters(model):,} parameters")
    if args.freeze == "trunk":
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("win."))
        log(f"frozen: everything but the win head ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters)")
    elif args.freeze == "actor":
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith(("policy.", "policy_deep.")))
        log(f"frozen: everything but the actor head ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters)")
    if str(args.batch) == "auto":
        args.batch = probe_batch(model, args, device)
        log(f"batch size {args.batch} (measured)")
    args.batch = int(args.batch)

    train_loader = DataLoader(
        Windows(train_sources, args.burn_in, args.chunk, args.seed + start_step), batch_size=args.batch, num_workers=args.workers,
        collate_fn=collate, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
    )
    val_loader = DataLoader(
        Windows(val_sources, args.burn_in, args.chunk, endless=False), batch_size=args.batch, num_workers=min(args.workers, 4), collate_fn=collate,
    )
    if args.compile:
        model.step = torch.compile(model.step)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and "role" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and not (p.ndim >= 2 and "role" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and device.type == "cuda")
    step = 0
    best, bad, best_state = math.inf, 0, None  # the monitored validation loss, validations since it improved, and the weights that had it
    if resume_state:
        opt.load_state_dict(resume_state["opt"])  # (the model was built from it above)
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        step = resume_state["step"]
        best, bad = resume_state.get("best", math.inf), resume_state.get("bad", 0)
        log(f"resumed at step {step}")
        del resume_state
        best_path = os.path.join(args.out, "weights_best.pt")
        if os.path.exists(best_path):
            best_state = torch.load(best_path, map_location="cpu")
    checkpointer = Checkpointer(latest)
    snapshot_args = vars(args)

    def snapshot() -> dict:
        return {"model": to_cpu(model.state_dict()), "opt": to_cpu(opt.state_dict()), "scaler": scaler.state_dict(), "step": step, "args": snapshot_args, "best": best, "bad": bad}

    model.train()
    start, window_start = time.time(), time.time()
    running = {"loss": 0.0, "kl": 0.0, "bc": 0.0, "bce": 0.0, "top1": 0.0, "anchor": 0.0, "opp": 0.0, "n": 0}
    try:
        for batch in train_loader:
            if step >= args.steps or (args.patience > 0 and bad >= args.patience):  # (also on a restart after an early stop)
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args)
            # (--accum N: the batch goes through the model in N micro-batches whose gradients add up; every term is a sum over the
            # micro-batch divided by the whole batch's count, so the loss and its gradient are the whole batch's)
            full_pol = batch["policy_ok"] & batch["scored"]
            full_kl = (full_pol & ~batch["bc_game"]) if args.kl_scope == "nonbot" else full_pol
            counts = {
                "pol": full_pol.float().sum().clamp_min(1),
                "kl": full_kl.float().sum().clamp_min(1),
                "sc": batch["scored"].float().sum().clamp_min(1),
                "bc": (batch["bc_ok"] & batch["scored"]).float().sum().clamp_min(1),
                "opp": ((batch["opp_action"] >= 0) & batch["scored"]).float().sum().clamp_min(1),
            }
            opt.zero_grad(set_to_none=True)
            parts = max(1, args.accum)
            size = batch["scored"].shape[0]
            bounds = [size * i // parts for i in range(parts + 1)]
            totals = {k: 0.0 for k in ("loss", "kl", "bc", "bce", "anchor", "opp", "top1")}
            for i0, i1 in zip(bounds[:-1], bounds[1:]):
                if i1 <= i0:
                    continue
                mb = {k: v[i0:i1] for k, v in batch.items()}
                with torch.autocast(device.type, dtype=amp_dtype(args), enabled=amp_enabled(args, device)):
                    out = run_batch(model, mb, args, ref)
                pol_mask = mb["policy_ok"] & mb["scored"]
                pol = pol_mask.float()
                kl_mask = (pol_mask & ~mb["bc_game"]).float() if args.kl_scope == "nonbot" else pol
                sc = mb["scored"].float()
                bcm = (mb["bc_ok"] & mb["scored"]).float()
                kl = (out["kl"] * kl_mask).sum() / counts["kl"]
                bc = (out["bc"] * bcm).sum() / counts["bc"]
                bce = (out["bce"] * sc).sum() / counts["sc"]
                anchor = (out["anchor"] * sc).sum() / counts["sc"] if ref is not None else kl.new_zeros(())
                loss = args.alpha * kl + args.bc_weight * bc + args.anchor_weight * anchor
                opp = kl.new_zeros(())
                if "opp" in out:
                    opp = (out["opp"] * ((mb["opp_action"] >= 0) & mb["scored"]).float()).sum() / counts["opp"]
                    if args.opp_weight > 0:
                        loss = loss + args.opp_weight * opp
                if args.value_weight > 0:  # (left out of the graph at 0, so the policy stage never touches the win head, not even to decay it)
                    loss = loss + args.value_weight * bce
                scaler.scale(loss).backward()
                for k, v in (("loss", loss), ("kl", kl), ("bc", bc), ("bce", bce), ("anchor", anchor), ("opp", opp)):
                    totals[k] += v.item()
                totals["top1"] += ((out["top1"] * pol).sum() / counts["pol"]).item()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)  # skips the update when fp16 gradients overflowed
            scaler.update()
            step += 1
            for k in ("loss", "kl", "bc", "bce", "anchor", "opp", "top1"):
                running[k] += totals[k]
            running["n"] += 1
            if step % 100 == 0:
                n = running["n"]
                rate = n * args.batch * (args.chunk + args.burn_in) / (time.time() - window_start)
                log(f"step {step} loss {running['loss'] / n:.4f} kl {running['kl'] / n:.4f} bc {running['bc'] / n:.4f} bce {running['bce'] / n:.4f} top1 {running['top1'] / n:.3f} anchor {running['anchor'] / n:.4f} "
                    + (f"opp {running['opp'] / n:.4f} " if model.cfg.opponent_head else "") + f"grad {norm:.2f} lr {lr_at(step, args):.2e} {rate:,.0f} steps/s")
                running = {k: 0.0 for k in running}
                running["n"] = 0
                window_start = time.time()
            stop = False
            if step % args.val_every == 0 or step == args.steps:
                result = validate(model, val_loader if not args.val_batches else _limit(val_loader, args.val_batches), args, device, ref)
                log(f"validation step {step} decisions {int(result['decisions'])} kl {result['kl']:.4f} top1 {result['top1']:.3f} "
                    f"anchor {result['anchor']:.4f} bce {result['bce']:.4f} brier {result['brier']:.4f} | imitated bots: {int(result['bc_decisions'])} decisions, "
                    f"played-action nll {result['bc']:.4f} accuracy {result['bc_top1']:.3f}"
                    + (f" | kl on the other games {result['kl_nonbot']:.4f}" if args.kl_scope == "nonbot" else "")
                    + (f" | opponent's next action: {int(result['opp_decisions'])} decisions, nll {result['opp']:.4f} accuracy {result['opp_top1']:.3f}"
                       if model.cfg.opponent_head else ""))
                log("calibration " + " ".join(f"[{i / 10:.1f}-{(i + 1) / 10:.1f}) n={c} pred={p:.3f} won={w:.3f}" for i, (c, p, w) in enumerate(result["calibration"]) if c))
                log("by source " + " ".join(f"{names[s]}: kl {k:.4f} bce {b:.4f}" for s, (k, b) in sorted(result["by_source"].items())))
                monitor = args.monitor if args.monitor != "auto" else ("bce" if args.stage == "value" else "kl")
                if monitor == "loss":
                    monitored = (args.alpha * (result["kl_nonbot"] if args.kl_scope == "nonbot" else result["kl"]) + args.bc_weight * result["bc"]
                                 + args.anchor_weight * result["anchor"] + args.value_weight * result["bce"] + args.opp_weight * result["opp"])
                else:
                    monitored = result[monitor]
                if monitored < best - args.min_delta:
                    best, bad = monitored, 0
                    best_state = to_cpu(model.state_dict())
                    save_atomic(best_state, os.path.join(args.out, "weights_best.pt"))
                else:
                    bad += 1
                log(f"best validation {monitor} {best:.4f}; {bad} validation(s) without progress of {args.min_delta:g}" + (f" (patience {args.patience})" if args.patience else ""))
                stop = args.patience > 0 and bad >= args.patience
                if backup and bad == 0:
                    backup.copy(os.path.join(args.out, "weights_best.pt"))
            backup_now = backup is not None and (step % args.backup_every == 0 or step == args.steps or stop)
            if step % args.ckpt_every == 0 or step == args.steps or backup_now:
                state = snapshot()
                checkpointer.submit(state)
                if backup_now:
                    backup.save_step(step, state)
                    backup.copy(os.path.join(args.out, "log.txt"))
            if step % args.save_every == 0 or step == args.steps:
                save_atomic(to_cpu(model.state_dict()), os.path.join(args.out, f"weights_{step}.pt"))
            if stop:
                log(f"early stop at step {step}: no progress for {bad} validations; the final weights are the best ones (validation {best:.4f})")
                break
    finally:
        checkpointer.close()  # the last submitted state reaches the disk even when the run is interrupted
    # weights_final.pt: the best validated weights after an early stop, else the last step's
    final = best_state if (args.patience > 0 and bad >= args.patience and best_state is not None) else to_cpu(model.state_dict())
    save_atomic(final, os.path.join(args.out, "weights_final.pt"))
    log(f"done at step {step}, {time.time() - start:.0f}s")
    if backup:
        backup.copy(os.path.join(args.out, "weights_final.pt"))
        backup.copy(os.path.join(args.out, "log.txt"))


def _limit(loader, n):
    class Limited:
        def __iter__(self):
            for i, b in enumerate(loader):
                if i >= n:
                    return
                yield b
    return Limited()


if __name__ == "__main__":
    import signal

    # Ctrl-C, and the launcher's terminate (SIGTERM) when it stops a run, end training the same way: the step in progress is
    # dropped, the last finished one is in latest.pt (the checkpoint writer flushes in main's finally), and a restart resumes there
    def _interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt)
    try:
        main()
    except KeyboardInterrupt:
        print(f"{time.strftime('%H:%M:%S')} interrupted: latest.pt holds the last finished step; run again with --resume to continue", flush=True)
        # (the checkpoint is on disk: stop the data loader's workers and exit at once, instead of waiting for them to wind down,
        # which can take long enough for the launcher to kill the process)
        import multiprocessing

        for worker in multiprocessing.active_children():
            worker.kill()
        os._exit(130)
