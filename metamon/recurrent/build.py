"""Turns (parsed trajectory, Exeggcute labels) pairs into training games: features encoded once, stored
compactly, so that training never runs the Python encoder.

One `<name>.npz` per game with
    ids_<key>      int16   [T, ...]   (model.ID_KEYS)
    floats_<key>   float16 [T, ...]   (model.FLOAT_KEYS)
    blend, judge   float16 [T, 9]     Exeggcute targets over this model's policy slots
    legal          uint8   [T, 9]     the legal policy slots
    policy_ok      bool [T]  the decision has a policy target (the observer acted, labels exist, and none
                             of the label mass fell on an action with no slot)
    win            float32 scalar   the observer's result: 1 win, 0.5 tie, 0 loss
    played         int8 [T]  what the observer actually played: a policy slot (0-8), 9 for "some move" (the log
                             does not say which: it was asleep, fully paralyzed, flinched, ...), -1 for no usable action
    kind           uint8 [T] the decision's kind (see decision_kind): normal, fight-or-switch, forced
    bc_game        bool scalar  the observer is one of the `--bc-players` (strong bots whose play is imitated)

Exeggcute's action indices (moves 0-3 by sorted name, then the alive reserves by sorted species) are
mapped to this model's slots (the active's sorted moves, then the five reserve slots by sorted species,
fainted ones included) with `features.action_slot_map`.

    python -m metamon.recurrent.build TRAJ_DIR LABEL_DIR OUT_DIR TABLES_DIR [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from multiprocessing import Pool

import numpy as np

from metamon.recurrent.tables import Tables
from metamon.recurrent.model import FLOAT_KEYS, ID_KEYS
from metamon.recurrent.records import iter_decisions

_TABLES = None
_BC_PLAYERS: set[str] = set()
DEFAULT_BC_PLAYERS = "Tauros-v1,Tauros-v2,SmallG1Online-v2,SmallG1Online-v3"
NAME_RE = re.compile(r"^[^_]+_[^_]*_(.+?)_vs_(.+?)_\d\d-\d\d-\d{4}_(?:WIN|LOSS|TIE)\.json\.lz4$")


# Decision kinds. What the player could do depends on its situation, and Metamon's own legal set does not say
# (it always lists every move and every alive switch):
#   FIGHT_OR_SWITCH  asleep or frozen: any move just fails, so the real choice is to fight or to switch out
#   FORCED_RECHARGE  recharging after Hyper Beam: the only option is the no-op
#   FORCED_TRAPPED   partially trapped: no switching, and every move is the same
#   NO_CHOICE        the game ended before the player chose (timeout)
NORMAL, FIGHT_OR_SWITCH, FORCED_RECHARGE, FORCED_TRAPPED, NO_CHOICE = 0, 1, 2, 3, 4
FORCED = (FORCED_RECHARGE, FORCED_TRAPPED, NO_CHOICE)
FIGHT = 9  # `played` code: some move, which one is unknown or does not matter


def decision_kind(d) -> int:
    if d.forced_switch:  # choosing a replacement: moves are not legal, nothing else is special
        return NORMAL
    if d.recharge_next:
        return FORCED_RECHARGE
    if d.trapped:
        return FORCED_TRAPPED
    if d.action_index == -1 and d.action_reason == "game_ended":
        return NO_CHOICE
    if d.own_status in ("slp", "frz"):
        return FIGHT_OR_SWITCH
    return NORMAL


def adjust(kind: int, legal: np.ndarray, blend: np.ndarray, judge: np.ndarray):
    """The decision's real options and targets, from the label row over Metamon's over-inclusive options."""
    legal, blend, judge = legal.copy(), blend.copy(), judge.copy()
    if kind == FORCED_RECHARGE:
        legal[:] = 0
        legal[0] = 1  # the no-op uses the first move slot
        blend[:] = judge[:] = legal
    elif kind == FORCED_TRAPPED:
        legal[4:] = 0
        blend[:] = judge[:] = legal / max(legal.sum(), 1)
    elif kind == FIGHT_OR_SWITCH and legal[:4].sum() > 0:
        # every move fails the same way: the fight option's probability is spread evenly over the moves
        moves = legal[:4] > 0
        for row in (blend, judge):
            row[:4] = moves * (row[:4].sum() / moves.sum())
    return legal, blend, judge


def played_code(d, kind: int, legal_row: np.ndarray) -> int:
    if kind in FORCED:
        return -1
    a = d.action_index
    if a == -1:  # a move was chosen but never ran (asleep, paralyzed, flinched, ...): which one is unknown
        return FIGHT if not d.forced_switch and legal_row[:4].sum() > 0 else -1
    if not 0 <= a < len(d.slot_map):
        return -1
    slot = int(d.slot_map[a])
    if slot < 0 or legal_row[slot] == 0:
        return -1
    return FIGHT if kind == FIGHT_OR_SWITCH and slot < 4 else slot


def observer_name(path: str) -> str | None:
    """The observer's username, from the parser's file name `<id>_<rating>_<player>_vs_<opponent>_<date>_<result>`."""
    m = NAME_RE.match(os.path.basename(path))
    return m.group(1) if m else None


def build_game(args) -> tuple[str, int] | tuple[str, str]:
    traj_path, label_path, out_path = args
    global _TABLES
    try:
        labels = np.load(label_path)
        n = len(labels["chosen"])
        decisions = []
        for d in iter_decisions(traj_path, _TABLES):
            if len(decisions) == n:
                break
            decisions.append(d)
        if len(decisions) != n:
            return traj_path, f"{len(decisions)} decisions but {n} labels"
        out = {}
        for key in ID_KEYS:
            out[f"ids_{key}"] = np.stack([d.features.ids[key] for d in decisions]).astype(np.int16)
        for key in FLOAT_KEYS:
            out[f"floats_{key}"] = np.stack([d.features.floats[key] for d in decisions]).astype(np.float16)
        mapped = {k: np.zeros((n, 9), np.float32) for k in ("blend", "judge", "legal")}
        lost = np.zeros(n, bool)  # label mass on an action that has no slot in this model
        for t, d in enumerate(decisions):
            for k in mapped:
                src = labels[k][t].astype(np.float64)
                for a, slot in enumerate(d.slot_map):
                    if slot >= 0:
                        mapped[k][t, slot] = src[a]
                    elif src[a] > 1e-6:
                        lost[t] = True
        kinds = np.array([decision_kind(d) for d in decisions], np.uint8)
        for t in range(n):
            mapped["legal"][t], mapped["blend"][t], mapped["judge"][t] = adjust(kinds[t], mapped["legal"][t], mapped["blend"][t], mapped["judge"][t])
        out["blend"] = mapped["blend"].astype(np.float16)
        out["judge"] = mapped["judge"].astype(np.float16)
        out["legal"] = mapped["legal"].astype(np.uint8)
        out["kind"] = kinds
        # forced decisions teach nothing (one option); the others are real choices even when the log shows no
        # action (the player chose a move that then failed)
        out["policy_ok"] = labels["has_blend"] & (mapped["legal"].sum(1) > 0) & ~lost & ~np.isin(kinds, FORCED)
        out["win"] = np.float32(decisions[0].win)
        out["played"] = np.array([played_code(d, kinds[t], mapped["legal"][t]) for t, d in enumerate(decisions)], np.int8)
        out["bc_game"] = np.bool_(observer_name(traj_path) in _BC_PLAYERS)
        np.savez_compressed(out_path, **out)
        return traj_path, n
    except Exception as exc:  # report and keep going
        return traj_path, f"{type(exc).__name__}: {exc}"


def _init(tables_dir: str, bc_players: tuple = ()) -> None:
    global _TABLES, _BC_PLAYERS
    _TABLES = Tables.load(tables_dir)
    _BC_PLAYERS = set(bc_players)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectories")
    ap.add_argument("labels")
    ap.add_argument("output")
    ap.add_argument("tables")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--bc-players", default=DEFAULT_BC_PLAYERS, help="comma-separated usernames whose own actions are imitated")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    jobs = []
    for label in sorted(glob.glob(os.path.join(args.labels, "*.npz"))):
        stem = os.path.basename(label)[: -len(".npz")]
        traj = glob.glob(os.path.join(args.trajectories, "**", stem + ".json.lz4"), recursive=True)
        out = os.path.join(args.output, stem + ".npz")
        if traj and not os.path.exists(out):
            jobs.append((traj[0], label, out))
    done = failed = 0
    bc_players = tuple(x for x in args.bc_players.split(",") if x)
    _init(args.tables, bc_players)  # fail here, not in a worker that the pool would respawn forever
    with Pool(args.workers, initializer=_init, initargs=(args.tables, bc_players)) as pool:
        for path, result in pool.imap_unordered(build_game, jobs, chunksize=4):
            if isinstance(result, str):
                failed += 1
                print(f"failed {os.path.basename(path)}: {result}", flush=True)
            else:
                done += 1
            if (done + failed) % 100 == 0:
                print(f"built {done}, failed {failed} of {len(jobs)}", flush=True)
    print(f"built {done}, failed {failed} of {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
