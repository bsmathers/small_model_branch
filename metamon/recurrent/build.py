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

Exeggcute's action indices (moves 0-3 by sorted name, then the alive reserves by sorted species) are
mapped to this model's slots (the active's sorted moves, then the five reserve slots by sorted species,
fainted ones included) with `features.action_slot_map`.

    python -m metamon.recurrent.build TRAJ_DIR LABEL_DIR OUT_DIR TABLES_DIR [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np

from metamon.recurrent.tables import Tables
from metamon.recurrent.model import FLOAT_KEYS, ID_KEYS
from metamon.recurrent.records import iter_decisions

_TABLES = None


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
        out["blend"] = mapped["blend"].astype(np.float16)
        out["judge"] = mapped["judge"].astype(np.float16)
        out["legal"] = mapped["legal"].astype(np.uint8)
        acted = np.array([d.action_index >= 0 for d in decisions])
        out["policy_ok"] = acted & labels["has_blend"] & (mapped["legal"].sum(1) > 0) & ~lost
        out["win"] = np.float32(decisions[0].win)
        np.savez_compressed(out_path, **out)
        return traj_path, n
    except Exception as exc:  # report and keep going
        return traj_path, f"{type(exc).__name__}: {exc}"


def _init(tables_dir: str) -> None:
    global _TABLES
    _TABLES = Tables.load(tables_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectories")
    ap.add_argument("labels")
    ap.add_argument("output")
    ap.add_argument("tables")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
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
    _init(args.tables)  # fail here, not in a worker that the pool would respawn forever
    with Pool(args.workers, initializer=_init, initargs=(args.tables,)) as pool:
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
