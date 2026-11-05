"""Verifies a folder of label files (.npz from metamon.recurrent.label) against the trajectories they were made from.

    python -m metamon.recurrent.check_labels LABELS_DIR [TRAJECTORIES_DIR] [--procs N] [--remove-bad]

Per file: it opens (a half-written or truncated file does not), has every array, has one row per decision of its trajectory (when the folder is
given), holds only finite numbers, and each row is internally consistent: the chosen action is legal, a forced decision (one legal action)
chose it, and the blend and judge distributions sum to 1 over legal actions. Prints the counts, the first few problems per kind, and writes the
bad files' paths to LABELS_DIR/bad_labels.txt. With --remove-bad it deletes exactly those files (so the labeller redoes them on its next run).
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np

KEYS = ("blend", "judge", "legal", "chosen", "judge_calls", "has_blend", "has_judge")


def check_one(args):
    path, traj_dir = args
    try:
        z = np.load(path)
        data = {k: z[k] for k in KEYS}  # (reads and decompresses every array: a truncated file fails here)
    except Exception as e:  # noqa: BLE001
        return path, f"unreadable: {type(e).__name__}: {str(e)[:80]}"
    n = len(data["chosen"])
    if any(len(data[k]) != n for k in KEYS):
        return path, "arrays of different lengths"
    if n == 0:
        return path, "no decisions"
    if traj_dir:
        stem = os.path.basename(path)[: -len(".npz")]
        traj = os.path.join(traj_dir, stem + ".json.lz4")
        if not os.path.exists(traj):
            return path, "no trajectory with this name"
        from metamon.recurrent.records import load_trajectory

        want = len(load_trajectory(traj)["legacy_states"]) - 1
        if want != n:
            return path, f"{n} rows, trajectory has {want} decisions"
    for k in ("blend", "judge"):
        if not np.isfinite(data[k]).all():
            return path, f"non-finite {k}"
    chosen, legal = data["chosen"].astype(np.int64), data["legal"] > 0.5
    if (chosen < 0).any() or (chosen >= legal.shape[1]).any():
        return path, "a decision without a chosen action"
    rows = np.arange(n)
    if not legal[rows, chosen].all():
        return path, "chosen action is not legal"
    forced = legal.sum(1) == 1
    if forced.any() and not (legal[forced].argmax(1) == chosen[forced]).all():
        return path, "a forced decision chose another action"
    for k, flag in (("blend", "has_blend"), ("judge", "has_judge")):
        have = data[flag].astype(bool)
        if have.any():
            sums = data[k][have].sum(1)
            if np.abs(sums - 1.0).max() > 1e-3:
                return path, f"{k} rows do not sum to 1"
            if (data[k][have][~legal[have]] > 1e-6).any():
                return path, f"{k} puts mass on an illegal action"
    return path, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("trajectories", nargs="?", default=None, help="the folder of .json.lz4 files the labels were made from (checks the row counts)")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--remove-bad", action="store_true", help="delete the files that fail (so the next labelling run redoes them)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.labels, "*.npz")))
    leftovers = glob.glob(os.path.join(args.labels, "*.part"))
    print(f"{len(files)} label files, {len(leftovers)} unfinished .part files", flush=True)
    bad: dict[str, list[str]] = {}
    with Pool(args.procs) as pool:
        for i, (path, problem) in enumerate(pool.imap_unordered(check_one, [(f, args.trajectories) for f in files], chunksize=64), 1):
            if problem:
                bad.setdefault(problem.split(":")[0].split(",")[0], []).append(path)
            if i % 20000 == 0:
                print(f"  checked {i}/{len(files)}", flush=True)
    total_bad = sum(len(v) for v in bad.values())
    print(f"checked {len(files)}: {len(files) - total_bad} ok, {total_bad} bad")
    for kind, paths in sorted(bad.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(paths):7d}  {kind}   e.g. {os.path.basename(paths[0])}")
    out = os.path.join(args.labels, "bad_labels.txt")
    with open(out, "w") as f:
        f.writelines(p + "\n" for paths in bad.values() for p in paths)
    print(f"bad files listed in {out}")
    if args.remove_bad and total_bad:
        for paths in bad.values():
            for p in paths:
                os.remove(p)
        print(f"removed {total_bad} files; run the labelling again to redo them")


if __name__ == "__main__":
    main()
