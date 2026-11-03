"""Folds the original parser's states and actions into the corrected trajectories.

Every trajectory then holds, per decision t:

    states[t]          the corrected state                 (input of the distilled model, via the extras)
    actions[t]         the corrected action label
    extras[t]          what the corrected parser adds (locks, traps, effects, Counter memory, parties ...)
    legacy_states[t]   the state the ORIGINAL parser produces for the same decision   (Exeggcute's input)
    legacy_actions[t]  the original parser's action label                              (Exeggcute's previous action)

Exeggcute was trained on the original parser's output, so it is run on the legacy pair only; the corrected parser's output is
what the new model is distilled from. Both parsers read the same log and see the same turns, so the lists have equal length;
a game where they disagree (or has no legacy twin) is moved to a `_rejected` sibling folder, never deleted.

    python tools/legacy_parse.py RAW_DIR LEGACY_DIR        # the original parser
    python -m metamon.recurrent.parse RAW_DIR TRAJ_DIR      # the corrected parser
    python -m metamon.recurrent.merge_legacy TRAJ_DIR LEGACY_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
from urllib.parse import quote

import lz4.frame

NAME_RE = re.compile(r"^([^_]+)_[^_]*_(.*?)_vs_(.*?)_\d\d-\d\d-\d{4}_(?:WIN|LOSS|TIE)\.json\.lz4$")


def legacy_path(legacy_dir: str, traj_name: str) -> str | None:
    m = NAME_RE.match(traj_name)
    return os.path.join(legacy_dir, f"{m.group(1)}_{quote(m.group(2), safe='')}.json.lz4") if m else None


def merge_one(traj_path: str, legacy_dir: str) -> str:
    """'merged', 'already' or the reason the game is rejected."""
    with lz4.frame.open(traj_path, "rb") as f:
        data = json.loads(f.read())
    if "legacy_states" in data:
        return "already"
    path = legacy_path(legacy_dir, os.path.basename(traj_path))
    if path is None or not os.path.exists(path):
        return "no legacy twin"
    with lz4.frame.open(path, "rb") as f:
        legacy = json.loads(f.read())
    n = len(data["states"])
    if not (len(legacy["states"]) == len(legacy["actions"]) == n == len(data["actions"]) == len(data["extras"])):
        return f"lengths differ: corrected {n}, legacy {len(legacy['states'])}"
    data["legacy_states"], data["legacy_actions"] = legacy["states"], legacy["actions"]
    tmp = traj_path + ".tmp"
    with lz4.frame.open(tmp, "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))
    os.replace(tmp, traj_path)
    return "merged"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj_dir")
    ap.add_argument("legacy_dir")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))
    names = sorted(f for f in os.listdir(args.traj_dir) if f.endswith(".json.lz4"))[i::n]
    rejected_dir = args.traj_dir.rstrip("/") + "_rejected"
    counts: dict[str, int] = {}
    for name in names:
        result = merge_one(os.path.join(args.traj_dir, name), args.legacy_dir)
        counts[result.split(":")[0]] = counts.get(result.split(":")[0], 0) + 1
        if result not in ("merged", "already"):
            os.makedirs(rejected_dir, exist_ok=True)
            os.replace(os.path.join(args.traj_dir, name), os.path.join(rejected_dir, name))
            with open(os.path.join(rejected_dir, "reasons.txt"), "a") as f:
                f.write(f"{name}\t{result}\n")
    print(f"legacy merge: {counts}")


if __name__ == "__main__":
    main()
