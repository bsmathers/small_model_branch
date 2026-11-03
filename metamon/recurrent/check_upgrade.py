"""Measures how well `upgrade.derive_extras` reproduces the raw-log parser's extras.

    python -m metamon.recurrent.check_upgrade DIR_WITH_TRUE_EXTRAS [--limit N]

For every file the extras are derived from its states and actions alone, then compared with the true ones
field by field over all states.
"""
import argparse
import collections
import glob
import json
import os
import re

import lz4.frame

from metamon.recurrent.records import _kind
from metamon.recurrent.upgrade import derive_extras

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def action_key(a: dict) -> tuple:
    kind = _kind(a)
    return (kind, norm(a["name"]) if kind in ("move", "switch") else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.directory, "**", "*.json.lz4"), recursive=True))
    if args.limit:
        files = files[: args.limit]
    n = 0
    hits = collections.Counter()
    conf = {"last_player_action": collections.Counter(), "last_opponent_action": collections.Counter()}
    cant = {"player_cant": collections.Counter(), "opponent_cant": collections.Counter()}
    trap = {"player_trapped": collections.Counter(), "opponent_trapped": collections.Counter()}
    for f in files:
        d = json.loads(lz4.frame.open(f).read())
        truth, derived = d["extras"], derive_extras(d["states"], d["actions"])
        for t, (a, b) in enumerate(zip(truth, derived)):
            n += 1
            for k in ("last_player_action", "last_opponent_action"):
                ka, kb = action_key(a[k]), action_key(b[k])
                hits[k] += ka == kb
                hits[k + "_kind"] += ka[0] == kb[0]
                conf[k][(ka[0], kb[0])] += 1
            for k in ("player_cant", "opponent_cant"):
                cant[k][(a[k], b[k])] += 1
                hits[k] += a[k] == b[k]
            for k in ("player_trapped", "opponent_trapped"):
                trap[k][(bool(a[k]), bool(b[k]))] += 1
                hits[k] += bool(a[k]) == bool(b[k])
            names = lambda team: [norm(m["name"]) for m in team if m]
            hits["own_party"] += sorted(names(a["player_team"])) == sorted(names(b["player_team"]))
            hits["opp_party"] += names(a["opponent_team"]) == names(b["opponent_team"])
            sc = lambda e, k: sorted(norm(c) for c in e[k])
            hits["own_conditions"] += sc(a, "player_conditions_all") == sc(b, "player_conditions_all")
            hits["opp_conditions"] += sc(a, "opponent_conditions_all") == sc(b, "opponent_conditions_all")
    print(f"{len(files)} files, {n} states")
    for k in ("last_player_action", "last_opponent_action"):
        print(f"{k}: exact {hits[k] / n:.4f}, kind only {hits[k + '_kind'] / n:.4f}")
        wrong = {key: c for key, c in conf[k].items() if key[0] != key[1]}
        print("   kind errors (true -> derived):", dict(sorted(wrong.items(), key=lambda x: -x[1])[:6]))
    for k, c in cant.items():
        tp = sum(v for (a, b), v in c.items() if a is not None and a == b)
        true_n = sum(v for (a, b), v in c.items() if a is not None)
        derived_n = sum(v for (a, b), v in c.items() if b is not None)
        wrong = sum(v for (a, b), v in c.items() if b is not None and a != b)
        print(f"{k}: exact {hits[k] / n:.4f}; reasons present {true_n}, recovered {tp} (recall {tp / max(true_n, 1):.3f}), derived {derived_n}, wrong {wrong}")
    for k, c in trap.items():
        tp, fp, fn = c[(True, True)], c[(False, True)], c[(True, False)]
        print(f"{k}: exact {hits[k] / n:.4f}; true {tp + fn}, recovered {tp}, false alarms {fp}")
    for k in ("own_party", "opp_party", "own_conditions", "opp_conditions"):
        print(f"{k}: exact {hits[k] / n:.4f}")


if __name__ == "__main__":
    main()
