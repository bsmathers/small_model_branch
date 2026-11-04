#!/usr/bin/env python
"""Parses raw Showdown logs with the ORIGINAL (upstream) replay parser: the states and actions that Exeggcute was trained on.

    python tools/legacy_parse.py RAW_DIR OUT_DIR [--every N] [--shard i/n] [--limit N]

Upstream's forward.py, backward.py and replay_state.py are the only parser files our parser changes. They are kept verbatim in
metamon/recurrent/legacy_parser/ (from upstream commit 83f2fb04), and an import hook points the three module names at them for
this process only, so everything else (state conversion, team prediction) is the one shared code base. Run as a script, not
with -m: the hook has to be installed before `metamon` is imported.

Output: OUT_DIR/<game id>_<observer>.json.lz4 with {"states", "actions"}, where the observer name is URL-quoted.
`python -m metamon.recurrent.merge_legacy` then folds these into the corrected trajectories.
"""
import argparse
import importlib.abc
import importlib.util
import json
import os
import random
import re
import sys
import warnings
import zlib
from datetime import datetime
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.path.abspath(os.path.join(HERE, "..", "metamon", "recurrent", "legacy_parser"))
PACKAGE = "metamon.backend.replay_parser"
LEGACY_MODULES = ("replay_state", "forward", "backward")


class LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(PACKAGE + ".") and fullname.split(".")[-1] in LEGACY_MODULES and fullname.count(".") == PACKAGE.count(".") + 1:
            file = os.path.join(LEGACY_DIR, fullname.split(".")[-1] + ".py")
            return importlib.util.spec_from_file_location(fullname, file)
        return None


sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))  # this checkout, not an installed copy of metamon
sys.meta_path.insert(0, LegacyFinder())


def wrap_log(path: str) -> dict:
    text = open(path, errors="ignore").read()
    players = {}
    for line in text.splitlines():
        if line.startswith("|player|"):
            parts = line.split("|")
            if len(parts) > 3 and parts[2] in ("p1", "p2"):
                players[parts[2]] = parts[3]
    m = re.search(r"^\|t:\|(\d+)", text, re.M)
    tier = re.search(r"^\|tier\|(.+)$", text, re.M)
    fmt = re.sub(r"[^a-z0-9]", "", tier.group(1).lower()) if tier else os.path.basename(path).split("-")[0]
    return {"players": [players["p1"], players["p2"]], "uploadtime": int(m.group(1)) if m else 0, "formatid": fmt, "log": text}


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":  # the same fixed hash seed as metamon.recurrent.parse
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import lz4.frame
    import numpy as np

    from metamon.backend.replay_parser import backward, forward, replay_state
    from metamon.backend.replay_parser.exceptions import BackwardException, ForwardException
    from metamon.backend.replay_parser.parse_replays import ReplayParser

    assert forward.__file__.startswith(os.path.abspath(LEGACY_DIR)), "the legacy parser was not loaded"
    files = sorted(os.path.join(args.raw_dir, f) for f in os.listdir(args.raw_dir) if f.endswith(".log"))
    files = files[:: max(1, args.every)]
    shard, shards = map(int, args.shard.split("/"))
    files = files[shard::shards]
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    parser = ReplayParser(replay_output_dir=None)
    done = ok = 0
    for path in files:
        gameid = os.path.basename(path).rsplit(".", 1)[0].replace("_", "-")
        names = wrap_log(path)["players"]
        if names[0] == names[1]:
            continue  # the two points of view cannot be told apart by name: left out (the merge drops such games)
        outputs = [os.path.join(args.out_dir, f"{gameid}_{quote(p, safe='')}.json.lz4") for p in names]
        if all(os.path.exists(o) for o in outputs):
            continue  # resumable
        data = wrap_log(path)
        random.seed(gameid)  # as in metamon.recurrent.parse: the team predictor is random
        np.random.seed(zlib.crc32(gameid.encode()) % (2**32))
        try:
            replay = forward.ParsedReplay(gameid=gameid, format=data["formatid"], time_played=datetime.fromtimestamp(int(data["uploadtime"])))
            replay = forward.forward_fill(replay, parser.clean_log(data), verbose=False)
            from_p1, from_p2 = backward.backward_fill(replay, team_predictor=parser.team_predictor)
            for pov, out in zip((from_p1, from_p2), outputs):
                replay_states, actions = parser.povreplay_to_state_action(pov)
                states, action_idxs = parser.state_action_to_obs_action_reward(replay_states, actions)
                with lz4.frame.open(out, "wb") as f:
                    f.write(json.dumps({"states": [s.to_dict() for s in states], "actions": action_idxs}).encode("utf-8"))
            ok += 1
        except Exception as e:  # (also covers ForwardException and BackwardException) one odd game must not stop a shard
            warnings.warn(f"{gameid}: {type(e).__name__}: {e}")
        done += 1
    print(f"legacy-parsed {ok} of {done} games")


if __name__ == "__main__":
    main()
