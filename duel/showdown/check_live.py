"""Checks the student's live inputs on Showdown against the training pipeline: each finished battle's log (recorded by
student_player.py) is parsed whole, as training parses a game (with the same own-HP conversion and the team brought), and the
inputs the student computed live at each decision are compared with the full parse's at the same index.

    python check_live.py BATTLES_DIR

Live inputs can only lack what the full parse fills in from later in the game (the opponent's later reveals); any other
difference is an adapter bug.
"""

import collections
import copy
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_player as sp  # noqa: E402  (also lifts the parser's finished-game check)
from metamon import interface  # noqa: E402
from metamon.backend.replay_parser import backward, forward  # noqa: E402
from metamon.backend.team_prediction.team import TeamSet  # noqa: E402
from metamon.recurrent.features import encode_observer  # noqa: E402
from metamon.recurrent.records import last_turn_from_extras  # noqa: E402
from metamon.recurrent.tables import Tables  # noqa: E402

tables = Tables.from_dex(1)
parser = sp.RecurrentReplayParser(replay_output_dir=None, verbose=False)
diffs, total, misaligned, games = collections.Counter(), 0, 0, 0
examples = []
for log in sorted(Path(sys.argv[1]).glob("*.log")):
    rec = json.loads(log.with_suffix(".json").read_text())
    if not rec.get("decisions"):
        continue
    raw = log.read_text().splitlines()
    side = "p1" if any(l.startswith("|player|p1|StudentEval") for l in raw) else "p2"
    lines = [sp.spectator_line(l, side) for l in raw]
    players = {l.split("|")[2]: l.split("|")[3] for l in lines if l.startswith("|player|") and len(l.split("|")) > 3 and l.split("|")[3]}
    tag = log.stem
    gameid = tag.replace("_", "-")
    data = {"players": [players["p1"], players["p2"]], "uploadtime": int(sp.PLAYED.timestamp()), "formatid": "gen1ou", "log": "\n".join(lines)}
    random.seed(gameid)
    np.random.seed(zlib.crc32(gameid.encode()) % (2**32))
    replay = forward.forward_fill(forward.ParsedReplay(gameid=gameid, format="gen1ou", time_played=sp.PLAYED), parser.clean_log(data), verbose=False)
    own = TeamSet.from_showdown_file(rec["team"], "gen1ou")
    a, b = backward.backward_fill(replay, team_predictor=sp.OwnTeam(parser.team_predictor, side == "p1", own))
    states, _, extras = parser.povreplay_to_seq_extras(a if side == "p1" else b)
    states = [s.to_dict() for s in states]
    games += 1
    for d in rec["decisions"]:
        t = d["t"]
        if t >= len(states) or extras[t]["turn"] != d["turn"]:
            misaligned += 1
            continue
        s, e = states[t], extras[t]
        state = interface.UniversalState.from_dict(copy.deepcopy(s))
        party = [interface.UniversalPokemon.from_dict(copy.deepcopy(m)) for m in e["player_team"] if m is not None]
        revealed = [interface.UniversalPokemon.from_dict(copy.deepcopy(m)) for m in e["opponent_team"] if m is not None]
        f = encode_observer(state, party, revealed, last_turn_from_extras(states[t - 1] if t else None, s, e), t + 1, tables, extras=e)
        full = {**{f"ids_{k}": v for k, v in f.ids.items()}, **{f"floats_{k}": v for k, v in f.floats.items()}}
        total += 1
        for k, v in d["features"].items():
            if not np.allclose(np.asarray(v, dtype=np.float32), np.asarray(full[k], dtype=np.float32), atol=1e-4):
                diffs[k] += 1
                if len(examples) < 6 and not k.startswith(("ids_opp", "floats_opp")):
                    examples.append(f"{tag} t={t} turn {d['turn']} forced {d['forced']}: {k}")
print(f"{games} games, {total} live decisions compared, {misaligned} misaligned")
for k, n in diffs.most_common():
    print(f"  {k}: differs at {n} ({n / max(total, 1):.1%})")
for x in examples:
    print("  e.g.", x)
