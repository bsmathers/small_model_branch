"""TaurosV0 as it plays on Showdown: upstream metamon's own client (ChallengeByUsername, battle backend "metamon", its observation
and action spaces, action temperature 1.0 = sampling from its policy), accepting challenges from one username.

The only change: initialize_agent gets attention_backend="sdpa" (the patch in duel/tauros_metamon.patch), because the released
config asks for CUDA-only FlashAttention; SDPA keeps its 32-turn sliding window.

    python tauros_acceptor.py --metamon-root UPSTREAM --cache CACHE --username U --opponent O --battles N --team-set b1_good_807
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--metamon-root", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--username", required=True)
ap.add_argument("--opponent", required=True)
ap.add_argument("--battles", type=int, required=True)
ap.add_argument("--team-set", default="b1_good_807")
ap.add_argument("--team-dir", default=None, help="a folder of team files (used instead of --team-set): the same files the student draws from")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--results", default=None)
ap.add_argument("--trajectories", default=None, help="save TaurosV0's own states and actions here (metamon's format)")
args = ap.parse_args()

os.environ["METAMON_CACHE_DIR"] = args.cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.abspath(args.metamon_root))

import metamon  # noqa: E402
from metamon.rl.pretrained import PretrainedModel, get_pretrained_model  # noqa: E402

_init = PretrainedModel.initialize_agent


def initialize_agent(self, *a, **kw):
    kw.setdefault("attention_backend", "sdpa")
    return _init(self, *a, **kw)


PretrainedModel.initialize_agent = initialize_agent

from metamon.rl.evaluate.__main__ import pretrained_vs_challenge  # noqa: E402

assert os.path.abspath(metamon.__file__).startswith(os.path.abspath(args.metamon_root)), metamon.__file__
# (a team file whose species line carries a gender tag, "Chansey (F)", is packed as the species "chanseyf" and rejected by the
# server; the repo's duel/teamset has those tags and nicknames removed, so --team-dir should point there)
teams = metamon.env.TeamSet(args.team_dir, "gen1ou") if args.team_dir else metamon.env.get_metamon_teams("gen1ou", args.team_set)
results = pretrained_vs_challenge(
    pretrained_model=get_pretrained_model("TaurosV0"), username=args.username, opponent_username=args.opponent, role="acceptor",
    battle_format="gen1ou", team_set=teams, total_battles=args.battles, action_temperature=args.temperature,
    save_results_to=args.results, save_trajectories_to=args.trajectories,
)
print("TAUROS RESULTS " + json.dumps(results, default=str), flush=True)
