"""The environment library from Python: random legal play in many games, the buffers' consistency, and the speed with the
Python side included.  python test_pkrl.py [GAMES] [STEPS] [THREADS]"""
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))
from pkrl import Env, random_actions  # noqa: E402

games, steps, threads = (int(x) for x in (sys.argv[1:] + ["4096", "200", "8"][len(sys.argv) - 1:])[:3])
root = "/Users/macuser/mcts_better_eval"
env = Env(games, threads, seed=1, tables=f"{root}/showdown_trial/duelwork/duel/recurrent_native_stage1",
          teams=(f"{root}/metamon/duel/teamset", f"{root}/metamon/selfplay/data/teams/hl_05_26/gen1ou"), share_a=0.5)
print("teams loaded per pool:", env.pool_sizes())
obs = env.reset()
g = torch.Generator().manual_seed(0)
t_lib = 0.0
rows = 0
first = obs["first"].sum().item()
assert first == obs["valid"].sum().item(), "every first row is a new game's"
t0 = time.time()
for _ in range(steps):
    valid = obs["valid"].bool()
    assert (obs["legal"][valid].sum(-1) >= 1).all(), "a valid row without a legal slot"
    assert (obs["n_events"][valid] <= env.MAX_EVENTS).all()
    rows += int(valid.sum())
    actions = random_actions(obs, g)
    t = time.time()
    obs = env.step(actions)
    t_lib += time.time() - t
elapsed = time.time() - t0
s = env.stats()
print(f"{games} games x {steps} steps on {threads} threads: {rows:,} decision rows in {elapsed:.2f} s ({t_lib:.2f} s in the library): "
      f"{rows / elapsed:,.0f} decisions/s overall, {rows / t_lib:,.0f} in the library")
print({k: v for k, v in s.items()})
print(f"P1 score {(s['p1_wins'] + 0.5 * s['ties']) / max(s['games'], 1):.3f} over {s['games']} games")
