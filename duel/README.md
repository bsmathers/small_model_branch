# Duel harness: the recurrent model against TaurosV0

Used by `metamon/recurrent/tournament.py` (and through it by `launch.sh`, every 5000 training steps). Copies, so that a clone
of this repo can play games on a rented machine:

- `harness/`: the Zig duel harness (pkmn-search-wide), built against [pkmn/engine](https://github.com/pkmn/engine) at
  `78dc891`. Changes from the original: server paths come from the environment (`DUEL_PYTHON`, `DUEL_TOOLS`, `TAUROS_METAMON`,
  `TAUROS_CACHE`, `TAUROS_DEVICE`; see `src/policy.zig`). Outside macOS it links no BLAS, and the value model (not used by
  `duel`) has a plain matrix product.
- `teamset/`: the 807 Gen 1 OU teams the games draw from (two per pair of games, sides swapped).
- `tools/recurrent_policy_server.py`: the student. Snapshot to this model's features, argmax of the actor head over the
  legal options. Same shim as the local shadow trial: Counter's damage memory and the opponent's volatiles and screens are
  not in the snapshot.
- `tools/metamon_tauros_server.py`, `tauros_inference.py`, `metamon_connector.py`, `snapshot_effects.py`: TaurosV0
  (checkpoint 62, argmax), from bot2's `selfplay/tools`.
- `tauros_metamon.patch`: bot2's changes to upstream metamon `0a00a75`. With them TaurosV0 runs with PyTorch SDPA (its
  32-turn sliding window kept) instead of CUDA-only FlashAttention, and loads its checkpoint offline. Applied to a fresh
  upstream checkout, the `metamon` package is identical to bot2's.

```
python -m metamon.recurrent.tournament setup --work WORK          # once: Zig 0.16.0, engine, patched metamon, checkpoint, build, 2 test games
python -m metamon.recurrent.tournament play WEIGHTS --work WORK --games 100 --procs 4
```

## Search with the recurrent model

`pkmn-search tournament` (the search, `harness/src/search.zig`, as P1 against TaurosV0 or Exeggcute) can use the recurrent
model instead of the windowed value model: `SEARCH_MODEL=recurrent`. Its values and both sides' actor priors then come from
`tools/recurrent_search_server.py`, which keeps one node per player view (the policy server's whole game memory and recurrent
state) and reads every node's snapshot with the policy server's own code (`prepare` / `legal_slots` / `finish`); a chain of
nodes along a game equals what the policy server computes in play (checked on 262 decisions: largest probability
difference 2e-6). Weights: `RECURRENT_SEARCH_WEIGHTS`, else `RECURRENT_WEIGHTS`. Paths on the server: `harness/src/remote.zig`
(client) and `Path.turn` in `path.zig`; snapshots are built by `snapshot.zig`, the same code as the duel's.

The opponent's history: at each root, P2's chain (its actor priors) used to start at the root with no history. Now each world
replays the real game from P2's side, every past decision point rebuilt in that world (`universe.pastWorld`: the world's
opponent team, the public HP, status and PP of that time; `search.Config.history`). The recurrent model replays the whole
game, the windowed one its window. `HISTORY_CHECK=1` (recurrent) compares, at every decision, the chain rebuilt in the
true world with the real opponent's own chain: they agree exactly (342 decisions).

The search's other models are not in this repo: run it from a folder with `models/belief` and `models/completion` (from
pkmn-search), e.g. `SEARCH_MODEL=recurrent ./zig-out/bin/pkmn-search tournament 1 10 100 priors visits tauros expert`
(seed, games, iterations, priors, root rule: visits | adaptive | widexact | klnash ..., opponent, team pool).

### Native recurrent model (`SEARCH_MODEL=native`)

The same model and features computed inside the harness, without Python: `harness/src/rmodel.zig` (the forward pass, one
position or a batch) and `harness/src/rnative.zig` (a line-by-line port of the policy server's per-snapshot code:
`prepare` / `finish` / `legal_slots`, the connector's state, the effect memory and `encode_observer`, reading the battle
directly). Export the weights and tables once per weights file:

```
python duel/tools/export_recurrent.py WEIGHTS DIR          # DIR/model.bin, model.json, tables/
SEARCH_MODEL=native RECURRENT_NATIVE_DIR=DIR ./zig-out/bin/pkmn-search tournament 1 10 60 priors klnash tauros expert
```

Checks: `python duel/tools/check_rmodel.py WEIGHTS DIR BINARY` (the forward pass against PyTorch: largest difference ~5e-6);
`NATIVE_CHECK=1` also runs the Python search server and compares every node's win logit and option logits, and on a
difference its features field by field (0 mismatches on 186,706 nodes: a `visits` game and 14 `klnash` decisions).

Threads: `SEARCH_THREADS=N` runs the `klnash` root's worlds on N threads (each world on one thread with its own memory and
its seed drawn as in one thread: the same result for any N; native model only, not with `NATIVE_CHECK`). The opponents'
history replays of the worlds advance together, one batch per past turn. `SEARCH_NO_BATCH=1` turns every batching off
(one position at a time), for checking: the same root values.

Speed (`klnash`, this Mac, quiet): about 1 searched decision per second on one thread (56-60 per minute), 107-108 per minute
with `SEARCH_THREADS=2` (1.85x); about 3.6 s per decision with the batched Python server and 8 s before. Accelerate runs on one thread (the harness sets `VECLIB_MAXIMUM_THREADS=1`: more threads cost more CPU here). On Linux
build with `-Dopenblas=true` (apt-get install libopenblas-dev); without a BLAS the portable kernel is about 17x slower.
