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
