# Recurrent policy/value model for Gen 1 OU

A 1.31M-parameter recurrent transformer distilled from the Exeggcute ensemble (its proposer blend) with
a win-probability head. Spec: `pkmn-search/docs/model_spec.md` (kept in the search repo).

## Launch

```
git clone git@github.com:bsmathers/small_model_branch.git && cd small_model_branch
./launch.sh --smoke     # a few minutes; prints stage timings and a projection for the full run
./launch.sh             # the full run (resumes if stopped: run the same command again)
```

`launch.sh` installs the requirements (`pip install -e .` plus a CUDA build of torch if none is present),
then `metamon.recurrent.launch` downloads the data from `bsmathers/raw_replays` and Exeggcute's checkpoints and
usage statistics, parses both points of view of every game, labels them with Exeggcute, builds the
training games and trains. Options: `--games` per source (default 30000), `--mix human=0.4 --mix strong=0.6` (the default),
`--steps` (20000), `--batch` (auto: the largest that fits in GPU memory), `--bc-weight` (1.0) and
`--bc-players` (see below), `--ckpt-every` (1: `latest.pt` is rewritten every iteration), `--precision auto` (bf16 on Ampere
or newer, fp16 before that, fp32 on CPU), `--work DIR`. Output: `WORK/run/weights_<step>.pt`; progress in
`WORK/run/log.txt` and `WORK/logs/`.

## What counts as a decision

Metamon always lists every move and every alive switch as legal, so the builder works out the real options
(`build.py`, `decision_kind`):

- **asleep, frozen, or losing the turn to a trapping move:** in Gen 1 these lock the Fight button (every move
  fails alike) but never prevent switching, so the real choice is to fight or to switch out. The target's move
  probability is spread evenly over the legal moves, and a recorded "no action" means "fight". A trapped
  turn counts only when the player was already trapped and the next state confirms the lost turn: the
  parser's flag alone never learns that a trap ran out, and a lost turn alone also happens when a fresh Wrap
  lands before the player moves. The user of a trapping move is also locked into it while the trap lasts, but the logs do not show
  when it ran out, so those decisions are left as normal ones;
- **recharge turn:** the only option is the no-op: no policy or imitation loss, legal set = the no-op;
- **no recorded action for another reason** (full paralysis, flinch, fainted before acting): the player chose
  a move that then failed, so Exeggcute's blend applies, and the imitation target is "some move" (the policy's
  total probability on moves);
- **timeouts** (the game ended before a choice): dropped.

## Training objective

Per decision: `alpha * KL(Exeggcute blend || policy)` (all games), `bc_weight * KL(played action || policy)`
(only games played by `Tauros-v1`, `Tauros-v2`, `SmallG1Online-v2` and `SmallG1Online-v3`, on their own
decisions), and a binary cross-entropy on the game result for the win head. Sources are mixed 40% human,
60% strong-bot by decisions.

Resuming: `latest.pt` holds the model, optimizer and step and is written every iteration by a background
thread (atomic replace; at most one write behind). Run the same command again to continue.

## Choosing a machine

- The model is tiny; a training step is 96 sequential small steps, limited by kernel-launch latency rather than
  compute. A big accelerator (A100, H100) is wasted. Any Ampere or newer card works (RTX 30xx/40xx, A-series,
  L4): bf16 needs compute capability 8.0 or higher. 12 GB of VRAM is enough to start; 24 GB allows a larger
  batch and more labelling processes.
- Parsing and labelling are CPU-bound (Python), so take a host with 16 or more vCPUs, 64 GB RAM and a fast
  disk. About 50 GB of disk covers the default sample (about 80 KB per game for trajectories, labels and built
  data, 2.5 GB of checkpoints, 4 GB of raw logs).
- Run `--smoke` first on the rented machine: it measures every stage and prints the projected hours.
