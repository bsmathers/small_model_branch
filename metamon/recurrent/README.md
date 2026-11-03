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

## Checking the parser against Pokemon Showdown

`tools/oracle/showdown_oracle.js` plays seeded random Gen 1 OU battles in the real simulator and records, at every decision,
the exact request Showdown sent, the hidden engine state, and the answer of Counter's own `damageCallback`;
`check_oracle.py` compares our parsed trajectories with it decision by decision (state, PP, switches, the moves the engine
offers, the lock shape of the request, Counter):

```
node tools/oracle/showdown_oracle.js SHOWDOWN_DIR TEAMS_DIR ORACLE_DIR 300 [FIRST_SEED]   # LOCK_MODE=1 / COUNTER_MODE=1 stress locks / Counter
python -m metamon.recurrent.parse ORACLE_LOGS TRAJ_DIR                                     # the *.log files in a folder of their own
python -m metamon.recurrent.check_oracle ORACLE_DIR TRAJ_DIR
```

On 1,200 battles (about 202,000 engine decisions, ordinary play plus sets that stress locks and Counter) the parser matches the engine exactly on HP, status, stat boosts, screens, PP,
the alive switches, whether each listed move is available (PP left, not disabled), Struggle, replacement states, and the
shape of the request (one forced move, a single Fight entry with switching allowed, or a normal list). Known gaps: a
Pokemon's moves that never appear in the log and teammates that never appear are unknowable (the parser fills them in with
guesses and the checker reports them separately); Mimic, which replaces a move slot (about 0.2% of PP values in a set where
15% of the Pokemon carry it); and the Counter cases below.

## Counter

Counter is bugged in Gen 1 and is handled as the engine does (`features.counter_fraction`, fed by the parser's
`last_damage`, `last_move` and `counter_selected`). It deals twice the battle's `lastDamage` (the damage of the last move
executed by either side), ignores type immunities, and works only if the target's last move *used* is Normal or Fighting
with power (Counter itself counts) and its last move *selected* is not Counter; the two differ after a switch or a turn in
which the target could not act. `lastDamage` is reset by every move except a short list (Recover, Reflect, Toxic... in
`SKIP_LASTDAMAGE`) and by a miss, and is set by move damage, by half the damage of a draining move, and by a confusion
self-hit, not by poison, burn, Leech Seed or recoil. Against the engine's own answer on about 82,000 evaluations (a set where 70% of the Pokemon carry Counter): right about
whether Counter works in 98.9%, and it catches 99.2% of the real successes; the damage is within 3% of the target's max HP in
99.6%. The misses are information the log does not contain: a paralyzed foe that had selected Counter, and an overkill (the
engine counts the uncapped damage, the log shows only the HP that was left). Super Fang (half the target's current HP) and
Psywave (up to 149) match the engine's maximum; Bide's damage (twice what it took while storing) is not modeled and gives 0.

## What counts as a decision

Metamon always lists every move and every alive switch as legal, so the builder works out the real options
(`build.py`, `decision_kind`):

The kind of a decision is the shape of the request Showdown would send, followed from the public log (`forward._track_lock`,
`records.lock_shape`):

- **a full lock, one option and no switching** (no policy or imitation loss): a pending recharge after Hyper Beam, the second
  turn of a charge move (Fly, Dig, Solar Beam, Razor Wind, Skull Bash, Sky Attack), Thrash and Petal Dance, and Rage;
- **a Fight entry with switching allowed:** asleep or frozen, using a trapping move (Wrap, Bind, Clamp, Fire Spin) or caught in one
  right after its first turn (later the request lists every move again, which is what the player sees), and Bide. All the
  moves are the same, so the target's move probability is spread evenly over the legal moves, and a recorded "no action" means
  "fight". Struggle (every move out of PP) is the same shape;
- the engine's own rules end a lock: full paralysis or a confusion self-hit end Bide, a charge, Thrash and a trap (not Rage or a
  recharge); a flinch, a sleep move that lands, or a trapping move (even a miss) cancels a pending recharge; Thrash and Petal Dance
  lock even into an immune target or on a miss, while Rage and a trap need to land;
- moves with no PP left or taken away by Disable are not available (`records.decision_facts`).

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
