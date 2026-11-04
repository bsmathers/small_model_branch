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
node tools/oracle/showdown_oracle.js SHOWDOWN_DIR TEAMS_DIR ORACLE_DIR 300 [FIRST_SEED]   # LOCK_MODE=1 / COUNTER_MODE=1 / SUB_MODE=1 stress locks / Counter / Substitute
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

## Trapping and Substitute

The partial-trapping rules follow the Smogon RBY partial-trapping article and the engine. The duration is hidden (2-5 uses,
weights 3:3:1:1), so the parser exports the probability that the victim is still trapped (`player_trapped`,
`opponent_trapped`: 0 when free, 1.0 after the first use, 0.625, 0.4, 0.5 after the 2nd to 4th, 0 after the 5th). A trap that
was not renewed in the turn is cleared, as is one on a Pokemon that makes a Substitute (`-end ... [partiallytrapped]`). On
about 600 fuzz games every 0 and every 1 matched the engine and the in-between values were calibrated (0.62 observed for 0.625,
0.39 for 0.4).

Substitute (Gen 1 mod): `player_effects` / `opponent_effects` list every volatile (substitute, confusion, leechseed,
focusenergy ...) and match the engine on all four in the Substitute fuzz set (`SUB_MODE=1`). The quirks handled:
creating a Substitute is not a hit (its HP cost leaves `lastDamage` alone); damage absorbed by a Substitute, or that breaks
it, sets `lastDamage` to an uncapped amount the log never shows, so Counter then uses the damage calculator's average roll
(`features.counter_fraction(estimate=...)`; in the checker those cases are counted but not graded); a confused Pokemon's
self-hit can land on the foe's Substitute and then no damage line names it (the parser reads a missing line as a self-hit and
ends Bide / charge / Thrash / the trap); a cure of bad poison is printed as `psn` (Haze); a last usable move that is disabled
means Struggle.

## Two parsers, one trajectory

Exeggcute was trained on the ORIGINAL (upstream) parser's output, bugs included (PP of 64 where Gen 1 has 61, boosts and
Reflect that outlive a fainted Pokemon, shifted action labels). So each trajectory stores both:

| key | parser | used by |
|---|---|---|
| `legacy_states[t]`, `legacy_actions[t]` | original | Exeggcute only (blend and judge labels, its previous-action input) |
| `states[t]`, `actions[t]`, `extras[t]` | corrected | the distilled model: its features, played action, decision kind |

`tools/legacy_parse.py` runs the original parser: upstream's `forward.py`, `backward.py` and `replay_state.py` (the only parser
files we changed) are kept verbatim in `legacy_parser/` and an import hook points the three module names at them for that
process, so everything else is shared code. On 276 trajectories it reproduces the output of an untouched upstream checkout
exactly (275 byte-identical; the 276th is a game whose two players share a username, which the legacy run skips).
`merge_legacy.py` folds it into the corrected files and moves any game whose two parsers disagree on length (or without a
legacy twin) to `<traj>_rejected/`. `label.py` reads only the legacy pair; labels from it are bit-identical to labelling the
upstream parser's own files. `build.py` maps the labels, which are indexed by the legacy state's actions, onto the corrected
state's slots by identity (a move by name, a reserve by species: `features.label_slot_map`); label mass on an action the
corrected state lacks counts as lost and the decision is excluded from the policy loss. The launcher runs `parse`, `legacy`,
`merge`, `label`, `build` in that order.

## Labelling speed

Profiled per phase (`label_batch`, 1,034 decisions): the five Exeggcute members' encoders are the real work; the router around
them used to take 48%. In decisions per second on one CPU process (24 games, steady state, noisy machine): 47 originally, 72 after the first
three changes, about 123 now. Everything below is exact up to float rounding: against the unmodified path, on 4,596 decisions, the blend
differs by at most 2e-6, the judge distribution by at most 1.3e-5, and the router picks the same action every time.

- the router's per-game reads (legal actions, cycle keys, previous reward, each member's probability table) use host tensors, so a
  GPU is never waited on game by game;
- the critic is evaluated once per member per step for all games and all actions (the critic is discrete, so one pass gives every
  action's value), instead of once per game and shortlist;
- `fast_tstep.py`: the members encode the seven Pokemon of a decision in one perceiver call with block-diagonal masks, which is a 7x
  larger attention than the same arithmetic done per Pokemon; the labeler encodes each Pokemon on its own, and only those whose tokens or
  numbers changed since the previous step of that game (about 70% are unchanged);
- the SigmaReparam linears of the trajectory transformers rebuild `gamma / sigma * W` (and, under autocast, re-cast it) on every call; the
  labeler computes it once (sigma under the labeler's own autocast, so it is bit-identical to the original layer: `test_fast_tstep.py`);
- the attention reads only the last 33 cached positions (the window) instead of copying the whole 128-slot cache every layer;
- `--batch` defaults to 256 games per process (about 11 MiB of caches per game).

`--encoder auto` (the default) times the fast and the members' own encoder for a few steps on the first games and keeps the faster one
(ties go to the fast one), so a machine where the original is quicker keeps it. The label stage prints decisions per second.

## Parser version

`metamon.recurrent.parse.PARSER_VERSION` is stamped into each work folder by the launcher; a folder made by another version is
refused (its trajectories, labels and built data would be stale).

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
