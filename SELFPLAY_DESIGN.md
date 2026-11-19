# Self-play from scratch: design

Goal: a ~5M-parameter Gen 1 OU policy/value model with a recurrent history and direct reading of the battle log, trained
from scratch by Ataraxos-style self-play on one RTX 4090 (24 GB). Zig does everything that is not the network (engine,
observations, log tokens, legal masks, rewards) in-process; PyTorch does the network (acting and training) on the GPU.

## Decisions (2026-11-18)

- **Format**: Gen 1 OU.
- **Teams**: each game's two teams drawn 50/50 from the 807-team pool (duel/teamset) and Metamon's high-ladder set
  `hl_05_26` (Gen 1 OU; Hugging Face dataset `jakegrigsby/metamon-teams`, file `hl_05_26/gen1ou.tar.gz`).
- **Recurrence**: truncated backpropagation through time. The state runs forward through the whole game, both while
  playing and in training: training chunks are 8 consecutive decisions of one player, each starting from the state recorded
  at its first decision (the whole game before it), with the gradient cut every 8 decisions; reset only at a game's start.
  Chunk length is a setting.
- **Batch**: as many chunks per gradient step as fit in GPU memory (measured on startup with a trial step, with headroom),
  not a fixed number; the learning rate's ceiling and floor scale with that batch as sqrt(batch / 24576). Concurrent games:
  4,096 to start, limited by CPU time and host memory rather than the GPU.
- **Policy head**: a pointer head after the core ("B-late"): the 16 core tokens as in the 1.3M model; each legal move scored
  by `query(state) · key(move)`, key = MLP(move embedding, PP and flags, type, the core's output for the active Pokemon);
  each legal switch by `query(state) · key(output of that reserve Pokemon's token)`. About +0.5% compute.
- **Inputs per decision**: the recurrent state; the log events since this player's previous decision; the board from its
  side. The board keeps the current-state features (own team exactly; the opponent's public view: seen species, revealed
  moves with use counts, public HP %, status, boosts, announced effects; the field token's damage, speed, screens, party
  counts, turn bucket). Dropped, left to the log and the recurrence: the hand-made last-turn token (action kinds, the move
  used, HP change, cannot-act reasons) and the history bookkeeping (Counter's damage memory, trap-turn probabilities, effect
  order).
- **The log**: not text. The pkmn engine's protocol log (built with logging; one `FixedLog` buffer per update, as
  duel/harness/src/harness.zig does) is a binary stream: an ArgType byte (Move, Switch, Cant, Faint, Turn, Damage, Heal,
  Status, CureStatus, Boost, ClearAllBoost, Fail, Miss, HitCount, Prepare, MustRecharge, Activate, FieldActivate, Start, End,
  OHKO, Crit, SuperEffective, Resisted, Immune, Transform, Win, Tie), a Pokemon identifier (side and slot), and typed
  arguments (move, species and level, HP and max HP, status, reason, amounts). Zig turns each event into a row of up to 12
  fields, from the deciding player's side: identifiers as own/opponent and slot, HP as exact for its own Pokemon and as
  Showdown's public percentage for the opponent's. No parsing, no strings.

## 1. Training algorithm (Ataraxos S3.4 / Table S7, scaled to one GPU)

Kept as published:

| | Ataraxos (move network) | Here |
|---|---|---|
| epochs per iteration | 1 | 1 |
| λ for advantages / outcome targets | 0.5 / 0.8 | 0.5 / 0.8 |
| importance-ratio clip | 0.2 | 0.2 |
| gradient-norm clip | 0.267 | 0.267 |
| reverse KL to the behavior policy | 0.1 | 0.1 |
| reverse KL to the magnet (uniform over legal actions) | 0.05 / iteration^0.3 | 0.05 / iteration_eq^0.3 |
| value | categorical win/loss/draw, cross-entropy on λ-return outcome probabilities, weight 1 | same |
| EMA of the weights (evaluation) | 0.999 | 0.999 |
| advantage filter | \|A\| ≥ 0.75 quantile (top 25%) and ≥ 0.01, over the iteration's data | **top 50%** (\|A\| ≥ median) and ≥ 0.01 |

Changed for one GPU:

| | Ataraxos (16 H100) | Here (1 × 4090) |
|---|---|---|
| concurrent games | 1,536 per GPU (24,576) | **4,096** |
| steps per iteration | 202 simulator steps (101 per player) | **64 turns** (both players decide each turn): ~500k decisions |
| transitions per gradient step (before filtering) | 24,576 (1,536 per GPU) | **as many as fit** (chunks of 8; ~4,096-16,000 expected) |
| gradient steps per iteration | 202 | ~122 |
| learning rate | clip(0.5 / i^1.1, 5e-6, 1e-4) | see below |

**Learning rate.** Ataraxos's schedule is a function of iterations, each iteration ≈ 5M transitions with a global batch of
24,576. Two adjustments, then an adaptive controller:

1. Schedule on data, not iterations: `i_eq = transitions_seen / 5.0e6` (one of our iterations is ≈ 0.1 of theirs).
2. Batch size: the ceiling and floor are scaled by sqrt(batch / 24576), the usual Adam heuristic (at 4,096 decisions per
   step: `lr_sched = clip(0.205 / i_eq^1.1, 2e-6, 4.1e-5)`; at 16,384: about 1.7x higher).
3. Adaptive: after each iteration, the KL between the behavior policy and the updated policy on the iteration's own data is
   measured. If it exceeds 2× a target, the next iteration's lr multiplier drops by 1.5×; below 0.5× the target, it rises
   by 1.5× (never above `lr_sched`). Target KL = 0.01 per iteration (an educated guess; logged so it can be tuned).

## 2. The model (~5.1M parameters, width 256)

One step per decision of one player, recurrent through a carried state token, as in the 1.3M model, with the "last turn"
features replaced by a reading of the battle log since that player's previous decision (the Gen 3 model's idea).

**Log encoder (≈1.0M).** The pkmn engine's own protocol log (binary events: move, switch, damage, heal, status, boost,
cant, faint, crit, miss, activate, start/end, ...) is tokenized in Zig per player's view: each event is up to 12 fields
(kind, actor side/slot, target side/slot, move, species, status, reason, two numbers, flags). Numbers are Showdown's: our
exact HP, the opponent's HP as its public percentage. Each field: a 32-wide embedding; the 12 are concatenated (384) →
MLP 384 → 512 → 256 → LayerNorm (≈0.33M). Then one transformer layer (width 256, 8 heads, ff 768) over up to 32 events plus
a summary token (≈0.66M); the summary is the update token.

**Core (≈3.9M).** 16 tokens: carried state, update token, 6 own Pokémon, 6 opponent Pokémon, field/request, value token.
The Pokémon and field tokens come from the existing native features (rnative.zig, already exact against the Python
pipeline). 6 pre-LayerNorm transformer layers, width 256, 8 heads (32 wide, like the native search's), ff 768.

**Heads.** Policy: the pointer head above (Decisions).
Value: 3 logits (win/loss/draw) from the value token. The state token's output is the next carried state.

**Training the recurrence.** On-policy data with one epoch, so the carried state recorded while acting is what the current
network would compute. Training runs chunks of 8 consecutive decisions of one player, starting from the recorded state
(BPTT over 8, no burn-in; the state itself carries the whole game). The advantage filter applies to the loss terms; filtered decisions still run forward (they carry
the state). Chunk length is a setting (1 = independent transitions, cheapest; 8 = default).

## 3. Memory and batch sizes (4090, 24 GB)

Training, bf16 autocast: saved activations ≈ 1 MB per decision (core 16 tokens × 6 layers × ~14 × 256 values × 2 bytes ≈
0.7 MB; log encoder ≈ 0.3 MB), ×1.5 for fp32 softmax/norm buffers ≈ 1.5 MB. Minibatch of 4,096 decisions ≈ 6 GB;
model + Adam + EMA ≈ 0.1 GB. Leaves room for allocator slack; the trainer measures the actual peak on a dry run and halves
the chunk count per step (with gradient accumulation, same step) if needed.

Acting: 4,096 games × 2 players = 8,192 decisions per forward, inference only: ~0.5 GB. Acting and training alternate
(synchronous), so their memory does not add up.

Host (pinned) memory per iteration: ~4 KB per decision (feature inputs in fp16, 32 × 12 event fields int16, behavior
log-probabilities, value, masks; the carried state only at chunk starts) × ~500k ≈ 2 GB.

Estimated speed (to be measured in stage 1): acting ≈ 64 turns × ~25 ms (GPU forward of 8,192 decisions with CUDA graphs;
Zig engine and observations on all cores in parallel ≈ 3 ms) ≈ 1.6 s; training ≈ 122 steps × ~40 ms ≈ 5 s. About 7 s per
iteration ≈ 70k decisions/s ≈ 250M decisions per hour (Ataraxos: about 1.9B steps per GPU-day; this would be ~6B per day).

## 4. Zig ↔ PyTorch: no adapters

- `libpkmnrl.so` (Zig, C ABI), loaded into the Python process with ctypes: `create(n_games, seed, team_pool)`,
  `step(actions*)`, all writing into caller-owned, preallocated, pinned NumPy buffers (observations, event tokens, legal
  masks, rewards, done flags, the acting player of each row). No processes, no JSON, no per-step allocation.
- Inside: the pkmn engine (78dc891, built with protocol logging), per-game PRNG, team sampling from the 807-team pool (both
  sides random), the existing native feature code (rnative.zig) for state tokens, the protocol-log tokenizer, the Showdown HP
  rules, parallelized over games with a thread pool.
- Python: `torch.from_numpy` on the pinned buffers, one non-blocking copy to the GPU, the policy forward captured as a
  CUDA graph (fixed batch shape), sampling on the GPU, actions copied back. One ctypes call and one graph replay per turn.

## 5. Build stages (each with its own checks)

1. **Environment library** — done (selfplay/zig, libpkrl): engine + observations + log events + masks. Checks
   (`pkrl-bench check | leak | leak-control | speed | trace`): determinism 0 of 300 steps differed; the opponent's hidden state
   changed 0 of 73,920 observations (a public change, the control: 72,874); 0 decode errors; ~3M decisions/s on 8 threads
   (random play, 2.5M/s through Python).
2. **Model** — done (selfplay/py/model.py, 5,460,003 parameters). Not yet: a CUDA-graph acting step (after the GPU profile).
3. **Self-play loop** — done (selfplay/py/train_selfplay.py): collection with the recorded states, λ-returns and advantages
   (test_targets.py against hand-worked values), filtering, the loss, the schedules, EMA, checkpoints, resume. Run on CPU only
   so far.
4. **Evaluation**: games against TaurosV0 every N iterations (the new model as a policy server for the duel harness, which
   then also has to send the protocol events), and against the 1.3M model. Not yet.
5. **Launcher** — done (selfplay/launch.py): setup, a GPU smoke test that measures the settings, then training after approval.

## Running it (GPU machine)

    python selfplay/launch.py --work /root/work/selfplay      # setup, smoke test, the proposed plan; asks before training
    python selfplay/launch.py --work /root/work/selfplay --yes

Setup: Zig 0.16, the pkmn engine at 78dc891 (selfplay/zig/engine), libpkrl, the hl_05_26 teams (Hugging Face
jakegrigsby/metamon-teams), the feature tables. The smoke test measures the environment's speed, the number of games with
the best acting throughput (1k–16k), the most 8-decision chunks per gradient step within 85% of GPU memory, and the turns per
iteration whose buffers fit next to the training (64 by default), then plays 2 real iterations; the plan goes to
WORK/plan.json and the training to WORK/run (resumed when started again). Any setting can be fixed instead (--games, --turns,
--batch-chunks).

## Open choices

- Format: Gen 1 OU (the pkmn engine and every tool here are Gen 1).

