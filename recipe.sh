#!/usr/bin/env bash
# One-stage training recipe from scratch at three model sizes, trained at the same time on one machine: distillation of
# Exeggcute, the win head and a weak imitation of the strong bots, all at once. Each run is a launch.sh run of its own (crash
# restarts, backups, a 100-game tournament against TaurosV0 every 5000 steps). Safe to run again: everything resumes.
#
#   ./recipe.sh                     # 200k, 1.3M and 3.3M at once
#   ./recipe.sh 200k 1m             # some of them
#   SEQUENTIAL=1 ./recipe.sh        # one after the other
#   ./recipe.sh 3m --steps 50000    # arguments after the sizes go to every launch.sh (and win over the recipe's)
#
# The recipe (one joint stage, --value-steps 0), per decision:
#   1.0  x KL(Exeggcute's blend || policy)                  every decision (human and strong-bot games, 40/60 by decisions)
#   0.25 x -log policy(the bot's played move)               only the strong bots' own decisions (Tauros-v1/v2, SmallG1Online-v2/v3):
#                                                           on those the policy settles near 0.8 x Exeggcute + 0.2 x the bot
#   1.0  x BCE(win head, game result)                       every decision; only a quarter of its gradient reaches the shared
#                                                           network (--value-trunk-scale 0.25): the win head learns fully, but the
#                                                           noisy result cannot pull the features away from the policy (training
#                                                           the whole network on the win loss cost 3-4 points of play before)
# Up to 100k steps of 256 windows (32 decisions of burn-in, 64 trained), AdamW, warmup 1000, cosine to 5%; validation every
# 2000 steps on 1 game in 20; early stopping and weights_best.pt follow the weighted validation loss (patience 10).
#
#   size  width layers heads feed-forward  parameters  lr     run folder (WORK/...)   logs (WORK/...)
#   200k     96      2     3          192     205,066  1e-3   run_recipe_200k         logs_recipe_200k
#   1m      192      4     6          384   1,312,234  3e-4   run_recipe_1m           logs_recipe_1m
#   3m      256      6     8          512   3,342,762  2e-4   run_recipe_3m           logs_recipe_3m
# (heads are 32 wide in every size; inputs, embeddings and heads are the same; smaller models take a larger learning rate)
#
# Running at once: the data checks run once first (launch.sh --check-only, and the tournament setup), then the three launchers
# share the built games, each with its own lock, run folder and logs, a third of the CPUs for data loading and fewer tournament
# processes. Training this model is bound by per-step overhead, not GPU arithmetic, so the runs mostly fill the card's idle time.
# GPU memory: the batch goes through the model in ACCUM_<SIZE> micro-batches (gradient accumulation, the same step): defaults 1
# (200k), 2 (1m), 4 (3m) when running at once, 1 when SEQUENTIAL; raise one if it runs out of memory (it resumes).
# Ctrl-C (or kill) stops all of them, each saving its checkpoint first.
#
# The models: WORK/run_recipe_<size>/weights_best.pt. Loaders (tournaments, the policy and search servers) read the size from the
# weights. The native Zig search (rmodel.zig) is built for width 192 only.
set -euo pipefail
cd "$(dirname "$0")"
WORK=${WORK:-$HOME/work}
export WORK
PYTHON=${PYTHON:-python3}
sizes=()
while [ $# -gt 0 ] && { [ "$1" = 200k ] || [ "$1" = 1m ] || [ "$1" = 3m ]; }; do sizes+=("$1"); shift; done
[ ${#sizes[@]} -gt 0 ] || sizes=(200k 1m 3m)
sequential=${SEQUENTIAL:-0}
[ ${#sizes[@]} -gt 1 ] || sequential=1

recipe=(--value-steps 0 --alpha 1 --bc-weight 0.25 --value-weight 1 --value-trunk-scale 0.25 --monitor loss --steps 100000 --val-every 2000 --patience 10)
size_args() {
  case "$1" in
    200k) echo "--width 96 --layers 2 --lr 1e-3 --accum ${ACCUM_200K:-1}" ;;
    1m)   echo "--width 192 --layers 4 --lr 3e-4 --accum ${ACCUM_1M:-$([ "$sequential" = 1 ] && echo 1 || echo 2)}" ;;
    3m)   echo "--width 256 --layers 6 --lr 2e-4 --accum ${ACCUM_3M:-$([ "$sequential" = 1 ] && echo 1 || echo 4)}" ;;
  esac
}
log() { echo "$(date +%H:%M:%S) recipe.sh: $*"; }

if [ "$sequential" = 1 ]; then
  for size in "${sizes[@]}"; do
    # shellcheck disable=SC2046
    ./launch.sh "${recipe[@]}" --policy-run "run_recipe_$size" $(size_args "$size") "$@"
  done
  exit 0
fi

# 1. the data (download and checks when needed; nothing to do when the built games are there) and the tournament setup, once
log "checking the data (launch.sh --check-only)"
./launch.sh --check-only --tournament-every 0 "$@"
if [ ! -f "$WORK/duel/.ready" ]; then
  log "setting up the tournament harness (log $WORK/logs/tournament_setup_recipe.txt)"
  mkdir -p "$WORK/logs"
  "$PYTHON" -m metamon.recurrent.tournament setup --work "$WORK" --cache "$WORK/metamon-cache" > "$WORK/logs/tournament_setup_recipe.txt" 2>&1 \
    || log "the tournament setup failed (see that log): training goes on without tournaments"
fi

# 2. the runs, at once
cpus=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
n=${#sizes[@]}
workers=$(( cpus / n > 2 ? cpus / n : 2 ))
tprocs=$(( cpus / (2 * n) > 1 ? cpus / (2 * n) : 1 ))
pids=()
stop() {
  log "stopping every run (each saves its checkpoint)"
  for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  wait
  exit 130
}
trap stop INT TERM
for size in "${sizes[@]}"; do
  log "starting $size ($(size_args "$size"); $workers data workers, $tprocs tournament processes): progress in $WORK/run_recipe_$size/log.txt, launcher log $WORK/logs_recipe_$size/launch.txt"
  # shellcheck disable=SC2046
  LAUNCH_LOCK="$WORK/.launch_recipe_$size.lock" ./launch.sh --train-only "${recipe[@]}" --policy-run "run_recipe_$size" $(size_args "$size") \
    --logs-name "logs_recipe_$size" --train-workers "$workers" --tournament-procs "$tprocs" --reuse-tournament-setup "$@" \
    > "$WORK/recipe_$size.out" 2>&1 &
  pids+=($!)
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then log "${sizes[$i]}: finished"; else code=$?; status=$code; log "${sizes[$i]}: stopped with exit code $code (see $WORK/recipe_${sizes[$i]}.out)"; fi
done
exit "$status"
