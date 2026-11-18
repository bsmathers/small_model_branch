#!/usr/bin/env bash
# Two runs of fine-tune stage A only (finetune.sh without the value stages), from the distilled weights
# (default WORK/run_v3/weights_best.pt, the first argument to change it):
#
#   1  full history   the run that gave weights_10000 (52.8% against TaurosV0): imitation of the strong bots' played moves (1)
#                     + KL to Exeggcute (0.25) + anchor to the starting policy (0.25), the bots' games only, 10000 steps with
#                     early stopping, but with a burn-in of 96 instead of 32: training windows are 160 decisions long, so a game
#                     of up to 160 decisions is trained with its whole history, as the model plays it (with windows of 96, 21%
#                     of the bots' decisions were trained on a memory restarted from zero; now 6%)    -> WORK/run_ft_fullhist_actor
#   2  second seed    the same with --seed 1: an independent second run, for an ensemble with 1
#                                                                                                     -> WORK/run_ft_fullhist_seed1_actor
#
# Each prints the usual 100-game tournaments every 5000 steps. Training costs about 1.7x as much per step as with burn-in 32.
# If the GPU runs out of memory with the longer windows, run again with --batch auto (it resumes).
#
#   ./imitation.sh
#   ./imitation.sh /path/to/start.pt --ft-actor-steps 15000
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK:-$HOME/work}"
weights="$WORK/run_v3/weights_best.pt"
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then weights="$1"; shift; fi
[ -f "$weights" ] || { echo "imitation.sh: no $weights (give the starting weights as the first argument)"; exit 2; }
common=(--train-only --finetune-from "$weights" --ft-actor-only --ft-bc-weight 1 --ft-kl-weight 0.25 --ft-actor-anchor 0.25
        --ft-actor-mix bots=1,human=0,otherbots=0 --ft-actor-steps 10000 --val-every 1000 --patience 3 --burn-in 96)
./launch.sh "${common[@]}" --seed 0 --finetune-run run_ft_fullhist "$@"
./launch.sh "${common[@]}" --seed 1 --finetune-run run_ft_fullhist_seed1 "$@"
