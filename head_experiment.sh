#!/usr/bin/env bash
# Experiment 1 alone: a new 3-layer actor head (192 -> 64 -> 64 -> 9, added to the linear actor head, output zero-initialised so it
# starts as START's policy) trained on the frozen model; only the actor (the new head and the linear head, 19k parameters) trains.
#
#   phase 1  1500 steps imitating Exeggcute: KL to its targets only, on the usual distillation data (human and strong games),
#            warmup 150 steps                                                                    -> WORK/exp_head_exeggcute
#   phase 2  from phase 1: imitation of the strong bots' played moves (1) + a little KL to Exeggcute (0.1), on the bots' games
#            only; up to 10000 steps with early stopping                                         -> WORK/exp_head
#   then     a 500-game tournament of WORK/exp_head/weights_final.pt against TaurosV0 (FINAL TOURNAMENT line,
#            WORK/tournaments/results.txt)
#
#   ./head_experiment.sh /root/work/run_ft_bots_b1_actor/weights_10000.pt --val-every 1000 --patience 3
#
# Options after START (launch.py): --exp-pre-steps 1500 --exp-pre-warmup 150 --exp-pre-mix none --exp-steps 10000 --exp-lr 3e-4
# --exp-kl-weight 0.1 --exp-bc-weight 1 --exp-mix bots=1,human=0,otherbots=0 --final-games 500. Safe to run again: it resumes.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -gt 0 ] && [ "${1#--}" = "$1" ] || { echo "head_experiment.sh: give the starting weights first (e.g. /root/work/run_ft_bots_b1_actor/weights_10000.pt)"; exit 2; }
start="$1"; shift
[ -f "$start" ] || { echo "head_experiment.sh: no $start"; exit 2; }
./launch.sh --train-only --finetune-from "$start" --experiment head "$@"
grep "FINAL TOURNAMENT head" "${WORK:-$HOME/work}/tournaments/results.txt" | tail -n 1 || true
