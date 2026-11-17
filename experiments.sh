#!/usr/bin/env bash
# Three fine-tuning experiments from one actor checkpoint, in order; each ends with a 500-game tournament against TaurosV0
# (--final-games) on the Zig duel harness, printed as a FINAL TOURNAMENT line and kept in WORK/tournaments/results.txt.
#
#   1  head     only the actor trains: a 3-layer actor head (192 -> 64 -> 64 -> 9, ~17k parameters) on the state
#               token, added to the linear head and zero-initialised so it starts as START's policy; embeddings, the
#               recurrent transformer and the win head frozen. Phase 1: 1500 steps imitating Exeggcute (KL only, usual
#               human/strong data; -> WORK/exp_head_exeggcute). Phase 2: imitates the strong bots' played moves (weight 1) with a
#               little KL to Exeggcute against overfitting (0.1), on the bots' games only. (head_experiment.sh runs this
#               one alone.)                                                                               -> WORK/exp_head
#   2  opphead  an opponent-action head added (the opponent's next move or switch, read from the next decision's record);
#               everything but the win head trains: imitation of the bots on their games (1), KL to Exeggcute on the other
#               games (1), the opponent head's cross-entropy on all (0.5). Data: bots, human and other bot games 3:2:1.
#               --exp-deep-actor adds the 3-layer actor head too (it trains with everything else).
#                                                                                                         -> WORK/exp_opphead
#   3  blend    no training: tournaments of START alone and of policy blends (each member keeps its own recurrent state;
#               the move probabilities are averaged): START+head, START+opphead, head+opphead, all three, START+weights_5000
#               (--blends; --blend-mode logit for a geometric blend)
#
#   ./experiments.sh /root/work/run_ft_bots_b1_actor/weights_10000.pt --val-every 1000 --patience 3
#
# Every argument after START goes to each experiment (launch.py: --exp-steps 10000 --exp-lr --exp-kl-weight --exp-bc-weight
# --exp-anchor --exp-opp-weight --exp-mix --final-games 500 --final-procs ...). Safe to run again: a finished experiment is skipped
# (training resumes from its latest.pt, a played tournament is not replayed). EXPERIMENTS="head blend" runs a subset.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -gt 0 ] && [ "${1#--}" = "$1" ] || { echo "experiments.sh: give the starting weights first (e.g. /root/work/run_ft_bots_b1_actor/weights_10000.pt)"; exit 2; }
start="$1"; shift
[ -f "$start" ] || { echo "experiments.sh: no $start"; exit 2; }
for experiment in ${EXPERIMENTS:-head opphead blend}; do
  echo "$(date +%H:%M:%S) experiments.sh: experiment $experiment"
  ./launch.sh --train-only --finetune-from "$start" --experiment "$experiment" "$@"
done
grep "FINAL TOURNAMENT" "${WORK:-$HOME/work}/tournaments/results.txt" | tail -n 12 || true
