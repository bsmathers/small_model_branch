#!/usr/bin/env bash
# Fine-tuning after policy distillation, from the best distilled weights. Three stages, each starting from the one before:
#
#   A  actor        trunk + actor head (the win head gets no gradient): 0.5 x KL to Exeggcute's targets + 0.5 x imitation of the
#                   strong bots' played moves (Tauros-v1/v2, SmallG1Online-v2/v3), anchored with weight 5 to the starting
#                   policy; early stopping on that weighted validation loss                         -> WORK/run_ft_actor
#   1  value head   only the win head MLP (fresh); embeddings, the recurrent transformer and the actor head frozen
#                                                                                                   -> WORK/run_ft_value_head
#   2  everything   the win BCE, anchored with weight 1 to stage A's policy                         -> WORK/run_ft_all
#
# The model is WORK/run_ft_all/weights_final.pt. Everything launch.sh does around training still happens: the checks, crash
# restarts (each stage resumes from its latest.pt), backups, and tournaments against TaurosV0 every 5000 steps of each stage.
#
#   ./finetune.sh                                   # from WORK/run_v3/weights_best.pt
#   ./finetune.sh /path/to/weights.pt --ft-actor-steps 10000 --ft-bc-weight 0.7
#
# Options (launch.py): --ft-kl-weight 0.5 --ft-bc-weight 0.5 --ft-actor-anchor 5 --ft-all-anchor 1 --ft-lr 1e-4 (stages A and 2;
# stage 1 uses --value-lr 3e-4) --ft-actor-steps 20000 --ft-value-head-steps 10000 --ft-all-steps 20000 --finetune-run run_ft
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK:-$HOME/work}"
weights="$WORK/run_v3/weights_best.pt"
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then weights="$1"; shift; fi
[ -f "$weights" ] || { echo "finetune.sh: no $weights (give the weights as the first argument)"; exit 1; }
exec ./launch.sh --train-only --finetune-from "$weights" "$@"
