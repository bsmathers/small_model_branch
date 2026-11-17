#!/usr/bin/env bash
# The value stages from a fine-tuned actor (default WORK/run_ft_bots_b1_actor/weights_10000.pt), skipping the actor stage:
#
#   1  value head   a fresh win head (LayerNorm -> 192x64 -> GELU -> 64x1) trained on the game results; embeddings, the recurrent
#                   transformer and the actor head frozen, so the policy is exactly the actor's. lr 3e-4 (--value-lr), up to
#                   10000 steps (--ft-value-head-steps)                                         -> WORK/run_value10k_value_head
#   2  everything   from stage 1: the win loss on all parameters, anchored with weight 1 (--ft-all-anchor) to the actor's policy
#                   (KL(actor || current)), lr 1e-4 (--ft-lr), up to 20000 steps (--ft-all-steps; 0 stops after stage 1)
#                                                                                               -> WORK/run_value10k_all
#
# Data: the usual human/strong mix. Early stopping follows the validation win loss (BCE) in stage 1 and the weighted loss in
# stage 2. The usual 100-game tournaments against TaurosV0 run every 5000 steps. Safe to run again: it resumes.
#
#   ./value.sh --val-every 1000 --patience 3
#   ./value.sh /path/to/actor.pt --ft-all-steps 0        # stage 1 only
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK:-$HOME/work}"
weights="$WORK/run_ft_bots_b1_actor/weights_10000.pt"
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then weights="$1"; shift; fi
[ -f "$weights" ] || { echo "value.sh: no $weights (give the actor weights as the first argument)"; exit 2; }
exec ./launch.sh --train-only --finetune-from "$weights" --ft-skip-actor --finetune-run run_value10k "$@"
