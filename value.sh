#!/usr/bin/env bash
# The value stages from a fine-tuned actor (default WORK/run_ft_bots_b1_actor/weights_10000.pt), skipping the actor stage:
#
#   1  value head   a fresh win head (LayerNorm -> 192x64 -> GELU -> 64x1) trained on the game results; embeddings, the recurrent
#                   transformer and the actor head frozen, so the policy is exactly the actor's. lr 3e-4 (--value-lr), up to
#                   10000 steps (--ft-value-head-steps)                                         -> WORK/run_value10k_value_head
#   2  everything   from stage 1: the win loss on all parameters, anchored with weight 1 (--ft-all-anchor) to the actor's policy
#                   (KL(actor || current)), lr 1e-4 (--ft-lr), up to 20000 steps (--ft-all-steps; 0 stops after stage 1)
#                                                                                               -> WORK/run_value10k_all
#   3  joint        from stage 2: everything trains on the win loss (1) + KL to Exeggcute's targets (0.25, --ft-joint-kl-weight)
#                   + imitation of the strong bots' played moves (1, --ft-joint-bc-weight), no anchor; lr 1e-4 (--ft-lr), up to
#                   20000 steps (--ft-joint-steps; 0 skips it), early stopping on the weighted sum   -> WORK/run_value10k_joint
#
# Data: the usual human/strong mix (stage 3: --ft-joint-mix). Early stopping follows the validation win loss (BCE) in stages 1
# and 2, the weighted sum in stage 3. The usual 100-game tournaments against TaurosV0 run every 5000 steps. Safe to run again:
# it resumes.
#
#   ./value.sh --val-every 1000 --patience 3
#   ./value.sh /path/to/actor.pt --ft-all-steps 0 --ft-joint-steps 0     # stage 1 only
#
# Rerunning after stages 1 and 2 finished skips them (each resumes at its end) and runs stage 3.
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK:-$HOME/work}"
weights="$WORK/run_ft_bots_b1_actor/weights_10000.pt"
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then weights="$1"; shift; fi
[ -f "$weights" ] || { echo "value.sh: no $weights (give the actor weights as the first argument)"; exit 2; }
exec ./launch.sh --train-only --finetune-from "$weights" --ft-skip-actor --finetune-run run_value10k --ft-joint-steps 20000 "$@"
