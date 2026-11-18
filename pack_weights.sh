#!/usr/bin/env bash
# Packs training runs' weights and logs on the training machine into one zip file, to download in one go.
#
#   ./pack_weights.sh                                   # the two imitation.sh runs (run_ft_fullhist_actor, run_ft_fullhist_seed1_actor)
#   ./pack_weights.sh run_ft_bots_b1_actor run_v3       # any runs in WORK
#
# For each run: every weights_*.pt (the 5000-step snapshots, weights_best.pt, weights_final.pt) and log.txt; latest.pt is left
# out (it also carries the optimizer state, about three times larger, and is only needed to resume training). Also
# WORK/tournaments/results.txt and WORK/logs/launch.txt, and SHA256SUMS of everything. Output: WORK/weights_<date>.zip; the
# end of the output prints its size and a download command. Nothing in WORK is changed or deleted.
#
# Environment: WORK (default ~/work), PYTHON (default python3; zips with Python's zipfile, so no zip install is needed).
set -euo pipefail
WORK=${WORK:-$HOME/work}
PYTHON=${PYTHON:-python3}
runs=("$@")
[ ${#runs[@]} -gt 0 ] || runs=(run_ft_fullhist_actor run_ft_fullhist_seed1_actor)

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
for run in "${runs[@]}"; do
  src="$WORK/$run"
  [ -d "$src" ] || { echo "pack_weights.sh: no run folder $src"; exit 1; }
  mkdir -p "$stage/$run"
  found=0
  for f in "$src"/weights_*.pt; do
    [ -e "$f" ] || continue
    cp "$f" "$stage/$run/"
    found=1
  done
  [ "$found" = 1 ] || echo "pack_weights.sh: warning: no weights_*.pt in $src"
  [ -f "$src/log.txt" ] && cp "$src/log.txt" "$stage/$run/"
  echo "pack_weights.sh: $run: $(ls "$stage/$run" | tr '\n' ' ')"
done
mkdir -p "$stage/logs"
[ -f "$WORK/tournaments/results.txt" ] && cp "$WORK/tournaments/results.txt" "$stage/logs/tournaments_results.txt"
[ -f "$WORK/logs/launch.txt" ] && cp "$WORK/logs/launch.txt" "$stage/logs/launch.txt"

(cd "$stage" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
out="$WORK/weights_$(date +%Y%m%d_%H%M%S).zip"
"$PYTHON" - "$stage" "$out" <<'EOF'
import os, sys, zipfile
stage, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out + ".part", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in sorted(os.walk(stage)):
        for name in sorted(files):
            path = os.path.join(root, name)
            z.write(path, os.path.relpath(path, stage))
os.replace(out + ".part", out)
EOF
size=$(du -h "$out" | cut -f1)
echo "pack_weights.sh: wrote $out ($size)"
echo "download (on your computer):  scp -P <port> root@<host>:$out ~/model_weights/"
