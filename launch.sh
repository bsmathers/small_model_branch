#!/usr/bin/env bash
# One-command launch for the recurrent policy/value model: installs the requirements, then runs
# metamon.recurrent.launch (download + unzip the finished trajectories from bsmathers/labeled_gen1_retrajectories, label,
# build, train; add --from-raw to parse raw logs instead). Safe to run again: every stage resumes.
#
#   ./launch.sh --smoke        # a few minutes; prints timings and a projection for the full run
#   ./launch.sh                # the full run
#   ./launch.sh --games 60000 --steps 30000 --mix human=0.4 --mix strong=0.6
#
# Environment: WORK (default /workspace/work or ~/work), PYTHON (default python3), HF_TOKEN (only if the data
# repo is private), SKIP_INSTALL=1 to skip the install step.
set -euo pipefail
cd "$(dirname "$0")"

WORK="${WORK:-$([ -d /workspace ] && echo /workspace/work || echo "$HOME/work")}"
export WORK
PYTHON="${PYTHON:-python3}"
mkdir -p "$WORK"

log() { echo "$(date +%H:%M:%S) launch.sh: $*"; }

if [ -z "${SKIP_INSTALL:-}" ]; then
  # A virtualenv that can see the system packages, so a preinstalled CUDA build of torch is reused.
  VENV="$WORK/venv"
  if [ ! -x "$VENV/bin/python" ]; then
    log "creating $VENV"
    "$PYTHON" -m venv --system-site-packages "$VENV"
  fi
  PYTHON="$VENV/bin/python"
  if [ ! -f "$VENV/.installed" ]; then
    log "installing requirements (a few minutes)"
    "$PYTHON" -m pip install --quiet --upgrade pip
    # torch first: reuse an existing CUDA build when it is new enough, else install the default Linux wheel
    if ! "$PYTHON" - <<'PY'
import sys
try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    sys.exit(0 if (major, minor) >= (2, 6) else 1)
except Exception:
    sys.exit(1)
PY
    then
      log "installing torch"
      "$PYTHON" -m pip install --quiet "torch>=2.6"
    fi
    # metamon and its dependencies (amago and poke-env come from GitHub, so git must be available)
    "$PYTHON" -m pip install --quiet -e .
    touch "$VENV/.installed"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { log "nvidia-smi sees a GPU but torch cannot use it; check that the torch build matches the driver (CUDA 12.x)"; exit 1; }
fi

log "starting"
exec "$PYTHON" -m metamon.recurrent.launch "$@"
