#!/usr/bin/env bash
# One-command launch for the recurrent policy/value model: installs the requirements, then runs
# metamon.recurrent.launch. By default it downloads the built training games (bsmathers/gen1ou-labels-backup, private)
# and trains; --data labels builds from downloaded trajectories + labels, --data trajectories labels them too, --from-raw
# parses raw logs. Safe to run again: every stage resumes.
#
#   ./launch.sh --steps 20000 --value-steps 20000 --init /path/to/latest.pt   # continue an earlier run's policy
#
#   With a vast.ai local volume attached (at /data or /workspace), backups go to <volume>/metamon-backup every 100 steps
#   (--backup-every; --backup-dir PATH for another mount, none to turn off). Rerunning the same command on a new
#   container with that volume resumes from it.
#
#   ./launch.sh --smoke        # a few minutes; prints timings and a projection for the full run
#   ./launch.sh                # the full run
#   ./launch.sh --games 60000 --steps 30000 --mix human=0.4 --mix strong=0.6
#
# Environment: WORK (default ~/work, on the container disk; never /workspace, which on vast.ai can be a small volume), PYTHON (default python3), HF_TOKEN (the default built-games
# repo is private; or log in once with `hf auth login`), SKIP_INSTALL=1 to skip the install step.
set -euo pipefail
cd "$(dirname "$0")"

WORK="${WORK:-$HOME/work}"
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
