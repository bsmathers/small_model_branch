#!/usr/bin/env bash
# One-command launch for the recurrent policy/value model. After a fresh clone:
#
#   git clone https://github.com/bsmathers/small_model_branch.git && cd small_model_branch && ./launch.sh
#
# With no arguments that is the full run: network and code checks, install, download the trajectories and Exeggcute's
# labels (bsmathers/labeled_gen1_retrajectories, bsmathers/gen1ou-labels-backup), check the labels against Exeggcute's own
# code, build every game, check the built games, then train (policy stage up to 100k steps, value stage up to 20k; each stops
# early without progress). The model ends up in ~/work/run_v3_value/weights_final.pt. Safe to run again: every stage resumes.
# --data built trains on the uploaded built games instead, --data trajectories relabels everything, --from-raw parses raw logs.
#
# Also, in the background, never stopping the run:
#   - every 5000 steps (--tournament-every) 100 games (--tournament-games) of the checkpoint against TaurosV0 on the Zig duel
#     harness (duel/, set up under WORK/duel at the start); each result is printed as a TOURNAMENT line and kept in
#     WORK/tournaments/results.txt
#   - with HF_WRITE_TOKEN=hf_... set: after the built-game check, the built games are zipped and uploaded to a public dataset
#     repo (--upload-repo, default <token's user>/gen1ou-recurrent-built), usable later with --data built --built-repo REPO
# A crash restarts the run by itself (RETRIES, default 3), which resumes where it stopped; so does running the same command again.
#
#   ./launch.sh --steps 20000 --value-steps 20000 --init /path/to/latest.pt   # continue an earlier run's policy
#
#   With a vast.ai local volume attached (at /data or /workspace), backups go to <volume>/metamon-backup every 100 steps
#   (--backup-every; --backup-dir PATH for another mount, none to turn off). Rerunning the same command on a new
#   container with that volume resumes from it.
#
#   It first checks the network (Hugging Face, and PyPI/GitHub before the install), and that this checkout is
#   current; the launcher then checks the repos, the downloaded labels and the built games before training (see its docstring).
#
#   ./launch.sh --smoke        # a few minutes; prints timings and a projection for the full run
#   ./launch.sh --check-only   # download, check the labels, build, check the built games; stop before training
#   ./launch.sh                # the full run
#   ./launch.sh --games 60000 --steps 30000 --mix human=0.4 --mix strong=0.6
#
# Environment: WORK (default ~/work, on the container disk; never /workspace, which on vast.ai can be a small volume), PYTHON (default python3),
# SKIP_INSTALL=1 to skip the install step. No Hugging Face token is needed: both dataset repos are public.
set -euo pipefail
cd "$(dirname "$0")"

WORK="${WORK:-$HOME/work}"
export WORK
PYTHON="${PYTHON:-python3}"
mkdir -p "$WORK"

log() { echo "$(date +%H:%M:%S) launch.sh: $*"; }

# ---- first checks: network, credentials, code version (a few seconds; each failure stops here, before anything is downloaded) ----
reach() {  # URL: any HTTP answer below 400 within 20 s
  local code
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || echo 000)
  if [ "$code" -ge 200 ] && [ "$code" -lt 400 ]; then log "network: $1 ok ($code)"; else log "network: cannot reach $1 (HTTP $code). This host cannot download the data (blocked region or no internet): use another machine"; exit 1; fi
}
command -v curl >/dev/null 2>&1 || { log "curl is missing (apt-get install -y curl)"; exit 1; }
reach https://huggingface.co/api/datasets/bsmathers/labeled_gen1_retrajectories
# the zips come from Hugging Face's file CDN, another host: download the first 16 MB of one and report the speed
probe=$(curl -sS -m 60 -L -r 0-16777215 -o /dev/null -w '%{http_code} %{size_download} %{speed_download}' \
  https://huggingface.co/datasets/bsmathers/labeled_gen1_retrajectories/resolve/main/human.zip 2>/dev/null || echo "000 0 0")
read -r p_code p_size p_speed <<< "$probe"
if [ "$p_code" != "206" ] && [ "$p_code" != "200" ] || [ "${p_size%.*}" -lt 1000000 ]; then
  log "network: cannot download from Hugging Face's file storage (HTTP $p_code, ${p_size%.*} bytes): use another machine"; exit 1
fi
log "network: Hugging Face download speed $(( ${p_speed%.*} / 1000000 )) MB/s"
if [ -z "${SKIP_INSTALL:-}" ] && [ ! -f "$WORK/venv/.installed" ]; then
  reach https://pypi.org/simple/pip/
  reach https://github.com
fi
# the tournaments' setup also needs GitHub (pkmn/engine, upstream metamon) and ziglang.org; without them only the tournaments are off
for url in https://github.com https://ziglang.org/download/index.json; do
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)
  if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then log "warning: cannot reach $url (HTTP $code): the tournaments against TaurosV0 will be off"; fi
done
if git rev-parse --git-dir >/dev/null 2>&1; then
  if git fetch -q origin 2>/dev/null; then
    behind=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo 0)
    if [ "$behind" != "0" ]; then log "this checkout is $behind commits behind origin: run 'git pull' first"; exit 1; fi
  else
    log "warning: could not reach the git remote to check that this checkout is current"
  fi
  if ! git diff --quiet HEAD -- metamon launch.sh 2>/dev/null; then log "warning: local changes in metamon/ or launch.sh (git status)"; fi
  log "code: $(git log -1 --format='%h %s')"
fi

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

# ---- the run, restarted after a crash ----
# Every stage resumes from what is on disk (finished downloads, labels, built games and tournaments are skipped; training continues
# from its latest.pt, written every step), so after a crash the same command simply runs again: up to RETRIES times (default 3),
# 60 s apart. Not after a failed data check (exit 3: it would fail again), a usage error (2) or Ctrl-C / kill (130, 143).
# Each attempt runs in its own process group, and whatever it leaves running (a stuck worker) is killed before the next one.
# One run per WORK at a time (a lock), so a second start cannot write into the same checkpoints.
RETRIES="${RETRIES:-3}"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$WORK/.launch.lock"
  flock -n 9 || { log "another launch.sh is already running on $WORK (lock $WORK/.launch.lock)"; exit 1; }
fi
child=""
on_signal() {
  # Ctrl-C (or kill): only the launcher is told; it stops its stages in order (training finishes writing latest.pt), then
  # whatever is left of this attempt's process group is killed. A second Ctrl-C does not wait for that.
  trap 'log "stopping now"; [ -n "$child" ] && kill -KILL -- "-$child" 2>/dev/null; exit 130' INT TERM
  if [ -n "$child" ]; then
    log "stopping: letting the current stage save its checkpoint (Ctrl-C again to stop at once)"
    kill -INT "$child" 2>/dev/null
    wait "$child" 2>/dev/null
    kill -KILL -- "-$child" 2>/dev/null
  fi
  log "stopped; run the same command again to resume"
  exit 130
}
trap on_signal INT TERM
attempt=0
while true; do
  attempt=$((attempt + 1))
  log "starting (attempt $attempt of $((RETRIES + 1)))"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$PYTHON" -m metamon.recurrent.launch "$@" &
  else
    "$PYTHON" -m metamon.recurrent.launch "$@" &
  fi
  child=$!
  set +e
  wait "$child"
  code=$?
  set -e
  kill -KILL -- "-$child" 2>/dev/null || true  # (anything of this attempt still running)
  child=""
  case "$code" in
    0) log "finished"; exit 0 ;;
    2|3|130|143) log "stopped with exit code $code: not restarting (see $WORK/logs/launch.txt)"; exit "$code" ;;
  esac
  if [ "$attempt" -gt "$RETRIES" ]; then log "failed $attempt times (exit code $code): giving up; see $WORK/logs/launch.txt"; exit "$code"; fi
  log "crashed with exit code $code; restarting in 60 s (it resumes where it stopped)"
  sleep 60
done
