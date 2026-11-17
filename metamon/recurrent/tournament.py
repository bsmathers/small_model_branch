"""Games between a training checkpoint of the recurrent model and TaurosV0, on the Zig duel harness (repo folder duel/).

The same games and the same student shim as the local shadow trial: the harness (a copy of pkmn-search-wide) plays Gen 1 OU on
the pkmn engine with teams from duel/teamset, and asks two policy servers for every decision: duel/tools/recurrent_policy_server.py
(this model, argmax of the actor head over the legal options) and duel/tools/metamon_tauros_server.py (TaurosV0, checkpoint 62,
argmax, run from upstream metamon at METAMON_COMMIT with duel/tauros_metamon.patch, which lets it run without FlashAttention).
Games come in pairs on the same two teams with the sides swapped; the same seeds every time, so checkpoints face the same games.

    python -m metamon.recurrent.tournament setup --work WORK [--cache CACHE]     # once (resumable): about 5 minutes
    python -m metamon.recurrent.tournament play WEIGHTS --work WORK [--games 100] [--procs 4] [--out result.json]

setup puts everything under WORK/duel: Zig ZIG_VERSION (unless it is on the PATH), pkmn/engine at ENGINE_COMMIT, the patched
upstream metamon, TaurosV0's checkpoint in CACHE/pretrained_models, the harness built from the repo's copy, and a 2-game test
with an untrained model (fails unless both servers answer every decision). It writes WORK/duel/.ready when everything works.
play splits the games over --procs harness processes (each runs its own two servers, one CPU thread each).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DUEL = REPO / "duel"
ZIG_VERSION = "0.16.0"
ENGINE_REPO, ENGINE_COMMIT = "https://github.com/pkmn/engine.git", "78dc891c49788e6ec9007d0f02247d2e04a03d29"
METAMON_REPO, METAMON_COMMIT = "https://github.com/UT-Austin-RPL/metamon.git", "0a00a759c9a4382a2877088d828302ec294a05a5"
TAUROS_REPO, TAUROS_FILE = "jakegrigsby/metamon", "tauros-v0/ckpts/policy_weights/policy_epoch_62.pt"
RESULT = re.compile(r"duel: A (\d+), B (\d+), draws (\d+) of (\d+) games; fallbacks A (\d+) B (\d+)")


def say(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def run(cmd: list, **kw) -> None:
    say("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def fetch_commit(url: str, commit: str, dest: Path) -> None:
    """A checkout of one commit (shallow), or nothing when dest already is one."""
    if (dest / ".git").exists():
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == commit:
            return
        shutil.rmtree(dest)
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    run(["git", "-C", tmp, "init", "-q"])
    run(["git", "-C", tmp, "fetch", "-q", "--depth", "1", url, commit])
    run(["git", "-C", tmp, "checkout", "-q", "FETCH_HEAD"])
    tmp.rename(dest)


def zig_binary(base: Path) -> str:
    found = shutil.which("zig")
    if found and subprocess.run([found, "version"], capture_output=True, text=True).stdout.strip() == ZIG_VERSION:
        return found
    machine = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}[platform.machine().lower()]
    system = {"Linux": "linux", "Darwin": "macos"}[platform.system()]
    key = f"{machine}-{system}"
    dest = base / f"zig-{key}-{ZIG_VERSION}"
    if (dest / "zig").exists():
        return str(dest / "zig")
    index = json.load(urllib.request.urlopen("https://ziglang.org/download/index.json", timeout=60))
    entry = index[ZIG_VERSION][key]
    archive = base / os.path.basename(entry["tarball"])
    say(f"downloading Zig {ZIG_VERSION} ({int(entry['size']) / 2**20:.0f} MiB)")
    urllib.request.urlretrieve(entry["tarball"], archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != entry["shasum"]:
        archive.unlink()
        raise SystemExit(f"Zig download has sha256 {digest}, expected {entry['shasum']}")
    with tarfile.open(archive) as t:
        try:
            t.extractall(base, filter="data")
        except TypeError:  # (a Python without extraction filters; the archive's checksum was verified above)
            t.extractall(base)
    archive.unlink()
    return str(dest / "zig")


def blend_spec(weights: str) -> str:
    """The student's weights for the policy server: one file, or a blend "a.pt@0.5,b.pt@0.5" (metamon/recurrent/blend.py); the
    paths made absolute (the harness runs in another folder)."""
    from metamon.recurrent.blend import parse_spec

    return ",".join(f"{Path(path).resolve()}@{weight:g}" for path, weight in parse_spec(weights))


def duel_env(work: Path, cache: Path, weights: str, device: str = "cpu", temperature: float = 0.0) -> dict:
    env = dict(os.environ)
    env.update({
        "DUEL_PYTHON": sys.executable, "DUEL_TOOLS": str(DUEL / "tools"), "TAUROS_METAMON": str(work / "duel" / "tauros-metamon"),
        "TAUROS_CACHE": str(cache), "TAUROS_DEVICE": device, "RECURRENT_WEIGHTS": blend_spec(weights),
        "METAMON_CACHE_DIR": str(cache), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0", "POLICY_TEMPERATURE": str(temperature),
    })
    env.pop("RECURRENT_SHADOW", None)
    env.pop("RECURRENT_DUMP", None)
    return env


def setup(work: Path, cache: Path) -> None:
    base = work / "duel"
    base.mkdir(parents=True, exist_ok=True)
    ready = base / ".ready"
    ready.unlink(missing_ok=True)
    zig = zig_binary(base)
    say(f"zig: {zig}")
    fetch_commit(ENGINE_REPO, ENGINE_COMMIT, base / "engine")
    say(f"engine: pkmn/engine {ENGINE_COMMIT[:7]}")
    tauros = base / "tauros-metamon"
    if not (tauros / ".patched").exists():
        shutil.rmtree(tauros, ignore_errors=True)
        fetch_commit(METAMON_REPO, METAMON_COMMIT, tauros)
        run(["git", "-C", tauros, "apply", DUEL / "tauros_metamon.patch"])
        (tauros / ".patched").write_text(METAMON_COMMIT + "\n")
    say(f"TaurosV0 code: upstream metamon {METAMON_COMMIT[:7]} + duel/tauros_metamon.patch")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=TAUROS_REPO, filename=TAUROS_FILE, cache_dir=str(cache / "pretrained_models"))
    say(f"TaurosV0 checkpoint: {path}")
    # the harness and the teams are copied out of the repo (the build writes next to the sources; ../engine and ../teamset
    # are where the harness looks), fresh every setup so a code update reaches the build
    for name in ("harness", "teamset"):
        shutil.copytree(DUEL / name, base / name, dirs_exist_ok=True)
    run([zig, "build", "-Doptimize=ReleaseFast"], cwd=base / "harness")
    # a 2-game test with an untrained model: both servers start and answer every decision
    import torch

    from metamon.recurrent.model import PolicyValueModel

    torch.manual_seed(0)
    test = base / "untrained.pt"
    torch.save(PolicyValueModel().state_dict(), test)
    result = play(work, cache, str(test), games=2, procs=1, log_dir=base / "test_logs")
    if result["fallbacks_student"] or result["fallbacks_tauros"]:
        raise SystemExit(f"test games: a server failed to answer ({result}); see {base / 'test_logs'}")
    ready.write_text(json.dumps({"zig": zig, "engine": ENGINE_COMMIT, "metamon": METAMON_COMMIT, "test": result}) + "\n")
    say(f"ready: 2 test games played in {result['seconds']:.0f}s")


def play(work: Path, cache: Path, weights: str, games: int = 100, procs: int = 4, device: str = "cpu", log_dir: Path | None = None,
         temperature: float = 0.0) -> dict:
    """Plays `games` games (rounded up to pairs), split over `procs` harness processes. Returns the totals. temperature 0: both
    sides play their argmax; T > 0: both sample (T = 1: each model's own action probabilities), seeded per process."""
    base = work / "duel"
    binary = base / "harness" / "zig-out" / "bin" / "pkmn-search"
    if not binary.exists():
        raise SystemExit(f"no harness at {binary}: run `python -m metamon.recurrent.tournament setup` first")
    pairs = max(1, (games + 1) // 2)
    procs = max(1, min(procs, pairs))
    split = [pairs // procs + (1 if k < pairs % procs else 0) for k in range(procs)]
    log_dir = Path(log_dir or base / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = duel_env(work, cache, weights, device, temperature)
    start, running, first = time.time(), [], 1
    for k, n in enumerate(split):
        log = open(log_dir / f"duel_{k}.txt", "w")
        cmd = [str(binary), "duel", str(first), str(2 * n), "recurrent", "tauros"]
        running.append((subprocess.Popen(cmd, cwd=base / "harness", env={**env, "POLICY_SEED": str(first)}, stdout=log, stderr=subprocess.STDOUT),
                        log, log_dir / f"duel_{k}.txt"))
        first += 2 * n  # (game g of a process plays seed first+g, its teams come from first+g/2+5000: no two processes share a game)
    totals = {"student": 0, "tauros": 0, "draws": 0, "games": 0, "fallbacks_student": 0, "fallbacks_tauros": 0}
    errors = []
    for p, log, path in running:
        p.wait()
        log.close()
        text = path.read_text(errors="replace")
        m = RESULT.search(text)
        if p.returncode != 0 or not m:
            errors.append(f"{path.name}: exit {p.returncode}, last lines: {' | '.join(text.strip().splitlines()[-3:])}")
            continue
        for key, value in zip(("student", "tauros", "draws", "games", "fallbacks_student", "fallbacks_tauros"), m.groups()):
            totals[key] += int(value)
    if errors:
        raise SystemExit("duel failed: " + "; ".join(errors))
    totals["seconds"] = time.time() - start
    # what was played, for comparing runs: the weights, the simulator, the policies' settings
    from metamon.recurrent.blend import describe as describe_blend

    members = describe_blend(blend_spec(weights))
    if len(members) == 1:
        totals["weights_sha256"] = members[0]["sha256"]
    else:
        totals["blend"] = members
        totals["blend_mode"] = os.environ.get("RECURRENT_BLEND_MODE", "mix")
    totals["engine"] = ENGINE_COMMIT
    totals["tauros_metamon"] = METAMON_COMMIT
    totals["temperature"] = temperature
    totals["tauros_fixes"] = os.environ.get("TAUROS_FIXES", "wait,prevmove")
    totals["tauros_filter"] = os.environ.get("TAUROS_FILTER", "0")
    totals["harness_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((DUEL / "harness" / "src").glob("*.zig")))).hexdigest()
    totals["tools_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((DUEL / "tools").glob("*.py")))).hexdigest()
    return totals


def describe(r: dict) -> str:
    n = max(1, r["games"])
    score = (r["student"] + 0.5 * r["draws"]) / n
    se = (score * (1 - score) / n) ** 0.5
    return (f"student {r['student']} - TaurosV0 {r['tauros']} ({r['draws']} draws) of {r['games']} games: student scores "
            f"{score:.1%} ± {se:.1%}; random-move fallbacks student {r['fallbacks_student']}, TaurosV0 {r['fallbacks_tauros']}; "
            f"{r['seconds'] / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "play"):
        p = sub.add_parser(name)
        if name == "play":
            p.add_argument("weights", help="a weights file, or a blend: a.pt@0.5,b.pt@0.5 (RECURRENT_BLEND_MODE=logit for the geometric blend)")
            p.add_argument("--games", type=int, default=100)
            p.add_argument("--procs", type=int, default=4)
            p.add_argument("--device", default="cpu", help="TaurosV0's device (the student runs on the CPU)")
            p.add_argument("--temperature", type=float, default=0.0, help="both sides: 0 plays the argmax, 1 samples from each model's own probabilities")
            p.add_argument("--out", default=None, help="write the totals here as JSON")
            p.add_argument("--log-dir", default=None)
        p.add_argument("--work", default=os.environ.get("WORK", os.path.expanduser("~/work")))
        p.add_argument("--cache", default=None, help="metamon cache (default WORK/metamon-cache)")
    args = ap.parse_args()
    work = Path(args.work).resolve()
    cache = Path(args.cache or work / "metamon-cache").resolve()
    if args.cmd == "setup":
        setup(work, cache)
        return
    result = play(work, cache, args.weights, args.games, args.procs, args.device, Path(args.log_dir) if args.log_dir else None, args.temperature)
    result["games_requested"] = args.games
    say(describe(result))
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
