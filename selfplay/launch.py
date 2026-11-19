"""One command for self-play training on a GPU machine (SELFPLAY_DESIGN.md).

    python selfplay/launch.py                    # setup, GPU smoke test, proposed settings, asks before training
    python selfplay/launch.py --yes              # the same, starting the training without asking
    python selfplay/launch.py --smoke-only       # stop after the smoke test
    python selfplay/launch.py --games 4096 --turns 64 --batch-chunks 1024    # fixed settings (skips their measurement)

1. Setup (each step skipped when done): Zig 0.16; the pkmn engine at 78dc891 in selfplay/zig/engine; libpkrl built; Metamon's
   high-ladder Gen 1 OU teams (hl_05_26, a public Hugging Face dataset) next to the repo's 807-team pool; the feature tables.
2. Smoke test on the GPU, measuring:
   - the environment's speed (random play) on all CPUs;
   - acting: the network's forward over every game's decisions plus the environment's step, for 1k to 16k games; the number of
     games is the one with the best decisions/s, stopping when doubling gains less than 10% or memory runs out;
   - training: the most 8-decision chunks per gradient step that fit (85% of GPU memory), and the time of one step;
   - turns per iteration: as many as the iteration's data (kept on the GPU) fits next to the training (at most 128, and enough
     for at least 8 gradient steps), defaulting to 64 when that fits;
   then plays 2 real iterations with those settings (WORK/smoke).
3. Prints the plan (settings and measured times) and asks for approval; then starts train_selfplay.py in WORK/run (it
   resumes when run again). The plan is kept in WORK/plan.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SP = REPO / "selfplay"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SP / "py"))


def say(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} launch: {text}", flush=True)


# ---------------------------------------------------------------- 1. setup


def setup(work: Path) -> dict:
    from metamon.recurrent.tournament import ENGINE_COMMIT, ENGINE_REPO, fetch_commit, zig_binary

    work.mkdir(parents=True, exist_ok=True)
    zig = zig_binary(work)
    say(f"zig: {zig}")
    engine = SP / "zig" / "engine"
    if engine.is_symlink() or (engine.exists() and not (engine / ".git").exists() and (engine / "build.zig").exists()):
        say(f"engine: using {engine} as it is (a link or a copy)")
    else:
        fetch_commit(ENGINE_REPO, ENGINE_COMMIT, engine)
        say(f"engine: pkmn/engine {ENGINE_COMMIT[:7]}")
    subprocess.run([zig, "build", "-Doptimize=ReleaseFast"], cwd=SP / "zig", check=True)
    lib = SP / "zig" / "zig-out" / "lib" / ("libpkrl.dylib" if sys.platform == "darwin" else "libpkrl.so")
    if not lib.exists():
        raise SystemExit(f"no {lib} after the build")
    say(f"library: {lib}")
    teams_b = fetch_teams(SP / "data" / "teams")
    tables = SP / "data" / "tables"
    if not (tables / "tables" / "species_ids.json").exists():
        from metamon.recurrent.tables import Tables

        Tables.from_dex(1).write(tables / "tables")
    say(f"tables: {tables}")
    return {"lib": str(lib), "tables": str(tables), "teams_a": str(REPO / "duel" / "teamset"), "teams_b": str(teams_b)}


def fetch_teams(root: Path) -> Path:
    """Metamon's hl_05_26 Gen 1 OU teams (public dataset jakegrigsby/metamon-teams), extracted with a path check."""
    dest = root / "hl_05_26" / "gen1ou"
    if dest.exists() and any(dest.glob("*.gen1ou_team")):
        return dest
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(repo_id="jakegrigsby/metamon-teams", filename="hl_05_26/gen1ou.tar.gz", repo_type="dataset",
                              local_dir=str(root / "download"))
    out = root / "hl_05_26"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            name = Path(m.name)
            if m.isdir():
                continue
            if not m.isfile() or name.is_absolute() or ".." in name.parts:
                raise SystemExit(f"unexpected entry in the team archive: {m.name}")
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(t.extractfile(m).read())
    say(f"teams: {sum(1 for _ in dest.glob('*.gen1ou_team'))} high-ladder teams in {dest}")
    return dest


# ---------------------------------------------------------------- 2. smoke test


def bytes_per_game_turn(env) -> int:
    """GPU bytes of the iteration buffers (train_selfplay.Collector) per game and turn: two rows."""
    row = env.IDS * 2 + env.FLOATS * 2 + env.MAX_EVENTS * env.FIELDS + 1 + env.SLOTS + 3 + 256 * 2 + 8 + env.SLOTS * 2 + 3 * 2
    return 2 * row + 2


def smoke(paths: dict, work: Path, args) -> dict:
    import torch

    from model import Config, SelfPlayModel, count_parameters
    from pkrl import Env, random_actions
    from train_selfplay import Collector, probe_chunks

    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU: the smoke test measures the GPU (--device cpu only checks that it runs)")
    if not cuda and not (args.games and args.batch_chunks):
        raise SystemExit("--device cpu needs --games and --batch-chunks (nothing to measure)")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def peak_reset():
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)

    prop = torch.cuda.get_device_properties(dev) if cuda else None
    total = prop.total_memory if cuda else 64 * 2**30
    cpus = os.cpu_count() or 1
    threads = args.threads or cpus
    name = prop.name if cuda else "cpu"
    say(f"device {name}, {total / 2**30:.1f} GiB; {cpus} CPUs")
    teams = (paths["teams_a"], paths["teams_b"])
    report = {"gpu": name, "gpu_gib": round(total / 2**30, 1), "cpus": cpus, "threads": threads}

    # the environment alone
    env = Env(4096, threads, 0, paths["tables"], teams, lib=paths["lib"])
    obs = env.reset()
    g = torch.Generator().manual_seed(0)
    t0, rows = time.time(), 0
    for _ in range(50):
        rows += int(obs["valid"].sum())
        obs = env.step(random_actions(obs, g))
    env_rate = rows / (time.time() - t0)
    say(f"environment: {env_rate:,.0f} decisions/s (4096 games, random play, {threads} threads); teams {env.pool_sizes()}")
    report["env_decisions_per_s"] = round(env_rate)
    env.close()

    model = SelfPlayModel(Config()).to(dev)
    say(f"model: {count_parameters(model):,} parameters")

    # acting: games per forward
    games = args.games
    if not games:
        best, best_rate = None, 0.0
        for n in (1024, 2048, 4096, 8192, 16384):
            try:
                env = Env(n, threads, 1, paths["tables"], teams, lib=paths["lib"])
                col = Collector(env, dev, model.cfg.d, 4)
                gen = torch.Generator(device=dev).manual_seed(0)
                col.run(model, True, gen)  # (warm-up)
                sync()
                t0 = time.time()
                b = col.run(model, True, gen)
                sync()
                rate = int(b["valid"].sum()) / (time.time() - t0)
                env.close()
                del col
                peak_reset()
            except torch.OutOfMemoryError:
                say(f"acting: {n} games do not fit")
                break
            say(f"acting: {n} games -> {rate:,.0f} decisions/s")
            if rate < best_rate * 1.10:
                break
            best, best_rate = n, rate
        games = best or 1024
        report["acting_decisions_per_s"] = round(best_rate)
    say(f"games: {games}")

    # training: chunks per step, and a step's time
    env = Env(games, threads, 2, paths["tables"], teams, lib=paths["lib"])
    col = Collector(env, dev, model.cfg.d, 8)
    gen = torch.Generator(device=dev).manual_seed(0)
    col.run(model, cuda, gen)
    targs = argparse.Namespace(chunk=8, batch_chunks=0, clip=0.2, kl_behavior=0.1, value_weight=1.0, value_all=False)
    chunks = args.batch_chunks or probe_chunks(model, col, targs, dev, True)
    peak_reset()
    import train_selfplay as ts

    T, R = col.buf["valid"].shape
    idx = torch.stack([torch.randint(0, T, (chunks, 8)), torch.randint(0, R, (chunks, 8))], -1).to(dev)
    adv = torch.ones(T, R, device=dev)
    tgt = torch.full((T, R, 3), 1 / 3, device=dev)
    keep = torch.ones(T, R, dtype=torch.bool, device=dev)
    model.train()
    times = []
    for _ in range(3):
        sync()
        t0 = time.time()
        loss, _, _ = ts.chunk_loss(model, col.buf, idx, adv, tgt, keep, targs, 0.01, cuda)
        loss.backward()
        model.zero_grad(set_to_none=True)
        sync()
        times.append(time.time() - t0)
    step_s = min(times)
    train_peak = torch.cuda.max_memory_allocated(dev) if cuda else 0
    say(f"training: {chunks} chunks of 8 decisions per step, {step_s * 1000:.0f} ms per step, peak {train_peak / 2**30:.1f} GiB")
    env.close()
    del col
    peak_reset()

    # turns per iteration: the iteration's buffers next to the training peak
    per_turn = games * bytes_per_game_turn(env)
    budget = 0.90 * total - train_peak - 2 * 2**30  # (headroom: the model, the acting batch, the allocator)
    t_max = max(1, int(budget // per_turn))
    t_min = max(4, -(-8 * chunks * 8 // int(2 * games * 0.9)))  # (at least 8 gradient steps per iteration)
    turns = args.turns or max(t_min, min(64, t_max, 128))
    if turns > t_max:
        say(f"warning: {turns} turns need {turns * per_turn / 2**30:.1f} GiB of buffers, more than the {budget / 2**30:.1f} GiB left")
    say(f"turns per iteration: {turns} (buffers {turns * per_turn / 2**30:.1f} GiB; fits up to {t_max})")
    report.update(games=games, turns=turns, batch_chunks=chunks, train_step_ms=round(step_s * 1000), train_peak_gib=round(train_peak / 2**30, 2),
                  buffers_gib=round(turns * per_turn / 2**30, 2))

    # two real iterations
    smoke_dir = work / "smoke"
    if smoke_dir.exists():
        for f in smoke_dir.glob("*"):
            f.unlink()
    cmd = train_cmd(paths, smoke_dir, games, turns, chunks, threads) + ["--iterations", "2", "--device", str(dev)]
    say("2 iterations: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    lines = [l for l in (smoke_dir / "log.txt").read_text().splitlines() if " iter " in l]
    report["smoke_iterations"] = lines
    return report


def train_cmd(paths: dict, out: Path, games: int, turns: int, chunks: int, threads: int) -> list[str]:
    return [sys.executable, str(SP / "py" / "train_selfplay.py"), str(out), "--games", str(games), "--turns", str(turns),
            "--batch-chunks", str(chunks), "--threads", str(threads), "--tables", paths["tables"], "--teams-a", paths["teams_a"],
            "--teams-b", paths["teams_b"]]


# ---------------------------------------------------------------- 3. plan and training


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.environ.get("WORK_SELFPLAY", os.path.expanduser("~/work/selfplay")))
    ap.add_argument("--yes", action="store_true", help="start the training without asking")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--games", type=int, default=0, help="concurrent games (0: measured)")
    ap.add_argument("--turns", type=int, default=0, help="turns per iteration (0: measured)")
    ap.add_argument("--batch-chunks", type=int, default=0, help="8-decision chunks per gradient step (0: measured)")
    ap.add_argument("--threads", type=int, default=0, help="environment threads (0: the CPUs)")
    ap.add_argument("--device", default="cuda", help="cpu: only checks that the smoke test runs (needs --games and --batch-chunks)")
    args = ap.parse_args()
    work = Path(args.work)
    paths = setup(work)
    report = smoke(paths, work, args)
    (work / "plan.json").write_text(json.dumps(report, indent=1))
    print("\n=== plan ===")
    for k in ("gpu", "gpu_gib", "cpus", "env_decisions_per_s", "acting_decisions_per_s", "games", "turns", "batch_chunks",
              "train_step_ms", "train_peak_gib", "buffers_gib"):
        if k in report:
            print(f"  {k:<24} {report[k]}")
    for line in report.get("smoke_iterations", []):
        print("  smoke:", line)
    cmd = train_cmd(paths, work / "run", report["games"], report["turns"], report["batch_chunks"], report["threads"])
    print("\ntraining command:\n  " + " ".join(cmd))
    if args.smoke_only:
        return
    if not args.yes:
        if not sys.stdin.isatty():
            print("(not interactive: run again with --yes to start it)")
            return
        if input("\nstart the training with these settings? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not started")
            return
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
