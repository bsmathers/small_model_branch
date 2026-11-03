"""Runs the whole job on one machine: fetch data and Exeggcute's checkpoints, parse, label, build, train.

Normally started through `launch.sh`, which installs the requirements first:

    ./launch.sh                    # the full run with the defaults below
    ./launch.sh --smoke            # a few minutes: tiny sample of every stage, prints timings and projections
    ./launch.sh --games 60000 --steps 30000 --mix human=0.5 --mix strong=0.5

Every stage is resumable: games already parsed, labelled or built are skipped, and training continues from
its last checkpoint, so a stopped run is continued by starting the same command again. Progress is plain
text in WORK/logs (one file per stage and shard) and in WORK/logs/launch.txt.

Layout under WORK: hf/ (downloaded zips), raw/<source>/ (logs), metamon-cache/ (checkpoints, usage stats),
<source>/{trajectories,labels,built}, tables/, run/ (checkpoints; weights_<step>.pt is the model).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import zipfile

SOURCES = ("human", "strong")


class Launcher:
    def __init__(self, args):
        self.args = args
        self.work = os.path.abspath(args.work)
        self.logs = os.path.join(self.work, "logs")
        os.makedirs(self.logs, exist_ok=True)
        self.cache = os.path.abspath(args.cache or os.path.join(self.work, "metamon-cache"))
        self.timings: dict[str, float] = {}
        self.py = sys.executable

    # ---- small helpers ----

    def log(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {text}"
        print(line, flush=True)
        with open(os.path.join(self.logs, "launch.txt"), "a") as f:
            f.write(line + "\n")

    def env(self, **extra) -> dict:
        env = dict(os.environ)
        env.update({"METAMON_CACHE_DIR": self.cache, "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"})
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def run_parallel(self, stage: str, commands: list[list[str]], envs: list[dict], progress=None) -> None:
        """Runs the commands at once, one log file each, and stops everything if one fails."""
        start = time.time()
        procs = []
        for i, (cmd, env) in enumerate(zip(commands, envs)):
            handle = open(os.path.join(self.logs, f"{stage}_{i}.txt"), "w")
            procs.append((subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
        last = 0.0
        try:
            while any(p.poll() is None for p, _ in procs):
                time.sleep(2)
                if progress and time.time() - last > 30:
                    last = time.time()
                    self.log(f"{stage}: {progress()} ({time.time() - start:.0f}s)")
                failed = [i for i, (p, _) in enumerate(procs) if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"{stage} shard {failed[0]} failed; see {self.logs}/{stage}_{failed[0]}.txt")
        except BaseException:
            for p, _ in procs:
                if p.poll() is None:
                    p.terminate()
            raise
        finally:
            for _, h in procs:
                h.close()
        bad = [i for i, (p, _) in enumerate(procs) if p.returncode != 0]
        if bad:
            raise RuntimeError(f"{stage} shard {bad[0]} failed; see {self.logs}/{stage}_{bad[0]}.txt")
        self.timings[stage] = self.timings.get(stage, 0.0) + time.time() - start

    def run_one(self, stage: str, cmd: list[str], env: dict) -> None:
        self.run_parallel(stage, [cmd], [env])

    # ---- stages ----

    def hardware(self) -> None:
        import torch

        a = self.args
        self.cpus = os.cpu_count() or 1
        self.gpus = []
        if a.device != "cpu" and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                self.gpus.append({"name": p.name, "mem": p.total_memory / 2**30, "cc": (p.major, p.minor)})
        self.device = "cuda" if self.gpus else "cpu"
        if a.precision != "auto":
            self.precision = a.precision
        elif self.gpus and self.gpus[0]["cc"][0] >= 8:
            self.precision = "bf16"
        elif self.gpus:
            self.precision = "fp16"  # bf16 is emulated or missing before Ampere
        else:
            self.precision = "fp32"
        free = __import__("shutil").disk_usage(self.work).free / 2**30
        self.log(f"hardware: {self.cpus} CPUs, " + (", ".join(f"{g['name']} {g['mem']:.0f} GiB (sm_{g['cc'][0]}{g['cc'][1]})" for g in self.gpus) or "no GPU") + f"; precision {self.precision}; {free:.0f} GiB free at {self.work}")
        if free < 40 and not a.smoke:
            self.log("warning: under 40 GiB free; the default sample needs about 25 GiB with the downloads")

    def prefetch(self) -> None:
        """Downloads the usage statistics and Exeggcute's checkpoints once, before any parallel stage
        (concurrent first downloads would race), then forces offline mode for the rest of the run."""
        marker = os.path.join(self.cache, ".prefetched")
        if os.path.exists(marker):
            return
        os.makedirs(self.cache, exist_ok=True)
        code = (
            "from metamon.data.download import download_usage_stats\n"
            "download_usage_stats(1)\n"
            "from metamon.recurrent.label import load_ensemble\n"
            "load_ensemble(%r, 'cpu', 2)\n" % self.cache
        )
        self.log("prefetching usage statistics and Exeggcute's checkpoints (about 2.5 GB the first time)")
        self.run_one("prefetch", [self.py, "-c", code], self.env())
        open(marker, "w").write("ok\n")

    def fetch_data(self) -> dict[str, str]:
        a = self.args
        raw = {}
        for name, zip_name, override in (("human", a.human_zip, a.raw_human), ("strong", a.strong_zip, a.raw_strong)):
            if override:
                raw[name] = os.path.abspath(override)
                continue
            dest = os.path.join(self.work, "raw", name)
            marker = os.path.join(dest, ".done")
            if not os.path.exists(marker):
                from huggingface_hub import hf_hub_download

                self.log(f"downloading {a.data_repo}/{zip_name}")
                path = hf_hub_download(repo_id=a.data_repo, repo_type="dataset", filename=zip_name, local_dir=os.path.join(self.work, "hf"))
                os.makedirs(dest, exist_ok=True)
                count = 0
                with zipfile.ZipFile(path) as z:
                    for member in z.namelist():
                        base = os.path.basename(member)
                        if member.startswith("__MACOSX") or not base.endswith(".log") or base.startswith("."):
                            continue
                        with z.open(member) as src, open(os.path.join(dest, base), "wb") as out:  # the basename only: no path tricks
                            out.write(src.read())
                        count += 1
                open(marker, "w").write(f"{count}\n")
                self.log(f"{name}: extracted {count} logs")
                os.remove(path)  # the zip is no longer needed
            raw[name] = dest
        return raw

    def count_files(self, directory: str, suffix: str) -> int:
        return sum(1 for n in os.listdir(directory) if n.endswith(suffix)) if os.path.isdir(directory) else 0

    def prepare(self, name: str, raw_dir: str, games: int) -> None:
        a = self.args
        total = self.count_files(raw_dir, ".log")
        if total == 0:
            raise RuntimeError(f"no .log files in {raw_dir}")
        every = max(1, total // games) if games else 1
        wanted = total // every
        src = os.path.join(self.work, name)
        traj, labels, built = (os.path.join(src, d) for d in ("trajectories", "labels", "built"))
        self.log(f"{name}: {total} logs, taking every {every}th (about {wanted} games, two points of view each)")

        # every stage resumes from what is on disk, so data made by an older parser must not be mixed in
        from metamon.recurrent.parse import PARSER_VERSION

        os.makedirs(src, exist_ok=True)
        stamp = os.path.join(src, ".parser_version")
        if os.path.exists(stamp):
            found = open(stamp).read().strip()
            if found != PARSER_VERSION:
                raise RuntimeError(
                    f"{src} was made by parser version {found!r}, the code is {PARSER_VERSION!r}: its trajectories, labels and "
                    f"built data are stale. Use another WORK folder or move {src} away."
                )
        elif any(os.path.isdir(os.path.join(src, d)) and os.listdir(os.path.join(src, d)) for d in ("trajectories", "labels", "built")):
            raise RuntimeError(f"{src} has data but no .parser_version stamp (made before parser versions): move it away or use another WORK folder.")
        with open(stamp, "w") as f:
            f.write(PARSER_VERSION)

        # parse, in parallel shards
        procs = a.parse_procs or self.cpus
        stage = f"parse_{name}"
        cmds = [[self.py, "-m", "metamon.recurrent.parse", raw_dir, traj, "--every", str(every), "--shard", f"{i}/{procs}"] for i in range(procs)]
        self.run_parallel(stage, cmds, [self.env(HF_HUB_OFFLINE=1)] * procs, lambda: f"{self.count_files(traj, '.lz4')} trajectories")

        # the original parser's states and actions, which Exeggcute reads (it was trained on them), folded into each trajectory
        legacy = os.path.join(src, "legacy")
        cmds = [[self.py, os.path.join("tools", "legacy_parse.py"), raw_dir, legacy, "--every", str(every), "--shard", f"{i}/{procs}"] for i in range(procs)]
        self.run_parallel(f"legacy_{name}", cmds, [self.env(HF_HUB_OFFLINE=1)] * procs, lambda: f"{self.count_files(legacy, '.lz4')} legacy trajectories")
        cmds = [[self.py, "-m", "metamon.recurrent.merge_legacy", traj, legacy, "--shard", f"{i}/{procs}"] for i in range(procs)]
        self.run_parallel(f"merge_{name}", cmds, [self.env(HF_HUB_OFFLINE=1)] * procs)
        n_traj = self.count_files(traj, ".lz4")
        self.log(f"{name}: parsed {n_traj} trajectories (games the two parsers disagree on are in {traj}_rejected)")

        # label, a few processes per GPU (each is mostly Python overhead around small batches)
        gpus = max(1, len(self.gpus))
        per_gpu = a.label_procs or (3 if self.cpus >= 12 else 2) if self.gpus else 1
        shards = gpus * per_gpu
        threads = max(1, self.cpus // shards) if not self.gpus else max(1, min(4, self.cpus // shards))
        stage = f"label_{name}"
        cmds, envs = [], []
        for k in range(shards):
            cmds.append([self.py, "-m", "metamon.recurrent.label", traj, labels, "--cache", self.cache, "--batch", str(a.label_batch),
                         "--device", self.device, "--threads", str(threads), "--precision", self.precision if self.gpus else "fp32", "--shard", f"{k}/{shards}"])
            envs.append(self.env(HF_HUB_OFFLINE=1, **({"CUDA_VISIBLE_DEVICES": k % gpus} if self.gpus else {})))
        self.run_parallel(stage, cmds, envs, lambda: f"{self.count_files(labels, '.npz')} of {n_traj} labelled")
        self.log(f"{name}: labelled {self.count_files(labels, '.npz')} trajectories")

        # build
        stage = f"build_{name}"
        self.run_one(stage, [self.py, "-m", "metamon.recurrent.build", traj, labels, built, os.path.join(self.work, "tables"), "--workers", str(self.cpus), "--bc-players", a.bc_players], self.env(HF_HUB_OFFLINE=1))
        self.log(f"{name}: built {self.count_files(built, '.npz')} games")
        self.counts = getattr(self, "counts", {})
        self.counts[name] = (wanted, n_traj)

    def train(self) -> None:
        a = self.args
        out = os.path.join(self.work, "run")
        cmd = [self.py, "-m", "metamon.recurrent.train", out]
        for name, weight in self.mix.items():
            cmd += ["--source", f"{name}={os.path.join(self.work, name, 'built')}:{weight}"]
        cmd += ["--steps", str(a.steps), "--batch", str(a.batch), "--workers", str(min(self.cpus, 12)), "--device", self.device,
                "--precision", self.precision if self.gpus else "fp32", "--target", "blend", "--alpha", str(a.alpha), "--bc-weight", str(a.bc_weight), "--lr", str(a.lr), "--ckpt-every", str(a.ckpt_every),
                "--val-every", str(a.val_every), "--save-every", str(a.save_every), "--val-fraction", str(a.val_fraction), "--resume"]
        if a.val_batches:
            cmd += ["--val-batches", str(a.val_batches)]
        self.log(f"training {a.steps} steps; progress in {out}/log.txt")
        self.run_one("train", cmd, self.env(HF_HUB_OFFLINE=1))

    def summary(self) -> None:
        a = self.args
        out = os.path.join(self.work, "run")
        weights = sorted((f for f in os.listdir(out) if f.startswith("weights_")), key=lambda f: int(f.split("_")[1].split(".")[0])) if os.path.isdir(out) else []
        self.log("stage times: " + ", ".join(f"{k} {v:.0f}s" for k, v in self.timings.items()))
        if weights:
            self.log(f"final weights: {os.path.join(out, weights[-1])}")
        for line in open(os.path.join(out, "log.txt")).read().splitlines()[-6:] if os.path.exists(os.path.join(out, "log.txt")) else []:
            self.log("  " + line)
        if a.smoke:
            self.project()

    def project(self) -> None:
        """Scales the smoke timings up to the default sizes. Rough: the startup cost of each stage is counted
        at full size, so the projection errs high."""
        a = self.args
        full_games = {n: a.games for n in SOURCES}
        scale_games = sum(full_games.values()) / max(1, sum(w for w, _ in self.counts.values()))
        parts = []
        for stage in ("parse", "legacy", "merge", "label", "build"):
            t = sum(v for k, v in self.timings.items() if k.startswith(stage + "_"))
            parts.append((stage, t * scale_games))
        smoke_steps = a.steps
        t_train = self.timings.get("train", 0.0)
        parts.append(("train (20000 steps)", t_train / max(1, smoke_steps) * 20000))
        self.log("projected for --games %d per source and 20000 steps (hours): " % a.games + ", ".join(f"{k} {v / 3600:.1f}" for k, v in parts) + f"; total {sum(v for _, v in parts) / 3600:.1f}")

    def main(self) -> None:
        a = self.args
        self.hardware()
        os.makedirs(os.path.join(self.work, "tables"), exist_ok=True)
        if not os.path.exists(os.path.join(self.work, "tables", "species_ids.json")):
            self.run_one("tables", [self.py, "-m", "metamon.recurrent.tables", os.path.join(self.work, "tables")], self.env())
        self.prefetch()
        raw = self.fetch_data()
        self.mix = dict(m.split("=") for m in a.mix) if a.mix else {"human": 0.4, "strong": 0.6}
        for name in SOURCES:
            if float(self.mix.get(name, 0)) > 0:
                self.prepare(name, raw[name], a.games)
        self.mix = {k: v for k, v in self.mix.items() if float(v) > 0}
        self.train()
        self.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.environ.get("WORK", "/workspace/work" if os.path.isdir("/workspace") else os.path.expanduser("~/work")))
    ap.add_argument("--cache", default=None, help="metamon cache (default WORK/metamon-cache)")
    ap.add_argument("--data-repo", default="bsmathers/raw_replays")
    ap.add_argument("--human-zip", default="human_logs.zip")
    ap.add_argument("--strong-zip", default="strong_model_replays.zip")
    ap.add_argument("--raw-human", default=None, help="a folder of .log files, instead of downloading")
    ap.add_argument("--raw-strong", default=None)
    ap.add_argument("--games", type=int, default=30000, help="games to take from each source (0: all)")
    ap.add_argument("--mix", action="append", help="name=weight, e.g. human=0.4 (default human=0.4 strong=0.6; weight 0 skips a source)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", default="auto", help="training batch: a number or auto (largest that fits in GPU memory)")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the distillation KL")
    ap.add_argument("--bc-weight", type=float, default=1.0, help="weight of the imitation term on the imitated bots' games")
    ap.add_argument("--bc-players", default="Tauros-v1,Tauros-v2,SmallG1Online-v2,SmallG1Online-v3", help="usernames whose played actions are imitated")
    ap.add_argument("--ckpt-every", type=int, default=1, help="steps between writes of latest.pt (1: every iteration)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=int, default=20, help="hold out 1 game in N (by battle id)")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--parse-procs", type=int, default=0, help="parser processes (default: one per CPU)")
    ap.add_argument("--label-procs", type=int, default=0, help="labelling processes per GPU (default 3, or 2 on small hosts)")
    ap.add_argument("--label-batch", type=int, default=64, help="games labelled in lockstep per process")
    ap.add_argument("--smoke", action="store_true", help="tiny sample of every stage, prints timings and a projection")
    args = ap.parse_args()
    if args.smoke:
        args.cache = args.cache or os.path.join(args.work, "metamon-cache")  # shared with the full run
        args.games = min(args.games, 150)
        args.steps, args.val_every, args.save_every, args.val_batches = min(args.steps, 150), 75, 150, 20
        args.val_fraction = min(args.val_fraction, 5)
        args.work = args.work.rstrip("/") + "_smoke"
    Launcher(args).main()


if __name__ == "__main__":
    main()
