"""Runs the whole job on one machine: fetch the data, then (depending on --data) label, build, train.

--data built (the default): download the built training games (bsmathers/gen1ou-labels-backup: <source>_built.zip, unzipped
into WORK/<source>/built) and train. No Exeggcute checkpoints, labelling or build: the zips are the output of those stages.
--data labels: download the finished trajectories and their labels (<source>_labels.zip), then build and train (labelling
skips every game that already has a label).
--data trajectories: download the finished trajectories (bsmathers/labeled_gen1_retrajectories: one zip per source, unzipped
into WORK/<source>/trajectories), label, build and train. `--from-raw` parses raw replay logs instead (bsmathers/raw_replays).
The backup repo is private: log in once with `hf auth login`, or set HF_TOKEN.

Normally started through `launch.sh`, which installs the requirements first:

    ./launch.sh                    # the full run with the defaults below
    ./launch.sh --smoke            # a few minutes: tiny sample of every stage, prints timings and projections
    ./launch.sh --games 60000 --steps 30000 --mix human=0.5 --mix strong=0.5

Every stage is resumable: games already parsed, labelled or built are skipped, and training continues from
its last checkpoint, so a stopped run is continued by starting the same command again. Progress is plain
text in WORK/logs (one file per stage and shard) and in WORK/logs/launch.txt.

Layout under WORK: hf/ (downloaded zips), raw/<source>/ (logs), metamon-cache/ (checkpoints, usage stats),
<source>/{trajectories,labels,built}, tables/, run/ (policy stage) and run_value/ (value stage; weights_<step>.pt is the model).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import shutil
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

    SINGLE_THREAD = {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "NUMEXPR_NUM_THREADS": 1}

    def env(self, **extra) -> dict:
        env = dict(os.environ)
        env.update({"METAMON_CACHE_DIR": self.cache, "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"})
        env.update({k: str(v) for k, v in extra.items()})
        return env

    @staticmethod
    def stop(procs, grace: float = 15.0) -> None:
        """Stops the children (Ctrl-C, or one of them failed): ask, wait a little, then kill what is left. Returns when none is running."""
        for p, _ in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + grace
        try:
            for p, _ in procs:
                while p.poll() is None and time.time() < deadline:
                    time.sleep(0.1)
        except KeyboardInterrupt:  # (a second Ctrl-C: do not wait)
            pass
        for p, _ in procs:
            if p.poll() is None:
                p.kill()
                p.wait()

    def run_parallel(self, stage: str, commands: list[list[str]], envs: list[dict], progress=None) -> None:
        """Runs the commands at once, one log file each, and stops everything if one fails."""
        start = time.time()
        procs = []
        for i, (cmd, env) in enumerate(zip(commands, envs)):
            handle = open(os.path.join(self.logs, f"{stage}_{i}.txt"), "w")
            try:
                procs.append((subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
            except BlockingIOError as e:
                self.stop(procs)
                raise RuntimeError(
                    f"{stage}: the container refused to start process {i + 1} of {len(commands)} ({e}); it limits processes or memory. "
                    f"Re-run with fewer, e.g. --parse-procs 4 --label-procs 1."
                ) from e
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
            self.stop(procs)
            raise
        finally:
            for _, h in procs:
                h.close()
        bad = [i for i, (p, _) in enumerate(procs) if p.returncode != 0]
        if bad:
            raise RuntimeError(f"{stage} shard {bad[0]} failed; see {self.logs}/{stage}_{bad[0]}.txt")
        self.timings[stage] = self.timings.get(stage, 0.0) + time.time() - start

    def start_background(self, stage: str, cmd: list[str], env: dict) -> None:
        handle = open(os.path.join(self.logs, f"{stage}_0.txt"), "w")
        # low priority: labelling (the stage that matters) takes the cores it wants, the build uses what is left
        self.background.append((stage, subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(15)), handle, time.time()))
        self.log(f"{stage}: started in the background (log {self.logs}/{stage}_0.txt)")

    def finish_background(self) -> None:
        """Waits for the background stages (the builds), showing their newest log line every 30 s."""
        last = 0.0
        while any(p.poll() is None for _, p, _, _ in self.background):
            time.sleep(2)
            for stage, p, _, _ in self.background:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f"{stage} failed; see {self.logs}/{stage}_0.txt")
            if time.time() - last > 30:
                last = time.time()
                for stage, p, _, start in self.background:
                    if p.poll() is None:
                        self.log(f"{stage}: {self.newest_line(stage)} ({time.time() - start:.0f}s)")
        for stage, p, handle, start in self.background:
            handle.close()
            if p.returncode != 0:
                raise RuntimeError(f"{stage} failed; see {self.logs}/{stage}_0.txt")
            self.log(f"{stage}: done ({time.time() - start:.0f}s): {self.newest_line(stage)}")
        self.background = []

    def newest_line(self, stage: str) -> str:
        try:
            with open(os.path.join(self.logs, f"{stage}_0.txt"), "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 400))
                lines = f.read().decode("utf-8", "replace").strip().splitlines()
            return lines[-1][:150] if lines else "(no output yet)"
        except OSError:
            return "(no log yet)"

    def run_one(self, stage: str, cmd: list[str], env: dict) -> None:
        last_line = lambda: self.newest_line(stage)  # (a stage writes to its own log file: show its newest line, so a long stage is not silent)
        self.run_parallel(stage, [cmd], [env], last_line)

    # ---- stages ----

    @staticmethod
    def _read(path: str):
        try:
            return open(path).read().strip()
        except OSError:
            return None

    def usable_cpus(self) -> int:
        """CPUs this container may use: the affinity mask and the cgroup quota (os.cpu_count() reports the whole host, which
        on a shared rented machine can be 100+ cores)."""
        n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        quota = self._read("/sys/fs/cgroup/cpu.max")  # cgroup v2: "<quota> <period>" or "max <period>"
        if quota and not quota.startswith("max"):
            q, period = quota.split()[:2]
            n = min(n, max(1, int(int(q) / int(period))))
        else:
            q, period = self._read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), self._read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            if q and period and int(q) > 0:
                n = min(n, max(1, int(int(q) / int(period))))
        return max(1, n)

    def process_cap(self) -> int:
        """How many worker processes the container can really run: each parser process holds about 0.3 GiB (measured),
        and a rented container limits the number of processes and threads (forking past it fails with EAGAIN)."""
        mem = None
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            v = self._read(path)
            if v and v.isdigit() and int(v) < 2**50:
                mem = int(v) / 2**30
        if mem is None:
            try:
                mem = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 2**20
            except (OSError, StopIteration):
                mem = 16.0
        cap = max(1, int(mem // 1))  # (1 GiB each leaves room for the label and build workers)
        pids = self._read("/sys/fs/cgroup/pids.max") or self._read("/sys/fs/cgroup/pids/pids.max")
        if pids and pids.isdigit():
            cap = min(cap, max(1, (int(pids) - 100) // 24))  # a worker plus its helper threads
        return max(1, min(cap, 32))

    def hardware(self) -> None:
        import torch

        a = self.args
        self.cpus = self.usable_cpus()
        self.proc_cap = self.process_cap()
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
        self.log(f"hardware: {self.cpus} CPUs (worker cap {self.proc_cap}), " + (", ".join(f"{g['name']} {g['mem']:.0f} GiB (sm_{g['cc'][0]}{g['cc'][1]})" for g in self.gpus) or "no GPU") + f"; precision {self.precision}; {free:.0f} GiB free at {self.work}")
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
            if float(self.mix.get(name, 0)) <= 0:
                continue  # a source with no weight in the mix is not downloaded
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

    def fetch_zip(self, name: str, kind: str, dest: str) -> int:
        """Downloads <name>_<kind>.zip from --built-repo and extracts its .npz files (and nothing else) into dest, flat: the
        basename only, so no member can write outside dest. Marks dest done, so a restart skips it; the zip is deleted after."""
        a = self.args
        marker = os.path.join(dest, f".{kind}_done")
        if os.path.exists(marker):
            count = open(marker).read().strip()
            self.log(f"{name}: {kind} already extracted ({count} files)")
            return int(count.split()[0])
        from huggingface_hub import hf_hub_download

        zip_name = f"{name}_{kind}.zip"
        self.log(f"downloading {a.built_repo}/{zip_name}")
        path = hf_hub_download(repo_id=a.built_repo, repo_type="dataset", filename=zip_name, local_dir=os.path.join(self.work, "hf"))
        os.makedirs(dest, exist_ok=True)
        count = 0
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                base = os.path.basename(info.filename)
                if info.is_dir() or base.startswith(".") or not base.endswith(".npz") or info.filename.startswith("__MACOSX"):
                    continue
                with z.open(info) as src, open(os.path.join(dest, base), "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                count += 1
        if count == 0:
            raise RuntimeError(f"no .npz files in {a.built_repo}/{zip_name}")
        open(marker, "w").write(f"{count}\n")
        self.log(f"{name}: extracted {count} {kind} files")
        os.remove(path)  # the zip is no longer needed
        return count

    def count_files(self, directory: str, suffix: str) -> int:
        return sum(1 for n in os.listdir(directory) if n.endswith(suffix)) if os.path.isdir(directory) else 0

    def fetch_trajectories(self, name: str, games: int) -> None:
        """Gets the finished trajectories (corrected + original-parser states, see merge_legacy.py) of one source: from a
        zip in the dataset repo, or from a local folder (--traj-human-dir / --traj-strong-dir). With `games`, keeps every
        k-th game (both points of view). Files labelled in the zip (.npz) are kept too, so the label stage skips them."""
        a = self.args
        src = os.path.join(self.work, name)
        traj, labels = os.path.join(src, "trajectories"), os.path.join(src, "labels")
        marker = os.path.join(src, ".trajectories_done")
        if os.path.exists(marker):
            self.log(f"{name}: trajectories already extracted ({open(marker).read().strip()})")
            return
        local = {"human": a.traj_human_dir, "strong": a.traj_strong_dir}[name]
        opened = None
        if local:
            members = [(f, os.path.join(d, f)) for d, _, fs in os.walk(local) for f in fs]
            read = lambda ref: open(ref, "rb")
        else:
            from huggingface_hub import HfApi, hf_hub_download

            zip_name = {"human": a.traj_human_zip, "strong": a.traj_strong_zip}[name]
            if not zip_name:
                files = [f for f in HfApi().list_repo_files(a.traj_repo, repo_type="dataset") if f.endswith(".zip")]
                hits = [f for f in files if name in os.path.basename(f).lower()]
                if len(hits) != 1:
                    raise RuntimeError(f"cannot pick the {name} zip in {a.traj_repo} (zips there: {files}); pass --traj-{name}-zip")
                zip_name = hits[0]
            self.log(f"downloading {a.traj_repo}/{zip_name}")
            path = hf_hub_download(repo_id=a.traj_repo, repo_type="dataset", filename=zip_name, local_dir=os.path.join(self.work, "hf"))
            opened = zipfile.ZipFile(path)
            members = [(os.path.basename(m), m) for m in opened.namelist() if not m.endswith("/") and not m.startswith("__MACOSX")]
            read = opened.open
        members = [(b, ref) for b, ref in members if not b.startswith(".") and (b.endswith(".json.lz4") or b.endswith(".npz"))]
        game = lambda b: b.split("_")[0]
        ids = sorted({game(b) for b, _ in members if b.endswith(".json.lz4")})
        if not ids:
            raise RuntimeError(f"no .json.lz4 trajectories found for {name} ({'folder ' + local if local else a.traj_repo})")
        every = max(1, len(ids) // games) if games else 1
        keep = set(ids[::every])
        self.log(f"{name}: {len(ids)} games in the dataset, taking every {every}th (about {len(keep)} games, two points of view each)")
        os.makedirs(traj, exist_ok=True)
        os.makedirs(labels, exist_ok=True)
        count = 0
        for b, ref in members:
            if game(b) not in keep:
                continue
            out = os.path.join(traj if b.endswith(".lz4") else labels, b)  # the basename only: no path tricks
            with read(ref) as f, open(out, "wb") as o:
                o.write(f.read())
            count += b.endswith(".lz4")
        if opened is not None:
            opened.close()
            os.remove(opened.filename)  # the zip is no longer needed
        self.log(f"{name}: extracted {count} trajectories")
        open(marker, "w").write(f"{count} trajectories, every {every}th game\n")

    def parse_raw(self, name: str, raw_dir: str, games: int, src: str, traj: str):
        """--from-raw: parse the .log files here (corrected parser + original parser + merge)."""
        a = self.args
        total = self.count_files(raw_dir, ".log")
        if total == 0:
            raise RuntimeError(f"no .log files in {raw_dir}")
        every = max(1, total // games) if games else 1
        wanted = total // every
        self.log(f"{name}: {total} logs, taking every {every}th (about {wanted} games, two points of view each)")
        # parse, in parallel shards
        procs = a.parse_procs or min(self.cpus, self.proc_cap)
        stage = f"parse_{name}"
        cmds = [[self.py, "-m", "metamon.recurrent.parse", raw_dir, traj, "--every", str(every), "--shard", f"{i}/{procs}"] for i in range(procs)]
        self.run_parallel(stage, cmds, [self.env(HF_HUB_OFFLINE=1, **self.SINGLE_THREAD)] * procs, lambda: f"{self.count_files(traj, '.lz4')} trajectories")

        # the original parser's states and actions, which Exeggcute reads (it was trained on them), folded into each trajectory
        legacy = os.path.join(src, "legacy")
        cmds = [[self.py, os.path.join("tools", "legacy_parse.py"), raw_dir, legacy, "--every", str(every), "--shard", f"{i}/{procs}"] for i in range(procs)]
        self.run_parallel(f"legacy_{name}", cmds, [self.env(HF_HUB_OFFLINE=1, **self.SINGLE_THREAD)] * procs, lambda: f"{self.count_files(legacy, '.lz4')} legacy trajectories")
        cmds = [[self.py, "-m", "metamon.recurrent.merge_legacy", traj, legacy, "--shard", f"{i}/{procs}"] for i in range(procs)]
        self.run_parallel(f"merge_{name}", cmds, [self.env(HF_HUB_OFFLINE=1, **self.SINGLE_THREAD)] * procs)
        n_traj = self.count_files(traj, ".lz4")
        self.log(f"{name}: parsed {n_traj} trajectories (games the two parsers disagree on are in {traj}_rejected)")

        return n_traj, wanted

    def prepare(self, name: str, raw_dir: str | None, games: int) -> None:
        a = self.args
        src = os.path.join(self.work, name)
        traj, labels, built = (os.path.join(src, d) for d in ("trajectories", "labels", "built"))

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

        if raw_dir is None:
            # finished trajectories (made with this parser version), no parsing here
            self.fetch_trajectories(name, games)
            if a.data == "labels":
                # labels made from these trajectories: the label stage then finds every game done (labels of games not kept are
                # unused: the build only reads labels of trajectories present)
                self.fetch_zip(name, "labels", labels)
            n_traj = self.count_files(traj, ".lz4")
            wanted = n_traj // 2
        else:
            n_traj, wanted = self.parse_raw(name, raw_dir, games, src, traj)

        # label, several processes per GPU. The members run once per game as a few big GPU calls; what is left is the router's per-game Python and
        # building observations, one core each, and measured on a 3090 the card sat idle most of the time with 3 processes. So the count is set by
        # cores (hyperthreads count half) and by memory: a process holds ~1 GiB of weights plus activations (one process reached 5.9 GiB at 16384 rows; 8192 rows is a guess at about half) and the
        # caching allocator grows with the longest game it meets (expandable segments and a cache release after each batch keep that in check).
        # A 24 GiB card gets 5 with 10 or more CPUs. Still a guess: the GPU memory in nvidia-smi and the labelled count over a few minutes tell.
        gpus = max(1, len(self.gpus))
        per_gpu = a.label_procs or max(1, min(self.cpus // 2 // gpus, int((self.gpus[0]["mem"] - 2) // 4))) if self.gpus else 1
        shards = gpus * per_gpu
        threads = max(1, self.cpus // shards) if not self.gpus else max(1, min(2, self.cpus // shards))
        stage = f"label_{name}"
        cmds, envs = [], []
        for k in range(shards):
            cmds.append([self.py, "-m", "metamon.recurrent.label", traj, labels, "--cache", self.cache, "--batch", str(a.label_batch or (256 if self.gpus else 64)),
                         "--device", self.device, "--threads", str(threads), "--precision", self.precision if self.gpus else "fp32", "--shard", f"{k}/{shards}"])
            envs.append(self.env(HF_HUB_OFFLINE=1, **({"CUDA_VISIBLE_DEVICES": k % gpus, "LABEL_SEQUENCE_ROWS": a.label_rows or 8192, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} if self.gpus else {})))
        self.run_parallel(stage, cmds, envs, lambda: f"{self.count_files(labels, '.npz')} of {n_traj} labelled")
        self.log(f"{name}: labelled {self.count_files(labels, '.npz')} trajectories")

        # build. It only reads this source's labels, so it runs in the background while the next source downloads and labels (the GPU
        # is the idle part of labelling, the CPU the busy one: the build slows labelling a little but the two together finish sooner).
        stage = f"build_{name}"
        if a.smoke or self.args.no_overlap:
            self.run_one(stage, [self.py, "-m", "metamon.recurrent.build", traj, labels, built, os.path.join(self.work, "tables"), "--workers", str(min(self.cpus, self.proc_cap)), "--bc-players", a.bc_players], self.env(HF_HUB_OFFLINE=1, **self.SINGLE_THREAD))
            self.log(f"{name}: built {self.count_files(built, '.npz')} games")
        else:
            self.start_background(stage, [self.py, "-m", "metamon.recurrent.build", traj, labels, built, os.path.join(self.work, "tables"), "--workers", str(max(2, self.cpus // 2)), "--bc-players", a.bc_players], self.env(HF_HUB_OFFLINE=1, **self.SINGLE_THREAD))
        self.counts = getattr(self, "counts", {})
        self.counts[name] = (wanted, n_traj)

    def train(self) -> None:
        """Two stages: the policy (distillation and imitation; the win head gets no gradient), then the win head, with the backbone held
        near the first stage's policy by a KL penalty (--anchor-weight). --value-steps 0 trains both at once in a single stage."""
        a = self.args
        out = os.path.join(self.work, a.policy_run)
        if a.value_steps <= 0:
            self.train_stage(out, a.steps, ["--stage", "joint"])
            return
        self.train_stage(out, a.steps, ["--stage", "policy"])
        weights = os.path.join(out, "weights_final.pt")
        if not os.path.exists(weights):
            raise RuntimeError(f"the policy stage left no {weights}")
        self.train_stage(os.path.join(self.work, a.value_run), a.value_steps, ["--stage", "value", "--init", weights, "--anchor-weight", str(a.anchor_weight), "--lr", str(a.value_lr)])

    def train_stage(self, out: str, steps: int, extra: list[str]) -> None:
        a = self.args
        cmd = [self.py, "-m", "metamon.recurrent.train", out]
        for name, weight in self.mix.items():
            cmd += ["--source", f"{name}={os.path.join(self.work, name, 'built')}:{weight}"]
        cmd += ["--steps", str(steps), "--batch", str(a.batch), "--workers", str(min(self.cpus, 12)), "--device", self.device,
                "--precision", self.precision if self.gpus else "fp32", "--target", "blend", "--alpha", str(a.alpha), "--bc-weight", str(a.bc_weight), "--lr", str(a.lr), "--ckpt-every", str(a.ckpt_every),
                "--val-every", str(a.val_every), "--save-every", str(a.save_every), "--val-fraction", str(a.val_fraction), "--resume"]
        if a.val_batches:
            cmd += ["--val-batches", str(a.val_batches)]
        cmd += ["--patience", str(a.patience), "--min-delta", str(a.min_delta)]
        if a.init and extra[1] != "value":
            cmd += ["--init", a.init]  # (the first stage starts from these weights; a later restart of the same command resumes from the run's own latest.pt instead)
        cmd += extra  # (later flags win: the value stage's --lr replaces the default one)
        stage = extra[1]
        self.log(f"training ({stage} stage) {steps} steps; progress in {out}/log.txt")
        self.run_one("train" if stage != "value" else "train_value", cmd, self.env(HF_HUB_OFFLINE=1))

    def summary(self) -> None:
        a = self.args
        out = os.path.join(self.work, a.value_run if a.value_steps > 0 else a.policy_run)
        weights = sorted((f for f in os.listdir(out) if f.startswith("weights_") and f.split("_")[1].split(".")[0].isdigit()), key=lambda f: int(f.split("_")[1].split(".")[0])) if os.path.isdir(out) else []
        self.log("stage times: " + ", ".join(f"{k} {v:.0f}s" for k, v in self.timings.items()))
        if weights:
            self.log(f"final weights: {os.path.join(out, 'weights_final.pt')}")
        for line in open(os.path.join(out, "log.txt")).read().splitlines()[-6:] if os.path.exists(os.path.join(out, "log.txt")) else []:
            self.log("  " + line)
        if a.smoke:
            self.project()

    def project(self) -> None:
        """Scales the smoke timings up to the default sizes. Rough: the startup cost of each stage is counted
        at full size, so the projection errs high."""
        a = self.args
        sources = [n for n in SOURCES if n in self.counts]
        scale_games = a.full_games * len(sources) / max(1, sum(w for w, _ in self.counts.values()))
        # each stage's fixed startup (loading the ensemble takes about a minute per label process) happens once, not per game
        startup = {"label": 60.0, "parse": 5.0, "legacy": 5.0, "merge": 1.0, "build": 3.0}
        parts = []
        for stage in ("parse", "legacy", "merge", "label", "build"):
            t = sum(v for k, v in self.timings.items() if k.startswith(stage + "_"))
            work = max(t - startup[stage] * len(sources), 0.3 * t)
            parts.append((stage, work * scale_games))
        t_train = self.timings.get("train", 0.0) + self.timings.get("train_value", 0.0)
        parts.append((f"train ({a.full_steps} steps)", t_train / max(1, a.steps + a.value_steps) * (a.full_steps + a.value_steps)))
        self.log(f"projected for --games {a.full_games} per source and {a.full_steps} steps (hours, rough): " + ", ".join(f"{k} {v / 3600:.1f}" for k, v in parts) + f"; total {sum(v for _, v in parts) / 3600:.1f}")

    def main(self) -> None:
        a = self.args
        self.hardware()
        os.makedirs(os.path.join(self.work, "tables"), exist_ok=True)
        if not os.path.exists(os.path.join(self.work, "tables", "species_ids.json")):
            self.run_one("tables", [self.py, "-m", "metamon.recurrent.tables", os.path.join(self.work, "tables")], self.env())
        self.mix = dict(m.split("=") for m in a.mix) if a.mix else {"human": 0.4, "strong": 0.6}
        self.background = []
        if a.data == "built" and not a.train_only and not (a.from_raw or a.raw_human or a.raw_strong or a.traj_human_dir or a.traj_strong_dir):
            for name, weight in self.mix.items():
                if float(weight) > 0:
                    self.fetch_zip(name, "built", os.path.join(self.work, name, "built"))
            a.train_only = True  # (the built games are the data: nothing to label or build)
        if a.train_only:  # (the built games of an earlier run are used as they are: no download, labelling or build)
            for name, weight in self.mix.items():
                built = os.path.join(self.work, name, "built")
                if float(weight) > 0 and not (os.path.isdir(built) and any(f.endswith(".npz") for f in os.listdir(built))):
                    raise SystemExit(f"--train-only: no built games in {built}")
        else:
            self.prefetch()
            raw = self.fetch_data() if a.from_raw or a.raw_human or a.raw_strong else {}
            try:
                for name in SOURCES:
                    if float(self.mix.get(name, 0)) > 0:
                        self.prepare(name, raw.get(name), a.games)
                self.finish_background()
            except BaseException:
                self.stop([(p, h) for _, p, h, _ in self.background])  # (Ctrl-C or a failure: no build is left running)
                raise
        self.mix = {k: v for k, v in self.mix.items() if float(v) > 0}
        self.train()
        self.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.environ.get("WORK", "/workspace/work" if os.path.isdir("/workspace") else os.path.expanduser("~/work")))
    ap.add_argument("--cache", default=None, help="metamon cache (default WORK/metamon-cache)")
    ap.add_argument("--data", choices=("built", "labels", "trajectories"), default="built",
                    help="what to download: built games (train at once), trajectories + labels (build, then train) or trajectories (label, build, train)")
    ap.add_argument("--built-repo", default="bsmathers/gen1ou-labels-backup", help="dataset repo with <source>_built.zip and <source>_labels.zip (private: hf auth login or HF_TOKEN)")
    ap.add_argument("--traj-repo", default="bsmathers/labeled_gen1_retrajectories", help="dataset repo with the finished trajectories (one zip per source)")
    ap.add_argument("--traj-human-zip", default=None, help="zip name in --traj-repo (default: the only zip with 'human' in its name)")
    ap.add_argument("--traj-strong-zip", default=None, help="zip name in --traj-repo (default: the only zip with 'strong' in its name)")
    ap.add_argument("--traj-human-dir", default=None, help="a local folder of finished trajectories, instead of downloading")
    ap.add_argument("--traj-strong-dir", default=None)
    ap.add_argument("--from-raw", action="store_true", help="parse raw replay logs here instead of using finished trajectories")
    ap.add_argument("--data-repo", default="bsmathers/raw_replays", help="(--from-raw) repo with the raw logs")
    ap.add_argument("--human-zip", default="human_logs.zip")
    ap.add_argument("--strong-zip", default="strong_model_replays.zip")
    ap.add_argument("--raw-human", default=None, help="a folder of .log files, instead of downloading")
    ap.add_argument("--raw-strong", default=None)
    ap.add_argument("--games", type=int, default=30000, help="games to take from each source (0: all). --data built always uses every built game")
    ap.add_argument("--mix", action="append", help="name=weight, e.g. human=0.4 (default human=0.4 strong=0.6; weight 0 skips a source)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", default="auto", help="training batch: a number or auto (largest that fits in GPU memory)")
    ap.add_argument("--value-steps", type=int, default=4000, help="steps of the second stage that trains the win head with the backbone anchored (0: one joint stage)")
    ap.add_argument("--patience", type=int, default=10, help="stop a stage after this many validations without progress (0: run all its steps)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="the smallest drop of the validation loss that counts as progress (KL in the first stage, win BCE in the second)")
    ap.add_argument("--train-only", action="store_true", help="skip every data stage and train on the games already built under WORK")
    ap.add_argument("--init", default=None, help="weights or a checkpoint (e.g. an earlier run's latest.pt) that the first training stage starts from")
    ap.add_argument("--policy-run", default="run", help="folder under WORK for the first stage's checkpoints and log")
    ap.add_argument("--value-run", default="run_value", help="folder under WORK for the second stage's")
    ap.add_argument("--anchor-weight", type=float, default=5.0, help="(second stage) weight of KL(first-stage policy || current policy)")
    ap.add_argument("--value-lr", type=float, default=3e-4, help="(second stage) learning rate")
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
    ap.add_argument("--label-procs", type=int, default=0, help="labelling processes per GPU (default: 5 on a 24 GiB card with 10+ CPUs, fewer with less memory or CPUs)")
    ap.add_argument("--label-rows", type=int, default=0, help="game-turns per parallel forward on a GPU (default 8192; lower it if a card runs out of memory)")
    ap.add_argument("--label-batch", type=int, default=0, help="games labelled in lockstep per process (default 256 on a GPU, 64 on a CPU: the lanes of a lockstep batch that finished early still cost their share of the math, and a CPU gains little from larger batches)")
    ap.add_argument("--no-overlap", action="store_true", help="build a source after labelling it, instead of in the background while the next source is prepared")
    ap.add_argument("--smoke", action="store_true", help="tiny sample of every stage, prints timings and a projection")
    args = ap.parse_args()
    args.full_games, args.full_steps = args.games, args.steps  # (the smoke run shrinks both; the projection scales back up)
    if args.smoke:
        args.cache = args.cache or os.path.join(args.work, "metamon-cache")  # shared with the full run
        args.games = min(args.games, 150)
        args.steps, args.val_every, args.save_every, args.val_batches = min(args.steps, 150), 75, 150, 20
        args.value_steps = min(args.value_steps, 75)
        args.val_fraction = min(args.val_fraction, 5)
        args.work = args.work.rstrip("/") + "_smoke"
    import signal

    def interrupt(*_):  # (Ctrl-C and `kill`/`pkill` both stop the workers: the stage loop handles the interrupt)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)  # (also when started in the background, where SIGINT begins ignored)
    signal.signal(signal.SIGTERM, interrupt)
    Launcher(args).main()


if __name__ == "__main__":
    main()
