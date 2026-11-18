"""Runs the whole job on one machine: fetch the data, then (depending on --data) label, build, train.

--data labels (the default): download the finished trajectories and their labels (<source>_labels.zip), then build and train
(labelling skips every game that already has a label).
--data built: download the built training games (bsmathers/gen1ou-labels-backup: <source>_built.zip, unzipped into
WORK/<source>/built) and train. No Exeggcute checkpoints, labelling or build: the zips are the output of those stages. Only
zips built with the current features.ENCODING_VERSION can be trained on (the ones uploaded before it are refused).
--data trajectories: download the finished trajectories (bsmathers/labeled_gen1_retrajectories: one zip per source, unzipped
into WORK/<source>/trajectories), label, build and train. `--from-raw` parses raw replay logs instead (bsmathers/raw_replays).
Both dataset repos are public: no Hugging Face token is needed.

Normally started through `launch.sh`, which installs the requirements first:

    ./launch.sh                    # the full run with the defaults below
    ./launch.sh --smoke            # a few minutes: tiny sample of every stage, prints timings and projections
    ./launch.sh --games 60000 --steps 30000 --mix human=0.5 --mix strong=0.5

Checks (each one stops the run when it fails; --skip-checks turns the last two off):
    launch.sh   network (Hugging Face API and a 16 MB test download from its file storage, PyPI and GitHub before the install),
                and that this checkout is not behind origin
    preflight   the dataset repos can be listed and hold every zip this run downloads
    labels      (--data labels, after the download, before any build) at most --max-unlabelled of the trajectories lack a label,
                and random whole games per source (--check-decisions decisions), relabelled here by Exeggcute's own code (verify_labels reference,
                CPU, fp32), match the downloaded labels within bf16 noise (verify_labels compare --preset bf16)
    built       (after the build, before training) --check-built-games random built games per source match their labels and
                trajectories by option name, with none of the build's code (verify_built)
--check-only stops after the built check, before training. A failed check exits with code 3 (launch.sh does not restart then).

In the background (SideJobs; a failure there is reported and the run goes on):
    tournaments  every --tournament-every steps of each training stage, --tournament-games games of that checkpoint against
                 TaurosV0 on the Zig duel harness (tournament.py; set up under WORK/duel while the data stages run). Each result
                 is a TOURNAMENT line here and in WORK/tournaments/results.txt
    upload       with HF_WRITE_TOKEN set, after the built check: the built games, one zip per source, to the public dataset repo
                 --upload-repo (upload_built.py), in the layout --data built downloads

Every stage is resumable: games already parsed, labelled or built are skipped, training continues from its latest.pt (written
every step; weight files are written to a temporary name and renamed, so a crash never leaves a truncated one), uploads and
tournaments already done are skipped. So a stopped run is continued by starting the same command again, and launch.sh does
that by itself after a crash. Progress is plain text in WORK/logs (one file per stage and shard) and in WORK/logs/launch.txt.

Layout under WORK: hf/ (downloaded zips), raw/<source>/ (logs), metamon-cache/ (checkpoints, usage stats),
<source>/{trajectories,labels,built}, tables/, run_v3/ (policy stage) and run_v3_value/ (value stage; weights_final.pt is the model),
duel/ (the tournaments' harness), tournaments/, upload/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import shutil
import time
import zipfile

SOURCES = ("human", "strong")
ENCODING_STAMP = ".encoding_version"  # (features.ENCODING_STAMP; features is not imported here before the install is known to work)


class SideJobs:
    """What runs beside the main stages, at low priority, and never stops the run when it fails (it is reported instead):
      - the tournament setup (tournament.py setup), started at the beginning;
      - the upload of the built games (upload_built.py), started after the built check;
      - a tournament of --tournament-games games against TaurosV0 for every weights_<step>.pt with step a multiple of
        --tournament-every that training writes, one at a time, oldest first. Results go to the terminal and to
        WORK/tournaments/results.txt; WORK/tournaments/<run>_<step>.json marks a checkpoint as played (a restart skips it)."""

    def __init__(self, launcher):
        self.l = launcher
        self.a = launcher.args
        self.dir = os.path.join(launcher.work, "tournaments")
        os.makedirs(self.dir, exist_ok=True)
        self.setup = None  # (Popen, log handle)
        self.ready = False
        self.upload = None
        self.match = None  # (Popen, log handle, run name, step, result path)
        self.watch: list[tuple[str, str]] = []  # (run name, folder) of the training stages so far

    def popen(self, cmd: list[str], log: str):
        handle = open(os.path.join(self.l.logs, log), "a")
        p = subprocess.Popen(cmd, env=self.l.env(), stdout=handle, stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(10))
        return p, handle

    def start_setup(self) -> None:
        if self.a.tournament_every <= 0 and not self.a.experiment:
            self.l.log("tournaments: off (--tournament-every 0)")
            return
        if self.a.reuse_tournament_setup and os.path.exists(os.path.join(self.l.work, "duel", ".ready")):
            # (several launchers at once: one setup made beforehand, not redone by each; a setup rebuilds the shared harness)
            self.ready = True
            self.l.log("tournaments: ready (the existing setup in WORK/duel)")
            return
        self.setup = self.popen([self.l.py, "-m", "metamon.recurrent.tournament", "setup", "--work", self.l.work, "--cache", self.l.cache], "tournament_setup.txt")
        self.l.log(f"tournaments: setting up the duel harness and TaurosV0 in the background (log {self.l.logs}/tournament_setup.txt)")

    def start_upload(self, sources: list[str]) -> None:
        if self.a.smoke:
            self.l.log("upload: skipped in a smoke run")
            return
        if not os.environ.get("HF_WRITE_TOKEN"):
            self.l.log("upload: HF_WRITE_TOKEN is not set, so the built games are not uploaded (set it and run again to upload them)")
            return
        self.upload = self.popen([self.l.py, "-m", "metamon.recurrent.upload_built", self.l.work, self.a.upload_repo, *sources], "upload_built.txt")
        self.l.log(f"upload: zipping and uploading the built games in the background (log {self.l.logs}/upload_built.txt)")

    def last_line(self, log: str) -> str:
        return self.l.newest_line_of(log)[1:-1]

    def poll(self) -> None:
        try:
            self._poll()
        except Exception as e:  # noqa: BLE001  (a side job never stops the run)
            self.l.log(f"background jobs: {type(e).__name__}: {e} (the run goes on)")

    def _poll(self) -> None:
        if self.setup and self.setup[0].poll() is not None:
            self.setup[1].close()
            self.ready = os.path.exists(os.path.join(self.l.work, "duel", ".ready"))
            self.l.log("tournaments: ready" if self.ready else
                       f"tournaments OFF: the setup failed ({self.last_line('tournament_setup')}; see {self.l.logs}/tournament_setup.txt). Training goes on")
            self.setup = None
        if self.upload and self.upload[0].poll() is not None:
            self.upload[1].close()
            code = self.upload[0].returncode
            self.l.log(f"upload: {'done' if code == 0 else 'FAILED (training goes on; run again to retry)'}: {self.last_line('upload_built')}")
            self.upload = None
        if self.match and self.match[0].poll() is not None:
            p, handle, run, step, result = self.match
            handle.close()
            self.match = None
            from metamon.recurrent.tournament import describe

            if p.returncode == 0 and os.path.exists(result):
                line = f"TOURNAMENT {run} step {step} vs TaurosV0: {describe(json.load(open(result)))}"
            else:
                line = f"TOURNAMENT {run} step {step}: FAILED ({self.last_line(f'tournament_{run}_{step}')}; see {self.l.logs}/tournament_{run}_{step}.txt)"
                with open(result, "w") as f:  # (not retried: the same checkpoint would fail the same way)
                    json.dump({"error": line}, f)
            self.l.log(line)
            with open(os.path.join(self.dir, "results.txt"), "a") as f:
                f.write(f"{time.strftime('%F %T')} {line}\n")
        if self.ready and self.match is None:
            pending = self.pending()
            if pending:
                run, step, weights = pending[0]
                result = os.path.join(self.dir, f"{run}_{step}.json")
                procs = self.a.tournament_procs or max(1, min(4, self.l.cpus // 4))
                cmd = [self.l.py, "-m", "metamon.recurrent.tournament", "play", weights, "--work", self.l.work, "--cache", self.l.cache,
                       "--games", str(self.a.tournament_games), "--procs", str(procs), "--device", self.a.tournament_device,
                       "--temperature", str(self.a.tournament_temperature), "--out", result,
                       "--log-dir", os.path.join(self.l.logs, f"tournament_{run}_{step}")]
                p, handle = self.popen(cmd, f"tournament_{run}_{step}.txt")
                self.match = (p, handle, run, step, result)
                self.l.log(f"tournament {run} step {step}: {self.a.tournament_games} games vs TaurosV0 started ({procs} processes)")

    def pending(self) -> list:
        out = []
        for order, (run, folder) in enumerate(self.watch):
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                m = re.fullmatch(r"weights_(\d+)\.pt", f)
                if m and int(m.group(1)) > 0 and int(m.group(1)) % self.a.tournament_every == 0 \
                        and not os.path.exists(os.path.join(self.dir, f"{run}_{m.group(1)}.json")):
                    out.append((order, int(m.group(1)), run, os.path.join(folder, f)))
        return [(run, step, path) for _, step, run, path in sorted(out)]

    def busy(self) -> bool:
        return bool(self.setup or self.upload or self.match or (self.ready and self.pending()))

    def drain(self) -> None:
        """At the end: waits for the upload and for the tournaments of the checkpoints training wrote."""
        last = 0.0
        while self.busy():
            if time.time() - last > 120:
                last = time.time()
                what = [w for w, on in (("the tournament setup", self.setup), ("the upload", self.upload), ("a tournament", self.match)) if on]
                self.l.log(f"waiting for {', '.join(what) or 'the next tournament'} ({len(self.pending()) if self.ready else 0} checkpoints left to play)")
            self.poll()
            time.sleep(5)

    def stop(self) -> None:
        jobs = [j for j in (self.setup, self.upload, self.match) if j]
        self.l.stop([(j[0], j[1]) for j in jobs])
        for j in jobs:
            j[1].close()


class Launcher:
    def __init__(self, args):
        self.args = args
        self.work = os.path.abspath(args.work)
        self.logs = os.path.join(self.work, args.logs_name)
        os.makedirs(self.logs, exist_ok=True)
        self.cache = os.path.abspath(args.cache or os.path.join(self.work, "metamon-cache"))
        self.timings: dict[str, float] = {}
        self.dataset_games: dict[str, int] = {}  # games per source in the downloaded dataset (for the smoke run's projection)
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
                if getattr(self, "side", None):
                    self.side.poll()  # (tournaments, the upload: started, reported)
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
            if getattr(self, "side", None):
                self.side.poll()
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

    CHECK_FAILED = 3  # exit code of a failed data check: launch.sh does not restart the run after it (it would fail again)

    def check_failed(self, text: str) -> None:
        self.log(text)
        raise SystemExit(self.CHECK_FAILED)

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
        total = __import__("shutil").disk_usage(self.work).total / 2**30
        if total < 25 and not a.smoke:
            raise SystemExit(f"WORK {self.work} is on a {total:.0f} GiB disk (a vast.ai volume?): the data needs the container disk. "
                             f"Use the default WORK (~/work) or another folder on the big disk")
        if free < 40 and not a.smoke and a.data != "built":
            self.log("warning: under 40 GiB free; the default sample needs about 25 GiB with the downloads")
        self.backup = self.find_backup()

    def find_backup(self) -> str | None:
        """--backup-dir auto: the vast.ai local volume, which the console mounts at /data (a template or `-v V.<id>:/path`
        can mount it elsewhere: pass that path). Only a separate mount counts: a /data folder on the container disk
        would vanish with it."""
        a = self.args
        if a.backup_dir == "none":
            self.log("backups off (--backup-dir none): checkpoints are only on the container disk")
            return None
        if a.backup_dir == "auto":
            path = next((p for p in ("/data", "/workspace") if os.path.ismount(p)), None)
            if path is None:
                self.log("WARNING: no local volume mounted at /data or /workspace, so checkpoints are only on the container disk. "
                         "Attach a vast.ai local volume or pass --backup-dir PATH")
                return None
        else:
            path = a.backup_dir
        os.makedirs(path, exist_ok=True)
        if os.stat(path).st_dev == os.stat(self.work).st_dev:
            self.log(f"WARNING: --backup-dir {path} is on the same disk as {self.work}: it will not survive losing that disk")
        root = os.path.join(path, "metamon-backup")
        os.makedirs(root, exist_ok=True)
        usage = __import__("shutil").disk_usage(root)
        self.log(f"backups to {root} (volume {usage.total / 2**30:.0f} GiB, {usage.free / 2**30:.1f} GiB free): every {a.backup_every} steps "
                 f"weights_<step>.pt (~5 MiB each) and latest.pt (overwritten), per stage")
        need = (a.steps + a.value_steps) / max(a.backup_every, 1) * 5.3 / 1024 + 0.1
        if need > usage.free / 2**30:
            self.log(f"warning: the numbered weights need about {need:.1f} GiB; when the volume is nearly full they stop and latest.pt goes on")
        return root

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
        zip_name = f"{name}_{kind}.zip"
        path = os.path.join(self.work, "hf", zip_name)
        if os.path.exists(path):
            # copied here by hand (e.g. scp from a machine that can read the private repo): no download, no token
            self.log(f"{name}: using {path}")
        else:
            from huggingface_hub import hf_hub_download

            self.log(f"downloading {a.built_repo}/{zip_name}")
            path = hf_hub_download(repo_id=a.built_repo, repo_type="dataset", filename=zip_name, local_dir=os.path.join(self.work, "hf"))
        os.makedirs(dest, exist_ok=True)
        count = 0
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                base = os.path.basename(info.filename)
                stamp = kind == "built" and base == ENCODING_STAMP  # (upload_built.py puts the folder's encoding stamp in the zip)
                if not stamp and (info.is_dir() or base.startswith(".") or not base.endswith(".npz") or info.filename.startswith("__MACOSX")):
                    continue
                with z.open(info) as src, open(os.path.join(dest, base), "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                count += not stamp
        if count == 0:
            raise RuntimeError(f"no .npz files in {a.built_repo}/{zip_name}")
        if kind == "built" and not os.path.exists(os.path.join(dest, ENCODING_STAMP)):
            self.log(f"{name}: {zip_name} has no {ENCODING_STAMP} stamp (zipped before upload_built.py): training will refuse it")
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
            note = open(marker).read().strip()
            self.log(f"{name}: trajectories already extracted ({note})")
            if " of " in note:
                self.dataset_games[name] = int(note.rsplit(" of ", 1)[1])
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
        open(marker, "w").write(f"{count} trajectories, every {every}th game of {len(ids)}\n")
        self.dataset_games[name] = len(ids)

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

    def preflight(self) -> None:
        """Before any download: the repos are reachable with this token and hold the files this run needs (a missing file or a bad
        token otherwise shows up an hour in, after the first source's download)."""
        a = self.args
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            user = api.whoami().get("name", "?")
        except Exception as e:
            user = None
            self.log(f"preflight: not logged in to Hugging Face ({type(e).__name__}); fine for public repos only")
        needed = []  # (repo, file or None for 'a zip with the source in its name')
        sources = [n for n in SOURCES if float(self.mix.get(n, 0)) > 0]
        if a.data == "built":
            needed += [(a.built_repo, f"{n}_built.zip") for n in sources]
        elif not (a.from_raw or a.raw_human or a.raw_strong):
            for n in sources:  # (what an earlier start of this run already extracted is not needed again)
                if not {"human": a.traj_human_dir, "strong": a.traj_strong_dir}[n] and not os.path.exists(os.path.join(self.work, n, ".trajectories_done")):
                    needed.append((a.traj_repo, {"human": a.traj_human_zip, "strong": a.traj_strong_zip}[n]))
                if a.data == "labels" and not os.path.exists(os.path.join(self.work, "hf", f"{n}_labels.zip")) \
                        and not os.path.exists(os.path.join(self.work, n, "labels", ".labels_done")):
                    needed.append((a.built_repo, f"{n}_labels.zip"))
        listed = {}
        for repo, f in needed:
            if repo not in listed:
                try:
                    listed[repo] = set(api.list_repo_files(repo, repo_type="dataset"))
                except Exception as e:
                    raise SystemExit(f"preflight: cannot list the dataset repo {repo} ({type(e).__name__}: {e}). "
                                     f"Check HF_TOKEN (logged in as {user}) and the network") from e
            if f is not None and f not in listed[repo]:
                self.check_failed(f"preflight: {repo} has no {f} (it has {sorted(x for x in listed[repo] if x.endswith('.zip'))})")
        self.log(f"preflight: Hugging Face user {user}; " + ("found " + ", ".join(f"{r}/{f or '<zip per source>'}" for r, f in needed) if needed else "nothing left to download"))

    def check_labels(self) -> None:
        """--data labels: the downloaded labels are trained on as they are, so before the build: (1) nearly every trajectory has one
        (the label stage would label the rest on the GPU), (2) a random sample of games, relabelled here by metamon's own ensemble
        step by step (verify_labels reference, fp32 on the CPU), matches the downloaded labels within bf16 noise (verify_labels
        compare --preset bf16): the same actions, the same probabilities on the same slots, labels of the right games."""
        a = self.args
        sources = [n for n in SOURCES if float(self.mix.get(n, 0)) > 0]
        marker = os.path.join(self.work, "check", ".labels_ok")
        if os.path.exists(marker):
            self.log(f"label check: passed before ({open(marker).read().strip()})")
            return
        import random

        import numpy as np

        cmds, envs, picked = [], [], {}
        for name in sources:
            traj, labels = os.path.join(self.work, name, "trajectories"), os.path.join(self.work, name, "labels")
            stems = sorted(f[: -len(".json.lz4")] for f in os.listdir(traj) if f.endswith(".json.lz4"))
            have = {f[: -len(".npz")] for f in os.listdir(labels) if f.endswith(".npz")} if os.path.isdir(labels) else set()
            missing = [s for s in stems if s not in have]
            self.log(f"label check {name}: {len(stems) - len(missing)} of {len(stems)} trajectories have a downloaded label")
            if len(missing) > a.max_unlabelled * len(stems):
                self.check_failed(f"label check: {len(missing)} of {len(stems)} {name} trajectories have no label (more than "
                                  f"{a.max_unlabelled:.0%}): the label zip does not belong to these trajectories, or is incomplete. "
                                  f"First missing: {missing[:3]}")
            # random games until they hold --check-decisions decisions (a fixed number of games can be a handful of short ones)
            pool = sorted(have & set(stems))
            random.Random(f"{a.check_seed}-{name}").shuffle(pool)
            sample, total = [], 0
            for s in pool:
                if total >= a.check_decisions:
                    break
                sample.append(s)
                total += len(np.load(os.path.join(labels, s + ".npz"))["chosen"])
            picked[name] = sample
            self.log(f"label check {name}: {len(sample)} random games, {total} decisions")
            ref_in, ref_out = os.path.join(self.work, "check", name, "trajectories"), os.path.join(self.work, "check", name, "reference")
            os.makedirs(ref_in, exist_ok=True)
            for s in sample:
                link = os.path.join(ref_in, s + ".json.lz4")
                if not os.path.lexists(link):
                    os.symlink(os.path.join(traj, s + ".json.lz4"), link)
            threads = max(1, self.cpus // len(sources))
            cmds.append([self.py, "-m", "metamon.recurrent.verify_labels", "reference", ref_in, ref_out, "--cache", self.cache,
                         "--device", "cpu", "--precision", "fp32", "--threads", str(threads)])
            envs.append(self.env(HF_HUB_OFFLINE=1, OMP_NUM_THREADS=threads))
        self.log(f"label check: relabelling {sum(map(len, picked.values()))} random games with Exeggcute's own code (CPU, fp32) "
                 f"to compare with the downloaded labels; progress in {self.logs}/check_labels_N.txt")
        self.run_parallel("check_labels", cmds, envs, lambda: " ".join(self.newest_line_of(f"check_labels_{i}") for i in range(len(cmds))))
        results = []
        for name in sources:
            cmd = [self.py, "-m", "metamon.recurrent.verify_labels", "compare", os.path.join(self.work, "check", name, "reference"),
                   os.path.join(self.work, name, "labels"), "--preset", "bf16", "--min-decisions", str(min(a.check_decisions, 1000))]
            r = subprocess.run(cmd, env=self.env(HF_HUB_OFFLINE=1), capture_output=True, text=True)
            text = (r.stdout + r.stderr).strip()
            with open(os.path.join(self.logs, f"check_labels_compare_{name}.txt"), "w") as f:
                f.write(text + "\n")
            for line in text.splitlines():
                if not line.startswith("  "):
                    self.log(f"label check {name}: {line}")
            if r.returncode != 0:
                reason = (text.splitlines() or ["?"])[-1]
                self.check_failed(f"label check FAILED for {name}: {reason} (details in {self.logs}/check_labels_compare_{name}.txt). "
                                  f"Stopping before the build and training")
            results.append(f"{name}: {len(picked[name])} games")
        open(marker, "w").write(", ".join(results) + "\n")
        self.log("label check: PASSED")

    def newest_line_of(self, log: str) -> str:
        try:
            with open(os.path.join(self.logs, f"{log}.txt"), "rb") as f:
                lines = f.read()[-400:].decode("utf-8", "replace").strip().splitlines()
            return f"[{lines[-1][:90]}]" if lines else "[starting]"
        except OSError:
            return "[no log]"

    def check_built(self) -> None:
        """After the build, before training: a random sample of built games against their labels and trajectories by option name,
        with none of the build's code (verify_built): the mapping from Exeggcute's actions to this model's slots, the slot meaning
        the features show, and the played action. A wrong mapping (the bug that sank an earlier run) fails here."""
        a = self.args
        for name in self.mix:
            if float(self.mix[name]) <= 0:
                continue
            src = os.path.join(self.work, name)
            cmd = [self.py, "-m", "metamon.recurrent.verify_built", os.path.join(src, "trajectories"), os.path.join(src, "labels"),
                   os.path.join(src, "built"), os.path.join(self.work, "tables"), "--games", str(a.check_built_games), "--seed", str(a.check_seed)]
            r = subprocess.run(cmd, env=self.env(HF_HUB_OFFLINE=1), capture_output=True, text=True)
            text = (r.stdout + r.stderr).strip()
            with open(os.path.join(self.logs, f"check_built_{name}.txt"), "w") as f:
                f.write(text + "\n")
            for line in text.splitlines()[:3] + text.splitlines()[-1:]:
                self.log(f"built check {name}: {line}")
            if r.returncode != 0:
                self.check_failed(f"built check FAILED for {name} (details in {self.logs}/check_built_{name}.txt). Stopping before training")
        self.log("built check: PASSED")

    def prepare(self, name: str, raw_dir: str | None, games: int) -> None:
        """Gets one source's trajectories (and, with --data labels, their labels) on disk."""
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
        self.counts = getattr(self, "counts", {})
        self.counts[name] = (wanted, n_traj)

    def label_and_build(self, name: str) -> None:
        """Labels what has no label yet (with --data labels, only games the label zip lacks), then builds the training games."""
        a = self.args
        src = os.path.join(self.work, name)
        traj, labels, built = (os.path.join(src, d) for d in ("trajectories", "labels", "built"))
        n_traj = self.counts[name][1]

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

    def train(self) -> None:
        """Two stages: the policy (distillation and imitation; the win head gets no gradient), then the win head, with the backbone held
        near the first stage's policy by a KL penalty (--anchor-weight). --value-steps 0 trains both at once in a single stage."""
        a = self.args
        out = os.path.join(self.work, a.policy_run)
        if a.finetune_from and a.experiment:
            self.experiment()
            return
        if a.finetune_from:
            self.finetune()
            return
        if a.value_steps <= 0:
            self.train_stage(out, a.steps, ["--stage", "joint"])
            return
        if a.value_from:
            # straight to the second stage, anchored to these policy weights (e.g. a stopped policy stage's weights_best.pt)
            weights = os.path.abspath(a.value_from)
            if not os.path.exists(weights):
                raise SystemExit(f"--value-from: no {weights}")
            self.log(f"skipping the policy stage: the value stage starts from and is anchored to {weights}")
        else:
            self.train_stage(out, a.steps, ["--stage", "policy"])
            weights = os.path.join(out, "weights_final.pt")
            if not os.path.exists(weights):
                raise RuntimeError(f"the policy stage left no {weights}")
        self.train_stage(os.path.join(self.work, a.value_run), a.value_steps, ["--stage", "value", "--init", weights, "--anchor-weight", str(a.anchor_weight),
                                                                              "--lr", str(a.value_lr), "--value-bc-weight", str(a.value_bc_weight)])

    def finetune(self) -> None:
        """--finetune-from WEIGHTS (finetune.sh): three stages after policy distillation, each starting from the one before.
          A  actor: trunk and actor head (the win head gets no gradient) on 0.5 x KL to Exeggcute's targets + 0.5 x imitation of the
             --bc-players' played moves, anchored with weight 2 to the policy of WEIGHTS; early stopping on that weighted loss. Data
             (--ft-actor-mix): only the games seen from the imitated bots' side (e.g. bots=0.5,human=0.3333,otherbots=0.1667 adds
             random subsets of human and other bot games)
          1  value head: only the win head MLP (fresh), everything else frozen, so the policy is A's
          2  everything: the win BCE, anchored with weight 1 to the policy of A (which stage 1 left unchanged)"""
        a = self.args
        start = os.path.abspath(a.finetune_from)
        if not os.path.exists(start):
            raise SystemExit(f"--finetune-from: no {start}")
        runs = [os.path.join(self.work, f"{a.finetune_run}_{n}") for n in ("actor", "value_head", "all", "joint")]
        if a.ft_skip_actor:  # (value.sh: WEIGHTS is already a fine-tuned actor)
            actor = start
            self.log(f"value stages from the actor {start}: stage 1 in {runs[1]}" + (f", stage 2 in {runs[2]}" if a.ft_all_steps > 0 else " (no stage 2)"))
        else:
            self.log(f"fine-tuning from {start}: stages in {', '.join(runs)}")
            self.train_stage(runs[0], a.ft_actor_steps, [
                "--stage", "policy", "--init", start, "--alpha", str(a.ft_kl_weight), "--bc-weight", str(a.ft_bc_weight),
                "--anchor-weight", str(a.ft_actor_anchor), "--lr", str(a.ft_lr), "--monitor", "loss"]
                + (["--rebalance", a.ft_actor_mix] if a.ft_actor_mix != "none" else []))
            actor = os.path.join(runs[0], "weights_final.pt")
        if a.ft_actor_only:
            self.log(f"--ft-actor-only: stopping after the actor stage ({actor})")
            return
        self.train_stage(runs[1], a.ft_value_head_steps, [
            "--stage", "value", "--init", actor, "--freeze", "trunk", "--anchor-weight", "0", "--lr", str(a.value_lr)])
        head = os.path.join(runs[1], "weights_final.pt")
        if a.ft_all_steps > 0:
            self.train_stage(runs[2], a.ft_all_steps, [
                "--stage", "value", "--init", head, "--keep-value-head", "--anchor-weight", str(a.ft_all_anchor), "--lr", str(a.ft_lr)])
        if a.ft_joint_steps > 0:
            # 3: everything, from stage 2 (or 1): the win BCE (0.5) + KL to Exeggcute (0.5) + imitation of the bots' played moves (1), no anchor;
            # the trained win head is kept, and early stopping follows the weighted sum of the three
            start3 = os.path.join(runs[2] if a.ft_all_steps > 0 else runs[1], "weights_final.pt")
            self.log(f"stage 3 (joint) from {start3}: win BCE {a.ft_joint_value_weight}, KL to Exeggcute {a.ft_joint_kl_weight}, imitation {a.ft_joint_bc_weight}, no anchor")
            self.train_stage(runs[3], a.ft_joint_steps, [
                "--stage", "joint", "--init", start3, "--alpha", str(a.ft_joint_kl_weight), "--bc-weight", str(a.ft_joint_bc_weight),
                "--value-weight", str(a.ft_joint_value_weight),
                "--anchor-weight", "0", "--lr", str(a.ft_lr), "--monitor", "loss"]
                + (["--rebalance", a.ft_joint_mix] if a.ft_joint_mix != "none" else []))

    EXPERIMENT_DEFAULTS = {
        # head: everything frozen but the actor: the linear actor head plus the deep actor head added on top of the state token
        # (model.DeepActor) imitate the bots' played moves with a little KL to Exeggcute against overfitting, on the bots' games
        "head": {"lr": 3e-4, "kl": 0.1, "bc": 1.0, "anchor": 0.0, "opp": 0.0, "mix": "bots=1,human=0,otherbots=0"},
        # opphead: an opponent-action head added; everything trains (but the win head): imitation of the bots on their games, KL to
        # Exeggcute on the others, the opponent head's cross-entropy on all
        "opphead": {"lr": 1e-4, "kl": 1.0, "bc": 1.0, "anchor": 0.0, "opp": 0.5, "mix": "bots=3,human=2,otherbots=1"},
    }
    DEFAULT_BLENDS = "start;start@0.5,head@0.5;start@0.5,opphead@0.5;head@0.5,opphead@0.5;start,head,opphead;start@0.5,ckpt5000@0.5"

    def experiment(self) -> None:
        """--experiment head|opphead|blend with --finetune-from START: training from START (head: --exp-pre-steps imitating Exeggcute
        first, then the bots; opphead: one stage; blend: none), then a tournament
        of --final-games games against TaurosV0 of its weights_final.pt (blend: of each --blends policy). Results: FINAL TOURNAMENT
        lines, WORK/tournaments/results.txt and WORK/tournaments/final_<name>.json (a rerun skips what is done)."""
        a = self.args
        start = os.path.abspath(a.finetune_from)
        if not os.path.exists(start):
            raise SystemExit(f"--finetune-from: no {start}")
        if a.experiment == "blend":
            self.blend_experiment(start)
            return
        d = self.EXPERIMENT_DEFAULTS[a.experiment]
        pick = lambda value, key: d[key] if value is None else value  # noqa: E731
        lr, kl, bc, anchor, opp = (pick(a.exp_lr, "lr"), pick(a.exp_kl_weight, "kl"), pick(a.exp_bc_weight, "bc"),
                                   pick(a.exp_anchor, "anchor"), pick(a.exp_opp_weight, "opp"))
        mix = a.exp_mix or d["mix"]
        out = os.path.join(self.work, a.exp_run or f"exp_{a.experiment}")
        self.log(f"experiment {a.experiment} from {start}: lr {lr}, KL to Exeggcute {kl}, imitation {bc}, anchor {anchor}, "
                 + (f"opponent head {opp}, " if a.experiment == "opphead" else "") + f"data {mix}; in {out}")
        init = start
        if a.experiment == "head" and a.exp_pre_steps > 0:
            # phase 1: the new head (on the frozen trunk) first learns Exeggcute's policy: KL only, on the distillation data
            pre = out + "_exeggcute"
            self.log(f"experiment head, phase 1: {a.exp_pre_steps} steps imitating Exeggcute (KL only, data "
                     f"{a.exp_pre_mix if a.exp_pre_mix != 'none' else 'the usual human/strong mix'}); in {pre}")
            self.train_stage(pre, a.exp_pre_steps, [
                "--stage", "policy", "--init", start, "--alpha", "1", "--bc-weight", "0", "--anchor-weight", "0", "--lr", str(lr),
                "--warmup", str(a.exp_pre_warmup), "--monitor", "loss", "--deep-actor", "--freeze", "actor"]
                + (["--rebalance", a.exp_pre_mix] if a.exp_pre_mix != "none" else []))
            init = os.path.join(pre, "weights_final.pt")
            self.log(f"experiment head, phase 2: imitating the strong bots from {init}")
        extra = ["--stage", "policy", "--init", init, "--alpha", str(kl), "--bc-weight", str(bc), "--anchor-weight", str(anchor),
                 "--lr", str(lr), "--monitor", "loss", "--rebalance", mix]
        if a.experiment == "head":
            extra += ["--deep-actor", "--freeze", "actor"]
        else:
            extra += ["--opponent-head", "--opp-weight", str(opp), "--kl-scope", "nonbot"] + (["--deep-actor"] if a.exp_deep_actor else [])
        self.train_stage(out, a.exp_steps, extra)
        self.final_tournament(a.exp_run or a.experiment, os.path.join(out, "weights_final.pt"))

    def blend_experiment(self, start: str) -> None:
        """Each --blends policy (";" between policies, "," between members, "@w" a member's weight) against TaurosV0. Members by name:
        start (--finetune-from), ckpt5000 (weights_5000.pt beside it), head and opphead (the experiments' weights_final.pt),
        distilled (WORK/run_v3/weights_best.pt), or a path."""
        names = {"start": start, "ckpt5000": os.path.join(os.path.dirname(start), "weights_5000.pt"),
                 "head": os.path.join(self.work, "exp_head", "weights_final.pt"), "opphead": os.path.join(self.work, "exp_opphead", "weights_final.pt"),
                 "distilled": os.path.join(self.work, "run_v3", "weights_best.pt")}
        for policy in [p.strip() for p in self.args.blends.split(";") if p.strip()]:
            members = []
            for part in policy.split(","):
                name, _, weight = part.strip().partition("@")
                members.append((names.get(name, os.path.abspath(os.path.expanduser(name))), weight or "1"))
            missing = [path for path, _ in members if not os.path.exists(path)]
            if missing:
                self.log(f"blend {policy}: skipped, no {', '.join(missing)}")
                continue
            self.final_tournament(re.sub(r"[^A-Za-z0-9.@_-]+", "+", policy), ",".join(f"{path}@{w}" for path, w in members))

    def final_tournament(self, label: str, spec: str) -> None:
        a = self.args
        result = os.path.join(self.side.dir, f"final_{label}.json")
        from metamon.recurrent.tournament import describe

        if os.path.exists(result):
            self.log(f"FINAL TOURNAMENT {label} vs TaurosV0 (played before): {describe(json.load(open(result)))}")
            return
        while self.side.setup:  # (the harness and TaurosV0, set up in the background at the start)
            self.side.poll()
            time.sleep(5)
        if not os.path.exists(os.path.join(self.work, "duel", ".ready")):
            raise RuntimeError(f"the tournament setup failed (see {self.logs}/tournament_setup.txt)")
        procs = a.final_procs or max(1, min(16, self.cpus // 2))
        self.log(f"final tournament {label}: {a.final_games} games vs TaurosV0 ({procs} processes; {spec})")
        cmd = [self.py, "-m", "metamon.recurrent.tournament", "play", spec, "--work", self.work, "--cache", self.cache,
               "--games", str(a.final_games), "--procs", str(procs), "--device", a.tournament_device,
               "--temperature", str(a.tournament_temperature), "--out", result + ".tmp", "--log-dir", os.path.join(self.logs, f"final_{label}")]
        self.run_one(f"final_{label}", cmd, self.env(RECURRENT_BLEND_MODE=a.blend_mode))
        os.replace(result + ".tmp", result)
        line = f"FINAL TOURNAMENT {label} vs TaurosV0: {describe(json.load(open(result)))}"
        self.log(line)
        with open(os.path.join(self.side.dir, "results.txt"), "a") as f:
            f.write(f"{time.strftime('%F %T')} {line} [{spec}]\n")

    def train_stage(self, out: str, steps: int, extra: list[str]) -> None:
        a = self.args
        cmd = [self.py, "-m", "metamon.recurrent.train", out]
        for name, weight in self.mix.items():
            cmd += ["--source", f"{name}={os.path.join(self.work, name, 'built')}:{weight}"]
        cmd += ["--burn-in", str(a.burn_in), "--seed", str(a.seed), "--accum", str(a.accum)]
        cmd += ["--steps", str(steps), "--batch", str(a.batch), "--workers", str(a.train_workers or min(self.cpus, 12)), "--device", self.device,
                "--precision", self.precision if self.gpus else "fp32", "--target", "blend", "--alpha", str(a.alpha), "--bc-weight", str(a.bc_weight), "--lr", str(a.lr), "--ckpt-every", str(a.ckpt_every),
                "--val-every", str(a.val_every), "--save-every", str(a.save_every), "--val-fraction", str(a.val_fraction), "--resume"]
        if a.val_batches:
            cmd += ["--val-batches", str(a.val_batches)]
        cmd += ["--patience", str(a.patience), "--min-delta", str(a.min_delta)]
        # the model's size (a fresh model; resumed or initialised weights keep theirs) and the win loss's weight and reach
        if (a.width, a.layers, a.ff) != (192, 4, 0):
            cmd += ["--width", str(a.width), "--layers", str(a.layers), "--ff", str(a.ff)]
        if a.value_weight is not None:
            cmd += ["--value-weight", str(a.value_weight)]
        if a.value_trunk_scale != 1.0:
            cmd += ["--value-trunk-scale", str(a.value_trunk_scale)]
        if a.monitor != "auto":
            cmd += ["--monitor", a.monitor]
        if self.backup:
            cmd += ["--backup-dir", os.path.join(self.backup, os.path.basename(out)), "--backup-every", str(a.backup_every)]
        if a.init and extra[1] != "value" and not a.finetune_from:
            cmd += ["--init", self.init_path()]  # (the first stage starts from these weights; a later restart of the same command resumes from the run's own latest.pt instead)
        cmd += extra  # (later flags win: the value stage's --lr replaces the default one)
        if a.tournament_every > 0:
            cmd += ["--save-every", str(math.gcd(a.save_every, a.tournament_every))]  # (later flags win: a weights_<step>.pt at every tournament step)
            self.side.watch.append((os.path.basename(out), out))
        stage = extra[1]
        self.log(f"training ({stage} stage) {steps} steps; progress in {out}/log.txt")
        self.run_one("train" if stage != "value" else "train_value", cmd, self.env(HF_HUB_OFFLINE=1))

    def init_path(self) -> str:
        """--init, kept on the backup volume too (BACKUP/init/<name>): a new container that lost the file uses the volume's copy."""
        a = self.args
        if not self.backup:
            return a.init
        kept = os.path.join(self.backup, "init", os.path.basename(a.init))
        if os.path.exists(a.init):
            if not os.path.exists(kept) or os.path.getsize(kept) != os.path.getsize(a.init):
                os.makedirs(os.path.dirname(kept), exist_ok=True)
                __import__("shutil").copyfile(a.init, kept + ".tmp")
                os.replace(kept + ".tmp", kept)
                self.log(f"copied --init to {kept}")
            return a.init
        if os.path.exists(kept):
            self.log(f"--init {a.init} is missing; using the backup's copy {kept}")
            return kept
        return a.init

    def summary(self) -> None:
        a = self.args
        if a.experiment == "blend":
            self.log("stage times: " + ", ".join(f"{k} {v:.0f}s" for k, v in self.timings.items()))
            return
        out = os.path.join(self.work, a.exp_run or f"exp_{a.experiment}") if a.experiment else \
            os.path.join(self.work, (f"{a.finetune_run}_actor" if a.ft_actor_only else f"{a.finetune_run}_joint" if a.ft_joint_steps > 0
                                     else f"{a.finetune_run}_all" if a.ft_all_steps > 0 else f"{a.finetune_run}_value_head") if a.finetune_from
                         else a.value_run if a.value_steps > 0 else a.policy_run)
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
        # --games 0 means every game: scale up to the dataset's size
        full = sum(a.full_games or self.dataset_games.get(n, 0) for n in sources)
        if not full:
            self.log("projection: the dataset size is unknown (local folders?); pass --games N for a projection")
            return
        scale_games = full / max(1, sum(w for w, _ in self.counts.values()))
        # each stage's fixed startup (loading the ensemble takes about a minute per label process) happens once, not per game
        startup = {"label": 60.0, "parse": 5.0, "legacy": 5.0, "merge": 1.0, "build": 3.0}
        parts = []
        for stage in ("parse", "legacy", "merge", "label", "build"):
            t = sum(v for k, v in self.timings.items() if k.startswith(stage + "_"))
            work = max(t - startup[stage] * len(sources), 0.3 * t)
            parts.append((stage, work * scale_games))
        t_train = self.timings.get("train", 0.0) + self.timings.get("train_value", 0.0)
        parts.append((f"train ({a.full_steps} + {a.full_value_steps} steps)", t_train / max(1, a.steps + a.value_steps) * (a.full_steps + a.full_value_steps)))
        games = a.full_games or "0 (all: " + ", ".join(f"{n} {self.dataset_games.get(n, '?')}" for n in sources) + ")"
        self.log(f"projected for --games {games} per source (hours, rough; training at most, it stops early without progress): " + ", ".join(f"{k} {v / 3600:.1f}" for k, v in parts) + f"; total {sum(v for _, v in parts) / 3600:.1f}")

    def main(self) -> None:
        a = self.args
        self.hardware()
        os.makedirs(os.path.join(self.work, "tables"), exist_ok=True)
        if not os.path.exists(os.path.join(self.work, "tables", "species_ids.json")):
            self.run_one("tables", [self.py, "-m", "metamon.recurrent.tables", os.path.join(self.work, "tables")], self.env())
        self.mix = dict(m.split("=") for m in a.mix) if a.mix else {"human": 0.4, "strong": 0.6}
        self.background = []
        self.side = SideJobs(self)
        try:
            self.stages()
        except BaseException:
            self.side.stop()  # (Ctrl-C or a failure: nothing is left running)
            raise

    def stages(self) -> None:
        a = self.args
        self.side.start_setup()
        uploadable = not a.train_only and a.data != "built"  # (games built here, not downloaded)
        if a.data == "built" and not a.train_only and not (a.from_raw or a.raw_human or a.raw_strong or a.traj_human_dir or a.traj_strong_dir):
            self.preflight()
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
            self.preflight()
            self.prefetch()
            raw = self.fetch_data() if a.from_raw or a.raw_human or a.raw_strong else {}
            sources = [n for n in SOURCES if float(self.mix.get(n, 0)) > 0]
            for name in sources:
                self.prepare(name, raw.get(name), a.games)
            if a.data == "labels" and not a.skip_checks:
                self.check_labels()
            try:
                for name in sources:
                    self.label_and_build(name)
                self.finish_background()
            except BaseException:
                self.stop([(p, h) for _, p, h, _ in self.background])  # (Ctrl-C or a failure: no build is left running)
                raise
        self.mix = {k: v for k, v in self.mix.items() if float(v) > 0}
        if not a.skip_checks and not a.train_only:
            self.check_built()
        if uploadable:
            self.side.start_upload(list(self.mix))
        if a.check_only:
            self.side.drain()
            self.log("--check-only: data downloaded and checked; stopping before training")
            return
        self.train()
        self.side.drain()
        self.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.environ.get("WORK", os.path.expanduser("~/work")), help="data and runs; on the container disk (never a small volume such as a vast.ai /workspace)")
    ap.add_argument("--cache", default=None, help="metamon cache (default WORK/metamon-cache)")
    ap.add_argument("--data", choices=("built", "labels", "trajectories"), default="labels",
                    help="what to download: built games (train at once), trajectories + labels (build, then train) or trajectories (label, build, train)")
    ap.add_argument("--built-repo", default="bsmathers/gen1ou-labels-backup", help="dataset repo with <source>_built.zip and <source>_labels.zip (public)")
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
    ap.add_argument("--games", type=int, default=0, help="games to take from each source (0: all, the default). --data built always uses every built game")
    ap.add_argument("--mix", action="append", help="name=weight, e.g. human=0.4 (default human=0.4 strong=0.6; weight 0 skips a source)")
    ap.add_argument("--steps", type=int, default=100000, help="first-stage steps at most (it stops earlier after --patience validations without progress)")
    ap.add_argument("--batch", default="256", help="training batch: a number or auto (largest that fits in GPU memory); the learning rate is set for 256")
    ap.add_argument("--value-steps", type=int, default=20000, help="steps of the second stage that trains the win head with the backbone anchored (0: one joint stage)")
    ap.add_argument("--patience", type=int, default=10, help="stop a stage after this many validations without progress (0: run all its steps)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="the smallest drop of the validation loss that counts as progress (KL in the first stage, win BCE in the second)")
    ap.add_argument("--train-only", action="store_true", help="skip every data stage and train on the games already built under WORK")
    ap.add_argument("--init", default=None, help="weights or a checkpoint (e.g. an earlier run's latest.pt) that the first training stage starts from")
    ap.add_argument("--policy-run", default="run_v3", help="folder under WORK for the first stage's checkpoints and log")
    ap.add_argument("--value-run", default="run_v3_value", help="folder under WORK for the second stage's")
    ap.add_argument("--anchor-weight", type=float, default=5.0, help="(second stage) weight of KL(first-stage policy || current policy)")
    ap.add_argument("--value-lr", type=float, default=3e-4, help="(second stage) learning rate")
    ap.add_argument("--value-bc-weight", type=float, default=0.0,
                    help="(second stage) weight of imitating the --bc-players' played actions on their games, besides the win head and the anchor")
    ap.add_argument("--finetune-from", default=None, help="(finetune.sh) run the three fine-tuning stages from these weights instead of the usual two")
    ap.add_argument("--finetune-run", default="run_ft_bots", help="prefix of the fine-tuning stages' folders: <prefix>_actor, _value_head, _all")
    ap.add_argument("--ft-kl-weight", type=float, default=0.5, help="(fine-tune A) weight of the KL to Exeggcute's targets")
    ap.add_argument("--ft-bc-weight", type=float, default=0.5, help="(fine-tune A) weight of imitating the --bc-players' played moves")
    ap.add_argument("--ft-actor-mix", default="bots=1,human=0,otherbots=0",
                    help="(fine-tune A) the data: every game seen from the imitated bots' side, plus random subsets of the human and the "
                         "other bot games; game counts and shares of the trained decisions in these ratios (none: the usual human/strong mix)")
    ap.add_argument("--ft-actor-anchor", type=float, default=2.0, help="(fine-tune A) weight of KL(--finetune-from policy || current policy)")
    ap.add_argument("--ft-all-anchor", type=float, default=1.0, help="(fine-tune 2) weight of KL(stage A policy || current policy)")
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="(fine-tune A and 2) learning rate; stage 1 uses --value-lr")
    ap.add_argument("--ft-actor-steps", type=int, default=20000)
    ap.add_argument("--ft-actor-only", action="store_true", help="(fine-tune) stop after stage A (no value stages)")
    ap.add_argument("--burn-in", type=int, default=32,
                    help="decisions before each training window's scored part that only build the recurrent state (the window is "
                         "burn-in + 64 long; a game shorter than that is trained whole, with its full history)")
    ap.add_argument("--seed", type=int, default=0, help="training seed (window sampling and the random subsets of --rebalance)")
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per training step (gradient accumulation): the same batch and results in 1/N of the GPU memory")
    ap.add_argument("--ft-skip-actor", action="store_true",
                    help="(value.sh) skip stage A: --finetune-from is already the fine-tuned actor; stages 1 and 2 start from it")
    ap.add_argument("--ft-value-head-steps", type=int, default=10000)
    ap.add_argument("--ft-all-steps", type=int, default=20000, help="(fine-tune 2) steps; 0 stops after stage 1")
    ap.add_argument("--ft-joint-steps", type=int, default=0,
                    help="(fine-tune 3) steps of a last stage from stage 2's weights: everything trains on the win BCE + KL to Exeggcute + "
                         "imitation of the bots, no anchor; 0 = no stage 3 (value.sh: 20000)")
    ap.add_argument("--ft-joint-kl-weight", type=float, default=0.5, help="(fine-tune 3) weight of the KL to Exeggcute's targets")
    ap.add_argument("--ft-joint-value-weight", type=float, default=0.5, help="(fine-tune 3) weight of the win BCE")
    ap.add_argument("--ft-joint-bc-weight", type=float, default=1.0, help="(fine-tune 3) weight of imitating the bots' played moves")
    ap.add_argument("--ft-joint-mix", default="none",
                    help="(fine-tune 3) data: none = the usual human/strong mix (the value stages' data), or a --rebalance spec")
    ap.add_argument("--experiment", choices=("head", "opphead", "blend"), default=None,
                    help="(with --finetune-from START; experiments.sh) instead of the three stages: head = only the actor trains (the linear head plus a "
                         "deep actor head on the frozen state token), imitating the bots with a little KL to Exeggcute; opphead = an opponent-action head, imitation on the "
                         "bots' games and KL to Exeggcute on the rest; blend = tournaments of policy blends (--blends). Each ends with a "
                         "--final-games tournament against TaurosV0")
    ap.add_argument("--exp-pre-steps", type=int, default=1500,
                    help="(head) phase 1: steps of the new head imitating Exeggcute (KL only) before the bot imitation; 0 skips it")
    ap.add_argument("--exp-pre-warmup", type=int, default=150, help="(head) phase 1's learning-rate warmup steps")
    ap.add_argument("--exp-pre-mix", default="none",
                    help="(head) phase 1's data: none = the usual human/strong mix (--mix), or a --rebalance spec such as bots=1,human=0,otherbots=0")
    ap.add_argument("--exp-deep-actor", action="store_true",
                    help="(opphead) also add the 3-layer actor head (192 -> 64 -> 64 -> 9); it trains with everything else")
    ap.add_argument("--exp-run", default=None, help="(experiment) its folder in WORK (default exp_<experiment>)")
    ap.add_argument("--exp-steps", type=int, default=10000, help="(experiment) training steps (early stopping still applies)")
    ap.add_argument("--exp-lr", type=float, default=None, help="(experiment) learning rate (default: head 3e-4, opphead 1e-4)")
    ap.add_argument("--exp-kl-weight", type=float, default=None, help="(experiment) KL to Exeggcute (default: head 0.1 on all decisions, opphead 1 off the bots' games)")
    ap.add_argument("--exp-bc-weight", type=float, default=None, help="(experiment) imitation of the bots' played moves (default 1)")
    ap.add_argument("--exp-anchor", type=float, default=None, help="(experiment) KL(START policy || current) (default 0)")
    ap.add_argument("--exp-opp-weight", type=float, default=None, help="(opphead) the opponent head's cross-entropy (default 0.5)")
    ap.add_argument("--exp-mix", default=None, help="(experiment) --rebalance data (default: head bots=1,human=0,otherbots=0; opphead bots=3,human=2,otherbots=1)")
    ap.add_argument("--blends", default=Launcher.DEFAULT_BLENDS,
                    help="(blend) policies, ';' between them, members 'name@weight' (names: start, ckpt5000, head, opphead, distilled, or a path)")
    ap.add_argument("--blend-mode", choices=("mix", "logit"), default="mix",
                    help="(blend) mix: the weighted average of the members' move probabilities; logit: the weighted geometric mean")
    ap.add_argument("--final-games", type=int, default=500, help="(experiment) games of each final tournament")
    ap.add_argument("--final-procs", type=int, default=0, help="(experiment) harness processes of a final tournament (default: half the CPUs, at most 16)")
    ap.add_argument("--value-from", default=None,
                    help="skip the policy stage: start the value stage from these policy weights (they are also the anchor's reference)")
    ap.add_argument("--logs-name", default="logs", help="folder under WORK for the launcher's logs (one per launcher when several share WORK)")
    ap.add_argument("--train-workers", type=int, default=0, help="training data loader processes (0: the CPUs, at most 12)")
    ap.add_argument("--reuse-tournament-setup", action="store_true",
                    help="use WORK/duel as it is when it is ready instead of setting it up again (several launchers at once)")
    ap.add_argument("--width", type=int, default=192, help="model width (a multiple of 32): 192 is the 1.31M model, 96 with --layers 2 is 205k, 256 with --layers 6 is 3.3M")
    ap.add_argument("--layers", type=int, default=4, help="model transformer blocks")
    ap.add_argument("--ff", type=int, default=0, help="model feed-forward width (0: twice the width)")
    ap.add_argument("--value-weight", type=float, default=None, help="weight of the win BCE in a joint stage (--value-steps 0; train.py's default: 1)")
    ap.add_argument("--value-trunk-scale", type=float, default=1.0,
                    help="share of the win loss's gradient that reaches the shared network (1: all; 0: the win head learns on the policy's features)")
    ap.add_argument("--monitor", choices=("auto", "kl", "bce", "loss"), default="auto",
                    help="what early stopping and weights_best.pt follow (train.py): loss = the stage's weighted loss")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the distillation KL")
    ap.add_argument("--bc-weight", type=float, default=0.0, help="weight of the imitation term on the imitated bots' games")
    ap.add_argument("--bc-players", default="Tauros-v1,Tauros-v2,SmallG1Online-v2,SmallG1Online-v3", help="usernames whose played actions are imitated")
    ap.add_argument("--ckpt-every", type=int, default=1, help="steps between writes of latest.pt (1: every iteration)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=int, default=20, help="hold out 1 game in N (by battle id)")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--backup-dir", default="auto",
                    help="separate storage for backups that outlive the container disk: auto = the vast.ai local volume at /data when one is mounted; "
                         "a path; or none. Holds metamon-backup/<run>/ (weights_<step>.pt every --backup-every steps, latest.pt overwritten each time, "
                         "weights_best.pt, weights_final.pt, log.txt) and metamon-backup/init/ (a copy of --init). A run whose WORK lost its "
                         "latest.pt resumes from the backup's")
    ap.add_argument("--backup-every", type=int, default=100, help="steps between backups")
    ap.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--parse-procs", type=int, default=0, help="parser processes (default: one per CPU)")
    ap.add_argument("--label-procs", type=int, default=0, help="labelling processes per GPU (default: 5 on a 24 GiB card with 10+ CPUs, fewer with less memory or CPUs)")
    ap.add_argument("--label-rows", type=int, default=0, help="game-turns per parallel forward on a GPU (default 8192; lower it if a card runs out of memory)")
    ap.add_argument("--label-batch", type=int, default=0, help="games labelled in lockstep per process (default 256 on a GPU, 64 on a CPU: the lanes of a lockstep batch that finished early still cost their share of the math, and a CPU gains little from larger batches)")
    ap.add_argument("--no-overlap", action="store_true", help="build a source after labelling it, instead of in the background while the next source is prepared")
    ap.add_argument("--smoke", action="store_true", help="tiny sample of every stage, prints timings and a projection")
    ap.add_argument("--check-only", action="store_true", help="download, check the labels, build, check the built games, then stop before training")
    ap.add_argument("--skip-checks", action="store_true", help="skip the label check and the built-game check (not recommended)")
    ap.add_argument("--max-unlabelled", type=float, default=0.01, help="(--data labels) stop when more than this share of the trajectories has no downloaded label")
    ap.add_argument("--check-decisions", type=int, default=2500, help="(--data labels) decisions per source (random whole games) relabelled from scratch to check the downloaded labels")
    ap.add_argument("--check-built-games", type=int, default=300, help="built games per source checked against their labels by name")
    ap.add_argument("--check-seed", type=int, default=0, help="seed of the checks' random samples")
    ap.add_argument("--tournament-every", type=int, default=5000, help="steps between tournaments against TaurosV0, in each training stage (0: none)")
    ap.add_argument("--tournament-games", type=int, default=100)
    ap.add_argument("--tournament-procs", type=int, default=0, help="harness processes per tournament (default: a quarter of the CPUs, at most 4)")
    ap.add_argument("--tournament-temperature", type=float, default=1.0, help="both sides' action temperature in tournaments (1: sample from each model's probabilities; 0: argmax)")
    ap.add_argument("--tournament-device", default="cpu", help="TaurosV0's device in tournaments (cpu, as in the local trials, or cuda)")
    ap.add_argument("--upload-repo", default="auto", help="public dataset repo the built games are uploaded to when HF_WRITE_TOKEN is set "
                                                          "(auto: <token's user>/gen1ou-recurrent-built)")
    args = ap.parse_args()
    args.full_games, args.full_steps, args.full_value_steps = args.games, args.steps, args.value_steps  # (the smoke run shrinks these; the projection scales back up)
    if args.smoke:
        args.cache = args.cache or os.path.join(args.work, "metamon-cache")  # shared with the full run
        args.games = min(args.games, 150) if args.games else 150  # (0 is every game)
        args.steps, args.val_every, args.save_every, args.val_batches = min(args.steps, 150), 75, 150, 20
        args.value_steps = min(args.value_steps, 75)
        args.val_fraction = min(args.val_fraction, 5)
        args.check_decisions, args.check_built_games = min(args.check_decisions, 300), min(args.check_built_games, 50)
        if args.tournament_every > 0:
            args.tournament_every, args.tournament_games = args.steps, min(args.tournament_games, 4)  # (one short tournament at the end of the policy stage)
        args.work = args.work.rstrip("/") + "_smoke"
    import signal

    def interrupt(*_):  # (Ctrl-C and `kill`/`pkill` both stop the workers: the stage loop handles the interrupt)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)  # (also when started in the background, where SIGINT begins ignored)
    signal.signal(signal.SIGTERM, interrupt)
    launcher = Launcher(args)
    try:
        launcher.main()
    except KeyboardInterrupt:
        # (the stage loops have stopped their processes by now; training wrote its latest.pt)
        launcher.log("stopped (interrupted). Every stage resumes where it stopped: run the same command again to continue")
        sys.exit(130)


if __name__ == "__main__":
    main()
