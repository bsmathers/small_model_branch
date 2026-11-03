"""One-command preparation and training for the recurrent model. Every step is resumable: finished
games are skipped, so an interrupted run can be started again with the same arguments.

    # per source (raw Showdown logs -> trajectories -> Exeggcute labels -> built games);
    # add --input-format parsed when the input is official parsed files (e.g. pac-tauros)
    python -m metamon.recurrent.pipeline prepare --name human --raw /data/raw/human --work /data/work \
        --cache /data/metamon-cache --device cuda --label-batch 64 [--shard 0/4] [--limit N]

    # train on the prepared sources with a mix of decisions
    python -m metamon.recurrent.pipeline train --work /data/work --out /data/runs/v1 \
        --mix human=0.5 --mix bots=0.5 --steps 100000 --batch 128 --precision bf16

`prepare` writes WORK/tables (once), WORK/<name>/{trajectories,labels,built}. Shards of the same source can
run on different GPUs; build once afterwards with `prepare --only build`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prepare(args) -> None:
    py = sys.executable
    work = os.path.join(args.work, args.name)
    tables = os.path.join(args.work, "tables")
    traj, labels, built = (os.path.join(work, d) for d in ("trajectories", "labels", "built"))
    only = args.only
    if not os.path.exists(os.path.join(tables, "species_ids.json")):
        os.makedirs(tables, exist_ok=True)
        run(py, "-m", "metamon.recurrent.tables", tables)
    if only in (None, "parse"):
        # raw Showdown logs go through the full parser; files already in the official parsed format (no raw
        # log exists) get their extras derived from the states, with less information (see upgrade.py)
        module = "metamon.recurrent.parse" if args.input_format == "raw" else "metamon.recurrent.upgrade"
        cmd = [py, "-m", module, args.raw, traj, "--shard", args.shard]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run(*cmd)
    if only in (None, "label"):
        cmd = [py, "-m", "metamon.recurrent.label", traj, labels, "--cache", args.cache, "--batch", str(args.label_batch),
               "--device", args.device, "--threads", str(args.threads), "--shard", args.shard,
               "--precision", args.precision]
        run(*cmd)
    if only in (None, "build"):
        run(py, "-m", "metamon.recurrent.build", traj, labels, built, tables, "--workers", str(args.workers))


def train(args) -> None:
    sources = [f"{m.split('=')[0]}={os.path.join(args.work, m.split('=')[0], 'built')}:{m.split('=')[1]}" for m in args.mix]
    cmd = [sys.executable, "-m", "metamon.recurrent.train", args.out]
    for s in sources:
        cmd += ["--source", s]
    cmd += ["--steps", str(args.steps), "--batch", str(args.batch), "--workers", str(args.workers), "--device", args.device]
    cmd += ["--target", args.target, "--alpha", str(args.alpha), "--lr", str(args.lr), "--resume"]
    cmd += ["--precision", args.precision]
    if args.compile:
        cmd.append("--compile")
    run(*cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--cache", default=os.environ.get("METAMON_CACHE_DIR", "."))
    p.add_argument("--device", default="cuda")
    p.add_argument("--label-batch", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--shard", default="0/1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", choices=("parse", "label", "build"))
    p.add_argument("--input-format", choices=("raw", "parsed"), default="raw", help="raw Showdown logs, or files in the official parsed format")
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    t = sub.add_parser("train")
    t.add_argument("--work", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--mix", action="append", required=True, help="name=weight")
    t.add_argument("--steps", type=int, default=100_000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--device", default="cuda")
    t.add_argument("--target", default="blend")
    t.add_argument("--alpha", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    t.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    {"prepare": prepare, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
