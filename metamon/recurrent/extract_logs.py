"""Extracts one format's replays from the metamon-raw-replays parquet shards into per-game `.log` files,
the input of `metamon.recurrent.parse`.

    python -m metamon.recurrent.extract_logs SHARD_DIR OUT_DIR [--format gen1ou]

Writes OUT_DIR/logs/<id>.log and OUT_DIR/manifest.tsv. A row is kept when its `formatid` is the format, or
when it is `MISSING` and the log's own `|tier|` line names the format. The source files are only read.
Each log gets the shard's upload time as a leading `|t:|<unix time>` line if it has none, because the parser
reads the game date from it.
"""
import argparse
import glob
import os
import re

import pyarrow.parquet as pq

TIER_LINE = re.compile(r"^\|tier\|(.+)$", re.M)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards")
    ap.add_argument("out")
    ap.add_argument("--format", default="gen1ou")
    args = ap.parse_args()
    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    kept = skipped_existing = 0
    seen = set()
    with open(os.path.join(args.out, "manifest.tsv"), "w") as manifest:
        manifest.write("id\tsource\tuploadtime\trating\tp1\tp2\tturns\trated\n")
        for shard in sorted(glob.glob(os.path.join(args.shards, "*.parquet"))):
            table = pq.read_table(shard, columns=["id", "formatid", "log", "players", "uploadtime", "rating"])
            for row in table.to_pylist():
                log, game_id = row["log"], row["id"]
                if row["formatid"] != args.format:
                    if row["formatid"] != "MISSING":
                        continue
                    tier = TIER_LINE.search(log)
                    if not tier or normalize(tier.group(1)) != args.format:
                        continue
                if game_id in seen or "/" in game_id or "_" in game_id:
                    continue  # the parser's output names are `<id>_...` and later steps split on "_"
                seen.add(game_id)
                if not re.search(r"^\|t:\|", log, re.M):
                    log = f"|t:|{row['uploadtime']}\n" + log
                path = os.path.join(log_dir, f"{game_id}.log")
                if os.path.exists(path):
                    skipped_existing += 1
                else:
                    with open(path, "w") as f:
                        f.write(log)
                turns = len(re.findall(r"^\|turn\|", log, re.M))
                p1, p2 = (row["players"] + ["", ""])[:2]
                source = game_id.split("-")[0]
                manifest.write(f"{game_id}\t{source}\t{row['uploadtime']}\t{row['rating']}\t{p1}\t{p2}\t{turns}\t{int(chr(10) + '|rated|' in log)}\n")
                kept += 1
            print(f"{os.path.basename(shard)}: {kept} kept so far", flush=True)
    print(f"kept {kept} games ({skipped_existing} already on disk)")


if __name__ == "__main__":
    main()
