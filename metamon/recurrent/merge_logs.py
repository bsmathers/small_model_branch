"""Merges replay collections by game id into one directory of per-game `.log` files.

    python -m metamon.recurrent.merge_logs OUT_DIR --logs DIR [--logs DIR ...] --parquet FILE [--parquet FILE ...]

Sources are given in priority order (all --logs first, then all --parquet): when an id appears in several,
the first source's log is kept. A `--logs` source is a directory with `logs/<id>.log` and `manifest.tsv`
(as written by extract_logs); a `--parquet` source has columns id, log, players, uploadtime and optionally
rating and views. The sources are only read. A leading `|t:|<upload time>` line is added to a log that has
none, because the parser reads the game date from it.

Writes OUT_DIR/logs/<id>.log and OUT_DIR/manifest.tsv (id, source, uploadtime, rating, views, p1, p2, turns,
rated, origin), where rating and views come from any source that has them and `origin` lists the sources
holding the id.
"""
import argparse
import csv
import os
import re
import shutil

import pyarrow.parquet as pq


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, re.M))


def usable(game_id: str) -> bool:
    # output names are `<id>_...` and later steps split on "_"
    return "/" not in game_id and "_" not in game_id and game_id != ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--logs", action="append", default=[])
    ap.add_argument("--parquet", action="append", default=[])
    args = ap.parse_args()
    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    meta: dict[str, dict] = {}  # id -> manifest fields
    origin: dict[str, list[str]] = {}
    written: set[str] = set()
    skipped = 0

    def note(game_id: str, name: str, fields: dict) -> None:
        origin.setdefault(game_id, []).append(name)
        m = meta.setdefault(game_id, {})
        for k, v in fields.items():
            if m.get(k) in (None, "", "None", "nan") and v not in (None, "", "None", "nan"):
                m[k] = v

    for directory in args.logs:
        name = os.path.basename(directory.rstrip("/"))
        with open(os.path.join(directory, "manifest.tsv")) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                game_id = row["id"]
                if not usable(game_id):
                    skipped += 1
                    continue
                src = os.path.join(directory, "logs", f"{game_id}.log")
                if game_id not in written and os.path.exists(src):
                    shutil.copyfile(src, os.path.join(log_dir, f"{game_id}.log"))
                    written.add(game_id)
                    meta[game_id] = {}
                note(game_id, name, {k: row[k] for k in ("uploadtime", "rating", "p1", "p2", "turns", "rated")})

    for path in args.parquet:
        name = os.path.splitext(os.path.basename(path))[0]
        pf = pq.ParquetFile(path)
        columns = [c for c in ("id", "log", "players", "uploadtime", "rating", "views") if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=2000, columns=columns):
            for row in batch.to_pylist():
                game_id, log = row["id"], row["log"]
                if not usable(game_id):
                    skipped += 1
                    continue
                players = (row.get("players") or []) + ["", ""]
                fields = {"uploadtime": row.get("uploadtime"), "rating": row.get("rating"), "views": row.get("views"), "p1": players[0], "p2": players[1]}
                if game_id not in written:
                    if not re.search(r"^\|t:\|", log, re.M):
                        log = f"|t:|{row.get('uploadtime') or 0}\n" + log
                    with open(os.path.join(log_dir, f"{game_id}.log"), "w") as f:
                        f.write(log)
                    written.add(game_id)
                    meta[game_id] = {}
                    fields["turns"] = count(r"^\|turn\|", log)
                    fields["rated"] = int("\n|rated|" in log)
                note(game_id, name, fields)

    with open(os.path.join(args.out, "manifest.tsv"), "w") as f:
        f.write("id\tsource\tuploadtime\trating\tviews\tp1\tp2\tturns\trated\torigin\n")
        for game_id in sorted(written):
            m = meta[game_id]
            vals = [m.get(k, "") for k in ("uploadtime", "rating", "views", "p1", "p2", "turns", "rated")]
            f.write("\t".join([game_id, game_id.split("-")[0], *map(str, vals), "+".join(origin[game_id])]) + "\n")
    print(f"wrote {len(written)} games ({skipped} ids skipped as unusable)")


if __name__ == "__main__":
    main()
