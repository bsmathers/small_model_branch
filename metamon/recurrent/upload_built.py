"""Uploads the built training games to a public Hugging Face dataset repo, one zip per source (<source>_built.zip), in the layout
`launch.py --data built --built-repo REPO` downloads: the games' .npz files and the folder's .encoding_version stamp, flat.

    HF_WRITE_TOKEN=hf_... python -m metamon.recurrent.upload_built WORK REPO human strong     # REPO auto: <token's user>/gen1ou-recurrent-built

Each game file is already compressed (np.savez_compressed), so the zip stores them as they are: compressing them again would
cost CPU time for nothing. Resumable: a source whose zip was uploaded is marked (WORK/upload/.<source>_uploaded) and skipped; a
zip that was written completely is uploaded again without rewriting it. The zip is deleted after the upload, to free the disk.
The token is read from HF_WRITE_TOKEN only, never printed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile

from metamon.recurrent.features import ENCODING_STAMP, ENCODING_VERSION


def say(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("repo", help="dataset repo, e.g. user/gen1ou-recurrent-built (created public if missing; auto: <the token's user>/gen1ou-recurrent-built)")
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args()
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise SystemExit("HF_WRITE_TOKEN is not set")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if args.repo == "auto":
        args.repo = f"{api.whoami()['name']}/gen1ou-recurrent-built"
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    out_dir = os.path.join(args.work, "upload")
    os.makedirs(out_dir, exist_ok=True)
    for name in args.sources:
        marker = os.path.join(out_dir, f".{name}_uploaded")
        if os.path.exists(marker):
            say(f"{name}: uploaded before ({open(marker).read().strip()})")
            continue
        built = os.path.join(args.work, name, "built")
        games = sorted(f for f in os.listdir(built) if f.endswith(".npz"))
        stamp = os.path.join(built, ENCODING_STAMP)
        if not games or not os.path.exists(stamp) or open(stamp).read().strip() != ENCODING_VERSION:
            raise SystemExit(f"{name}: {built} has no games of encoding {ENCODING_VERSION!r}")
        path = os.path.join(out_dir, f"{name}_built.zip")
        done = path + ".complete"  # (the zip was written to the end: a restart uploads it as it is)
        if not os.path.exists(done):
            size = sum(os.path.getsize(os.path.join(built, f)) for f in games)
            free = shutil.disk_usage(out_dir).free
            if free < size * 1.05 + 2**30:
                raise SystemExit(f"{name}: the zip needs {size / 2**30:.1f} GiB and {free / 2**30:.1f} GiB is free")
            say(f"{name}: zipping {len(games)} games ({size / 2**30:.2f} GiB)")
            tmp = path + ".partial"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
                z.write(stamp, ENCODING_STAMP)
                for i, f in enumerate(games):
                    z.write(os.path.join(built, f), f)
                    if (i + 1) % 20000 == 0:
                        say(f"{name}: zipped {i + 1} of {len(games)}")
            os.replace(tmp, path)
            open(done, "w").write(f"{len(games)}\n")
        say(f"{name}: uploading {os.path.getsize(path) / 2**30:.2f} GiB to {args.repo}")
        api.upload_file(path_or_fileobj=path, path_in_repo=f"{name}_built.zip", repo_id=args.repo, repo_type="dataset",
                        commit_message=f"{name}: {len(games)} built games, encoding {ENCODING_VERSION}")
        open(marker, "w").write(f"{len(games)} games to {args.repo}, encoding {ENCODING_VERSION}\n")
        os.remove(path)
        os.remove(done)
        say(f"{name}: uploaded {len(games)} games to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())
