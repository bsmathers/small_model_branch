"""
Pull Pokémon Showdown replays from HuggingFace dataset and convert them back to raw replay format.
"""

import os
import json
import shutil
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset, config


def write_replay_to_disk(
    replay_data: dict,
    output_dir: str,
    format_id: Optional[str] = None,
    replay_id: Optional[str] = None,
) -> bool:
    # Create directory structure like output_dir/gen4/ou/
    gen_num = format_id[3]
    tier = format_id[4:]
    output_path = Path(output_dir) / f"gen{gen_num}" / tier
    os.makedirs(output_path, exist_ok=True)
    file_path = str(output_path / replay_id) + ".json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(replay_data, f, indent=2)
    return True


def process_dataset(
    dataset_id: str, output_dir: str, revision: str, remove_from_hf_cache: bool = True
) -> None:
    print(f"Loading dataset {dataset_id}...")
    # TODO: load specific (dated) versions
    dataset = load_dataset(
        dataset_id, revision=revision, download_mode="force_redownload"
    )
    main_split = next(iter(dataset.keys()))
    dataset = dataset[main_split]
    print(f"\nProcessing {len(dataset)} replays...")
    format_counts = defaultdict(int)
    for replay_data in tqdm(dataset):
        replay_id = replay_data.get("id", None)
        format_id = replay_data.get("formatid", None)
        format_str = replay_data.get("format", None)
        if replay_id is None or replay_id == "MISSING":
            print(f"Skipping replay because it has no `id`")
            continue
        if format_id is None or format_id == "MISSING":
            if format_str is not None and format_str != "MISSING":
                # attempt to get format from format string (usually [Gen X] Tier)
                format_id = (
                    format_str.lower()
                    .replace(" ", "")
                    .replace("[", "")
                    .replace("]", "")
                    .strip()
                )
            elif replay_id is not None and replay_id != "MISSING":
                # attempt to get format from replay ID (usually genXTier-IDNUMBER)
                potential_formatids = replay_id.split("-")
                for potential in potential_formatids:
                    if re.match(r"gen[0-9].*", potential):
                        format_id = potential.strip().lower()
                        break

        if not re.match(r"gen[0-9].*", format_id):
            print(
                f"Skipping replay {replay_id} because we cannot recover its format ({format_id}, {format_str})"
            )
            continue
        format_counts[format_id] += 1
        write_replay_to_disk(
            replay_data,
            output_dir,
            format_id=format_id,
            replay_id=replay_id,
        )

    print("\nReplays by format:")
    for fmt, count in sorted(format_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {fmt}: {count:,} replays")

    if remove_from_hf_cache:
        print("\nClearing dataset from cache...")
        cache_dir = os.path.join(
            config.HF_DATASETS_CACHE, dataset_id.replace("/", "___")
        )
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared cache for {dataset_id}")
        else:
            print(f"No cache found for {dataset_id}")
