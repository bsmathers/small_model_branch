#!/usr/bin/env python3
"""Exports a recurrent model (metamon.recurrent) for the harness's native search backend (harness/src/rmodel.zig, rnative.zig).

    python export_recurrent.py WEIGHTS OUT_DIR

writes OUT_DIR/model.bin (every tensor as little-endian float32, concatenated), OUT_DIR/model.json (each tensor's name, shape
and offset in floats, and the model's configuration) and OUT_DIR/tables/*.json (metamon.recurrent.tables.Tables.from_dex(1):
the ids, base stats, move power and type, and type chart the features are computed with). The opponent-action head is left
out (the search does not use it).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metamon.recurrent.model import PolicyValueModel  # noqa: E402
from metamon.recurrent.tables import Tables  # noqa: E402


def main() -> None:
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = PolicyValueModel.from_state(torch.load(weights, map_location="cpu"))
    tensors, chunks, offset = [], [], 0
    for name, t in model.state_dict().items():
        if name.startswith("opponent."):
            continue
        a = t.detach().to(torch.float32).contiguous().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
        offset += a.size
        chunks.append(a.reshape(-1))
    blob = np.concatenate(chunks).astype("<f4")
    (out / "model.bin").write_bytes(blob.tobytes())
    cfg = model.cfg
    meta = {
        "weights": str(weights.resolve()),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "floats": int(offset),
        "config": {"d": cfg.d, "layers": cfg.layers, "heads": cfg.heads, "ff": cfg.ff, "n_species": cfg.n_species, "n_moves": cfg.n_moves,
                   "n_status": cfg.n_status, "emb": cfg.emb, "status_emb": cfg.status_emb, "n_actions": cfg.n_actions,
                   "deep_actor": cfg.deep_actor, "actor_width": cfg.actor_width},
        "tensors": tensors,
    }
    (out / "model.json").write_text(json.dumps(meta, indent=1))
    Tables.from_dex(1).write(out / "tables")
    print(f"exported {weights} -> {out}: {len(tensors)} tensors, {offset:,} floats")


if __name__ == "__main__":
    main()
