#!/usr/bin/env python3
"""Checks the harness's native recurrent model (rmodel.zig) against PyTorch: random inputs and states through both.

    python check_rmodel.py WEIGHTS EXPORT_DIR HARNESS_BINARY [CASES]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metamon.recurrent.model import FLOAT_KEYS, ID_KEYS, PolicyValueModel  # noqa: E402

weights, export_dir, binary = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 50
model = PolicyValueModel.from_state(torch.load(weights, map_location="cpu")).eval()
rng = np.random.default_rng(0)
shapes = {"own_species": (6,), "own_status": (6,), "own_moves": (6, 4), "opp_species": (6,), "opp_status": (6,), "opp_moves": (6, 4)}
limits = {"species": 152, "status": 8, "move": 168}
cases, expected = [], []
for _ in range(n):
    ids = {}
    for k in ID_KEYS:
        top = limits["species"] if "species" in k else limits["status"] if "status" in k else limits["move"]
        ids[k] = rng.integers(0, top, size=shapes.get(k, (1,)))
    floats = {"own": rng.random((6, 10), dtype=np.float32), "opp": rng.random((6, 7), dtype=np.float32),
              "last": rng.random(34, dtype=np.float32), "field": rng.random(39, dtype=np.float32)}
    h = rng.standard_normal(192).astype(np.float32)
    with torch.no_grad():
        tokens = model.observation_tokens({k: torch.from_numpy(v)[None, None] for k, v in ids.items()},
                                          {k: torch.from_numpy(v)[None, None] for k, v in floats.items()})
        logits, win, h2 = model.step(tokens[:, 0], torch.from_numpy(h)[None])
    cases.append({"ids": {k: v.tolist() for k, v in ids.items()}, "floats": {k: v.tolist() for k, v in floats.items()}, "h": h.tolist()})
    expected.append(np.concatenate([[float(win[0])], logits[0].numpy(), h2[0].numpy()]))
path = Path(export_dir) / "rmodel_cases.json"
path.write_text(json.dumps(cases))
out = subprocess.run([binary, "rmodel-check", export_dir, str(path)], capture_output=True, text=True)
lines = [line for line in out.stderr.strip().splitlines() if line.strip()]
single = [np.array([float(v) for v in line.split()]) for line in lines if not line.startswith("B ")]
batch = [np.array([float(v) for v in line.split()[1:]]) for line in lines if line.startswith("B ")]
if len(single) != n or len(batch) != n:
    raise SystemExit(f"harness gave {len(single)} + {len(batch)} lines: {out.stderr[-500:]}")
for name, got in (("one at a time", single), ("as one batch", batch)):
    diff = np.max(np.abs(np.stack(got) - np.stack(expected)), axis=0)
    print(f"{n} random steps, {name}: largest difference win {diff[0]:.2e}, logits {diff[1:10].max():.2e}, next state {diff[10:].max():.2e}")
