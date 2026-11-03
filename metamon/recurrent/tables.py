"""Lookup tables for the recurrent model's features: integer ids and Gen 1 game data.

The JSON files are generated once on the machine that has the engine's data (see `Tables.load`),
so that the Python and Zig sides use the same ids and numbers:
  species_ids.json : {"name": id}   ids from 1; 0 is the empty slot
  move_ids.json    : {"name": id}   ids from 1; 0 is "none"
  base_stats.json  : {"name": {"hp": h, "atk": a, "def": d, "spe": s, "spa": x}}
  moves.json       : {"name": [base_power, "type"]}
  type_chart.json  : {"attacker": {"defender": multiplier}}  (multipliers 0, 0.5, 1 or 2)
Names are matched after normalisation (lowercase, letters and digits only).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass
class Tables:
    species_ids: dict[str, int]
    move_ids: dict[str, int]
    base: dict[str, dict[str, int]]
    moves: dict[str, tuple[int, str]]
    chart: dict[str, dict[str, float]]

    @classmethod
    def load(cls, directory: str | Path) -> "Tables":
        d = Path(directory)

        def read(file: str):
            data = json.loads((d / file).read_text())
            return {normalize(k): v for k, v in data.items()}

        return cls(
            species_ids=read("species_ids.json"),
            move_ids=read("move_ids.json"),
            base=read("base_stats.json"),
            moves={k: (int(v[0]), normalize(v[1])) for k, v in read("moves.json").items()},
            chart={normalize(a): {normalize(b): float(m) for b, m in row.items()} for a, row in json.loads((d / "type_chart.json").read_text()).items()},
        )

    @classmethod
    def from_dex(cls, gen: int = 1) -> "Tables":
        """Tables from metamon's bundled Showdown dex. Ids are the 1-based rank of the normalised name
        in sorted order, so they do not depend on any data sample; the Zig side loads the JSON written
        by `write` so that both sides use the same ids."""
        from metamon.backend.showdown_dex.dex import Dex

        dex = Dex(gen)
        pokedex = dex.load_pokedex(gen)
        base = {}
        for key, entry in pokedex.items():
            if not 1 <= entry.get("num", 0) <= 151 or "forme" in entry:
                continue  # the dex also lists later-generation species and alternate formes (megas...)
            stats = entry["baseStats"]
            base[normalize(key)] = {"hp": stats["hp"], "atk": stats["atk"], "def": stats["def"], "spe": stats["spe"], "spa": stats["spa"]}
        species = sorted(base)
        move_data = dex.load_moves(gen)
        moves = {normalize(k): (int(v.get("basePower", 0)), normalize(v.get("type", "normal"))) for k, v in move_data.items()}
        chart_raw = dex.load_type_chart(gen)  # chart_raw[defender][attacker]
        chart: dict[str, dict[str, float]] = {}
        for defender, row in chart_raw.items():
            for attacker, mult in row.items():
                chart.setdefault(normalize(attacker), {})[normalize(defender)] = float(mult)
        return cls(
            species_ids={n: i + 1 for i, n in enumerate(species)},
            move_ids={n: i + 1 for i, n in enumerate(sorted(moves))},
            base=base,
            moves=moves,
            chart=chart,
        )

    def write(self, directory: str | Path) -> None:
        """Writes the JSON files that `load` reads (and that the Zig side reads)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "species_ids.json").write_text(json.dumps(self.species_ids, indent=1, sort_keys=True))
        (d / "move_ids.json").write_text(json.dumps(self.move_ids, indent=1, sort_keys=True))
        (d / "base_stats.json").write_text(json.dumps(self.base, indent=1, sort_keys=True))
        (d / "moves.json").write_text(json.dumps({k: [v[0], v[1]] for k, v in self.moves.items()}, indent=1, sort_keys=True))
        (d / "type_chart.json").write_text(json.dumps(self.chart, indent=1, sort_keys=True))

    def species_id(self, name: str) -> int:
        key = normalize(name)
        if key == "":
            return 0
        try:
            return self.species_ids[key]
        except KeyError as exc:
            raise KeyError(f"unknown species {name!r}") from exc

    def move_id(self, name: str) -> int:
        key = normalize(name)
        if key in ("", "none", "nomove"):
            return 0
        try:
            return self.move_ids[key]
        except KeyError as exc:
            raise KeyError(f"unknown move {name!r}") from exc

    def base_stats(self, name: str) -> dict[str, int]:
        try:
            return self.base[normalize(name)]
        except KeyError as exc:
            raise KeyError(f"no base stats for {name!r}") from exc

    def move_power_type(self, name: str) -> tuple[int, str]:
        key = normalize(name)
        if key in ("", "none", "nomove"):
            return (0, "notype")
        try:
            return self.moves[key]
        except KeyError as exc:
            raise KeyError(f"no move data for {name!r}") from exc

    def effectiveness(self, attack_type: str, defend_type: str) -> float:
        if defend_type in ("", "notype", "typeless"):
            return 1.0
        return self.chart.get(normalize(attack_type), {}).get(normalize(defend_type), 1.0)


if __name__ == "__main__":
    import sys

    out = sys.argv[1]
    Tables.from_dex(1).write(out)
    print(f"wrote Gen 1 tables to {out}")
