"""Training windows from built games (see build.py), mixed across sources.

A source is a directory of built games with a mixing weight. The weight is a share of the scored decisions,
and inside a source every decision is scored equally often (see plan_window and draw_weights).

A window has `burn_in + chunk` steps. A game that fits in one window is scored whole (padded at the end, the
padding masked out of every loss). A longer game is scored a chunk at a time: the chunk's first decision c is
uniform over [-(chunk - 1), n - 1], the chunk is cut to the game, and up to `burn_in` decisions before it only
build the recurrent state and carry no loss. Every decision of the game is then in `chunk` of the n + chunk - 1
possible chunks, the first and last ones included (a chunk start drawn among the full windows alone would score
the beginning and the end of a long game only a few times).
"""

from __future__ import annotations

import glob
import os
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from metamon.recurrent.features import check_encoding
from metamon.recurrent.model import FLOAT_KEYS, ID_KEYS


@dataclass
class Source:
    name: str
    paths: list[str]
    lengths: np.ndarray
    weight: float


def battle_id(path: str) -> str:
    """The game a file belongs to: the parser's names start with the battle id, and a game parsed from
    both players' points of view gives two files with the same id."""
    return os.path.basename(path).split("_")[0]


def is_validation(path: str, fraction: int = 20) -> bool:
    """A game is held out by a hash of its battle id (1 in `fraction`), the same on every machine and for
    both points of view of a game."""
    return zlib.crc32(battle_id(path).encode()) % fraction == 0


def scan(name: str, directory: str, weight: float, validation: bool, fraction: int = 20) -> Source:
    check_encoding(directory)  # (games built by code with another feature layout must not be trained on)
    paths =[p for p in sorted(glob.glob(os.path.join(directory, "**", "*.npz"), recursive=True)) if is_validation(p, fraction) == validation]
    if not paths:
        raise ValueError(f"source {name}: no {'validation' if validation else 'training'} games in {directory}")
    lengths = np.array([_length(p) for p in paths])
    return Source(name, paths, lengths, weight)


def _length(path: str) -> int:
    with np.load(path) as z:
        return int(z["legal"].shape[0])


def load_game(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def window(game: dict, start: int, size: int) -> dict[str, np.ndarray]:
    """One window as arrays of length `size` (padded at the end), with a validity mask."""
    n = game["legal"].shape[0]
    end = min(start + size, n)
    out: dict[str, np.ndarray] = {}

    def cut(a: np.ndarray) -> np.ndarray:
        piece = a[start:end]
        if len(piece) < size:
            piece = np.concatenate([piece, np.zeros((size - len(piece),) + piece.shape[1:], piece.dtype)])
        return piece

    for k in ID_KEYS:
        out[f"ids_{k}"] = cut(game[f"ids_{k}"]).astype(np.int64)
    for k in FLOAT_KEYS:
        out[f"floats_{k}"] = cut(game[f"floats_{k}"]).astype(np.float32)
    for k in ("blend", "judge", "legal"):
        out[k] = cut(game[k]).astype(np.float32)
    valid = np.zeros(size, bool)
    valid[: end - start] = True
    out["valid"] = valid
    out["policy_ok"] = cut(game["policy_ok"].astype(bool)) & valid
    # the action actually played, on games of the imitated bots (older built games have neither key)
    played = cut(game["played"].astype(np.int64)) if "played" in game else np.full(size, -1, np.int64)
    played = np.where(valid, played, -1)
    out["played"] = played
    out["bc_ok"] = (played >= 0) & bool(game["bc_game"] if "bc_game" in game else False)
    out["start"] = np.array(start)
    out["win"] = np.array(float(game["win"]), np.float32)
    return out


def plan_window(n: int, burn_in: int, chunk: int, rng) -> tuple[int, int, int]:
    """A training window of a game of n decisions: (start in the game, scored steps [lo, hi) within the window)."""
    if n <= burn_in + chunk:
        return 0, 0, n
    c = int(rng.integers(-(chunk - 1), n))  # the chunk's first decision; before the game start, the chunk is cut
    first, last = max(c, 0), min(c + chunk, n)
    start = max(first - burn_in, 0)
    return start, first - start, last - start


def draw_weights(lengths: np.ndarray, burn_in: int, chunk: int) -> np.ndarray:
    """How often each game is drawn (unnormalised), so that each of its decisions is scored equally often: a game that fits
    in a window is scored whole in every draw, a longer one scores `chunk` of its decisions in n + chunk - 1 equally likely
    chunks, so it is drawn (n + chunk - 1) / chunk times as often."""
    lengths = np.asarray(lengths, float)
    return np.where(lengths <= burn_in + chunk, 1.0, (lengths + chunk - 1) / chunk)


class Windows(IterableDataset):
    """An endless stream of windows (training), or one pass over fixed windows (validation)."""

    def __init__(self, sources: list[Source], burn_in: int, chunk: int, seed: int = 0, endless: bool = True):
        self.sources, self.burn_in, self.chunk, self.seed, self.endless = sources, burn_in, chunk, seed, endless
        self.size = burn_in + chunk

    def __iter__(self):
        info = get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)
        if not self.endless:
            yield from self._validation(wid, nworkers)
            return
        rng = np.random.default_rng([self.seed, wid])
        games = [draw_weights(s.lengths, self.burn_in, self.chunk) for s in self.sources]
        # a draw from a source scores (its decisions / its total draw weight) decisions on average: the source is picked
        # in inverse proportion to that, so that its share of the scored decisions is its weight
        weights = np.array([s.weight * g.sum() / s.lengths.sum() for s, g in zip(self.sources, games)], float)
        weights /= weights.sum()
        games = [g / g.sum() for g in games]
        while True:
            k = rng.choice(len(self.sources), p=weights)
            src = self.sources[k]
            i = rng.choice(len(src.paths), p=games[k])
            start, lo, hi = plan_window(int(src.lengths[i]), self.burn_in, self.chunk, rng)
            yield self._item(load_game(src.paths[i]), start, lo, hi, k)

    def _validation(self, wid: int, nworkers: int):
        """Every game, covered by consecutive windows that score each decision once."""
        for src_idx, src in enumerate(self.sources):
            for i, path in enumerate(src.paths):
                if (i + src_idx) % nworkers != wid:
                    continue
                game = load_game(path)
                n = int(src.lengths[i])
                start = 0
                while True:
                    yield self._item(game, start, 0 if start == 0 else self.burn_in, self.size, src_idx)
                    if start + self.size >= n:
                        break
                    start += self.chunk

    def _item(self, game: dict, start: int, lo: int, hi: int, source: int) -> dict:
        w = window(game, start, self.size)
        # the steps before lo only build the recurrent state (lo is 0 when the window starts the game)
        steps = np.arange(self.size)
        w["scored"] = w["valid"] & (steps >= lo) & (steps < hi)
        w["source"] = np.array(source)
        return w


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0]}
