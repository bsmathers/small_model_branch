"""Python side of the self-play environment library (selfplay/zig, libpkrl): many Gen 1 OU games, stepped in Zig, writing
every pending decision into these preallocated tensors in place (pinned host memory when CUDA is available, so the copy to
the GPU is a single asynchronous transfer). One ctypes call per step; no copies, no serialization.

    env = Env(n_games=4096, threads=16, seed=0, tables="…/recurrent_native_stage1",
              teams=("…/duel/teamset", "…/selfplay/data/teams/hl_05_26/gen1ou"), share_a=0.5)
    obs = env.reset()                    # dict of tensors, rows 2g + p (game g, player p)
    obs = env.step(actions)              # actions: uint8 tensor [rows], a policy slot per valid row

Rows (see selfplay/zig/src/env.zig): valid, forced, first [R]; ids [R, IDS] int16; floats [R, FLOATS] float32; events
[R, MAX_EVENTS, FIELDS] uint8 with n_events [R]; legal [R, 9]. Per game: ended, outcome (P1's side: 1, -1, 0), turn.
The tensors are reused by the next call: copy (e.g. to the GPU) what must be kept.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEFAULT_LIB = HERE.parent / "zig" / "zig-out" / "lib" / ("libpkrl.dylib" if os.uname().sysname == "Darwin" else "libpkrl.so")


class _Layout(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("ids", "floats", "max_events", "fields", "slots")]


class _Buffers(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("valid", "forced", "first", "ids", "floats", "events", "n_events", "legal",
                                                 "ended", "outcome", "turn")]


STAT_FIELDS = ("decisions", "games", "p1_wins", "ties", "cut_off", "invalid_actions", "dropped_events", "decode_errors",
               "engine_errors")


class _Stats(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in STAT_FIELDS]


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.pkrl_layout.argtypes = [ctypes.POINTER(_Layout)]
    lib.pkrl_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_char_p, ctypes.c_float, ctypes.c_uint32]
    lib.pkrl_create.restype = ctypes.c_void_p
    lib.pkrl_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Buffers)]
    lib.pkrl_reset.restype = ctypes.c_int32
    lib.pkrl_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_Buffers)]
    lib.pkrl_step.restype = ctypes.c_int32
    lib.pkrl_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stats)]
    lib.pkrl_destroy.argtypes = [ctypes.c_void_p]
    lib.pkrl_error.restype = ctypes.c_char_p
    lib.pkrl_pool_sizes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32 * 2)]
    return lib


class Env:
    def __init__(self, n_games: int, threads: int, seed: int, tables: str, teams: tuple[str, ...], share_a: float = 0.5,
                 max_turns: int = 1000, lib: str | None = None, pin: bool | None = None):
        self.lib = _load(Path(lib) if lib else DEFAULT_LIB)
        layout = _Layout()
        self.lib.pkrl_layout(ctypes.byref(layout))
        self.IDS, self.FLOATS, self.MAX_EVENTS, self.FIELDS, self.SLOTS = (layout.ids, layout.floats, layout.max_events,
                                                                            layout.fields, layout.slots)
        teams_a = str(teams[0]).encode()
        teams_b = str(teams[1]).encode() if len(teams) > 1 and teams[1] else b""
        self.handle = self.lib.pkrl_create(n_games, threads, seed, str(tables).encode(), teams_a, teams_b, share_a, max_turns)
        if not self.handle:
            raise RuntimeError(self.lib.pkrl_error().decode())
        self.n_games, self.rows = n_games, 2 * n_games
        pin = torch.cuda.is_available() if pin is None else pin
        R, G = self.rows, n_games

        def t(shape, dtype):
            return torch.zeros(shape, dtype=dtype, pin_memory=pin)

        self.obs = {
            "valid": t((R,), torch.uint8), "forced": t((R,), torch.uint8), "first": t((R,), torch.uint8),
            "ids": t((R, self.IDS), torch.int16), "floats": t((R, self.FLOATS), torch.float32),
            "events": t((R, self.MAX_EVENTS, self.FIELDS), torch.uint8), "n_events": t((R,), torch.uint8),
            "legal": t((R, self.SLOTS), torch.uint8),
            "ended": t((G,), torch.uint8), "outcome": t((G,), torch.int8), "turn": t((G,), torch.int16),
        }
        self._bufs = _Buffers(*(ctypes.c_void_p(self.obs[n].data_ptr()) for n, _ in _Buffers._fields_))
        self._actions = t((R,), torch.uint8)

    def reset(self) -> dict:
        if self.lib.pkrl_reset(self.handle, ctypes.byref(self._bufs)) != 0:
            raise RuntimeError(self.lib.pkrl_error().decode())
        return self.obs

    def step(self, actions: torch.Tensor) -> dict:
        """actions: [rows] uint8 (a policy slot per valid, unforced row), on any device."""
        if actions.device.type != "cpu" or actions.dtype != torch.uint8:
            self._actions.copy_(actions.to(torch.uint8), non_blocking=False)
            ptr = self._actions.data_ptr()
        else:
            ptr = actions.contiguous().data_ptr()
        if self.lib.pkrl_step(self.handle, ctypes.c_void_p(ptr), ctypes.byref(self._bufs)) != 0:
            raise RuntimeError(self.lib.pkrl_error().decode())
        return self.obs

    def pool_sizes(self) -> tuple[int, int]:
        out = (ctypes.c_uint32 * 2)()
        self.lib.pkrl_pool_sizes(self.handle, ctypes.byref(out))
        return out[0], out[1]

    def stats(self) -> dict:
        s = _Stats()
        self.lib.pkrl_stats(self.handle, ctypes.byref(s))
        return {n: getattr(s, n) for n in STAT_FIELDS}

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.pkrl_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def random_actions(obs: dict, generator: torch.Generator | None = None) -> torch.Tensor:
    """A uniformly random legal slot per row (rows without a legal slot get 0)."""
    legal = obs["legal"].bool()
    noise = torch.rand(legal.shape, generator=generator).masked_fill(~legal, -1.0)
    return noise.argmax(-1).to(torch.uint8)
