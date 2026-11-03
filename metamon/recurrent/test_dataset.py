"""Data-split and window tests: python -m metamon.recurrent.test_dataset"""
import numpy as np

from metamon.recurrent.dataset import battle_id, is_validation, window
from metamon.recurrent.model import FLOAT_KEYS, ID_KEYS


def test_both_points_of_view_land_in_the_same_split():
    a = "gen1ou-1234_1500_alice_vs_bob_01-01-2026_WIN.json.lz4"
    b = "gen1ou-1234_1480_bob_vs_alice_01-01-2026_LOSS.json.lz4"
    assert battle_id(a) == battle_id(b) == "gen1ou-1234"
    assert all(is_validation(a, f) == is_validation(b, f) for f in range(2, 40))
    held = sum(is_validation(f"gen1ou-{i}_x.npz") for i in range(2000))
    assert 60 < held < 140  # about 1 in 20


def _game(n):
    g = {f"ids_{k}": np.ones((n, 2), np.int16) for k in ID_KEYS}
    g |= {f"floats_{k}": np.ones((n, 2), np.float16) for k in FLOAT_KEYS}
    g |= {k: np.ones((n, 9), np.float16) for k in ("blend", "judge", "legal")}
    g |= {"policy_ok": np.ones(n, bool), "win": np.float32(0.5)}
    return g


def test_short_games_are_padded_and_masked():
    w = window(_game(10), 0, 96)
    assert w["valid"].sum() == 10 and w["policy_ok"].sum() == 10
    assert w["ids_own_species"].shape == (96, 2) and (w["blend"][10:] == 0).all()
    assert w["win"] == 0.5


def test_window_cuts_the_requested_range():
    g = _game(200)
    g["blend"] = np.tile(np.arange(200, dtype=np.float16)[:, None], (1, 9))
    w = window(g, 50, 96)
    assert w["blend"][0, 0] == 50 and w["blend"][95, 0] == 145 and w["valid"].all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "tests passed")
