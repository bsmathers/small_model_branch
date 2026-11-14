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


def test_every_decision_is_scored_equally_often():
    """Across games of different lengths, both sides of the window size, and two sources, every decision is scored at the
    same rate, and each source's share of the scored decisions is its weight."""
    import os
    import tempfile

    from metamon.recurrent.dataset import Source, Windows

    burn_in, chunk = 4, 8
    lengths = {"a": [3, 10, 12, 30], "b": [7, 50]}
    with tempfile.TemporaryDirectory() as d:
        sources = []
        for name, weight in (("a", 0.25), ("b", 0.75)):
            paths = []
            for j, n in enumerate(lengths[name]):
                g = _game(n)
                g["blend"] = np.tile(np.arange(n, dtype=np.float16)[:, None], (1, 9))  # the decision's index,
                g["blend"][:, 1] = j  # and the game's
                paths.append(os.path.join(d, f"{name}{j}.npz"))
                np.savez(paths[-1], **g)
            sources.append(Source(name, paths, np.array(lengths[name]), weight))
        counts = {(s.name, j): np.zeros(n) for s in sources for j, n in enumerate(s.lengths)}
        stream = iter(Windows(sources, burn_in, chunk, seed=1))
        for _ in range(60000):
            w = next(stream)
            s = sources[int(w["source"])]
            j = int(w["blend"][0, 1])
            counts[(s.name, j)][w["blend"][w["scored"], 0].astype(int)] += 1
    allc = np.concatenate(list(counts.values()))
    assert allc.min() > 0
    for name in ("a", "b"):
        per = np.concatenate([c for (n, _), c in counts.items() if n == name])
        assert abs(per.std() / per.mean()) < 0.08, (name, per)  # flat inside a source, ends of the long games included
    share_a = sum(c.sum() for (n, _), c in counts.items() if n == "a") / allc.sum()
    assert abs(share_a - 0.25) < 0.02, share_a


def test_plan_window_cuts_chunks_at_the_ends_of_long_games():
    from metamon.recurrent.dataset import plan_window

    class R:
        def __init__(self, c):
            self.c = c

        def integers(self, lo, hi):
            assert lo <= self.c < hi
            return self.c

    assert plan_window(10, 4, 8, R(0)) == (0, 0, 10)  # fits in a window: scored whole
    assert plan_window(30, 4, 8, R(-7)) == (0, 0, 1)  # the chunk [-7, 1) is cut to the first decision
    assert plan_window(30, 4, 8, R(2)) == (0, 2, 10)  # two steps of context before it
    assert plan_window(30, 4, 8, R(10)) == (6, 4, 12)  # a full burn-in
    assert plan_window(30, 4, 8, R(29)) == (25, 4, 5)  # the last decision alone


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "tests passed")
