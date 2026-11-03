"""Model checks: python -m metamon.recurrent.test_model"""
import torch

from metamon.recurrent.model import Config, PolicyValueModel, count_parameters


def fake_batch(b=3, t=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    cfg = Config()
    r = lambda hi, *s: torch.randint(0, hi, s, generator=g)
    ids = {
        "own_species": r(cfg.n_species, b, t, 6), "own_status": r(8, b, t, 6), "own_moves": r(cfg.n_moves, b, t, 6, 4),
        "opp_species": r(cfg.n_species, b, t, 6), "opp_status": r(8, b, t, 6), "opp_moves": r(cfg.n_moves, b, t, 6, 4),
    }
    for who in ("own", "opp"):
        ids[f"last_{who}_move"] = r(cfg.n_moves, b, t, 1)
        ids[f"last_{who}_species"] = r(cfg.n_species, b, t, 1)
        ids[f"last_{who}_status"] = r(8, b, t, 1)
    n = lambda *s: torch.randn(*s, generator=g)
    floats = {"own": n(b, t, 6, 10), "opp": n(b, t, 6, 7), "last": n(b, t, 34), "field": n(b, t, 39)}
    return ids, floats


def test_parameter_count():
    assert count_parameters(PolicyValueModel()) == 1_310_698 + 192 * 8  # field 31 wide plus 8 effect inputs


def test_shapes_and_burn_in():
    m = PolicyValueModel().eval()
    ids, floats = fake_batch()
    logits, wins, h = m(ids, floats)
    assert logits.shape == (3, 7, 9) and wins.shape == (3, 7) and h.shape == (3, 192)
    logits2, wins2, h2 = m(ids, floats, burn_in=3)
    assert logits2.shape == (3, 4, 9)
    # burn-in only changes what is returned, not the computation
    assert torch.allclose(logits[:, 3:], logits2, atol=1e-5) and torch.allclose(h, h2, atol=1e-5)


def test_chunked_equals_whole():
    m = PolicyValueModel().eval()
    ids, floats = fake_batch()
    logits, _, h = m(ids, floats)
    cut = lambda d, a, b: {k: v[:, a:b] for k, v in d.items()}
    l1, _, h1 = m(cut(ids, 0, 4), cut(floats, 0, 4))
    l2, _, h2 = m(cut(ids, 4, 7), cut(floats, 4, 7), h0=h1)
    assert torch.allclose(logits[:, 4:], l2, atol=1e-5) and torch.allclose(h, h2, atol=1e-5)


def test_history_matters_and_batch_independent():
    m = PolicyValueModel().eval()
    ids, floats = fake_batch(b=2)
    a, _, _ = m(ids, floats)
    ids1 = {k: v[:1] for k, v in ids.items()}
    f1 = {k: v[:1] for k, v in floats.items()}
    b, _, _ = m(ids1, f1)
    assert torch.allclose(a[:1], b, atol=1e-5)  # no leakage across games in a batch
    ids2 = {k: v.clone() for k, v in ids.items()}
    ids2["opp_species"][:, 0] = (ids2["opp_species"][:, 0] + 1) % 152
    c, _, _ = m(ids2, floats)
    assert not torch.allclose(a[:, -1], c[:, -1], atol=1e-5)  # a change at step 0 reaches step 6 through h


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "tests passed")
