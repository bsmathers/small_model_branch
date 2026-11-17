"""The optional heads (deep actor, opponent head), policy blending and the opponent-action labels."""

import numpy as np
import pytest
import torch

from metamon.recurrent.blend import PolicyBlend, parse_spec
from metamon.recurrent.dataset import N_MOVE_IDS, opponent_actions
from metamon.recurrent.model import Config, PolicyValueModel


def inputs(seed: int = 0, t: int = 3):
    g = torch.Generator().manual_seed(seed)
    shapes = {"own_species": (6,), "own_status": (6,), "own_moves": (6, 4), "opp_species": (6,), "opp_status": (6,), "opp_moves": (6, 4)}
    from metamon.recurrent.model import ID_KEYS

    ids = {k: torch.randint(0, 8, (1, t, *shapes.get(k, (1,))), generator=g) for k in ID_KEYS}
    floats = {k: torch.rand(1, t, *w, generator=g) for k, w in (("own", (6, 10)), ("opp", (6, 7)), ("last", (34,)), ("field", (39,)))}
    return ids, floats


def test_new_heads_leave_the_policy_unchanged():
    torch.manual_seed(0)
    base = PolicyValueModel().eval()
    grown = PolicyValueModel.from_state(base.state_dict(), deep_actor=True, opponent_head=True).eval()
    ids, floats = inputs()
    with torch.no_grad():
        a, wa, _ = base(ids, floats)
        b, wb, _, states = grown(ids, floats, states=True)
    assert torch.equal(a, b) and torch.equal(wa, wb)
    assert grown.opponent(states).shape == (1, 3, Config().n_opponent)
    # a checkpoint with the heads gets them back without asking
    again = PolicyValueModel.from_state({"model": grown.state_dict()})
    assert again.cfg.deep_actor and again.cfg.opponent_head


def test_from_state_rejects_other_weights():
    state = PolicyValueModel().state_dict()
    state.pop("policy.weight")
    with pytest.raises(RuntimeError):
        PolicyValueModel.from_state(state)


def test_blend(tmp_path):
    torch.manual_seed(1)
    paths = []
    for i in range(2):
        m = PolicyValueModel()
        torch.nn.init.normal_(m.policy.weight, std=0.5)
        paths.append(str(tmp_path / f"m{i}.pt"))
        torch.save(m.state_dict(), paths[-1])
    assert parse_spec(f"{paths[0]}@0.25,{paths[1]}") == [(paths[0], 0.25), (paths[1], 1.0)]
    ids, floats = inputs(t=1)
    legal = [0, 2, 5]
    one = PolicyBlend.load(paths[0])
    each = [PolicyBlend.load(p).decide(ids, floats, PolicyBlend.load(p).initial_state(), legal)[0].exp() for p in paths]
    p0, _ = one.decide(ids, floats, one.initial_state(), legal)
    assert torch.allclose(p0.exp(), each[0]) and abs(p0.exp().sum().item() - 1) < 1e-5
    assert torch.all(p0.exp()[0, [1, 3, 4, 6, 7, 8]] == 0)
    mix = PolicyBlend.load(f"{paths[0]}@0.25,{paths[1]}@0.75")
    pm, h = mix.decide(ids, floats, mix.initial_state(), legal)
    assert len(h) == 2
    assert torch.allclose(pm.exp(), 0.25 * each[0] + 0.75 * each[1], atol=1e-6)
    geo = PolicyBlend.load(f"{paths[0]}@0.5,{paths[1]}@0.5", mode="logit")
    pg, _ = geo.decide(ids, floats, geo.initial_state(), legal)
    expect = (each[0][0, legal] * each[1][0, legal]).sqrt()
    assert torch.allclose(pg.exp()[0, legal], expect / expect.sum(), atol=1e-6)


def test_opponent_actions():
    n = 4
    game = {
        "legal": np.ones((n, 9)),
        "ids_last_opp_move": np.array([[0], [55], [0], [0]]),  # the record at decision t is the turn before it
        "ids_last_opp_species": np.array([[0], [0], [12], [0]]),
    }
    assert opponent_actions(game).tolist() == [55, N_MOVE_IDS + 12, -1, -1]
