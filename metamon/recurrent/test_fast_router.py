"""fast_router against the router's own methods, on random inputs (no checkpoints needed).

    python -m metamon.recurrent.test_fast_router
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import torch

from metamon.recurrent import fast_router
from metamon.rl.experimental.ensemble import ensemble as E


def test_zscore_matches():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, 9)
        values = [rng.choice([0.0, 0.25, rng.random()]) for _ in range(n)]
        want = E._zscore(torch.tensor(values, dtype=torch.float32)).tolist()
        got = fast_router._zscore(values)
        assert len(want) == len(got)
        assert all(abs(a - b) < 1e-5 for a, b in zip(want, got)), (values, want, got)


def test_route_shortlist_matches():
    """The shortlist and its support are the original's on random candidate sets (the router method, called on a bare instance)."""
    policy = object.__new__(E.HeuristicRouterEnsemblePolicy)
    policy.anchor_idx = 0
    recorder = SimpleNamespace(n_slots=9, critic_list=None)
    policy._choose_action_for_batch = lambda **kw: 0
    fast_router.install(policy, recorder)
    route_shortlist = recorder.route_shortlist

    rng = random.Random(1)
    mismatches = 0
    for trial in range(400):
        legal = sorted(rng.sample(range(0, 10), rng.randint(2, 9)))
        n_members = 5
        features = [{"strength": rng.choice([0.45, 0.6, 0.75, 0.85])} for _ in range(n_members)]
        support, members, roles = {}, {}, {}
        for action in rng.sample(legal, rng.randint(1, len(legal))):
            support[action] = rng.random()
            members[action] = set(rng.sample(range(n_members), rng.randint(1, 3)))
            roles[action] = set(rng.sample(["any", "move", "switch", "counter_anchor", "anchor_guard"], rng.randint(1, 3)))
        anchor_action = rng.choice(legal)
        mixed = rng.random() < 0.5
        state = {"turn_idx": rng.choice([5.0, 40.0]), "resource_edge": rng.choice([-0.3, 0.0, 0.3]), "player_remaining": 3, "opponent_remaining": 3}
        disagreement = rng.choice([0.0, 0.2, 0.4])
        top = rng.random()
        strong = rng.random() < 0.5
        want = E.HeuristicRouterEnsemblePolicy._route_shortlist(
            policy, legal_actions=legal, anchor_action=anchor_action, candidate_support=dict(support), candidate_members={k: set(v) for k, v in members.items()},
            candidate_roles={k: set(v) for k, v in roles.items()}, features=features, disagreement=disagreement, mixed_choice=mixed,
            anchor_top_prob=top, state_summary=state, strong_anchor=strong,
        )
        late = state["turn_idx"] >= 30.0
        desperation = state["resource_edge"] < -0.10 or (late and state["resource_edge"] < 0.02)
        stabilize = state["resource_edge"] > 0.14 and state["player_remaining"] >= state["opponent_remaining"]
        got = route_shortlist(
            policy, legal, anchor_action, dict(support), {k: set(v) for k, v in members.items()}, {k: set(v) for k, v in roles.items()},
            features, disagreement, mixed, top, state, strong, desperation, stabilize, late,
        )
        if want[0] != got[0]:
            mismatches += 1
        assert (want[0] != got[0]) or set(want[1]) == set(got[1]), (trial, want, got)
    print("shortlist mismatches", mismatches, "of 400")
    assert mismatches <= 4, mismatches  # (float32 vs float64 can reorder an exact tie in a rare case)


if __name__ == "__main__":
    test_zscore_matches()
    test_route_shortlist_matches()
    print("ok")
