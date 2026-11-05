from types import SimpleNamespace

import torch
import torch.nn as nn

from metamon.recurrent import label


class FakeActor(nn.Module):
    def forward(self, emb, straight_from_obs=None):
        # probs [lanes, T, gammas=1, actions=3] = a function of the embedding
        return SimpleNamespace(probs=torch.softmax(emb[..., :1].unsqueeze(2).expand(-1, -1, 1, 3) * torch.tensor([1.0, 2.0, 3.0]), dim=-1))


class FakePolicy:
    pass_obs_keys_to_actor = ["illegal_actions"]

    def __init__(self):
        self.actor = FakeActor()
        self.calls = 0

    def get_state_embedding(self, obs, rl2s, time_idxs, hidden_state=None):
        self.calls += 1
        return obs["x"] + time_idxs.float(), hidden_state


def _steps(lanes, horizon):
    out = []
    for t in range(horizon):
        batch = {"x": torch.full((lanes, 1, 1), float(t) + 0.5), "illegal_actions": torch.zeros(lanes, 1, 3, dtype=torch.bool)}
        out.append((batch, torch.zeros(lanes, 1, 4), [[0, 1]] * lanes, None))
    return out


def test_member_outputs_are_served_per_step_and_restored():
    policy = SimpleNamespace(members=[SimpleNamespace(policy=FakePolicy(), device=torch.device("cpu"))])
    mp = policy.members[0].policy
    original_forward = mp.actor.forward
    steps = _steps(lanes=3, horizon=5)
    restore = label.member_sequence_outputs(policy, steps)
    assert mp.calls == 1  # the whole game in one call
    for t in range(5):
        mp._label_step = t
        emb, hidden = mp.get_state_embedding(obs=None, rl2s=None, time_idxs=None, hidden_state="h")
        assert hidden == "h" and emb.shape == (3, 1, 1) and torch.all(emb == 2 * t + 0.5)
        probs = mp.actor(emb).probs
        assert probs.shape == (3, 1, 1, 3)
        expected = torch.softmax(torch.tensor([2 * t + 0.5]) * torch.tensor([1.0, 2.0, 3.0]), dim=-1)
        assert torch.allclose(probs[0, 0, 0], expected)
    restore()
    assert mp.actor.forward == original_forward and "get_state_embedding" not in vars(mp)
    emb, _ = mp.get_state_embedding(obs={"x": torch.zeros(1, 1, 1)}, rl2s=None, time_idxs=torch.zeros(1, 1, 1))
    assert mp.calls == 2  # the real method again


def test_long_games_run_in_lane_chunks_with_the_same_result():
    steps = _steps(lanes=5, horizon=4)
    results = []
    for rows in (10_000, 8):  # one chunk, then 2 lanes per chunk
        label.SEQUENCE_ROWS = rows
        policy = SimpleNamespace(members=[SimpleNamespace(policy=FakePolicy(), device=torch.device("cpu"))])
        mp = policy.members[0].policy
        restore = label.member_sequence_outputs(policy, steps)
        mp._label_step = 2
        results.append((mp.get_state_embedding(None, None, None)[0].clone(), mp.actor(None).probs.clone(), mp.calls))
        restore()
    label.SEQUENCE_ROWS = 24576
    assert torch.equal(results[0][0], results[1][0]) and torch.allclose(results[0][1], results[1][1])
    assert results[0][2] == 1 and results[1][2] == 3  # (5 lanes, 2 per chunk)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
