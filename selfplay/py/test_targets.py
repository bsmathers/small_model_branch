"""targets() on a hand-worked game: one game (rows 0 = P1, 1 = P2), three decisions each, every prediction 50/50 (win/loss),
P1 wins after the third. lambda 0.5 for advantages, 0.8 for the outcome targets."""
import sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_selfplay import targets

T, R, G = 4, 2, 1
b = {
    "valid": torch.tensor([[1, 1], [1, 1], [1, 1], [0, 0]], dtype=torch.bool),
    "probs": torch.tensor([[0.5, 0.5, 0.0]]).repeat(T, R, 1),
    "ended": torch.tensor([[0], [0], [1], [0]], dtype=torch.bool),
    "outcome": torch.tensor([[0], [0], [1], [0]], dtype=torch.int8),
}
boot = {"valid": torch.zeros(R, dtype=torch.bool), "first": torch.zeros(R, dtype=torch.bool), "probs": torch.zeros(R, 3)}
adv, tgt = targets(b, boot, 0.5, 0.8)
want_adv = torch.tensor([[0.25, -0.25], [0.5, -0.5], [1.0, -1.0], [0.0, 0.0]])
want_p1 = torch.tensor([[0.82, 0.18, 0.0], [0.9, 0.1, 0.0], [1.0, 0.0, 0.0]])
want_p2 = torch.tensor([[0.18, 0.82, 0.0], [0.1, 0.9, 0.0], [0.0, 1.0, 0.0]])
assert torch.allclose(adv, want_adv), adv
assert torch.allclose(tgt[:3, 0], want_p1), tgt[:3, 0]
assert torch.allclose(tgt[:3, 1], want_p2), tgt[:3, 1]
# a game cut off by the iteration's end bootstraps from the next decision's prediction (here 0.8 win for P1)
b["ended"][:] = False
boot = {"valid": torch.ones(R, dtype=torch.bool), "first": torch.zeros(R, dtype=torch.bool),
        "probs": torch.tensor([[0.8, 0.2, 0.0], [0.2, 0.8, 0.0]])}
b["valid"][3] = False
adv, tgt = targets(b, boot, 0.5, 0.8)
assert torch.allclose(adv[2], torch.tensor([0.6, -0.6])), adv[2]  # (0.8 - 0.2) - 0
print("targets: ok (terminal outcome, both players' sides, bootstrap)")
