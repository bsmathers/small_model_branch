"""fast_tstep checks: python -m metamon.recurrent.test_fast_tstep"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from metamon.recurrent import fast_tstep


def _policy():
    from amago.nets.transformer import SigmaReparam

    torch.manual_seed(0)
    module = nn.Sequential(SigmaReparam(24, 40), SigmaReparam(40, 8)).eval()
    return SimpleNamespace(members=[SimpleNamespace(device=torch.device("cpu"), policy=module)]), module


def test_frozen_sigma_reparam_equals_the_original_layer_in_fp32_and_under_autocast():
    for half in (None, torch.bfloat16):
        policy, module = _policy()
        x = torch.randn(5, 24)
        with torch.no_grad(), torch.autocast("cpu", dtype=half, enabled=half is not None):
            want = module(x)  # the original layers: sigma and the effective weight rebuilt on the call
        assert fast_tstep.freeze_sigma_reparam(policy, half) == 2
        with torch.no_grad(), torch.autocast("cpu", dtype=half, enabled=half is not None):
            got = module(x)
        assert torch.equal(got, want), (half, (got - want).abs().max())


def test_independent_pokemon_encoding_reuses_unchanged_pokemon():
    # the cache bookkeeping on its own: a Pokemon is re-encoded only when its tokens or numbers change
    calls = []

    class Enc(nn.Module):
        NUM_POKEMON = 3
        _pokemon_role_emb = None

        def __init__(self):
            super().__init__()
            self.pokemon_proj = nn.Linear(1, 1)

    enc = Enc()
    original = fast_tstep._encode_independent
    fast_tstep._encode_independent = lambda e, text, nums, slots: (calls.append(len(text)), text.float().sum(-1, keepdim=True) + nums.sum(-1, keepdim=True))[1]
    try:
        # build only the cache closure by calling _patch on an encoder that has the attributes it needs
        for name in ("extra_emb", "entity_type_emb", "fusion", "fusion_out_norm"):
            setattr(enc, name, nn.Identity())
        enc._entity_type_ids = torch.zeros(1)
        enc._encode_global = lambda *a, **k: None
        fast_tstep._patch(enc)
        cache_fn = enc._pokemon_cache
        # reach the closure through the patched implementation's cell contents
        pokemon_embeddings = [c.cell_contents for c in enc._fast_impl.__closure__ if callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "pokemon_embeddings"][0]
        text = torch.arange(2 * 3 * 4).view(2, 3, 4)
        nums = torch.zeros(2, 3, 2)
        first = pokemon_embeddings(text, nums)
        assert calls == [6] and first.shape == (2, 3, 1)
        text2 = text.clone()
        text2[1, 2] += 1  # one Pokemon changes
        second = pokemon_embeddings(text2, nums)
        assert calls == [6, 1]
        assert torch.equal(second[0], first[0]) and torch.equal(second[1, :2], first[1, :2]) and not torch.equal(second[1, 2], first[1, 2])
        fast_tstep.reset(SimpleNamespace(members=[SimpleNamespace(policy=SimpleNamespace(tstep_encoder=enc))]))
        pokemon_embeddings(text2, nums)
        assert calls == [6, 1, 6]  # a new batch of games starts from nothing
    finally:
        fast_tstep._encode_independent = original


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
