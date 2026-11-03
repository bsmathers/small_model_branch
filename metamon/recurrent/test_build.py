"""Decision kinds, adjusted options and played-action codes: python -m metamon.recurrent.test_build"""
from types import SimpleNamespace as NS

import numpy as np

from metamon.recurrent.build import (
    FIGHT, FIGHT_OR_SWITCH, FORCED_RECHARGE, FORCED_TRAPPED, NO_CHOICE, NORMAL, adjust, decision_kind, played_code,
)


def dec(**kw):
    base = dict(forced_switch=False, recharge_next=False, trapped=False, action_index=0, action_reason=None,
                own_status="nostatus", slot_map=np.array([0, 1, 2, 3, 4, 5, 6, 7, -1]))
    base.update(kw)
    return NS(**base)


def test_kinds():
    assert decision_kind(dec()) == NORMAL
    assert decision_kind(dec(own_status="slp", action_index=-1)) == FIGHT_OR_SWITCH
    assert decision_kind(dec(own_status="frz")) == FIGHT_OR_SWITCH
    assert decision_kind(dec(recharge_next=True)) == FORCED_RECHARGE
    assert decision_kind(dec(recharge_next=True, own_status="slp")) == FORCED_RECHARGE  # forced beats asleep
    assert decision_kind(dec(trapped=True, own_status="slp")) == FORCED_TRAPPED
    assert decision_kind(dec(action_index=-1, action_reason="game_ended")) == NO_CHOICE
    assert decision_kind(dec(action_index=-1, action_reason="par")) == NORMAL  # paralyzed: a real choice that then failed
    assert decision_kind(dec(forced_switch=True, trapped=True, own_status="slp")) == NORMAL  # choosing a replacement


def row(moves, switches):
    r = np.zeros(9, np.float32)
    r[:4], r[4:4 + len(switches)] = moves, switches
    return r


def test_asleep_spreads_the_fight_probability_evenly_over_the_legal_moves():
    legal = row([1, 1, 1, 0], [1, 1, 0, 0, 0])  # three legal moves (one slot empty), two switches
    blend = row([0.5, 0.1, 0.0, 0.0], [0.3, 0.1])
    legal2, blend2, _ = adjust(FIGHT_OR_SWITCH, legal, blend, blend)
    assert np.allclose(blend2[:4], [0.2, 0.2, 0.2, 0.0]) and np.allclose(blend2[4:6], [0.3, 0.1])
    assert abs(blend2.sum() - 1) < 1e-6 and (legal2 == legal).all()


def test_recharge_leaves_only_the_no_op_and_trapped_only_the_moves():
    legal = row([1, 1, 1, 1], [1, 1, 1, 0, 0])
    r, _, _ = adjust(FORCED_RECHARGE, legal, legal / legal.sum(), legal / legal.sum())
    assert list(r) == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    t, _, _ = adjust(FORCED_TRAPPED, legal, legal / legal.sum(), legal / legal.sum())
    assert list(t[:4]) == [1, 1, 1, 1] and t[4:].sum() == 0


def test_played_codes():
    legal = row([1, 1, 1, 1], [1, 1, 1, 0, 0])
    assert played_code(dec(action_index=-1), NORMAL, legal) == FIGHT  # paralyzed / flinched: some move
    assert played_code(dec(action_index=2), NORMAL, legal) == 2
    assert played_code(dec(action_index=1), FIGHT_OR_SWITCH, legal) == FIGHT  # any move while asleep is the same
    assert played_code(dec(action_index=5), FIGHT_OR_SWITCH, legal) == 5  # switching out stays a switch
    assert played_code(dec(action_index=-1), FORCED_RECHARGE, legal) == -1
    assert played_code(dec(action_index=0), FORCED_RECHARGE, legal) == -1  # the no-op is not a choice
    assert played_code(dec(action_index=-1), NO_CHOICE, legal) == -1
    assert played_code(dec(action_index=-1, forced_switch=True), NORMAL, row([0, 0, 0, 0], [1, 1, 0, 0, 0])) == -1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "tests passed")
