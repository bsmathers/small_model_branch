"""Decision kinds, adjusted options and played-action codes: python -m metamon.recurrent.test_build"""
from types import SimpleNamespace as NS

import numpy as np

from metamon.recurrent.records import lock_shape
from metamon.recurrent.build import (
    FIGHT, FIGHT_OR_SWITCH, FORCED_RECHARGE, NO_CHOICE, NORMAL, adjust, decision_kind, played_code,
)


def dec(**kw):
    base = dict(forced_switch=False, lock_shape=None, struggling=False, disabled_move=None, action_index=0, action_reason=None,
                own_status="nostatus", slot_map=np.array([0, 1, 2, 3, 4, 5, 6, 7, -1]))
    base.update(kw)
    return NS(**base)


def test_kinds():
    assert decision_kind(dec()) == NORMAL
    assert decision_kind(dec(own_status="slp", action_index=-1)) == FIGHT_OR_SWITCH
    assert decision_kind(dec(own_status="frz")) == FIGHT_OR_SWITCH
    assert decision_kind(dec(lock_shape="full")) == FORCED_RECHARGE
    assert decision_kind(dec(lock_shape="full", own_status="slp")) == FORCED_RECHARGE  # a full lock beats sleep
    assert decision_kind(dec(lock_shape="semi")) == FIGHT_OR_SWITCH  # trapped, trapping or Bide: switching stays allowed
    assert decision_kind(dec(struggling=True)) == FIGHT_OR_SWITCH  # every move is out of PP: Struggle, and switching is allowed
    assert decision_kind(dec(action_index=-1, action_reason="game_ended")) == NO_CHOICE
    assert decision_kind(dec(action_index=-1, action_reason="par")) == NORMAL  # paralyzed: a real choice that then failed
    assert decision_kind(dec(forced_switch=True, lock_shape="semi", own_status="slp")) == NORMAL  # choosing a replacement


def test_lock_shapes_follow_the_engine():
    lock = lambda kind, age=0, trapped=0: {"lock": kind, "age": age, "trapped_age": trapped, "disabled": None}
    for full in ("recharge", "charge", "thrash", "rage"):  # one option and no switching
        assert lock_shape(lock(full)) == "full"
    assert lock_shape(lock("bide")) == "semi"
    # the engine tells the player it is locked only right after the first use of a trap; later the request lists every move
    assert lock_shape(lock("trap_user", age=1)) == "semi" and lock_shape(lock("trap_user", age=2)) is None
    assert lock_shape(lock(None, trapped=1)) == "semi" and lock_shape(lock(None, trapped=2)) is None
    assert lock_shape(lock(None)) is None and lock_shape(None) is None


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


def test_recharge_leaves_only_the_no_op():
    legal = row([1, 1, 1, 1], [1, 1, 1, 0, 0])
    r, _, _ = adjust(FORCED_RECHARGE, legal, legal / legal.sum(), legal / legal.sum())
    assert list(r) == [1, 0, 0, 0, 0, 0, 0, 0, 0]


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
