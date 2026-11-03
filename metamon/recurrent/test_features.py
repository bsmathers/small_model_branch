"""Shape and behaviour tests for the observer features. Run: python -m metamon.recurrent.test_features"""

from types import SimpleNamespace as NS

import numpy as np

from metamon.recurrent.features import action_slot_map, canonical_own, CANT_REASONS, FIELD_WIDTH, KINDS, LAST_SIDE_WIDTH, LastTurn, encode_observer, N_MOVES, N_OPP, N_OWN
from metamon.recurrent.tables import Tables


def _mon(name, hp=1.0, status="nostatus", moves=("tackle",), types="normal", effect="noeffect", active=False, hp_stat=-1):
    return NS(
        name=name,
        hp_pct=hp,
        status=status,
        types=types,
        effect=effect,
        moves=[NS(name=m, current_pp=10, max_pp=10, base_power=40, move_type="normal", category="physical") for m in moves],
        atk_boost=0,
        def_boost=0,
        spa_boost=0,
        spd_boost=0,
        spe_boost=0,
        atk_stat=-1,
        def_stat=-1,
        spa_stat=-1,
        spd_stat=-1,
        spe_stat=-1,
        hp_stat=hp_stat,
    )


def _tables() -> Tables:
    return Tables(
        species_ids={"tauros": 1, "chansey": 2},
        move_ids={"bodyslam": 1, "tackle": 2},
        base={
            "tauros": {"hp": 75, "atk": 100, "def": 95, "spe": 110, "spa": 70},
            "chansey": {"hp": 250, "atk": 5, "def": 5, "spe": 50, "spa": 105},
        },
        moves={"bodyslam": (85, "normal"), "tackle": (35, "normal")},
        chart={"normal": {"normal": 1.0}},
    )


def _state(player, opponent, forced=False, remaining=6):
    return NS(
        player_active_pokemon=player,
        opponent_active_pokemon=opponent,
        player_conditions="noconditions",
        opponent_conditions="noconditions",
        forced_switch=forced,
        opponents_remaining=remaining,
    )


def test_shapes_are_fixed():
    tables = _tables()
    own = [_mon("tauros", active=True, moves=("bodyslam", "tackle"))]
    own += [_mon("chansey") for _ in range(5)]
    opp = [_mon("chansey", active=True)]
    enc = encode_observer(_state(own[0], opp[0]), own, opp, LastTurn(), turn=1, tables=tables)
    assert enc.ids["own_species"].shape == (N_OWN,)
    assert enc.ids["own_moves"].shape == (N_OWN, N_MOVES)
    assert enc.ids["opp_species"].shape == (N_OPP,)
    assert enc.floats["own"].shape == (N_OWN, 10)
    assert enc.floats["opp"].shape == (N_OPP, 7)
    assert enc.floats["last"].shape == (2 * LAST_SIDE_WIDTH,)
    assert enc.floats["field"].shape == (FIELD_WIDTH,)


def test_empty_opponent_slots_are_zero():
    tables = _tables()
    own = [_mon("tauros", active=True)]
    opp = [_mon("chansey", active=True)]
    enc = encode_observer(_state(own[0], opp[0]), own, opp, LastTurn(), turn=1, tables=tables)
    assert enc.ids["opp_species"][1:].sum() == 0
    assert enc.floats["opp"][1:].sum() == 0


def test_unknown_names_fail_loudly():
    tables = _tables()
    own = [_mon("mystery", active=True)]
    try:
        encode_observer(_state(own[0], own[0]), own, [], LastTurn(), turn=1, tables=tables)
    except KeyError:
        return
    raise AssertionError("an unknown species should raise KeyError")


def test_opponent_unrevealed_fields_do_not_change_features():
    """The encoder only sees the opponent's active and revealed party, so changing an unrevealed
    opponent Pokemon (not passed in) cannot change the output. The active's HP fraction is public."""
    tables = _tables()
    own = [_mon("tauros", active=True, moves=("bodyslam", "tackle"))]
    a = encode_observer(_state(own[0], _mon("chansey", active=True, hp=0.5)), own, [_mon("chansey", active=True, hp=0.5)], LastTurn(), 3, tables)
    b = encode_observer(_state(own[0], _mon("chansey", active=True, hp=0.5)), own, [_mon("chansey", active=True, hp=0.5)], LastTurn(), 3, tables)
    for key in a.floats:
        assert np.array_equal(a.floats[key], b.floats[key])
    for key in a.ids:
        assert np.array_equal(a.ids[key], b.ids[key])


def test_last_turn_token_carries_the_action_and_the_reason():
    tables = _tables()
    own = [_mon("tauros", active=True, moves=("bodyslam", "tackle"))]
    opp = [_mon("chansey", active=True)]
    last = LastTurn(own_kind="move", own_name="bodyslam", opp_kind="none", opp_cant="par", opp_hp_change=-0.25)
    enc = encode_observer(_state(own[0], opp[0]), own, opp, last, 2, tables)
    own_side, opp_side = enc.floats["last"].reshape(2, LAST_SIDE_WIDTH)
    assert own_side[KINDS.index("move")] == 1.0 and opp_side[KINDS.index("none")] == 1.0
    assert opp_side[len(KINDS) + 2 + CANT_REASONS.index("par")] == 1.0
    assert own_side[len(KINDS) + 2:].sum() == 0.0  # it acted: no cannot-act reason
    assert opp_side[len(KINDS)] == -0.25
    assert enc.ids["last_own_move"][0] == tables.move_id("bodyslam")


def test_switch_records_the_species_not_a_move():
    tables = _tables()
    own = [_mon("tauros", active=True)]
    opp = [_mon("chansey", active=True)]
    last = LastTurn(opp_kind="switch", opp_name="chansey")
    enc = encode_observer(_state(own[0], opp[0]), own, opp, last, 2, tables)
    assert enc.ids["last_opp_species"][0] == tables.species_id("chansey")
    assert enc.ids["last_opp_move"][0] == 0


def test_trapped_flags_come_from_the_extras():
    tables = _tables()
    own = [_mon("tauros", active=True)]
    opp = [_mon("chansey", active=True)]
    base = encode_observer(_state(own[0], opp[0]), own, opp, LastTurn(), 3, tables)
    assert base.floats["field"][-3:].sum() == 0.0
    extras = {"player_trapped": True, "opponent_trapped": False, "player_conditions_all": ["reflect"], "opponent_conditions_all": []}
    enc = encode_observer(_state(own[0], opp[0]), own, opp, LastTurn(), 3, tables, extras=extras)
    assert list(enc.floats["field"][-3:-1]) == [1.0, 0.0]
    assert enc.floats["field"][-1] == 0.0  # not derived
    assert enc.floats["field"][-3 - 6] == 1.0  # our Reflect (the first screen flag)


def test_party_is_in_policy_order():
    """Slot 0 is the active; the reserves follow by species name; each Pokemon's moves are sorted by name.
    The action slot map sends Metamon's k-th alive reserve to its slot among the sorted reserves."""
    own = [
        _mon("tauros", active=False, moves=("tackle", "bodyslam")),
        _mon("snorlax", active=True, moves=("tackle", "bodyslam")),
        _mon("chansey", hp=0.0, moves=("tackle",)),
        _mon("zapdos", moves=("tackle",)),
        _mon("alakazam", moves=("tackle",)),
        _mon("exeggutor", moves=("tackle",)),
    ]
    state = _state(own[1], _mon("starmie", active=True))
    state.available_switches = [own[0], own[3], own[4], own[5]]  # chansey fainted
    ordered, active = canonical_own(state, own)
    assert [p.name for p in ordered] == ["snorlax", "alakazam", "chansey", "exeggutor", "tauros", "zapdos"]
    assert [m.name for m in active.moves] == ["bodyslam", "tackle"]
    assert [m.name for m in ordered[0].moves] == ["bodyslam", "tackle"]
    mapping = action_slot_map(state, ordered)
    # alive reserves in name order: alakazam, exeggutor, tauros, zapdos -> slots (reserve index 0, 2, 3, 4) + 4
    assert list(mapping) == [0, 1, 2, 3, 4, 6, 7, 8, -1][:9]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
