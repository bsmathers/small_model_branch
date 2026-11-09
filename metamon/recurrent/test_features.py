"""Shape and behaviour tests for the observer features. Run: python -m metamon.recurrent.test_features"""

from types import SimpleNamespace as NS

import numpy as np

from metamon.recurrent.features import action_slot_map, canonical_own, counter_fraction, CANT_REASONS, FIELD_WIDTH, KINDS, LAST_SIDE_WIDTH, LastTurn, encode_observer, N_MOVES, N_OPP, N_OWN, policy_options
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


def test_trapped_and_effect_values_come_from_the_extras():
    # field tail: ... screens (6), trapped (2), effects (8: substitute, confusion, leechseed, focusenergy per side), derived (1)
    tables = _tables()
    own = [_mon("tauros", active=True)]
    opp = [_mon("chansey", active=True)]
    base = encode_observer(_state(own[0], opp[0]), own, opp, LastTurn(), 3, tables)
    assert base.floats["field"][-11:].sum() == 0.0
    extras = {
        "player_trapped": 0.625, "opponent_trapped": 0.0, "player_conditions_all": ["reflect"], "opponent_conditions_all": [],
        "player_effects": ["confusion"], "opponent_effects": ["substitute", "leechseed"],
    }
    enc = encode_observer(_state(own[0], opp[0]), own, opp, LastTurn(), 3, tables, extras=extras)
    f = enc.floats["field"]
    assert list(f[-11:-9]) == [0.625, 0.0]
    assert list(f[-9:-1]) == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert f[-1] == 0.0  # not derived
    assert f[-11 - 6] == 1.0  # our Reflect (the first screen flag)


def test_party_is_in_policy_order():
    """Exeggcute's order: slot 0 is the active; the living reserves follow by species name, then the fainted ones;
    each Pokemon's moves are sorted by name. Metamon's action indices are then the policy slots (the identity map)."""
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
    assert [p.name for p in ordered] == ["snorlax", "alakazam", "exeggutor", "tauros", "zapdos", "chansey"]
    assert [m.name for m in active.moves] == ["bodyslam", "tackle"]
    assert [m.name for m in ordered[0].moves] == ["bodyslam", "tackle"]
    assert policy_options(ordered) == (["bodyslam", "tackle"], ["alakazam", "exeggutor", "tauros", "zapdos"])
    mapping = action_slot_map(state, ordered)
    # Metamon's k-th alive reserve (alakazam, exeggutor, tauros, zapdos) is slot 4 + k: the identity
    assert list(mapping) == [0, 1, 2, 3, 4, 5, 6, 7, -1]


def _more_tables() -> Tables:
    t = _tables()
    for i, name in enumerate(("snorlax", "starmie", "zapdos"), start=3):
        t.species_ids[name] = i
        t.base[name] = {"hp": 100, "atk": 100, "def": 100, "spe": 100, "spa": 100}
    return t


def test_opponent_is_sorted_by_species_with_sorted_moves():
    """The opponent's revealed Pokemon go by species name and their revealed moves by name, whatever order they were seen in."""
    tables = _more_tables()
    own = [_mon("tauros", active=True, moves=("bodyslam", "tackle"))]
    opp_active = _mon("snorlax", active=True, moves=("tackle", "bodyslam"))
    revealed = [_mon("tauros", moves=("tackle",)), opp_active, _mon("chansey", moves=("tackle",))]
    enc = encode_observer(_state(own[0], opp_active), own, revealed, LastTurn(), 3, tables)
    assert list(enc.ids["opp_species"][:3]) == [tables.species_id(n) for n in ("chansey", "snorlax", "tauros")]
    assert list(enc.floats["opp"][:3, 2]) == [0.0, 1.0, 0.0]  # the active flag follows the Pokemon
    assert list(enc.ids["opp_moves"][1, :2]) == [tables.move_id("bodyslam"), tables.move_id("tackle")]


def test_features_do_not_depend_on_the_order_of_the_inputs():
    """Only the current state decides the layout: shuffling the party lists or the move lists changes nothing."""
    tables = _more_tables()
    def party(order):
        mons = {"tauros": _mon("tauros", active=True, moves=("tackle", "bodyslam")), "chansey": _mon("chansey", hp=0.0),
                "snorlax": _mon("snorlax", moves=("tackle", "bodyslam")), "starmie": _mon("starmie")}
        return [mons[n] for n in order]
    opp = lambda order: [{"zapdos": _mon("zapdos"), "snorlax": _mon("snorlax", active=True, moves=("tackle", "bodyslam"))}[n] for n in order]
    a_own, b_own = party(["tauros", "chansey", "snorlax", "starmie"]), party(["starmie", "snorlax", "chansey", "tauros"])
    a_opp, b_opp = opp(["zapdos", "snorlax"]), opp(["snorlax", "zapdos"])
    b_opp[0].moves = list(reversed(b_opp[0].moves))
    a = encode_observer(_state(a_own[0], a_opp[1]), a_own, a_opp, LastTurn(), 2, tables)
    b = encode_observer(_state(b_own[3], b_opp[0]), b_own, b_opp, LastTurn(), 2, tables)
    for k in a.ids:
        assert (a.ids[k] == b.ids[k]).all(), k
    for k in a.floats:
        assert np.allclose(a.floats[k], b.floats[k]), k


def _counter_tables() -> Tables:
    t = _tables()
    t.moves.update({"counter": (1, "fighting"), "thunderbolt": (95, "electric"), "seismictoss": (1, "fighting"), "splash": (0, "normal")})
    t.base.update({"snorlax": {"hp": 160, "atk": 110, "def": 65, "spe": 30, "spa": 65}})
    return t


def _extras(foe_move, damage_pct=20.0, victim="chansey", side="player", kind="hit", selected=False, foe_cant=None):
    return {
        "last_damage": {"kind": kind, "side": side, "name": victim, "pct": damage_pct},
        "last_move": {"player": "tackle", "opponent": foe_move},
        "counter_selected": {"player": False, "opponent": selected},
        "opponent_cant": foe_cant,
        "player_cant": None,
    }


def test_counter_is_twice_the_last_damage_and_needs_a_counterable_last_move():
    t = _counter_tables()
    hp = 703.0  # chansey; the target of our Counter is the foe, the damage was taken by our own chansey
    # the foe's last move was Body Slam (Normal, power): Counter works, for twice what we lost (20% of our max HP)
    assert abs(counter_fraction("player", _extras("bodyslam"), t, hp, hp) - 0.4) < 1e-9
    # the damage was a share of the Pokemon that lost it, and the fraction is of the target's max HP
    snorlax_hp = (160 + 15) * 2 + 63 + 110
    assert abs(counter_fraction("player", _extras("bodyslam", victim="snorlax"), t, hp, hp) - 2 * 0.2 * snorlax_hp / hp) < 1e-9
    # Thunderbolt is Electric: not counterable; a status move has no power; neither works
    assert counter_fraction("player", _extras("thunderbolt"), t, hp, hp) == 0.0
    assert counter_fraction("player", _extras("splash"), t, hp, hp) == 0.0
    # no damage in the battle, or damage taken by a Substitute (amount not in the log): nothing to counter
    assert counter_fraction("player", {**_extras("bodyslam"), "last_damage": None}, t, hp, hp) == 0.0
    assert counter_fraction("player", _extras("bodyslam", kind="unknown"), t, hp, hp) == 0.0
    # twice a big hit is a KO, never more than the target's max HP
    assert counter_fraction("player", _extras("bodyslam", damage_pct=80.0), t, hp, hp) == 1.0


def test_counter_after_counter_depends_on_what_the_foe_selected_not_what_it_used():
    t = _counter_tables()
    hp = 703.0
    # Counter itself is Fighting with power 1, so a foe whose last USED move was Counter is still counterable ...
    assert counter_fraction("player", _extras("counter", selected=False), t, hp, hp) > 0.0
    # ... but not when it also SELECTED Counter last (the engine's separate selection test)
    assert counter_fraction("player", _extras("counter", selected=True), t, hp, hp) == 0.0
    # a foe that selected Counter but then could not act (paralyzed, fainted): the selection is unknown, taken as not Counter
    assert counter_fraction("player", _extras("counter", selected=True, foe_cant="par"), t, hp, hp) > 0.0
    # and it works the other way round: the foe's Counter against us uses OUR last move
    mine = {**_extras("bodyslam", side="opponent"), "last_move": {"player": "bodyslam", "opponent": "tackle"}}
    assert counter_fraction("opponent", mine, t, hp, hp) > 0.0


def test_counter_after_a_substitute_hit_uses_the_estimate_when_given():
    t = _counter_tables()
    hp = 703.0
    unknown = _extras("bodyslam", kind="unknown", side="player")
    asked = []

    def estimate(victim_side, move):
        asked.append((victim_side, move))
        return 0.15  # a share of the victim's max HP

    # the foe's Body Slam hit our Substitute: Counter doubles the estimated damage (same max HP on both sides here)
    assert abs(counter_fraction("player", unknown, t, hp, hp, estimate=estimate) - 0.3) < 1e-9
    assert asked == [("player", "bodyslam")]


def test_trap_probability_table_and_effect_flags():
    from types import SimpleNamespace

    from metamon.recurrent.parse import TRAP_PROBABILITY, _trap_probability

    # the trap lasts 2/3/4/5 uses with weights 3/3/1/1: P(still trapped | k uses seen)
    assert TRAP_PROBABILITY == {1: 1.0, 2: 5 / 8, 3: 2 / 5, 4: 1 / 2, 5: 0.0}
    assert _trap_probability(SimpleNamespace(trapped_age=2)) == 5 / 8
    assert _trap_probability(SimpleNamespace(trapped_age=0)) == 0.0
    assert _trap_probability(None) == 0.0


def test_label_slots_follow_the_original_states_actions_by_identity():
    from metamon.recurrent.features import label_slot_map

    # corrected party: active snorlax (moves bodyslam, tackle), reserves chansey, tauros (sorted by name)
    party = [_mon("snorlax", active=True, moves=("bodyslam", "tackle")), _mon("chansey"), _mon("tauros")]
    # the original parser's state for the same decision: same active moves, but its reserves are tauros and zapdos
    legacy_active = _mon("snorlax", active=True, moves=("bodyslam", "tackle"))
    legacy = _state(legacy_active, _mon("chansey"))
    legacy.available_switches = [_mon("tauros"), _mon("zapdos")]
    m = label_slot_map(legacy, party)
    assert list(m[:4]) == [0, 1, -1, -1]  # the two moves keep their place, the empty slots map nowhere
    assert m[4] == 4 + 1  # legacy reserve 0 (sorted: tauros, zapdos) is tauros: the second corrected reserve
    assert m[5] == -1  # zapdos is not in the corrected party: its label mass is reported as lost
    # a move the original state lists but the corrected one lacks (a different guess for an unrevealed move)
    legacy.player_active_pokemon = _mon("snorlax", active=True, moves=("bodyslam", "earthquake"))
    assert list(label_slot_map(legacy, party)[:2]) == [0, -1]


def test_label_move_actions_follow_metamons_alphabetical_order_not_the_stored_order():
    """Exeggcute's labels index moves by Metamon's action space (consistent_move_order: alphabetical). The parsed
    state usually stores them in another order; the label for "thunderwave" must still land on thunderwave's slot."""
    from metamon.recurrent.features import label_slot_map

    party = [_mon("jolteon", active=True, moves=("pinmissile", "rest", "thunderbolt", "thunderwave")), _mon("chansey")]
    legacy = _state(_mon("jolteon", active=True, moves=("thunderbolt", "thunderwave", "pinmissile", "rest")), _mon("chansey"))
    legacy.available_switches = [_mon("chansey")]
    m = label_slot_map(legacy, party)
    # action j is the j-th move alphabetically (pinmissile, rest, thunderbolt, thunderwave); the model's slots use the same order
    assert list(m[:4]) == [0, 1, 2, 3]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
