"""Observer features for the recurrent model (see pkmn-search/docs/model_spec.md).

Everything here is computed from the observer's public information only:
  - the observer's own full party (available to the player),
  - the opponent's active Pokemon and living reserves from the battle state,
  - the opponent's revealed party, tracked by the caller from public history,
  - the last turn's public record (actions, moves, HP changes, faints).

Layout (per decision):
  ids    : integer ids, embedded by the model (species, moves, statuses)
  floats : fixed-width float features

The damage features use the Gen 1 damage formula at its maximum roll (255/255), with maximum
DVs and maximum stat experience, the same assumptions as the Zig engine's damage estimates.
"""

from __future__ import annotations

from types import SimpleNamespace

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from metamon.recurrent.tables import Tables

N_OWN = 6
N_OPP = 6
N_MOVES = 4
N_ACTIONS = 9
TURN_EDGES = (2, 4, 8, 16, 32, 64, 128)  # turn buckets: <2, <4, ..., <128, >=128 -> 8 buckets

# status ids (8 values): the seven Gen 1 statuses plus "fnt" (fainted), which parsed data uses
STATUS_IDS = {"nostatus": 0, "slp": 1, "brn": 2, "frz": 3, "par": 4, "psn": 5, "tox": 6, "fnt": 7}
# Gen 1 physical types; every other type is special.
PHYSICAL_TYPES = {"normal", "fighting", "poison", "ground", "flying", "bug", "rock", "ghost", "steel"}
# Action kinds of the last turn, and the reasons a side did not act (from the parser's extras).
KINDS = ("move", "switch", "recharge", "none")
CANT_REASONS = (
    "par", "slp", "frz", "recharge", "partiallytrapped", "flinch",
    "fainted", "wokeup", "opponent_fainted", "confusion", "game_ended",
)
LAST_SIDE_WIDTH = len(KINDS) + 1 + 1 + len(CANT_REASONS)  # kind, hp change, fainted, cant reason
EFFECTS = ("substitute", "confusion", "leechseed", "focusenergy")  # volatile effects the model sees, per side
FIELD_WIDTH = 39  # damage 8, speed 2, turn 8, party 2, forced 2, screens 6, trapped 2, effects 8, derived-extras flag 1
# Screens. The parsed data keeps side conditions as "noconditions" and records Reflect and the
# other screens as a Pokemon's effect, so both places are read.
SCREENS = ("reflect", "lightscreen", "mist")
SPEED_SCALE = 500.0


@dataclass
class LastTurn:
    """The public record of the turn that produced this state (see records.last_turn_from_extras).

    Actions are recorded by identity, not by move-slot index: a move name, or the species switched
    in. The slot index depends on the mover's whole moveset, which is partly unknown for the
    opponent and, in parsed replays, partly guessed for the observer.
    """

    own_kind: str = "none"  # one of KINDS
    own_name: str = ""  # move name, or the species switched in
    own_cant: Optional[str] = None  # why it did not act (one of CANT_REASONS), None if it acted
    opp_kind: str = "none"
    opp_name: str = ""
    opp_cant: Optional[str] = None
    own_hp_change: float = 0.0  # change in own active HP fraction, clipped to [-1, 1]
    opp_hp_change: float = 0.0
    own_status_inflicted: str = "nostatus"
    opp_status_inflicted: str = "nostatus"
    own_fainted: bool = False
    opp_fainted: bool = False


@dataclass
class Encoded:
    ids: dict[str, np.ndarray] = field(default_factory=dict)
    floats: dict[str, np.ndarray] = field(default_factory=dict)


def _stat_of(base: int, *, hp: bool = False) -> float:
    """Unboosted Gen 1 stat at level 100 with maximum DVs and maximum stat experience."""
    if hp:
        return float((base + 15) * 2 + 63 + 100 + 10)
    return float((base + 15) * 2 + 63 + 5)


def _boosted(stat: float, boost: int) -> float:
    """Gen 1 stat-stage multiplier, for boost in [-6, 6]."""
    boost = max(-6, min(6, boost))
    if boost >= 0:
        return stat * (2 + boost) / 2
    return stat * 2 / (2 - boost)


def _boost(value: int) -> float:
    return max(-6, min(6, value)) / 6.0


def _name(pokemon) -> str:
    return pokemon.name


def _types(pokemon) -> list[str]:
    return [t for t in pokemon.types.split(" ") if t and t != "notype"]


# fixed damage (the user's level, 100), one-hit KOs, and the moves that halve the target's defense
FIXED_DAMAGE = {"seismictoss": 100, "nightshade": 100, "sonicboom": 20, "dragonrage": 40}
OHKO_MOVES = {"fissure", "guillotine", "horndrill"}
HALVES_DEFENSE = {"explosion", "selfdestruct"}


def _damage_fraction(attacker, defender, move, tables: Tables, attacker_stats, defender_stats) -> float:
    """Max-roll Gen 1 damage as a fraction of the defender's max HP (0 for non-damaging moves)."""
    name = move.name
    power, move_type = tables.move_power_type(name)
    if name in FIXED_DAMAGE or name in OHKO_MOVES:
        # the type chart still applies (a Ghost is immune to Normal and Fighting moves)
        # (Night Shade ignores it: the engine hits Normal types with it)
        immune = name != "nightshade" and any(tables.effectiveness(move_type, t) == 0 for t in _types(defender))
        if immune:
            return 0.0
        if name in OHKO_MOVES:
            return 1.0
        return min(1.0, FIXED_DAMAGE[name] / max(1.0, defender_stats["hp"]))
    if power <= 0:
        return 0.0
    special = move_type not in PHYSICAL_TYPES
    if special:
        atk = _boosted(attacker_stats["spa"], attacker.spa_boost)
        dfn = _boosted(defender_stats["spd"], defender.spd_boost)
    else:
        atk = _boosted(attacker_stats["atk"], attacker.atk_boost)
        dfn = _boosted(defender_stats["def"], defender.def_boost)
    if atk >= 256 or dfn >= 256:
        atk = max(1, atk // 4)
        dfn = max(1, dfn // 4)
    if name in HALVES_DEFENSE:
        dfn = max(1, dfn // 2)
    level_term = 2 * 100 // 5 + 2
    damage = (level_term * power * int(atk)) // max(1, int(dfn))
    damage = damage // 50 + 2
    if move_type in _types(attacker):
        damage = damage * 3 // 2
    multiplier = 1.0
    for def_type in _types(defender):
        multiplier *= tables.effectiveness(move_type, def_type)
    damage = int(damage * multiplier)
    if multiplier > 0:
        damage = max(1, damage)
    return min(1.0, damage / max(1.0, defender_stats["hp"]))


def counter_fraction(user: str, extras: Optional[dict], tables: Tables, own_hp: float, opp_hp: float, max_hp=None, estimate=None) -> float:
    """What Counter used by `user` ("player" or "opponent") would do now, as a fraction of the target's max HP.

    Gen 1 Counter (data/mods/gen1/moves.ts) deals twice the battle's lastDamage, the damage of the last move
    executed by either side, and ignores type immunities (a Ghost is hit). It succeeds when three things hold:
      - lastDamage is not 0;
      - the target's last move USED is Normal or Fighting with power (Counter itself qualifies: it is Fighting
        with power 1, a quirk of the engine);
      - the target's last move SELECTED was not Counter.
    The last two memories differ when the target selected a move that did not run (it could not act) or switched since
    running Counter, and the log never shows selections: the parser follows them (a move run sets the selection, a
    switch leaves it, a turn without a move leaves it unknown and taken as "not Counter", which the engine confirms
    in most such cases). `own_hp` and `opp_hp` are the max HP of
    the two active Pokemon; `max_hp(side, species)` can supply a Pokemon's true max HP instead of the max-DV estimate.
    """
    if not extras or not extras.get("last_damage"):
        return 0.0
    damage, last_move = extras["last_damage"], extras.get("last_move") or {}
    target = "opponent" if user == "player" else "player"
    moved = last_move.get(target)
    if not moved or moved not in tables.moves:
        return 0.0
    power, move_type = tables.move_power_type(moved)
    if power <= 0 or move_type not in ("normal", "fighting"):
        return 0.0
    if (extras.get("counter_selected") or {}).get(target) and extras.get(f"{target}_cant") is None:
        return 0.0  # its last move selection was Counter (it did not select anything new that failed to run): the test fails
    if damage["kind"] == "unknown":
        # taken by a Substitute: the engine tracks the uncapped amount, which the log does not show. `estimate(side, move)`
        # (the damage calculator's average roll, as a fraction of the max HP of the side that took it) stands in for it
        if estimate is None:
            return 0.0
        hitter = "opponent" if damage["side"] == "player" else "player"
        hit_move = last_move.get(hitter)
        if not hit_move or hit_move not in tables.moves:
            return 0.0
        taken = estimate(damage["side"], hit_move)
        victim_hp = own_hp if damage["side"] == "player" else opp_hp
        target_hp = opp_hp if user == "player" else own_hp
        return min(1.0, 2.0 * taken * victim_hp / max(1.0, target_hp))
    # the damage in HP: a share of the max HP of the Pokemon that lost it (or healed, for a drain)
    try:
        victim_hp = max_hp(damage["side"], damage["name"]) if max_hp else _stat_of(tables.base_stats(damage["name"])["hp"], hp=True)
    except KeyError:
        victim_hp = own_hp if damage["side"] == "player" else opp_hp
    target_hp = opp_hp if user == "player" else own_hp
    return min(1.0, 2.0 * damage["pct"] / 100.0 * victim_hp / max(1.0, target_hp))


def _move_damage(user: str, attacker, defender, move, tables: Tables, attacker_stats, defender_stats, extras, own_hp: float, opp_hp: float) -> float:
    """Max-roll damage of one move as a fraction of the defender's max HP, with the Gen 1 moves whose damage is
    not a function of the two Pokemon's stats handled as the engine does."""
    key = "".join(c for c in move.name.lower() if c.isalnum())
    if key == "counter":
        def estimate(victim_side: str, move_name: str) -> float:
            # the average Gen 1 roll (217..255 of 255) of the move that hit the Substitute, as a share of the victim's HP
            hit = SimpleNamespace(name=move_name)
            if victim_side == user:  # the foe hit us
                return 0.93 * _damage_fraction(defender, attacker, hit, tables, defender_stats, attacker_stats)
            return 0.93 * _damage_fraction(attacker, defender, hit, tables, attacker_stats, defender_stats)

        return counter_fraction(user, extras, tables, own_hp, opp_hp, estimate=estimate)
    if key == "superfang":  # half of the target's current HP, at least 1 (ignores type immunity)
        hp = defender_stats["hp"]
        return max(1.0, float(int(defender.hp_pct * hp) // 2)) / max(1.0, hp) if defender.hp_pct > 0 else 0.0
    if key == "psywave":  # a random 1 to 149 at level 100
        return min(1.0, 149.0 / max(1.0, defender_stats["hp"]))
    if key == "bide":  # twice the damage taken while it was stored: not visible in the state
        return 0.0
    return _damage_fraction(attacker, defender, move, tables, attacker_stats, defender_stats)


def _own_stats(pokemon, tables: Tables) -> dict[str, float]:
    """The observer's own unboosted stats: the actual values when present, else the Gen 1 estimate."""
    base = tables.base_stats(_name(pokemon))
    out = {}
    for key in ("atk", "def", "spe", "spa"):
        actual = getattr(pokemon, f"{key}_stat", -1)
        out[key] = float(actual) if actual and actual > 0 else _stat_of(base[key])
    out["spd"] = out["spa"]  # Gen 1 has one special stat
    hp_actual = getattr(pokemon, "hp_stat", -1)
    out["hp"] = float(hp_actual) if hp_actual and hp_actual > 0 else _stat_of(base["hp"], hp=True)
    return out


def _opp_stats(pokemon, tables: Tables) -> dict[str, float]:
    """The opponent's stats are not public: estimate them from the base stats at maximum DVs."""
    base = tables.base_stats(_name(pokemon))
    out = {k: _stat_of(base[k]) for k in ("atk", "def", "spe", "spa")}
    out["spd"] = out["spa"]
    out["hp"] = _stat_of(base["hp"], hp=True)
    return out


def _turn_bucket(turn: int) -> np.ndarray:
    out = np.zeros(len(TURN_EDGES) + 1, dtype=np.float32)
    idx = next((i for i, edge in enumerate(TURN_EDGES) if turn < edge), len(TURN_EDGES))
    out[idx] = 1.0
    return out


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def canonical_own(state, own_party: Sequence) -> tuple[list, object]:
    """The observer's party in the order the policy outputs refer to.

    Slot 0 is the active Pokemon; slots 1-5 are the others sorted by species name, fainted ones included.
    Every Pokemon's moves are sorted by name. Policy logit j < 4 is then the active's j-th move (slot 0's
    move j) and logit 4 + i is reserve slot 1 + i. Returns (party in slot order, the active Pokemon).
    """
    import copy

    from metamon.interface import move_name, pokemon_name

    active_name = state.player_active_pokemon.name
    active = [p for p in own_party if p.name == active_name]
    if not active:  # the party list should always hold the active; fall back to the state's own copy
        active = [state.player_active_pokemon]
    others = sorted((p for p in own_party if p.name != active_name), key=lambda p: pokemon_name(p.name))
    ordered = []
    for p in active + others:
        q = copy.copy(p)
        q.moves = sorted(p.moves, key=lambda m: move_name(m.name))
        ordered.append(q)
    active_copy = copy.copy(state.player_active_pokemon)
    active_copy.moves = sorted(state.player_active_pokemon.moves, key=lambda m: move_name(m.name))
    return ordered, active_copy


def action_slot_map(state, ordered_party: Sequence) -> np.ndarray:
    """For each action index of Metamon's action space (moves 0-3, then the alive reserves in sorted
    species order), the policy slot of this model: moves keep their index; the k-th alive reserve maps to
    4 + (its position among the party's non-active members in sorted order). -1 where there is no match."""
    from metamon.interface import pokemon_name

    mapping = np.full(N_ACTIONS, -1, dtype=np.int64)
    mapping[:N_MOVES] = np.arange(N_MOVES)
    reserve_names = [p.name for p in ordered_party[1:]]
    for k, p in enumerate(sorted(state.available_switches, key=lambda p: pokemon_name(p.name))):
        if 4 + k < N_ACTIONS and p.name in reserve_names:
            mapping[4 + k] = 4 + reserve_names.index(p.name)
    return mapping


def encode_observer(
    state,
    own_party: Sequence,
    opp_revealed: Sequence,
    last: LastTurn,
    turn: int,
    tables: Tables,
    extras: Optional[dict] = None,
) -> Encoded:
    """Features for one decision, from the observer's view.

    state        : the metamon UniversalState from the observer's side (active Pokemon, living
                   reserves, conditions, forced-switch flag, opponents_remaining).
    own_party    : the observer's full party (6 entries; fainted members included), in any order; it is
                   put in policy order by canonical_own.
    opp_revealed : the opponent's revealed Pokemon in order of first appearance (max 6), from history.
    last         : the public record of the previous turn.
    turn         : the battle turn number of this decision.
    extras       : the parser's extras for this state (trapped flags, all side conditions). Without
                   them the trapped flags are 0 and the screens come from the effect field only.
    """
    if len(own_party) > N_OWN or len(opp_revealed) > N_OPP:
        raise ValueError("a party has at most six Pokemon")
    own_party, own_active = canonical_own(state, own_party)
    opp_active = state.opponent_active_pokemon
    ids: dict[str, np.ndarray] = {
        "own_species": np.zeros(N_OWN, dtype=np.int64),
        "own_status": np.zeros(N_OWN, dtype=np.int64),
        "own_moves": np.zeros((N_OWN, N_MOVES), dtype=np.int64),
        "opp_species": np.zeros(N_OPP, dtype=np.int64),
        "opp_status": np.zeros(N_OPP, dtype=np.int64),
        "opp_moves": np.zeros((N_OPP, N_MOVES), dtype=np.int64),
        "last_own_move": np.zeros(1, dtype=np.int64),
        "last_opp_move": np.zeros(1, dtype=np.int64),
        "last_own_species": np.zeros(1, dtype=np.int64),
        "last_opp_species": np.zeros(1, dtype=np.int64),
        "last_own_status": np.zeros(1, dtype=np.int64),
        "last_opp_status": np.zeros(1, dtype=np.int64),
    }
    own = np.zeros((N_OWN, 10), dtype=np.float32)  # hp, active, atk, def, spe, spa, pp x4
    for slot, pokemon in enumerate(own_party):
        is_active = pokemon.name == own_active.name
        live = pokemon.hp_pct > 0
        ids["own_species"][slot] = tables.species_id(_name(pokemon))
        ids["own_status"][slot] = STATUS_IDS[pokemon.status]
        for j, move in enumerate(pokemon.moves[:N_MOVES]):
            ids["own_moves"][slot, j] = tables.move_id(move.name)
            own[slot, 6 + j] = move.current_pp / max(1, move.max_pp) if move.max_pp else 0.0
        own[slot, 0] = pokemon.hp_pct if live else 0.0
        own[slot, 1] = 1.0 if is_active else 0.0
        own[slot, 2:6] = [_boost(pokemon.atk_boost), _boost(pokemon.def_boost), _boost(pokemon.spe_boost), _boost(pokemon.spa_boost)]

    opp = np.zeros((N_OPP, 7), dtype=np.float32)  # seen, hp, active, atk, def, spe, spa
    for slot, pokemon in enumerate(opp_revealed):
        is_active = pokemon.name == opp_active.name
        ids["opp_species"][slot] = tables.species_id(_name(pokemon))
        ids["opp_status"][slot] = STATUS_IDS[pokemon.status]
        for j, move in enumerate(pokemon.moves[:N_MOVES]):
            ids["opp_moves"][slot, j] = tables.move_id(move.name)
        opp[slot, 0] = 1.0
        opp[slot, 1] = pokemon.hp_pct
        opp[slot, 2] = 1.0 if is_active else 0.0
        opp[slot, 3:7] = [_boost(pokemon.atk_boost), _boost(pokemon.def_boost), _boost(pokemon.spe_boost), _boost(pokemon.spa_boost)]

    last_floats = np.zeros((2, LAST_SIDE_WIDTH), dtype=np.float32)
    for side, kind, name, cant, hp_change, fainted, status in (
        (0, last.own_kind, last.own_name, last.own_cant, last.own_hp_change, last.own_fainted, last.own_status_inflicted),
        (1, last.opp_kind, last.opp_name, last.opp_cant, last.opp_hp_change, last.opp_fainted, last.opp_status_inflicted),
    ):
        who = "own" if side == 0 else "opp"
        last_floats[side, :len(KINDS)] = _one_hot(KINDS.index(kind), len(KINDS))
        last_floats[side, len(KINDS)] = float(np.clip(hp_change, -1, 1))
        last_floats[side, len(KINDS) + 1] = float(fainted)
        if cant is not None:
            last_floats[side, len(KINDS) + 2 + CANT_REASONS.index(cant)] = 1.0
        if kind == "move":
            ids[f"last_{who}_move"][0] = tables.move_id(name)
        elif kind == "switch":
            ids[f"last_{who}_species"][0] = tables.species_id(name)
        ids[f"last_{who}_status"][0] = STATUS_IDS[status]

    # damage: our active's known moves against theirs, and theirs (revealed) against ours
    own_stats = _own_stats(own_active, tables)
    opp_stats = _opp_stats(opp_active, tables)
    damage = np.zeros(8, dtype=np.float32)
    for j, move in enumerate(own_active.moves[:N_MOVES]):
        damage[j] = _move_damage("player", own_active, opp_active, move, tables, own_stats, opp_stats, extras, own_stats["hp"], opp_stats["hp"])
    for j, move in enumerate(opp_active.moves[:N_MOVES]):
        damage[4 + j] = _move_damage("opponent", opp_active, own_active, move, tables, opp_stats, own_stats, extras, own_stats["hp"], opp_stats["hp"])

    speed = np.array([own_stats["spe"] / SPEED_SCALE, opp_stats["spe"] / SPEED_SCALE], dtype=np.float32)
    party = np.array(
        [
            sum(1 for p in own_party if p.hp_pct > 0) / N_OWN,
            state.opponents_remaining / N_OPP,
        ],
        dtype=np.float32,
    )
    forced = np.array([float(bool(state.forced_switch)), 0.0], dtype=np.float32)  # only our own forced switch is observable
    own_conditions = " ".join((extras or {}).get("player_conditions_all", [])) + " " + str(state.player_conditions)
    opp_conditions = " ".join((extras or {}).get("opponent_conditions_all", [])) + " " + str(state.opponent_conditions)
    screens = np.array(
        [float(sc in own_conditions or own_active.effect == sc) for sc in SCREENS]
        + [float(sc in opp_conditions or opp_active.effect == sc) for sc in SCREENS],
        dtype=np.float32,
    )
    trapped = np.array(
        [float((extras or {}).get("player_trapped", False)), float((extras or {}).get("opponent_trapped", False))],
        dtype=np.float32,
    )
    # 1 when the extras were derived from the official parsed states (upgrade.py) instead of the raw log:
    # the cannot-act reasons and trapped flags are then partly unknown
    derived = np.array([float((extras or {}).get("derived", False))], dtype=np.float32)
    # volatile effects of the two active Pokemon (the parser's full list; the official state keeps only one)
    effects = np.array(
        [float(fx in set((extras or {}).get(key) or ())) for key in ("player_effects", "opponent_effects") for fx in EFFECTS],
        dtype=np.float32,
    )
    field_floats = np.concatenate([damage, speed, _turn_bucket(turn), party, forced, screens, trapped, effects, derived])
    assert field_floats.shape == (FIELD_WIDTH,)

    return Encoded(
        ids=ids,
        floats={
            "own": own,
            "opp": opp,
            "last": last_floats.reshape(-1),
            "field": field_floats,
        },
    )
