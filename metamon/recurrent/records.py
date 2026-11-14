"""Turns parsed trajectories (the output of metamon.recurrent.parse) into encoder inputs and labels.

One Decision per state: the observer's features at that state, and what happened next (the action
label by name and by the official index, the reason when there was none, and the game result).
Everything in `features` depends only on the state and its extras up to that turn; the labels are
the future and are kept apart.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import lz4.frame
import numpy as np

from metamon import interface
from metamon.recurrent.features import CANT_REASONS, KINDS, STATUS_IDS, Encoded, LastTurn, action_slot_map, canonical_own, encode_observer, label_slot_map
from metamon.recurrent.tables import Tables


try:  # (several times faster than json; the same values)
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def _loads(raw: bytes):
    if orjson is not None:
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:  # (e.g. a NaN, which json.dumps writes and orjson refuses)
            pass
    return json.loads(raw)


# Every extras field the features and labels read (parse.PARSER_VERSION 3). Trajectories carry no version of their own, and
# several of these are read with .get (a missing lock reads as no lock), so a trajectory from an older parser is refused here.
REQUIRED_EXTRAS = frozenset({
    "action_reason", "counter_selected", "forced_switch", "last_damage", "last_move", "last_opponent_action", "last_player_action",
    "next_player_action", "opponent_cant", "opponent_conditions_all", "opponent_effects", "opponent_lock", "opponent_team",
    "opponent_trapped", "player_cant", "player_conditions_all", "player_effects", "player_lock", "player_team", "player_trapped",
})


def load_trajectory(path: str) -> dict:
    """A trajectory written by metamon.recurrent.parse: states, actions and extras."""
    with lz4.frame.open(path, "rb") as f:
        data = _loads(f.read())
    if "extras" not in data:
        raise ValueError(f"{os.path.basename(path)} has no extras: parse the raw log with metamon.recurrent.parse")
    if "legacy_states" not in data:
        raise ValueError(f"{os.path.basename(path)} has no legacy_states: run metamon.recurrent.merge_legacy (see its docstring)")
    # (extras derived from the official parsed states by upgrade.py are marked "derived" and hold fewer fields)
    missing = REQUIRED_EXTRAS - set(data["extras"][0]) if data["extras"] and not data["extras"][0].get("derived") else set()
    if missing:
        raise ValueError(f"{os.path.basename(path)} lacks extras {sorted(missing)}: made by an older parser (see parse.PARSER_VERSION); parse it again")
    return data


def load_for_labelling(path: str) -> dict:
    """Only what the labeller reads (the original parser's states and actions): the corrected parser's states and the extras are about half
    of a loaded trajectory's memory (about 64 KiB per decision is left)."""
    data = load_trajectory(path)
    return {"legacy_states": data["legacy_states"], "legacy_actions": data["legacy_actions"]}


def _kind(summary: dict) -> str:
    kind = summary["kind"]
    if kind == "noop":
        kind = "recharge"
    return kind if kind in KINDS else "none"


def _status(name: str) -> str:
    return name if name in STATUS_IDS else "nostatus"


def _no_move_if_cant(kind: str, name: str, cant: Optional[str]) -> tuple[str, str]:
    """A side that could not act ran no move. Logs with |choice| lines (about a fifth of the human games, none of the bots')
    make the parser fall back to the CHOSEN move when none ran, giving "move X" together with the reason it failed; every other
    log, and the duel server, give "none" with the reason. One convention: the chosen move is dropped. (A switch keeps its reason:
    a replacement after a faint carries the faint's.)"""
    if kind == "move" and cant is not None:
        return "none", ""
    return kind, name


def last_turn_from_extras(prev_state: Optional[dict], state: dict, extras: dict) -> LastTurn:
    """The record of the turn that produced `state`, from the previous state and this state's extras."""
    if prev_state is None:
        return LastTurn()
    own_prev, own_now = prev_state["player_active_pokemon"], state["player_active_pokemon"]
    opp_prev, opp_now = prev_state["opponent_active_pokemon"], state["opponent_active_pokemon"]

    def change(prev: dict, now: dict) -> tuple[float, str, bool]:
        if prev["name"] != now["name"]:
            return 0.0, "nostatus", now["status"] == "fnt"
        inflicted = now["status"] if prev["status"] == "nostatus" and now["status"] != "fnt" else "nostatus"
        return now["hp_pct"] - prev["hp_pct"], _status(inflicted), now["status"] == "fnt"

    own_hp, own_status, own_fainted = change(own_prev, own_now)
    opp_hp, opp_status, opp_fainted = change(opp_prev, opp_now)
    own, opp = extras["last_player_action"], extras["last_opponent_action"]
    own_cant, opp_cant = extras["player_cant"], extras["opponent_cant"]
    own_cant = own_cant if own_cant in CANT_REASONS else None
    opp_cant = opp_cant if opp_cant in CANT_REASONS else None
    own_kind, own_name = _no_move_if_cant(_kind(own), own["name"], own_cant)
    opp_kind, opp_name = _no_move_if_cant(_kind(opp), opp["name"], opp_cant)
    return LastTurn(
        own_kind=own_kind,
        own_name=own_name,
        own_cant=own_cant,
        opp_kind=opp_kind,
        opp_name=opp_name,
        opp_cant=opp_cant,
        own_hp_change=own_hp,
        opp_hp_change=opp_hp,
        own_status_inflicted=own_status,
        opp_status_inflicted=opp_status,
        own_fainted=own_fainted,
        opp_fainted=opp_fainted,
    )


@dataclass
class Decision:
    features: Encoded  # inputs: depend only on the state and its extras up to this turn
    t: int  # index of the state in its trajectory
    length: int  # number of states in the trajectory
    action_index: int  # the official move-slot index (-1: no action); depends on guessed moves
    next_action: dict  # the action that followed, by name: {"kind", "name"}
    action_reason: Optional[str]  # why there was no action, when the index is -1
    forced_switch: bool
    win: float  # the observer's result: 1.0 win, 0.5 tie, 0.0 loss
    slot_map: np.ndarray  # Metamon action index (corrected state) -> this model's policy slot (-1: not a legal option)
    label_slots: np.ndarray  # the same for the labels, which are indexed by the ORIGINAL parser's state (Exeggcute's)
    own_status: str  # the observer's active Pokemon's status when it decides ("slp", "frz", ...)
    lock_shape: Optional[str]  # the lock on the observer's active Pokemon: "full", "semi" or None (see lock_shape)
    disabled_move: Optional[str]  # the move Disable took away from the observer's active Pokemon
    struggling: bool  # every move is out of PP, so the only move is Struggle
    unusable_moves: tuple = ()  # policy move slots (0-3) the player cannot pick: out of PP or disabled (see unusable_move_slots)


def result_from_filename(path: str) -> float:
    """The observer's result from the parser's file name: 1 win, 0 loss, 0.5 tie."""
    base = os.path.basename(path)
    for suffix, value in (("_WIN.json.lz4", 1.0), ("_LOSS.json.lz4", 0.0), ("_TIE.json.lz4", 0.5)):
        if base.endswith(suffix):
            return value
    raise ValueError(f"cannot read the result from {base}")


FULL_LOCKS = ("recharge", "charge", "thrash", "rage")  # one option, no switching
# How old a trap can be while the engine still tells the player it is locked. Later the request shows every move
# (the trap may have ended without the player being told), which is also what the player sees.
CERTAIN_TRAP_AGE = 1


def lock_shape(lock: Optional[dict]) -> Optional[str]:
    """The shape of the engine's request for a Pokemon with these locks, apart from sleep and freeze:
    "full" (the locked move only, no switching), "semi" (a single Fight entry, switching allowed) or None."""
    if not lock:
        return None
    if lock["lock"] in FULL_LOCKS:
        return "full"
    if lock["lock"] == "bide" or (lock["lock"] == "trap_user" and lock["age"] <= CERTAIN_TRAP_AGE) or 0 < lock["trapped_age"] <= CERTAIN_TRAP_AGE:
        return "semi"
    return None


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def decision_facts(states: list[dict], actions: list[int], extras: list[dict], t: int) -> dict:
    """What decides the kind of decision t (see build.decision_kind), derived from the trajectory alone."""
    s, e = states[t], extras[t]
    return dict(
        forced_switch=bool(e["forced_switch"]),
        action_index=actions[t],
        action_reason=e["action_reason"],
        own_status=s["player_active_pokemon"]["status"],
        lock_shape=lock_shape(e.get("player_lock")),
        disabled_move=(e.get("player_lock") or {}).get("disabled"),
        # every move has run out of PP: the engine offers Struggle only, and switching stays allowed
        # (a disabled move does not count as usable: the last move with PP can be the disabled one)
        struggling=bool(s["player_active_pokemon"]["moves"])
        and all(
            m["current_pp"] == 0 or _norm(m["name"]) == _norm((e.get("player_lock") or {}).get("disabled") or "")
            for m in s["player_active_pokemon"]["moves"]
        ),
    )


def unusable_move_slots(active, lock: Optional[dict]) -> tuple:
    """The policy move slots of `active` (a Pokemon of canonical_own's party, moves in slot order) that the engine does not
    offer: a move with no PP left, or the disabled move. Metamon's legal set (and so the labels' legal mask) lists every move.
    A trapping move keeps going at 0 PP while its trap lasts (Gen 1), so PP is not checked while the player is trapping."""
    disabled = _norm((lock or {}).get("disabled") or "")
    trapping = (lock or {}).get("lock") == "trap_user"
    return tuple(
        j
        for j, m in enumerate(active.moves[:4])
        if (m.current_pp == 0 and m.max_pp and not trapping) or (disabled and _norm(m.name) == disabled)
    )


def iter_decisions(path: str, tables: Tables) -> Iterator[Decision]:
    traj = load_trajectory(path)
    win = result_from_filename(path)
    states, actions, extras = traj["states"], traj["actions"], traj["extras"]
    prev = None
    for t, (s, a, e) in enumerate(zip(states, actions, extras)):
        state = interface.UniversalState.from_dict(copy.deepcopy(s))
        own_party = [interface.UniversalPokemon.from_dict(copy.deepcopy(m)) for m in e["player_team"] if m is not None]
        opp_revealed = [interface.UniversalPokemon.from_dict(copy.deepcopy(m)) for m in e["opponent_team"] if m is not None]
        ordered_party, _ = canonical_own(state, own_party)
        features = encode_observer(state, own_party, opp_revealed, last_turn_from_extras(prev, s, e), t + 1, tables, extras=e)
        yield Decision(
            features=features,
            t=t,
            length=len(states),
            next_action=e["next_player_action"],
            win=win,
            slot_map=action_slot_map(state, ordered_party),
            label_slots=label_slot_map(interface.UniversalState.from_dict(copy.deepcopy(traj["legacy_states"][t])), ordered_party),
            unusable_moves=unusable_move_slots(ordered_party[0], e.get("player_lock")),
            **decision_facts(states, actions, extras, t),
        )
        prev = s
