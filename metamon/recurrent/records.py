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
from metamon.recurrent.features import CANT_REASONS, KINDS, STATUS_IDS, Encoded, LastTurn, action_slot_map, canonical_own, encode_observer
from metamon.recurrent.tables import Tables


def load_trajectory(path: str) -> dict:
    """A trajectory written by metamon.recurrent.parse: states, actions and extras."""
    with lz4.frame.open(path, "rb") as f:
        data = json.loads(f.read())
    if "extras" not in data:
        raise ValueError(f"{os.path.basename(path)} has no extras: parse the raw log with metamon.recurrent.parse")
    return data


def _kind(summary: dict) -> str:
    kind = summary["kind"]
    if kind == "noop":
        kind = "recharge"
    return kind if kind in KINDS else "none"


def _status(name: str) -> str:
    return name if name in STATUS_IDS else "nostatus"


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
    return LastTurn(
        own_kind=_kind(own),
        own_name=own["name"],
        own_cant=own_cant if own_cant in CANT_REASONS else None,
        opp_kind=_kind(opp),
        opp_name=opp["name"],
        opp_cant=opp_cant if opp_cant in CANT_REASONS else None,
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
    slot_map: np.ndarray  # Metamon action index -> this model's policy slot (-1: not a legal option)
    own_status: str  # the observer's active Pokemon's status when it decides ("slp", "frz", ...)
    trapped: bool  # the observer was already trapped (Wrap, Bind, ...) when it decided, and lost the turn to it; it can still switch
    recharge_next: bool  # the observer must recharge this turn: its only option is the no-op


def result_from_filename(path: str) -> float:
    """The observer's result from the parser's file name: 1 win, 0 loss, 0.5 tie."""
    base = os.path.basename(path)
    for suffix, value in (("_WIN.json.lz4", 1.0), ("_LOSS.json.lz4", 0.0), ("_TIE.json.lz4", 0.5)):
        if base.endswith(suffix):
            return value
    raise ValueError(f"cannot read the result from {base}")


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
            action_index=a,
            next_action=e["next_player_action"],
            action_reason=e["action_reason"],
            forced_switch=bool(e["forced_switch"]),
            win=win,
            slot_map=action_slot_map(state, ordered_party),
            own_status=s["player_active_pokemon"]["status"],
            # Both signals are needed. The parser's flag alone never learns that a trap ran out (or that the player
            # switched away); a lost turn alone also happens when a fresh Wrap lands before the player moves, while
            # it was still free to use any move.
            trapped=bool(
                e["player_trapped"]
                and t + 1 < len(extras)
                and not s["forced_switch"]
                and extras[t + 1]["player_cant"] == "partiallytrapped"
            ),
            recharge_next=bool(
                t + 1 < len(extras)
                and not s["forced_switch"]
                and (extras[t + 1]["player_cant"] == "recharge" or extras[t + 1]["last_player_action"]["kind"] == "recharge")
            ),
        )
        prev = s
