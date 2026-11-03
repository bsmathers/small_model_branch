"""Trajectory generator for the recurrent model, from raw Showdown protocol logs.

For one point of view (the player with username `pov_username`), the decisions are:
  - one per turn: the public state at the end of the turn, paired with both sides' actions in the
    next turn (actions are recorded by name: a move, or a switch target);
  - one per forced switch: the state right after the faint, paired with the replacement.

A side that could not act in a turn records a `none` action with the protocol's |cant| reason:
par, slp, frz, recharge, partiallytrapped, or flinch. Those are the -1 actions in the parsed data.

Trapping: Gen 1 logs have no trap start or end line. A side is trapped from a
|cant|...|partiallytrapped line on its active until that side switches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Mon:
    name: str
    hp: float = 1.0
    status: str = "nostatus"
    moves: list[str] = field(default_factory=list)


@dataclass
class SideState:
    active: Optional[str] = None
    trapped: bool = False
    mons: dict[str, Mon] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def mon(self, name: str) -> Mon:
        if name not in self.mons:
            self.mons[name] = Mon(name)
            self.order.append(name)
        return self.mons[name]

    def snapshot(self) -> dict:
        act = self.mons.get(self.active or "")
        return {
            "active": self.active,
            "active_hp": act.hp if act else 0.0,
            "active_status": act.status if act else "nostatus",
            "trapped": self.trapped,
            "party": [
                {"name": n, "hp": self.mons[n].hp, "status": self.mons[n].status, "moves": list(self.mons[n].moves)}
                for n in self.order
            ],
        }


def _side(ident: str) -> str:
    return ident[:2]


def _name(ident: str) -> str:
    return ident.split(": ", 1)[1] if ": " in ident else ident


def _hp_status(value: str) -> tuple[float, Optional[str]]:
    parts = value.split(" ")
    cur, _, mx = parts[0].partition("/")
    try:
        frac = int(cur) / max(1, int(mx))
    except ValueError:
        frac = 0.0
    return frac, (parts[1] if len(parts) > 1 else None)


def _action(kind: str, name: str = "", reason: str = "") -> dict:
    return {"kind": kind, "name": name, "reason": reason}


def _pov_side(lines: list[str], username: str) -> str:
    for line in lines:
        if line.startswith("|player|"):
            parts = line.split("|")
            if len(parts) > 3 and parts[3] == username:
                return parts[2]
    raise ValueError(f"player {username!r} not found")


def generate(path: Path, pov_username: str) -> list[dict]:
    lines = path.read_text(errors="ignore").splitlines()
    pov = _pov_side(lines, pov_username)
    opp = "p2" if pov == "p1" else "p1"
    states = {"p1": SideState(), "p2": SideState()}

    # split into blocks at |turn| lines (block 0 = before the first turn: the lead switch-ins)
    blocks: list[list[str]] = [[]]
    for line in lines:
        if line.startswith("|turn|"):
            blocks.append([])
        blocks[-1].append(line)

    decisions: list[dict] = []
    prev_snapshot = None  # snapshot at the end of the previous block, awaiting its next actions
    prev_turn = 0
    for block_index, block in enumerate(blocks):
        turn = prev_turn
        actions: dict[str, dict] = {}
        faint_pending = {"p1": False, "p2": False}
        mid_decisions: list[dict] = []
        for line in block:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            kind = parts[1]
            if kind == "turn":
                turn = int(parts[2])
                continue
            if kind in ("switch", "drag"):
                side = _side(parts[2])
                name = _name(parts[2])
                forced = faint_pending[side] and actions.get(side, {}).get("kind") == "faint"
                if forced:
                    # forced replacement: the state is the position after the faint; the switch
                    # is that decision's action, not part of this block's turn actions
                    mid_decisions.append({"snapshot": _snap(states, pov, opp, turn), "side": side, "replacement": parts[3]})
                faint_pending[side] = False
                st = states[side]
                if st.active is not None and st.active != name:
                    st.trapped = False
                st.active = name
                mon = st.mon(name)
                if len(parts) > 4:
                    mon.hp, status = _hp_status(parts[4])
                    if status:
                        mon.status = status
                if not forced:
                    actions[side] = _action("switch", parts[3])
            elif kind == "move":
                side = _side(parts[2])
                st = states[side]
                mon = st.mon(_name(parts[2]))
                if parts[3] not in mon.moves:
                    mon.moves.append(parts[3])
                actions.setdefault(side, _action("move", parts[3]))
            elif kind == "cant":
                side = _side(parts[2])
                reason = parts[3] if len(parts) > 3 else ""
                actions[side] = _action("none", reason=reason)
                if reason == "partiallytrapped":
                    states[side].trapped = True
            elif kind in ("-damage", "-heal"):
                side = _side(parts[2])
                hp, status = _hp_status(parts[3])
                mon = states[side].mon(_name(parts[2]))
                mon.hp = hp
                if status:
                    mon.status = status
            elif kind == "-status":
                states[_side(parts[2])].mon(_name(parts[2])).status = parts[3]
            elif kind == "-curestatus":
                states[_side(parts[2])].mon(_name(parts[2])).status = "nostatus"
            elif kind == "faint":
                side = _side(parts[2])
                mon = states[side].mon(_name(parts[2]))
                mon.hp = 0.0
                mon.status = "fnt"
                faint_pending[side] = True
                actions[side] = _action("faint")
        # the block's actions are the next actions of the previous block's decision
        if prev_snapshot is not None:
            decisions.append(_decision(prev_snapshot, actions, opp, pov))
        for mid in mid_decisions:
            # the forced replacement is itself a decision, whose next action is that switch
            replacement = _action("switch", mid["replacement"])
            snap = mid["snapshot"]
            if mid["side"] == pov:
                decisions.append(_decision(snap, {pov: replacement}, opp, pov, own_only=True))
        prev_snapshot = _snap(states, pov, opp, turn)
        prev_turn = turn
    return decisions


def _snap(states, pov: str, opp: str, turn: int) -> dict:
    return {"turn": turn, "own": states[pov].snapshot(), "opp": states[opp].snapshot()}


def _decision(snap: dict, actions: dict, opp: str, pov: str, own_only: bool = False) -> dict:
    out = {
        "turn": snap["turn"],
        "pov": pov,
        "own": snap["own"],
        "opp": snap["opp"],
        "next_own": actions.get(pov, _action("none", reason="no-action")),
        "next_opp": actions.get(opp, _action("none", reason="no-action")),
    }
    if out["next_own"]["kind"] == "faint":
        out["next_own"] = _action("none", reason="no-action")
    if out["next_opp"]["kind"] == "faint":
        out["next_opp"] = _action("none", reason="no-action")
    return out


def write_jsonl(records: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
