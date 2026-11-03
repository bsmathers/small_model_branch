"""Upgrades trajectories in the official parsed format (states + actions, no raw log) to the format of
`metamon.recurrent.parse`, by deriving the extras from the states.

The official format is what the human replays and the pac-tauros dataset are published in. Their states
carry the previous move of each active Pokemon and the active/reserve lists, from which most of what the
model reads can be rebuilt:

  derived exactly        the party lists (own party with fainted members, opponent's revealed Pokemon),
                         the forced-switch flag
  derived by rules       the last action of each side (a switch when the active changed, else the move
                         the state names, if it is fresh), the cannot-act reasons that a status or a
                         recharge move explain, the partial-trapping flags (from trap-move hits)
  not recoverable        flinch, full paralysis, confusion self-hits, wake-up turns, faint-ended turns,
                         timeouts, and side conditions other than the most recent one

Every upgraded file is marked `"derived": true` in each extras entry, and the encoder turns that into an
input flag so the model can tell. Accuracy of each rule against the raw-log parser is measured by
`check_upgrade.py`.

    python -m metamon.recurrent.upgrade IN_DIR OUT_DIR [--limit N] [--shard i/n]
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os

import lz4.frame

TRAP_MOVES = {"wrap", "bind", "clamp", "firespin"}
LOCK_MOVES = TRAP_MOVES | {"thrash", "petaldance", "bide"}  # continue on later turns without spending PP
RECHARGE_MOVES = {"hyperbeam"}
# moves that can make the target flinch in Gen 1
FLINCH_MOVES = {"bite", "headbutt", "stomp", "rollingkick", "hyperfang", "lowkick", "bonemerang_", "rockslide", "waterfall", "boneclub", "icepunch_"}
NONE_NAMES = {"", "none", "nomove"}
NO_ACTION = {"kind": "none", "name": ""}


def _speed(mon: dict) -> float:
    """Relative speed from the base stat (paralysis quarters it in Gen 1)."""
    return mon["base_spe"] * (0.25 if mon["status"] == "par" else 1.0)


def _used_move(before: dict, after: dict) -> str | None:
    """The move a Pokemon used between two states, from its move list: a move whose PP dropped, or one
    that was not listed before (revealed by its first use). The list is the Pokemon's own, so this holds
    only when it was in play in both states."""
    pp = {m["name"]: m["current_pp"] for m in before["moves"]}
    for m in after["moves"]:
        if m["name"] not in pp or m["current_pp"] < pp[m["name"]]:
            return m["name"]
    return None


def derive_extras(states: list[dict], actions: list[int]) -> list[dict]:
    out = []
    own_seen: dict[str, dict] = {}
    opp_seen: dict[str, dict] = {}
    opp_order: list[str] = []
    trapped = {"player": False, "opponent": False}
    for t, state in enumerate(states):
        prev = states[t - 1] if t > 0 else None
        own, opp = state["player_active_pokemon"], state["opponent_active_pokemon"]

        # ---- party lists
        current_own = {m["name"]: m for m in [own, *state["available_switches"]]}
        for name, mon in current_own.items():
            own_seen[name] = copy.deepcopy(mon)
        for name, mon in own_seen.items():
            if name not in current_own:  # dropped from the state: fainted
                mon["hp_pct"] = 0.0
                mon["status"] = "fnt"
        if opp["name"] not in opp_seen:
            opp_order.append(opp["name"])
        opp_seen[opp["name"]] = copy.deepcopy(opp)

        last = {"player": NO_ACTION, "opponent": NO_ACTION}
        cant = {"player": None, "opponent": None}
        if prev is None:
            # both leads are switched in at the start
            last = {"player": {"kind": "switch", "name": own["name"]}, "opponent": {"kind": "switch", "name": opp["name"]}}
        elif prev["forced_switch"]:
            # the state after a forced replacement belongs to the turn that caused it: both records are
            # the previous state's
            for w in ("player", "opponent"):
                last[w] = out[-1][f"last_{w}_action"]
                cant[w] = out[-1][f"{w}_cant"]
        else:
            before = {"player": prev["player_active_pokemon"], "opponent": prev["opponent_active_pokemon"]}
            now = {"player": own, "opponent": opp}
            other = {"player": "opponent", "opponent": "player"}
            switched = {w: before[w]["name"] != now[w]["name"] for w in before}
            a = actions[t - 1]
            # a replacement after a faint is not a choice: the count of living opponents dropped
            opp_fainted_swap = switched["opponent"] and state["opponents_remaining"] < prev["opponents_remaining"]
            faster = {w: _speed(before[w]) > _speed(before[other[w]]) for w in before}

            # pass 1: what each side did
            for w in ("player", "opponent"):
                if switched[w]:
                    if not (w == "opponent" and opp_fainted_swap):
                        last[w] = {"kind": "switch", "name": now[w]["name"]}
                    continue
                used = _used_move(before[w], now[w])
                prior = out[-1][f"last_{w}_action"]
                if w == "player" and a == -1:
                    continue  # the label says there was no action
                if used is not None:
                    last[w] = {"kind": "move", "name": used}
                elif prior["kind"] == "move" and norm_name(prior["name"]) in LOCK_MOVES and not switched[other[w]]:
                    last[w] = prior  # a trap or lock move continues without spending PP

            # pass 2: why a side that did not act could not
            for w in ("player", "opponent"):
                if last[w]["kind"] != "none" or switched[w] and not (w == "opponent" and opp_fainted_swap):
                    continue
                o = other[w]
                prior = out[-1][f"last_{w}_action"]
                status_before = before[w]["status"]
                if w == "opponent" and opp_fainted_swap:
                    cant[w] = "fainted" if (faster[o] and last[o]["kind"] == "move") else None
                    if cant[w] is None:  # it acted, then fainted: the move is the one its list shows used last
                        last[w] = {"kind": "move", "name": prev["opponent_prev_move"]["name"]} if prev["opponent_prev_move"]["name"] not in NONE_NAMES else NO_ACTION
                elif w == "player" and a == -1 and now[w]["hp_pct"] == 0:
                    cant[w] = "fainted"
                elif prior["kind"] == "move" and norm_name(prior["name"]) in RECHARGE_MOVES:
                    cant[w] = "recharge"
                elif status_before == "slp":
                    cant[w] = "slp" if now[w]["status"] == "slp" else "wokeup"
                elif status_before == "frz":
                    cant[w] = "frz"
                elif last[o]["kind"] == "move" and norm_name(last[o]["name"]) in TRAP_MOVES:
                    cant[w] = "partiallytrapped"
                elif status_before == "par":
                    cant[w] = "par"
                elif last[o]["kind"] == "move" and norm_name(last[o]["name"]) in FLINCH_MOVES and faster[o]:
                    cant[w] = "flinch"
                elif w == "player" and a != -1 and now[w]["hp_pct"] == before[w]["hp_pct"] and False:
                    cant[w] = None

            # partial trapping: a trap move dealing damage traps the target until either side switches or the
            # trapper uses another move
            for victim in ("player", "opponent"):
                trapper = other[victim]
                if switched[victim] or switched[trapper]:
                    trapped[victim] = False
                elif last[trapper]["kind"] == "move":
                    if norm_name(last[trapper]["name"]) in TRAP_MOVES:
                        if now[victim]["hp_pct"] < before[victim]["hp_pct"]:
                            trapped[victim] = True
                    else:
                        trapped[victim] = False
        conds = lambda c: [] if c in ("noconditions", "") else [c]
        out.append(
            {
                "derived": True,
                "turn": t,
                "last_player_action": last["player"],
                "last_opponent_action": last["opponent"],
                "player_cant": cant["player"],
                "opponent_cant": cant["opponent"],
                "next_player_action": NO_ACTION,
                "action_reason": None,
                # the trap length is hidden and the age of a derived trap unknown: 0.5 stands for "probably trapped"
                "player_trapped": 0.5 if trapped["player"] else 0.0,
                "opponent_trapped": 0.5 if trapped["opponent"] else 0.0,
                "player_effects": [],  # (not recoverable from the official states)
                "opponent_effects": [],
                "player_conditions_all": conds(state["player_conditions"]),
                "opponent_conditions_all": conds(state["opponent_conditions"]),
                "player_team": list(copy.deepcopy(own_seen).values()),
                "opponent_team": [copy.deepcopy(opp_seen[n]) for n in opp_order],
                "forced_switch": bool(state["forced_switch"]),
            }
        )
    return out


def norm_name(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def upgrade_file(path: str, out_dir: str) -> bool:
    with lz4.frame.open(path, "rb") as f:
        data = json.loads(f.read())
    if "extras" in data:
        return False  # already in the new format
    data["extras"] = derive_extras(data["states"], data["actions"])
    # the official files ARE the original parser's output (what Exeggcute was trained on)
    data["legacy_states"], data["legacy_actions"] = data["states"], data["actions"]
    name = os.path.basename(path)
    if not name.endswith(("_WIN.json.lz4", "_LOSS.json.lz4", "_TIE.json.lz4")):
        # no result in the name: read it from the final state
        last = data["states"][-1]
        result = "WIN" if last.get("battle_won") else "LOSS" if last.get("battle_lost") else None
        if result is None:
            return False
        name = name[: -len(".json.lz4")] + f"_{result}.json.lz4"
    os.makedirs(out_dir, exist_ok=True)
    with lz4.frame.open(os.path.join(out_dir, name), "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.input, "**", "*.json.lz4"), recursive=True))
    i, n = map(int, args.shard.split("/"))
    files = files[i::n]
    if args.limit:
        files = files[: args.limit]
    files = [f for f in files if not glob.glob(os.path.join(args.output, os.path.basename(f)[: -len(".json.lz4")] + "*"))]
    done = 0
    for f in files:
        done += upgrade_file(f, args.output)
        if done and done % 1000 == 0:
            print(f"upgraded {done} of {len(files)}", flush=True)
    print(f"upgraded {done} of {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
