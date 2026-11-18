#!/usr/bin/env python3
"""The small recurrent policy (metamon.recurrent, 1.3M parameters) as a snapshot policy server for the duel harness.

Protocol (the same as bot2's metamon_tauros_server.py): one JSON line per request. {"op": "reset"} starts a game
and is answered "ok"; an observer snapshot with "legal" (lowercase option names: "bodyslam", "switch tauros",
"no move") is answered with one of the legal names: the argmax of the actor head over the legal options.

The shim. TaurosV0 reads the snapshot through bot2's SnapshotEncoder (snapshot -> metamon UniversalState ->
tokens). This model was trained on parsed replays: per decision, metamon.recurrent.features.encode_observer reads
the UniversalState, the observer's full party, the opponent's revealed Pokemon (all put in Exeggcute's order, which
depends only on the current state: see features.canonical_own / canonical_opponent), the public
record of the last turn (each side's action, why it could not act, HP and status changes) and the parser's extras
(effects, trapped flags, side conditions, Counter's damage memory), and the model carries a recurrent state from
decision to decision. Here:
  - the UniversalState comes from the same SnapshotEncoder, so both models read the snapshot the same way;
  - the party comes from snapshot.user.pokemon, the revealed opponents from snapshot.opponent.revealed;
  - the last-turn record is reconstructed by comparing consecutive snapshots: our action is the one we sent
    (a move counts as used when its PP went down), the opponent's from its species and last_move; the reasons for
    not acting follow the parser's names (par, slp, wokeup, frz, fainted, opponent_fainted, recharge, ...);
  - a snapshot where we only wait for the opponent's forced switch ("no move" with opponent_force_switch) is not a
    decision in parsed replays (no request is sent there): it is answered without stepping the model;
  - "turn" is the decision count (records.iter_decisions passes t + 1).
Not reproduced (the snapshot does not carry it): the opponent's volatile effects and screens, Counter's damage
memory (Counter's damage feature reads 0), and the opponent's PP (a paralysed opponent that repeats its last move
is taken to have moved).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

# (repo copy) this repository's metamon, and this folder's metamon_connector / snapshot_effects (copied from bot2's selfplay/tools)
METAMON_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(METAMON_ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from metamon.interface import move_name, pokemon_name  # noqa: E402
from metamon.recurrent.features import CANT_REASONS, LastTurn, canonical_own, encode_observer, policy_options  # noqa: E402
from metamon.recurrent.features import _move_name, _pokemon_name  # noqa: E402  (memoized move_name / pokemon_name)
from metamon.recurrent.blend import PolicyBlend  # noqa: E402
from metamon.recurrent.tables import Tables  # noqa: E402
from metamon_connector import SnapshotEncoder  # noqa: E402
from metamon.recurrent.features import EFFECTS  # noqa: E402
from metamon.recurrent.parse import TRAP_PROBABILITY  # noqa: E402
from snapshot_effects import SnapshotEffectMemory  # noqa: E402

TRAPPING = {"wrap", "bind", "firespin", "clamp"}


def public_volatiles(side: dict) -> list:
    """A side's public volatiles (knowledge.zig publicVolatileList): "binding" means that side's active is using a trapping
    move. (The older "volatiles" list names the same engine flag "partiallytrapped", which reads as the opposite.)"""
    return list(side.get("public_volatiles") or [])


def effect_payload(snap: dict, side: str, species: str) -> dict:
    """The input of SnapshotEffectMemory for one side's active, from the public volatiles: a Pokemon caught in the other
    side's trap carries "partiallytrapped" (named after the trapping move, as the replay parser's states do)."""
    other = "opponent" if side == "user" else "user"
    mine, theirs = public_volatiles(snap[side]), public_volatiles(snap[other])
    vols = [v for v in mine if v != "binding"] + (["partiallytrapped"] if "binding" in theirs else [])
    return {"base_species": species, "volatiles": vols, "last_move": snap[side].get("last_move", "none"),
            "trapping_move": snap[other].get("last_move", "none"), "turn": snap.get("turn", 0)}


def counter_damage(snap: dict, game, state, tables):
    """(repo copy) Counter's memory in the parser's form ({"kind", "side", "name", "pct"}) from the engine's lastDamage, so that
    features.counter_fraction reads 2 x lastDamage / the target's max HP. The side that took it: ours when our active lost
    exactly that much HP since the last decision, else the opponent's. pct is relative to the max-HP estimate the feature
    uses for that Pokemon, so the feature recovers the exact amount."""
    from metamon.recurrent.features import _stat_of

    damage = int(snap.get("last_damage") or 0)
    if damage <= 0:
        return None
    side = "opponent"
    if game.prev is not None:
        before = game.prev[0]["user"]["pokemon"][0]
        after = snap["user"]["pokemon"][0]
        if norm(before["id"]) == norm(after["id"]) and int(before["hp"]) - int(after["hp"]) == damage:
            side = "player"
        elif norm(before["id"]) == norm(after["id"]) and int(before["hp"]) > int(after["hp"]) and game.last.opp_hp_change >= 0:
            side = "player"
    mon = state.player_active_pokemon if side == "player" else state.opponent_active_pokemon
    name = norm(mon.base_species or mon.name)
    try:
        estimate = _stat_of(tables.base_stats(name)["hp"], hp=True)
    except KeyError:
        return None
    if side == "opponent":
        # (repo copy) damage to the opponent is public only as its HP percentage's drop (Showdown's rounded percentage); the
        # engine's exact amount, or what a Substitute absorbed, is not: none when its active changed or showed no drop
        if game.prev is None:
            return None
        pb, pa = game.prev[0]["opponent"], snap["opponent"]
        if norm(pb["active_species"]) != norm(pa["active_species"]):
            return None
        drop = float(pb["active"]["hp"]) - float(pa["active"]["hp"])
        if drop <= 0:
            return None
        return {"kind": "hit", "side": side, "name": name, "pct": 100.0 * drop}
    return {"kind": "hit", "side": side, "name": name, "pct": 100.0 * damage / max(estimate, 1.0)}


def showdown_pct(hp: int, top: int) -> int:
    """Pokémon Showdown's HP Percentage Mod: rounded up, and 99 for anything below full."""
    if hp <= 0:
        return 0
    pct = -(-100 * hp // max(top, 1))
    return 99 if pct >= 100 and hp < top else min(pct, 100)


def norm(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


class _Space:  # SnapshotEncoder keeps its per-battle effect memory on the observation space object
    pass


class Game:
    """Per-game memory: the recurrent state and what the last-turn record needs."""

    def __init__(self, policy):
        self.h = policy.initial_state()  # (one recurrent state per blended member)
        self.space = _Space()
        self.prev = None  # (snapshot, UniversalState) of the last stepped decision
        self.sent = None  # what we answered there
        self.last = LastTurn()
        self.decisions = 0
        self.counter_selected = {"player": False, "opponent": False}  # (repo copy) Counter's selection memory, as the parser keeps it
        self.opp_faint_switch = False  # a wait state for the opponent's forced switch came since the last decision
        self.opp_species = None  # the opponent's active and the last move it was seen using since it came in
        self.opp_move = ""
        # the single active effect of each side (screens are read from it), selected as the replay parser's states select it;
        # kept apart from the SnapshotEncoder's own memory, which reads the older volatiles list
        self.effects = SnapshotEffectMemory()
        # trap age per trapping side: (trapper, uses so far, battle turn of the last counted use), as the parser's trapped_age
        self.trap = {"user": None, "opponent": None}
        # each side's first real action (a move run or a recharge) since our last decision: a turn that ends in the
        # opponent's faint reaches us in two snapshots (waiting for its replacement, then the replacement), and only the
        # second one is a decision
        self.actions: dict = {}

    def fork(self) -> "Game":
        """An independent copy (the search server's nodes): the recurrent state is replaced, never changed in place, and the
        stored previous snapshot is not changed after it is stored, so only the memories that change in place are copied."""
        other = copy.copy(self)
        other.space = copy.deepcopy(self.space)
        other.effects = copy.deepcopy(self.effects)
        other.counter_selected = dict(self.counter_selected)
        other.trap = dict(self.trap)
        other.actions = dict(self.actions)
        return other

    def note_actions(self, snap: dict) -> None:
        for side, act in (snap.get("last_actions") or {}).items():
            if side not in self.actions or (self.actions[side]["executed"] == "none" and not self.actions[side]["recharged"]):
                self.actions[side] = act

    def trap_probability(self, snap: dict, trapper_side: str) -> float:
        """The replay parser's trapped flag for the Pokemon caught in `trapper_side`'s trap: P(still trapped) after the
        number of uses so far (parse.TRAP_PROBABILITY). A use is counted once per battle turn while the trapper's Binding
        is on; it is reset when the trapper changes or stops."""
        if "binding" not in public_volatiles(snap[trapper_side]):
            self.trap[trapper_side] = None
            return 0.0
        trapper = norm(snap[trapper_side]["pokemon"][0]["id"]) if trapper_side == "user" else norm(snap["opponent"]["active_species"])
        turn = int(snap.get("turn", 0))
        prev = self.trap[trapper_side]
        if prev is None or prev[0] != trapper:
            uses = 1
        else:
            uses = prev[1] + (1 if turn != prev[2] else 0)
        self.trap[trapper_side] = (trapper, uses, turn)
        return TRAP_PROBABILITY.get(uses, 0.0)


def own_cant(prev_snap, prev_state, snap, state, sent, prev_last: LastTurn, action=None):
    """(kind, name, cant) of our own last action."""
    if sent.startswith("switch "):
        name = sent[7:]
        # a replacement after a faint keeps the reason of the turn the Pokemon fainted in (as the parser does)
        return "switch", name, (prev_last.own_cant if prev_snap.get("force_switch") else None)
    before, after = prev_snap["user"]["pokemon"][0], snap["user"]["pokemon"][0]
    if action is not None:  # the engine's record of the update (main.zig lastAction)
        if action["recharged"]:
            return "recharge", "", "recharge"
        if action["executed"] != "none":
            return "move", action["executed"], None
        if sent == "no move":
            return "none", "", None
        return ("none", "", cant_reason(prev_snap, prev_state, snap, state, "user"))
    if sent == "no move":
        if "mustrecharge" in prev_snap["user"].get("volatiles", []):
            return "recharge", "", "recharge"
        return "none", "", None
    used = False
    if norm(before["id"]) == norm(after["id"]):
        pp = lambda mon: {norm(m["id"]): m.get("pp") for m in mon.get("moves", [])}
        b, a = pp(before), pp(after)
        key = norm(sent)
        used = key in b and key in a and a[key] is not None and b[key] is not None and a[key] < b[key]
        if key == "struggle" and norm(snap["user"].get("last_move")) == "struggle":
            used = True
    if not used and norm(sent) in TRAPPING and "binding" in public_volatiles(prev_snap["user"]) and "binding" in public_volatiles(snap["user"]):
        used = True  # our trap went on (a continuation uses no PP; the log shows the move "[from]" itself)
    if used:
        return "move", sent, None
    status_before = prev_state.player_active_pokemon.status
    if after["hp"] <= 0 or snap.get("force_switch"):
        return "none", "", "fainted"
    if status_before == "slp":
        return "none", "", ("wokeup" if state.player_active_pokemon.status == "nostatus" else "slp")
    if status_before == "frz":
        return "none", "", "frz"
    if status_before == "par":
        return "none", "", "par"
    if state.opponent_active_pokemon.hp_pct <= 0 or snap.get("opponent_force_switch"):
        return "none", "", "opponent_fainted"
    if "confusion" in snap["user"].get("volatiles", []):
        return "none", "", "confusion"
    # caught in the opponent's trap (its Binding: the older "volatiles" list's "partiallytrapped" is OUR Binding, the reverse)
    if "binding" in public_volatiles(snap["opponent"]) or "binding" in public_volatiles(prev_snap["opponent"]):
        return "none", "", "partiallytrapped"
    return "none", "", None


FLINCH_MOVES = {"bite", "stomp", "headbutt", "rollingkick", "hyperfang", "lowkick", "boneclub"}


def cant_reason(prev_snap, prev_state, snap, state, side: str):
    """Why `side` ran no move in the update between two snapshots (it did not switch either), in the parser's names, from what
    the log would show: the order follows the engine's checks before a move."""
    other = "opponent" if side == "user" else "user"
    mine_before = prev_state.player_active_pokemon if side == "user" else prev_state.opponent_active_pokemon
    mine_now = state.player_active_pokemon if side == "user" else state.opponent_active_pokemon
    theirs_now = state.opponent_active_pokemon if side == "user" else state.player_active_pokemon
    if mine_now.hp_pct <= 0 or snap.get("force_switch" if side == "user" else "opponent_force_switch"):
        return "fainted"
    status_before, status_now = mine_before.status, mine_now.status
    if status_before == "slp":
        return "slp" if status_now == "slp" else "wokeup"  # (waking takes the turn; it can be paralysed right after)
    if status_before == "frz":
        return "frz"
    if status_now in ("slp", "frz"):  # put to sleep or frozen before it could act
        return status_now
    if theirs_now.hp_pct <= 0 or snap.get("opponent_force_switch" if side == "user" else "force_switch"):
        return "opponent_fainted"
    if "binding" in public_volatiles(snap[other]) or "binding" in public_volatiles(prev_snap[other]):
        return "partiallytrapped"  # caught in the other side's trap
    other_action = (snap.get("last_actions") or {}).get(other) or {}
    if norm(other_action.get("executed")) in FLINCH_MOVES and "substitute" not in public_volatiles(prev_snap[side]):
        return "flinch"
    if status_before == "par":
        return "par"
    if "confusion" in public_volatiles(prev_snap[side]):
        return "confusion"  # hurt itself
    return None


def opp_cant(prev_snap, prev_state, snap, state, faint_switch: bool, prev_last: LastTurn, known_move: str = "", action=None):
    before_species, after_species = norm(prev_snap["opponent"]["active_species"]), norm(snap["opponent"]["active_species"])
    moved_now, moved_before = norm(snap["opponent"].get("last_move")), norm(prev_snap["opponent"].get("last_move"))
    if prev_snap.get("force_switch") and not prev_snap.get("opponent_force_switch"):
        # only our replacement came in: the parser keeps the opponent's action of the turn we fainted in
        return prev_last.opp_kind, prev_last.opp_name, prev_last.opp_cant
    if action is not None:  # the engine's record of the update(s) since our last decision (main.zig lastAction)
        if faint_switch:  # it fainted and was replaced: it moved first, or fainted before acting
            return ("move", action["executed"], None) if action["executed"] != "none" else ("none", "", "fainted")
        if before_species != after_species:
            return "switch", snap["opponent"]["active_species"], None
        if action["recharged"]:
            return "recharge", "", "recharge"
        if action["executed"] != "none":
            return "move", action["executed"], None
        return "none", "", cant_reason(prev_snap, prev_state, snap, state, "opponent")
    if faint_switch:  # it fainted and was replaced: it moved first (its last move changed) or fainted before acting
        return ("move", snap["opponent"]["last_move"], None) if moved_now != moved_before and moved_now not in ("", "none") else ("none", "", "fainted")
    if before_species != after_species:
        return "switch", snap["opponent"]["active_species"], None
    if state.opponent_active_pokemon.hp_pct <= 0:
        return "none", "", "fainted"
    # The engine clears last_used_move at times (a faint on either side), so "none" is not proof of not acting: the move is
    # taken from memory (known_move: the last move this opponent was seen using) and acting is judged from its effects.
    name = moved_now if moved_now not in ("", "none") else known_move
    if moved_now not in ("", "none") and moved_now != moved_before:
        return "move", name, None
    status_before, opp_before, opp_now = prev_state.opponent_active_pokemon.status, prev_state.opponent_active_pokemon, state.opponent_active_pokemon
    ours_before = next((m for m in prev_snap["user"]["pokemon"] if norm(m["id"]) == norm(snap["user"]["pokemon"][0]["id"])), None)
    ours_now = snap["user"]["pokemon"][0]
    hit = ours_before is not None and (ours_now["hp"] < ours_before["hp"] or (ours_before["status"] == "none" and ours_now["status"] not in ("none",)))
    acted = hit or snap.get("force_switch") or opp_now.hp_pct > opp_before.hp_pct + 1e-6 or (
        (opp_now.atk_boost, opp_now.def_boost, opp_now.spa_boost, opp_now.spe_boost) != (opp_before.atk_boost, opp_before.def_boost, opp_before.spa_boost, opp_before.spe_boost))
    if status_before == "slp":
        return "none", "", ("wokeup" if opp_now.status == "nostatus" else "slp")
    if status_before == "frz" and not acted:
        return "none", "", "frz"
    if not acted and "binding" in public_volatiles(snap["user"]):
        return "none", "", "partiallytrapped"  # caught in our trap: it cannot move (switching was handled above)
    if status_before == "par" and not acted:
        return "none", "", "par"
    if not acted and prev_last.opp_kind == "move" and norm(prev_last.opp_name) == "hyperbeam" and not prev_snap.get("force_switch"):
        return "recharge", "", "recharge"
    return ("move", name, None) if name else ("none", "", None)


def change(prev_mon, now_mon):
    if prev_mon.name != now_mon.name:
        return 0.0, "nostatus", now_mon.status == "fnt"
    inflicted = now_mon.status if prev_mon.status == "nostatus" and now_mon.status != "fnt" else "nostatus"
    return now_mon.hp_pct - prev_mon.hp_pct, inflicted, now_mon.status == "fnt"


def make_encoder():
    """The snapshot encoder and the move/species tables, set up as training saw the data."""
    tables = Tables.from_dex(1)
    encoder = SnapshotEncoder(METAMON_ROOT, observation_space=_Space())
    # this metamon's UniversalPokemon also takes the actual stats; the parsed training data has none (-1 everywhere)
    from types import SimpleNamespace
    from metamon.interface import UniversalMove, UniversalPokemon, UniversalState

    def pokemon_without_stats(*a, **kw):
        for k in ("hp_stat", "atk_stat", "def_stat", "spa_stat", "spd_stat", "spe_stat"):
            kw.setdefault(k, -1)
        return UniversalPokemon(*a, **kw)

    pokemon_without_stats.universal_types = UniversalPokemon.universal_types
    encoder._universal = SimpleNamespace(UniversalMove=UniversalMove, UniversalPokemon=pokemon_without_stats, UniversalState=UniversalState)
    return encoder, tables


class Prepared:
    """One decision's model input and what the rest of the loop reads."""

    def __init__(self, enc, state, party, opp_active, ids, floats):
        self.enc, self.state, self.party, self.opp_active, self.ids, self.floats = enc, state, party, opp_active, ids, floats


def prepare(game: Game, snap: dict, encoder, tables):
    """Everything before the model step for one snapshot: updates the game's memories and returns the model input, or None
    for a wait state (only the opponent's forced switch: not a decision in parsed replays, so the model is not stepped)."""
    legal = snap["legal"]
    game.note_actions(snap)
    if legal == ["no move"] and snap.get("opponent_force_switch") and not snap.get("force_switch"):
        game.opp_faint_switch = True  # waiting: not a decision of a parsed replay
        return None
    # (repo copy) own HP as training saw it: recorded games show every Pokemon's HP in percent (Showdown's HP
    # Percentage Mod), not exactly; the raw snapshot keeps the exact values for the checks below
    # (only the own party's entries change: copied, not the whole snapshot)
    enc_snap = dict(snap)
    enc_snap["user"] = dict(snap["user"])
    enc_snap["user"]["pokemon"] = [dict(m) for m in snap["user"]["pokemon"]]
    for m in enc_snap["user"]["pokemon"]:
        m["hp_pct"] = showdown_pct(int(m.get("hp", 0)), int(m.get("maxhp", 1))) / 100.0
    encoder.begin_reuse()  # (the reserves state() builds are reused for the party below: same payloads, same objects)
    try:
        state = encoder.state(enc_snap, game.space)
        # the full party: the active from the state (it carries the boosts), the others from the snapshot
        party = [state.player_active_pokemon] + [encoder._pokemon(m, False, exact_hp=True) for m in enc_snap["user"]["pokemon"][1:]]
    finally:
        encoder.end_reuse()
    opp_active = state.opponent_active_pokemon
    revealed = []  # (in the snapshot's order: encode_observer sorts them by species)
    for r in snap["opponent"]["revealed"][:6]:
        if norm(r["species"]) == norm(opp_active.base_species) or norm(r["species"]) == norm(opp_active.name):
            revealed.append(opp_active)
        else:
            revealed.append(encoder._pokemon({"id": r["species"], "hp_pct": r.get("hp", 1.0), "status": r.get("status", "none"), "moves": r.get("moves", [])}, False, exact_hp=False))
    if game.prev is not None:
        prev_snap, prev_state = game.prev
        acts = game.actions if "last_actions" in snap else {}
        ok, on, oc = own_cant(prev_snap, prev_state, snap, state, game.sent, game.last, acts.get("user"))
        pk, pn, pc = opp_cant(prev_snap, prev_state, snap, state, game.opp_faint_switch, game.last, game.opp_move, acts.get("opponent"))
        own_hp, own_status, own_fainted = change(prev_state.player_active_pokemon, state.player_active_pokemon)
        opp_hp, opp_status, opp_fainted = change(prev_state.opponent_active_pokemon, state.opponent_active_pokemon)
        game.last = LastTurn(
            own_kind=ok, own_name=on, own_cant=oc if oc in CANT_REASONS else None,
            opp_kind=pk, opp_name=pn, opp_cant=pc if pc in CANT_REASONS else None,
            own_hp_change=own_hp, opp_hp_change=opp_hp, own_status_inflicted=own_status, opp_status_inflicted=opp_status,
            own_fainted=own_fainted, opp_fainted=opp_fainted,
        )
        # (repo copy) the parser's Counter selection memory: a move that ran sets it, a turn without one (cant) clears
        # it, a switch leaves it
        for who, kind, name, cant in (("player", ok, on, oc), ("opponent", pk, pn, pc)):
            if kind == "move":
                game.counter_selected[who] = norm(name) == "counter"
            elif cant is not None:
                game.counter_selected[who] = False
    # Effects as the replay parser's states carry them. Screens are read from each active's single effect (in Gen 1
    # Reflect, Light Screen and Mist are volatiles; the parser's side-condition lists are always empty), so the
    # effects are re-selected from the public volatiles, the opponent's included.
    own_species, opp_species = norm(state.player_active_pokemon.base_species), norm(state.opponent_active_pokemon.base_species)
    state.player_active_pokemon.effect = game.effects.select(effect_payload(snap, "user", own_species), "user")
    state.opponent_active_pokemon.effect = game.effects.select(effect_payload(snap, "opponent", opp_species), "opponent")
    opp_active.effect = state.opponent_active_pokemon.effect
    extras = {
        "last_damage": counter_damage(snap, game, state, tables),
        "last_move": {"player": norm(snap["user"].get("last_move")), "opponent": norm(snap["opponent"].get("last_move"))},
        "counter_selected": dict(game.counter_selected),
        "player_cant": game.last.own_cant,
        "opponent_cant": game.last.opp_cant,
        "player_effects": [v for v in public_volatiles(snap["user"]) if v in EFFECTS],
        "opponent_effects": [v for v in public_volatiles(snap["opponent"]) if v in EFFECTS],
        "player_trapped": game.trap_probability(snap, "opponent"),  # caught in the opponent's trap
        "opponent_trapped": game.trap_probability(snap, "user"),
        "player_conditions_all": [],
        "opponent_conditions_all": [],
    }
    game.decisions += 1
    canonical = canonical_own(state, party)  # (computed once: the features and legal_slots both need it)
    enc = encode_observer(state, party, revealed, game.last, game.decisions, tables, extras=extras, canonical=canonical)
    ids ={k: torch.from_numpy(v)[None, None] for k, v in enc.ids.items()}
    floats = {k: torch.from_numpy(v)[None, None] for k, v in enc.floats.items()}
    prep = Prepared(enc, state, party, opp_active, ids, floats)
    prep.canonical = canonical
    return prep


def legal_slots(prep: Prepared, legal: list) -> dict:
    """Option name -> policy slot, with the training code's own definition (Metamon's action indices: the active's moves
    sorted by name, then the living reserves sorted by species); options without a slot are left out."""
    moves, reserves = policy_options(prep.canonical[0])
    slots = {}
    for option in legal:
        if option.startswith("switch "):
            name = _pokemon_name(option[7:])
            if name in reserves:
                slots[option] = 4 + reserves.index(name)
        elif _move_name(option) in moves:
            slots[option] = moves.index(_move_name(option))
    return slots


def finish(game: Game, snap: dict, prep: Prepared, reply: str) -> None:
    """The memories updated after a decision: the opponent's active and last move, and what we answered."""
    species = norm(snap["opponent"]["active_species"])
    if species != game.opp_species:
        game.opp_species, game.opp_move = species, ""
    if game.last.opp_kind == "move" and game.last.opp_name:
        game.opp_move = norm(game.last.opp_name)
    if norm(snap["opponent"].get("last_move")) not in ("", "none"):
        game.opp_move = norm(snap["opponent"]["last_move"])
    game.prev, game.sent, game.opp_faint_switch = (snap, prep.state), reply, False
    game.actions = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.environ.get("RECURRENT_WEIGHTS", os.path.expanduser("~/model_weights/weights_best.pt")),
                    help="a weights file, or a blend of several: a.pt@0.5,b.pt@0.5 (metamon/recurrent/blend.py)")
    ap.add_argument("--blend-mode", default=os.environ.get("RECURRENT_BLEND_MODE", "mix"), choices=("mix", "logit"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--shadow", default=os.environ.get("RECURRENT_SHADOW"),
                    help="blend|judge: Exeggcute plays (the replies are its), this model only scores the same snapshots; prints the agreement")
    args = ap.parse_args()
    teacher = None
    if args.shadow:
        import subprocess
        # RECURRENT_SHADOW_SERVER: a teacher script that may reply "<option>\t<json {option: prob}>" (the probabilities go to the dump)
        teacher_cmd = ([os.environ["RECURRENT_SHADOW_SERVER"]] if os.environ.get("RECURRENT_SHADOW_SERVER")
                       else ["/Users/macuser/mcts_better_eval/pkmn-search/tools/exeggcute_variant_server.py", "--player", args.shadow])
        teacher = subprocess.Popen(["/Users/macuser/bot2/.venv-metamon/bin/python", "-u", *teacher_cmd],
                                   cwd="/Users/macuser/bot2", stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env={**os.environ, "TORCHDYNAMO_DISABLE": "1"})
    agree = [0, 0]

    def ask_teacher(raw: str) -> str:
        teacher.stdin.write(raw if raw.endswith("\n") else raw + "\n")
        teacher.stdin.flush()
        return teacher.stdout.readline().strip()
    torch.set_num_threads(args.threads)
    # (repo copy) POLICY_TEMPERATURE: 0 (default) plays the argmax, T > 0 samples from softmax(logits / T); the draws are seeded
    # by POLICY_SEED, so a rerun plays the same games
    import random

    temperature = float(os.environ.get("POLICY_TEMPERATURE", "0"))
    rng = random.Random(f"student-{os.environ.get('POLICY_SEED', '0')}")
    model = PolicyBlend.load(args.weights, args.blend_mode)
    encoder, tables = make_encoder()
    print(f"recurrent policy: {args.weights}" + (f" (blend of {model.size}, {args.blend_mode})" if model.size > 1 else ""), file=sys.stderr, flush=True)
    game = Game(model)
    for raw in sys.stdin:
        try:
            snap = json.loads(raw)
            told = ask_teacher(raw) if teacher else None
            told_probs = None
            if told and "\t" in told:
                told, probs_json = told.split("\t", 1)
                told_probs = json.loads(probs_json)
            if snap.get("op") == "reset":
                if teacher and agree[1]:
                    print(f"SHADOW agreement {agree[0]}/{agree[1]} = {agree[0] / agree[1]:.3f}", file=sys.stderr, flush=True)
                game = Game(model)
                if os.environ.get("RECURRENT_DUMP"):
                    with open(os.environ["RECURRENT_DUMP"], "a") as f:
                        f.write(json.dumps({"reset": True}) + "\n")
                print(told or "ok", flush=True)
                continue
            legal = snap["legal"]
            prep = prepare(game, snap, encoder, tables)
            if prep is None:
                if os.environ.get("RECURRENT_DUMP") and os.environ.get("RECURRENT_DUMP_SNAP"):  # (the whole snapshot stream)
                    with open(os.environ["RECURRENT_DUMP"], "a") as f:
                        f.write(json.dumps({"wait": True, "snap": snap}) + "\n")
                print(told or "no move", flush=True)
                continue
            enc, state, party, opp_active, ids, floats = prep.enc, prep.state, prep.party, prep.opp_active, prep.ids, prep.floats
            slots = legal_slots(prep, legal)
            # the policy's log-probabilities over the legal slots (one model: its logits up to a constant, so the sampling
            # below is unchanged; a blend: the blended distribution)
            logits, game.h = model.decide(ids, floats, game.h, sorted(set(slots.values())))
            scored = [(float(logits[0, s]), option) for option, s in slots.items()]
            reply = max(scored)[1] if scored else legal[0]
            if temperature > 0 and len(scored) > 1:  # (repo copy) sample from softmax(logits / T) over the legal options
                weights = torch.softmax(torch.tensor([v for v, _ in scored]) / temperature, 0).tolist()
                reply = rng.choices([o for _, o in scored], weights=weights)[0]
            student_probs = dict(zip([o for _, o in scored], torch.softmax(torch.tensor([s for s, _ in scored]), 0).tolist())) if scored else {}
            if os.environ.get("RECURRENT_TRACE"):
                p = torch.softmax(torch.tensor([s for s, _ in scored]), 0).tolist() if scored else []
                ranked = sorted(zip(p, [o for _, o in scored]), reverse=True)[:4]
                print(f"TRACE d{game.decisions} {state.player_active_pokemon.name} {state.player_active_pokemon.hp_pct:.2f} vs {opp_active.name} {opp_active.hp_pct:.2f} "
                      f"last own={game.last.own_kind}:{game.last.own_name}:{game.last.own_cant} opp={game.last.opp_kind}:{game.last.opp_name}:{game.last.opp_cant} "
                      f"unmapped={[o for o in legal if o not in slots]} -> {reply} | " + " ".join(f"{o}={q:.2f}" for q, o in ranked), file=sys.stderr, flush=True)
            if os.environ.get("RECURRENT_DUMP"):  # features, the option -> slot map and (shadow mode) the teacher's reply
                with open(os.environ["RECURRENT_DUMP"], "a") as f:
                    f.write(json.dumps({**{f"ids_{k}": v.tolist() for k, v in enc.ids.items()}, **{f"floats_{k}": v.tolist() for k, v in enc.floats.items()},
                                        "legal": legal, "slots": slots, "told": told, "told_probs": told_probs, "reply": reply, "student_probs": student_probs,
                                        **({"snap": snap} if os.environ.get("RECURRENT_DUMP_SNAP") else {})}) + "\n")
            if teacher:
                if len(slots) > 1 and told in legal:
                    agree[0] += reply == told
                    agree[1] += 1
                reply = told if told in legal else reply
            finish(game, snap, prep, reply)
            print(reply, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"!error {type(exc).__name__}: {exc}", flush=True)
            print(f"recurrent server error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
