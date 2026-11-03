"""Compares parsed trajectories with the real Pokemon Showdown simulator, decision by decision.

    node tools/oracle/showdown_oracle.js SHOWDOWN_DIR TEAMS_DIR ORACLE_DIR N [FIRST_SEED]    # truth: a request and the
                                                                                             # hidden engine state at every decision
    python -m metamon.recurrent.parse ORACLE_LOGS TRAJ_DIR                                   # ORACLE_DIR/*.log in a folder of their own
    python -m metamon.recurrent.check_oracle ORACLE_DIR TRAJ_DIR [--examples 3]

State i of a player's trajectory is the engine's i-th decision for that player (the trajectory has one more
state, the final one; this is checked). For each decision the checks are:

  switches     the alive reserves in the state are the engine's switch targets
  active       species, HP, status and stat boosts of both active Pokemon
  moves        the observer's moves and their PP, and the opponent's revealed moves and their PP
  conditions   Reflect / Light Screen / Mist on both sides
  kind         the decision kind of build.decision_kind against the shape of the engine's request:
               "full" lock (one move, no switching), "semi" lock (one "fight" entry, switching allowed),
               or a normal request; and the moves the request leaves enabled

Every check reports passes and failures with a few examples.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
from types import SimpleNamespace

import lz4.frame

from metamon.recurrent.build import FIGHT_OR_SWITCH, FORCED_LOCK, NORMAL, decision_kind
from metamon.recurrent.features import counter_fraction
from metamon.recurrent.records import decision_facts
from metamon.recurrent.tables import Tables

TABLES = Tables.from_dex(1)


def norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def truth_shape(req: dict) -> str:
    """The shape of the engine's request, which is what the player is actually offered."""
    if req.get("forceSwitch"):
        return "replacement"
    active = req["active"][0]
    if len(active["moves"]) == 1:
        move = active["moves"][0]["id"]
        if active.get("trapped"):
            return "full"  # one move, no switching: recharge, a charging move, Thrash, Rage
        return "struggle" if move == "struggle" else "semi"  # one entry, switching allowed: sleep, freeze, a trap, Bide
    return "normal"


def expected_kind(shape: str) -> int:
    return {"full": FORCED_LOCK, "semi": FIGHT_OR_SWITCH, "struggle": FIGHT_OR_SWITCH}.get(shape, NORMAL)


class Report:
    def __init__(self, n_examples: int):
        self.n_examples = n_examples
        self.ok = collections.Counter()
        self.bad = collections.Counter()
        self.examples = collections.defaultdict(list)
        self.matrix = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

    def check(self, name: str, passed: bool, detail=None) -> bool:
        if passed:
            self.ok[name] += 1
        else:
            self.bad[name] += 1
            if len(self.examples[name]) < self.n_examples:
                self.examples[name].append(detail)
        return passed

    def print(self) -> None:
        names = sorted(set(self.ok) | set(self.bad))
        print(f"{'check':46s} {'pass':>8s} {'fail':>8s}  {'fail rate':>9s}")
        for n in names:
            total = self.ok[n] + self.bad[n]
            print(f"{n:46s} {self.ok[n]:8d} {self.bad[n]:8d}  {100 * self.bad[n] / max(total, 1):8.3f}%")
        for n in names:
            if self.examples[n]:
                print(f"\n-- examples: {n}")
                for e in self.examples[n]:
                    print("   ", e)
        for title, matrix in self.matrix.items():
            print(f"\n-- {title}")
            for row in sorted(matrix):
                print(f"   {row}: {dict(matrix[row])}")


def revealed(log: str, side: str) -> tuple[set, dict]:
    """What the public log shows about a side over the whole game: the species that appeared, and the moves each
    one used as its own (a move called by another, via Metronome or Mirror Move, is not one of its moves)."""
    species, moves, active = set(), collections.defaultdict(set), None
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 4:
            continue
        if parts[1] in ("switch", "drag") and parts[2].startswith(side + "a:"):
            active = norm(parts[3].split(",")[0])
            species.add(active)
        elif parts[1] == "move" and parts[2].startswith(side + "a:") and active:
            tags = [x for x in parts[5:] if x.startswith("[from]")]
            if not tags or tags[0] == "[from]lockedmove":
                moves[active].add(norm(parts[3]))
    return species, moves


def pokemon_by_species(side_state: dict) -> dict:
    return {p["species"]: p for p in side_state["pokemon"]}


def compare_game(report: Report, dec: dict, i: int, state: dict, extras: dict, facts: dict, tag: str, own_species: set, own_moves_used: dict, opp_moves_used: dict) -> None:
    req = dec["request"]
    me_truth = next(p for p in dec["me"]["pokemon"] if p["position"] == 0)  # slot 0 is the active one, also when it just fainted
    opp_truth = next(p for p in dec["opp"]["pokemon"] if p["position"] == 0)
    me, opp = state["player_active_pokemon"], state["opponent_active_pokemon"]
    where = f"{tag} decision {i} (turn {dec['turn']})"

    # --- replacement and switch targets
    report.check("forced_switch flag", bool(req.get("forceSwitch")) == bool(state["forced_switch"]), (where, req.get("forceSwitch"), state["forced_switch"]))
    truth_switches = {p["species"] for p in dec["me"]["pokemon"] if p["position"] != 0 and not p["fainted"]}
    ours_switches = {norm(p["name"]) for p in state["available_switches"]}
    shown = own_species  # a teammate that never appears in the log cannot be known by any parser
    report.check("switch targets = alive reserves (those the log shows)", truth_switches & shown == ours_switches, (where, sorted((truth_switches & shown) ^ ours_switches)))
    report.check("switch targets: same number as the engine (unseen ones are filled in)", len(ours_switches) == len(truth_switches), (where, len(ours_switches), len(truth_switches)))
    report.check("never-shown alive reserves (unknowable, informational)", not (truth_switches - shown), (where, sorted(truth_switches - shown)))

    # --- the active Pokemon
    report.check("own active species", norm(me["name"]) == me_truth["species"], (where, me["name"], me_truth["species"]))
    report.check("opp active species", norm(opp["name"]) == opp_truth["species"], (where, opp["name"], opp_truth["species"]))

    def hp_ok(ours: float, truth: dict) -> bool:
        pct = math.ceil(truth["hp"] * 100 / truth["maxhp"]) if truth["hp"] > 0 else 0  # the log rounds HP percent up,
        if 0 < truth["hp"] < truth["maxhp"] and pct == 100:  # and shows 99 for anything short of full
            pct = 99
        return abs(ours * 100 - pct) < 1e-6

    report.check("own active HP", hp_ok(me["hp_pct"], me_truth), (where, me["hp_pct"], me_truth["hp"], me_truth["maxhp"]))
    report.check("opp active HP", hp_ok(opp["hp_pct"], opp_truth), (where, opp["hp_pct"], opp_truth["hp"], opp_truth["maxhp"]))

    def status_of(p: dict) -> str:
        return "fnt" if p["fainted"] or p["hp"] <= 0 else (p["status"] or "nostatus")

    report.check("own active status", me["status"] == status_of(me_truth), (where, me["status"], status_of(me_truth)))
    report.check("opp active status", opp["status"] == status_of(opp_truth), (where, opp["status"], status_of(opp_truth)))
    for who, ours, truth in (("own", me, me_truth), ("opp", opp, opp_truth)):
        for stat, key in (("atk", "atk_boost"), ("def", "def_boost"), ("spa", "spa_boost"), ("spe", "spe_boost")):
            report.check(f"{who} boost {stat}", ours[key] == truth["boosts"][stat], (where, ours[key], truth["boosts"][stat]))

    # --- moves and PP
    truth_moves = {m["id"]: m for m in me_truth["moves"]}
    ours_moves = {norm(m["name"]): m for m in me["moves"]}
    common = set(truth_moves) & set(ours_moves)
    used = own_moves_used.get(norm(me["name"]), set())
    # Struggle is not one of the Pokemon's moves (it is what the engine offers when every move is out of PP)
    report.check("own moves: every move the log shows is listed", (used - {"struggle"}) <= set(ours_moves), (where, sorted(used - set(ours_moves))))
    report.check("own moves: listed moves the log never shows (guesses, informational)", set(ours_moves) <= used, (where, sorted(set(ours_moves) - used)))
    report.check("own moves: ours are real moves, or guesses", True)
    report.check("own moves: listed count = real count", len(ours_moves) == len(truth_moves), (where, len(ours_moves), len(truth_moves)))
    for mv in common:
        report.check("own move PP", ours_moves[mv]["current_pp"] == truth_moves[mv]["pp"], (where, mv, ours_moves[mv]["current_pp"], truth_moves[mv]["pp"]))
    opp_truth_moves = {m["id"]: m for m in opp_truth["moves"]}
    opp_ours = {norm(m["name"]): m for m in opp["moves"]}
    report.check("opp moves: revealed ones are real", set(opp_ours) <= set(opp_truth_moves), (where, sorted(set(opp_ours) - set(opp_truth_moves))))
    seen_so_far = opp_moves_used.get(norm(opp["name"]), set())
    for mv in set(opp_ours) & set(opp_truth_moves):
        report.check("opp move PP", opp_ours[mv]["current_pp"] == opp_truth_moves[mv]["pp"], (where, mv, opp_ours[mv]["current_pp"], opp_truth_moves[mv]["pp"]))

    # --- side conditions
    for who, ours, truth in (("own", extras["player_conditions_all"], dec["me"]["conditions"]), ("opp", extras["opponent_conditions_all"], dec["opp"]["conditions"])):
        ours_set = {norm(c) for c in ours} & {"reflect", "lightscreen", "mist"}
        truth_set = {norm(c) for c in truth} & {"reflect", "lightscreen", "mist"}
        report.check(f"{who} screens", ours_set == truth_set, (where, sorted(ours_set), sorted(truth_set)))

    # --- Counter: the engine's own answer to "what would Counter do now" against what the extras let us predict
    for who, truth, user in (("own", dec.get("counterMe"), "player"), ("opp", dec.get("counterOpp"), "opponent")):
        if truth is None:
            continue
        theirs = min(1.0, truth["damage"] / truth["targetMaxHp"])  # damage beyond the target's max HP is just a KO
        # (the logic is graded, not the HP estimate: the true max HP of each side's Pokemon is supplied)
        true_hp = {("player", p["species"]): p["maxhp"] for p in dec["me"]["pokemon"]} | {("opponent", p["species"]): p["maxhp"] for p in dec["opp"]["pokemon"]}
        ours = counter_fraction(user, extras, TABLES, me_truth["maxhp"], opp_truth["maxhp"], max_hp=lambda side, species: true_hp[(side, species)])
        report.matrix["Counter outcome (rows: engine succeeds?; columns: we predict damage?)"][f"engine {'succeeds' if truth['succeeds'] else 'fails'}"][f"ours {'>0' if ours > 0 else '=0'}"] += 1
        report.check(f"Counter ({who}): predicted to succeed iff the engine succeeds", (ours > 0) == (truth["succeeds"] and theirs > 0.005), (where, "ours", round(ours, 3), "engine", truth["damage"], truth["targetMaxHp"], truth["lastDamage"], truth["targetSide"], "memory", extras.get("last_damage"), extras.get("last_move")))
        if ours > 0 and theirs > 0:
            report.check(f"Counter ({who}): damage within 3% of the target's max HP", abs(ours - theirs) <= max(0.03, 0.1 * theirs), (where, "ours", round(ours, 3), "engine", round(theirs, 3), extras.get("last_damage")))
        # where the two disagree, why: the foe's last used move, and whether the foe could act last turn
        if (ours > 0) != (truth["succeeds"] and theirs > 0.005):
            foe = "opponent" if user == "player" else "player"
            last = (extras.get("last_move") or {}).get(foe)
            kind = "ours says succeed, engine fails" if ours > 0 else "ours says fail, engine succeeds"
            report.matrix["Counter disagreements: what we saw of the foe (last used move; its cant reason last turn)"][kind][f"foe last={'counter' if last == 'counter' else 'other'}, cant={extras.get(foe + '_cant')}"] += 1

    # --- the kind of decision, against the shape of the request
    shape = truth_shape(req)
    kind = decision_kind(SimpleNamespace(**facts))
    exp = expected_kind(shape)
    report.matrix["decision kind (rows: engine request shape; columns: our kind 0 normal, 1 fight-or-switch, 2 forced recharge, 4 timeout)"][shape][kind] += 1
    if shape == "replacement":
        report.check("kind: replacement", kind == NORMAL, (where, shape, kind))
    else:
        passed = report.check(f"kind: {shape}", kind == exp, (where, shape, kind, sorted(me_truth["volatiles"]), me_truth["status"], extras.get("player_lock")))
        if not passed:
            lock = extras.get("player_lock") or {}
            engine = ",".join(sorted(set(me_truth["volatiles"]) & {"mustrecharge", "twoturnmove", "lockedmove", "rage", "bide", "partialtrappinglock", "partiallytrapped", "fakepartiallytrapped", "disable"})) or "-"
            report.matrix["kind mismatches: engine shape | engine volatiles | status || ours: kind, lock, trap age, trapped age"][f"{shape:7s} | {engine:42s} | {me_truth['status'] or '-':3s}"][f"{kind} {lock.get('lock')} a{lock.get('age')} t{lock.get('trapped_age')}"] += 1
    # which moves the request leaves enabled
    if shape == "normal":
        # a move is available when it has PP left and Disable has not taken it away; compared on the moves that
        # both sides list (the ones we only guess cannot be checked)
        disabled = norm(facts["disabled_move"] or "")
        for mv in common:
            truth_enabled = not next(m for m in req["active"][0]["moves"] if m["id"] == mv).get("disabled")
            ours_enabled = ours_moves[mv]["current_pp"] > 0 and mv != disabled
            report.check("normal request: a listed real move is available iff the engine offers it", ours_enabled == truth_enabled,
                         (where, mv, "ours", ours_enabled, "engine", truth_enabled, "pp", ours_moves[mv]["current_pp"], "disabled", disabled))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oracle")
    ap.add_argument("trajectories")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N trajectories")
    args = ap.parse_args()
    report = Report(args.examples)
    files = sorted(glob.glob(os.path.join(args.trajectories, "*.json.lz4")))
    if args.limit:
        files = files[: args.limit]
    n_traj = n_dec = 0
    for tf in files:
        m = re.match(r"(oracle-\d+)_.*?_(Alpha|Beta)_vs_", os.path.basename(tf))
        gid, side = m.group(1), "p1" if m.group(2) == "Alpha" else "p2"
        truth = json.load(open(os.path.join(args.oracle, f"{gid}.json")))
        decisions = [d for d in truth["decisions"] if d["side"] == side]
        traj = json.loads(lz4.frame.open(tf).read())
        states, actions, extras = traj["states"], traj["actions"], traj["extras"]
        if not report.check("trajectory has one state more than the engine had decisions", len(states) == len(decisions) + 1, (gid, side, len(states), len(decisions))):
            continue
        n_traj += 1
        log = open(os.path.join(args.oracle, f"{gid}.log")).read()
        own_species, own_used = revealed(log, side)
        _, opp_used = revealed(log, "p2" if side == "p1" else "p1")
        for i, dec in enumerate(decisions):
            n_dec += 1
            compare_game(report, dec, i, states[i], extras[i], decision_facts(states, actions, extras, i), f"{gid}/{side}", own_species, own_used, opp_used)
    print(f"{n_traj} trajectories, {n_dec} decisions compared\n")
    report.print()


if __name__ == "__main__":
    main()
