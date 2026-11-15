"""Checks a sample of built training games against their labels and trajectories, by option NAME, without the build's code.

The build moves Exeggcute's targets from Exeggcute's action indices (its own state: moves sorted by name, then the alive
reserves sorted by species) onto this model's policy slots. A wrong mapping is invisible to every other check: the targets are
still valid distributions over legal slots, training converges, and the model learns the wrong move for every feature. (This is
the bug that sank an earlier run.) So this re-derives what each slot means from the two ends only:

    Exeggcute's side  the names of its actions, from the trajectory's original-parser state (legacy_states), sorted as Metamon sorts
                      them (lowercase, letters and digits only)
    the model's side  the names the model SEES in each slot, decoded from the built game's own feature ids (the active's move ids
                      for slots 0-3, the species ids of the living reserves for slots 4-8) through the tables

and requires, on every normal decision with a policy target, that the built target restricted to the options both sides name is
the label's target over the same names (up to the renormalisation the build applies, and float16 storage). It also checks that
the action the built game says was played is the one the trajectory names.

    python -m metamon.recurrent.verify_built TRAJ_DIR LABEL_DIR BUILT_DIR TABLES_DIR [--games 200] [--seed 0]

Exits with an error when more than --max-bad of the checked decisions disagree. Expect 0 bad decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import numpy as np

N_MOVES, N_SLOTS = 4, 9
NORMAL = 0  # build.NORMAL (decision kinds other than normal reshape the target on purpose: see build.adjust)
FAINTED = 7  # features.STATUS_IDS["fnt"]
TOLERANCE = 0.02  # float16 storage and renormalisation of the built target


def norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def reverse_table(path: str) -> dict[int, str]:
    return {int(v): k for k, v in json.load(open(path)).items()}


def exeggcute_names(legacy_state: dict) -> list:
    """The names of Exeggcute's actions 0-8 in this state (None where there is no action)."""
    moves = sorted(norm(m["name"]) for m in legacy_state["player_active_pokemon"]["moves"])[:N_MOVES]
    switches = sorted(norm(p["name"]) for p in legacy_state["available_switches"])[: N_SLOTS - N_MOVES]
    return (moves + [None] * (N_MOVES - len(moves)) + switches + [None] * (N_SLOTS - N_MOVES - len(switches)))[:N_SLOTS]


def model_names(built, t: int, moves: dict[int, str], species: dict[int, str]) -> list:
    """The names of the model's policy slots at decision t, from the features the model is given."""
    own_moves, own_species = built["ids_own_moves"][t], built["ids_own_species"][t]
    status, hp = built["ids_own_status"][t], built["floats_own"][t][:, 0]
    names = [moves.get(int(i)) if int(i) else None for i in own_moves[0][:N_MOVES]]
    living = [species.get(int(own_species[s])) for s in range(1, len(own_species)) if own_species[s] and hp[s] > 0 and status[s] != FAINTED]
    return names + (living + [None] * (N_SLOTS - N_MOVES))[: N_SLOTS - N_MOVES]


def check_game(traj_path: str, label_path: str, built_path: str, moves, species) -> dict:
    import lz4.frame

    with lz4.frame.open(traj_path, "rb") as f:
        traj = json.loads(f.read())
    labels, built = np.load(label_path), np.load(built_path)
    n = len(built["kind"])
    out = {"decisions": 0, "bad_target": 0, "bad_legal": 0, "played_checked": 0, "bad_played": 0, "examples": []}
    if len(labels["chosen"]) != n:
        out["examples"].append(f"{os.path.basename(built_path)}: {n} built decisions, {len(labels['chosen'])} labels")
        out["bad_target"] += 1
        return out
    for t in range(n):
        ours = model_names(built, t, moves, species)
        # the played action against the trajectory's own record of it
        played = int(built["played"][t])
        nxt = traj["extras"][t].get("next_player_action") or {}
        if 0 <= played < N_SLOTS and nxt.get("kind") in ("move", "switch"):
            out["played_checked"] += 1
            if ours[played] != norm(nxt.get("name", "")):
                out["bad_played"] += 1
                if len(out["examples"]) < 5:
                    out["examples"].append(f"{os.path.basename(built_path)} t={t}: played slot {played} is {ours[played]}, the log says {nxt}")
        if int(built["kind"][t]) != NORMAL or not bool(built["policy_ok"][t]):
            continue
        theirs = exeggcute_names(traj["legacy_states"][t])
        legal_ours = {ours[s] for s in range(N_SLOTS) if built["legal"][t, s] and ours[s] is not None}
        legal_theirs = {theirs[a] for a in range(N_SLOTS) if labels["legal"][t, a] > 0 and theirs[a] is not None}
        out["decisions"] += 1
        # the build only removes options (no slot in this model, out of PP, disabled): every option it keeps is one Exeggcute had
        if not legal_ours <= legal_theirs or any(built["legal"][t, s] and ours[s] is None for s in range(N_SLOTS)):
            out["bad_legal"] += 1
            if len(out["examples"]) < 5:
                out["examples"].append(f"{os.path.basename(built_path)} t={t}: built legal {sorted(legal_ours)} not within the label's {sorted(legal_theirs)}")
            continue
        for key in ("blend", "judge"):
            label_row, built_row = labels[key][t].astype(np.float64), built[key][t].astype(np.float64)
            q = {theirs[a]: label_row[a] for a in range(N_SLOTS) if theirs[a] in legal_ours}
            p = {ours[s]: built_row[s] for s in range(N_SLOTS) if ours[s] in legal_ours and built["legal"][t, s]}
            qs, ps = sum(q.values()), sum(p.values())
            if qs <= 0 and ps <= 0:
                continue  # (no judge at this decision)
            worst = max(abs(q.get(k, 0.0) / max(qs, 1e-12) - p.get(k, 0.0) / max(ps, 1e-12)) for k in legal_ours)
            if worst > TOLERANCE:
                out["bad_target"] += 1
                if len(out["examples"]) < 5:
                    out["examples"].append(
                        f"{os.path.basename(built_path)} t={t} {key}: label {({k: round(float(v / qs), 3) for k, v in q.items()} if qs else q)} "
                        f"vs built {({k: round(float(v / ps), 3) for k, v in p.items()} if ps else p)}"
                    )
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectories")
    ap.add_argument("labels")
    ap.add_argument("built")
    ap.add_argument("tables")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bad", type=float, default=0.001, help="largest allowed share of checked decisions that disagree (each check)")
    args = ap.parse_args()
    moves = reverse_table(os.path.join(args.tables, "move_ids.json"))
    species = reverse_table(os.path.join(args.tables, "species_ids.json"))
    trajectories = {}
    for root, _, names in os.walk(args.trajectories):
        for name in names:
            if name.endswith(".json.lz4"):
                trajectories.setdefault(name[: -len(".json.lz4")], os.path.join(root, name))
    stems = sorted(n[: -len(".npz")] for n in os.listdir(args.built) if n.endswith(".npz"))
    stems = [s for s in stems if s in trajectories and os.path.exists(os.path.join(args.labels, s + ".npz"))]
    if not stems:
        raise SystemExit(f"no built games in {args.built} with both a trajectory and a label")
    picked = random.Random(args.seed).sample(stems, min(args.games, len(stems)))
    totals = {"decisions": 0, "bad_target": 0, "bad_legal": 0, "played_checked": 0, "bad_played": 0}
    examples = []
    for stem in picked:
        r = check_game(trajectories[stem], os.path.join(args.labels, stem + ".npz"), os.path.join(args.built, stem + ".npz"), moves, species)
        for k in totals:
            totals[k] += r[k]
        examples += r["examples"][: max(0, 10 - len(examples))]
    d, pc = max(totals["decisions"], 1), max(totals["played_checked"], 1)
    print(f"{len(picked)} built games: {totals['decisions']} normal decisions with a target, {totals['played_checked']} played actions")
    print(f"targets that differ from the label by name: {totals['bad_target']} ({totals['bad_target'] / d:.3%}); "
          f"legal options Exeggcute did not have: {totals['bad_legal']} ({totals['bad_legal'] / d:.3%}); "
          f"played action differs from the log: {totals['bad_played']} ({totals['bad_played'] / pc:.3%})")
    for e in examples:
        print("  " + e)
    bad = (totals["decisions"] == 0 or totals["bad_target"] > args.max_bad * d or totals["bad_legal"] > args.max_bad * d
           or totals["bad_played"] > args.max_bad * pc)
    if bad:
        raise SystemExit("the built games do not match their labels by name: do not train on them")
    print("built games match their labels and trajectories by name")


if __name__ == "__main__":
    sys.exit(main())
