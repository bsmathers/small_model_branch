#!/usr/bin/env python3
"""Raw greedy Tauros using the same inference path as the search evaluator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tauros_inference import TaurosInference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metamon-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=62)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention", choices=("auto", "vanilla", "sdpa"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    # (repo copy) POLICY_TEMPERATURE: 0 (default) plays the argmax, T > 0 samples (T = 1: TaurosV0's own action distribution,
    # restricted to the legal options); the draws are seeded by POLICY_SEED
    import os
    import random

    temperature = float(os.environ.get("POLICY_TEMPERATURE", "0"))
    rng = random.Random(f"tauros-{os.environ.get('POLICY_SEED', '0')}")
    engine = TaurosInference(
        args.metamon_root, args.cache_dir, args.checkpoint, args.device,
        attention_backend=args.attention,
    )
    print(
        f"tauros device={engine.device} attention={engine.attention_backend}",
        file=sys.stderr,
        flush=True,
    )
    # (repo copy) two fixes that make TaurosV0's inputs match how it plays on Showdown (TAUROS_FIXES=none turns both off):
    #   wait      a snapshot where we only wait for the opponent's replacement ("no move", the opponent must switch, we need
    #             not) is not a decision on Showdown (the client gets a wait request): no model step, as the student's server
    #   prevmove  player_prev_move / opponent_prev_move as metamon's parser keeps them (on Showdown TaurosV0's states come from
    #             it): each Pokemon's own last executed move, kept across switches. The engine's last_used_move is cleared
    #             for both sides at every switch-in and faint instead
    fixes = set(os.environ.get("TAUROS_FIXES", "wait,prevmove").split(","))
    print(f"tauros fixes: {sorted(fixes)}", file=sys.stderr, flush=True)
    history = engine.begin()
    previous_state = None
    last_turn = None
    last_used = {"user": {}, "opponent": {}}  # side -> species -> last executed move
    acting = {"user": None, "opponent": None}  # each side's active at the previous snapshot (the one that acted since)
    for raw in sys.stdin:
        try:
            snapshot = json.loads(raw)
            # Explicit reset avoids relying on a turn-number decrease at a
            # game boundary (including very short or failed games).
            if snapshot.get("op") == "reset":
                history, previous_state, last_turn = engine.begin(), None, None
                last_used, acting = {"user": {}, "opponent": {}}, {"user": None, "opponent": None}
                print("ok", flush=True)
                continue
            # every snapshot (waiting ones too) carries the last update's actions: note who ran which move
            for side, act in (snapshot.get("last_actions") or {}).items():
                executed = str((act or {}).get("executed", "none")).lower()
                if side in acting and acting[side] and executed not in ("", "none"):
                    last_used[side][acting[side]] = executed
            acting = {"user": str(snapshot["user"]["pokemon"][0]["id"]).lower(), "opponent": str(snapshot["opponent"]["active_species"]).lower()}
            if "wait" in fixes and snapshot.get("legal") == ["no move"] and snapshot.get("opponent_force_switch") and not snapshot.get("force_switch"):
                print("no move", flush=True)
                continue
            turn = int(snapshot.get("turn", 0))
            if last_turn is not None and turn < last_turn:
                history = engine.begin()
                previous_state = None
            last_turn = turn
            state = engine.encoder.state(snapshot, history.space)
            if "prevmove" in fixes:
                state.player_prev_move = engine.encoder._move({"id": last_used["user"].get(acting["user"], "None")})
                state.opponent_prev_move = engine.encoder._move({"id": last_used["opponent"].get(acting["opponent"], "None")})
            choices = engine.legal_actions(snapshot, state)
            legal = list(choices)
            reward = (
                0.0
                if previous_state is None
                else float(engine.reward_function(previous_state, state))
            )
            probabilities = engine.probabilities(
                snapshot, legal, history, reward=reward, state=state
            )
            # Tie order is the canonical Tauros action order, not engine order.
            selected = max(legal, key=lambda i: (probabilities[legal.index(i)], -i))
            if temperature > 0 and len(legal) > 1:  # (repo copy) sample from the policy's probabilities ** (1 / T)
                weights = [max(float(q), 0.0) ** (1.0 / temperature) for q in probabilities]
                if sum(weights) > 0:
                    selected = rng.choices(legal, weights=weights)[0]
            history.commit(selected)
            previous_state = state
            print(choices[selected], flush=True)
        except Exception as exc:
            print(f"!error {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
