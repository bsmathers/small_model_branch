#!/usr/bin/env python3
"""The epoch-20 actor (bot2's compact Tauros multi-head, tauros_v1_v2_finetune_epoch020.pt) as a duel player that samples.

bot2's compact_tauros_policy_server.py plays the argmax after a hand-written veto of "dominated" moves. This wrapper runs the
same Engine (32-step context, full inference, its own snapshot reader and memories, untouched) and replaces only the final
choice: with POLICY_TEMPERATURE = T > 0 it samples from softmax(logits / T) over the legal options (T = 1: the actor's own
probabilities), seeded by POLICY_SEED; with T = 0 it plays the legal argmax. No veto filter: the pure actor.

Input fixes (EPOCH20_FIXES, default "wait,prevmove,effects,hp"; "none" turns them off). bot2's snapshot reader was written
for this harness's snapshots, but epoch 20 was trained on metamon's parsed replay states, and the reader differs from them in
four ways (measured on its input tokens: training validation frames against 399 harness games):
  wait      a snapshot where it only waits for the opponent's replacement is not a state in parsed replays: no model step
            (as TaurosV0's and the student's servers)
  prevmove  actor_previous_move / opponent_previous_move are each Pokemon's own last executed move, kept across switches (the
            parser's last_used_move); the snapshot's last_move is the engine's, cleared at every switch-in and faint: "nomove"
            was 49% of the harness steps against 14% in training
  effects   the effects come from the public volatile lists of both sides (the opponent's "active" entry has no volatiles, so
            its Reflect, Substitute, confusion... never reached the model), restricted to the effects of the training
            vocabulary (no recharge, Thrash lock, Rage, Mist or trap: the trap was also put on the trapper, as an unknown
            token), Transform named "typechange" as in training
  hp        own HP as the replays show it (Showdown's HP percentage), not exact

Protocol: as the other duel players ({"op": "reset"} -> "ok"; a snapshot with "legal" -> one of the legal names).
Environment: BOT2_ROOT (default /Users/macuser/bot2), EPOCH20_CHECKPOINT (default the epoch-20 file in bot2's replay_bc
experiments), POLICY_TEMPERATURE, POLICY_SEED.
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch

BOT2 = os.environ.get("BOT2_ROOT", "/Users/macuser/bot2")
sys.path.insert(0, os.path.join(BOT2, "selfplay", "tools"))
import compact_tauros_policy_server as srv  # noqa: E402
from tauros_fqe_inference import actor_state_from_snapshot  # noqa: E402

CHECKPOINT = os.environ.get("EPOCH20_CHECKPOINT", os.path.join(
    BOT2, "replay_bc/experiments/tauros_multihead_epoch89_tauros_v1_v2_lr1e5_no_l2sp_20260909/saved_models/tauros_v1_v2_finetune_epoch020.pt"))


# effects of the training vocabulary (pokemon.effect=...), by the public volatile lists' names
TRAINED_EFFECTS = {"bide", "confusion", "focusenergy", "leechseed", "lightscreen", "reflect", "substitute", "typechange"}


def showdown_pct(hp: int, top: int) -> int:
    """Pokemon Showdown's HP Percentage Mod: rounded up, 99 for anything below full, 0 only at 0 HP."""
    if hp <= 0:
        return 0
    pct = -(-100 * hp // max(top, 1))
    return 99 if pct >= 100 and hp < top else min(pct, 100)


class Memory:
    """What the input fixes keep across a game's snapshots."""

    def __init__(self):
        self.last_used = {"user": {}, "opponent": {}}  # side -> species -> its last executed move
        self.acting = {"user": None, "opponent": None}  # each side's active at the previous snapshot (the one that acted since)

    def note(self, snapshot: dict) -> None:
        # every snapshot (waiting ones too) carries the last update's actions: note who ran which move
        for side, act in (snapshot.get("last_actions") or {}).items():
            executed = str((act or {}).get("executed", "none")).lower()
            if side in self.acting and self.acting[side] and executed not in ("", "none"):
                self.last_used[side][self.acting[side]] = executed
        self.acting = {"user": str(snapshot["user"]["pokemon"][0]["id"]).lower(), "opponent": str(snapshot["opponent"]["active_species"]).lower()}


def fixed_snapshot(snapshot: dict, memory: Memory, fixes: set) -> dict:
    """The snapshot as the fixes make the reader see it (a copy: the original is not changed)."""
    snap = dict(snapshot)
    user = snap["user"] = dict(snapshot["user"])
    opponent = snap["opponent"] = dict(snapshot["opponent"])
    if "prevmove" in fixes:
        user["last_move"] = memory.last_used["user"].get(memory.acting["user"], "None")
        opponent["last_move"] = memory.last_used["opponent"].get(memory.acting["opponent"], "None")
    if "effects" in fixes:
        def trained(side):
            names = ["typechange" if v == "transform" else v for v in (side.get("public_volatiles") or [])]
            return [v for v in names if v in TRAINED_EFFECTS]
        user["volatiles"] = trained(snapshot["user"])
        user["conditions"] = {}  # (the screens are in the public list: not counted twice)
        opponent["active"] = dict(snapshot["opponent"].get("active") or {}, volatiles=trained(snapshot["opponent"]))
    if "hp" in fixes:
        user["pokemon"] = [dict(m, hp=showdown_pct(int(m.get("hp", 0)), int(m.get("maxhp", 1))), maxhp=100) for m in snapshot["user"]["pokemon"]]
    return snap


def choose(engine, snapshot: dict, temperature: float, rng: random.Random, veto_snapshot: dict | None = None) -> str:
    """`veto_snapshot`: apply bot2's hand-written veto of dominated moves (as its own server does by default), read from it."""
    mapping = srv.legal_actions(snapshot)  # action index -> option name
    state = actor_state_from_snapshot(snapshot, allow_forced_switch=True, effect_memory=engine.effect_memory)
    groups = engine.memory.encode(state)
    engine.steps.append([[engine.vocab.get(token, 1) for token in group] for group in groups])
    engine.total_steps += 1
    with torch.inference_mode():
        logits = engine._actor_logits(engine._representation(), snapshot=snapshot).float()
    indices = sorted(mapping)
    if veto_snapshot is not None:
        blocked = srv.dominated_actor_indices(veto_snapshot)
        indices = [i for i in indices if i not in blocked] or sorted(mapping)
    scores = [float(logits[i]) for i in indices]
    if temperature <= 0 or len(indices) == 1:
        return mapping[max(indices, key=lambda i: (float(logits[i]), -i))]  # (bot2's rule: highest logit, ties to the lower index)
    weights = torch.softmax(torch.tensor(scores) / temperature, 0).tolist()
    return mapping[rng.choices(indices, weights=weights)[0]]


def main() -> None:
    torch.set_num_threads(1)
    # EPOCH20_TEMPERATURE overrides POLICY_TEMPERATURE (which the duel's other player reads too); EPOCH20_FILTER=1: bot2's veto
    temperature = float(os.environ.get("EPOCH20_TEMPERATURE", os.environ.get("POLICY_TEMPERATURE", "0")))
    veto = os.environ.get("EPOCH20_FILTER", "0") == "1"
    rng = random.Random(f"epoch20-{os.environ.get('POLICY_SEED', '0')}")
    engine = srv.Engine(CHECKPOINT, "cpu", 32, "full")
    fixes = set(os.environ.get("EPOCH20_FIXES", "wait,prevmove,effects,hp").split(",")) - {"none", ""}
    print(f"epoch-20 actor: {CHECKPOINT}, temperature {temperature:g}, {"bot2's veto filter" if veto else "no veto filter"}, input fixes {sorted(fixes)}", file=sys.stderr, flush=True)
    last_turn = None
    memory = Memory()
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            if request.get("op") == "reset":
                engine.reset()
                last_turn = None
                memory = Memory()
                print("ok", flush=True)
                continue
            turn = int(request.get("turn", 0))
            if last_turn is not None and turn < last_turn:  # (as bot2's server: a new game without a reset)
                engine.reset()
                memory = Memory()
            last_turn = turn
            memory.note(request)
            if "wait" in fixes and request.get("legal") == ["no move"] and request.get("opponent_force_switch") and not request.get("force_switch"):
                print("no move", flush=True)
                continue
            print(choose(engine, fixed_snapshot(request, memory, fixes), temperature, rng, request if veto else None), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"!error {type(exc).__name__}: {exc}", flush=True)
            print(f"epoch-20 server error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
