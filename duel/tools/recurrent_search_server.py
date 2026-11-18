#!/usr/bin/env python3
"""The recurrent model (metamon.recurrent) for the duel harness's search: values and action logits of search nodes.

A node is one player's view at one decision point: the policy server's whole per-game memory (recurrent state, last-turn
record, effect, trap and Counter memories; recurrent_policy_server.Game) after reading that decision's snapshot. A child
is made from its parent by forking the parent's memory, recording what the player chose there ("sent", as the policy
server records its reply), and reading the child's snapshot. Every snapshot goes through the policy server's own code
(prepare / legal_slots / finish), so a chain of nodes along the real game is exactly what the policy server computes.

Protocol: one JSON object per line, one reply line each.
  {"op": "step", "parent": P, "sent": NAME, "snap": SNAPSHOT}
      P = -1 starts a game (sent is ignored). SNAPSHOT is the harness's duel snapshot (with "legal" and "last_actions").
      Reply: "<id> <win logit> <logit of legal[0]> ... <logit of legal[n-1]>", the logits of options without a policy
      slot as -1e30. A wait state (only the opponent's forced switch) is not a decision: the node keeps its parent's
      recurrent state and value, and its only option gets logit 0.
  {"op": "chain", "parent": P, "steps": [{"sent": NAME, "snap": SNAPSHOT}, ...]}
      Several steps in one request (the first from P); only the last node is kept. Same reply, for the last node.
  {"op": "batch", "items": [{"parent": P, "sent": NAME, "snap": SNAPSHOT}, ...]}
      Independent steps (each parent an existing node), their model steps run as one batch. Reply: the items' replies (as
      for "step") joined by " ; ".
  {"op": "mark"}               -> the next node id
  {"op": "release", "mark": M} -> "ok"; forgets every node with id >= M (the search's nodes, after a decision)
  {"op": "reset"}              -> "ok"; forgets every node
Errors are answered "!error <message>".

Weights: RECURRENT_SEARCH_WEIGHTS, else RECURRENT_WEIGHTS (default ~/model_weights/weights_best.pt).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

from recurrent_policy_server import Game, finish, legal_slots, make_encoder, prepare  # (also puts the repo's metamon on sys.path)
from metamon.recurrent.blend import PolicyBlend  # noqa: E402

UNMAPPED = -1e30


class Node:
    __slots__ = ("game", "snap", "prep", "win")

    def __init__(self, game, snap, prep, win):
        self.game, self.snap, self.prep, self.win = game, snap, prep, win


class Server:
    def __init__(self, weights: str):
        self.policy = PolicyBlend.load(weights)
        if self.policy.size != 1:
            raise SystemExit("the search server takes one weights file, not a blend")
        self.model = self.policy.models[0]
        self.encoder, self.tables = make_encoder()
        self.nodes: dict[int, Node] = {}
        self.next_id = 0

    def _begin(self, parent: Node | None, sent: str | None, snap: dict):
        """Everything of a step before the model: the child's game (its memories updated) and its model input (None for a wait state)."""
        if parent is None:
            game = Game(self.policy)
        else:
            game = parent.game.fork()
            if parent.prep is not None:  # (a wait state got no reply in the policy server either)
                finish(game, parent.snap, parent.prep, sent if sent is not None else snap["legal"][0])
        return game, prepare(game, snap, self.encoder, self.tables)

    def _end(self, parent: Node | None, snap: dict, game, prep, logits=None, win=None, h=None) -> tuple[Node, list[float]]:
        legal = snap["legal"]
        if prep is None:
            return Node(game, snap, None, parent.win if parent is not None else 0.0), [0.0] * len(legal)
        game.h = [h]
        slots = legal_slots(prep, legal)
        row = logits.tolist()
        return Node(game, snap, prep, float(win)), [row[slots[o]] if o in slots else UNMAPPED for o in legal]

    @torch.no_grad()
    def step(self, parent: Node | None, sent: str | None, snap: dict) -> tuple[Node, list[float]]:
        game, prep = self._begin(parent, sent, snap)
        if prep is None:
            return self._end(parent, snap, game, None)
        tokens = self.model.observation_tokens(prep.ids, prep.floats)
        logits, win, h = self.model.step(tokens[:, 0], game.h[0])
        return self._end(parent, snap, game, prep, logits[0], win[0], h)

    @torch.no_grad()
    def batch(self, items: list) -> list[tuple[Node, list[float]]]:
        """Independent steps; the model runs once over all of them that are decisions."""
        begun = []
        for item in items:
            parent = None if int(item.get("parent", -1)) < 0 else self.nodes[int(item["parent"])]
            game, prep = self._begin(parent, item.get("sent"), item["snap"])
            begun.append((parent, item["snap"], game, prep))
        live = [k for k, b in enumerate(begun) if b[3] is not None]
        out: list = [None] * len(begun)
        if live:
            ids = {k: torch.cat([begun[j][3].ids[k] for j in live]) for k in begun[live[0]][3].ids}
            floats = {k: torch.cat([begun[j][3].floats[k] for j in live]) for k in begun[live[0]][3].floats}
            h0 = torch.cat([begun[j][2].h[0] for j in live])
            tokens = self.model.observation_tokens(ids, floats)
            logits, win, h = self.model.step(tokens[:, 0], h0)
            for row, j in enumerate(live):
                parent, snap, game, prep = begun[j]
                out[j] = self._end(parent, snap, game, prep, logits[row], win[row], h[row:row + 1])
        for j, (parent, snap, game, prep) in enumerate(begun):
            if prep is None:
                out[j] = self._end(parent, snap, game, None)
        return out

    positions = 0

    def store(self, node: Node) -> int:
        self.positions += 1
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = node
        return nid

    def handle(self, req: dict) -> str:
        op = req.get("op")
        if op in ("step", "chain"):
            parent = None if int(req.get("parent", -1)) < 0 else self.nodes[int(req["parent"])]
            steps = req["steps"] if op == "chain" else [req]
            node, logits = parent, []
            for item in steps:
                node, logits = self.step(node, item.get("sent"), item["snap"])
            nid = self.store(node)
            return f"{nid} {node.win:.6g} " + " ".join(f"{v:.6g}" for v in logits)
        if op == "explain":  # (a step, plus its model input, for checking the harness's native features: "<reply>\t<json>")
            parent = None if int(req.get("parent", -1)) < 0 else self.nodes[int(req["parent"])]
            node, logits = self.step(parent, req.get("sent"), req["snap"])
            nid = self.store(node)
            enc = node.prep.enc if node.prep is not None else None
            feats = {} if enc is None else {"ids": {k: v.tolist() for k, v in enc.ids.items()}, "floats": {k: v.tolist() for k, v in enc.floats.items()}}
            return f"{nid} {node.win:.6g} " + " ".join(f"{v:.6g}" for v in logits) + "\t" + json.dumps(feats)
        if op == "batch":
            replies = []
            for node, logits in self.batch(req["items"]):
                nid = self.store(node)
                replies.append(f"{nid} {node.win:.6g} " + " ".join(f"{v:.6g}" for v in logits))
            return " ; ".join(replies)
        if op == "mark":
            return str(self.next_id)
        if op == "release":
            mark = int(req["mark"])
            for nid in [n for n in self.nodes if n >= mark]:
                del self.nodes[nid]
            return "ok"
        if op == "reset":
            self.nodes.clear()
            return "ok"
        raise ValueError(f"unknown op {op!r}")


def main() -> None:
    torch.set_num_threads(int(os.environ.get("RECURRENT_SEARCH_THREADS", "1")))
    weights = os.environ.get("RECURRENT_SEARCH_WEIGHTS") or os.environ.get("RECURRENT_WEIGHTS") or os.path.expanduser("~/model_weights/weights_best.pt")
    server = Server(weights)
    print(f"recurrent search server: {weights}", file=sys.stderr, flush=True)
    # RECURRENT_SEARCH_RECORD=PATH: every request, for profiling; the time spent answering is printed at each reset
    record = open(os.environ["RECURRENT_SEARCH_RECORD"], "a") if os.environ.get("RECURRENT_SEARCH_RECORD") else None
    busy, requests, started = 0.0, 0, time.time()
    stats_every = int(os.environ.get("RECURRENT_SEARCH_STATS", "0"))
    for raw in sys.stdin:
        t = time.perf_counter()
        if record:
            record.write(raw)
        try:
            req = json.loads(raw)
            if req.get("op") == "reset" and requests:
                print(f"recurrent search server: {requests} requests, {busy:.1f} s answering of {time.time() - started:.1f} s", file=sys.stderr, flush=True)
            reply = server.handle(req)
        except Exception as exc:  # noqa: BLE001
            reply = f"!error {type(exc).__name__}: {exc}"
            print(f"recurrent search server error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print(reply, flush=True)
        busy += time.perf_counter() - t
        requests += 1
        if stats_every and requests % stats_every == 0:  # (RECURRENT_SEARCH_STATS=N: the share of the time spent answering)
            print(f"recurrent search server: {requests} requests, {server.positions} positions, {busy:.1f} s answering of "
                  f"{time.time() - started:.1f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
