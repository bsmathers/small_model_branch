"""The recurrent student as a poke-env player on a real Pokémon Showdown server.

The student's inputs at a decision come from the SAME code that made its training data: the battle log received so far goes
through the training parser (metamon.recurrent.parse: forward fill, backward fill, extras) and the same feature code
(features.encode_observer, records.last_turn_from_extras), and the model is run over the whole game so far from its first state
(the training sequence semantics). What differs from training, and how:

  - the log ends at the decision: a turn decision is cut after its |turn| line and the empty turn that line opens is dropped; a
    replacement after our faint is the log as it stands, with forced_switch set from the request (the log shows it only later).
    Checked on 20 recorded games (prefix_fidelity.py): the inputs are identical to the full-game parse at 99% of 1964 decisions;
    the rest differ only in the opponent's information that the full-game parse fills in from later in the game.
  - our own team: the team we brought (its file), given to the parser in place of the team predictor's guess for our side. The
    opponent's side gets the standard predictor, as in training.
  - our own HP: the player's stream shows it exactly (253/353); recorded games (training) show percentages. Converted with
    Showdown's HP Percentage Mod rule (ceil, and 99 for anything below full).

The choice: softmax(logits / temperature) over the legal options that map to policy slots (canonical_own / policy_options, as in
training), sampled; temperature 0 plays the argmax. An option with no slot (recharge, struggle) is taken when it is the only
one; anything else unmappable is counted as a fallback (a random legal option).
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import random
import re
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder

from metamon import interface
from metamon.backend.replay_parser import backward, checks, forward
from metamon.backend.team_prediction.team import TeamSet
from metamon.interface import move_name, pokemon_name
from metamon.recurrent.features import canonical_own, encode_observer, policy_options
from metamon.recurrent.model import PolicyValueModel
from metamon.recurrent.parse import RecurrentReplayParser
from metamon.recurrent.records import last_turn_from_extras
from metamon.recurrent.tables import Tables

# an unfinished game is the point here (the parser otherwise refuses logs without a result)
checks.check_finished = lambda replay: None
forward.check_finished = lambda replay: None

PLAYED = datetime(2026, 3, 1)  # the date the opponent's team predictor uses (usage statistics exist up to 2026-04)
HP = re.compile(r"^(\d+)/(\d+)(\s.*)?$")


def percent_hp(token: str) -> str:
    """Showdown's HP Percentage Mod: ceil(100 * hp / max), and 99 for anything below full."""
    m = HP.match(token)
    if not m or m.group(2) == "100":
        return token
    hp, top = int(m.group(1)), int(m.group(2))
    pct = math.ceil(100 * hp / top) if top else 0
    if pct == 100 and hp < top:
        pct = 99
    return f"{pct}/100{m.group(3) or ''}"


def spectator_line(line: str, side: str) -> str:
    """A line of the player's stream as a recorded game shows it: our own HP in percent."""
    parts = line.split("|")
    if len(parts) > 3 and parts[2].startswith(f"{side}a:"):
        parts = parts[:3] + [percent_hp(p) for p in parts[3:]]
    return "|".join(parts)


class OwnTeam:
    """The standard predictor, except for our side (p1: the first call of a backward fill, p2: the second): the team we brought."""

    def __init__(self, base, p1: bool, own: TeamSet):
        self.base, self.p1, self.own, self.calls = base, p1, own, 0

    def predict(self, team, **kw):
        self.calls += 1
        if (self.calls == 1) == self.p1:
            return copy.deepcopy(self.own)
        return self.base.predict(team, **kw)


class TeamFiles(Teambuilder):
    """A random team file per battle (seeded); remembers which, for the parser."""

    def __init__(self, files: list[Path], seed: int):
        self.files, self.rng, self.last = files, random.Random(seed), None

    def yield_team(self) -> str:
        self.last = self.rng.choice(self.files)
        return self.join_team(self.parse_showdown_team(self.last.read_text()))


class StudentPlayer(Player):
    def __init__(self, weights: str, team_dir: str, temperature: float = 1.0, seed: int = 0, log_dir: str | None = None, **kw):
        files = sorted(Path(team_dir).glob("*_team")) or sorted(Path(team_dir).iterdir())
        self.teams = TeamFiles(files, seed)
        super().__init__(battle_format="gen1ou", team=self.teams, **kw)
        self.temperature, self.rng = temperature, random.Random(f"student-{seed}")
        # (weights with the optional actor MLP or opponent head load too; blends play on the Zig harness only)
        self.model = PolicyValueModel.from_state(torch.load(weights, map_location="cpu"))
        self.model.eval()
        self.tables = Tables.from_dex(1)
        self.parser = RecurrentReplayParser(replay_output_dir=None, verbose=False)
        self.lines: dict[str, list[str]] = defaultdict(list)
        self.team_of: dict[str, Path] = {}
        self.cut_turn: dict[str, int] = {}  # battle -> the turn number its last turn decision was cut at
        self.stats = defaultdict(int)
        self.unmapped = defaultdict(int)  # option -> count, for the report
        self.log_dir = Path(log_dir) if log_dir else None
        self.records: dict[str, dict] = {}

    # ---- the raw log ----
    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">").strip()
        if tag not in self.team_of and self.teams.last is not None:
            self.team_of[tag] = self.teams.last  # (one battle at a time: the team just sent with the challenge)
        for m in split_messages[1:]:
            if len(m) > 1 and m[1] not in ("request", "split"):
                self.lines[tag].append("|".join(m))
        await super()._handle_battle_message(split_messages)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        rec = self.records.setdefault(battle.battle_tag, {})
        rec.update({"won": battle.won, "turns": battle.turn, "team": str(self.team_of.get(battle.battle_tag))})
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{battle.battle_tag}.log").write_text("\n".join(self.lines[battle.battle_tag]) + "\n")
            (self.log_dir / f"{battle.battle_tag}.json").write_text(json.dumps(rec))

    # ---- the decision ----
    async def choose_move(self, battle):
        tag, side = battle.battle_tag, battle.player_role
        forced = bool(battle.force_switch)
        # The request can arrive before the log lines it follows (poke-env then calls this at once, with its turn counter
        # stale too), so the wait is on the log itself: a turn decision needs a |turn| line newer than the one the previous
        # turn decision was cut at; a replacement needs our faint after the last |turn| line. (Messages are handled in their
        # own tasks, so the lines keep arriving while this waits.)
        def ready() -> bool:
            lines = self.lines[tag]
            turns = [i for i, l in enumerate(lines) if l.startswith("|turn|")]
            if not forced:
                return bool(turns) and int(lines[turns[-1]].split("|")[2]) > self.cut_turn.get(tag, 0)
            start = turns[-1] if turns else 0
            return any(l.startswith(f"|faint|{side}a:") for l in lines[start:])

        if not ready():
            self.stats["waited_for_log"] += 1
            for _ in range(500):
                await asyncio.sleep(0.01)
                if ready():
                    break
            else:
                self.stats["log_wait_timeouts"] += 1
        try:
            return self.decide(battle, forced)
        except Exception as e:  # noqa: BLE001
            self.stats["errors"] += 1
            if self.stats["errors"] <= 5:
                print(f"student error in {tag} turn {battle.turn}: {type(e).__name__}: {e}", flush=True)
            return self.choose_random_move(battle)

    def decide(self, battle, forced: bool):
        tag, side = battle.battle_tag, battle.player_role
        lines = [spectator_line(l, side) for l in self.lines[tag]]
        if not forced:
            last_turn = max(i for i, l in enumerate(lines) if l.startswith("|turn|"))
            self.cut_turn[tag] = int(lines[last_turn].split("|")[2])
            lines = lines[: last_turn + 1]
        players = {}
        for l in lines:
            if l.startswith("|player|"):
                p = l.split("|")
                if len(p) > 3 and p[3]:
                    players[p[2]] = p[3]
        data = {"players": [players.get("p1", "p1"), players.get("p2", "p2")], "uploadtime": int(PLAYED.timestamp()),
                "formatid": "gen1ou", "log": "\n".join(lines)}
        gameid = tag.replace("_", "-")
        random.seed(gameid)
        np.random.seed(zlib.crc32(gameid.encode()) % (2**32))
        replay = forward.ParsedReplay(gameid=gameid, format="gen1ou", time_played=PLAYED)
        replay = forward.forward_fill(replay, self.parser.clean_log(data), verbose=False)
        own = TeamSet.from_showdown_file(str(self.team_of[tag]), "gen1ou")
        pov1, pov2 = backward.backward_fill(replay, team_predictor=OwnTeam(self.parser.team_predictor, side == "p1", own))
        states, _, extras = self.parser.povreplay_to_seq_extras(pov1 if side == "p1" else pov2)
        states = [s.to_dict() for s in states]
        if not forced:
            states, extras = states[:-1], extras[:-1]  # (the empty turn the |turn| line opened)
        else:
            states[-1]["forced_switch"] = True
            extras[-1]["forced_switch"] = True
        logits, ordered = self.run(states, extras)
        rec = self.records.setdefault(tag, {"decisions": []})
        rec["decisions"].append({"t": len(states) - 1, "turn": extras[-1]["turn"], "forced": forced, "cut_lines": len(lines),
                                 "features": self.last_features})
        moves, reserves = policy_options(ordered)
        slots = {}
        fight = None
        for m in battle.available_moves:
            if move_name(m.id) in moves:
                slots[m] = moves.index(move_name(m.id))
            elif m.id == "fight" and moves:
                # Showdown offers one "Fight" when every move would do the same (asleep, frozen, ...): training's
                # fight-or-switch decisions, where the moves' probabilities add up to the fight option's
                fight = m
        for p in battle.available_switches:
            name = pokemon_name(p.species)
            if name in reserves:
                slots[p] = 4 + reserves.index(name)
        options = list(battle.available_moves) + list(battle.available_switches)
        self.stats["decisions"] += 1
        if not slots and fight is None:
            if len(options) == 1:
                self.stats["single_unmapped"] += 1
                return self.create_order(options[0])
            self.stats["fallbacks"] += 1
            return self.choose_random_move(battle)
        if len(slots) + (fight is not None) < len(options):
            self.stats["unmapped_options"] += 1
            for o in options:
                if o not in slots and o is not fight:
                    name = f"move {move_name(o.id)}" if hasattr(o, "id") and o in battle.available_moves else f"switch {pokemon_name(o.species)}"
                    self.unmapped[f"{name} (slots: moves {moves}, reserves {reserves})"] += 1
        keys = list(slots) + ([fight] if fight is not None else [])

        def distribution(temperature: float) -> list:
            # softmax over the slots the options cover (Fight covers every move slot), at the temperature; Fight gets the
            # sum of its move slots' probabilities (scaling the slots first, then adding: at T != 1 that is not the same
            # as scaling a log-sum-exp)
            covered = sorted({slots[k] for k in slots} | (set(range(len(moves))) if fight is not None else set()))
            p = torch.softmax(logits[covered] / temperature, 0)
            at = {slot: float(q) for slot, q in zip(covered, p)}
            return [at[slots[k]] if k is not fight else sum(at[j] for j in range(len(moves))) for k in keys]

        if self.temperature > 0:
            pick = self.rng.choices(keys, weights=distribution(self.temperature))[0]
        else:
            probs = distribution(1.0)
            pick = keys[max(range(len(keys)), key=lambda i: probs[i])]
        return self.create_order(pick)

    def run(self, states, extras):
        """The model over the whole game so far, as in training (float16 inputs); returns the last decision's logits."""
        feats, ordered = [], None
        for t, (s, e) in enumerate(zip(states, extras)):
            state = interface.UniversalState.from_dict(copy.deepcopy(s))
            party = [interface.UniversalPokemon.from_dict(copy.deepcopy(m)) for m in e["player_team"] if m is not None]
            revealed = [interface.UniversalPokemon.from_dict(copy.deepcopy(m)) for m in e["opponent_team"] if m is not None]
            feats.append(encode_observer(state, party, revealed, last_turn_from_extras(states[t - 1] if t else None, s, e), t + 1, self.tables, extras=e))
            if t == len(states) - 1:
                ordered, _ = canonical_own(state, party)
        ids = {k: torch.from_numpy(np.stack([f.ids[k] for f in feats]).astype(np.int64))[None] for k in feats[0].ids}
        floats = {k: torch.from_numpy(np.stack([f.floats[k] for f in feats]).astype(np.float16).astype(np.float32))[None] for k in feats[0].floats}
        with torch.no_grad():
            tokens = self.model.observation_tokens(ids, floats)
            h = torch.zeros(1, self.model.cfg.d)
            for t in range(tokens.shape[1]):
                logits, _, h = self.model.step(tokens[:, t], h)
        last = feats[-1]
        self.last_features = {**{f"ids_{k}": v.tolist() for k, v in last.ids.items()}, **{f"floats_{k}": v.tolist() for k, v in last.floats.items()}}
        return logits[0], ordered
