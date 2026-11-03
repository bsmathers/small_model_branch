"""Trajectory parser for the recurrent model: the official metamon parser plus extra per-state fields.

This forks the official pipeline (forward fill -> backward fill -> POV states and actions) and keeps
its `states` and `actions` identical. It adds a parallel `extras` list, one entry per state:

  last_player_action / last_opponent_action
        what each side did in the turn that produced this state: {"kind", "name"} with kind one of
        move, switch, recharge, noop, none. Both sides, by name (public).
  player_cant / opponent_cant
        why that side did not act in the turn that produced this state, or None if it acted:
        par, slp, frz, recharge, flinch (from |cant| lines); partiallytrapped (from |cant| or from a
        trap move that hit it, since continuation turns print no |cant|); slp/frz when it was put to
        sleep or frozen before acting; wokeup; fainted (knocked out before acting); opponent_fainted
        (the other side fainted first, which ends the turn).
  turn
        the battle turn number of this state (several states share one when forced switches split a turn).
  next_player_action
        the action that followed this state, by name (kind and name), independent of the move-slot
        index in `actions`. The official index depends on the moves the team predictor guessed for
        unrevealed slots, so the name is the reliable label.
  action_reason
        for a state whose action label is -1: the reason (same vocabulary as above) in the turn that
        followed, i.e. the cause of the missing action. Over 40 games, 1,271 of 1,274 -1 actions get a
        reason (80 of them are final states, which have no action by design).
  player_trapped / opponent_trapped
        partial trapping (Bind, Wrap, Fire Spin, Clamp): set when a trap move hits the Pokemon (first
        use or a [from] continuation) or a |cant|...|partiallytrapped line names it, cleared when it
        moves or switches out. Gen 1 logs have no trap start or end line.
  player_conditions_all / opponent_conditions_all
        every active side condition (the official state keeps only the most recent one as a string).
  player_team / opponent_team
        the observer's full party (fainted members included) and the opponent's revealed party, as
        UniversalPokemon dicts, in party order.
  forced_switch
        whether this state is a forced-switch decision.

Usage:
  python -m metamon.recurrent.parse RAW_DIR OUT_DIR [--limit N] [--pov USERNAME]
RAW_DIR holds Showdown protocol logs (*.log) or replay JSON files; OUT_DIR is created if needed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
import warnings
import zlib
from datetime import datetime
from typing import Optional

import lz4.frame

from metamon import interface
from metamon.interface import clean_no_numbers
from metamon.backend.replay_parser import backward, forward
from metamon.backend.replay_parser.exceptions import BackwardException, ForwardException
from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser.replay_state import Action


def summarize_action(action: Optional[Action]) -> dict:
    if action is None:
        return {"kind": "none", "name": ""}
    if action.is_noop:
        return {"kind": "recharge" if action.name == "Recharge" else "noop", "name": action.name}
    if action.is_switch or action.is_revival:
        target = action.target.name if action.target is not None else action.name
        return {"kind": "switch", "name": target}
    return {"kind": "move", "name": action.name}


def _pokemon_dict(pokemon) -> Optional[dict]:
    if pokemon is None:
        return None
    return dataclasses.asdict(interface.UniversalPokemon.from_ReplayPokemon(pokemon))


def _lock_dict(pokemon) -> Optional[dict]:
    """The locks on a Pokemon when its player decides (see forward._track_lock)."""
    if pokemon is None:
        return None
    return {"lock": pokemon.lock, "age": pokemon.lock_age, "trapped_age": pokemon.trapped_age, "disabled": pokemon.disabled_move}


def _plain(text: Optional[str]) -> Optional[str]:
    return None if text is None else "".join(c for c in text.lower() if c.isalnum())


def _memory(turn, p1: bool) -> tuple[Optional[dict], dict, dict]:
    """Counter's memory (see Turn.last_damage) from the observer's point of view."""
    ld = turn.last_damage
    if ld is not None:
        ld = {**ld, "side": "player" if (ld["side"] == "p1") == p1 else "opponent", "name": _plain(ld["name"])}
    mine, theirs = (turn.last_move_1, turn.last_move_2) if p1 else (turn.last_move_2, turn.last_move_1)
    sel_mine, sel_theirs = (turn.counter_selected_1, turn.counter_selected_2) if p1 else (turn.counter_selected_2, turn.counter_selected_1)
    return ld, {"player": _plain(mine), "opponent": _plain(theirs)}, {"player": bool(sel_mine), "opponent": bool(sel_theirs)}


def _conditions(conditions: dict) -> list[str]:
    return sorted(clean_no_numbers(c.name) for c in conditions)


class RecurrentReplayParser(ReplayParser):
    """The official parser with extra per-state fields (see the module docstring)."""

    def build_extras(self, replay: backward.POVReplay, replay_states, action_idxs) -> list[dict]:
        p1 = replay.from_p1_pov
        extras = []
        for i, (turn, state) in enumerate(zip(replay.povturnlist, replay_states)):
            own = turn.active_pokemon_1 if p1 else turn.active_pokemon_2
            opp = turn.active_pokemon_2 if p1 else turn.active_pokemon_1
            own_moves = turn.moves_1 if p1 else turn.moves_2
            opp_moves = turn.moves_2 if p1 else turn.moves_1
            own_choices = turn.choices_1 if p1 else turn.choices_2
            opp_choices = turn.choices_2 if p1 else turn.choices_1
            own_cant = (turn.cant_1 if p1 else turn.cant_2)[0]
            opp_cant = (turn.cant_2 if p1 else turn.cant_1)[0]
            own_team = turn.pokemon_1 if p1 else turn.pokemon_2
            opp_team = turn.pokemon_2 if p1 else turn.pokemon_1
            own_trapped = bool(own[0] is not None and own[0].partially_trapped)
            reason = replay.cantlist[i] if action_idxs[i] == -1 else None
            if action_idxs[i] == -1 and reason is None and own_trapped:
                # trapped and nothing else explains it: the trapper switched out on a turn the
                # trapped Pokemon still lost (no move, |cant| or trap line is printed for it)
                reason = "partiallytrapped"
            if action_idxs[i] == -1 and reason is None and own[0] is not None:
                status = str(getattr(own[0].status, "name", "")).lower()
                if status in ("slp", "frz"):
                    # asleep or frozen and nothing else explains it (the Gen 1 trap-and-switch bug
                    # leaves a frozen Pokemon without any |cant| line); it cannot act in that state
                    reason = status
            last_damage, last_move, counter_selected = _memory(turn, p1)
            extras.append(
                {
                    "last_damage": last_damage,
                    "last_move": last_move,
                    "counter_selected": counter_selected,
                    "last_player_action": summarize_action(own_moves[0] or own_choices[0]),
                    "last_opponent_action": summarize_action(opp_moves[0] or opp_choices[0]),
                    "turn": turn.turn_number,
                    "next_player_action": summarize_action(replay.actionlist[i][0]),
                    "player_cant": own_cant,
                    "opponent_cant": opp_cant,
                    "action_reason": reason,
                    "player_lock": _lock_dict(own[0]),
                    "opponent_lock": _lock_dict(opp[0]),
                    "player_trapped": own_trapped,
                    "opponent_trapped": bool(opp[0] is not None and opp[0].partially_trapped),
                    "player_conditions_all": _conditions(turn.conditions_1 if p1 else turn.conditions_2),
                    "opponent_conditions_all": _conditions(turn.conditions_2 if p1 else turn.conditions_1),
                    "player_team": [_pokemon_dict(m) for m in own_team],
                    "opponent_team": [_pokemon_dict(m) for m in opp_team],
                    "forced_switch": bool(state.force_switch),
                }
            )
        return extras

    def povreplay_to_seq_extras(self, replay: backward.POVReplay):
        replay_states, actions = self.povreplay_to_state_action(replay)
        universal_states, action_idxs = self.state_action_to_obs_action_reward(replay_states, actions)
        extras = self.build_extras(replay, replay_states, action_idxs)
        return universal_states, action_idxs, extras

    def save_to_disk(self, replay, time_played, player_username, opponenent_username):
        if replay.result is None:
            raise ForwardException(f"{replay.gameid}: the log has no result")
        universal_states, action_idxs, extras = self.povreplay_to_seq_extras(replay)
        won = replay.result  # WIN, LOSS or TIE
        filename = f"{replay.gameid}_{replay.rating}_{player_username}_vs_{opponenent_username}_{time_played.strftime('%m-%d-%Y')}_{won}"
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        output_json = {
            "states": [state.to_dict() for state in universal_states],
            "actions": action_idxs,
            "extras": extras,
        }
        with lz4.frame.open(os.path.join(self.output_dir, f"{filename}.json.lz4"), "wb") as f:
            f.write(json.dumps(output_json).encode("utf-8"))

    # -- raw Showdown protocol logs (*.log), which have no replay JSON wrapper --------------------

    @staticmethod
    def wrap_log(path: str) -> dict:
        """The replay-JSON fields the parser reads, taken from a plain protocol log."""
        text = open(path, errors="ignore").read()
        players = {}
        for line in text.splitlines():
            if line.startswith("|player|"):
                parts = line.split("|")
                if len(parts) > 3 and parts[2] in ("p1", "p2"):
                    players[parts[2]] = parts[3]
        m = re.search(r"^\|t:\|(\d+)", text, re.M)
        tier = re.search(r"^\|tier\|(.+)$", text, re.M)
        fmt = re.sub(r"[^a-z0-9]", "", tier.group(1).lower()) if tier else os.path.basename(path).split("-")[0]
        # "Gen 1 OU" -> "gen1ou"
        return {
            "players": [players["p1"], players["p2"]],
            "uploadtime": int(m.group(1)) if m else 0,
            "formatid": fmt,
            "log": text,
        }

    def parse_data(self, gameid: str, data: dict, pov_username: Optional[str] = None):
        p1_username, p2_username = data["players"]
        time_played = datetime.fromtimestamp(int(data["uploadtime"]))
        replay = forward.ParsedReplay(gameid=gameid, format=data["formatid"], time_played=time_played)
        log = self.clean_log(data)
        # The official team predictor samples unrevealed moves with the global RNG, so two parses of
        # the same game differ. Seeding per game makes the output reproducible.
        random.seed(gameid)
        try:
            import numpy as np

            np.random.seed(zlib.crc32(gameid.encode()) % (2**32))
        except ImportError:
            pass
        try:
            replay = forward.forward_fill(replay, log, verbose=self.verbose)
            from_p1, from_p2 = backward.backward_fill(replay, team_predictor=self.team_predictor)
            if pov_username in (None, p1_username):
                self.save_to_disk(from_p1, time_played, p1_username, p2_username)
            if pov_username in (None, p2_username):
                self.save_to_disk(from_p2, time_played, p2_username, p1_username)
            return True
        except (ForwardException, BackwardException) as e:
            self.add_exception_to_history(e, gameid)
            warnings.warn(f"{gameid}: {e}")
            return False

    def parse_replay(self, path: str, pov_username: Optional[str] = None):
        # output names are `<gameid>_<rating>_...` and later steps split on "_", so the id has none
        gameid = os.path.basename(path).rsplit(".", 1)[0].replace("_", "-")
        if path.endswith(".log"):
            data = self.wrap_log(path)
        else:
            with open(path) as f:
                data = json.load(f)
        return self.parse_data(gameid, data, pov_username)


def main() -> None:
    # Python randomises string hashing per process, which changes the team predictor's set iteration
    # order. A fixed hash seed (plus the per-game seed above) makes the output reproducible.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "metamon.recurrent.parse", *sys.argv[1:]])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="take every Nth file (to sample across a large directory)")
    ap.add_argument("--pov", default=None, help="only write this username's point of view")
    ap.add_argument("--shard", default="0/1", help="i/n: parse every n-th file starting at i")
    args = ap.parse_args()
    files = sorted(os.path.join(args.raw_dir, f) for f in os.listdir(args.raw_dir) if f.endswith((".log", ".json")))
    files = files[:: max(1, args.every)]
    shard, shards = map(int, args.shard.split("/"))
    files = files[shard::shards]
    if args.limit:
        files = files[: args.limit]
    # a game whose output already exists is skipped, so an interrupted run resumes
    done = set()
    if os.path.isdir(args.out_dir):
        done = {name.split("_")[0] for name in os.listdir(args.out_dir)}
    files = [f for f in files if os.path.basename(f).rsplit(".", 1)[0].replace("_", "-") not in done]
    parser = RecurrentReplayParser(replay_output_dir=args.out_dir)
    ok = sum(bool(parser.parse_replay(f, args.pov)) for f in files)
    print(f"parsed {ok} of {len(files)} games; errors: {parser.summarize_errors()}")


if __name__ == "__main__":
    main()
