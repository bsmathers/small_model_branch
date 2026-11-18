"""Public engine snapshot -> Metamon observation connector.

The engine snapshot is already redacted. This adapter translates it into the
same ``UniversalState`` representation used by Metamon's live battle backend;
it must never inspect or reconstruct hidden opponent team data.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

import torch

from snapshot_effects import SnapshotEffectMemory


_STATUS = {
    "none": "nostatus",
    "sleep": "slp",
    "poison": "psn",
    "toxic": "tox",
    "burn": "brn",
    "freeze": "frz",
    "paralyze": "par",
    "fnt": "fnt",
}

@functools.lru_cache(maxsize=None)
def _norm_str(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _norm(value: object) -> str:
    return _norm_str(str(value))  # (repo copy) memoized: the search server normalizes the same few hundred names millions of times


def _status(value: object) -> str:
    return _STATUS.get(str(value).lower(), "nostatus")


def _deduplicated_types(values) -> list[str]:
    result = []
    for value in values or []:
        name = _norm(value)
        if name in {"", "none", "notype"} or name in result:
            continue
        result.append(name)
    return result


class SnapshotEncoder:
    def __init__(self, metamon_root: Path, observation_space=None):
        sys.path.insert(0, str(metamon_root.resolve()))
        from poke_env.environment import Move, Pokemon
        from metamon.interface import DefaultObservationSpace, TokenizedObservationSpace
        from metamon.tokenizer import get_tokenizer

        self.Move = Move
        self.Pokemon = Pokemon
        self._species_feature_cache = {}
        # (repo copy) True for TaurosV0 (tauros_inference): Toxic and Disable from the duel snapshot's "toxic" / "disabled",
        # as native metamon shows them. Off for the recurrent model's servers, whose features (and rnative.zig) keep the
        # snapshot's poison status and no Disable effect
        self.duel_extras = False
        self._universal = __import__(
            "metamon.interface",
            fromlist=["UniversalMove", "UniversalPokemon", "UniversalState"],
        )
        self.space = observation_space or TokenizedObservationSpace(
            base_obs_space=DefaultObservationSpace(),
            tokenizer=get_tokenizer("DefaultObservationSpace-v0"),
        )

    def _move(self, payload: dict | None):
        UniversalMove = self._universal.UniversalMove
        if not payload or _norm(payload.get("id", "None")) in {"", "none"}:
            return UniversalMove.blank_move()
        # (repo copy) a move is fixed by these four fields and nothing changes a UniversalMove after it is made, so equal
        # payloads share one object (the search server builds ~46 moves per node otherwise)
        key = (payload.get("id"), payload.get("max_pp"), payload.get("pp"), payload.get("uses"))
        cache = self.__dict__.setdefault("_move_cache", {})
        made = cache.get(key)
        if made is None:
            made = cache[key] = self._make_move(payload)
        return made

    def _make_move(self, payload: dict):
        UniversalMove = self._universal.UniversalMove
        move_id = str(payload["id"])
        try:
            move = self.Move(move_id.lower(), gen=1)
            maximum = int(payload.get("max_pp", move.max_pp))
            if "pp" in payload:
                current = int(payload["pp"])
            elif "uses" in payload:
                current = maximum - int(payload["uses"])
            else:
                current = maximum
            maximum = max(maximum, 0)
            current = max(0, min(current, maximum))
            return UniversalMove(
                name=move.id,
                move_type=move.type.name.lower(),
                category=move.category.name.lower(),
                base_power=move.base_power,
                accuracy=move.accuracy,
                priority=move.priority,
                current_pp=current,
                max_pp=maximum,
            )
        except (KeyError, ValueError):
            maximum = max(0, int(payload.get("max_pp", payload.get("pp", 0))))
            current = max(
                0,
                min(int(payload.get("pp", maximum - int(payload.get("uses", 0)))), maximum),
            )
            return UniversalMove(
                name=_norm(move_id),
                move_type="nomove",
                category="nomove",
                base_power=0,
                accuracy=1.0,
                priority=0,
                current_pp=current,
                max_pp=maximum,
            )

    def _species_features(self, species: str) -> tuple[list[str], dict[str, int]]:
        cached = self._species_feature_cache.get(species)
        if cached is not None:
            return cached
        try:
            pokemon = self.Pokemon(gen=1, species=species)
            types = [kind.name.lower() for kind in pokemon.types]
            result = (types, dict(pokemon.base_stats))
        except (KeyError, ValueError):
            result = ([], {name: 0 for name in ("atk", "spa", "def", "spd", "spe", "hp")})
        self._species_feature_cache[species] = result
        return result

    @staticmethod
    def _active_effect(payload: dict, side: str, space) -> str:
        """Match UniversalPokemon's single selected active effect.

        poke-env stores an age/action counter per effect and UniversalPokemon
        chooses the lowest counter, breaking ties by insertion order. We retain
        those public counters alongside Metamon's per-battle observation memory.
        Search forks clone this memory, so speculative branches remain isolated.
        """
        if space is None:
            return SnapshotEffectMemory().select(payload, side)
        memory = getattr(space, "_snapshot_effect_memory", None)
        if memory is None:
            memory = SnapshotEffectMemory()
            setattr(space, "_snapshot_effect_memory", memory)
        return memory.select(payload, side)

    def _pokemon(
        self,
        payload: dict,
        active: bool,
        *,
        exact_hp: bool,
        effect: str = "noeffect",
    ):
        # (repo copy) a reserve built twice from the same payload object (by state() and again by a caller that wants the whole
        # party) is built once: `reuse` maps id(payload) to what was made from it, while the caller keeps it on (begin_reuse /
        # end_reuse around one snapshot, so the payload objects outlive their ids). Only non-active Pokemon, which nothing
        # changes after they are made, are shared.
        reuse = self.__dict__.get("_reuse")
        if reuse is not None and not active:
            key = (id(payload), exact_hp, effect)
            made = reuse.get(key)
            if made is None:
                made = reuse[key] = self._make_pokemon(payload, active, exact_hp=exact_hp, effect=effect)
            return made
        return self._make_pokemon(payload, active, exact_hp=exact_hp, effect=effect)

    def begin_reuse(self) -> None:
        self._reuse = {}

    def end_reuse(self) -> None:
        self._reuse = None

    def _make_pokemon(
        self,
        payload: dict,
        active: bool,
        *,
        exact_hp: bool,
        effect: str = "noeffect",
    ):
        UniversalPokemon = self._universal.UniversalPokemon
        moves = [
            self._move(move)
            for move in payload.get("moves", [])
            if _norm(move.get("id", "None")) not in {"", "none"}
        ]
        name = _norm(payload.get("id", payload.get("species", "unknown"))) or "unknown"
        base_species = _norm(payload.get("base_species", name)) or name
        inferred_types, stats = self._species_features(name)
        raw_types = _deduplicated_types(payload.get("types")) or inferred_types
        types = UniversalPokemon.universal_types(list(raw_types))
        if "hp_pct" in payload:
            hp_pct = float(payload["hp_pct"])
        else:
            maximum_hp = max(int(payload.get("maxhp", 1)), 1)
            hp_pct = float(payload.get("hp", 0)) / maximum_hp
        hp_pct = max(0.0, min(1.0, hp_pct))
        status = _status(payload.get("status", "none"))
        if exact_hp and hp_pct <= 0:
            status = "fnt"
        boosts = payload.get("boosts", {}) if active else {}
        special = int(boosts.get("spc", boosts.get("spa", 0)))
        return UniversalPokemon(
            name=name,
            base_species=base_species,
            hp_pct=hp_pct,
            types=types,
            tera_type="notype",
            # (repo copy) native metamon gives the opponent's Pokemon "unknownitem" and our own "noitem" in Gen 1 (every state
            # of its Showdown trajectories and of the parsed replays); the opponent's are the ones built with exact_hp=False
            item="noitem" if exact_hp else "unknownitem",
            ability="noability",
            lvl=int(payload.get("level", 100)),
            status=status,
            effect=effect,
            moves=moves,
            atk_boost=int(boosts.get("atk", 0)),
            spa_boost=special,
            def_boost=int(boosts.get("def", 0)),
            spd_boost=int(boosts.get("spd", special)),
            spe_boost=int(boosts.get("spe", 0)),
            accuracy_boost=int(boosts.get("accuracy", 0)),
            evasion_boost=int(boosts.get("evasion", 0)),
            base_atk=int(stats["atk"]),
            base_spa=int(stats["spa"]),
            base_def=int(stats["def"]),
            base_spd=int(stats["spd"]),
            base_spe=int(stats["spe"]),
            base_hp=int(stats["hp"]),
        )

    def _public_effects(self, snapshot: dict, side: str) -> list:
        """(repo copy) A side's active effects from the public volatile lists, as the replay parser would see them: a trap's
        "binding" belongs to its user (the older "volatiles" list called it "partiallytrapped", the reverse), and the Pokemon
        caught in the other side's trap carries "partiallytrapped" (named after the trapping move by active_effects)."""
        other = "opponent" if side == "user" else "user"
        mine = list(snapshot[side].get("public_volatiles") or [])
        theirs = list(snapshot[other].get("public_volatiles") or [])
        # Native metamon's states never carry a recharge effect (it is not in TaurosV0's vocabulary: an unknown token) and
        # carry a trap's name in 0.04% of states (Gen 1 logs have no trap start line), against 3% here: both are left out,
        # as is the lock of Thrash
        effects = [v for v in mine if v not in ("binding", "mustrecharge", "lockedmove")]
        # (repo copy) Disable (the snapshot's volatile lists leave it out; the duel snapshot's "disabled" names the move)
        if self.duel_extras and (snapshot.get("disabled") or {}).get(side) and "disable" not in effects:
            effects.append("disable")
        return effects

    def state(self, snapshot: dict, space=None):
        UniversalState = self._universal.UniversalState
        user = snapshot["user"]
        opponent = snapshot["opponent"]
        own = user["pokemon"]
        own_active_payload = dict(own[0])
        own_active_payload["turn"] = snapshot.get("turn", 0)
        own_active_payload["boosts"] = user.get("boosts", {})
        # (repo copy) effects from the public lists for both sides; the snapshot's opponent "active" has no volatiles, so
        # the opponent's effects (screens, Substitute, confusion, Leech Seed, recharge, traps) were all lost before
        own_active_payload["volatiles"] = self._public_effects(snapshot, "user") if "public_volatiles" in user else user.get("volatiles", [])
        own_active_payload["conditions"] = user.get("conditions", {})
        own_active_payload["last_move"] = user.get("last_move", "None")
        own_active_payload["trapping_move"] = opponent.get("last_move", "None")
        # (repo copy) Toxic is Showdown's "tox" status (the snapshot's status calls it poison; the duel snapshot flags it)
        toxic = (snapshot.get("toxic") or {}) if self.duel_extras else {}
        if toxic.get("user"):
            own_active_payload["status"] = "toxic"
        active = self._pokemon(
            own_active_payload,
            True,
            exact_hp=True,
            effect=self._active_effect(own_active_payload, "user", space),
        )
        switches = [
            self._pokemon(mon, False, exact_hp=True)
            for mon in own[1:]
            if int(mon.get("hp", 0)) > 0 and str(mon.get("status", "none")).lower() != "fnt"
        ]

        active_species = _norm(opponent.get("active_species", "unknown")) or "unknown"
        active_public = opponent.get("active", {})
        active_base_species = _norm(active_public.get("base_species", active_species)) or active_species
        revealed = opponent.get("revealed", [])
        known = next(
            (mon for mon in revealed if _norm(mon.get("species", "")) == active_base_species),
            {},
        )
        foe_payload = {
            "id": active_species,
            "base_species": active_base_species,
            "types": active_public.get("types", []),
            "hp_pct": active_public.get("hp", known.get("hp", 1.0)),
            "status": "toxic" if toxic.get("opponent") else active_public.get("status", known.get("status", "none")),
            "moves": known.get("moves", []),
            "boosts": active_public.get("boosts", {}),
            "volatiles": self._public_effects(snapshot, "opponent") if "public_volatiles" in opponent else active_public.get("volatiles", []),
            "conditions": active_public.get("conditions", {}),
            "last_move": opponent.get("last_move", "None"),
            "trapping_move": user.get("last_move", "None"),
            "turn": snapshot.get("turn", 0),
        }
        foe = self._pokemon(
            foe_payload,
            True,
            exact_hp=False,
            effect=self._active_effect(foe_payload, "opponent", space),
        )
        return UniversalState(
            format="gen1ou",
            player_active_pokemon=active,
            opponent_active_pokemon=foe,
            available_switches=switches,
            player_prev_move=self._move({"id": user.get("last_move", "None")}),
            opponent_prev_move=self._move({"id": opponent.get("last_move", "None")}),
            opponents_remaining=6
            - sum(str(mon.get("status", "none")).lower() == "fnt" for mon in revealed),
            # Reflect, Light Screen, and Mist are emitted as Pokemon effects in
            # Gen 1, not as poke-env side conditions.
            player_conditions="noconditions",
            opponent_conditions="noconditions",
            weather="noweather",
            battle_field="nofield",
            forced_switch=bool(snapshot.get("force_switch", False)),
            battle_won=False,
            battle_lost=False,
            can_tera=False,
            opponent_teampreview=[],
        )

    def observation(self, snapshot: dict, space=None):
        target = self.space if space is None else space
        return target.state_to_obs(self.state(snapshot, target))

    def encode(self, snapshot: dict) -> tuple[list[int], list[float]]:
        observation = self.observation(snapshot)
        return (
            torch.as_tensor(observation["text_tokens"]).view(-1).tolist(),
            torch.as_tensor(observation["numbers"]).view(-1).tolist(),
        )
