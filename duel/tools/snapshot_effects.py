"""Canonical projection of engine snapshot effects onto Metamon semantics.

Generation I screens are active-Pokemon effects even though ActorSnapshot also
publishes their counters under ``conditions``.  Metamon's released observation
space exposes one effect token, choosing the effect with the lowest stored
counter and then dictionary insertion order. In Metamon's Gen 1 replay parser
those counters are not aged at turn boundaries, so first-observed insertion
order is the training-compatible rule. ``SnapshotEffectMemory`` preserves that
order across live turns and, importantly, can be deep-copied for search forks.
"""

from __future__ import annotations

import re


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


# Names emitted by public_snapshot that correspond to poke-env/Metamon effects.
# Transform and the Gen-I stat-nullification markers are represented elsewhere
# in Metamon and must not displace a genuine UniversalPokemon.effect token.
_DIRECT_EFFECTS = {
    "bide",
    "confusion",
    "disable",
    "focusenergy",
    "leechseed",
    "lightscreen",
    "lockedmove",
    "mimic",
    "mist",
    "mustrecharge",
    "partiallytrapped",
    "rage",
    "reflect",
    "substitute",
    "typechange",
}
_SCREENS = ("reflect", "lightscreen", "mist")
_TRAPPING_MOVES = {"bind", "clamp", "firespin", "wrap"}

_BOOST_MOVES = {
    "agility": ("spe",),
    "amnesia": ("spc", "spa", "spd"),
    "swordsdance": ("atk",),
    "meditate": ("atk",),
    "sharpen": ("atk",),
    "growth": ("spc", "spa", "spd"),
    "acidarmor": ("def",),
    "barrier": ("def",),
    "defensecurl": ("def",),
    "harden": ("def",),
    "withdraw": ("def",),
    "doubleteam": ("evasion",),
    "minimize": ("evasion",),
}


def _enabled(value: object) -> bool:
    if isinstance(value, str):
        return norm(value) not in {"", "0", "false", "none", "noeffect"}
    return bool(value)


def active_effects(payload: dict) -> list[str]:
    """Return active Metamon-compatible effects in stable snapshot order."""
    result: list[str] = []

    explicit = norm(payload.get("effect", ""))
    if explicit not in {"", "none", "noeffect"}:
        result.append(explicit)

    last_move = norm(payload.get("last_move", ""))
    trapping_move = norm(payload.get("trapping_move", ""))
    for raw in payload.get("volatiles", ()) or ():
        effect = norm(raw)
        if effect in {"lockedmove", "partiallytrapped"}:
            source = last_move if effect == "lockedmove" else trapping_move
            if source in _TRAPPING_MOVES:
                effect = source
        if effect in _DIRECT_EFFECTS or effect in _TRAPPING_MOVES:
            if effect not in result:
                result.append(effect)

    conditions = payload.get("conditions") or {}
    if isinstance(conditions, dict):
        for effect in _SCREENS:
            if _enabled(conditions.get(effect, False)) and effect not in result:
                result.append(effect)
    return result


class SnapshotEffectMemory:
    """Stateful single-effect selector matching Metamon's lossy interface."""

    def __init__(self) -> None:
        self.clock = 0
        self.sides: dict[str, dict] = {}

    def __deepcopy__(self, memo):
        # (repo copy) the search server copies this memory for every node: copying its two levels directly is ~10x faster
        # than the generic deepcopy, with the same result
        other = SnapshotEffectMemory.__new__(SnapshotEffectMemory)
        other.clock = self.clock
        other.sides = {side: {"identity": r["identity"], "turn": r["turn"], "effects": dict(r["effects"])} for side, r in self.sides.items()}
        return other

    def select(self, payload: dict, side: str) -> str:
        active = active_effects(payload)
        identity = norm(payload.get("base_species", payload.get("id", "unknown")))
        turn = int(payload.get("turn", 0))
        record = self.sides.get(side)
        if record is None or record["identity"] != identity:
            record = {"identity": identity, "turn": turn, "effects": {}}
            self.sides[side] = record

        tracked = record["effects"]
        record["turn"] = turn

        for effect in tuple(tracked):
            if effect not in active:
                del tracked[effect]
        for effect in active:
            if effect not in tracked:
                self.clock += 1
                tracked[effect] = self.clock
        if not tracked:
            return "noeffect"
        return min(tracked, key=tracked.get)


def selected_effect(payload: dict) -> str:
    """Stateless fallback for tests and one-shot callers.

    Stateful inference should use ``SnapshotEffectMemory``.  The first active
    effect is deterministic and follows ActorSnapshot's stable field order.
    """
    effects = active_effects(payload)
    return effects[0] if effects else "noeffect"


def dominated_move_names(snapshot: dict) -> set[str]:
    """Moves whose effect is already active or whose own boost is capped.

    This intentionally excludes status, healing, immunity, and Substitute
    heuristics: simultaneous opponent actions can make those choices useful.
    """
    user = snapshot.get("user") or {}
    roster = user.get("pokemon") or []
    if not roster:
        return set()
    payload = dict(roster[0])
    payload.update(
        {
            "volatiles": user.get("volatiles", []),
            "conditions": user.get("conditions", {}),
        }
    )
    active = set(active_effects(payload))
    dominated = {
        move
        for move in ("reflect", "lightscreen", "mist", "focusenergy")
        if move in active
    }
    boosts = user.get("boosts") or {}
    for move, aliases in _BOOST_MOVES.items():
        present = [int(boosts[name]) for name in aliases if name in boosts]
        if present and all(value >= 6 for value in present):
            dominated.add(move)
    return dominated


def dominated_actor_indices(snapshot: dict) -> set[int]:
    """Return compact actor indices for currently dominated legal moves."""
    user = snapshot.get("user") or {}
    roster = user.get("pokemon") or []
    if not roster:
        return set()
    moves = sorted(
        norm(move.get("id", ""))
        for move in roster[0].get("moves", [])
        if norm(move.get("id", "")) not in {"", "none"}
    )
    dominated = dominated_move_names(snapshot)
    return {index for index, move in enumerate(moves) if move in dominated}


def _effective_speed(speed: object, boost: object, status: object) -> float | None:
    try:
        value = float(speed)
        stage = max(-6, min(6, int(boost)))
    except (TypeError, ValueError):
        return None
    value *= (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
    if norm(status) in {"par", "paralyze", "paralyzed", "paralysis"}:
        value /= 4
    return value


def guaranteed_unexecutable_actor_indices(snapshot: dict) -> set[int]:
    """Return move slots which cannot execute in the public Gen 1 state.

    This is a mechanics projection, not a strategic blacklist. A sleeping
    Pokémon consumes the turn even when it wakes in Gen 1. Likewise, an
    already-bound slower Pokémon cannot execute a selected move while an
    unimpaired faster trapping user continues the sequence. Switch indices
    begin at four and are intentionally never removed.
    """
    user = snapshot.get("user") or {}
    roster = user.get("pokemon") or []
    if not roster:
        return set()
    active = roster[0]
    moves = [
        move
        for move in active.get("moves", [])
        if norm(move.get("id", "")) not in {"", "none"}
    ]
    move_indices = set(range(len(moves)))
    if norm(active.get("status", "")) in {"slp", "sleep", "asleep"}:
        return move_indices

    volatiles = {norm(value) for value in user.get("volatiles", ()) or ()}
    if "partiallytrapped" not in volatiles:
        return set()
    opponent = (snapshot.get("opponent") or {}).get("active") or {}
    opponent_volatiles = {
        norm(value) for value in opponent.get("volatiles", ()) or ()
    }
    opponent_status = norm(opponent.get("status", ""))
    # Paralysis, sleep, freeze, or confusion can interrupt the trapping user,
    # so the victim's move is not guaranteed to fail in those states.
    if opponent_status in {
        "par", "paralyze", "paralyzed", "paralysis", "slp", "sleep",
        "asleep", "frz", "freeze", "frozen",
    } or "confusion" in opponent_volatiles:
        return set()
    own_speed = _effective_speed(
        active.get("speed"), (user.get("boosts") or {}).get("spe", 0), active.get("status")
    )
    foe_speed = _effective_speed(
        opponent.get("speed"),
        (opponent.get("boosts") or {}).get("spe", 0),
        opponent.get("status"),
    )
    if own_speed is not None and foe_speed is not None and own_speed < foe_speed:
        return move_indices
    return set()


def filter_probabilities(
    probabilities: list[float], legal_indices: list[int], snapshot: dict
) -> tuple[list[float], int]:
    """Remove semantic no-ops while preserving every legal stay-in choice.

    A move selected during sleep or a faster partial-trapping sequence cannot
    execute, but selecting it is still how the player chooses to remain active
    and advance the status/trapping turn. Search must simulate that failed
    execution; suppressing the choice would incorrectly force a switch.
    """
    if len(probabilities) != len(legal_indices):
        raise ValueError("probabilities and legal indices must align")
    blocked = dominated_actor_indices(snapshot)
    keep = [index not in blocked for index in legal_indices]
    if not any(keep):
        return probabilities, 0
    output = [value if allowed else 0.0 for value, allowed in zip(probabilities, keep)]
    total = sum(output)
    if total <= 0:
        count = sum(keep)
        output = [1.0 / count if allowed else 0.0 for allowed in keep]
    else:
        output = [value / total for value in output]
    return output, keep.count(False)
