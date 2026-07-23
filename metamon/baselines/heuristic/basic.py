import random
import warnings
from typing import Optional, List

import poke_env
from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.environment import (
    Pokemon,
    PokemonType,
    Move,
    MoveCategory,
    Battle,
    DoubleBattle,
    SideCondition,
    Weather,
    Status,
    Effect,
)

from metamon.baselines import register_baseline, GEN_DATA, Baseline


@register_baseline()
class RandomBaseline(Baseline):
    """
    picks a totally random move
    """

    def randomize(self):
        """
        Add extra diversity by randomizing some aspect of the agent's decision-making.
        """
        pass

    def choose_move(self, battle):
        """
        See poke-env documentation. The `metamon.env.ShowdownEnv` provides many examples
        of the kinds of information contained in the `battle` object.
        """
        return self.choose_random_move(battle)


@register_baseline()
class MaxBPBaseline(Baseline):
    """
    (usually) picks the move with the highest base power
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        if random.random() > self.random_action_prob:
            if battle.available_moves:
                best_move = max(
                    battle.available_moves, key=lambda move: move.base_power
                )
                return self.create_order(best_move)
        return self.choose_random_move(battle)


@register_baseline()
class PokeEnvHeuristic(Baseline, SimpleHeuristicsPlayer):
    """
    Calls the heuristic agent included in the poke-env repo
    """

    def randomize(self):
        pass

    def choose_move(self, battle):
        return super().choose_move(battle)


@register_baseline()
class Gen1Trainer(Baseline):
    def randomize(self):
        pass

    def choose_move(self, battle):
        if battle.force_switch and battle.available_switches:
            return self.create_order(battle.available_switches[0])

        elif battle.available_moves:
            # otherwise pick a random move (but doesn't randomly switch)
            return self.create_order(random.choice(battle.available_moves))

        return self.choose_random_move(battle)


@register_baseline()
class Gen1TrainerGoodSwitching(Baseline):
    def randomize(self):
        pass

    def choose_move(self, battle):
        if battle.force_switch and battle.available_switches:
            switch_scores = self.switch_scores(
                switches=battle.available_switches,
                battle=battle,
                def_type_disadvantage_w=1.0,
                off_type_advantage_w=1.0,
                speed_w=0.001,
            )
            best_switch = max(switch_scores, key=switch_scores.get)
            return self.create_order(best_switch)

        elif battle.available_moves:
            # otherwise pick a random move (but doesn't randomly switch)
            return self.create_order(random.choice(battle.available_moves))

        return self.choose_random_move(battle)


@register_baseline()
class Gen1BossAI(Baseline):
    """
    An interpretation of the "Good AI" from Generation 1

    Based on the info here:
    http://wiki.pokemonspeedruns.com/index.php/Pok%C3%A9mon_Red/Blue/Yellow_Trainer_AI
    """

    def randomize(self):
        pass

    def choose_move(self, battle):
        if battle.force_switch and battle.available_switches:
            return self.create_order(battle.available_switches[0])

        if battle.available_moves:
            opponent = battle.opponent_active_pokemon
            first_turn = battle.active_pokemon.first_turn
            opp_has_status = opponent.status is not None
            move_scores = {move: 10 for move in battle.available_moves}
            boost_scores = self.boost_move_scores(battle) if first_turn else None

            for move, score in move_scores.items():
                # "Modification 1"
                if (
                    move.category == MoveCategory.STATUS
                    and move.volatile_status is not None
                    and move.base_power == 0
                    and opp_has_status
                ):
                    score += 5
                # "Modification 2"
                if first_turn:
                    # in the actual games this apparently was bugged and happens on the 2nd turn.
                    # this logic actually hurts performance vs. Gen1Trainer, but I guess it gives
                    # the appearance of planning...
                    if (
                        boost_scores[move] != 0
                        or move.heal > 0
                        or move.id in ["reflect", "lightscreen", "transform"]
                    ):
                        score -= 1
                # "Modification 3"
                # there is a bug with dual-type effectiveness checks in Gen1 which we are ignoring
                effective = self.type_advantage(move.type, opponent, battle) >= 1
                score = score + (-1 if effective else 1)
                move_scores[move] = score

            lowest = min(move_scores.values())
            candidates = [m for m in move_scores if move_scores[m] == lowest]
            return self.force_use_gimmick(
                battle, self.create_order(random.choice(candidates))
            )

        return self.choose_random_move(battle)


@register_baseline()
class BasicSwitcher(Baseline):
    """
    a simple agent that knows its type charts.

    picks a pokemon from its party that has the best type advantage against the active opponent,
    and then picks moves with the highest base power adjusted for type matchups.
    """

    def randomize(self):
        pass

    def choose_move(self, battle):
        opponent = battle.opponent_active_pokemon

        if battle.available_switches:
            active_types = battle.active_pokemon.types
            active_type_adv = max(
                [self.type_advantage(t, opponent, battle) for t in active_types]
            )

            best_switch, best_switch_adv = None, -float("inf")
            for switch in battle.available_switches:
                switch_adv = max(
                    [self.type_advantage(t, opponent, battle) for t in switch.types]
                )
                if switch_adv > best_switch_adv:
                    best_switch = switch
                    best_switch_adv = switch_adv

            if best_switch_adv > active_type_adv:
                # found a pokemon with a better type advantage than our current choice, let's switch it in!
                return self.create_order(best_switch)

        if battle.available_moves:
            # pick a move while considering the type advantage we switched this pokemon in to exploit
            best_move = max(
                [
                    (
                        move,
                        move.base_power
                        * self.type_advantage(move.type, opponent, battle),
                    )
                    for move in battle.available_moves
                ],
                key=lambda x: x[1],
            )[0]
            return self.create_order(best_move)

        return self.choose_random_move(battle)


class SwitchToBestMove(Baseline):
    """
    switches to pokemon with best move, taking into account type advantage
    then uses the best move
    """

    def randomize(self):
        pass

    def choose_move(self, battle):
        opponent = battle.opponent_active_pokemon

        def get_best_move(
            pokemon, opponent, battle
        ) -> tuple[Optional[poke_env.environment.move.Move], float]:
            if not pokemon.moves:
                return None, -float("inf")

            return max(
                [
                    (
                        move,
                        move.base_power
                        * self.type_advantage(move.type, opponent, battle)
                        * (
                            1.5 if move.type in pokemon.types else 1.0
                        ),  # same type attack bonus
                    )
                    for move in pokemon.moves.values()
                ],
                key=lambda x: x[1],
            )

        active_best_move = get_best_move(battle.active_pokemon, opponent, battle)

        if battle.available_switches:
            best_switch, switch_best_move_power = None, -float("inf")
            for switch in battle.available_switches:
                best_move_power = get_best_move(switch, opponent, battle)[1]
                if best_move_power > switch_best_move_power:
                    best_switch = switch
                    switch_best_move_power = best_move_power

            if switch_best_move_power > active_best_move[1]:
                # found a pokemon with a better move!
                return self.create_order(best_switch)

        if active_best_move[0]:
            self.create_order(active_best_move[0])

        return self.choose_random_move(battle)


class RiskTaker(Baseline):
    """
    (usually) choses the move with the lowest accuracy
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        if random.random() > self.random_action_prob:
            if battle.available_moves:
                best_move = min(battle.available_moves, key=lambda move: move.accuracy)
                return self.create_order(best_move)
        return self.choose_random_move(battle)


class NotRiskTaker(Baseline):
    """
    prefers to choose high accuracy moves
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        if random.random() > self.random_action_prob:
            if battle.available_moves:
                best_move = max(battle.available_moves, key=lambda move: move.accuracy)
                return self.create_order(best_move)
        return self.choose_random_move(battle)


class PreferPhysical(Baseline):
    """
    (usually) picks the physcial move with the highest base power
    otherwise, picks any move with the highest base power
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        if random.random() > self.random_action_prob:
            physcial_moves = [
                move
                for move in battle.available_moves
                if move.category == MoveCategory.PHYSICAL
            ]

            moves = physcial_moves if physcial_moves else battle.available_moves

            if moves:
                best_move = max(moves, key=lambda move: move.base_power)
                return self.create_order(best_move)
        return self.choose_random_move(battle)


class PreferSpecial(Baseline):
    """
    (usually) picks the special move with the highest base power
    otherwise, picks any move with the highest base power
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        if random.random() > self.random_action_prob:
            special_moves = [
                move
                for move in battle.available_moves
                if move.category == MoveCategory.SPECIAL
            ]

            moves = special_moves if special_moves else battle.available_moves

            if moves:
                best_move = max(moves, key=lambda move: move.base_power)
                return self.create_order(best_move)
        return self.choose_random_move(battle)


class PreferStatus(Baseline):
    """
    (usually) picks the status move with the highest base power
    otherwise, picks any move with the highest base power
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        if random.random() > self.random_action_prob:
            status_moves = [
                move
                for move in battle.available_moves
                if move.category == MoveCategory.STATUS
            ]

            if status_moves:
                return self.create_order(random.choice(status_moves))

            if battle.available_moves:
                best_move = max(
                    battle.available_moves, key=lambda move: move.base_power
                )
                return self.create_order(best_move)
        return self.choose_random_move(battle)


class PreferPriority(Baseline):
    """
    picks high speed moves
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        if random.random() > self.random_action_prob:
            if battle.available_moves:
                best_move = max(battle.available_moves, key=lambda move: move.priority)
                return self.create_order(best_move)
        return self.choose_random_move(battle)


class PreferPrioritySmart(MaxBPBaseline):
    """
    picks fast moves when they can kill
    """

    def randomize(self):
        self.random_action_prob = random.uniform(0, 0.1)

    def choose_move(self, battle):
        user = battle.active_pokemon
        target = battle.opponent_active_pokemon

        if random.random() > self.random_action_prob:
            if battle.available_moves:
                killing_moves = [
                    move
                    for move in battle.available_moves
                    if self.damage_equation(
                        user,
                        move,
                        target,
                        battle,
                        critical_hit=False,
                        rng="random",
                        assume_stats="uniform",
                    )
                    > target.current_hp
                ]
                if killing_moves:
                    best_move = max(killing_moves, key=lambda move: move.priority)
                    return self.create_order(best_move)
        return super().choose_move(battle)


class _ParameterizedPokeEnvHeuristic(Baseline, SimpleHeuristicsPlayer):
    """
    Makes the poke-env heuristic agent exploitable by introducing randomness in its decision-making.
    Gives the poke-env heuristic agent parameters to control how likely it is to use certain strategies.
    p_dynamax: the probability of considering dynamaxing
    p_switch: the probability of considering switching
    p_setup_entry_hazard: the probability of considering setting up entry hazards
    p_remove_entry_hazard: the probability of considering removing entry hazards
    p_setup_move: the probability of considering setup moves
    p_consider_STAB: the probability of considering same type attack bonus in move selection
    p_consider_physical_special_ratio: the probability of considering the physical/special ratio in move selection
    p_consider_accuracy: the probability of considering accuracy in move selection
    p_consider_expected_hits: the probability of considering expected hits in move selection
    p_consider_damage_multiplier: the probability of considering damage multiplier in move selection
    ...
    """

    def __init__(
        self,
        *args,
        p_dynamax: float = 0.2,
        p_switch: float = 0.2,
        p_setup_entry_hazard: float = 0.2,
        p_remove_entry_hazard: float = 0.2,
        p_setup_move: float = 0.2,
        p_consider_STAB: float = 0.2,
        p_consider_physical_special_ratio: float = 0.2,
        p_consider_accuracy: float = 0.2,
        p_consider_expected_hits: float = 0.2,
        p_consider_damage_multiplier: float = 0.2,
        **kwargs,
    ):
        super(Baseline, self).__init__(*args, **kwargs)
        self.p_dynamax = p_dynamax
        self.p_switch = p_switch
        self.p_setup_entry_hazard = p_setup_entry_hazard
        self.p_remove_entry_hazard = p_remove_entry_hazard
        self.p_setup_move = p_setup_move
        self.p_consider_STAB = p_consider_STAB
        self.p_consider_physical_special_ratio = p_consider_physical_special_ratio
        self.p_consider_accuracy = p_consider_accuracy
        self.p_consider_expected_hits = p_consider_expected_hits
        self.p_consider_damage_multiplier = p_consider_damage_multiplier
        pass

    def randomize(self):
        pass

    def _should_dynamax(self, *args, **kwargs):
        if random.random() < self.p_dynamax:
            return super()._should_dynamax(*args, **kwargs)
        return False

    def _should_switch_out(self, *args, **kwargs):
        if random.random() < self.p_switch:
            return super()._should_switch_out(*args, **kwargs)
        return False

    def choose_move(self, battle):
        if isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)

        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Rough estimation of damage ratio
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(
            opponent, "def"
        )
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(
            opponent, "spd"
        )

        if battle.available_moves and (
            not self._should_switch_out(battle) or not battle.available_switches
        ):
            n_remaining_mons = len(
                [m for m in battle.team.values() if m.fainted is False]
            )
            n_opp_remaining_mons = 6 - len(
                [m for m in battle.opponent_team.values() if m.fainted is True]
            )

            should_setup_entry_hazard: bool = (
                random.random() < self.p_setup_entry_hazard
            )

            should_remove_entry_hazard: bool = (
                random.random() < self.p_remove_entry_hazard
            )

            # Entry hazard...
            for move in battle.available_moves:
                # ...setup
                if (
                    should_setup_entry_hazard
                    and n_opp_remaining_mons >= 3
                    and move.id in self.ENTRY_HAZARDS
                    and self.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions
                ):
                    return self.create_order(move)

                # ...removal
                elif (
                    should_remove_entry_hazard
                    and battle.side_conditions
                    and move.id in self.ANTI_HAZARDS_MOVES
                    and n_remaining_mons >= 2
                ):
                    return self.create_order(move)

            should_setup_move: bool = random.random() < self.p_setup_move

            # Setup moves
            if (
                should_setup_move
                and active.current_hp_fraction == 1
                and self._estimate_matchup(active, opponent) > 0
            ):
                for move in battle.available_moves:
                    if (
                        move.boosts
                        and sum(move.boosts.values()) >= 2
                        and move.target == "self"
                        and min(
                            [active.boosts[s] for s, v in move.boosts.items() if v > 0]
                        )
                        < 6
                    ):
                        return self.create_order(move)

            # same type attack bonus
            consider_STAB: bool = random.random() < self.p_consider_STAB

            consider_physical_special_ratio: bool = (
                random.random() < self.p_consider_physical_special_ratio
            )

            consider_accuracy: bool = random.random() < self.p_consider_accuracy

            consider_expected_hits: bool = (
                random.random() < self.p_consider_expected_hits
            )

            consider_damage_mult: bool = (
                random.random() < self.p_consider_damage_multiplier
            )

            move = max(
                battle.available_moves,
                key=lambda m: m.base_power
                * (1.5 if m.type in active.types and consider_STAB else 1)
                * (
                    (
                        physical_ratio
                        if m.category == MoveCategory.PHYSICAL
                        else special_ratio
                    )
                    if consider_physical_special_ratio
                    else 1
                )
                * (m.accuracy if consider_accuracy else 1)
                * (m.expected_hits if consider_expected_hits else 1)
                * (opponent.damage_multiplier(m) if consider_damage_mult else 1),
            )
            return self.create_order(
                move, dynamax=self._should_dynamax(battle, n_remaining_mons)
            )

        if battle.available_switches:
            switches: List[Pokemon] = battle.available_switches
            return self.create_order(
                max(
                    switches,
                    key=lambda s: self._estimate_matchup(s, opponent),
                )
            )

        return self.choose_random_move(battle)


class EasyPokeEnvHeuristic(_ParameterizedPokeEnvHeuristic):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            p_dynamax=0.1,
            p_switch=0.1,
            p_setup_entry_hazard=0.1,
            p_remove_entry_hazard=0.1,
            p_setup_move=0.5,
            p_consider_STAB=0.1,
            p_consider_physical_special_ratio=0.2,
            p_consider_accuracy=0.1,
            p_consider_expected_hits=0.2,
            p_consider_damage_multiplier=0.3,
            **kwargs,
        )


class GruntRandomSwitching(Baseline):

    def randomize(self):
        pass

    def choose_move(self, battle):
        user = battle.active_pokemon
        target = battle.opponent_active_pokemon

        if battle.force_switch and battle.available_switches:
            return self.create_order(random.choice(battle.available_switches))

        if battle.available_moves:
            move = max(
                battle.available_moves,
                key=lambda m: self.damage_roll(
                    user=user, move=m, target=target, battle=battle
                ),
            )
            return self.force_use_gimmick(battle, self.create_order(move))
        return self.choose_random_move(battle)


@register_baseline()
class Grunt(Baseline):
    """
    Simple but decent baseline that knows the damage equation
    and type chart. Picks the highest damage move and the best
    type matchup when forced to switch.
    """

    def randomize(self):
        pass

    def choose_move(self, battle):
        user = battle.active_pokemon
        target = battle.opponent_active_pokemon

        if battle.force_switch and battle.available_switches:
            switch_scores = self.switch_scores(
                switches=battle.available_switches,
                battle=battle,
                def_type_disadvantage_w=1.0,
                off_type_advantage_w=1.0,
            )
            best_switch = max(switch_scores, key=switch_scores.get)
            return self.create_order(best_switch)

        if battle.available_moves:
            move = max(
                battle.available_moves,
                key=lambda m: self.damage_roll(
                    user=user, move=m, target=target, battle=battle
                ),
            )
            return self.force_use_gimmick(battle, self.create_order(move))
        return self.choose_random_move(battle)


@register_baseline()
class GymLeader(Baseline):
    def randomize(self):
        pass

    def choose_move(self, battle):
        user = battle.active_pokemon
        target = battle.opponent_active_pokemon

        if (
            battle.force_switch or self.should_emergency_switch(battle, False, False)
        ) and battle.available_switches:
            # outspeed KO switch
            outspeed_ko = self.find_outspeed_ko_switch(battle, assume_worst=True)
            if outspeed_ko is not None:
                return self.create_order(outspeed_ko)

            # regular type switch with ties broken by speed
            switch_scores = self.switch_scores(
                switches=battle.available_switches,
                battle=battle,
                def_type_disadvantage_w=1.0,
                off_type_advantage_w=1.0,
                speed_w=0.001,
            )
            best_switch = max(switch_scores, key=switch_scores.get)
            return self.create_order(best_switch)

        if battle.available_moves:
            # knockout move
            ko_move = self.find_ko_move(user, target, battle, assume_worst=True)
            if ko_move is not None and ko_move in battle.available_moves:
                return self.create_order(ko_move)

            if user.first_turn and user.current_hp_fraction > 0.9:
                # setup move
                boost_scores = self.boost_move_scores(battle)
                max_boost_move = max(boost_scores, key=boost_scores.get)
                if boost_scores[max_boost_move] > 1 and random.random() < 0.5:
                    return self.create_order(max_boost_move)

            if user.current_hp_fraction < 0.33:
                # heal move
                heal_scores = self.heal_move_scores(battle)
                max_heal_move = max(heal_scores, key=heal_scores.get)
                if heal_scores[max_heal_move] > 0.33 and random.random() < 0.75:
                    return self.create_order(max_heal_move)

            # highest damaging move
            move = max(
                battle.available_moves,
                key=lambda m: self.expected_damage(
                    user=user, move=m, target=target, battle=battle
                ),
            )
            return self.force_use_gimmick(battle, self.create_order(move))

        return self.choose_random_move(battle)


@register_baseline()
class SmogonSwitcher(Grunt):
    def switch_scores(self, *args, **kwargs):
        if "check_w" in kwargs:
            kwargs["check_w"] = 100.0
        else:
            kwargs["check_w"] = 100.0
        return super().switch_scores(*args, **kwargs)


@register_baseline()
class BugCatcher(Baseline):
    """
    An actively bad trainer that always picks the least
    damaging move. When forced to switch, picks the pokemon
    in its party with the worst type matchup vs the player.
    """

    def randomize(self):
        pass

    def choose_move(self, battle):
        user = battle.active_pokemon
        target = battle.opponent_active_pokemon

        if battle.force_switch and battle.available_switches:
            # if we have to switch, pick the worst matchup considering types
            switch_scores = self.switch_scores(
                switches=battle.available_switches,
                battle=battle,
                def_type_disadvantage_w=1.0,
                off_type_advantage_w=1.0,
            )
            worst_switch = min(switch_scores, key=switch_scores.get)
            return self.create_order(worst_switch)
        if battle.available_moves:
            # if we can move, pick the least damaging move
            move = min(
                battle.available_moves,
                key=lambda m: self.damage_roll(
                    user=user, move=m, target=target, battle=battle
                ),
            )
            return self.create_order(move)

        return self.choose_random_move(battle)
