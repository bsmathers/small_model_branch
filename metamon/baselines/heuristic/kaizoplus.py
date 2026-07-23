import random

import poke_env
from poke_env.player import Player
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.move import Move
from poke_env.environment.effect import Effect
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.weather import Weather
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status

from metamon.baselines import register_baseline, Baseline


@register_baseline()
class KaizoPlus(Baseline):
    def randomize(self):
        pass

    @property
    def _switch_score_kwargs(self):
        return dict(
            check_w=9.0,
            def_type_disadvantage_w=1.0,
            off_type_advantage_w=1.0,
            speed_w=0.001,
        )

    # trimmed down kaizo to bump boosts, heals, hazards, etc
    # with score modification from smogon + damage calcs
    def score_moves(self, battle: Battle):
        # variables for manual scoring
        rng = random.random
        user = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        user_hp = user.current_hp_fraction
        opp_hp = opp.current_hp_fraction
        user_last_move = user.previous_move if not user.first_turn else None
        opp_last_move = opp.previous_move if not opp.first_turn else None
        highest_damage_move = max(
            battle.available_moves,
            key=lambda m: self.damage_equation(
                user,
                m,
                opp,
                battle,
                critical_hit=False,
                rng="max",
                assume_stats="uniform",
            ),
        )
        opp_highest_damage_move = (
            None
            if len(opp.moves) == 0
            else max(
                opp.moves.values(),
                key=lambda m: self.damage_equation(
                    opp,
                    m,
                    user,
                    battle,
                    critical_hit=False,
                    rng="max",
                    assume_stats="uniform",
                ),
            )
        )
        boost_scores = self.boost_move_scores(battle, speed_w=0.5)
        max_boost_move = max(boost_scores, key=boost_scores.get)
        opp_has_hazards = False
        user_has_hazards = False
        hazards = [
            SideCondition.SPIKES,
            SideCondition.TOXIC_SPIKES,
            SideCondition.STEALTH_ROCK,
            SideCondition.STICKY_WEB,
        ]
        for hazard in hazards:
            if hazard in battle.side_conditions:
                user_has_hazards = True
            if hazard in battle.opponent_side_conditions:
                opp_has_hazards = True

        user_has_prot_det = any(
            m.id in ["protect", "detect"] for m in battle.available_moves
        )

        # for u-turn and baton pass
        has_better_switch = False
        if battle.available_switches:
            switch_scores = self.switch_scores(
                switches=battle.available_switches,
                battle=battle,
                **self._switch_score_kwargs,
            )
            active_score = self.switch_scores(
                switches=[battle.active_pokemon],
                battle=battle,
                **self._switch_score_kwargs,
            )[battle.active_pokemon]
            best_switch = max(switch_scores, key=switch_scores.get)
            has_better_switch = active_score < switch_scores[best_switch]

        should_emergency_switch = self.should_emergency_switch(battle, False, False)

        move_scores = {}
        for move in battle.available_moves:
            type_adv_mult = self.type_advantage(move.type, opp, battle)
            ineffective = type_adv_mult == 0
            resists_move = type_adv_mult < 1
            effective = type_adv_mult > 1
            exp_damage = self.expected_damage(user, move, opp, battle)
            worst_case_damage = self.worst_case_damage(user, move, opp, battle)
            exp_can_kill = exp_damage >= opp.current_hp
            always_can_kill = worst_case_damage >= opp.current_hp
            self_destruct = move.self_destruct is not None
            self_destruct_loses = len(battle.available_switches) == 0
            slower = self.outspeed_chance(user, opp, battle) <= 0.5
            faster = not slower
            priority = move.priority > 0 and move.id != "fakeout"
            is_boost = {}
            is_boost["atk"] = (
                move.boosts
                and sum(move.boosts.values()) >= 2
                and "atk" in move.boosts
                and move.boosts["atk"] > 0
            )
            is_boost["def"] = (
                move.boosts
                and sum(move.boosts.values()) >= 2
                and "def" in move.boosts
                and move.boosts["def"] > 0
            )
            is_boost["spa"] = (
                move.boosts
                and sum(move.boosts.values()) >= 2
                and "spa" in move.boosts
                and move.boosts["spa"] > 0
            )
            is_boost["spd"] = (
                move.boosts and "spd" in move.boosts and move.boosts["spd"] > 0
            )

            score = 0
            if always_can_kill:
                score = 8 + (2 if priority else 0) if not self_destruct else -1
                if move == highest_damage_move:
                    score += 1
            elif exp_can_kill:
                score = 4 + (2 if priority else 0) if not self_destruct else -1
                if move == highest_damage_move:
                    score += 1
            elif move.category == MoveCategory.STATUS:
                score = 0
            elif move == highest_damage_move:
                score = 1
            elif move.base_power > 1 and move != highest_damage_move:
                score = -1
            elif ineffective or (self_destruct and self_destruct_loses):
                score = -12

            # don't bother with other stuff, kill asap if opponent is boosted
            if move == highest_damage_move and (
                opp.boosts["atk"] >= 3
                or opp.boosts["spa"] >= 3
                or opp.boosts["def"] >= 3
                or opp.boosts["spd"] >= 3
            ):
                score += 2

            # note: our other baselines rarely use sleep, so this doesn't actually do much lol
            if user.status == Status.SLP and not move.sleep_usable:
                score = -12

            # begin manual scoring

            always_hit_moves = [
                "aerialace",
                "aurasphere",
                "clearsmog",
                "falsesurrender" "feintattack",
                "faintattack",
                "magicalleaf",
                "magnetbomb",
                "shadowpunch",
                "shockwave",
                "swift",
                "vitalthrow",
                "trumpcard",
            ]
            if move.id in always_hit_moves:
                if user.boosts["accuracy"] <= -3 and rng() < 0.61:
                    score += 1
                if user.boosts["accuracy"] <= -5:
                    score += 2

            if is_boost["atk"]:
                if user_hp == 1:
                    if user.boosts["atk"] <= 2 and move == max_boost_move:
                        score += 4 if rng() < 0.5 else 3
                elif user_hp > 70:
                    pass
                elif user_hp > 40:
                    if rng() < 0.84:
                        score -= 2
                else:
                    score -= 2

            if is_boost["spa"]:
                if user_hp == 1:
                    if user.boosts["spa"] <= 2 and move == max_boost_move:
                        score += 4 if rng() < 0.5 else 3
                elif user_hp > 70:
                    pass
                elif user_hp > 40:
                    if rng() < 0.84:
                        score -= 2
                else:
                    score -= 2

            if is_boost["def"]:
                if user_hp == 1 and user.boosts["def"] <= 2 and move == max_boost_move:
                    score += 4 if rng() < 0.5 else 3

                if user.boosts["def"] >= 3 and rng() < 0.61:
                    score -= 1

                if user_hp <= 0.4:
                    score -= 2
                elif user_hp < 0.7 or rng() < 0.22:
                    final_check_chance = 1
                    if opp_last_move:
                        if opp_last_move.category == MoveCategory.PHYSICAL:
                            final_check_chance = 0.59
                        elif opp_last_move.category == MoveCategory.STATUS:
                            final_check_chance = 0.77
                    else:
                        final_check_chance = 0.7

                    if rng() < final_check_chance:
                        score -= 2

            if is_boost["spd"]:
                if user_hp == 1:
                    if slower and user.boosts["spd"] <= 2 and move == max_boost_move:
                        score += 4 if rng() < 0.5 else 3
                elif user_hp < 0.4:
                    score -= 2
                elif user_hp > 0.7:
                    if rng() < 0.22:
                        score -= 2
                elif rng() < 0.7:
                    score -= 2

            speed_lower_moves = [
                "bubble",
                "bubblebeam",
                "bulldoze",
                "cottonspore",
                "drumbeating",
                "electroweb",
                "glaciate",
                "lowsweep",
                "pounce",
                "icywind",
                "mudshot",
                "rocktomb",
                "stringshot",
                "scaryface",
                "syrupbomb",
                "tarshot",
                "toxicthread",
            ]
            if move.id in speed_lower_moves:
                if slower:
                    if rng() < 0.5:
                        score += 1
                else:
                    score -= 3

            hp_res_moves = [
                "milkdrink",
                "softboiled",
                "moonlight",
                "morningsun",
                "recover",
                "slackoff",
                "swallow",
                "synthesis",
                "roost",
                "healorder",
                "shoreup",
                "lunarblessing",
                "wish",
            ]
            if move.id in hp_res_moves or move.heal > 0.2:
                if user_hp == 1:
                    score -= 3
                elif user_hp > 0.5:
                    if rng() < 0.88:
                        score -= 3
                elif user_hp > 0.33:
                    if rng() < 0.5:
                        score += 2
                else:
                    score += 3

            confusing_moves = [
                "confuseray",
                "supersonic",
                "dynamicpunch",
                "shadowpanic",
                "teeterdance",
                "sweetkiss",
                "flatter",
                "swagger",
                "hurricane",
            ]
            if move.id in confusing_moves:
                if move.id == "flatter" or move.id == "swagger":
                    if rng() < 0.5:
                        score += 1
                if opp_hp <= 0.7:
                    if rng() < 0.5:
                        score -= 1
                if opp_hp <= 0.5:
                    if rng() < 0.5:
                        score -= 1
                if opp_hp <= 0.3:
                    if rng() < 0.5:
                        score -= 1

            par_moves = ["glare", "stunspore", "thunderwave"]
            if move.id in par_moves:
                if slower:
                    if Status.PAR != opp.status and rng() < 0.5:
                        score += 2
                else:
                    if user_hp < 0.7:
                        score -= 1

            sleep_moves = [
                "grasswhistle",
                "hypnosis",
                "lovelykiss",
                "sing",
                "sleeppowder",
                "yawn",
            ]
            if move.id in sleep_moves and opp.status != Status.SLP:
                has_dream = False

                for m in battle.available_moves:
                    if m.id == "dreameater" or m.id == "nightmare":
                        has_dream = True

                if has_dream and rng() < 0.5:
                    score += 2

            if move.id == "dreameater" or move.id == "nightmare":
                if opp.status == Status.SLP:
                    if rng() < 0.5:
                        score += 2

            if move.id == "flail" or move.id == "reversal":
                if slower:
                    if user_hp > 0.6:
                        score -= 1
                    elif user_hp > 0.4:
                        pass
                    else:
                        score += 1
                else:
                    if user_hp > 0.33:
                        score -= 1
                    elif user_hp > 0.08:
                        pass
                    else:
                        score += 1

            if move.id == "stealthrock":
                if (
                    PokemonType.FLYING not in opp.types
                    and SideCondition.STEALTH_ROCK
                    not in battle.opponent_side_conditions
                ):
                    if rng() < 0.5:
                        score += 2
                else:
                    score -= 2

            if move.id == "spikes":
                if PokemonType.FLYING not in opp.types and (
                    SideCondition.SPIKES not in battle.opponent_side_conditions
                    or battle.opponent_side_conditions[SideCondition.SPIKES] < 3
                ):
                    if rng() < 0.2:
                        score += 2
                else:
                    score -= 2

            if move.id == "toxicspikes":
                if (
                    PokemonType.FLYING not in opp.types
                    and PokemonType.POISON not in opp.types
                    and PokemonType.STEEL not in opp.types
                    and (
                        SideCondition.TOXIC_SPIKES
                        not in battle.opponent_side_conditions
                        or battle.opponent_side_conditions[SideCondition.TOXIC_SPIKES]
                        < 2
                    )
                ):
                    if rng() < 0.2:
                        score += 2
                else:
                    score -= 2

            if move.id == "defog":
                if opp_has_hazards and rng() < 0.8:
                    score -= 1
                if user_has_hazards and (
                    user_hp < 0.5 or has_better_switch or should_emergency_switch
                ):
                    if rng() < 0.8:
                        score += 2
                if (
                    SideCondition.LIGHT_SCREEN in battle.opponent_side_conditions
                    or SideCondition.REFLECT in battle.opponent_side_conditions
                ):
                    score += 1

            if move.id == "encore":
                if slower:
                    score -= 2
                else:
                    encore_moves = [
                        "attract",
                        "camouflage",
                        "charge",
                        "confuseray",
                        "conversion",
                        "conversion2",
                        "detect",
                        "dreameater",
                        "encore",
                        "endure",
                        "fakeout",
                        "followme",
                        "foresight",
                        "glare",
                        "growth",
                        "harden",
                        "haze",
                        "healbell",
                        "imprison",
                        "ingrain",
                        "knockoff",
                        "lightscreen",
                        "meanlook",
                        "mudsport",
                        "poisonpowder",
                        "protect",
                        "recycle",
                        "refresh",
                        "rest",
                        "roar",
                        "roleplay",
                        "safeguard",
                        "skillswap",
                        "stunspore",
                        "superfang",
                        "supersonic",
                        "swagger",
                        "sweetkiss",
                        "teeterdance",
                        "thief",
                        "thunderwave",
                        "toxic",
                        "watersport",
                        "willowisp",
                    ]

                    if opp_last_move:
                        if opp_last_move.id in encore_moves:
                            if rng() < 0.88:
                                score += 3
                        else:
                            score -= 2
                    else:
                        score -= 2

            if move.id == "batonpass":
                hp_thresh = 0.7 if slower else 0.6
                if (
                    user.boosts["atk"] >= 3
                    or user.boosts["spa"] >= 3
                    or user.boosts["def"] >= 3
                    or user.boosts["spd"] >= 3
                    or user.boosts["evasion"] >= 3
                ):
                    if user_hp < hp_thresh and has_better_switch:
                        score += 3
                elif (
                    user.boosts["atk"] >= 2
                    or user.boosts["spa"] >= 2
                    or user.boosts["def"] >= 2
                    or user.boosts["spd"] >= 2
                    or user.boosts["evasion"] >= 2
                ):
                    if user_hp > hp_thresh:
                        score -= 2
                else:
                    score -= 2

            if move.id == "taunt":
                if ineffective and rng() < 0.5:
                    score -= 1
                if (
                    opp_last_move
                    and self.type_advantage(opp_last_move.type, user, battle) > 1
                    and rng() < 0.5
                ):
                    score -= 1

            if move.id == "uturn":
                if highest_damage_move != move and not has_better_switch:
                    score -= 2
                if user_has_hazards:
                    score -= 1
                if has_better_switch and slower:
                    score += 1
                if should_emergency_switch:
                    score += 2
                if user_hp < 0.4 and rng() < 0.3:
                    score += 1

            if move.id == "rapidspin":
                if user_has_hazards and (
                    user_hp < 0.5 or has_better_switch or should_emergency_switch
                ):
                    score += 3 if rng() < 0.5 else 1

            if move.id == "sleeptalk":
                if user.status == Status.SLP:
                    score += 10
                else:
                    score -= 5

            if move.id == "snore":
                if user.status == Status.SLP:
                    score += 8
                else:
                    score -= 5

            if move.id == "lightscreen" or move.id == "reflect":
                if user_hp < 0.9:
                    score -= 2
                elif opp_last_move:
                    thresh = 0.8 if faster else 0.5
                    opp_best_attack_type = None
                    if opp_highest_damage_move:
                        opp_best_attack_type = opp_highest_damage_move.category

                    if (
                        move.id == "reflect"
                        and opp_best_attack_type == MoveCategory.PHYSICAL
                        and not SideCondition.REFLECT in battle.side_conditions
                    ):
                        if rng() < thresh:
                            score += 2
                    elif (
                        move.id == "lightscreen"
                        and opp_best_attack_type == MoveCategory.SPECIAL
                        and not SideCondition.LIGHT_SCREEN in battle.side_conditions
                    ):
                        if rng() < thresh:
                            score += 2

            recharge_moves = ["blastburn", "frenzyplant", "hydrocannon", "hyperbeam"]
            if move.id in recharge_moves:
                if resists_move:
                    score -= 1
                else:
                    hp_thresh = 0.6 if slower else 0.41
                    if user_hp > hp_thresh:
                        score -= 1

            if move.id == "revenge":
                if (
                    user.status == Status.SLP
                    or Effect.CONFUSION in user.effects
                    or rng() < 0.73
                ):
                    score -= 2
                else:
                    score += 2

            trap_moves = [
                "sandtomb",
                "whirlpool",
                "wrap",
                "meanlook",
                "spiderweb",
                "anchorshot",
                "spiritshackle",
                "jawlock",
                "shadowhold",
                "thousandwaves",
                "block",
                "fairylock",
            ]
            if move.id in trap_moves:
                if opp.status == Status.TOX and not (
                    (move.id == "fairylock" or move.id == "thousandwaves")
                    and PokemonType.GHOST in opp.types
                ):
                    if rng() < 0.5:
                        score += 1

            if move.id == "watersport":
                if user_hp > 0.5 and PokemonType.FIRE in opp.types:
                    score += 2
                else:
                    score -= 1

            if move.id == "raindance":
                has_swim = False
                for m in battle.available_moves:
                    if m.id == "swiftswim":
                        has_swim = True
                if has_swim and slower:
                    score += 1

                if user_hp < 0.4:
                    score -= 1
                elif battle.weather != Weather.RAINDANCE:
                    score += 1

            if move.id == "hail":
                if user_hp < 0.4:
                    score -= 1
                elif battle.weather != Weather.HAIL:
                    score += 1

            if move.id == "sunnyday":
                if user_hp < 0.4:
                    score -= 1
                elif battle.weather != Weather.SUNNYDAY:
                    score += 1

            if move.id == "earthquake":
                if opp_last_move and opp_last_move.id == "dig" and faster:
                    score += 1
                    if rng() < 0.5:
                        score += 1

            if move.id == "earthpower":
                if opp_hp == 1:
                    if opp.boosts["spd"] > -3 and rng() < 0.5:
                        score += 1
                if opp_hp > 0.7:
                    if opp.boosts["spd"] > -3 and rng() < 0.3:
                        score += 1

                if opp.boosts["spd"] <= -2 and rng() < 0.6:
                    score -= 1

            if move.id == "substitute":
                if user_hp < 0.5 and rng() < 0.61:
                    score -= 1
                if user_hp < 0.7 and rng() < 0.61:
                    score -= 1
                if user_hp < 0.9 and rng() < 0.61:
                    score -= 1

            move_scores[move] = score

        return move_scores

    def should_emergency_switch(
        self, battle: Battle, must_be_slower: bool, must_be_low_hp: bool
    ) -> bool:
        # additional emergency switch logic based off PokeEnvHeuristic
        active = battle.active_pokemon
        if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
            return True
        if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
            return True
        if active.boosts["spa"] <= -3 and active.stats["atk"] <= active.stats["spa"]:
            return True
        return super().should_emergency_switch(battle, must_be_slower, must_be_low_hp)

    def choose_move(self, battle: Battle):
        using_best_matchup = True

        if battle.available_switches:
            switch_scores = self.switch_scores(
                switches=battle.available_switches,
                battle=battle,
                **self._switch_score_kwargs,
            )
            active_score = self.switch_scores(
                switches=[battle.active_pokemon],
                battle=battle,
                **self._switch_score_kwargs,
            )[battle.active_pokemon]
            best_switch = max(switch_scores, key=switch_scores.get)

            if active_score < switch_scores[best_switch]:
                using_best_matchup = False

            if battle.force_switch or (
                self.should_emergency_switch(battle, False, False)
                and not using_best_matchup
            ):
                return self.create_order(best_switch)

        if battle.available_moves:
            move_scores = self.score_moves(battle)
            best_move = max(move_scores, key=move_scores.get)

            order = self.create_order(best_move)

            # rough dynamax logic
            if (
                using_best_matchup
                and battle.can_dynamax
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                order.dynamax = True

            # note: additional gimmick scoring could also be considered, but might be hard to score for
            # also they're banned in ou anyways
            return self.force_use_gimmick(battle, order)

        return self.choose_random_move(battle)
