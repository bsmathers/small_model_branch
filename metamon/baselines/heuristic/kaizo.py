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
class EmeraldKaizo(Baseline):
    """
    Based on Emerald Kaizo AI, with bug fixes and modifications
    """

    def randomize(self):
        pass

    def kaizo_score_moves(self, battle: Battle, risky: bool = False):
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

        # calculate some stuff for later move scoring
        user_has_damaging = any(m.base_power > 1 for m in battle.available_moves)
        user_has_prot_det = any(
            m.id in ["protect", "detect"] for m in battle.available_moves
        )

        physical_types = [
            PokemonType.NORMAL,
            PokemonType.FIGHTING,
            PokemonType.FLYING,
            PokemonType.POISON,
            PokemonType.GROUND,
            PokemonType.GHOST,
            PokemonType.BUG,
            PokemonType.STEEL,
        ]

        user_only_physical = True
        user_only_special = True

        for type in user.types:
            if type in physical_types:
                user_only_special = False
            else:
                user_only_physical = False

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
                move.boosts and "atk" in move.boosts and move.boosts["atk"] > 0
            )
            is_boost["def"] = (
                move.boosts and "def" in move.boosts and move.boosts["def"] > 0
            )
            is_boost["spa"] = (
                move.boosts and "spa" in move.boosts and move.boosts["spa"] > 0
            )
            is_boost["spd"] = (
                move.boosts and "spd" in move.boosts and move.boosts["spd"] > 0
            )

            score = 0
            if always_can_kill:
                score = 5 + (2 if priority else 0) if not self_destruct else -1
                if move == highest_damage_move:
                    score += 1
            elif exp_can_kill:
                score = 4 + (2 if priority else 0) if not self_destruct else -1
                if move == highest_damage_move:
                    score += 1
            elif move.category == MoveCategory.STATUS:
                score = 2 if rng() < 0.7 else 0
            elif move == highest_damage_move:
                score = 2 if rng() < 0.7 else 0
            elif move.base_power > 1 and move != highest_damage_move:
                score = -1
            elif ineffective or (self_destruct and self_destruct_loses):
                score = -12

            # need to confirm move ids, i think they're right though
            if move.id == "skillswap":
                swap_moves = [
                    "battlearmor",
                    "chlorophyll",
                    "cutecharm",
                    "effectspore",
                    "hugepower",
                    "marvelscale",
                    "purepower",
                    "raindish",
                    "sandveil",
                    "shedskin",
                    "shielddust",
                    "speedboost",
                    "static",
                    "swift swim",
                    "wonderguard",
                ]

                user_has_move = any(m.id in swap_moves for m in battle.available_moves)
                # should check against what we've seen the opponent use
                # but that is very hard to do in poke env, so we cheat
                opp_has_move = any(m.id in swap_moves for m in opp.moves.values())
                if user_has_move or not opp_has_move:
                    score -= 1
                if opp_has_move and rng() < 0.8:
                    score += 2

            always_hit_moves = [
                "aerialace",
                "aurasphere",
                "clearsmog",
                "falsesurrender" "feintattack",
                "faintattack",  # they changed the name lol
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
                    score += 1

            atk_boost_moves = [
                "coil",
                "honeclaws",
                "howl",
                "swordsdance",
                "meditate",
                "sharpen",
                "shiftgear",
                "workup",
                "meteormash",
            ]
            if move.id in atk_boost_moves or is_boost["atk"]:
                if user_hp == 1:
                    if user.boosts["atk"] <= 2 and rng() < 0.5:
                        score += 1
                elif user_hp > 70:
                    pass
                elif user_hp > 40:
                    if rng() < 0.84:
                        score -= 2
                else:
                    score -= 2

            spa_boost_moves = [
                "nastyplot",
                "torchsong",
                "chargebeam",
                "fierydance",
                "geomancy",
                "growth",
                "maxooze" "quiverdance",
                "workup",
            ]
            if move.id in spa_boost_moves or is_boost["spa"]:
                if user_hp == 1:
                    if user.boosts["spa"] <= 2 and effective and rng() < 0.5:
                        score += 1
                elif user_hp > 70:
                    pass
                elif user_hp > 40:
                    if rng() < 0.84:
                        score -= 2
                else:
                    score -= 2

            if move.id == "charm":
                if opp.boosts["atk"] < 0:
                    score -= 1
                    if user_hp < 0.9:
                        score -= 1
                    if opp.boosts["atk"] < -3 and rng() < 0.8:
                        score -= 1
                else:
                    if user_hp < 0.7:
                        score -= 2
                    else:
                        opp_has_physical = False
                        for type in opp.types:
                            if type in physical_types:
                                opp_has_physical = True

                        if not opp_has_physical:
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
                    if user_hp < hp_thresh and rng() < 0.69:
                        score += 2
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

            if move.id == "bellydrum":
                if user_hp < 0.9:
                    score -= 2

            if move.id == "brickbreak":
                if Effect.REFLECT in user.effects:
                    score += 1

            if move.id == "solarbeam":
                if user_has_prot_det or resists_move:
                    score -= 2
                if user_hp < 0.38:
                    score -= 1

            # alluring voice can be added
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

            if move.id == "conversion":
                if user_hp < 0.9 or rng() < 0.8:
                    score -= 2

            if move.id == "counter" or move.id == "mirrorcoat":
                if opp.status == Status.SLP or Effect.CONFUSION in opp.effects:
                    score -= 1
                else:
                    if user_hp < 0.3 and rng() < 0.96:
                        score -= 1
                    if user_hp < 0.5 and rng() < 0.61:
                        score -= 1

                    has_counter = False
                    has_mirrorcoat = False
                    for m in battle.available_moves:
                        if m.id == "counter":
                            has_counter = True
                        if m.id == "mirrorcoat":
                            has_mirrorcoat = True

                    # weird kaizo special logic for wobbuffet
                    if has_counter and has_mirrorcoat and rng() < 0.61:
                        score += 4

                    if not (has_counter and has_mirrorcoat) and opp_last_move:
                        prev_cat = opp_last_move.category
                        if prev_cat == MoveCategory.STATUS:
                            if move.id == "mirrorcoat":
                                if user_only_special and rng() < 0.5:
                                    score += 4
                            else:
                                if user_only_physical and rng() < 0.5:
                                    score += 4
                        elif prev_cat == MoveCategory.PHYSICAL:
                            if move.id == "mirrorcoat":
                                score -= 1
                            elif rng() < 0.61:
                                score += 1
                        elif prev_cat == MoveCategory.SPECIAL:
                            if move.id == "counter":
                                score -= 1
                            elif rng() < 0.61:
                                score += 1

            if move.id == "curse":
                if PokemonType.GHOST in user.types:
                    if rng() < 0.69:
                        score -= 1
                else:
                    if user.boosts["def"] >= 3 and rng() < 0.5:
                        score += 1
                    if user.boosts["def"] >= 2 and rng() < 0.5:
                        score += 1
                    if user.boosts["def"] >= 1 and rng() < 0.5:
                        score += 1
                    if rng() < 0.5:
                        score += 1

            defense_boost_moves = [
                "acidarmor",
                "barrier",
                "bulkup",
                "cosmicpower",
                "cottonguard",
                "defendorder",
                "defensecurl",
                "diamondstorm",
                "harden",
                "irondefense",
                "shelter",
                "skullbash",
                "withdraw",
            ]
            if move.id in defense_boost_moves or is_boost["def"]:
                if user_hp == 1 and user.boosts["def"] <= 2 and rng() < 0.5:
                    score += 2

                # idk why it checks special defense but it does
                if user.boosts["spd"] >= 3 and rng() < 0.61:
                    score -= 1

                if user_hp <= 0.4:
                    score -= 2
                elif user_hp < 0.7 or rng() < 0.22:
                    final_check_chance = 1
                    if not opp.first_turn and opp_last_move:
                        if opp_last_move.category == MoveCategory.PHYSICAL:
                            final_check_chance = 0.59
                        elif opp_last_move.category == MoveCategory.STATUS:
                            final_check_chance = 0.77
                    else:
                        final_check_chance = 0.7

                    if rng() < final_check_chance:
                        score -= 2

            if move.id == "tickle":
                if user_hp < 0.7:
                    score -= 2

                # tickle lowers attack as well as defense
                if (user_hp < 0.7 or opp.boosts["atk"] <= -3) and rng() < 0.8:
                    score -= 2

            if move.id == "moonblast":
                if user_hp < 0.5 and rng() < 0.5:
                    score -= 2

                if (user_hp < 0.5 or opp.boosts["spa"] <= -3) and rng() < 0.8:
                    score -= 2

            if move.id == "destinybond":
                score -= 1

                if faster:
                    if user_hp < 0.7:
                        if rng() < 0.5:
                            score += 1
                    if user_hp < 0.5:
                        if rng() < 0.5:
                            score += 1
                    if user_hp < 0.3:
                        if rng() < 0.61:
                            score += 2

            if move.id == "disable":
                if opp_last_move:
                    if faster:
                        if opp_last_move.category == MoveCategory.STATUS:
                            if rng() < 0.61:
                                score -= 1
                        else:
                            score += 1

            if move.id == "dragondance":
                if slower:
                    if rng() < 0.5:
                        score += 1
                else:
                    if user_hp < 0.5 and rng() < 0.73:
                        score -= 1

            draining_moves = [
                "absorb",
                "drainpunch",
                "drainingkiss",
                "hornleech" "gigadrain",
                "megadrain",
                "leechlife",
                "oblivionwing",
                "paraboliccharge",
            ]
            if move.id in draining_moves:
                if resists_move and rng() < 0.8:
                    score -= 3

            if move.id == "dreameater":
                if resists_move:
                    score -= 1

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

            if move.id == "endeavor":
                if opp_hp < 0.7:
                    score -= 1
                else:
                    hp_thresh = 0.5 if slower else 0.4
                    if user_hp < hp_thresh:
                        score += 1
                    else:
                        score -= 1

            if move.id == "endure":
                if user_hp > 0.04 and user_hp < 0.34:
                    if rng() < 0.73:
                        score += 1
                else:
                    score -= 1

            if move.id == "doubleteam":
                if user_hp > 0.9 and rng() < 0.61:
                    score += 3
                if user.boosts["evasion"] >= 3 and rng() < 0.5:
                    score -= 1
                if user_hp < 0.4:
                    score -= 2
                if (
                    user.boosts["evasion"] != 0
                    and user_hp > 0.4
                    and user_hp < 0.7
                    and rng() < 0.73
                ):
                    score -= 2

            if move.id == "sweetscent":
                if user_hp < 0.7 or opp.boosts["evasion"] <= -3:
                    if rng() < 0.8:
                        score -= 2
                else:
                    score -= 2

            if move.id == "facade":
                good_status = [Status.BRN, Status.PAR, Status.PSN, Status.TOX]
                # Kaizo is bugged and checks target status instead of user
                # I swapped
                if user.status in good_status:
                    score += 1

            if move.id == "fakeout":
                score += 2

            if move.id == "focuspunch":
                if resists_move:
                    score -= 1
                if opp.status == Status.SLP:
                    score += 1

            if move.id == "roar" or move.id == "whirlwind":
                if (
                    opp.boosts["atk"] >= 3
                    or opp.boosts["spa"] >= 3
                    or opp.boosts["def"] >= 3
                    or opp.boosts["spd"] >= 3
                    or opp.boosts["evasion"] >= 3
                ):
                    if rng() < 0.5:
                        score += 2
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
            if move.id in hp_res_moves:
                if user_hp == 1:
                    score -= 3
                else:
                    if faster:
                        score -= 8
                    else:
                        if user_hp > 0.7:
                            if rng() < 0.88:
                                score -= 3
                        else:
                            if rng() < 0.92:
                                score += 2

            if move.id == "healbell" or move.id == "takeheart":
                if user.status == None:
                    score -= 5

            high_crit_moves = [
                "blazekick",
                "aeroblast",
                "crabhammer",
                "crosschop",
                "dragonclaw",
                "drillpeck",
                "drillrun",
                "karatechop",
                "leafblade",
                "razorleaf",
                "slash",
                "xscissors",
            ]
            if move.id in high_crit_moves:
                if type_adv_mult == 1:
                    if rng() < 0.25:
                        score += 1
                elif effective:
                    if rng() < 0.5:
                        score += 1

            if move.id == "imprison":
                if rng() < 0.61:
                    score += 2

            if move.id == "knockoff":
                if user_hp > 0.3 and rng() < 0.27:
                    score += 1

            if move.id == "leechseed" or move.id == "toxic":
                if user_has_damaging and user_hp < 0.5:
                    r = rng()
                    if r < 0.04:
                        pass
                    elif r < 0.35:
                        score -= 3
                    else:
                        score -= 6

                if user_has_prot_det and rng() < 0.77:
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

            if move.id == "magiccoat":
                if user_hp < 0.3 and rng() < 0.61:
                    score -= 1
                if user_hp == 1:
                    if rng() < 0.39:
                        score += 1
                elif rng() < 0.88:
                    score -= 1

            if move.id == "painsplit":
                if opp_hp < 0.8:
                    score -= 1
                else:
                    hp_thresh = 0.6 if slower else 0.4
                    if user_hp < hp_thresh:
                        score += 1
                    else:
                        score -= 1

            par_moves = ["glare", "stunspore", "thunderwave"]
            if move.id in par_moves:
                if slower:
                    if Status.PAR != opp.status and rng() < 0.92:
                        score += 3
                else:
                    if user_hp < 0.7:
                        score -= 1

            if move.id == "poisonpowder":
                if user_hp < 0.5 or opp_hp < 0.5:
                    score -= 1

            if move.id == "protect" or move.id == "detect":
                if user_last_move:
                    prev = user_last_move.id

                    if prev == "protect" or prev == "detect":
                        if rng() < 0.8:
                            score -= 2

            if move.id == "pursuit":
                if (
                    user_hp == 1
                    and PokemonType.GHOST in opp.types
                    or PokemonType.PSYCHIC in opp.types
                    and rng() < 0.5
                ):
                    score += 1

            recharge_moves = ["blastburn", "frenzyplant", "hydrocannon", "hyperbeam"]
            if move.id in recharge_moves:
                if resists_move:
                    score -= 1
                else:
                    hp_thresh = 0.6 if slower else 0.41
                    if user_hp > hp_thresh:
                        score -= 1

            if move.id == "recycle":
                score -= 2

            if move.id == "rest":
                if slower:
                    if user_hp == 1:
                        score -= 8
                    elif user_hp > 0.71:
                        score -= 5
                    elif user_hp > 0.6 and rng() < 0.8:
                        score -= 3
                    elif rng() < 0.96:
                        score += 3
                else:
                    if user_hp == 1:
                        score -= 8
                    elif user_hp > 0.5:
                        score -= 5
                    elif user_hp > 0.4 and rng() < 0.73:
                        score -= 3
                    elif rng() < 0.96:
                        score += 3

            if move.id == "revenge":
                if (
                    user.status == Status.SLP
                    or Effect.CONFUSION in user.effects
                    or rng() < 0.73
                ):
                    score -= 2
                else:
                    score += 2

            if move.id == "lightscreen" or move.id == "reflect":
                if user_hp < 0.5:
                    score -= 2

            if move.id == "explosion" or move.id == "selfdestruct":
                if user_hp > 0.8:
                    if rng() < 0.8:
                        score -= 1 if slower else 3
                elif user_hp > 0.5:
                    if rng() < 0.8:
                        score -= 1
                elif user_hp > 0.3:
                    if rng() < 0.5:
                        score += 1
                else:
                    r = rng()
                    if r < 0.4:
                        score += 2
                    elif r < 0.9:
                        score += 1

            semi_invuln_moves = ["bounce", "dig", "dive", "fly"]
            if move.id in semi_invuln_moves:
                if user_has_prot_det:
                    score -= 1
                else:
                    if (
                        faster
                        or Effect.LEECH_SEED in opp.effects
                        or opp.status == Status.TOX
                        or battle.weather == Weather.SANDSTORM
                        or battle.weather == Weather.HAIL
                    ):
                        if rng() < 0.69:
                            score += 1

            if move.id == "sleeptalk":
                if user.status == Status.SLP:
                    score += 10
                else:
                    score -= 5

            sleep_moves = [
                "grasswhistle",
                "hypnosis",
                "lovelykiss",
                "sing",
                "sleeppowder",
                "yawn",
            ]
            if move.id in sleep_moves:
                has_dream = False
                # Kaizo checks the opponent's moves, that is a bug
                for m in battle.available_moves:
                    if m.id == "dreameater" or m.id == "nightmare":
                        has_dream = True

                if has_dream and rng() < 0.5:
                    score += 1

            if move.id == "snore":
                if user.status == Status.SLP:
                    score += 2

            if move.id == "tailglow":
                if user.boosts["spa"] >= 3 and rng() < 0.61:
                    score -= 1

                if user_hp == 1:
                    if user.boosts["spa"] <= 2 and rng() < 0.5:
                        score += 1
                elif user_hp > 0.7:
                    pass
                elif user_hp > 0.4:
                    if rng() < 0.84:
                        score -= 2
                else:
                    score -= 2

            spd_boost_moves = ["cosmicpower", "stockpile", "calmmind", "amnesia"]
            if move.id in spd_boost_moves or is_boost["spd"]:
                if user_hp == 1:
                    if user.boosts["spd"] <= 2 and rng() < 0.5:
                        score += 2
                    if user.boosts["spd"] >= 3 and rng() < 0.61:
                        score -= 1
                elif user_hp < 0.4:
                    score -= 2
                elif user_hp > 0.7:
                    if rng() < 0.22:
                        score -= 2
                elif rng() < 0.7:
                    score -= 2

            if move.id == "faketears":
                if user_hp < 0.7 or opp.boosts["spd"] <= -3:
                    if rng() < 0.8:
                        score -= 2
                if user_hp < 0.7:
                    score -= 2

            if move.id == "agility":
                if slower:
                    if rng() < 0.73:
                        score += 3
                else:
                    score -= 3

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
                    if rng() < 0.73:
                        score += 2
                else:
                    score -= 3

            if move.id == "substitute":
                if user_hp < 0.5 and rng() < 0.61:
                    score -= 1
                if user_hp < 0.7 and rng() < 0.61:
                    score -= 1
                if user_hp < 0.9 and rng() < 0.61:
                    score -= 1

            if move.id == "superfang":
                if opp_hp > 0.5:
                    score -= 1

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

            if move.id == "vitalthrow":
                if slower:
                    if user_hp > 0.4 and user_hp < 0.6:
                        if rng() < 0.24:
                            score -= 1
                    elif user_hp <= 0.4:
                        if rng() < 0.2:
                            score -= 1

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

            if move.id == "acrobatics":
                # fairly sure the item check works
                if user.item == None and rng() < 0.7:
                    score += 1

            if move.id == "uturn":
                if ineffective and rng() < 0.7:
                    score += 1
                if effective:
                    score -= 1
                if user_hp < 0.4 and rng() < 0.3:
                    score += 1
                if user_hp < 0.1:
                    score += 1

            if move.id == "gyroball":
                if faster:
                    score -= 2

            if move.id == "haze":
                if (
                    opp.boosts["atk"] >= 3
                    or opp.boosts["spa"] >= 3
                    or opp.boosts["def"] >= 3
                    or opp.boosts["spd"] >= 3
                    or opp.boosts["evasion"] >= 3
                ):
                    score += 2

            if move.id == "bodypress":
                if user.boosts["def"] >= 3 and rng() < 0.8:
                    score += 1

            if move.id == "foulplay":
                if opp.boosts["atk"] >= 3 and rng() < 0.8:
                    score += 1

            if move.id == "taunt":
                if ineffective and rng() < 0.5:
                    score -= 1
                if (
                    opp_last_move
                    and self.type_advantage(opp_last_move.type, user, battle) > 1
                    and rng() < 0.5
                ):
                    score -= 1

            if move.id == "stealthrock":
                if (
                    PokemonType.FLYING not in opp.types
                    and SideCondition.STEALTH_ROCK
                    not in battle.opponent_side_conditions
                ):
                    if rng() < 0.4:
                        score += 1
                else:
                    score -= 1

            if move.id == "spikes":
                if PokemonType.FLYING not in opp.types and (
                    SideCondition.SPIKES not in battle.opponent_side_conditions
                    or battle.opponent_side_conditions[SideCondition.SPIKES] < 3
                ):
                    if rng() < 0.4:
                        score += 1
                else:
                    score -= 1

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
                    if rng() < 0.4:
                        score += 1
                else:
                    score -= 1

            if move.id == "defog":
                if opp_has_hazards and rng() < 0.8:
                    score -= 1
                if user_has_hazards and rng() < 0.5:
                    score += 1

            if move.id == "earthquake":
                if opp_last_move and opp_last_move.id == "dig" and faster:
                    score += 2
                    if rng() < 0.5:
                        score += 2

            if move.id == "earthpower":
                if opp_hp == 1:
                    if opp.boosts["spd"] > -3 and rng() < 0.5:
                        score += 1
                elif opp_hp > 0.7:
                    if opp.boosts["spd"] > -3 and rng() < 0.3:
                        score += 1

                if opp.boosts["spd"] <= -2 and rng() < 0.6:
                    score -= 1

            # Risky ai
            if risky:
                risky_moves = [
                    "attract",
                    "counter",
                    "destinybond",
                    "focuspunch",
                    "mirrorcoat",
                    "sleeppowder",
                    "hypnosis",
                    "lovelykiss",
                    "spore",
                    "selfdestruct",
                    "explosion",
                    "drillrun",
                    "xscissor",
                    "crosschop",
                    "dragonclaw",
                    "confuseray",
                    "teeterdance",
                    "ancientpower",
                ]
                if move.id in risky_moves and rng() < 0.5:
                    score += 2

            move_scores[move] = score

        #     if logging:
        #         if move.id not in move_score_modified_map:
        #             move_score_modified_map[move.id] = (0, 0)

        #         if orig_score != score:
        #             moves_scored_modified += 1
        #             move_score_modified_map[move.id] = (move_score_modified_map[move.id][0] + 1, move_score_modified_map[move.id][1])

        #         move_score_modified_map[move.id] = (move_score_modified_map[move.id][0], move_score_modified_map[move.id][1] + 1)
        #         moves_scored_total += 1

        # if logging and not logged and (len([m for m in battle.team.values() if m.fainted is False]) == 1 or len([m for m in battle.opponent_team.values() if m.fainted is False]) == 1):
        #     print(f"total moves scored: {moves_scored_total}")
        #     print(f"total moves scores modified: {moves_scored_modified}")
        #     print(f"move modify ratio: {moves_scored_modified/moves_scored_total}")
        #     print(f"modified moves by name:")
        #     print(sorted(move_score_modified_map.items(), key=lambda item: item[1][1], reverse=True))
        #     logged = True

        return move_scores

    def kaizo_forced_switch(self, battle):
        # Step 1: the best type matchup *for the opponent* with a supereffective move
        opp = battle.opponent_active_pokemon
        switch_scores = {}
        for switch in battle.available_switches:
            switch_score = 10
            for type in opp.types:
                # the annoying thing about pokemon is that it is so likely
                # that this entire behavior is a bug caused by switching
                # this line of code around on accident.
                # (self.type_advantage(user.type, opp, battle))
                matchup = self.type_advantage(type, switch, battle)
                twice = 2 if opp.types[-1] is None else 1
                if matchup == 0:
                    switch_score = 0
                elif matchup < 1:
                    switch_score /= 2 * twice
                elif matchup >= 2:
                    switch_score *= 2 * twice
            switch_scores[switch] = switch_score

        scores_high_low = sorted(
            switch_scores.items(), key=lambda x: x[1], reverse=True
        )
        for switch, score in scores_high_low:
            if score > 0:
                for move in switch.moves.values():
                    if (
                        self.type_advantage(move.type, opp, battle) >= 2
                        and move.base_power > 1
                    ):
                        return switch

        # Step 2: "highest damaging move"
        # this is super bugged in emerald so let's do it differently
        best_damage, best_mon = -float("inf"), None
        for switch in battle.available_switches:
            for move in switch.moves.values():
                damage = self.damage_equation(
                    switch,
                    move,
                    opp,
                    battle,
                    critical_hit=False,
                    rng="max",
                    assume_stats="uniform",
                    disable_rounding=False,
                )
                if damage > best_damage:
                    best_damage = damage
                    best_mon = switch
        return best_mon

    def kaizo_unforced_switch(self, battle):
        candidates = []
        for mon in battle.available_switches:
            for move in mon.moves.values():
                if (
                    self.type_advantage(
                        move.type, battle.opponent_active_pokemon, battle
                    )
                    > 1
                    and move.base_power > 1
                ):
                    candidates.append(mon)
        if candidates:
            return random.choice(candidates)

        return random.choice(battle.available_switches)

    def choose_move(self, battle: Battle):
        if battle.force_switch and battle.available_switches:
            switch = self.kaizo_forced_switch(battle)
            return self.create_order(switch)

        if (
            self.should_emergency_switch(battle, False, False)
            and random.random() < 0.9
            and battle.available_switches
        ):
            switch = self.kaizo_unforced_switch(battle)
            return self.create_order(switch)

        if battle.available_moves:
            move_scores = self.kaizo_score_moves(battle)
            best_move = max(move_scores, key=move_scores.get)
            return self.force_use_gimmick(battle, self.create_order(best_move))

        return self.choose_random_move(battle)


class EKRisky(EmeraldKaizo):
    """
    Based on Kaizo's Risky AI,
    same as the normal AI but with preference for risky moves
    """

    def randomize(self):
        pass

    def kaizo_score_moves(self, battle, risky=True):
        # override move scoring to risky
        return super().kaizo_score_moves(battle, risky=True)
