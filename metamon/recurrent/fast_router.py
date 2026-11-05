"""A plain-Python version of the ensemble router's per-game decision, for labelling.

`HeuristicRouterEnsemblePolicy._choose_action_for_batch` (ensemble.py) decides one game at a time, and almost all of its time goes to torch
calls on tensors of two to nine numbers (a sort, a std, a dozen `.item()`s per member and game): each costs a few microseconds of
dispatch for no arithmetic. At labelling scale the members run in parallel on the GPU, and this per-game Python is what the GPU waits for.

This module makes the same decision with the member probabilities read once per step as plain lists, and everything else in Python floats.
The logic is the router's, line for line (same thresholds, same orderings, same tie-breaks); `_build_proposer_variants`,
`_select_proposer_variants`, `_stall_penalties` and the `_record_anchor_decision` bookkeeping are the router's own methods, called unchanged.
Arithmetic differs only in precision (float64 where the router used float32 tensors), which can move a score in its 7th digit.
`test_fast_router.py` checks the decisions and the label arrays against the original on recorded games.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-8


def _zscore(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [0.0] * n
    mean = sum(values) / n
    var = sum((v - mean) * (v - mean) for v in values) / n
    std = math.sqrt(var)
    if std < EPS:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def _argmax(values: list[float]) -> int:
    best, best_i = values[0], 0
    for i in range(1, len(values)):
        if values[i] > best:
            best, best_i = values[i], i
    return best_i


def install(policy, recorder) -> None:
    """Replaces the router's per-game decision with the fast one. `recorder` supplies the critic table and receives the label record."""
    from metamon.rl.experimental.ensemble import ensemble as E

    cls = type(policy)
    normalize_with_floor = E._normalize_with_floor
    n_slots = recorder.n_slots
    original = policy._choose_action_for_batch  # (kept so it can be switched back)
    recorder.exact_choose = original

    def rows_of(member_steps):
        """The step's probabilities as [games][actions] lists, per member (read from the device once per step)."""
        cache = member_steps[0].get("rows")
        if cache is None:
            for step in member_steps:
                probs = step["probs"]
                step["np"] = probs.float().cpu().numpy()
                step["rows"] = step["np"].tolist()
            cache = member_steps[0]["rows"]
        return [step["rows"] for step in member_steps]

    def choose(self, batch_idx, legal_actions, member_steps, sample, tracker, current_state_key, state_summary):
        rows_all = rows_of(member_steps)
        anchor_idx = self.anchor_idx
        n_members = len(member_steps)
        n_legal = len(legal_actions)
        log_legal = max(math.log(n_legal), 1.0)
        is_move, is_switch = self._is_move_action, self._is_switch_action

        # ---- per-member features
        features, top_actions, normed = [], [], []
        for m, step in enumerate(member_steps):
            row = rows_all[m][batch_idx]
            p = [row[a] for a in legal_actions]
            total = max(sum(p), EPS)
            p = [x / total for x in p]
            normed.append(p)
            entropy = -sum(x * math.log(max(x, EPS)) for x in p) / log_legal
            top_idx = _argmax(p)
            ordered = sorted(p, reverse=True)
            spec = step["member"].spec
            features.append(
                {
                    "entropy": entropy,
                    "certainty": 1.0 - entropy,
                    "top_prob": ordered[0],
                    "margin": ordered[0] - ordered[1] if len(ordered) > 1 else 1.0,
                    "top_action": legal_actions[top_idx],
                    "strength": spec.gxe,
                    "spec": spec,
                }
            )
            top_actions.append(legal_actions[top_idx])

        anchor = features[anchor_idx]
        default_anchor_action = anchor["top_action"]
        default_anchor_margin = anchor["margin"]
        default_anchor_top_prob = anchor["top_prob"]

        vote_weights: dict[int, float] = {}
        vote_counts: dict[int, int] = {}
        for info in features:
            action = info["top_action"]
            vote_weights[action] = vote_weights.get(action, 0.0) + (0.80 * info["strength"] + 0.20 * info["certainty"])
            vote_counts[action] = vote_counts.get(action, 0) + 1
        consensus_action = max(sorted(vote_weights), key=lambda action: (vote_counts[action], vote_weights[action], -action))
        consensus_count = vote_counts[consensus_action]
        anchor_action = default_anchor_action
        if (
            consensus_action != default_anchor_action
            and consensus_count >= 2
            and vote_weights[consensus_action] > vote_weights.get(default_anchor_action, 0.0)
            and default_anchor_top_prob < 0.62
        ):
            anchor_action = consensus_action

        anchor_prob_map = dict(zip(legal_actions, normed[anchor_idx]))
        anchor_top_prob = anchor_prob_map.get(anchor_action, default_anchor_top_prob)
        if anchor_action == default_anchor_action:
            anchor_margin = default_anchor_margin
        else:
            sorted_anchor_probs = sorted(anchor_prob_map.values(), reverse=True)
            anchor_margin = max(anchor_top_prob - (sorted_anchor_probs[0] if sorted_anchor_probs else 0.0), 0.0)
        for info, p in zip(features, normed):
            move_mass = switch_mass = counter = 0.0
            for action, x in zip(legal_actions, p):
                if is_move(action):
                    move_mass += x
                if is_switch(action):
                    switch_mass += x
                if action != anchor_action:
                    counter += x
            info["move_mass"], info["switch_mass"], info["counter_anchor_mass"] = move_mass, switch_mass, counter
        forced_switch = all(action >= 4 for action in legal_actions)
        stall_penalties = self._stall_penalties(tracker=tracker, current_state_key=current_state_key, forced_switch=forced_switch)
        late_turn = state_summary["turn_idx"] >= 30.0
        desperation = state_summary["resource_edge"] < -0.10 or (late_turn and state_summary["resource_edge"] < 0.02)
        stabilize = state_summary["resource_edge"] > 0.14 and (state_summary["player_remaining"] >= state_summary["opponent_remaining"])
        disagreement = 1.0 - (max(top_actions.count(action) for action in set(top_actions)) / max(len(top_actions), 1))

        proposer_scores, judge_scores = [], []
        for idx, info in enumerate(features):
            proposer = info["spec"].proposer_bias * (
                0.70 * info["strength"] + 0.20 * info["entropy"] + 0.10 * float(info["top_action"] != anchor_action)
            )
            judge = info["spec"].judge_bias * (0.85 * info["strength"] + 0.15 * info["certainty"])
            if forced_switch:
                proposer *= 1.05
                judge *= 1.05
            if disagreement > 0.35:
                proposer *= 1.05
            if idx == anchor_idx:
                proposer *= 1.35
                judge *= 1.45
            if desperation:
                if idx == anchor_idx:
                    proposer *= 0.95
                    judge *= 0.96
                else:
                    proposer *= 1.04 + 0.06 * float(info["top_action"] != anchor_action)
                    judge *= 1.02
            if stabilize:
                if idx == anchor_idx:
                    proposer *= 1.04
                    judge *= 1.08
                elif info["top_action"] != anchor_action:
                    proposer *= 0.94
                    judge *= 0.97
            if info["strength"] < 0.50:
                proposer *= 0.75
                judge *= 0.50
            proposer_scores.append(proposer)
            judge_scores.append(judge)

        anchor_judge_score = judge_scores[anchor_idx]
        best_other_judge = max((judge_scores[idx] for idx in range(n_members) if idx != anchor_idx), default=0.0)
        anchor_dominant_judge = anchor_judge_score > 0.0 and best_other_judge <= 0.20 * anchor_judge_score
        if anchor_dominant_judge:
            num_judges = 1
        elif n_members <= 3:
            num_judges = n_members
        else:
            num_judges = min(n_members, 2 + int(disagreement > 0.35 and anchor_margin < 0.15) + int(desperation and n_members >= 4))
        judge_order = [anchor_idx]
        for idx in sorted((i for i in range(n_members) if i != anchor_idx), key=lambda i: judge_scores[i], reverse=True):
            if len(judge_order) >= num_judges:
                break
            judge_order.append(idx)

        strong_anchor = features[anchor_idx]["strength"] >= 0.82
        proposer_floor = 0.18 if desperation else 0.22
        judge_floor = 0.40 if desperation else 0.45
        if strong_anchor and not desperation:
            proposer_floor += 0.05
            judge_floor += 0.08
        proposer_weights = normalize_with_floor(proposer_scores, anchor_pos=anchor_idx, floor=proposer_floor)
        judge_weights = normalize_with_floor([judge_scores[idx] for idx in judge_order], anchor_pos=judge_order.index(anchor_idx), floor=judge_floor)
        proposer_variants, mixed_choice = self._build_proposer_variants(
            legal_actions=legal_actions,
            proposer_weights=proposer_weights,
            features=features,
            anchor_action=anchor_action,
            forced_switch=forced_switch,
        )
        proposer_variants = self._select_proposer_variants(proposer_variants=proposer_variants, disagreement=disagreement, forced_switch=forced_switch)

        candidate_support: dict[int, float] = {}
        candidate_members: dict[int, set[int]] = {}
        candidate_roles: dict[int, set[str]] = {}
        for variant in proposer_variants:
            if not variant.allowed_actions or variant.weight <= 0.0:
                continue
            proposer_idx = variant.proposer_idx
            info = features[proposer_idx]
            shortlist_k = self._shortlist_k_for_variant(
                info=info, proposer_idx=proposer_idx, role=variant.role, allowed_actions=variant.allowed_actions, mixed_choice=mixed_choice
            )
            allowed = list(variant.allowed_actions)
            row = rows_all[proposer_idx][batch_idx]
            vals = [row[a] for a in allowed]
            total = max(sum(vals), EPS)
            vals = [v / total for v in vals]
            order = sorted(range(len(allowed)), key=lambda i: -vals[i])  # (stable: ties keep the allowed-list order)
            for i in order[:shortlist_k]:
                action = allowed[i]
                candidate_support[action] = candidate_support.get(action, 0.0) + variant.weight * vals[i]
                candidate_members.setdefault(action, set()).add(proposer_idx)
                candidate_roles.setdefault(action, set()).add(variant.role)
        candidate_support.setdefault(anchor_action, anchor_top_prob)
        candidate_members.setdefault(anchor_action, set()).add(anchor_idx)
        candidate_roles.setdefault(anchor_action, set()).add("anchor_guard")

        shortlist, proposal_support = route_shortlist(
            self, legal_actions, anchor_action, candidate_support, candidate_members, candidate_roles, features,
            disagreement, mixed_choice, anchor_top_prob, state_summary, strong_anchor, desperation, stabilize, late_turn,
        )
        if not shortlist:
            shortlist = [consensus_action]
            proposal_support = {consensus_action: 1.0}

        def judge(candidates):
            single = len(judge_order) == 1
            critic_mix = (0.35 if forced_switch else 0.15) if single else (0.25 if forced_switch else 0.10)
            proposal_bonus = 0.18 if single else 0.50
            actor_mix = 1.0 - critic_mix
            scores = [0.0] * len(candidates)
            for weight, judge_idx in zip(judge_weights, judge_order):
                step = member_steps[judge_idx]
                row = rows_all[judge_idx][batch_idx]
                actor = _zscore([row[a] for a in candidates])
                critic = _zscore(recorder.critic_list(step, batch_idx, candidates))
                for k in range(len(scores)):
                    scores[k] += weight * (actor_mix * actor[k] + critic_mix * critic[k])
            bonus = _zscore([proposal_support.get(a, 0.0) for a in candidates])
            return [s + proposal_bonus * b for s, b in zip(scores, bonus)]

        def penalised(candidates, scores):
            if not stall_penalties:
                return scores
            return [s - stall_penalties.get(a, 0.0) for a, s in zip(candidates, scores)]

        recorder.judge_calls = 1
        raw = judge(shortlist)
        last_shortlist, last_raw = shortlist, raw
        final_scores = penalised(shortlist, raw)
        full_rerank = False
        if len(shortlist) < n_legal:
            ordered = sorted(final_scores, reverse=True)
            margin = ordered[0] - ordered[1] if len(ordered) > 1 else float("inf")
            if anchor_dominant_judge or disagreement > 0.25 or margin < 0.16:
                shortlist = list(legal_actions)
                recorder.judge_calls = 2
                raw = judge(shortlist)
                last_shortlist, last_raw = shortlist, raw
                final_scores = penalised(shortlist, raw)
                full_rerank = True
        recorder.last_judge = (last_shortlist, last_raw)
        recorder.last_weights = proposer_weights
        recorder.last_rows = [rows_all[m][batch_idx] for m in range(n_members)]

        best_local = _argmax(final_scores)
        best_action = shortlist[best_local]

        def record(final_action):
            self._record_anchor_decision(
                default_anchor_action=default_anchor_action,
                anchor_action=anchor_action,
                proposed_best_action=best_action,
                final_action=final_action,
                forced_switch=forced_switch,
                single_judge=len(judge_order) == 1,
                full_rerank=full_rerank,
                shortlist_size=len(shortlist),
                proposer_variant_count=len(proposer_variants),
                judge_count=len(judge_order),
            )

        if best_action == anchor_action:
            record(best_action)
            return best_action

        anchor_score = float("-inf")
        if anchor_action in shortlist:
            anchor_score = final_scores[shortlist.index(anchor_action)]
        best_score = final_scores[best_local]
        members_for_best = candidate_members.get(best_action, set())
        supporting_members = sum(idx != anchor_idx for idx in members_for_best)
        supporting_roles = len(candidate_roles.get(best_action, set()))
        alt_strength_sum = sum(features[idx]["strength"] for idx in members_for_best if idx != anchor_idx)
        allow_override = (
            alt_strength_sum >= 1.25
            or (
                alt_strength_sum >= 0.62
                and supporting_roles >= 2
                and anchor_top_prob < 0.58
                and disagreement > 0.18
                and best_score - anchor_score > 0.10
            )
            or (supporting_members >= 2 and supporting_roles >= 2 and best_score - anchor_score > 0.14)
            or (desperation and alt_strength_sum >= 0.62 and supporting_roles >= 2 and best_score - anchor_score > 0.08)
        )
        if strong_anchor and not desperation:
            allow_override = allow_override and (
                alt_strength_sum >= 1.45 or (supporting_members >= 2 and supporting_roles >= 2 and best_score - anchor_score > 0.12)
            )
        override_margin = 0.05 if anchor_margin < 0.08 else 0.10
        if strong_anchor and not desperation:
            override_margin += 0.03
        if len(judge_order) >= 2 and len(shortlist) <= 4 and not full_rerank:
            override_margin = max(0.03, override_margin - 0.015)
            if supporting_members >= 1 and supporting_roles >= 2:
                allow_override = allow_override or (alt_strength_sum >= 0.62 and best_score - anchor_score > 0.07)
        if desperation:
            override_margin = max(0.03, override_margin - 0.03)
        if stabilize:
            override_margin += 0.02
        if not forced_switch and (not allow_override or best_score - anchor_score < override_margin):
            record(anchor_action)
            return anchor_action
        record(best_action)
        return best_action

    def route_shortlist(
        self, legal_actions, anchor_action, candidate_support, candidate_members, candidate_roles, features,
        disagreement, mixed_choice, anchor_top_prob, state_summary, strong_anchor, desperation, stabilize, late_turn,
    ):
        anchor_idx = self.anchor_idx
        if not candidate_support:
            return [anchor_action], {anchor_action: anchor_top_prob}
        actions = sorted(candidate_support)
        support = [candidate_support[a] for a in actions]
        members = [float(len(candidate_members.get(a, ()))) for a in actions]
        roles = [float(len(candidate_roles.get(a, ()))) for a in actions]
        non_anchor = [float(sum(m != anchor_idx for m in candidate_members.get(a, ()))) for a in actions]
        strength = [sum(features[m]["strength"] for m in candidate_members.get(a, ())) for a in actions]

        z_non_anchor, z_strength = _zscore(non_anchor), _zscore(strength)
        zs = (_zscore(support), _zscore(members), _zscore(roles), z_non_anchor, z_strength)
        scores = [0.45 * a + 0.25 * b + 0.10 * c + 0.15 * d + 0.10 * e for a, b, c, d, e in zip(*zs)]
        if desperation:
            scores = [s + 0.08 * z for s, z in zip(scores, z_non_anchor)]
        if stabilize:
            scores = [s + 0.05 * z for s, z in zip(scores, z_strength)]
        if anchor_action in actions:
            bonus = 0.04 + 0.03 * stabilize - 0.02 * desperation
            if strong_anchor and not desperation:
                bonus += 0.03
            scores[actions.index(anchor_action)] += bonus

        shortlist_k = min(len(actions), 3 + int(len(legal_actions) >= 5) + int(disagreement > 0.30) + int(mixed_choice))
        shortlist_k = min(len(actions), shortlist_k + int(desperation) + int(late_turn and mixed_choice))
        ranked = sorted(
            range(len(actions)),
            key=lambda i: (scores[i], support[i], strength[i], members[i], -actions[i]),
            reverse=True,
        )
        shortlist = [actions[i] for i in ranked[:shortlist_k]]
        if anchor_action not in shortlist:
            if len(shortlist) < shortlist_k:
                shortlist.append(anchor_action)
            else:
                replace_idx = len(shortlist) - 1
                for i in range(len(shortlist) - 1, -1, -1):
                    if shortlist[i] != anchor_action:
                        replace_idx = i
                        break
                shortlist[replace_idx] = anchor_action
        if mixed_choice:
            for predicate in (self._is_move_action, self._is_switch_action):
                if any(predicate(a) for a in shortlist):
                    continue
                candidates = [i for i, a in enumerate(actions) if predicate(a)]
                if not candidates:
                    continue
                best = max(candidates, key=lambda i: (scores[i], support[i], strength[i], -actions[i]))
                replace_idx = len(shortlist) - 1
                for i in range(len(shortlist) - 1, -1, -1):
                    if shortlist[i] != anchor_action and not predicate(shortlist[i]):
                        replace_idx = i
                        break
                shortlist[replace_idx] = actions[best]
        shortlist = list(dict.fromkeys(shortlist))
        proposal_support = {a: candidate_support.get(a, 0.0) for a in shortlist}
        proposal_support.setdefault(anchor_action, anchor_top_prob)
        return shortlist, proposal_support

    recorder.route_shortlist = route_shortlist  # (for the test)
    policy._choose_action_for_batch = choose.__get__(policy, cls)
