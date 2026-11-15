"""End-to-end check of the labeller (label.py) against Exeggcute's own code, on real trajectories.

The labeller makes the router's decisions fast in four ways: every member runs over a whole game in one parallel forward
(LABEL_SEQUENCE), the router's per-game decision is re-implemented in plain Python (fast_router.py), the members' timestep
encoder can be swapped for an equivalent one (fast_tstep.py), and the critic is evaluated for a whole step at once. Each was
tested on its own; this checks the whole labeller against a reference that uses none of them:

    reference   metamon's ensemble exactly as it plays a battle: one game at a time, step by step, carrying its own hidden state
                (KV caches, stall trackers), the router's own methods, the members' own encoder and critic calls. The labels are
                read off the router with hooks written here, independently of label.Recorder.

    python -m metamon.recurrent.verify_labels reference TRAJ_DIR REF_DIR [--device cpu] [--precision fp32]
    python -m metamon.recurrent.label TRAJ_DIR TEST_DIR --device cpu --precision fp32
    python -m metamon.recurrent.verify_labels compare REF_DIR TEST_DIR

`compare` prints, over every decision: the share where the chosen action differs, the largest differences of the blend and judge
distributions, and the decisions that differ most; it exits with an error when they differ beyond the given tolerances.
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time

import numpy as np

N_SLOTS = 9


class ReferenceCapture:
    """Reads the router's intermediate results, without changing what it does."""

    def __init__(self, policy):
        self.weights = None
        self.judge = None
        self.member_steps = None
        self.last = None
        build, judge, choose = policy._build_proposer_variants, policy._judge_shortlist, policy._choose_action_for_batch

        def build_hook(**kw):
            self.weights = [float(w) for w in kw["proposer_weights"]]
            return build(**kw)

        def judge_hook(**kw):
            shortlist, scores = judge(**kw)
            self.judge = ([int(a) for a in shortlist], scores.detach().double().cpu().numpy().copy())
            return shortlist, scores

        def choose_hook(**kw):
            self.weights, self.judge = None, None
            self.member_steps = kw["member_steps"]
            chosen = choose(**kw)
            self.last = {"chosen": int(chosen), "weights": self.weights, "judge": self.judge, "legal": list(kw["legal_actions"]),
                         "probs": [s["probs"][kw["batch_idx"]].detach().double().cpu().numpy().reshape(-1).copy() for s in self.member_steps]}
            return chosen

        policy._build_proposer_variants = build_hook
        policy._judge_shortlist = judge_hook
        policy._choose_action_for_batch = choose_hook


def reference_game(spec, policy, capture, traj: dict, device, half) -> dict:
    import torch
    from metamon.interface import UniversalAction, UniversalState

    action_space = spec.action_space
    n_actions = action_space.gym_space.n
    space = copy.deepcopy(spec.observation_space)
    space.reset()
    states = [UniversalState.from_dict(copy.deepcopy(s)) for s in traj["legacy_states"]]
    actions = traj["legacy_actions"]
    n = len(states) - 1  # a decision per state but the last
    out = {k: np.zeros((n, N_SLOTS), np.float32) for k in ("blend", "judge", "legal")}
    out |= {"chosen": np.full(n, -1, np.int16), "has_blend": np.zeros(n, bool), "has_judge": np.zeros(n, bool)}
    prev_action = np.zeros(n_actions, np.float32)
    prev_reward = 0.0
    hidden = None
    autocast = torch.autocast(device.type, dtype=half or torch.bfloat16, enabled=half is not None and device.type in ("cuda", "cpu"))
    for t in range(n):
        s = states[t]
        obs = space.state_to_obs(s)
        legal = sorted({action_space.action_to_agent_output(s, a) for a in UniversalAction.maybe_valid_actions(s)})
        mask = np.ones(n_actions, bool)
        mask[legal] = False
        obs["illegal_actions"] = mask
        batch = {k: torch.as_tensor(np.asarray(v))[None, None] for k, v in obs.items()}
        rl2 = torch.as_tensor(np.concatenate([[prev_reward], prev_action]).astype(np.float32)).view(1, 1, -1)
        time_idxs = torch.full((1, 1, 1), t, dtype=torch.long)
        capture.last = None
        with torch.inference_mode(), autocast:
            chosen, hidden = policy.get_actions(batch, rl2, time_idxs, hidden_state=hidden, sample=False)
        out["chosen"][t] = int(chosen.reshape(-1)[0])
        out["legal"][t, [a for a in legal if a < N_SLOTS]] = 1.0
        rec = capture.last
        slot_mask = np.zeros(N_SLOTS)
        slot_mask[[a for a in legal if a < N_SLOTS]] = 1.0
        if rec is not None and rec["weights"] is not None:
            blend = np.zeros(N_SLOTS)
            for w, p in zip(rec["weights"], rec["probs"]):
                if w > 0:
                    q = p[:N_SLOTS] * slot_mask
                    blend += w * q / max(q.sum(), 1e-12)
            if blend.sum() > 0:
                out["blend"][t] = blend / blend.sum()
                out["has_blend"][t] = True
        if rec is not None and rec["judge"] is not None:
            shortlist, scores = rec["judge"]
            e = np.exp(scores - scores.max())
            e /= e.sum()
            for a, p in zip(shortlist, e):
                if a < N_SLOTS:
                    out["judge"][t, a] = p
            out["has_judge"][t] = True
        if len(legal) == 1:  # (the labeller's convention: a forced decision's label is its one action)
            out["blend"][t] = 0.0
            out["blend"][t, legal[0]] = 1.0
            out["has_blend"][t] = True
        a = actions[t]
        prev_action = np.zeros(n_actions, np.float32)
        if a >= 0:
            prev_action[action_space.action_to_agent_output(s, UniversalAction(action_idx=a))] = 1.0
        prev_reward = float(spec.reward_function(states[t], states[t + 1]))
    return out


def reference(args) -> None:
    import torch
    from metamon.recurrent.label import load_ensemble
    from metamon.recurrent.records import load_for_labelling

    spec, policy = load_ensemble(args.cache, args.device, args.threads)
    capture = ReferenceCapture(policy)
    half = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "**", "*.json.lz4"), recursive=True))
    start = time.time()
    for i, f in enumerate(files):
        dest = os.path.join(args.output, os.path.basename(f).split(".json")[0] + ".npz")
        if os.path.exists(dest):
            continue
        out = reference_game(spec, policy, capture, load_for_labelling(f), device, half)
        np.savez_compressed(dest, **out)
        print(f"reference {i + 1}/{len(files)} games ({len(out['chosen'])} decisions), {time.time() - start:.0f}s", flush=True)


def compare(args) -> None:
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.reference, "*.npz")))
    if not names:
        raise SystemExit(f"no reference labels in {args.reference}")
    if args.preset == "bf16":  # labels made in bf16 (as on vast) against the fp32 reference: near-ties flip, see PRESETS
        for k, v in PRESETS["bf16"].items():
            setattr(args, k, v)
    rows = []
    totals = {"decisions": 0, "chosen_differs": 0, "missing": 0, "legal_differs": 0, "has_differs": 0, "permuted": 0}
    worst_blend = worst_judge = 0.0
    blend_l1, blend_kl, judge_l1 = [], [], []
    for name in names:
        test_path = os.path.join(args.test, name)
        if not os.path.exists(test_path):
            totals["missing"] += 1
            continue
        r, s = np.load(os.path.join(args.reference, name)), np.load(test_path)
        n = len(r["chosen"])
        if len(s["chosen"]) != n:
            raise SystemExit(f"{name}: {len(s['chosen'])} decisions, the reference has {n}")
        totals["decisions"] += n
        totals["legal_differs"] += int((r["legal"] != s["legal"]).any(1).sum())
        totals["has_differs"] += int(((r["has_blend"] != s["has_blend"]) | (r["has_judge"] != s["has_judge"])).sum())
        differs = r["chosen"] != s["chosen"]
        totals["chosen_differs"] += int(differs.sum())
        db = np.abs(r["blend"].astype(np.float64) - s["blend"]).max(1)
        dj = np.abs(r["judge"].astype(np.float64) - s["judge"]).max(1)
        worst_blend, worst_judge = max(worst_blend, float(db.max())), max(worst_judge, float(dj.max()))
        rb, sb = r["blend"].astype(np.float64), s["blend"].astype(np.float64)
        both = r["has_blend"] & s["has_blend"]
        blend_l1 += list(np.abs(rb - sb).sum(1)[both])
        blend_kl += list((rb * (np.log(np.maximum(rb, 1e-12)) - np.log(np.maximum(sb, 1e-12)))).sum(1)[both])
        both_j = r["has_judge"] & s["has_judge"]
        judge_l1 += list(np.abs(r["judge"].astype(np.float64) - s["judge"]).sum(1)[both_j])
        # the same probabilities on other slots (a reordering of the actions, the bug class that hides best): flagged on its own
        far = both & (np.abs(rb - sb).sum(1) > 0.2)
        totals["permuted"] += int((far & (np.abs(np.sort(rb, 1) - np.sort(sb, 1)).sum(1) < 0.05)).sum())
        for t in range(n):
            if differs[t] or db[t] > args.blend_tol or dj[t] > args.judge_tol:
                rows.append((name, t, int(r["chosen"][t]), int(s["chosen"][t]), float(db[t]), float(dj[t])))
    d = max(totals["decisions"], 1)
    print(f"{len(names)} games, {totals['decisions']} decisions ({totals['missing']} games missing from {args.test})")
    print(f"chosen differs: {totals['chosen_differs']} ({totals['chosen_differs'] / d:.4%}); legal differs: {totals['legal_differs']}; "
          f"has_blend/has_judge differ: {totals['has_differs']}")
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    print(f"largest |blend difference| {worst_blend:.2e}, largest |judge difference| {worst_judge:.2e}")
    print(f"mean blend L1 {mean(blend_l1):.2e}, mean KL(reference || labels) {mean(blend_kl):.2e}, mean judge L1 {mean(judge_l1):.2e}; "
          f"decisions whose blend looks reordered: {totals['permuted']}")
    for row in sorted(rows, key=lambda r: -max(r[4], r[5]))[:15]:
        print(f"  {row[0]} t={row[1]} chosen {row[2]} vs {row[3]}, blend diff {row[4]:.3e}, judge diff {row[5]:.3e}")
    problems = []
    if totals["missing"]:
        problems.append(f"{totals['missing']} games have no label")
    if totals["legal_differs"] or totals["has_differs"]:
        problems.append("the legal actions or the label presence differ (labels of another state or game)")
    if totals["permuted"]:
        problems.append(f"{totals['permuted']} decisions have the reference's probabilities on other actions (reordered)")
    if totals["chosen_differs"] > args.chosen_tol * d:
        problems.append(f"chosen action differs in {totals['chosen_differs'] / d:.2%} (allowed {args.chosen_tol:.2%})")
    if worst_blend > args.blend_tol or worst_judge > args.judge_tol:
        problems.append(f"largest difference blend {worst_blend:.3f} / judge {worst_judge:.3f} (allowed {args.blend_tol} / {args.judge_tol})")
    if mean(blend_l1) > args.mean_blend_tol or mean(blend_kl) > args.mean_kl_tol or mean(judge_l1) > args.mean_judge_tol:
        problems.append(f"mean differences blend L1 {mean(blend_l1):.4f} / KL {mean(blend_kl):.5f} / judge L1 {mean(judge_l1):.4f} "
                        f"(allowed {args.mean_blend_tol} / {args.mean_kl_tol} / {args.mean_judge_tol})")
    if totals["decisions"] < args.min_decisions:
        problems.append(f"only {totals['decisions']} decisions compared (at least {args.min_decisions})")
    if problems:
        raise SystemExit("the labels differ from the reference: " + "; ".join(problems))
    print("labels match the reference within the tolerances")


# Measured on 24 games (6,375 decisions): labeller in bf16 against the fp32 reference. Chosen differs 0.39% (all near-ties), largest blend
# difference 0.044, largest judge difference 0.71 (a near-tie flips the router's pick, and the judge then scores another shortlist), mean blend
# L1 and KL as printed by `compare`. The limits are several times those, and far below what a wrong label set gives (a reordering moves most of
# the mass: L1 near 1 on most decisions).
PRESETS = {"bf16": {"chosen_tol": 0.02, "blend_tol": 0.25, "judge_tol": 1.0, "mean_blend_tol": 0.03, "mean_kl_tol": 2e-3, "mean_judge_tol": 0.15}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("reference")
    r.add_argument("input")
    r.add_argument("output")
    r.add_argument("--cache", default=os.environ.get("METAMON_CACHE_DIR", "."))
    r.add_argument("--device", default="cpu")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("test")
    c.add_argument("--blend-tol", type=float, default=1e-3, help="largest allowed |difference| of a blend probability")
    c.add_argument("--judge-tol", type=float, default=1e-3)
    c.add_argument("--chosen-tol", type=float, default=0.0, help="largest allowed share of decisions with another chosen action")
    c.add_argument("--mean-blend-tol", type=float, default=1e-3, help="largest allowed mean L1 distance of the blends")
    c.add_argument("--mean-kl-tol", type=float, default=1e-4, help="largest allowed mean KL(reference blend || tested blend)")
    c.add_argument("--mean-judge-tol", type=float, default=1e-3, help="largest allowed mean L1 distance of the judge distributions")
    c.add_argument("--min-decisions", type=int, default=1)
    c.add_argument("--preset", choices=("exact", "bf16"), default="exact", help="bf16: the tolerances for bf16 labels against the fp32 reference (overrides the others)")
    args = ap.parse_args()
    {"reference": reference, "compare": compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
