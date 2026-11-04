"""Exeggcute labels for parsed trajectories.

For every decision of a parsed trajectory this runs Metamon's Exeggcute ensemble (the heuristic router) on
the sequence of states, exactly as the router would see them in a battle, and records what the router
computes on the way to its action:

  blend   sum_m w_m * p_m over the legal actions, with each member's distribution masked to the legal
          options and renormalised, and w_m the router's proposer weights
  scores  the judge's final scores for the shortlisted actions (the router plays their argmax), as a
          softmax over the shortlist; zero outside it
  chosen  the router's action
  legal   the legal actions (from the parsed state)

Inputs to the members are what they get in a battle: the observation, and the previous reward and previous
action (RL2 input). In a replay the previous action is the one that was actually played, which was not
Exeggcute's; this is the standard offline setup and the labels are Exeggcute's answer given that history.
A missing action (-1) enters as zeros.

Games are labelled in lockstep batches (all lanes advance one decision per call), longest games first, so
that the members run on [B, 1, ...] tensors. Output: one `<name>.npz` per input file.

    python -m metamon.recurrent.label IN_DIR OUT_DIR [--batch 32] [--device cuda] [--limit N]
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from metamon.recurrent.records import load_trajectory

N_SLOTS = 9


def _use_sdpa() -> None:
    """The released members select CUDA-only FlashAttention (fp16/bf16 only) with a 32-step window. This
    swaps in PyTorch SDPA with the same window, which runs in fp32 and needs no flash_attn install; the
    weights are unchanged because the attention has no parameters."""
    import amago
    import gin

    original = amago.cli_utils.use_config

    def use_config(*a, **kw):
        result = original(*a, **kw)
        from metamon.recurrent.sdpa_attention import SDPAAttention

        gin.bind_parameter("amago.nets.traj_encoders.TformerTrajEncoder.attention_type", SDPAAttention)
        try:
            window = gin.query_parameter("amago.nets.transformer.FlashAttention.window_size")
            gin.bind_parameter("metamon.recurrent.sdpa_attention.SDPAAttention.window_size", window)
        except ValueError:
            pass
        return result

    amago.cli_utils.use_config = use_config


def load_ensemble(cache_dir: str, device: str, threads: int, attention: str = "sdpa"):
    """The Exeggcute ensemble policy plus its observation space and reward function."""
    os.environ["METAMON_CACHE_DIR"] = os.path.abspath(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["METAMON_ENSEMBLE_MEMBER_DEVICES"] = device
    import torch

    torch.set_num_threads(threads)
    import gymnasium

    sys.modules.setdefault("gym", gymnasium)
    from metamon.rl.pretrained import get_pretrained_model

    if attention == "sdpa":
        _use_sdpa()
    spec = get_pretrained_model("Exeggcute")
    # the released preset kept an internal class name; the public checkpoint family is TaurosV0
    type(spec).MEMBER_SPECS = [
        replace(m, model_name="TaurosV0") if m.model_name == "TaurosV5" else m
        for m in type(spec).MEMBER_SPECS
    ]
    experiment = spec.initialize_agent(log=False)
    policy = experiment.policy_aclr.eval()
    return spec, policy


class Recorder:
    """Hooks on the router that capture its intermediate results without changing its behaviour."""

    def __init__(self, policy):
        self.policy = policy
        self.weights = None
        self.judge = None
        self.judge_calls = 0
        self.member_steps = None
        self.batch_idx = None
        self.record: dict[int, dict] = {}

        build = policy._build_proposer_variants
        judge = policy._judge_shortlist
        choose = policy._choose_action_for_batch

        def build_hook(**kw):
            self.weights = list(kw["proposer_weights"])
            return build(**kw)

        def judge_hook(**kw):
            self.member_steps = kw["member_steps"]
            self.batch_idx = kw["batch_idx"]
            shortlist, scores = judge(**kw)
            self.judge = (list(shortlist), scores.detach().float().cpu().numpy())
            self.judge_calls += 1
            return shortlist, scores

        def choose_hook(**kw):
            # The router's per-game loop reads each member's probabilities with dozens of tiny .item() calls. On a GPU every
            # one of them waits for the device; moving the [batch, actions] tables to the CPU once per step (a no-op after
            # the first game of the batch) gives the same numbers without the waiting.
            for step in kw["member_steps"]:
                if step["probs"].device.type != "cpu":
                    step["probs"] = step["probs"].cpu()
            self.weights = None
            self.judge = None
            self.judge_calls = 0
            chosen = choose(**kw)
            idx = kw["batch_idx"]
            rec = {"chosen": int(chosen), "judge_calls": self.judge_calls}
            legal = kw["legal_actions"]
            if self.weights is not None and self.member_steps is not None:
                blend = np.zeros(N_SLOTS, np.float64)
                for m, w in enumerate(self.weights):
                    if w <= 0:
                        continue
                    p = self.member_steps[m]["probs"][idx].float().cpu().numpy()[:N_SLOTS]
                    mask = np.zeros(N_SLOTS)
                    mask[[a for a in legal if a < N_SLOTS]] = 1.0
                    p = p * mask
                    blend += w * p / max(p.sum(), 1e-12)
                if blend.sum() > 0:
                    rec["blend"] = blend / blend.sum()
            if self.judge is not None:
                shortlist, scores = self.judge
                z = scores - scores.max()
                e = np.exp(z)
                e /= e.sum()
                judge_p = np.zeros(N_SLOTS, np.float64)
                for a, p in zip(shortlist, e):
                    judge_p[a] = p
                rec["judge"] = judge_p
            self.record[idx] = rec
            return chosen

        # The router scores a shortlist with the critic one game at a time: a tiny network call per game, and the call
        # recomputes the same state embedding for every candidate. The critic of this ensemble is discrete, so one pass gives the
        # value of EVERY action of a state. Computing that table for all games of the step at once, then reading the shortlist
        # rows, is the same arithmetic in one call per member instead of one per game and candidate.
        self._critic_tables: dict[int, tuple] = {}
        score_one = policy._score_candidates_with_critic

        def critic_table(step) -> torch.Tensor:
            entry = self._critic_tables.get(id(step))
            if entry is not None and entry[0] is step:
                return entry[1]
            if len(self._critic_tables) > 64:
                self._critic_tables.clear()
            member_policy = step["member"].policy
            traj_emb = step["traj_emb"]  # [games, 1, d]
            games, actions = traj_emb.shape[0], policy.action_dim
            device = traj_emb.device
            gammas = len(member_policy.gammas)
            state = traj_emb.repeat_interleave(actions, dim=0)  # row g * actions + a: state of game g, action a
            action_tensor = torch.arange(actions, device=device).repeat(games)
            one_hot = F.one_hot(action_tensor, num_classes=actions).float().view(1, games * actions, 1, 1, actions)
            critic_actions = member_policy.actor.policy_dist.action_from_buffer(one_hot.repeat(1, 1, 1, gammas, 1))
            values = member_policy.critics(state, critic_actions)
            if hasattr(member_policy.critics, "bin_dist_to_raw_vals"):
                values = member_policy.critics.bin_dist_to_raw_vals(values)
            else:
                values = member_policy.popart(values, normalized=False)
            table = values.mean(dim=3)[0, :, 0, -1, 0].detach().view(games, actions).float().cpu()
            self._critic_tables[id(step)] = (step, table)
            return table

        def score_hook(step, batch_idx, shortlist):
            return critic_table(step)[batch_idx, shortlist]

        self.exact_critic = score_one  # (the original, kept for the equivalence check)
        policy._score_candidates_with_critic = score_hook

        policy._build_proposer_variants = build_hook
        policy._judge_shortlist = judge_hook
        policy._choose_action_for_batch = choose_hook


def label_batch(spec, policy, recorder, trajectories: list[dict], half=None) -> list[dict]:
    """Labels a batch of loaded trajectories in lockstep; returns one dict of arrays per trajectory."""
    import torch
    from metamon.interface import UniversalAction, UniversalState

    from metamon.recurrent import fast_tstep

    fast_tstep.reset(policy)  # (a no-op unless the fast encoder is installed) the cached embeddings belong to the previous games
    device = next(iter(m.device for m in policy.members))
    action_space = spec.action_space
    n_actions = action_space.gym_space.n
    lanes = len(trajectories)
    spaces = [copy.deepcopy(spec.observation_space) for _ in range(lanes)]
    # Exeggcute reads what the ORIGINAL parser produced (it was trained on that); see merge_legacy.py
    states = [[UniversalState.from_dict(copy.deepcopy(s)) for s in t["legacy_states"]] for t in trajectories]
    lengths = [len(s) - 1 for s in states]  # a decision per state but the last
    horizon = max(lengths)
    for sp in spaces:
        sp.reset()

    out = [
        {k: np.zeros((n, N_SLOTS), np.float32) for k in ("blend", "judge", "legal")}
        | {"chosen": np.full(n, -1, np.int16), "judge_calls": np.zeros(n, np.int8), "has_blend": np.zeros(n, bool), "has_judge": np.zeros(n, bool)}
        for n in lengths
    ]
    prev_action = np.zeros((lanes, n_actions), np.float32)
    prev_reward = np.zeros(lanes, np.float32)
    hidden = None
    for t in range(horizon):
        obs_list, legal_list = [], []
        for i in range(lanes):
            ti = min(t, lengths[i] - 1)  # finished lanes repeat their last decision; outputs are ignored
            s = states[i][ti]
            obs = spaces[i].state_to_obs(s)
            legal = sorted(
                {
                    action_space.action_to_agent_output(s, a)
                    for a in UniversalAction.maybe_valid_actions(s)
                }
            )
            mask = np.ones(n_actions, bool)
            mask[legal] = False
            obs["illegal_actions"] = mask
            obs_list.append(obs)
            legal_list.append(legal)
            if t < lengths[i]:
                out[i]["legal"][t, [a for a in legal if a < N_SLOTS]] = 1.0
        # The tensors stay on the CPU: the router reads them game by game on the host (legal actions, cycle keys, the previous
        # reward), and each such read of a device tensor waits for the device. The members move what they need to their own
        # device once per step.
        batch = {k: torch.as_tensor(np.stack([o[k] for o in obs_list]))[:, None] for k in obs_list[0]}
        rl2 = torch.as_tensor(np.concatenate([prev_reward[:, None], prev_action], axis=1)).view(lanes, 1, -1)
        time_idxs = torch.full((lanes, 1, 1), t, dtype=torch.long)
        recorder.record.clear()
        with torch.inference_mode(), torch.autocast(device.type, dtype=half or torch.bfloat16, enabled=half is not None and device.type in ("cuda", "cpu")):
            actions, hidden = policy.get_actions(batch, rl2, time_idxs, hidden_state=hidden, sample=False)
        actions = actions.reshape(lanes).tolist()
        for i in range(lanes):
            if t >= lengths[i]:
                continue
            rec = recorder.record.get(i)
            o = out[i]
            o["chosen"][t] = actions[i]
            if rec is not None:
                o["judge_calls"][t] = rec["judge_calls"]
                if "blend" in rec:
                    o["blend"][t] = rec["blend"]
                    o["has_blend"][t] = True
                if "judge" in rec:
                    o["judge"][t] = rec["judge"]
                    o["has_judge"][t] = True
            # a trajectory decision with one legal action has no router work; its label is that action
            if len(legal_list[i]) == 1:
                o["blend"][t, legal_list[i][0]] = 1.0
                o["has_blend"][t] = True
        # the next step's RL2 input: reward of this transition, action actually played in the replay
        for i in range(lanes):
            if t < lengths[i]:
                a = trajectories[i]["legacy_actions"][t]
                prev_action[i] = 0.0
                if a >= 0:
                    prev_action[i, action_space.action_to_agent_output(states[i][t], UniversalAction(action_idx=a))] = 1.0
                prev_reward[i] = float(spec.reward_function(states[i][t], states[i][t + 1]))
    return out


def pick_encoder(spec, policy, recorder, trajectories: list[dict], half, steps: int = 10) -> None:
    """Times `steps` decisions of the first games with each encoder (a first pass warms it up: compilation, allocator) and keeps the
    faster one; the fast one wins ties. The labels are discarded."""
    from metamon.recurrent import fast_tstep

    short = [{**t, "legacy_states": t["legacy_states"][: steps + 1], "legacy_actions": t["legacy_actions"][: steps + 1]} for t in trajectories]
    elapsed = {}
    for name in ("fast", "exact"):
        fast_tstep.enable(policy, name == "fast")
        label_batch(spec, policy, recorder, short, half)
        _sync(policy)
        start = time.perf_counter()
        label_batch(spec, policy, recorder, short, half)
        _sync(policy)
        elapsed[name] = time.perf_counter() - start
    use_fast = elapsed["fast"] <= 1.05 * elapsed["exact"]
    fast_tstep.enable(policy, use_fast)
    print(f"encoder: fast {elapsed['fast']:.1f}s, exact {elapsed['exact']:.1f}s for {steps} decisions of {len(short)} games -> using {'fast' if use_fast else 'exact'}", flush=True)


def _sync(policy) -> None:
    import torch

    for member in policy.members:
        if member.device.type == "cuda":
            torch.cuda.synchronize(member.device)
        elif member.device.type == "mps":
            torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cache", default=os.environ.get("METAMON_CACHE_DIR", "."))
    ap.add_argument("--batch", type=int, default=256, help="games labelled in lockstep (the members' fixed per-step cost is shared by all of them; the caches take about 11 MiB per game)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16", help="runs the members under CUDA autocast")
    ap.add_argument("--attention", choices=("sdpa", "flash"), default="sdpa", help="sdpa runs in fp32; flash needs flash_attn and half precision")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--encoder", choices=("auto", "fast", "exact"), default="auto",
        help="timestep encoder: the members' own (exact), the equivalent one in fast_tstep.py (fast), or whichever is faster on this machine, "
        "timed on a few steps of the first games (auto)",
    )
    ap.add_argument("--shard", default="0/1", help="i/n: label every n-th file starting at i")
    args = ap.parse_args()
    import torch

    files = sorted(glob.glob(os.path.join(args.input, "**", "*.json.lz4"), recursive=True))
    i, n = map(int, args.shard.split("/"))
    files = files[i::n]
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.output, exist_ok=True)
    files = [f for f in files if not os.path.exists(os.path.join(args.output, os.path.basename(f).split(".json")[0] + ".npz"))]
    spec, policy = load_ensemble(args.cache, args.device, args.threads, args.attention)
    half = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    recorder = Recorder(policy)
    from metamon.recurrent import fast_tstep

    fast_tstep.install(policy)
    fast_tstep.enable(policy, args.encoder != "exact")
    print(f"froze {fast_tstep.freeze_sigma_reparam(policy, half)} SigmaReparam layers", flush=True)

    # longest games first inside each batch group keeps the lockstep padding small
    lengths = {f: len(load_trajectory(f)["legacy_states"]) for f in files}
    files.sort(key=lambda f: lengths[f])
    if args.encoder == "auto" and files:
        pick_encoder(spec, policy, recorder, [load_trajectory(f) for f in files[len(files) // 2 : len(files) // 2 + args.batch]], half)
    done = decisions = 0
    start = time.time()
    for b in range(0, len(files), args.batch):
        group = files[b : b + args.batch]
        trajectories = [load_trajectory(f) for f in group]
        results = label_batch(spec, policy, recorder, trajectories, half=half)
        for f, r in zip(group, results):
            np.savez_compressed(os.path.join(args.output, os.path.basename(f).split(".json")[0] + ".npz"), **r)
        done += len(group)
        decisions += sum(len(r["chosen"]) for r in results)
        print(f"labelled {done}/{len(files)} games, {time.time() - start:.0f}s ({decisions / max(time.time() - start, 1e-9):.0f} decisions/s)", flush=True)


if __name__ == "__main__":
    main()
