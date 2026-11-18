"""One inference implementation for raw Tauros and branch-local search policies.

Callers may provide the previous transition's shaped reward.  Search callers
that cannot reconstruct a transition retain the explicit zero default.
All other recurrent inputs and observation memory belong to History, never
to the shared model. Tokenizers/model weights are immutable and shared.
"""
from __future__ import annotations

import copy
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from metamon_connector import SnapshotEncoder
from snapshot_effects import filter_probabilities


class ProfileLog:
    """Optional cumulative plain-text timings; nested timings are not additive.

    Accelerator work is synchronized only when profiling is enabled. Each line
    is flushed so a running daemon can be inspected without waiting for exit.
    """

    def __init__(self, path=None, device="cpu"):
        self.file = Path(path).open("a", encoding="utf-8") if path else None
        self.device = torch.device(device)
        self.counts = {}
        self.phase = ""
        if self.file:
            self.file.write("profile start elapsed=wall_seconds nested=yes\n")
            self.file.flush()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextlib.contextmanager
    def measure(self, name):
        if self.file is None:
            yield
            return
        name = f"{self.phase}.{name}" if self.phase else name
        self._sync()
        start = time.perf_counter()
        failed = False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            count, total, errors = self.counts.get(name, (0, 0.0, 0))
            count, total, errors = count + 1, total + elapsed, errors + int(failed)
            self.counts[name] = count, total, errors
            self.file.write(f"{name} count={count} elapsed_s={elapsed:.6f} total_s={total:.6f} mean_ms={total * 1000 / count:.3f} errors={errors}\n")
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def profile_measure(profile, name):
    return profile.measure(name) if profile is not None else contextlib.nullcontext()


def clone_space(space):
    result = copy.copy(space)
    if hasattr(space, "base_obs_space"):
        result.base_obs_space = clone_space(space.base_obs_space)
    for name in ("revealed_opponents", "_snapshot_effect_memory"):
        if hasattr(space, name):
            setattr(result, name, copy.deepcopy(getattr(space, name)))
    return result


EAGER_METHODS = ("forward", "training_forward", "_inference_with_cache",
                 "_forward_without_cache", "inner_forward", "_inner_forward_impl",
                 "actor_network_forward", "critic_network_forward")


def use_eager_inference(model):
    """Unwrap training compile decorators on this serving model instance.

    In particular VanillaAttention's native cache routine slices at end.max(),
    which otherwise triggers compiler work as the context grows. Keep exactly
    the installed attention math and native cache rolling behavior, without
    altering AMAGO classes, installed files, or the process-wide compiler.
    Also unwrap timestep/actor helpers so the first serving call cannot compile
    them lazily. The returned count includes torch.compiler.disable wrappers.
    """
    count = 0
    for module in model.modules():
        for name in EAGER_METHODS:
            method = getattr(module, name, None)
            original = getattr(method, "_torchdynamo_orig_callable", None)
            if original is not None:
                setattr(module, name, original if getattr(original, "__self__", None) is not None
                        else MethodType(original, module))
                count += 1
    return count


@dataclass
class History:
    space: object
    previous_action: np.ndarray
    hidden: object = None
    decision_index: int = 0
    pending: tuple[int, ...] | None = None
    hidden_shared: bool = False

    def fork(self):
        # Transformer KV caches are immutable between policy evaluations. Share
        # them across speculative branches and copy only when a branch is
        # actually advanced. This makes abandoned/pruned forks effectively free.
        self.hidden_shared = self.hidden is not None
        return History(clone_space(self.space), self.previous_action.copy(),
                       self.hidden, self.decision_index, self.pending,
                       self.hidden is not None)

    def writable_hidden(self):
        if self.hidden_shared:
            self.hidden = clone_hidden(self.hidden)
            self.hidden_shared = False
        return self.hidden

    def validate_commit(self, action: int):
        if self.pending is None or type(action) is not int or action not in self.pending:
            raise ValueError("action commit must follow a policy evaluation and be legal")

    def commit(self, action: int):
        self.validate_commit(action)
        self.previous_action.fill(0)
        self.previous_action[action] = 1
        self.decision_index += 1
        self.pending = None


def clone_hidden(hidden):
    """Clone mutable recurrent state, copying only the live KV-cache prefix."""
    if hidden is None:
        return None
    if not all(hasattr(hidden, name) for name in ("key_cache", "val_cache", "seq_lens")):
        return hidden.clone() if isinstance(hidden, torch.Tensor) else copy.deepcopy(hidden)
    cloned = copy.copy(hidden)
    cloned.seq_lens = hidden.seq_lens.clone()
    live = int(hidden.seq_lens.max().item())
    for name in ("key_cache", "val_cache"):
        source = getattr(hidden, name)
        cache = copy.copy(source)
        cache.data = torch.empty_like(source.data)
        if live:
            cache.data[:, :, :live].copy_(source.data[:, :, :live])
        setattr(cloned, name, cache)
    return cloned


def compact_attention_cache(hidden, window_left: int):
    """Keep only the capacity needed for current token plus a left window."""
    capacity = window_left + 2
    for name in ("key_cache", "val_cache"):
        cache = getattr(hidden, name)
        if cache.data.shape[2] <= capacity:
            continue
        cache.data = cache.data[:, :, :capacity].clone()
        cache.max_seq_len = capacity


def batch_hidden_states(states):
    if not states:
        raise ValueError("cannot batch an empty hidden-state list")
    first = states[0]
    combined = copy.copy(first)
    combined.seq_lens = torch.cat([state.seq_lens for state in states], dim=0)
    for name in ("key_cache", "val_cache"):
        cache = copy.copy(getattr(first, name))
        cache.data = torch.cat([getattr(state, name).data for state in states], dim=1)
        setattr(combined, name, cache)
    return combined


def split_hidden_state(hidden, count):
    output = []
    for index in range(count):
        item = copy.copy(hidden)
        item.seq_lens = hidden.seq_lens[index:index + 1]
        for name in ("key_cache", "val_cache"):
            cache = copy.copy(getattr(hidden, name))
            cache.data = getattr(hidden, name).data[:, index:index + 1]
            setattr(item, name, cache)
        output.append(item)
    return output


class TaurosInference:
    def __init__(self, root: Path, cache: Path, checkpoint=62, device="cpu", profile=None,
                 attention_backend="auto", model_name="TaurosV0"):
        self.profile = profile
        os.environ["METAMON_CACHE_DIR"] = str(cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
        sys.path.insert(0, str(root.resolve()))
        from metamon.rl.pretrained import get_pretrained_model
        self.device = torch.device(device)
        self.model_name = model_name
        if attention_backend == "auto" and model_name == "TaurosV0":
            # Tauros was trained with FlashAttention's (32, 0) sliding window.
            # PyTorch SDPA preserves that architecture on both CPU and MPS;
            # AMAGO's VanillaAttention is full-context and is not equivalent.
            attention_backend = "sdpa"
        elif attention_backend == "auto":
            # Other released transformer checkpoints also select CUDA-only
            # FlashAttention. SDPA preserves their configured context window.
            attention_backend = "sdpa"
        if attention_backend not in {None, "vanilla", "sdpa"}:
            raise ValueError("attention backend must be auto, vanilla, or sdpa")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available to this Python process")
        with contextlib.redirect_stdout(sys.stderr):
            factory = get_pretrained_model(model_name)
            experiment = factory.initialize_agent(
                checkpoint=checkpoint,
                log=False,
                action_temperature=1.0,
                attention_backend=attention_backend,
            )
        self.attention_backend = attention_backend or "native"
        self.policy = experiment.policy.to(self.device).eval()
        self.reward_function = factory.reward_function
        # AMAGO compiles training/full-sequence helpers. Recurrent serving has
        # changing cache state and tiny sequence dimensions; on MPS these
        # wrappers graph-break around cache management and are slower than eager.
        self.eager_decorators_unwrapped = use_eager_inference(self.policy)
        self.encoder = SnapshotEncoder(root, factory.observation_space)
        self.encoder.duel_extras = True  # (repo copy) Toxic and Disable as native metamon shows them
        self.action_count = factory.action_space.gym_space.n
        self.attention_window = None
        if self.attention_backend == "sdpa":
            from metamon.rl.sdpa_attention import SDPAAttention
            windows = {
                tuple(module.window_size)
                for module in self.policy.modules()
                if isinstance(module, SDPAAttention)
            }
            if not windows:
                # Recurrent Metamon checkpoints accept the compatibility
                # argument but do not contain an attention module.
                self.attention_backend = "native"
            elif len(windows) != 1:
                raise RuntimeError(f"expected one SDPA window configuration, found {windows}")
            else:
                self.attention_window = next(iter(windows))

    @property
    def inference_traj_encoder(self):
        return (self.policy._tortoise_traj_encoder
                if getattr(self.policy, "use_tortoise_for_inference", False)
                else self.policy.traj_encoder)

    def begin(self):
        space = clone_space(self.encoder.space)
        space.reset()
        # None selects AMAGO's training/full-sequence path and never creates a
        # cache. Initialize the exact trajectory encoder used by get_actions.
        hidden = self.inference_traj_encoder.init_hidden_state(1, self.device)
        if hidden is None and getattr(self, "model_name", "TaurosV0") == "TaurosV0":
            raise RuntimeError("Tauros trajectory encoder did not initialize an inference cache")
        if hidden is not None and getattr(self, "attention_backend", "vanilla") == "sdpa":
            compact_attention_cache(hidden, self.attention_window[0])
        return History(space, np.zeros(self.action_count, dtype=np.float32), hidden)

    def inputs(self, snapshot, legal, history, reward=0.0, state=None):
        if history.pending is not None:
            raise ValueError("previous policy evaluation has not been committed")
        if not legal or len(set(legal)) != len(legal) or any(type(i) is not int or i < 0 or i >= self.action_count for i in legal):
            raise ValueError("invalid Tauros legal action indices")
        obs = (
            self.encoder.observation(snapshot, history.space)
            if state is None
            else history.space.state_to_obs(state)
        )
        obs["illegal_actions"] = np.ones(self.action_count, dtype=bool)
        obs["illegal_actions"][legal] = False
        tensors = {k: torch.as_tensor(v, device=self.device)[None, None] for k, v in obs.items()}
        rl2 = np.concatenate(
            (np.asarray([reward], dtype=np.float32), history.previous_action)
        )
        return tensors, torch.as_tensor(rl2, device=self.device).view(1, 1, -1), torch.tensor([[[history.decision_index]]], device=self.device)

    def probabilities(self, snapshot, legal, history, reward=0.0, state=None):
        with profile_measure(self.profile, "encoding.tauros"):
            tensors, rl2, time = self.inputs(
                snapshot, legal, history, reward=reward, state=state
            )
        with profile_measure(self.profile, "forward.tauros"), torch.inference_mode():
            history.writable_hidden()
            # Match MultiTaskAgent.get_actions, including tortoise inference.
            if getattr(self.policy, "use_tortoise_for_inference", False):
                step = self.policy._tortoise_tstep_encoder(obs=tensors, rl2s=rl2)
                embedding, hidden = self.policy._tortoise_traj_encoder(step, time_idxs=time, hidden_state=history.hidden)
                actor = self.policy._tortoise_actor
            else:
                embedding, hidden = self.policy.get_state_embedding(obs=tensors, rl2s=rl2, time_idxs=time, hidden_state=history.hidden)
                actor = self.policy.actor
            dist = actor(embedding, straight_from_obs={k: tensors[k] for k in self.policy.pass_obs_keys_to_actor})
            probabilities = dist.probs[0, -1, -1, legal].detach().float().cpu().numpy()
        mass = float(probabilities.sum())
        if not np.isfinite(probabilities).all() or mass <= 0:
            raise ValueError("Tauros legal actions have no finite probability mass")
        history.hidden = hidden
        history.pending = tuple(legal)
        normalized = (probabilities / mass).tolist()
        if os.environ.get("TAUROS_FILTER", "0") != "1":
            # (repo copy) native metamon on Showdown masks illegal actions only; the strategic filter (no Reflect while Reflect
            # is up, no boosting at +6) made a different policy, so it is off unless TAUROS_FILTER=1
            return normalized
        filtered, _ = filter_probabilities(normalized, legal, snapshot)
        return filtered

    def probabilities_batch(self, snapshots, legal_lists, histories):
        if not snapshots or not (len(snapshots) == len(legal_lists) == len(histories)):
            raise ValueError("Tauros batch inputs must be non-empty and aligned")
        with profile_measure(self.profile, "encoding.tauros_batch"):
            encoded = [
                self.inputs(snapshot, legal, history)
                for snapshot, legal, history in zip(snapshots, legal_lists, histories)
            ]
            tensor_keys = encoded[0][0].keys()
            if any(item[0].keys() != tensor_keys for item in encoded[1:]):
                raise ValueError("Tauros batch observations have inconsistent fields")
            tensors = {
                key: torch.cat([item[0][key] for item in encoded], dim=0)
                for key in tensor_keys
            }
            rl2 = torch.cat([item[1] for item in encoded], dim=0)
            time_idxs = torch.cat([item[2] for item in encoded], dim=0)
            hidden = batch_hidden_states([history.hidden for history in histories])
        with profile_measure(self.profile, "forward.tauros_batch"), torch.inference_mode():
            if getattr(self.policy, "use_tortoise_for_inference", False):
                step = self.policy._tortoise_tstep_encoder(obs=tensors, rl2s=rl2)
                embedding, hidden = self.policy._tortoise_traj_encoder(
                    step, time_idxs=time_idxs, hidden_state=hidden
                )
                actor = self.policy._tortoise_actor
            else:
                embedding, hidden = self.policy.get_state_embedding(
                    obs=tensors, rl2s=rl2, time_idxs=time_idxs, hidden_state=hidden
                )
                actor = self.policy.actor
            dist = actor(
                embedding,
                straight_from_obs={key: tensors[key] for key in self.policy.pass_obs_keys_to_actor},
            )
            all_probabilities = dist.probs[:, -1, -1].detach().float().cpu().numpy()
        split = split_hidden_state(hidden, len(histories))
        output = []
        for row, snapshot, legal, history, next_hidden in zip(
            all_probabilities, snapshots, legal_lists, histories, split
        ):
            probabilities = row[legal]
            mass = float(probabilities.sum())
            if not np.isfinite(probabilities).all() or mass <= 0:
                raise ValueError("Tauros legal actions have no finite probability mass")
            history.hidden = next_hidden
            history.hidden_shared = False
            history.pending = tuple(legal)
            normalized = (probabilities / mass).tolist()
            filtered, _ = filter_probabilities(normalized, legal, snapshot)
            output.append(filtered)
        return output

    def legal_actions(self, snapshot, state=None):
        from metamon.interface import consistent_move_order, consistent_pokemon_order
        if state is None:
            state = self.encoder.state(snapshot)
        moves = consistent_move_order(state.player_active_pokemon.moves)
        switches = consistent_pokemon_order(state.available_switches)
        # (repo copy) as metamon on Showdown (UniversalAction.definitely_valid_actions / action_idx_to_BattleOrder): when the
        # only move is Struggle, every move index (one per known move, up to 4) is Struggle (Recharge: every index), so the
        # policy's mass on all of them goes to it (mapping it to index 0 alone moved that mass to the switches); the index sampled is the one
        # committed as the previous action. "no move" (the pass while the opponent replaces a fainted Pokemon) is index 0.
        all_moves = list(range(min(4, len(moves)))) or [0]
        legal = snapshot["legal"]
        result = {}
        for response in legal:
            name = response.lower()
            if name.startswith("switch "):
                indices = [next((4 + i for i, mon in enumerate(switches) if mon.name == name[7:]), None)]
            elif name == "struggle":
                indices = all_moves
            elif name == "recharge":
                # every index is Recharge, the switch ones too (metamon's state lists the bench whether or not it is trapped)
                indices = all_moves + [4 + i for i in range(len(switches))]
            elif name == "no move":
                indices = [0]
            else:
                indices = [next((i for i, move in enumerate(moves) if move.name == name), None)]
                # a locked move (Thrash, Rage, a two-turn move) called by Metronome / Mirror Move is not in the moveset; as the
                # only option its index does not matter to the outcome
                if indices == [None] and len(legal) == 1:
                    indices = [0]
            for index in indices:
                if index is None or index in result:
                    raise ValueError(f"cannot uniquely map legal action: {response}")
                result[index] = response
        return result


class ReplayGRUInference:
    """Replay-trained Metamon IL actor, NOT a Tauros-distilled policy.

    The checkpoint stores a GRUModel object and its tokenizer. Its feed-forward
    turn embedding and GRU consume DefaultObservationSpace text/numbers only;
    previous_action/decision_index are retained for the RPC commit contract but
    are not model inputs. Each History owns the observation memory and hidden
    state. Calling constituent layers avoids the training torch.compile wrapper.
    """

    def __init__(self, root: Path, checkpoint: Path, device="cpu", profile=None):
        sys.path.insert(0, str(root.resolve()))
        from metamon.interface import DefaultObservationSpace, TokenizedObservationSpace

        self.device = torch.device(device)
        self.profile = profile
        self.policy = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if not all(hasattr(self.policy, key) for key in ("turn_embedding", "rnn", "out")):
            raise TypeError("rollout checkpoint must be a serialized Metamon GRUModel")
        if not isinstance(self.policy.rnn, torch.nn.GRU):
            raise TypeError("rollout checkpoint must contain a GRU actor")
        self.policy.eval().to(self.device)
        embedding = self.policy.turn_embedding
        embedding.tokenizer.freeze()
        # Use the exact checkpoint vocabulary, not the Tauros tokenizer.
        space = TokenizedObservationSpace(base_obs_space=DefaultObservationSpace(),
                                          tokenizer=embedding.tokenizer)
        self.encoder = SnapshotEncoder(root, space)
        self.action_count = int(self.policy.out.out_features)

    def begin(self):
        space = clone_space(self.encoder.space)
        space.reset()
        return History(space, np.zeros(self.action_count, dtype=np.float32))

    def probabilities(self, snapshot, legal, history):
        if history.pending is not None:
            raise ValueError("previous policy evaluation has not been committed")
        if not legal or any(type(i) is not int or i < 0 or i >= self.action_count for i in legal) or len(set(legal)) != len(legal):
            raise ValueError("invalid GRU legal action indices")
        # Encoding updates revealed-opponent memory; publish it only on success.
        space = clone_space(history.space)
        with profile_measure(self.profile, "encoding.gru"):
            obs = self.encoder.observation(snapshot, space)
            tokens = torch.as_tensor(obs["text_tokens"], device=self.device, dtype=torch.long).view(1, 1, -1)
            numbers = torch.as_tensor(obs["numbers"], device=self.device, dtype=torch.float32).view(1, 1, -1)
            embedding = self.policy.turn_embedding
            if tokens.shape[-1] != embedding.text_features or numbers.shape[-1] != embedding.numerical_features:
                raise ValueError("GRU checkpoint does not match DefaultObservationSpace features")
            vocab_size = embedding.token_embedding.text_emb.num_embeddings
            # Metamon shifts IDs by +1: -1 is its valid unknown token.
            if (tokens < -1).any() or (tokens >= vocab_size - 1).any() or not torch.isfinite(numbers).all():
                raise ValueError("invalid GRU observation features")
        with profile_measure(self.profile, "forward.gru"), torch.inference_mode():
            embedded = embedding(token_inputs=tokens, numerical_inputs=numbers)
            recurrent, hidden = self.policy.rnn(embedded, history.hidden)
            logits = self.policy.out(recurrent)[0, -1, legal]
            probabilities = torch.softmax(logits, dim=-1).float().cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("GRU legal actions have no finite probability mass")
        history.space = space
        history.hidden = hidden.detach()
        history.pending = tuple(legal)
        return probabilities.tolist()
