//! Determinized UCT search over simultaneous moves for Generation I battles.
//!
//! Each determinization completes the opponent's unrevealed move slots by naive completion
//! (sampling from per-species candidate moves), then a tree is grown with UCB1 on each side's
//! marginal action values. Leaves are scored by an `Evaluator`: random rollouts by default, or a
//! learned value function through the same hook.

const std = @import("std");
const pkmn = @import("pkmn");
const vm = @import("value_model.zig");
const know = @import("knowledge.zig");
const pathmod = @import("path.zig");
const chance_enum = @import("chance_enum.zig");
const remote = @import("remote.zig");
const rnative = @import("rnative.zig");
const rmodel = @import("rmodel.zig");
const universe = @import("universe.zig");

test {
    _ = chance_enum;
}

pub const Battle = pkmn.gen1.Battle(pkmn.gen1.PRNG);
pub const Move = pkmn.gen1.Move;
pub const MAXC = pkmn.MAX_CHOICES;
const MAX_DEPTH = 64;

/// Returns P1's expected outcome in [0, 1] (1 = P1 wins) for a non-terminal battle.
pub const Evaluator = *const fn (battle: *const Battle, result: pkmn.Result, rand: *std.Random) f32;

pub const Config = struct {
    iterations: u32 = 1200,
    worlds: u32 = 4,
    exploration: f32 = 1.4,
    max_depth: u32 = 24,
    /// Opponent move slots that are known to be in the set, per party slot.
    revealed: [6][4]bool = @splat(@splat(false)),
    evaluator: Evaluator = rolloutValue,
    /// Subtracted from P1's reward each time P1's simulated choice is a switch. Small enough not
    /// to change which line wins, but enough to break ties between switching and attacking.
    switch_penalty: f32 = 0.01,
    /// When set, leaves are scored by the learned value model along each node's path (see
    /// path.zig) instead of by `evaluator`.
    model: ?*const vm.Model = null,
    vocab: ?*const std.StringHashMap(u32) = null,
    /// The observer's knowledge at the root decision (merged along each path below it).
    root_knowledge: know.Knowledge = .{},
    /// The real game's path at the root decision. Its history and memories are the root's, so
    /// the root's value sees the game so far.
    root_path: ?*const pathmod.Path = null,
    /// Number of most recent real steps the root's window keeps (0 = all of them).
    root_window: usize = 16,
    /// Opponent universes to search in (one per determinization). When empty, each world is the
    /// true battle with naive completion.
    universes: []const Battle = &.{},
    /// When set (with a model), every node gets the epoch-20 actor's distribution over each side's
    /// options, computed from the same forward pass as its value, and PUCT uses it at every depth.
    actor_priors: bool = false,
    /// How the final root action is chosen: most-visited, or the bot2 rule
    /// `log(prior) + root_value_scale * (Q - V)`.
    root_final: FinalRule = .visits,
    root_value_scale: f32 = 50,
    /// Adaptive racing at the root: after each round, P1 root actions whose paired deficit to the
    /// leader is significant (over the universes) are eliminated, and the budget goes to the rest.
    root_racing: bool = false,
    race_round: u32 = 25,
    race_z: f32 = 2,
    adaptive_candidates: usize = 4,
    adaptive_min_sweeps: u32 = 6,
    adaptive_confidence: f32 = 1.5,
    adaptive_margin: f32 = 0.01,
    adaptive_variance_floor: f32 = 0.03,
    sample_opponent: bool = false,
    first_play_value: bool = false,
    fpu_reduction: f32 = 0.15,
    block_history: bool = false,
    /// Wide root (adaptive only): each own action is valued against EVERY legal opponent reply in
    /// every world, combined with the opponent actor's probabilities, instead of one sampled reply.
    wide: bool = false,
    /// Fresh worlds, not used to pick the challenger. An override must also beat the actor on them.
    confirm_universes: []const Battle = &.{},
    /// Variance reduction (wide only). Chance draws per (action, reply) cell per sweep, all sharing the
    /// sweep's seeds across own actions. At depth 1 repeated draws mostly land on already-expanded
    /// chance children, so extra draws cost an engine update, not a forward pass.
    chance_samples: u32 = 1,
    /// Replies enumerated in probability order until this mass is covered (at least two); one more
    /// reply is drawn from the remaining mass and weighted by it (unbiased). 1 = enumerate all.
    reply_mass: f32 = 1,
    /// Children carry only the observer's value path, not the opponent-perspective actor path (only
    /// valid with max_depth 1, where children are never expanded).
    leaf_only_value: bool = false,
    /// Policy continuation at leaves: both sides play their actor's top legal option for this many
    /// more turns (chance sampled) before the value model is read. 0 = read the value immediately.
    continuation: u32 = 0,
    /// Exact chance (wide only): each (own action, reply) cell's turn is split into its important
    /// chance outcomes, weighted by their probabilities (chance_enum.zig), instead of
    /// `chance_samples` random draws. Repeated sweeps of one world then add nothing, so adaptive
    /// racing stops after one round over the worlds.
    exact_chance: bool = false,
    chance_min_path: f32 = 0.05,
    chance_max_leaves: u8 = 16,
    /// Threads of the kl_nash root (the worlds are independent: each world's cells run on one thread, with its own memory and
    /// its seed drawn as in one thread, so the result does not depend on the count). Only with the native recurrent model.
    threads: u32 = 1,
    /// The model is the recurrent one (remote.zig: values and actor logits from tools/recurrent_search_server.py) instead
    /// of `model`; `root_path` must then be the real game's P1 path on the server.
    recurrent: bool = false,
    /// The real game's decision points so far, oldest first, the last one the root (with the choices played from each). P2's
    /// root chain in each world replays them from P2's side with that world's opponent team (universe.pastWorld), so the
    /// opponent's actor sees the game so far; empty: P2's chain starts at the root, with no history.
    history: []const HistoryPoint = &.{},
    /// KL-anchored root game (root_final = .kl_nash): our strategy is anchored to our actor with
    /// temperature `kl_tau`, the opponent's (one per world) to its actor in that world with `kl_tau_opp`.
    kl_tau: f32 = 0.02,
    kl_tau_opp: f32 = 0.02,
    /// Play the anchored equilibrium's most likely action instead of drawing from it.
    kl_argmax: bool = false,
};

pub const FinalRule = enum { visits, prior_value, nash, adaptive, kl_nash };

/// One decision point of the real game: the state, the update result there, the observer's knowledge (P1's orientation),
/// and the choices played from it (unused for the last point, the root).
pub const HistoryPoint = struct {
    battle: Battle,
    result: pkmn.Result,
    knowledge: know.Knowledge,
    c1: pkmn.Choice = .{},
    c2: pkmn.Choice = .{},
};

/// Whether nodes carry model paths (the windowed model or the recurrent one).
/// SEARCH_NO_BATCH=1: the recurrent model's chance leaves one at a time (for checking the batched evaluation).
var no_batch_flag: ?bool = null;
fn noBatch() bool {
    if (no_batch_flag == null) no_batch_flag = if (std.c.getenv("SEARCH_NO_BATCH")) |v| v[0] == '1' else false;
    return no_batch_flag.?;
}

fn modelOn(cfg: Config) bool {
    return cfg.model != null or cfg.recurrent;
}

pub const Outcome = struct {
    depth_total: u64 = 0,
    max_depth: usize = 0,
    reused: u64 = 0,
    simulations: u32 = 0,
    sweeps: u32 = 0,
    actor_choice: u8 = 0,
    override_lower: f32 = 0,
    early_stop: bool = false,
    confirm_lower: f32 = 0,
    confirm_ran: bool = false,
    /// Chance leaves evaluated (exact chance).
    leaves: u64 = 0,
    /// The root strategy the choice was drawn from (kl_nash), and the anchored game's value.
    strategy: [MAXC]f32 = @splat(0),
    game_value: f32 = 0,
    residual: f32 = 0,
    choice: u8,
    visits: [MAXC]f32,
    /// Mean P1 value per root action (0 for unvisited).
    value: [MAXC]f32,
};

const Node = struct {
    battle: Battle,
    result: pkmn.Result,
    n: [2]u8,
    choices: [2][MAXC]pkmn.Choice,
    visits: [MAXC][MAXC]u32,
    /// Sum of P1 rewards for each (P1 action, P2 action) pair.
    wins: [MAXC][MAXC]f32,
    child: [MAXC][MAXC]u32,
    sibling: u32 = 0,
    terminal: bool,
    terminal_value: f32,
    initial_value: f32 = 0.5,
    /// The observer's path state at this node (only when a value model is configured).
    path: ?pathmod.Path = null,
    /// The same node from P2's perspective (only when priors are on).
    path2: ?pathmod.Path = null,
    /// Actor distributions over each side's options at this node (valid when `has_prior`).
    prior: [2][MAXC]f32 = @splat(@splat(0)),
    has_prior: bool = false,
    /// Value after a policy continuation from this node (computed once, when first needed).
    cont_value: f32 = 0.5,
    has_cont: bool = false,
};

const Tree = struct {
    allocator: std.mem.Allocator,
    nodes: []Node,
    count: u32,
    depth_total: u64 = 0,
    max_depth: usize = 0,
    reused: u64 = 0,
    leaves: u64 = 0,

    fn reset(self: *Tree) void {
        self.count = 1; // node 0 is reserved so that child == 0 means "no child"
    }

    /// Creates a node for `battle` at a decision point described by `result`.
    fn alloc(self: *Tree, battle: Battle, result: pkmn.Result) u32 {
        const idx = self.count;
        self.count += 1;
        const node = &self.nodes[idx];
        node.battle = battle;
        node.result = result;
        node.visits = @splat(@splat(0));
        node.wins = @splat(@splat(0));
        node.child = @splat(@splat(0));
        node.terminal = false;
        node.terminal_value = 0.5;
        node.initial_value = 0.5;
        node.path = null;
        node.path2 = null;
        node.has_prior = false;
        node.has_cont = false;
        node.sibling = 0;
        if (result.type != .None) {
            node.terminal = true;
            node.terminal_value = terminalValue(result);
            node.n = .{ 0, 0 };
            return idx;
        }
        var b = battle;
        node.n[0] = b.choices(.P1, result.p1, &node.choices[0]);
        node.n[1] = b.choices(.P2, result.p2, &node.choices[1]);
        if (node.n[0] == 0 or node.n[1] == 0) {
            node.terminal = true;
            node.terminal_value = 0.5;
        }
        return idx;
    }
};

fn terminalValue(result: pkmn.Result) f32 {
    return switch (result.type) {
        .Win => 1,
        .Lose => 0,
        else => 0.5,
    };
}

/// Heuristic leaf value: the difference in remaining HP fractions across each side's party,
/// squashed into (0, 1). Unlike random rollouts it always prefers lines that deal damage, so the
/// search does not stall when every rollout reaches the same result.
pub fn hpValue(battle: *const Battle, result: pkmn.Result, rand: *std.Random) f32 {
    _ = result;
    _ = rand;
    var diff: f32 = 0;
    for (battle.sides, 0..) |side, i| {
        var frac: f32 = 0;
        for (side.pokemon) |pk| {
            if (pk.stats.hp == 0) continue;
            frac += @as(f32, @floatFromInt(pk.hp)) / @as(f32, @floatFromInt(pk.stats.hp));
        }
        diff += if (i == 0) frac else -frac;
    }
    return 1.0 / (1.0 + @exp(-diff));
}

/// Plays random moves for both sides from a decision point until the battle ends.
pub fn rolloutValue(battle: *const Battle, result: pkmn.Result, rand: *std.Random) f32 {
    var b = battle.*;
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    var choices1: [MAXC]pkmn.Choice = undefined;
    var choices2: [MAXC]pkmn.Choice = undefined;
    var r = result;
    var turns: u32 = 0;
    while (r.type == .None) : (turns += 1) {
        if (turns >= 300) return 0.5;
        const n1 = b.choices(.P1, r.p1, &choices1);
        const n2 = b.choices(.P2, r.p2, &choices2);
        if (n1 == 0 or n2 == 0) return 0.5;
        const c1 = choices1[rand.uintLessThan(u8, n1)];
        const c2 = choices2[rand.uintLessThan(u8, n2)];
        r = b.update(c1, c2, &options) catch return 0.5;
    }
    return terminalValue(r);
}

fn firstPlay(value: f32, side: usize, prior: f32, reduction: f32) f32 {
    return @max(0, (if (side == 0) value else 1 - value) - reduction * @sqrt(prior));
}

/// Index of `side`'s action to explore next by UCB1 over its marginal (averaged over the other
/// side's actions). Unvisited actions are tried first.
fn pickAction(tree: *const Tree, node: *const Node, side: usize, c: f32, prior: ?*const [MAXC]f32, mask: ?*const [MAXC]bool, fpu: ?f32) u8 {
    const mine_n = node.n[side];
    const theirs_n = node.n[1 - side];
    var total: f32 = 0;
    var counts: [MAXC]f32 = @splat(0);
    var sums: [MAXC]f32 = @splat(0);
    var a: u8 = 0;
    while (a < mine_n) : (a += 1) {
        var b: u8 = 0;
        while (b < theirs_n) : (b += 1) {
            const i: usize = if (side == 0) a else b;
            const j: usize = if (side == 0) b else a;
            const n: f32 = @floatFromInt(node.visits[i][j]);
            const w1 = node.wins[i][j];
            counts[a] += n;
            sums[a] += if (side == 0) w1 else n - w1;
        }
        total += counts[a];
    }
    _ = tree;
    var best: u8 = 0;
    var best_score: f32 = -std.math.inf(f32);
    a = 0;
    if (prior) |p| {
        // PUCT: mean value plus an exploration bonus scaled by the actor's prior
        while (a < mine_n) : (a += 1) {
            if (mask) |m| if (!m[a]) continue;
            const mean = if (counts[a] == 0) (if (fpu) |reduction| firstPlay(node.initial_value, side, p[a], reduction) else 0) else sums[a] / counts[a];
            const score = mean + c * p[a] * @sqrt(total + 1) / (1 + counts[a]);
            if (score > best_score) {
                best_score = score;
                best = a;
            }
        }
        return best;
    }
    while (a < mine_n) : (a += 1) {
        if (mask) |m| if (!m[a]) continue;
        const score = if (counts[a] == 0)
            std.math.inf(f32)
        else
            sums[a] / counts[a] + c * @sqrt(@log(total) / counts[a]);
        if (score > best_score) {
            best_score = score;
            best = a;
        }
    }
    return best;
}

const Step = struct { node: u32, a1: u8, a2: u8 };

/// Leaf value of node `idx`: the model's win probability along its path when a model is set,
/// otherwise the configured evaluator.
fn leafValue(tree: *Tree, idx: u32, cfg: Config, rand: *std.Random) !f32 {
    const node = &tree.nodes[idx];
    if (node.terminal) return node.terminal_value;
    if (cfg.continuation > 0 and modelOn(cfg) and node.path != null and node.path2 != null) {
        if (!node.has_cont) {
            node.cont_value = try continuationValue(tree.allocator, node, cfg, rand);
            node.has_cont = true;
        }
        return node.cont_value;
    }
    if (modelOn(cfg)) {
        if (node.path) |*p| return @floatCast(p.value());
    }
    return cfg.evaluator(&node.battle, node.result, rand);
}

/// Legal option with the highest actor logit for `side` (the first option when none maps to a slot).
fn greedyChoice(battle: *const Battle, side: usize, choices: []const pkmn.Choice, logits: *const [vm.ACTIONS]f32) pkmn.Choice {
    var best = choices[0];
    var best_logit: f32 = -std.math.inf(f32);
    for (choices) |choice| {
        const index = know.actorIndex(battle, side, choice) orelse continue;
        if (logits[index] > best_logit) {
            best_logit = logits[index];
            best = choice;
        }
    }
    return best;
}

/// Both sides play their actor's top option for `cfg.continuation` turns from `node`, with fresh
/// chance each turn; returns the terminal result or P1's value at the end. The opponent's actor reads
/// its own-perspective chain, so it plays from this world's (sampled) team.
fn continuationValue(allocator: std.mem.Allocator, node: *const Node, cfg: Config, rand: *std.Random) !f32 {
    // (one draw from the search's stream seeds this continuation's own chance, so a batched evaluation of several leaves
    // plays exactly the same continuations)
    var stream = std.Random.DefaultPrng.init(rand.int(u64));
    const own = stream.random();
    var battle = node.battle;
    var result = node.result;
    // Paths point at their parents, so every continuation step's path lives in the arena (a stack
    // variable reused across steps would make a path its own parent).
    var p1: *const pathmod.Path = &node.path.?;
    var p2: *const pathmod.Path = &node.path2.?;
    var buf1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var buf2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    for (0..cfg.continuation) |step| {
        const n1 = battle.choices(.P1, result.p1, &buf1);
        const n2 = battle.choices(.P2, result.p2, &buf2);
        if (n1 == 0 or n2 == 0) return 0.5;
        const c1 = greedyChoice(&battle, 0, buf1[0..n1], &p1.actor);
        const c2 = greedyChoice(&battle, 1, buf2[0..n2], &p2.actor);
        var next = battle;
        next.rng = pkmn.gen1.PRNG.init(own.int(u64));
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        const r = try next.update(c1, c2, &options);
        if (r.type != .None) return terminalValue(r);
        const p1_next = try allocator.create(pathmod.Path);
        p1_next.* = try pathmod.Path.turn(allocator, cfg.model, cfg.vocab, p1, &battle, &next, r, c1, c2, .{}, 0);
        // the opponent's chain is only needed to choose its next move
        if (step + 1 < cfg.continuation) {
            const p2_next = try allocator.create(pathmod.Path);
            p2_next.* = try pathmod.Path.turn(allocator, cfg.model, cfg.vocab, p2, &battle, &next, r, c1, c2, .{}, 1);
            p2 = p2_next;
        }
        p1 = p1_next;
        battle = next;
        result = r;
    }
    return @floatCast(p1.value());
}

/// The actor's distribution over each side's options at `node`: a softmax over the logits of the
/// options' action slots (the policy server's rule, without its dominated-action filter).
fn computePriors(node: *Node) void {
    if (node.path) |*path| node.initial_value = @floatCast(path.value());
    if (node.path == null or node.path2 == null) return;
    for (0..2) |s| {
        const logits = if (s == 0) node.path.?.actor else node.path2.?.actor;
        var max: f32 = -std.math.inf(f32);
        var vals: [MAXC]f32 = @splat(-std.math.inf(f32));
        for (0..node.n[s]) |k| {
            const idx = know.actorIndex(&node.battle, s, node.choices[s][k]) orelse continue;
            vals[k] = logits[idx];
            max = @max(max, vals[k]);
        }
        var total: f32 = 0;
        for (0..node.n[s]) |k| {
            node.prior[s][k] = if (std.math.isInf(vals[k])) 0 else @exp(vals[k] - max);
            total += node.prior[s][k];
        }
        for (0..node.n[s]) |k| node.prior[s][k] = if (total > 0) node.prior[s][k] / total else 1 / @as(f32, @floatFromInt(node.n[s]));
    }
    node.has_prior = true;
}

const Child = struct { idx: u32, existed: bool };

/// The child of node `cur` for the action pair (`a1`, `a2`) and the post-turn state `next`: an
/// existing chance child with the same state (the RNG aside), or a new node with its paths. Chance
/// outcomes share an action edge, but have separate descendant trees.
fn childFor(tree: *Tree, cur: u32, a1: u8, a2: u8, next: *const Battle, r: pkmn.Result, cfg: Config) !Child {
    return childNode(tree, cur, a1, a2, next, r, cfg, false);
}

/// childFor; with `deferPaths` a new child gets no model paths (the caller attaches them, batched).
fn childNode(tree: *Tree, cur: u32, a1: u8, a2: u8, next: *const Battle, r: pkmn.Result, cfg: Config, deferPaths: bool) !Child {
    const node = &tree.nodes[cur];
    var existing = node.child[a1][a2];
    while (existing != 0) : (existing = tree.nodes[existing].sibling) {
        var candidate = tree.nodes[existing].battle;
        candidate.rng = next.rng;
        if (std.meta.eql(candidate, next.*) and std.meta.eql(tree.nodes[existing].result, r)) break;
    }
    if (existing != 0) {
        tree.reused += 1;
        return .{ .idx = existing, .existed = true };
    }
    if (tree.count >= tree.nodes.len) return error.TreeFull;
    const child = tree.alloc(next.*, r);
    tree.nodes[child].sibling = node.child[a1][a2];
    node.child[a1][a2] = child;
    if (deferPaths) return .{ .idx = child, .existed = false };
    if (modelOn(cfg)) {
        if (!tree.nodes[child].terminal) {
            const parent_path = &tree.nodes[cur].path.?;
            tree.nodes[child].path = try pathmod.Path.turn(tree.allocator, cfg.model, cfg.vocab, parent_path, &tree.nodes[cur].battle, &tree.nodes[child].battle, r, node.choices[0][a1], node.choices[1][a2], .{}, 0);
            if (cfg.actor_priors and !cfg.leaf_only_value) {
                const parent_path2 = &tree.nodes[cur].path2.?;
                tree.nodes[child].path2 = try pathmod.Path.turn(tree.allocator, cfg.model, cfg.vocab, parent_path2, &tree.nodes[cur].battle, &tree.nodes[child].battle, r, node.choices[0][a1], node.choices[1][a2], .{}, 1);
                computePriors(&tree.nodes[child]);
            }
        }
    }
    return .{ .idx = child, .existed = false };
}

/// Exact-chance value of root cell (`a`, `b`): the turn's chance leaves (chance_enum.enumerate,
/// unforced events drawn from `seed`), each valued like a depth-1 leaf, weighted by probability.
fn exactCellValue(tree: *Tree, root: u32, cfg: Config, seed: u64, a: u8, b: u8, rand: *std.Random) !f32 {
    if (cfg.recurrent and !noBatch()) return exactCellValueBatched(tree, root, cfg, seed, a, b, rand);
    var leaves: [chance_enum.MAX_LEAVES]chance_enum.Leaf = undefined;
    const battle = tree.nodes[root].battle;
    const n = try chance_enum.enumerate(&battle, tree.nodes[root].choices[0][a], tree.nodes[root].choices[1][b], seed, .{ .min_path = cfg.chance_min_path, .max_leaves = cfg.chance_max_leaves }, &leaves);
    var total: f32 = 0;
    for (leaves[0..n]) |*leaf| {
        const c = try childFor(tree, root, a, b, &leaf.battle, leaf.result, cfg);
        total += leaf.weight * try leafValue(tree, c.idx, cfg, rand);
    }
    tree.leaves += n;
    tree.depth_total += n;
    tree.max_depth = @max(tree.max_depth, 1);
    tree.nodes[root].visits[a][b] += 1;
    tree.nodes[root].wins[a][b] += total;
    return total;
}

/// exactCellValue with the recurrent model: the cell's chance leaves are independent, so their paths (both sides) are made in
/// one request to the server, and their policy continuations advance together, one request per continuation turn. The same
/// values as one leaf at a time (up to the batch's rounding), with the continuations' chance drawn in another order.
fn exactCellValueBatched(tree: *Tree, root: u32, cfg: Config, seed: u64, a: u8, b: u8, rand: *std.Random) !f32 {
    const L = chance_enum.MAX_LEAVES;
    var leaves: [L]chance_enum.Leaf = undefined;
    const battle = tree.nodes[root].battle;
    const c1_root = tree.nodes[root].choices[0][a];
    const c2_root = tree.nodes[root].choices[1][b];
    const n = try chance_enum.enumerate(&battle, c1_root, c2_root, seed, .{ .min_path = cfg.chance_min_path, .max_leaves = cfg.chance_max_leaves }, &leaves);
    const allocator = tree.allocator;
    const with_p2 = cfg.actor_priors and !cfg.leaf_only_value;

    // the leaves' nodes, and the paths of the new ones in one batch
    var idx: [L]u32 = undefined;
    var fresh: [L]u32 = undefined;
    var n_fresh: usize = 0;
    for (leaves[0..n], 0..) |*leaf, i| {
        const c = try childNode(tree, root, a, b, &leaf.battle, leaf.result, cfg, true);
        idx[i] = c.idx;
        if (!c.existed and !tree.nodes[c.idx].terminal) {
            fresh[n_fresh] = c.idx;
            n_fresh += 1;
        }
    }
    if (n_fresh > 0) {
        const per: usize = if (with_p2) 2 else 1;
        const specs = try allocator.alloc(pathmod.Path.TurnSpec, n_fresh * per);
        for (fresh[0..n_fresh], 0..) |ci, k| {
            const child = &tree.nodes[ci];
            specs[k * per] = .{ .parent = &tree.nodes[root].path.?, .before = &tree.nodes[root].battle, .after = &child.battle, .result = child.result, .c1 = c1_root, .c2 = c2_root, .side = 0 };
            if (with_p2) specs[k * per + 1] = .{ .parent = &tree.nodes[root].path2.?, .before = &tree.nodes[root].battle, .after = &child.battle, .result = child.result, .c1 = c1_root, .c2 = c2_root, .side = 1 };
        }
        const paths = try allocator.alloc(pathmod.Path, specs.len);
        try pathmod.Path.turnBatch(allocator, cfg.model, cfg.vocab, specs, paths);
        for (fresh[0..n_fresh], 0..) |ci, k| {
            tree.nodes[ci].path = paths[k * per];
            if (with_p2) {
                tree.nodes[ci].path2 = paths[k * per + 1];
                computePriors(&tree.nodes[ci]);
            }
        }
    }

    // the leaves' values; the policy continuations of those that need one, advanced together
    var value: [L]f32 = undefined;
    var cont: [L]usize = undefined; // leaf indices whose continuation is computed here
    var n_cont: usize = 0;
    for (0..n) |i| {
        const node = &tree.nodes[idx[i]];
        if (node.terminal) {
            value[i] = node.terminal_value;
        } else if (cfg.continuation > 0 and node.path != null and node.path2 != null) {
            if (node.has_cont) {
                value[i] = node.cont_value;
            } else {
                var dup = false; // (two leaves can reach the same node: its continuation is computed once)
                for (cont[0..n_cont]) |j| dup = dup or idx[j] == idx[i];
                if (!dup) {
                    cont[n_cont] = i;
                    n_cont += 1;
                }
            }
        } else if (node.path) |*pp| {
            value[i] = @floatCast(pp.value());
        } else {
            value[i] = cfg.evaluator(&node.battle, node.result, rand);
        }
    }
    if (n_cont > 0) {
        var state_battle: [L]Battle = undefined;
        var state_result: [L]pkmn.Result = undefined;
        var p1: [L]*const pathmod.Path = undefined;
        var p2: [L]*const pathmod.Path = undefined;
        var done: [L]bool = @splat(false);
        var cv: [L]f32 = @splat(0.5);
        var before: [L]Battle = undefined;
        var streams: [L]std.Random.DefaultPrng = undefined;
        for (cont[0..n_cont], 0..) |i, k| {
            streams[k] = std.Random.DefaultPrng.init(rand.int(u64)); // (as continuationValue: one draw per leaf, in leaf order)
            const node = &tree.nodes[idx[i]];
            state_battle[k] = node.battle;
            state_result[k] = node.result;
            p1[k] = &node.path.?;
            p2[k] = &node.path2.?;
        }
        var buf1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var buf2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        for (0..cfg.continuation) |step| {
            const want_p2 = step + 1 < cfg.continuation;
            var specs: [2 * L]pathmod.Path.TurnSpec = undefined;
            var owner: [2 * L]usize = undefined;
            var ns: usize = 0;
            for (0..n_cont) |k| {
                if (done[k]) continue;
                const n1 = state_battle[k].choices(.P1, state_result[k].p1, &buf1);
                const n2 = state_battle[k].choices(.P2, state_result[k].p2, &buf2);
                if (n1 == 0 or n2 == 0) {
                    cv[k] = 0.5;
                    done[k] = true;
                    continue;
                }
                const c1 = greedyChoice(&state_battle[k], 0, buf1[0..n1], &p1[k].actor);
                const c2 = greedyChoice(&state_battle[k], 1, buf2[0..n2], &p2[k].actor);
                before[k] = state_battle[k];
                state_battle[k].rng = pkmn.gen1.PRNG.init(streams[k].random().int(u64));
                var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
                const r = try state_battle[k].update(c1, c2, &options);
                state_result[k] = r;
                if (r.type != .None) {
                    cv[k] = terminalValue(r);
                    done[k] = true;
                    continue;
                }
                specs[ns] = .{ .parent = p1[k], .before = &before[k], .after = &state_battle[k], .result = r, .c1 = c1, .c2 = c2, .side = 0 };
                owner[ns] = k;
                ns += 1;
                if (want_p2) {
                    specs[ns] = .{ .parent = p2[k], .before = &before[k], .after = &state_battle[k], .result = r, .c1 = c1, .c2 = c2, .side = 1 };
                    owner[ns] = k;
                    ns += 1;
                }
            }
            if (ns == 0) break;
            // (paths point at their parents, so each continuation step's paths live in the tree's arena)
            const paths = try allocator.alloc(pathmod.Path, ns);
            try pathmod.Path.turnBatch(allocator, cfg.model, cfg.vocab, specs[0..ns], paths);
            for (0..ns) |j| {
                if (specs[j].side == 0) p1[owner[j]] = &paths[j] else p2[owner[j]] = &paths[j];
            }
        }
        for (cont[0..n_cont], 0..) |i, k| {
            if (!done[k]) cv[k] = @floatCast(p1[k].value());
            tree.nodes[idx[i]].cont_value = cv[k];
            tree.nodes[idx[i]].has_cont = true;
        }
        for (0..n) |i| {
            const node = &tree.nodes[idx[i]];
            if (!node.terminal and node.has_cont) value[i] = node.cont_value;
        }
    }
    var total: f32 = 0;
    for (leaves[0..n], 0..) |*leaf, i| total += leaf.weight * value[i];
    tree.leaves += n;
    tree.depth_total += n;
    tree.max_depth = @max(tree.max_depth, 1);
    tree.nodes[root].visits[a][b] += 1;
    tree.nodes[root].wins[a][b] += total;
    return total;
}

fn simulate(tree: *Tree, root: u32, cfg: Config, rand: *std.Random, live: *const [MAXC]bool, forced_root: ?u8, forced_reply: ?u8) !f32 {
    var path: [MAX_DEPTH + 1]Step = undefined;
    var depth: usize = 0;
    var cur = root;
    var value: f32 = 0.5;
    while (true) {
        const node = &tree.nodes[cur];
        if (node.terminal) {
            value = node.terminal_value;
            break;
        }
        const a1 = if (cur == root and forced_root != null) forced_root.? else pickAction(tree, node, 0, cfg.exploration, if (node.has_prior) &node.prior[0] else null, if (cur == root) live else null, if (cfg.first_play_value) cfg.fpu_reduction else null);
        const a2 = if (cur == root and forced_reply != null) forced_reply.? else if (cfg.sample_opponent and node.has_prior) sampleStrategy(&node.prior[1], node.n[1], rand) else pickAction(tree, node, 1, cfg.exploration, if (node.has_prior) &node.prior[1] else null, null, if (cfg.first_play_value) cfg.fpu_reduction else null);
        path[depth] = .{ .node = cur, .a1 = a1, .a2 = a2 };
        depth += 1;
        // Resample chance on every traversal, independently of the real game's RNG.
        var next = node.battle;
        next.rng = pkmn.gen1.PRNG.init(rand.int(u64));
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        const r = try next.update(node.choices[0][a1], node.choices[1][a2], &options);
        const c = try childFor(tree, cur, a1, a2, &next, r, cfg);
        if (c.existed) {
            cur = c.idx;
            if (depth >= cfg.max_depth) {
                value = try leafValue(tree, cur, cfg, rand);
                break;
            }
            continue;
        }
        value = try leafValue(tree, c.idx, cfg, rand);
        break;
    }
    tree.depth_total += depth;
    tree.max_depth = @max(tree.max_depth, depth);
    for (path[0..depth]) |step| {
        const n = &tree.nodes[step.node];
        const penalty: f32 = if (n.choices[0][step.a1].type == .Switch) cfg.switch_penalty else 0;
        n.visits[step.a1][step.a2] += 1;
        n.wins[step.a1][step.a2] += value - penalty;
    }
    return value;
}

/// Opponent move candidates per species for naive completion (a demo table; in a full system this
/// comes from usage statistics for each species).
pub fn candidateMoves(species: pkmn.gen1.Species) []const Move {
    return switch (species) {
        .Tauros => &.{ .BodySlam, .HyperBeam, .Blizzard, .Earthquake, .Rest },
        .Chansey => &.{ .Reflect, .SeismicToss, .SoftBoiled, .ThunderWave, .IceBeam },
        .Snorlax => &.{ .BodySlam, .Reflect, .Rest, .IceBeam, .Earthquake },
        .Exeggutor => &.{ .SleepPowder, .Psychic, .Explosion, .DoubleEdge, .Stomp },
        .Starmie => &.{ .Recover, .ThunderWave, .Blizzard, .Thunderbolt, .Psychic },
        .Alakazam => &.{ .Psychic, .SeismicToss, .ThunderWave, .Recover, .Thunderbolt },
        else => &.{},
    };
}

/// Naive completion: fills P2's unrevealed move slots with random candidate moves for its species.
pub fn complete(battle: *Battle, revealed: [6][4]bool, rand: *std.Random) void {
    for (&battle.sides[1].pokemon, 0..) |*mon, i| {
        if (mon.species == .None) continue;
        const cands = candidateMoves(mon.species);
        if (cands.len == 0) continue;
        for (0..4) |j| {
            if (revealed[i][j] or mon.moves[j].id == .None) continue;
            var tries: u32 = 0;
            while (tries < 32) : (tries += 1) {
                const m = cands[rand.uintLessThan(usize, cands.len)];
                var taken = false;
                for (mon.moves) |slot| {
                    if (slot.id == m) taken = true;
                }
                if (!taken) {
                    const slot = pkmn.gen1.MoveSlot{ .id = m, .pp = Move.pp(m) };
                    mon.moves[j] = slot;
                    // The active Pokémon keeps its own copy of the move slots; keep it in sync.
                    if (battle.sides[1].order[0] == i + 1) battle.sides[1].active.moves[j] = slot;
                    break;
                }
            }
        }
    }
}

/// Runs the determinized search at a decision point of `truth`. `truth_result` is the request
/// state returned by the last update. Returns the best P1 choice index into the choice list that
/// `battle.choices(.P1, truth_result.p1, ...)` produces for the truth battle.
const MAX_WORLDS = 32;

/// Byte sizes of a search node and of its path (for memory-cost checks).
pub fn nodeBytes() usize {
    return @sizeOf(Node);
}
pub fn pathBytes() usize {
    return @sizeOf(pathmod.Path);
}

/// One search tree per universe. Every root action is evaluated in every universe (paired).
const Root = struct {
    trees: []Tree,
    roots: []u32,
};

/// Duration counters are private RNG outcomes, even on our own side. Draw a
/// latent state instead of allowing the search to inspect the real countdown.
/// These are broad priors; conditioning on public elapsed durations is future work.
fn randomizeHiddenTimers(world: *Battle, rand: *std.Random) void {
    for (&world.sides) |*side| {
        for (&side.pokemon) |*mon| {
            if (pkmn.gen1.Status.is(mon.status, .SLP) and !pkmn.gen1.Status.is(mon.status, .EXT))
                mon.status = pkmn.gen1.Status.slp(1 + rand.uintLessThan(u3, 7));
        }
        const v = &side.active.volatiles;
        if (v.Confusion) v.confusion = 1 + rand.uintLessThan(u3, 4);
        if (v.disable_move != 0) v.disable_duration = 1 + rand.uintLessThan(u4, 8);
        if (v.Binding or v.Bide or v.Thrashing)
            v.attacks = 1 + rand.uintLessThan(u3, if (v.Binding) @as(u3, 4) else @as(u3, 2));
    }
}

fn setupRoots(allocator: std.mem.Allocator, cfg: Config, truth: *const Battle, truth_result: pkmn.Result, per_world: u32, rand: *std.Random) !Root {
    const world_count: usize = if (cfg.universes.len > 0) cfg.universes.len else cfg.worlds;
    const trees = try allocator.alloc(Tree, world_count);
    const roots = try allocator.alloc(u32, world_count);
    // The real game's P1 root view is the same in every universe (only P2's hidden team differs), so
    // it is built once and shared.
    var shared_view: ?pathmod.Path = null;
    if (cfg.recurrent) {
        if (cfg.root_path == null) return error.RecurrentNeedsRootPath;
        shared_view = cfg.root_path.?.*; // (the recurrent state already holds the whole game)
    } else if (cfg.model != null and cfg.root_path != null) {
        const rp = cfg.root_path.?;
        const window = if (cfg.block_history) @min(rp.memory.step, 16 + rp.memory.step % 16) else cfg.root_window;
        shared_view = if (window == 0) rp.* else try pathmod.Path.rootView(allocator, cfg.model.?, rp, window);
        shared_view.?.block_window = cfg.block_history;
    }
    for (0..world_count) |w| {
        var world = if (cfg.universes.len > 0) cfg.universes[w] else truth.*;
        if (cfg.universes.len == 0) complete(&world, cfg.revealed, rand);
        randomizeHiddenTimers(&world, rand);
        trees[w] = Tree{ .allocator = allocator, .nodes = try allocator.alloc(Node, per_world + 2), .count = 1 };
        roots[w] = trees[w].alloc(world, truth_result);
        if (modelOn(cfg)) {
            const node = &trees[w].nodes[roots[w]];
            if (shared_view != null) {
                node.path = shared_view.?;
            } else {
                const snapshot = try know.observerSnapshot(allocator, &node.battle, cfg.root_knowledge, false);
                node.path = try pathmod.Path.extend(allocator, cfg.model.?, cfg.vocab.?, null, cfg.root_knowledge, snapshot);
            }
            if (cfg.actor_priors and !batchedOpponentRoots(cfg)) {
                node.path2 = try opponentRoot(allocator, cfg, &node.battle, truth_result);
                node.path2.?.block_window = cfg.block_history;
                computePriors(node);
            }
        }
    }
    if (modelOn(cfg) and cfg.actor_priors and batchedOpponentRoots(cfg)) {
        const worlds = try allocator.alloc(*const Battle, world_count);
        const paths = try allocator.alloc(pathmod.Path, world_count);
        for (0..world_count) |w| worlds[w] = &trees[w].nodes[roots[w]].battle;
        try opponentRoots(allocator, cfg, worlds, truth_result, paths);
        for (0..world_count) |w| {
            const node = &trees[w].nodes[roots[w]];
            node.path2 = paths[w];
            node.path2.?.block_window = cfg.block_history;
            computePriors(node);
        }
    }
    return .{ .trees = trees, .roots = roots };
}

/// With the native recurrent model (and a history to replay), the worlds' P2 root chains advance together, one batch per
/// past decision point (opponentRoots), instead of one world after the other (opponentRoot): the same paths.
fn batchedOpponentRoots(cfg: Config) bool {
    return cfg.recurrent and rnative.active != null and remote.active == null and cfg.history.len >= 2 and !noBatch();
}

/// opponentRoot for several worlds at once (the recurrent model replays the whole game).
fn opponentRoots(allocator: std.mem.Allocator, cfg: Config, worlds: []const *const Battle, truth_result: pkmn.Result, out: []pathmod.Path) !void {
    _ = truth_result; // (each point's own result is used; the last point is the root)
    const h = cfg.history;
    const W = worlds.len;
    const cur = try allocator.alloc(*pathmod.Path, W);
    const prev = try allocator.alloc(Battle, W);
    const here = try allocator.alloc(Battle, W);
    const specs = try allocator.alloc(pathmod.Path.TurnSpec, W);
    for (0..W) |w| {
        here[w] = universe.pastWorld(&h[0].battle, h[0].knowledge, worlds[w], cfg.root_knowledge);
        cur[w] = try allocator.create(pathmod.Path);
        cur[w].* = try pathmod.Path.turn(allocator, cfg.model, cfg.vocab, null, &here[w], &here[w], h[0].result, .{}, .{}, h[0].knowledge, 1);
    }
    for (1..h.len) |i| {
        const point = h[i];
        for (0..W) |w| {
            prev[w] = here[w];
            // the last point is the root: the world itself (its hidden timers drawn), the others rebuilt in it
            here[w] = if (i + 1 == h.len) worlds[w].* else universe.pastWorld(&point.battle, point.knowledge, worlds[w], cfg.root_knowledge);
            specs[w] = .{ .parent = cur[w], .before = &prev[w], .after = &here[w], .result = point.result, .c1 = h[i - 1].c1, .c2 = h[i - 1].c2, .side = 1 };
        }
        const next = try allocator.alloc(pathmod.Path, W);
        try pathmod.Path.turnBatch(allocator, cfg.model, cfg.vocab, specs, next);
        for (0..W) |w| cur[w] = &next[w];
    }
    for (0..W) |w| out[w] = cur[w].*;
}

/// P2's root path in the world `world`. With the real game's history (cfg.history), the chain replays the game from P2's
/// side, each past decision point as it would have been in this world (universe.pastWorld: the world's opponent team, the
/// public HP, status and PP of that time), so the opponent's actor reads a history consistent with the team it holds here;
/// the windowed model replays the last `root_window` points (its window), the recurrent one the whole game. Without a
/// history (or with a single point) the chain starts at the root.
pub fn opponentRoot(allocator: std.mem.Allocator, cfg: Config, world: *const Battle, truth_result: pkmn.Result) !pathmod.Path {
    const h = cfg.history;
    if (h.len < 2) return pathmod.Path.turn(allocator, cfg.model, cfg.vocab, null, world, world, truth_result, .{}, .{}, cfg.root_knowledge, 1);
    const first = if (cfg.recurrent or cfg.root_window == 0) 0 else h.len -| cfg.root_window;
    var parent: ?*const pathmod.Path = null;
    var prev: Battle = undefined;
    for (h[first..], first..) |point, i| {
        // the last point is the root: the world itself (its hidden timers drawn), the others rebuilt in this world
        const here = if (i + 1 == h.len) world.* else universe.pastWorld(&point.battle, point.knowledge, world, cfg.root_knowledge);
        const node = try allocator.create(pathmod.Path);
        if (parent) |p| {
            node.* = try pathmod.Path.turn(allocator, cfg.model, cfg.vocab, p, &prev, &here, point.result, h[i - 1].c1, h[i - 1].c2, .{}, 1);
        } else {
            node.* = try pathmod.Path.turn(allocator, cfg.model, cfg.vocab, null, &here, &here, point.result, .{}, .{}, point.knowledge, 1);
        }
        parent = node;
        prev = here;
    }
    return parent.?.*;
}

const Cell = struct { visits: f32, wins: f32 };

/// Visits and P1 reward summed over P2's root actions, for P1 action `a1` in one universe.
fn rowCell(tree: *const Tree, root: u32, a1: usize, n2: usize) Cell {
    const node = &tree.nodes[root];
    var c = Cell{ .visits = 0, .wins = 0 };
    for (0..n2) |b| {
        c.visits += @floatFromInt(node.visits[a1][b]);
        c.wins += node.wins[a1][b];
    }
    return c;
}

fn countLive(live: *const [MAXC]bool, n1: usize) usize {
    var k: usize = 0;
    for (0..n1) |a| {
        if (live[a]) k += 1;
    }
    return k;
}

/// Paired racing: the leader is the live action with the best mean value over the universes. An
/// action is eliminated when its paired deficit to the leader (one difference per universe) is
/// significant at `z` standard errors.
fn eliminate(root: Root, live: *[MAXC]bool, n1: usize, z: f32) void {
    const W = root.trees.len;
    std.debug.assert(W <= MAX_WORLDS);
    var val: [MAXC][MAX_WORLDS]f32 = undefined;
    var ok: [MAXC][MAX_WORLDS]bool = @splat(@splat(false));
    var mean: [MAXC]f32 = @splat(0);
    var leader: usize = n1;
    for (0..n1) |a| {
        if (!live[a]) continue;
        var sum: f32 = 0;
        var k: f32 = 0;
        for (0..W) |w| {
            const c = rowCell(&root.trees[w], root.roots[w], a, root.trees[w].nodes[root.roots[w]].n[1]);
            if (c.visits > 0) {
                val[a][w] = c.wins / c.visits;
                ok[a][w] = true;
                sum += val[a][w];
                k += 1;
            }
        }
        if (k == 0) continue; // an unsearched action cannot lead
        mean[a] = sum / k;
        if (leader == n1 or mean[a] > mean[leader]) leader = a;
    }
    if (leader == n1) return;
    for (0..n1) |a| {
        if (!live[a] or a == leader) continue;
        var d: [MAX_WORLDS]f32 = undefined;
        var k: usize = 0;
        for (0..W) |w| {
            if (ok[leader][w] and ok[a][w]) {
                d[k] = val[leader][w] - val[a][w];
                k += 1;
            }
        }
        if (k < 2) continue;
        var m: f32 = 0;
        for (d[0..k]) |x| m += x;
        m /= @floatFromInt(k);
        var v: f32 = 0;
        for (d[0..k]) |x| v += (x - m) * (x - m);
        const se = @sqrt(v / @as(f32, @floatFromInt(k - 1))) / @sqrt(@as(f32, @floatFromInt(k)));
        if (m - z * se > 0) live[a] = false;
    }
}

/// Each sampled team is an opponent information type. P1 uses one strategy across
/// worlds; P2 has a distinct response in each world. Never align unrelated move slots.
fn pooledNash(root: Root, live: *const [MAXC]bool, n1: usize) [MAXC]f32 {
    var payoff: [MAX_WORLDS][MAXC][MAXC]f32 = undefined;
    var counts: [MAX_WORLDS]usize = @splat(0);
    for (root.trees, root.roots, 0..) |*tree, r, w| {
        const node = &tree.nodes[r];
        counts[w] = node.n[1];
        for (0..n1) |a| {
            const row = rowCell(tree, r, a, counts[w]);
            const mean = if (row.visits > 0) row.wins / row.visits else 0.5;
            for (0..counts[w]) |b| {
                payoff[w][a][b] = if (node.visits[a][b] > 0)
                    node.wins[a][b] / @as(f32, @floatFromInt(node.visits[a][b])) else mean;
            }
        }
    }
    return solveNash(&payoff, live, n1, counts[0..root.trees.len]);
}

fn solveNash(payoff: *const [MAX_WORLDS][MAXC][MAXC]f32, live: *const [MAXC]bool, n1: usize, counts: []const usize) [MAXC]f32 {
    const iters = 2000;
    var regret1: [MAXC]f32 = @splat(0);
    var regret2: [MAX_WORLDS][MAXC]f32 = @splat(@splat(0));
    var sum_p: [MAXC]f32 = @splat(0);
    for (0..iters) |_| {
        var p: [MAXC]f32 = @splat(0);
        var pos: f32 = 0;
        for (0..n1) |a| if (live[a]) {
            p[a] = @max(regret1[a], 0);
            pos += p[a];
        };
        for (0..n1) |a| if (live[a]) {
            p[a] = if (pos > 0) p[a] / pos else 1 / @as(f32, @floatFromInt(countLive(live, n1)));
        };
        var util1: [MAXC]f32 = @splat(0);
        for (counts, 0..) |n2, w| {
            var q: [MAXC]f32 = @splat(0);
            pos = 0;
            for (0..n2) |b| {
                q[b] = @max(regret2[w][b], 0);
                pos += q[b];
            }
            for (0..n2) |b| q[b] = if (pos > 0) q[b] / pos else 1 / @as(f32, @floatFromInt(n2));
            var util2: [MAXC]f32 = @splat(0);
            var v2: f32 = 0;
            for (0..n2) |b| {
                for (0..n1) |a| if (live[a]) {
                    util1[a] += payoff[w][a][b] * q[b] / @as(f32, @floatFromInt(counts.len));
                    util2[b] -= p[a] * payoff[w][a][b];
                };
                v2 += q[b] * util2[b];
            }
            for (0..n2) |b| regret2[w][b] += util2[b] - v2;
        }
        var v1: f32 = 0;
        for (0..n1) |a| v1 += p[a] * util1[a];
        for (0..n1) |a| if (live[a]) {
            regret1[a] += util1[a] - v1;
            sum_p[a] += p[a] / @as(f32, iters);
        };
    }
    return sum_p;
}

fn sampleStrategy(strategy: *const [MAXC]f32, n: usize, rand: *std.Random) u8 {
    var draw = rand.float(f32);
    var last: u8 = 0;
    for (0..n) |a| {
        if (strategy[a] <= 0) continue;
        last = @intCast(a);
        draw -= strategy[a];
        if (draw <= 0) return last;
    }
    return last;
}

const Paired = struct {
    n: u32 = 0,
    mean: f32 = 0,
    m2: f32 = 0,
    fn add(self: *Paired, x: f32) void {
        self.n += 1;
        const delta = x - self.mean;
        self.mean += delta / @as(f32, @floatFromInt(self.n));
        self.m2 += delta * (x - self.mean);
    }
    fn lower(self: Paired, prior_shift: f32, cfg: Config) f32 {
        if (self.n < cfg.adaptive_min_sweeps) return -std.math.inf(f32);
        const variance = self.m2 / @as(f32, @floatFromInt(self.n - 1));
        const se = @sqrt(@max(variance, cfg.adaptive_variance_floor * cfg.adaptive_variance_floor) / @as(f32, @floatFromInt(self.n)));
        return self.mean + prior_shift - cfg.adaptive_confidence * se;
    }
};

/// Value of own root action `a` in one world against every legal opponent reply, weighted by the
/// world's actor probabilities for the replies. All replies share the sweep seed (mixed per reply).
fn replyWeighted(tree: *Tree, root: u32, cfg: Config, seed: u64, live: *const [MAXC]bool, a: u8) !f32 {
    if (cfg.chance_samples > 1 or cfg.reply_mass < 1 or cfg.exact_chance) return replyReduced(tree, root, cfg, seed, live, a);
    const node = &tree.nodes[root];
    var total: f32 = 0;
    var mass: f32 = 0;
    for (0..node.n[1]) |b| {
        const p = node.prior[1][b];
        if (p <= 0) continue;
        var prng = std.Random.DefaultPrng.init(seed ^ (@as(u64, b) +% 1) *% 0x9E3779B97F4A7C15);
        var r = prng.random();
        total += p * try simulate(tree, root, cfg, &r, live, a, @intCast(b));
        mass += p;
    }
    return if (mass > 0) total / mass else 0.5;
}

/// Mean of `chance_samples` draws of one (own action, reply) cell; draw m uses mixSeed(seed, m) for
/// every own action, so candidates see the same seed sequence.
fn cellValue(tree: *Tree, root: u32, cfg: Config, seed: u64, live: *const [MAXC]bool, a: u8, b: u8) !f32 {
    if (cfg.exact_chance) {
        var prng = std.Random.DefaultPrng.init(mixSeed(seed, 0xc4a));
        var r = prng.random();
        return exactCellValue(tree, root, cfg, seed, a, b, &r);
    }
    var sum: f32 = 0;
    const m_count = @max(1, cfg.chance_samples);
    for (0..m_count) |m| {
        var prng = std.Random.DefaultPrng.init(mixSeed(seed, m));
        var r = prng.random();
        sum += try simulate(tree, root, cfg, &r, live, a, b);
    }
    return sum / @as(f32, @floatFromInt(m_count));
}

/// Likely replies enumerated, the remaining mass covered by one paired draw, chance averaged per cell.
fn replyReduced(tree: *Tree, root: u32, cfg: Config, seed: u64, live: *const [MAXC]bool, a: u8) !f32 {
    const node = &tree.nodes[root];
    const n2: usize = node.n[1];
    const prior = node.prior[1];
    var order: [MAXC]u8 = undefined;
    for (0..n2) |b| order[b] = @intCast(b);
    std.mem.sort(u8, order[0..n2], prior, struct {
        fn less(p: [MAXC]f32, x: u8, y: u8) bool { return p[x] > p[y] or (p[x] == p[y] and x < y); }
    }.less);
    var mass: f32 = 0;
    for (0..n2) |b| mass += prior[b];
    if (mass <= 0) return 0.5;
    var head: usize = 0;
    var head_mass: f32 = 0;
    while (head < n2 and prior[order[head]] > 0 and (head < 2 or head_mass < cfg.reply_mass * mass)) : (head += 1)
        head_mass += prior[order[head]];
    var total: f32 = 0;
    for (order[0..head]) |b| total += prior[b] * try cellValue(tree, root, cfg, mixSeed(seed, @as(u64, b) + 1000), live, a, b);
    const tail = mass - head_mass;
    if (head < n2 and tail > 1e-6) {
        // the same tail reply for every own action (it depends only on the sweep seed)
        var prng = std.Random.DefaultPrng.init(mixSeed(seed, 0x7a11));
        var u = prng.random().float(f32) * tail;
        var pick = order[n2 - 1];
        for (order[head..n2]) |b| {
            if (u < prior[b]) { pick = b; break; }
            u -= prior[b];
        }
        total += tail * try cellValue(tree, root, cfg, mixSeed(seed, @as(u64, pick) + 1000), live, a, pick);
    }
    return total / mass;
}

/// Simulations per own action in a sweep (an upper bound when replies are pruned).
fn sweepCost(cfg: Config, root: Root, size: usize) usize {
    const draws: usize = if (cfg.exact_chance) 1 else @max(1, cfg.chance_samples);
    return if (cfg.wide) size * root.trees[0].nodes[root.roots[0]].n[1] * draws else size;
}

/// Paired sweeps through the existing chance trees. Actor is always retained as
/// a control, including after it would otherwise be eliminated. Budget is per world.
fn adaptiveSearch(root: Root, cfg: Config, rand: *std.Random, n1: usize, budget: u32, confirm: ?Root) !Outcome {
    const prior = root.trees[0].nodes[root.roots[0]].prior[0];
    var ranked: [MAXC]u8 = undefined;
    for (0..n1) |a| ranked[a] = @intCast(a);
    std.mem.sort(u8, ranked[0..n1], prior, struct {
        fn less(p: [MAXC]f32, a: u8, b: u8) bool { return p[a] > p[b] or (p[a] == p[b] and a < b); }
    }.less);
    const actor = ranked[0];
    var live: [MAXC]bool = @splat(false);
    for (ranked[0..@min(n1, cfg.adaptive_candidates)]) |a| live[a] = true;
    var pairs: [MAXC][MAXC]Paired = @splat(@splat(.{}));
    var sums: [MAXC]f32 = @splat(0);
    var counts: [MAXC]f32 = @splat(0);
    var spent: [MAX_WORLDS]u32 = @splat(0);
    var out = Outcome{ .choice = actor, .actor_choice = actor, .visits = @splat(0), .value = @splat(0) };
    var rollout_cfg = cfg;
    rollout_cfg.sample_opponent = true;
    while (true) {
        const w = out.sweeps % root.trees.len;
        const size = countLive(&live, n1);
        const cost = sweepCost(cfg, root, size);
        if (spent[w] + cost > budget) break;
        const seed = rand.int(u64);
        var values: [MAXC]f32 = @splat(0);
        for (0..n1) |offset| {
            const a = (offset + out.sweeps) % n1;
            if (!live[a]) continue;
            if (cfg.wide) {
                values[a] = try replyWeighted(&root.trees[w], root.roots[w], rollout_cfg, seed, &live, @intCast(a));
            } else {
                var prng = std.Random.DefaultPrng.init(seed);
                var paired_rng = prng.random();
                values[a] = try simulate(&root.trees[w], root.roots[w], rollout_cfg, &paired_rng, &live, @intCast(a), null);
            }
            sums[a] += values[a]; counts[a] += 1;
        }
        for (0..n1) |a| if (live[a]) {
            for (0..n1) |b| if (live[b] and a != b) { pairs[a][b].add(values[a] - values[b]); };
        };
        spent[w] += @intCast(cost); out.simulations += @intCast(cost); out.sweeps += 1;
        var leader = actor;
        var best: f32 = -std.math.inf(f32);
        for (0..n1) |a| if (live[a]) {
            const score = sums[a] / counts[a] + @log(@max(prior[a], 1e-8)) / cfg.root_value_scale;
            if (score > best) { best = score; leader = @intCast(a); }
        };
        // Complete a round over all sampled teams before elimination or stopping.
        if (out.sweeps % root.trees.len != 0) continue;
        var clear = true;
        for (0..n1) |a| if (live[a] and a != leader) {
            const shift = @log(@max(prior[leader], 1e-8) / @max(prior[a], 1e-8)) / cfg.root_value_scale;
            if (pairs[leader][a].lower(shift, cfg) > cfg.adaptive_margin) {
                if (a != actor) live[a] = false;
            } else clear = false;
        };
        if (clear) { out.early_stop = true; break; }
        // with exact chance a second round over the same worlds repeats the first one's values
        if (cfg.exact_chance) break;
    }
    var best: f32 = -std.math.inf(f32);
    var proposed = actor;
    for (0..n1) |a| {
        out.visits[a] = counts[a];
        out.value[a] = if (counts[a] > 0) sums[a] / counts[a] else 0;
        if (!live[a] or counts[a] == 0) continue;
        const score = out.value[a] + @log(@max(prior[a], 1e-8)) / cfg.root_value_scale;
        if (score > best) { best = score; proposed = @intCast(a); }
    }
    if (proposed != actor) {
        const shift = @log(@max(prior[proposed], 1e-8) / @max(prior[actor], 1e-8)) / cfg.root_value_scale;
        out.override_lower = pairs[proposed][actor].lower(shift, cfg);
        if (out.override_lower > cfg.adaptive_margin) {
            out.choice = proposed;
            if (confirm) |fresh| {
                // The challenger was selected on the main worlds; test it against the actor on new ones.
                out.confirm_ran = true;
                var diff = Paired{};
                var only: [MAXC]bool = @splat(false);
                only[actor] = true;
                only[proposed] = true;
                // at least adaptive_min_sweeps paired samples, cycling over the fresh worlds
                const rounds = @max(fresh.trees.len, cfg.adaptive_min_sweeps);
                for (0..rounds) |i| {
                    const tree = &fresh.trees[i % fresh.trees.len];
                    const r = fresh.roots[i % fresh.trees.len];
                    const seed = rand.int(u64);
                    const va = if (cfg.wide) try replyWeighted(tree, r, rollout_cfg, seed, &only, actor) else blk: {
                        var prng = std.Random.DefaultPrng.init(seed);
                        var g = prng.random();
                        break :blk try simulate(tree, r, rollout_cfg, &g, &only, actor, null);
                    };
                    const vp = if (cfg.wide) try replyWeighted(tree, r, rollout_cfg, seed, &only, proposed) else blk: {
                        var prng = std.Random.DefaultPrng.init(seed);
                        var g = prng.random();
                        break :blk try simulate(tree, r, rollout_cfg, &g, &only, proposed, null);
                    };
                    diff.add(vp - va);
                    out.simulations += if (cfg.wide) @intCast(2 * tree.nodes[r].n[1] * @max(1, cfg.chance_samples)) else 2;
                }
                out.confirm_lower = diff.lower(shift, cfg);
                if (out.confirm_lower <= cfg.adaptive_margin) out.choice = actor;
            }
        }
    }
    for (root.trees) |tree| {
        out.depth_total += tree.depth_total;
        out.max_depth = @max(out.max_depth, tree.max_depth);
        out.reused += tree.reused;
        out.leaves += tree.leaves;
    }
    return out;
}

/// Solution of the KL-anchored root game (see solveKL).
const KLSolution = struct {
    /// Our strategy over the n1 root actions.
    x: [MAXC]f32,
    /// Each action's value against the opponent's anchored responses (averaged over worlds).
    q: [MAXC]f32,
    /// The anchored game's value (including both KL terms).
    value: f32,
    iterations: u32,
    /// Largest gap between x and softmax(log pi + q / tau1) (0 at the exact solution).
    residual: f32,
};

/// Priors floored at 1e-6 and renormalized (uniform when none are positive), as natural logs.
fn logPrior(p: []const f32, out: []f64) void {
    var total: f64 = 0;
    for (p) |v| total += @max(@as(f64, v), 1e-6);
    for (p, 0..) |v, i| out[i] = @log(@max(@as(f64, v), 1e-6) / total);
}

/// The opponent's anchored response in one world to our strategy x: y = softmax(log sigma - M^T x / tau2).
/// Returns min_y [x^T M y + tau2 KL(y || sigma)] = -tau2 log sum_b sigma_b exp(-(M^T x)_b / tau2).
fn anchoredResponse(m: *const [MAXC][MAXC]f32, n1: usize, n2: usize, x: []const f64, log_sigma: []const f64, tau2: f64, y: []f64) f64 {
    var z: [MAXC]f64 = undefined;
    var top: f64 = -std.math.inf(f64);
    for (0..n2) |b| {
        var u: f64 = 0;
        for (0..n1) |a| u += x[a] * m[a][b];
        z[b] = log_sigma[b] - u / tau2;
        top = @max(top, z[b]);
    }
    var total: f64 = 0;
    for (0..n2) |b| {
        y[b] = @exp(z[b] - top);
        total += y[b];
    }
    for (0..n2) |b| y[b] /= total;
    return -tau2 * (top + @log(total));
}

/// The KL-anchored Bayesian root game. Each world is an opponent type (its sampled team): one
/// strategy x for us across worlds, one y_w per world for them. P1 maximizes
///   sum_w p_w [x^T M_w y_w + tau2 KL(y_w || sigma_w)] - tau1 KL(x || pi)
/// with each y_w minimizing its bracket. The opponent's anchored response has a closed form, so
/// this is a strictly concave problem in x alone, solved by mirror ascent with backtracking. At the
/// solution x = softmax(log pi + q / tau1): our actor's prior pulled toward the actions that do well
/// against an opponent who is itself pulled from its actor toward its best reply. Large
/// temperatures keep both sides at their actors; small ones approach the unanchored equilibrium.
fn solveKL(payoff: *const [MAX_WORLDS][MAXC][MAXC]f32, n1: usize, counts: []const usize, own_prior: []const f32, opp_prior: *const [MAX_WORLDS][MAXC]f32, tau1_in: f32, tau2_in: f32) KLSolution {
    const W = counts.len;
    const tau1: f64 = tau1_in;
    const tau2: f64 = tau2_in;
    const pw: f64 = 1 / @as(f64, @floatFromInt(W));
    var log_pi: [MAXC]f64 = undefined;
    logPrior(own_prior[0..n1], log_pi[0..n1]);
    var log_sigma: [MAX_WORLDS][MAXC]f64 = undefined;
    for (0..W) |w| logPrior(opp_prior[w][0..counts[w]], log_sigma[w][0..counts[w]]);

    const Eval = struct { f: f64, g: [MAXC]f64 };
    const S = struct {
        fn eval(p: *const [MAX_WORLDS][MAXC][MAXC]f32, n1_: usize, cs: []const usize, lx: []const f64, lpi: []const f64, ls: *const [MAX_WORLDS][MAXC]f64, t1: f64, t2: f64, w_: f64) Eval {
            var x: [MAXC]f64 = undefined;
            for (0..n1_) |a| x[a] = @exp(lx[a]);
            var out = Eval{ .f = 0, .g = @splat(0) };
            for (cs, 0..) |n2, w| {
                var y: [MAXC]f64 = undefined;
                out.f += w_ * anchoredResponse(&p[w], n1_, n2, x[0..n1_], ls[w][0..n2], t2, y[0..n2]);
                for (0..n1_) |a| {
                    var r: f64 = 0;
                    for (0..n2) |b| r += p[w][a][b] * y[b];
                    out.g[a] += w_ * r;
                }
            }
            var kl: f64 = 0;
            for (0..n1_) |a| kl += x[a] * (lx[a] - lpi[a]);
            out.f -= t1 * kl;
            return out;
        }
    };

    var lx: [MAXC]f64 = undefined;
    @memcpy(lx[0..n1], log_pi[0..n1]);
    var cur = S.eval(payoff, n1, counts, lx[0..n1], log_pi[0..n1], &log_sigma, tau1, tau2, pw);
    var beta: f64 = 0.5;
    var it: u32 = 0;
    while (it < 20000) : (it += 1) {
        // mirror ascent step: log x <- (1 - beta) log x + beta (log pi + q / tau1), normalized
        var nx: [MAXC]f64 = undefined;
        var top: f64 = -std.math.inf(f64);
        for (0..n1) |a| {
            nx[a] = (1 - beta) * lx[a] + beta * (log_pi[a] + cur.g[a] / tau1);
            top = @max(top, nx[a]);
        }
        var total: f64 = 0;
        for (0..n1) |a| total += @exp(nx[a] - top);
        const lse = top + @log(total);
        var change: f64 = 0;
        for (0..n1) |a| {
            nx[a] -= lse;
            change = @max(change, @abs(nx[a] - lx[a]));
        }
        if (change < 1e-7) break;
        const next = S.eval(payoff, n1, counts, nx[0..n1], log_pi[0..n1], &log_sigma, tau1, tau2, pw);
        if (next.f >= cur.f - 1e-12) {
            @memcpy(lx[0..n1], nx[0..n1]);
            cur = next;
            beta = @min(1, beta * 1.5);
        } else {
            beta *= 0.5;
            if (beta < 1e-12) break;
        }
    }
    var sol = KLSolution{ .x = @splat(0), .q = @splat(0), .value = @floatCast(cur.f), .iterations = it, .residual = 0 };
    var top: f64 = -std.math.inf(f64);
    for (0..n1) |a| top = @max(top, log_pi[a] + cur.g[a] / tau1);
    var total: f64 = 0;
    for (0..n1) |a| total += @exp(log_pi[a] + cur.g[a] / tau1 - top);
    for (0..n1) |a| {
        sol.x[a] = @floatCast(@exp(lx[a]));
        sol.q[a] = @floatCast(cur.g[a]);
        const target = @exp(log_pi[a] + cur.g[a] / tau1 - top) / total;
        sol.residual = @max(sol.residual, @as(f32, @floatCast(@abs(target - @exp(lx[a])))));
    }
    return sol;
}

/// Root rule .kl_nash: every (own action, reply) cell in every world is valued (exact chance when
/// configured), then the KL-anchored Bayesian game is solved (solveKL) and our action is drawn from
/// its strategy (or its argmax with kl_argmax). All own actions of a world share a reply's seed.
/// The cells of one world of the kl_nash root (its payoff rows); `cells` counts them.
fn klNashWorld(tree: *Tree, r: u32, cfg: Config, seed: u64, n1: usize, payoff: *[MAXC][MAXC]f32, cells: *u32) !void {
    const all: [MAXC]bool = @splat(true);
    for (0..tree.nodes[r].n[1]) |b| {
        const cell_seed = mixSeed(seed, @as(u64, b) + 1000);
        for (0..n1) |a| payoff[a][b] = try cellValue(tree, r, cfg, cell_seed, &all, @intCast(a), @intCast(b));
        cells.* += @intCast(n1);
    }
}

const WorldJob = struct {
    root: Root,
    cfg: Config,
    seeds: []const u64,
    n1: usize,
    payoff: *[MAX_WORLDS][MAXC][MAXC]f32,
    threads: usize,
    cells: [16]u32 = @splat(0),
    failed: [16]?anyerror = @splat(null),

    /// Thread `t` takes the worlds t, t + threads, ...
    fn run(job: *WorldJob, t: usize) void {
        var w = t;
        while (w < job.root.trees.len) : (w += job.threads) {
            klNashWorld(&job.root.trees[w], job.root.roots[w], job.cfg, job.seeds[w], job.n1, &job.payoff[w], &job.cells[t]) catch |e| {
                job.failed[t] = e;
                return;
            };
        }
    }
};

fn klNashSearch(root: Root, cfg: Config, rand: *std.Random, n1: usize) !Outcome {
    const W = root.trees.len;
    var payoff: [MAX_WORLDS][MAXC][MAXC]f32 = undefined;
    var counts: [MAX_WORLDS]usize = undefined;
    var opp_prior: [MAX_WORLDS][MAXC]f32 = undefined;
    var seeds: [MAX_WORLDS]u64 = undefined;
    var cells: u32 = 0;
    for (root.trees, root.roots, 0..) |*tree, r, w| {
        counts[w] = tree.nodes[r].n[1];
        opp_prior[w] = tree.nodes[r].prior[1];
        seeds[w] = rand.int(u64); // (in world order, as one thread draws them)
    }
    const threads: usize = @min(@max(cfg.threads, 1), @min(W, 16));
    if (threads <= 1 or remote.active != null or rnative.active == null) {
        for (root.trees, root.roots, 0..) |*tree, r, w| try klNashWorld(tree, r, cfg, seeds[w], n1, &payoff[w], &cells);
    } else {
        // every world allocates from its own arena while the threads run (the search's arena is not thread-safe)
        var arenas: [MAX_WORLDS]std.heap.ArenaAllocator = undefined;
        for (0..W) |w| {
            arenas[w] = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            root.trees[w].allocator = arenas[w].allocator();
        }
        defer for (0..W) |w| arenas[w].deinit();
        _ = noBatch();
        know.warmNames();
        rmodel.warm();
        var job = WorldJob{ .root = root, .cfg = cfg, .seeds = seeds[0..W], .n1 = n1, .payoff = &payoff, .threads = threads };
        var spawned: [16]std.Thread = undefined;
        for (1..threads) |t| spawned[t] = try std.Thread.spawn(.{}, WorldJob.run, .{ &job, t });
        job.run(0);
        for (1..threads) |t| spawned[t].join();
        for (0..threads) |t| {
            if (job.failed[t]) |e| return e;
            cells += job.cells[t];
        }
    }
    const own_prior = root.trees[0].nodes[root.roots[0]].prior[0];
    const sol = solveKL(&payoff, n1, counts[0..W], own_prior[0..n1], &opp_prior, cfg.kl_tau, cfg.kl_tau_opp);
    var actor: u8 = 0;
    var best: u8 = 0;
    for (0..n1) |a| {
        if (own_prior[a] > own_prior[actor]) actor = @intCast(a);
        if (sol.x[a] > sol.x[best]) best = @intCast(a);
    }
    var out = Outcome{ .choice = if (cfg.kl_argmax) best else sampleStrategy(&sol.x, n1, rand), .actor_choice = actor, .visits = sol.x, .value = sol.q, .strategy = sol.x, .game_value = sol.value, .residual = sol.residual, .simulations = cells, .sweeps = sol.iterations };
    for (root.trees) |tree| {
        out.depth_total += tree.depth_total;
        out.max_depth = @max(out.max_depth, tree.max_depth);
        out.reused += tree.reused;
        out.leaves += tree.leaves;
    }
    return out;
}

fn mixSeed(a: u64, b: u64) u64 {
    var x = a ^ (b +% 0x9E3779B97F4A7C15) *% 0xBF58476D1CE4E5B9;
    x ^= x >> 31;
    return x *% 0x94D049BB133111EB;
}

/// Variance diagnostic for the wide root. In every world, for `k_samples` chance seeds, every own
/// action is evaluated against every opponent reply. A (world, seed, reply) seed is shared by all own
/// actions (the pairing the wide search uses); a second, action-specific seed gives an unpaired draw
/// of the same cell. One line per cell: `D tag w k a b p paired unpaired depth_paired`.
pub fn diagnose(parent_allocator: std.mem.Allocator, truth: *const Battle, truth_result: pkmn.Result, cfg: Config, rand: *std.Random, k_samples: u32, tag: u32) !void {
    var arena = std.heap.ArenaAllocator.init(parent_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    if (cfg.universes.len == 0 or cfg.universes.len > MAX_WORLDS) return error.TooManyUniverses;
    const budget: u32 = @intCast(2 * @as(usize, k_samples) * MAXC * MAXC + 4);
    const root = try setupRoots(allocator, cfg, truth, truth_result, budget, rand);
    const r0 = &root.trees[0].nodes[root.roots[0]];
    const n1: usize = r0.n[0];
    var actor: usize = 0;
    for (0..n1) |a| if (r0.prior[0][a] > r0.prior[0][actor]) {
        actor = a;
    };
    std.debug.print("H {d} n1 {d} actor {d} prior", .{ tag, n1, actor });
    for (0..n1) |a| std.debug.print(" {d:.5}", .{r0.prior[0][a]});
    std.debug.print("\n", .{});
    var rcfg = cfg;
    rcfg.sample_opponent = true;
    var all: [MAXC]bool = @splat(true);
    _ = &all;
    for (root.trees, root.roots, 0..) |*tree, r, w| {
        const n2: usize = tree.nodes[r].n[1];
        for (0..k_samples) |k| {
            for (0..n2) |b| {
                const p = tree.nodes[r].prior[1][b];
                if (p <= 0) continue;
                const shared = mixSeed(mixSeed(tag, w), mixSeed(k, b));
                for (0..n1) |a| {
                    const before = tree.depth_total;
                    var prng = std.Random.DefaultPrng.init(shared);
                    var g = prng.random();
                    const v = try simulate(tree, r, rcfg, &g, &all, @intCast(a), @intCast(b));
                    const depth = tree.depth_total - before;
                    var prng2 = std.Random.DefaultPrng.init(mixSeed(shared, a + 101));
                    var g2 = prng2.random();
                    const u = try simulate(tree, r, rcfg, &g2, &all, @intCast(a), @intCast(b));
                    std.debug.print("D {d} {d} {d} {d} {d} {d:.5} {d:.5} {d:.5} {d}\n", .{ tag, w, k, a, b, p, v, u, depth });
                }
            }
        }
    }
}

pub fn search(
    parent_allocator: std.mem.Allocator,
    truth: *const Battle,
    truth_result: pkmn.Result,
    cfg: Config,
    rand: *std.Random,
) !Outcome {
    // everything a decision allocates (snapshots, token strings, path steps) is freed together
    var arena = std.heap.ArenaAllocator.init(parent_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    if (cfg.root_final == .adaptive and (!cfg.actor_priors or !modelOn(cfg) or cfg.adaptive_candidates == 0 or
        cfg.adaptive_min_sweeps < 2 or cfg.root_value_scale <= 0)) return error.InvalidAdaptiveConfig;
    if (cfg.root_final == .kl_nash and (!cfg.actor_priors or !modelOn(cfg) or cfg.kl_tau <= 0 or cfg.kl_tau_opp <= 0)) return error.InvalidKLConfig;
    if (cfg.recurrent and remote.active == null and rnative.active == null) return error.NoRecurrentModel;
    // the recurrent server's nodes made by this search are forgotten when it returns (the real game's path was made before)
    const server_mark: i64 = if (cfg.recurrent and remote.active != null) try remote.active.?.mark() else -1;
    defer if (cfg.recurrent and remote.active != null) remote.active.?.release(server_mark) catch {};
    if (cfg.exact_chance and (cfg.max_depth != 1 or cfg.chance_max_leaves == 0 or cfg.chance_max_leaves > chance_enum.MAX_LEAVES)) return error.InvalidExactChance;
    if (cfg.max_depth == 0 or cfg.max_depth > MAX_DEPTH) return error.InvalidDepth;
    if (cfg.leaf_only_value and cfg.max_depth != 1) return error.LeafOnlyNeedsDepthOne;
    if (cfg.root_racing and cfg.race_round == 0) return error.InvalidRaceRound;
    if (cfg.universes.len == 0 and cfg.worlds == 0) return error.NoUniverses;
    if (truth_result.type != .None) return error.TerminalBattle;
    if (cfg.model != null and cfg.vocab == null) return error.MissingVocabulary;
    // a forced move (one legal option for P1) needs no search
    {
        var forced_buf: [MAXC]pkmn.Choice = undefined;
        if (truth.choices(.P1, truth_result.p1, &forced_buf) == 1) return Outcome{ .choice = 0, .visits = @splat(0), .value = @splat(0) };
    }

    const world_count: usize = if (cfg.universes.len > 0) cfg.universes.len else cfg.worlds;
    if (world_count > MAX_WORLDS) return error.TooManyUniverses;
    // every universe gets the full iteration budget; with exact chance a tree must also hold every
    // root cell's leaves
    const exact_nodes: u32 = if (cfg.exact_chance) @as(u32, MAXC) * @as(u32, MAXC) * @as(u32, cfg.chance_max_leaves) + 8 else 0;
    const per_world = @max(1, cfg.iterations, exact_nodes);
    const root = try setupRoots(allocator, cfg, truth, truth_result, per_world, rand);
    const r0 = &root.trees[0].nodes[root.roots[0]];
    const n1: usize = r0.n[0];
    if (n1 == 0) return error.NoLegalActions;
    for (root.trees, root.roots) |*tree, r| {
        const node = &tree.nodes[r];
        if (node.n[0] != n1 or !std.mem.eql(pkmn.Choice, node.choices[0][0..n1], r0.choices[0][0..n1]))
            return error.InconsistentOwnChoices;
    }

    if (cfg.root_final == .kl_nash) return klNashSearch(root, cfg, rand, n1);

    if (cfg.root_final == .adaptive) {
        var confirm: ?Root = null;
        if (cfg.confirm_universes.len > 0) {
            if (cfg.confirm_universes.len > MAX_WORLDS) return error.TooManyUniverses;
            var fresh_cfg = cfg;
            fresh_cfg.universes = cfg.confirm_universes;
            const per_fresh = (@max(cfg.confirm_universes.len, cfg.adaptive_min_sweeps) + cfg.confirm_universes.len - 1) / cfg.confirm_universes.len;
            const draws: usize = if (cfg.exact_chance) cfg.chance_max_leaves else @max(1, cfg.chance_samples);
            confirm = try setupRoots(allocator, fresh_cfg, truth, truth_result, @intCast(2 * @as(usize, r0.n[1]) * draws * per_fresh + 8), rand);
        }
        return adaptiveSearch(root, cfg, rand, n1, per_world, confirm);
    }

    var live: [MAXC]bool = @splat(false);
    for (0..n1) |a| live[a] = true;
    // with racing, budget is spent in rounds so that eliminated actions stop receiving visits
    const round: u32 = if (cfg.root_racing and cfg.root_final != .nash) cfg.race_round else per_world;
    var spent: u32 = 0;
    var rounds: u32 = 0;
    while (spent < per_world) {
        const step = @min(round, per_world - spent);
        for (root.trees, root.roots) |*tree, r| {
            var it: u32 = 0;
            while (it < step) : (it += 1) _ = try simulate(tree, r, cfg, rand, &live, null, null);
        }
        spent += step;
        rounds += 1;
        if (cfg.root_racing and cfg.root_final != .nash) {
            if (rounds >= 2) eliminate(root, &live, n1, cfg.race_z);
            if (countLive(&live, n1) <= 1) break;
        }
    }

    var total_visits: [MAXC]f32 = @splat(0);
    var total_wins: [MAXC]f32 = @splat(0);
    for (root.trees, root.roots) |*tree, r| {
        const node = &tree.nodes[r];
        for (0..n1) |a| {
            for (0..node.n[1]) |b| {
                total_visits[a] += @floatFromInt(node.visits[a][b]);
                total_wins[a] += node.wins[a][b];
            }
        }
    }
    var value: [MAXC]f32 = @splat(0);
    for (0..MAXC) |a| {
        if (total_visits[a] > 0) value[a] = total_wins[a] / total_visits[a];
    }

    if (cfg.root_final == .nash) {
        // Only actions that were actually searched enter the matrix. An unvisited row has no
        // estimate, and filling it with a neutral value would let it win the equilibrium.
        var solve_live = live;
        for (0..n1) |a| {
            if (total_visits[a] == 0) solve_live[a] = false;
        }
        const strategy = pooledNash(root, &solve_live, n1);
        const best = sampleStrategy(&strategy, n1, rand);
        return Outcome{ .choice = best, .visits = total_visits, .value = value };
    }

    if (cfg.actor_priors and cfg.root_final == .prior_value) {
        // bot2's root rule: log(actor prior) + scale * (Q - V), over the actions that were visited
        var v_total: f32 = 0;
        var n_total: f32 = 0;
        for (0..MAXC) |a_i| {
            if (total_visits[a_i] > 0) {
                v_total += total_wins[a_i];
                n_total += total_visits[a_i];
            }
        }
        const v_mean = if (n_total > 0) v_total / n_total else 0.5;
        var best_score: f32 = -std.math.inf(f32);
        var best: u8 = 0;
        for (0..MAXC) |a_i| {
            if (total_visits[a_i] == 0) continue;
            const q = total_wins[a_i] / total_visits[a_i];
            var prior: f32 = 0;
            for (root.trees, root.roots) |*tree, r| prior += tree.nodes[r].prior[0][a_i];
            prior /= @floatFromInt(root.trees.len);
            const score = @log(@max(prior, 1e-6)) + cfg.root_value_scale * (q - v_mean);
            if (score > best_score) {
                best_score = score;
                best = @intCast(a_i);
            }
        }
        return Outcome{ .choice = best, .visits = total_visits, .value = value };
    }

    var out = Outcome{ .choice = 0, .visits = total_visits, .value = value };
    var best_visits: f32 = -1;
    var a: u8 = 0;
    while (a < MAXC) : (a += 1) {
        if (total_visits[a] > best_visits) {
            best_visits = total_visits[a];
            out.choice = a;
        }
    }
    return out;
}

test "Nash retains a mixed strategy and conditions opponent responses on their team" {
    var payoff: [MAX_WORLDS][MAXC][MAXC]f32 = @splat(@splat(@splat(0)));
    var live: [MAXC]bool = @splat(false);
    live[0] = true;
    live[1] = true;
    payoff[0][0][0] = 1;
    payoff[0][1][1] = 1;
    const mixed = solveNash(&payoff, &live, 2, &.{2});
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mixed[0], 0.02);
    var prng = std.Random.DefaultPrng.init(42);
    var rand = prng.random();
    var first: usize = 0;
    for (0..1000) |_| { if (sampleStrategy(&mixed, 2, &rand) == 0) first += 1; }
    try std.testing.expect(first > 400 and first < 600);
    // Slot pooling would value row 0 at 0.5 and choose it. A type-aware
    // opponent chooses its zero column in each world, so row 1 (0.4) wins.
    payoff[0][0][0] = 1; payoff[0][0][1] = 0;
    payoff[1][0][0] = 0; payoff[1][0][1] = 1;
    for (0..2) |w| { payoff[w][1][0] = 0.4; payoff[w][1][1] = 0.4; }
    const typed = solveNash(&payoff, &live, 2, &.{2, 2});
    try std.testing.expect(typed[1] > 0.98);
}

test "KL-anchored game: closed form, symmetry, anchoring and opponent types" {
    var payoff: [MAX_WORLDS][MAXC][MAXC]f32 = @splat(@splat(@splat(0)));
    var opp: [MAX_WORLDS][MAXC]f32 = @splat(@splat(0));
    // one reply: x = softmax(log pi + v / tau1)
    payoff[0][0][0] = 0.6;
    payoff[0][1][0] = 0.5;
    opp[0][0] = 1;
    var sol = solveKL(&payoff, 2, &.{1}, &.{ 0.5, 0.5 }, &opp, 0.1, 0.1);
    const expected: f32 = @floatCast(1 / (1 + @exp(@as(f64, -1))));
    try std.testing.expectApproxEqAbs(expected, sol.x[0], 1e-4);
    // matching pennies with uniform anchors stays at one half
    payoff[0][0][0] = 1; payoff[0][0][1] = 0;
    payoff[0][1][0] = 0; payoff[0][1][1] = 1;
    opp[0][0] = 0.5; opp[0][1] = 0.5;
    sol = solveKL(&payoff, 2, &.{2}, &.{ 0.5, 0.5 }, &opp, 0.05, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sol.x[0], 1e-3);
    // hot temperatures keep our actor's prior
    sol = solveKL(&payoff, 2, &.{2}, &.{ 0.8, 0.2 }, &opp, 50, 50);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), sol.x[0], 1e-2);
    // an opponent anchored to a predictable reply is exploited: they favor column 0 strongly
    opp[0][0] = 0.95; opp[0][1] = 0.05;
    sol = solveKL(&payoff, 2, &.{2}, &.{ 0.5, 0.5 }, &opp, 0.05, 1);
    try std.testing.expect(sol.x[0] > 0.9);
    // the same anchor with a cold opponent temperature is not exploitable: back toward one half
    const cold = solveKL(&payoff, 2, &.{2}, &.{ 0.5, 0.5 }, &opp, 0.05, 0.01);
    try std.testing.expect(cold.x[0] < sol.x[0] and cold.x[0] < 0.75);
    // type-aware: row 0 is good in each world only against the reply that type would not pick
    payoff[0][0][0] = 1; payoff[0][0][1] = 0;
    payoff[1][0][0] = 0; payoff[1][0][1] = 1;
    for (0..2) |w| { payoff[w][1][0] = 0.4; payoff[w][1][1] = 0.4; }
    opp[0] = @splat(0); opp[1] = @splat(0);
    opp[0][0] = 0.5; opp[0][1] = 0.5; opp[1][0] = 0.5; opp[1][1] = 0.5;
    sol = solveKL(&payoff, 2, &.{ 2, 2 }, &.{ 0.5, 0.5 }, &opp, 0.01, 0.01);
    try std.testing.expect(sol.x[1] > 0.95);
    try std.testing.expect(sol.residual < 1e-4);
    // a random 9 x 9 x 8 game with cold temperatures still converges
    var prng = std.Random.DefaultPrng.init(3);
    const rr = prng.random();
    var counts: [8]usize = undefined;
    for (0..8) |w| {
        counts[w] = 9;
        for (0..9) |a| for (0..9) |b| { payoff[w][a][b] = 0.3 + 0.4 * rr.float(f32); };
        var t: f32 = 0;
        for (0..9) |b| { opp[w][b] = rr.float(f32) * rr.float(f32); t += opp[w][b]; }
        for (0..9) |b| opp[w][b] /= t;
    }
    var pi: [9]f32 = undefined;
    var t: f32 = 0;
    for (&pi) |*v| { v.* = rr.float(f32) * rr.float(f32) * rr.float(f32); t += v.*; }
    for (&pi) |*v| v.* /= t;
    sol = solveKL(&payoff, 9, &counts, &pi, &opp, 0.02, 0.02);
    std.debug.print("random game: iterations {d}, residual {e}\n", .{ sol.iterations, sol.residual });
    try std.testing.expect(sol.residual < 1e-3);
}

test "search ignores real RNG and supports different opponent choice counts" {
    const a = [_]pkmn.gen1.helpers.Pokemon{.{ .species = .Tauros, .moves = &.{.BodySlam, .Blizzard} }};
    const b = [_]pkmn.gen1.helpers.Pokemon{.{ .species = .Chansey, .moves = &.{.SeismicToss, .ThunderWave} }};
    var battle = pkmn.gen1.helpers.Battle.init(3, &a, &b);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    const result = try battle.update(.{}, .{}, &options);
    var changed = battle;
    changed.rng = pkmn.gen1.PRNG.init(999);
    var p1 = std.Random.DefaultPrng.init(123);
    var p2 = std.Random.DefaultPrng.init(123);
    var r1 = p1.random(); var r2 = p2.random();
    const worlds1 = [_]Battle{battle}; const worlds2 = [_]Battle{changed};
    const cfg = Config{ .iterations = 100, .universes = &worlds1, .evaluator = hpValue, .root_final = .nash };
    var cfg2 = cfg; cfg2.universes = &worlds2;
    const out1 = try search(std.testing.allocator, &battle, result, cfg, &r1);
    const out2 = try search(std.testing.allocator, &changed, result, cfg2, &r2);
    try std.testing.expectEqual(out1.choice, out2.choice);
    try std.testing.expectEqualSlices(f32, &out1.visits, &out2.visits);
    try std.testing.expectEqualSlices(f32, &out1.value, &out2.value);
    changed.sides[1].pokemon[0].moves[1] = .{};
    changed.sides[1].active.moves[1] = .{};
    const worlds3 = [_]Battle{battle, changed};
    cfg2.universes = &worlds3;
    _ = try search(std.testing.allocator, &battle, result, cfg2, &r2);
    cfg2.worlds = 0; cfg2.universes = &.{};
    try std.testing.expectError(error.NoUniverses, search(std.testing.allocator, &battle, result, cfg2, &r2));
    cfg2.worlds = 1; cfg2.max_depth = 65;
    try std.testing.expectError(error.InvalidDepth, search(std.testing.allocator, &battle, result, cfg2, &r2));
    cfg2.max_depth = 1; cfg2.root_racing = true; cfg2.race_round = 0;
    try std.testing.expectError(error.InvalidRaceRound, search(std.testing.allocator, &battle, result, cfg2, &r2));
}

test "recharge and blocked opponent choices do not reveal private moves" {
    const public = @import("public_state.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const mons = [_]pkmn.gen1.helpers.Pokemon{.{ .species = .Tauros, .moves = &.{.BodySlam, .HyperBeam} }};
    var battle = pkmn.gen1.helpers.Battle.init(3, &mons, &mons);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try battle.update(.{}, .{}, &options);
    var history = public.History.init(arena.allocator());
    try history.record(&battle);
    const recharge = pkmn.Choice{ .type = .Move, .data = 0 };
    _ = know.Knowledge.afterTransition(&battle, &battle, recharge, recharge);
    try history.noteTransition(&battle, &battle, recharge, recharge);
    const move = pkmn.Choice{ .type = .Move, .data = 2 };
    try history.noteTransition(&battle, &battle, move, move);
    try std.testing.expect(history.frames.items[0].opponent_action == null);
    try std.testing.expectEqual(@as(usize, 0), history.revealed[1][0].items.len);
}

test "empty known slots are completed and opponent perspective keeps oriented knowledge" {
    const comp = @import("completion.zig");
    const belief = @import("belief.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var table = comp.Table{ .allocator = alloc, .by_species = std.StringHashMap([]comp.Entry).init(alloc) };
    var entries = [_]comp.Entry{ .{.move = .BodySlam, .rate = 1}, .{.move = .Blizzard, .rate = 1}, .{.move = .Earthquake, .rate = 1}, .{.move = .HyperBeam, .rate = 1} };
    try table.by_species.put("tauros", &entries);
    var rng = belief.Rng.init(1);
    const filled = try table.complete("tauros", &.{.None, .None, .None, .None}, &rng);
    try std.testing.expectEqual(@as(usize, 4), filled.len);
    try std.testing.expect(std.mem.indexOfScalar(Move, filled, .None) == null);
    const mons = [_]pkmn.gen1.helpers.Pokemon{.{.species = .Tauros, .moves = &.{.BodySlam, .HyperBeam}}};
    var battle = pkmn.gen1.helpers.Battle.init(1, &mons, &mons);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try battle.update(.{}, .{}, &options);
    var k = know.initial(&battle);
    k.moves[0][0][0] = .BodySlam;
    k.moves[1][0][0] = .HyperBeam;
    const swapped = know.swapSides(&battle);
    const expected = try know.observerSnapshot(alloc, &swapped, know.swapKnowledge(k), false);
    const actual = try know.perspectiveSnapshot(alloc, &battle, know.swapKnowledge(k), 1);
    const policy = @import("policy.zig");
    var left: std.ArrayList(u8) = .empty; var right: std.ArrayList(u8) = .empty;
    try policy.writeJson(alloc, &left, expected); try policy.writeJson(alloc, &right, actual);
    try std.testing.expectEqualStrings(left.items, right.items);
    try std.testing.expectEqual(@as(?usize, 0), know.actorIndex(&battle, 0, .{.type = .Move, .data = 0}));
}

test "public move consumption accumulates by move identity" {
    const mons = [_]pkmn.gen1.helpers.Pokemon{.{.species = .Chansey, .moves = &.{.Splash, .ThunderWave}}};
    var battle = pkmn.gen1.helpers.Battle.init(7, &mons, &mons);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try battle.update(.{}, .{}, &options);
    var knowledge = know.initial(&battle);
    const move = pkmn.Choice{.type = .Move, .data = 1};
    for (0..3) |_| {
        const before = battle;
        _ = try battle.update(move, move, &options);
        knowledge = know.Knowledge.merge(knowledge, know.Knowledge.afterTransition(&before, &battle, move, move));
    }
    try std.testing.expectEqual(Move.Splash, knowledge.moves[1][0][0]);
    try std.testing.expectEqual(@as(u8, 3), knowledge.used[1][0][0]);
}

test "private sleep confusion disable and trapping counters are not copied into search" {
    const mons = [_]pkmn.gen1.helpers.Pokemon{.{.species = .Chansey, .moves = &.{.Splash}}};
    var a = pkmn.gen1.helpers.Battle.init(7, &mons, &mons);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try a.update(.{}, .{}, &options);
    a.sides[1].pokemon[0].status = pkmn.gen1.Status.slp(1);
    a.sides[1].active.volatiles.Confusion = true;
    a.sides[1].active.volatiles.confusion = 1;
    a.sides[1].active.volatiles.disable_move = 1;
    a.sides[1].active.volatiles.disable_duration = 1;
    a.sides[1].active.volatiles.Binding = true;
    a.sides[1].active.volatiles.attacks = 1;
    var b = a;
    b.sides[1].pokemon[0].status = pkmn.gen1.Status.slp(7);
    b.sides[1].active.volatiles.confusion = 4;
    b.sides[1].active.volatiles.disable_duration = 8;
    b.sides[1].active.volatiles.attacks = 4;
    var p1 = std.Random.DefaultPrng.init(42); var p2 = std.Random.DefaultPrng.init(42);
    var r1 = p1.random(); var r2 = p2.random();
    randomizeHiddenTimers(&a, &r1); randomizeHiddenTimers(&b, &r2);
    try std.testing.expect(std.meta.eql(a, b));
}

test "paired confidence requires evidence and respects prior penalty" {
    const cfg = Config{};
    var pair = Paired{};
    for (0..5) |_| pair.add(0.1);
    try std.testing.expect(std.math.isNegativeInf(pair.lower(0, cfg)));
    pair.add(0.1);
    try std.testing.expect(pair.lower(0, cfg) > cfg.adaptive_margin);
    try std.testing.expect(pair.lower(-0.2, cfg) < cfg.adaptive_margin);
    var tie = Paired{};
    for (0..20) |_| tie.add(0);
    try std.testing.expect(tie.lower(0, cfg) < 0);
}

test "adaptive sweeps preserve actor on ties and respect world budgets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const mons = [_]pkmn.gen1.helpers.Pokemon{.{ .species = .Chansey, .moves = &.{.Splash, .Growl} }};
    var battle = pkmn.gen1.helpers.Battle.init(1, &mons, &mons);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    const result = try battle.update(.{}, .{}, &options);
    var prng = std.Random.DefaultPrng.init(11); var rand = prng.random();
    const worlds = [_]Battle{battle, battle};
    const cfg = Config{ .universes = &worlds, .max_depth = 3, .evaluator = hpValue, .switch_penalty = 0 };
    const roots = try setupRoots(alloc, cfg, &battle, result, 30, &rand);
    for (roots.trees, roots.roots) |*tree, r| {
        tree.nodes[r].has_prior = true;
        tree.nodes[r].prior[0][0] = 0.9; tree.nodes[r].prior[0][1] = 0.1;
        tree.nodes[r].prior[1][0] = 0.5; tree.nodes[r].prior[1][1] = 0.5;
    }
    const out = try adaptiveSearch(roots, cfg, &rand, 2, 30, null);
    try std.testing.expectEqual(@as(u8, 0), out.choice);
    try std.testing.expect(out.early_stop);
    try std.testing.expect(out.sweeps >= cfg.adaptive_min_sweeps);
    try std.testing.expect(out.simulations <= 60);
    try std.testing.expectEqual(out.visits[0], out.visits[1]);
    const tiny = try adaptiveSearch(roots, cfg, &rand, 2, 1, null);
    try std.testing.expectEqual(@as(u32, 0), tiny.simulations);
    try std.testing.expectEqual(@as(u8, 0), tiny.choice);

    var decisive_cfg = cfg;
    decisive_cfg.max_depth = 1;
    decisive_cfg.evaluator = struct {
        fn evaluate(b: *const Battle, _: pkmn.Result, _: *std.Random) f32 {
            return if (b.sides[0].last_selected_move == .Growl) 0.9 else 0.1;
        }
    }.evaluate;
    const decisive_roots = try setupRoots(alloc, decisive_cfg, &battle, result, 30, &rand);
    for (decisive_roots.trees, decisive_roots.roots) |*tree, r| {
        tree.nodes[r].has_prior = true;
        tree.nodes[r].prior[0][0] = 0.9; tree.nodes[r].prior[0][1] = 0.1;
        tree.nodes[r].prior[1][0] = 0.5; tree.nodes[r].prior[1][1] = 0.5;
    }
    const decisive = try adaptiveSearch(decisive_roots, decisive_cfg, &rand, 2, 30, null);
    try std.testing.expectEqual(@as(u8, 1), decisive.choice);
    try std.testing.expect(decisive.override_lower > decisive_cfg.adaptive_margin);
}

test "first-play estimates follow the known-good state value and reduction" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.625), firstPlay(0.7, 0, 0.25, 0.15), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.225), firstPlay(0.7, 1, 0.25, 0.15), 1e-6);
    try std.testing.expectEqual(@as(f32, 0), firstPlay(0.01, 0, 1, 0.15));
}
