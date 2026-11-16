//! Per-node observer state for the search tree and for the real game. Everything the value model
//! conditions on that depends on history lives in a `Path`: the knowledge, the roster and effect
//! memories, the step vector, and the model's cached keys/values for this node's window position.
//!
//! The window of a node is the chain of steps from its root's window to the node. A node whose
//! window is not yet full extends its parent incrementally: one position run against the cached
//! rows of the earlier positions (NNUE-style). A node whose window is full shifts every position,
//! so the cache is not reusable there and the whole window is recomputed.
//!
//! A search root is a `rootView`: the last `window` real steps, recomputed once as the root's own
//! window (the search then fills in history along each trajectory incrementally).

const std = @import("std");
const pkmn = @import("pkmn");
const vm = @import("value_model.zig");
const vs = @import("value_state.zig");
const know = @import("knowledge.zig");

pub const WINDOW = vm.MAX_WINDOW;

/// Scratch memory for the forward passes' temporaries. Each pass returns plain values (a logit and a
/// representation) and writes its keys and values into caller-owned rows, so nothing it allocates
/// outlives the call: the buffer is reset before every pass instead of growing a tree arena.
var scratch_buffer: [1 << 23]u8 = undefined;
var scratch_fba = std.heap.FixedBufferAllocator.init(&scratch_buffer);

fn scratch() std.mem.Allocator {
    scratch_fba.reset();
    return scratch_fba.allocator();
}

pub const Path = struct {
    knowledge: know.Knowledge,
    memory: vs.PlayerMemory,
    effects: vs.EffectMemory,
    step: []f32,
    parent: ?*const Path,
    /// Number of steps in this node's window (its own step included).
    window_len: usize,
    block_window: bool = false,
    /// Win logit of this node (computed when the node is built).
    logit: f32,
    /// The node's representation and the actor's logits over the nine action slots (same pass).
    rep: [vm.DIM]f32 = @splat(0),
    actor: [vm.ACTIONS]f32 = @splat(0),
    /// Keys and values of this node's own window position (valid when window_len < WINDOW).
    kv: vm.Kv,
    /// Root only: the steps of its window, oldest first.
    root_steps: []const []const f32 = &.{},
    /// Root only: keys and values of every position of the root's window.
    root_rows: []vm.Kv = &.{},

    /// Node for an observer snapshot reached from `parent` (null for a root).
    pub fn extend(
        allocator: std.mem.Allocator,
        model: *const vm.Model,
        vocab: *const std.StringHashMap(u32),
        parent: ?*const Path,
        knowledge: know.Knowledge,
        snapshot: vs.V,
    ) !Path {
        return build(allocator, model, vocab, parent, knowledge, snapshot);
    }

    /// The node reached from `parent` after the turn in which `before` became `after`.
    pub fn child(
        allocator: std.mem.Allocator,
        model: *const vm.Model,
        vocab: *const std.StringHashMap(u32),
        parent: *const Path,
        before: *const know.Battle,
        after: *const know.Battle,
        c1: pkmn.Choice,
        c2: pkmn.Choice,
    ) !Path {
        const step = know.Knowledge.afterTransition(before, after, c1, c2);
        const merged = know.Knowledge.merge(parent.knowledge, step);
        const snapshot = try know.observerSnapshot(allocator, after, merged, false);
        return build(allocator, model, vocab, parent, merged, snapshot);
    }

    /// As `child`, for the chain of `side` (1 = P2): the knowledge and snapshot are in P2's
    /// orientation, so the node is P2's observer state for the same turn.
    pub fn childSide(
        allocator: std.mem.Allocator,
        model: *const vm.Model,
        vocab: *const std.StringHashMap(u32),
        parent: *const Path,
        before: *const know.Battle,
        after: *const know.Battle,
        c1: pkmn.Choice,
        c2: pkmn.Choice,
        side: usize,
    ) !Path {
        if (side == 0) return child(allocator, model, vocab, parent, before, after, c1, c2);
        const step = know.swapKnowledge(know.Knowledge.afterTransition(before, after, c1, c2));
        const merged = know.Knowledge.merge(parent.knowledge, step);
        const snapshot = try know.perspectiveSnapshot(allocator, after, merged, 1);
        return build(allocator, model, vocab, parent, merged, snapshot);
    }

    /// Win probability at this node.
    pub fn value(self: *const Path) f64 {
        return 1.0 / (1.0 + @exp(-@as(f64, self.logit)));
    }

    /// A root whose window is the last `window` steps of `node` (all of them when fewer). The
    /// rows are recomputed for those steps, so the node's own memories and knowledge are kept
    /// while its history is cut to the window.
    pub fn rootView(allocator: std.mem.Allocator, model: *const vm.Model, node: *const Path, window: usize) !Path {
        // The existing cache already represents this exact window.
        if (window >= node.window_len) return node.*;
        const steps = try lastSteps(allocator, node, @min(window, WINDOW));
        const rows = try allocator.alloc(vm.Kv, steps.len);
        const fwd = try model.windowForwardState(scratch(), steps, rows);
        var view = node.*;
        view.parent = null;
        view.window_len = steps.len;
        view.logit = fwd.logit;
        view.rep = fwd.rep;
        view.actor = model.actorLogits(&fwd.rep);
        view.kv = rows[steps.len - 1];
        view.root_steps = steps;
        view.root_rows = rows;
        return view;
    }

    fn build(
        allocator: std.mem.Allocator,
        model: *const vm.Model,
        vocab: *const std.StringHashMap(u32),
        parent: ?*const Path,
        knowledge: know.Knowledge,
        snapshot: vs.V,
    ) !Path {
        // the node gets its own memories: it must never change the parent's
        var memory = if (parent) |p| try p.memory.cloneInto(allocator) else vs.PlayerMemory{ .allocator = allocator };
        var effects = if (parent) |p| p.effects else vs.EffectMemory{};
        const state = try vs.actorState(allocator, snapshot, &effects);
        const groups = try memory.encode(state);
        var slices: [vm.GROUPS][]const u32 = undefined;
        for (groups, 0..) |group, g| {
            const ids = try allocator.alloc(u32, group.len);
            for (group, ids) |token, *dst| dst.* = vocab.get(token) orelse 1;
            slices[g] = ids;
        }
        const step = try model.encodeStep(allocator, &slices);

        var node = Path{
            .knowledge = knowledge,
            .memory = memory,
            .effects = effects,
            .step = step,
            .parent = parent,
            .window_len = 1,
            .block_window = if (parent) |p| p.block_window else false,
            .logit = 0,
            .kv = undefined,
        };
        if (parent == null) {
            const steps = try allocator.alloc([]const f32, 1);
            steps[0] = step;
            const rows = try allocator.alloc(vm.Kv, 1);
            const fwd = try model.windowForwardState(scratch(), steps, rows);
            node.logit = fwd.logit;
            node.rep = fwd.rep;
            node.actor = model.actorLogits(&fwd.rep);
            node.root_steps = steps;
            node.root_rows = rows;
            node.kv = rows[0];
            node.window_len = 1;
        } else {
            const limit: usize = if (node.block_window) 31 else WINDOW;
            if (parent.?.window_len < limit) {
                const prefix = try prefixOf(allocator, parent.?);
                const fwd = try model.tokenForwardState(scratch(), step, prefix.len, prefix, &node.kv);
                node.logit = fwd.logit;
                node.rep = fwd.rep;
                node.actor = model.actorLogits(&fwd.rep);
                node.window_len = prefix.len + 1;
            } else {
                const keep: usize = if (node.block_window) 16 else WINDOW;
                const steps = try appendLast(allocator, parent.?, step, keep);
                const rows = try allocator.alloc(vm.Kv, steps.len);
                const fwd = try model.windowForwardState(scratch(), steps, rows);
                node.parent = null;
                node.root_steps = steps;
                node.root_rows = rows;
                node.kv = rows[steps.len - 1];
                node.logit = fwd.logit;
                node.rep = fwd.rep;
                node.actor = model.actorLogits(&fwd.rep);
                node.window_len = steps.len;
            }
        }
        return node;
    }

    /// Cached rows of the positions before the child's, oldest first: the root's rows, then each
    /// intermediate node's own row.
    fn prefixOf(allocator: std.mem.Allocator, parent: *const Path) ![]*const vm.Kv {
        var chain: std.ArrayList(*const vm.Kv) = .empty;
        var node: *const Path = parent;
        while (node.parent) |p| {
            try chain.append(allocator, &node.kv);
            node = p;
        }
        const root_len = node.root_rows.len;
        const out = try allocator.alloc(*const vm.Kv, root_len + chain.items.len);
        for (0..root_len) |i| out[i] = &node.root_rows[i];
        for (0..chain.items.len) |i| out[root_len + i] = chain.items[chain.items.len - 1 - i];
        return out;
    }

    /// The last `limit` steps of the window ending at `from`.
    fn lastSteps(allocator: std.mem.Allocator, from: *const Path, limit: usize) ![]const []const f32 {
        var reversed: std.ArrayList([]const f32) = .empty;
        var cur: ?*const Path = from;
        while (cur) |node| : (cur = node.parent) {
            if (reversed.items.len == limit) break;
            if (node.parent == null) {
                var i = node.root_steps.len;
                while (i > 0 and reversed.items.len < limit) {
                    i -= 1;
                    try reversed.append(allocator, node.root_steps[i]);
                }
                break;
            }
            try reversed.append(allocator, node.step);
        }
        std.mem.reverse([]const f32, reversed.items);
        return reversed.items;
    }

    /// The last `limit` steps of the window ending with `step` after `from`.
    fn appendLast(allocator: std.mem.Allocator, from: *const Path, step: []f32, limit: usize) ![]const []const f32 {
        const previous = try lastSteps(allocator, from, limit - 1);
        const out = try allocator.alloc([]const f32, previous.len + 1);
        @memcpy(out[0..previous.len], previous);
        out[previous.len] = step;
        return out;
    }
};
