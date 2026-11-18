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
const remote = @import("remote.zig");
const rnative = @import("rnative.zig");
const rmodel = @import("rmodel.zig");
const snapmod = @import("snapshot.zig");

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
    /// The recurrent search server's node for this path (remote.zig), -1 with the windowed model.
    remote_id: i64 = -1,
    /// The native recurrent model's node for this path (rnative.zig).
    native_node: ?*rnative.Node = null,

    /// The node of `side`'s chain (0 = P1, the observer; 1 = P2) at the decision point `after` (`result` is the update's
    /// result there), reached from `parent` by the turn (`c1`, `c2`) played from `before`. A null `parent` starts a chain
    /// at `after` with `knowledge` (in P1's orientation; `before`, `c1` and `c2` are then unused). Runs the windowed model
    /// (`model`, `vocab`) or, when remote.active is set, the recurrent model.
    pub fn turn(
        allocator: std.mem.Allocator,
        model: ?*const vm.Model,
        vocab: ?*const std.StringHashMap(u32),
        parent: ?*const Path,
        before: *const know.Battle,
        after: *const know.Battle,
        result: pkmn.Result,
        c1: pkmn.Choice,
        c2: pkmn.Choice,
        knowledge: know.Knowledge,
        side: usize,
    ) !Path {
        if (rnative.active) |nat| return nativeTurn(allocator, nat, parent, before, after, result, c1, c2, knowledge, side);
        if (remote.active) |server| return remoteTurn(allocator, server, parent, before, after, result, c1, c2, knowledge, side);
        if (parent) |p| return childSide(allocator, model.?, vocab.?, p, before, after, c1, c2, side);
        const k = if (side == 0) knowledge else know.swapKnowledge(knowledge);
        const snapshot = if (side == 0) try know.observerSnapshot(allocator, after, k, false) else try know.perspectiveSnapshot(allocator, after, k, 1);
        return extend(allocator, model.?, vocab.?, null, k, snapshot);
    }

    /// One `turn` on the recurrent search server, before the server's reply: the node's knowledge, its snapshot (the duel's
    /// snapshot for `side`, what the policy server reads in play) and what `side` chose at the parent (the server records it as
    /// what it sent there).
    const Pending = struct {
        parent: ?*const Path,
        after: *const know.Battle,
        side: usize,
        knowledge: know.Knowledge,
        names: []const []const u8,
        choices: [pkmn.CHOICES_SIZE]pkmn.Choice,
        sent: []const u8,
        json: []u8,
    };

    fn pending(
        allocator: std.mem.Allocator,
        parent: ?*const Path,
        before: *const know.Battle,
        after: *const know.Battle,
        result: pkmn.Result,
        c1: pkmn.Choice,
        c2: pkmn.Choice,
        knowledge: know.Knowledge,
        side: usize,
    ) !Pending {
        // the node's knowledge in the side's orientation (as childSide keeps it)
        var k = if (side == 0) knowledge else know.swapKnowledge(knowledge);
        var last = [2]snapmod.LastAction{ .{}, .{} };
        var sent: []const u8 = "";
        if (parent) |p| {
            const step = know.Knowledge.afterTransition(before, after, c1, c2);
            k = know.Knowledge.merge(p.knowledge, if (side == 0) step else know.swapKnowledge(step));
            last = .{ try snapmod.lastAction(allocator, before, after, 0, c1), try snapmod.lastAction(allocator, before, after, 1, c2) };
            sent = try snapmod.duelName(allocator, before, side, if (side == 0) c1 else c2);
        }
        var out = Pending{ .parent = parent, .after = after, .side = side, .knowledge = k, .names = &.{}, .choices = undefined, .sent = sent, .json = &.{} };
        out.names = try snapmod.legalNames(allocator, after, result, side, &out.choices);
        out.json = try snapmod.serializeDuelSnapshot(allocator, after, if (side == 0) k else know.swapKnowledge(k), out.names, side == 1, last);
        return out;
    }

    fn fromReply(allocator: std.mem.Allocator, p: *const Pending, reply: remote.Reply) Path {
        var node = Path{
            .knowledge = p.knowledge,
            .memory = vs.PlayerMemory{ .allocator = allocator },
            .effects = .{},
            .step = &.{},
            .parent = p.parent,
            .window_len = 0,
            .logit = reply.win_logit,
            .kv = undefined,
            .remote_id = reply.id,
        };
        // the options' logits in the actor's slots (the slot order is the same rule on both sides; options without a slot stay -inf)
        node.actor = @splat(-std.math.inf(f32));
        for (0..p.names.len) |i| {
            if (i >= reply.n or reply.logits[i] <= remote.UNMAPPED) continue;
            const idx = know.actorIndex(p.after, p.side, p.choices[i]) orelse continue;
            node.actor[idx] = reply.logits[i];
        }
        return node;
    }

    fn remoteTurn(
        allocator: std.mem.Allocator,
        server: *remote.Remote,
        parent: ?*const Path,
        before: *const know.Battle,
        after: *const know.Battle,
        result: pkmn.Result,
        c1: pkmn.Choice,
        c2: pkmn.Choice,
        knowledge: know.Knowledge,
        side: usize,
    ) !Path {
        const p = try pending(allocator, parent, before, after, result, c1, c2, knowledge, side);
        const reply = try server.step(if (parent) |pp| pp.remote_id else -1, p.sent, p.json);
        return fromReply(allocator, &p, reply);
    }

    /// `turn` with the native recurrent model (rnative.zig): the same node as the search server's, computed in-process. With
    /// the search server running too (NATIVE_CHECK=1), every node is also made there and the two are compared.
    fn nativeTurn(
        allocator: std.mem.Allocator,
        nat: *rnative.Native,
        parent: ?*const Path,
        before: *const know.Battle,
        after: *const know.Battle,
        result: pkmn.Result,
        c1: pkmn.Choice,
        c2: pkmn.Choice,
        knowledge: know.Knowledge,
        side: usize,
    ) !Path {
        const node = try allocator.create(rnative.Node);
        const pre = nativePre(nat, parent, before, after, result, c1, c2, knowledge, side, &node.snap);
        var logits: [pkmn.CHOICES_SIZE]f32 = undefined;
        var input: rmodel.Input = undefined;
        try nat.step(if (parent) |p| p.native_node else null, pre.sent, &node.snap, node, &logits, &input);
        var path = nativePath(allocator, parent, after, side, pre.k, &node.snap, node, &logits);
        if (remote.active) |server| try checkAgainstServer(allocator, server, &path, parent, before, after, result, c1, c2, knowledge, side, &node.snap, &logits, &input, node.decision);
        return path;
    }

    const NativePre = struct { k: know.Knowledge, sent: rnative.Option };

    /// The node's knowledge (in the side's orientation) and what its player chose at the parent; its snapshot goes to `snap`.
    fn nativePre(nat: *rnative.Native, parent: ?*const Path, before: *const know.Battle, after: *const know.Battle, result: pkmn.Result, c1: pkmn.Choice, c2: pkmn.Choice, knowledge: know.Knowledge, side: usize, snap: *rnative.Snap) NativePre {
        var k = if (side == 0) knowledge else know.swapKnowledge(knowledge);
        var last = [2]rnative.Act{ .{}, .{} };
        var sent = rnative.Option{};
        if (parent) |p| {
            const step = know.Knowledge.afterTransition(before, after, c1, c2);
            k = know.Knowledge.merge(p.knowledge, if (side == 0) step else know.swapKnowledge(step));
            last = .{ rnative.actOf(before, after, 0, c1), rnative.actOf(before, after, 1, c2) };
            sent = rnative.optionOf(before, side, if (side == 0) c1 else c2);
        }
        rnative.snapInto(snap, nat.tables, after, result, if (side == 0) k else know.swapKnowledge(k), side, last);
        return .{ .k = k, .sent = sent };
    }

    fn nativePath(allocator: std.mem.Allocator, parent: ?*const Path, after: *const know.Battle, side: usize, k: know.Knowledge, snap: *const rnative.Snap, node: *rnative.Node, logits: *const [pkmn.CHOICES_SIZE]f32) Path {
        var path = Path{
            .knowledge = k,
            .memory = vs.PlayerMemory{ .allocator = allocator },
            .effects = .{},
            .step = &.{},
            .parent = parent,
            .window_len = 0,
            .logit = node.win,
            .kv = undefined,
            .native_node = node,
        };
        path.actor = @splat(-std.math.inf(f32));
        for (0..snap.n_legal) |i| {
            if (logits[i] <= rnative.UNMAPPED / 2) continue;
            const idx = know.actorIndex(after, side, snap.choices[i]) orelse continue;
            path.actor[idx] = logits[i];
        }
        return path;
    }

    pub var check_nodes: u64 = 0;
    pub var check_mismatches: u64 = 0;
    pub var check_worst: f32 = 0;

    /// NATIVE_CHECK: the same node on the search server; its win logit and option logits against the native ones, and on a
    /// difference its model input, field by field.
    fn checkAgainstServer(allocator: std.mem.Allocator, server: *remote.Remote, path: *Path, parent: ?*const Path, before: *const know.Battle, after: *const know.Battle, result: pkmn.Result, c1: pkmn.Choice, c2: pkmn.Choice, knowledge: know.Knowledge, side: usize, snap: *const rnative.Snap, logits: *const [pkmn.CHOICES_SIZE]f32, input: *const rmodel.Input, decision: bool) !void {
        const p = try pending(allocator, parent, before, after, result, c1, c2, knowledge, side);
        const r = try server.explain(if (parent) |pp| pp.remote_id else -1, p.sent, p.json);
        path.remote_id = r[0].id;
        check_nodes += 1;
        var worst: f32 = @abs(r[0].win_logit - path.logit);
        if (r[0].n != snap.n_legal) worst = std.math.inf(f32);
        for (0..@min(r[0].n, snap.n_legal)) |i| {
            const a = logits[i];
            const b = r[0].logits[i];
            if ((a <= rnative.UNMAPPED / 2) != (b <= remote.UNMAPPED)) {
                worst = std.math.inf(f32);
            } else if (b > remote.UNMAPPED) worst = @max(worst, @abs(a - b));
        }
        check_worst = @max(check_worst, if (std.math.isInf(worst)) 0 else worst);
        if (worst <= 2e-3) return;
        check_mismatches += 1;
        if (check_mismatches > 8) return;
        std.debug.print("NATIVE_CHECK mismatch #{d} (node {d}, side {d}, turn {d}, decision {any}): win {d:.5} vs server {d:.5}; logits", .{ check_mismatches, check_nodes, side, after.turn, decision, path.logit, r[0].win_logit });
        for (0..snap.n_legal) |i| std.debug.print(" {d:.4}/{d:.4}", .{ logits[i], if (i < r[0].n) r[0].logits[i] else -9 });
        std.debug.print("\n", .{});
        if (!decision) return;
        const feats = std.json.parseFromSliceLeaky(std.json.Value, allocator, r[1], .{}) catch return;
        if (feats != .object or feats.object.get("ids") == null) {
            std.debug.print("  the server did not step the model here (a wait state there)\n", .{});
            return;
        }
        const ids = feats.object.get("ids").?.object;
        const fl = feats.object.get("floats").?.object;
        const num = struct {
            fn f(v: std.json.Value) f64 {
                return switch (v) {
                    .float => |q| q,
                    .integer => |q| @floatFromInt(q),
                    else => 0,
                };
            }
        }.f;
        const id_fields = .{ "own_species", "own_status", "opp_species", "opp_status" };
        inline for (id_fields) |name| {
            const mine = @field(input, name);
            for (ids.get(name).?.array.items, 0..) |v, i| if (@as(i64, mine[i]) != v.integer) std.debug.print("  ids.{s}[{d}]: native {d}, server {d}\n", .{ name, i, mine[i], v.integer });
        }
        inline for (.{ "own_moves", "opp_moves" }) |name| {
            const mine = @field(input, name);
            for (ids.get(name).?.array.items, 0..) |row, i| for (row.array.items, 0..) |v, j| if (@as(i64, mine[i][j]) != v.integer) std.debug.print("  ids.{s}[{d}][{d}]: native {d}, server {d}\n", .{ name, i, j, mine[i][j], v.integer });
        }
        inline for (.{ "last_own_move", "last_opp_move", "last_own_species", "last_opp_species", "last_own_status", "last_opp_status" }) |name| {
            const v = ids.get(name).?.array.items[0].integer;
            if (@as(i64, @field(input, name)) != v) std.debug.print("  ids.{s}: native {d}, server {d}\n", .{ name, @field(input, name), v });
        }
        for (fl.get("own").?.array.items, 0..) |row, i| for (row.array.items, 0..) |v, j| if (@abs(input.own[i][j] - num(v)) > 1e-5) std.debug.print("  floats.own[{d}][{d}]: native {d}, server {d}\n", .{ i, j, input.own[i][j], num(v) });
        for (fl.get("opp").?.array.items, 0..) |row, i| for (row.array.items, 0..) |v, j| if (@abs(input.opp[i][j] - num(v)) > 1e-5) std.debug.print("  floats.opp[{d}][{d}]: native {d}, server {d}\n", .{ i, j, input.opp[i][j], num(v) });
        for (fl.get("last").?.array.items, 0..) |v, j| if (@abs(input.last[j] - num(v)) > 1e-5) std.debug.print("  floats.last[{d}]: native {d}, server {d}\n", .{ j, input.last[j], num(v) });
        for (fl.get("field").?.array.items, 0..) |v, j| if (@abs(input.field[j] - num(v)) > 1e-5) std.debug.print("  floats.field[{d}]: native {d}, server {d}\n", .{ j, input.field[j], num(v) });
    }

    /// One child turn, for `turnBatch`.
    pub const TurnSpec = struct {
        parent: *const Path,
        before: *const know.Battle,
        after: *const know.Battle,
        result: pkmn.Result,
        c1: pkmn.Choice,
        c2: pkmn.Choice,
        side: usize,
    };

    /// `turn` for several independent children: one request to the recurrent server, whose model steps them as one batch (the
    /// windowed model builds them one by one).
    pub fn turnBatch(allocator: std.mem.Allocator, model: ?*const vm.Model, vocab: ?*const std.StringHashMap(u32), specs: []const TurnSpec, out: []Path) !void {
        if (rnative.active) |nat| {
            // the native model steps the batch's decisions together (in NATIVE_CHECK mode one by one, each checked)
            if (remote.active == null) {
                const pres = try allocator.alloc(NativePre, specs.len);
                const parents = try allocator.alloc(?*const rnative.Node, specs.len);
                const sents = try allocator.alloc(rnative.Option, specs.len);
                const nodes = try allocator.alloc(*rnative.Node, specs.len);
                const logits = try allocator.alloc([pkmn.CHOICES_SIZE]f32, specs.len);
                for (specs, 0..) |s, i| {
                    nodes[i] = try allocator.create(rnative.Node);
                    pres[i] = nativePre(nat, s.parent, s.before, s.after, s.result, s.c1, s.c2, .{}, s.side, &nodes[i].snap);
                    parents[i] = s.parent.native_node;
                    sents[i] = pres[i].sent;
                }
                try nat.stepMany(allocator, parents, sents, nodes, logits);
                for (specs, out, 0..) |s, *o, i| o.* = nativePath(allocator, s.parent, s.after, s.side, pres[i].k, &nodes[i].snap, nodes[i], &logits[i]);
                return;
            }
        } else {
            if (remote.active) |server| {
                const pend = try allocator.alloc(Pending, specs.len);
                const items = try allocator.alloc(remote.Item, specs.len);
                for (specs, pend, items) |s, *p, *item| {
                    p.* = try pending(allocator, s.parent, s.before, s.after, s.result, s.c1, s.c2, .{}, s.side);
                    item.* = .{ .parent = s.parent.remote_id, .sent = p.sent, .snap = p.json };
                }
                const replies = try server.batch(allocator, items);
                for (pend, replies, out) |*p, r, *o| o.* = fromReply(allocator, p, r);
                return;
            }
        }
        for (specs, out) |s, *o| o.* = try turn(allocator, model, vocab, s.parent, s.before, s.after, s.result, s.c1, s.c2, .{}, s.side);
    }

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
        // (the recurrent model's state already holds the whole game: there is no window to cut)
        if (node.remote_id >= 0) return node.*;
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
