//! The duel harness's snapshots (what the policy servers read), shared by the duel loop (main.zig) and the search's
//! recurrent-model paths (path.zig): option names, each side's last action, and the serialized observer snapshot.

const std = @import("std");
const pkmn = @import("pkmn");
const know = @import("knowledge.zig");
const vs = @import("value_state.zig");
const policy = @import("policy.zig");

const Battle = know.Battle;

pub fn optionName(allocator: std.mem.Allocator, battle: *const Battle, side: usize, choice: pkmn.Choice) ![]const u8 {
    const active = battle.sides[side].order[0] - 1;
    if (choice.type == .Move) {
        // data 0 is the no-op a recharging Pokémon must take (Hyper Beam); it is not a move slot
        if (choice.data == 0) return "recharge";
        return @tagName(battle.sides[side].pokemon[active].moves[choice.data - 1].id);
    }
    if (choice.type == .Switch) {
        return std.fmt.allocPrint(allocator, "switch {s}", .{@tagName(battle.sides[side].pokemon[battle.sides[side].order[choice.data - 1] - 1].species)});
    }
    return "pass";
}

/// What one side did in the update that produced the current state, as the battle log shows it (public): the move it ran
/// ("none" when it ran none: a switch, sleep, full paralysis, a recharge...), and whether that turn was a recharge.
/// Snapshot readers cannot rebuild this from `last_move`: the engine clears both sides' last_used_move at every switch-in
/// and faint, and its own at a sleep or freeze turn.
pub const LastAction = struct { executed: []const u8 = "none", recharged: bool = false };

pub fn lastAction(allocator: std.mem.Allocator, before: *const Battle, after: *const Battle, side: usize, c: pkmn.Choice) !LastAction {
    const b = before.sides[side];
    const a = after.sides[side];
    if (c.type != .Move or b.order[0] == 0) return .{};
    if (b.active.volatiles.Recharging) return .{ .recharged = true };
    const slot = b.order[0] - 1;
    const forced = b.active.volatiles.Thrashing or b.active.volatiles.Rage or b.active.volatiles.Charging;
    if (!forced and c.data >= 1 and c.data <= 4) {
        const mv = b.active.moves[c.data - 1];
        if (mv.id == .None) return .{};
        // a selected move that runs costs PP (also when it misses); one that never ran (asleep, paralysed...) does not
        if (a.pokemon[slot].moves[c.data - 1].pp < mv.pp) return .{ .executed = try std.ascii.allocLowerString(allocator, @tagName(mv.id)) };
        // a trapping move going on costs no PP
        if (b.active.volatiles.Binding and a.last_used_move == mv.id) return .{ .executed = try std.ascii.allocLowerString(allocator, @tagName(mv.id)) };
        return .{};
    }
    // a locked move (Thrash, Rage, the second turn of a charge move) runs as the last selected one
    const locked = b.last_selected_move;
    if (locked != .None and locked != .SKIP_TURN and a.last_used_move == locked) return .{ .executed = try std.ascii.allocLowerString(allocator, @tagName(locked)) };
    return .{};
}

/// serializeSnapshot plus the last update's actions of both sides (`last_actions`, from P1's side or swapped).
pub fn serializeDuelSnapshot(galloc: std.mem.Allocator, battle: *const Battle, k: know.Knowledge, legal: []const []const u8, swap: bool, last: [2]LastAction) ![]u8 {
    var view = battle.*;
    var kk = k;
    var mine = last[0];
    var theirs = last[1];
    if (swap) {
        std.mem.swap(@TypeOf(view.sides[0]), &view.sides[0], &view.sides[1]);
        kk = know.Knowledge{ .seen = .{ k.seen[1], k.seen[0] }, .moves = .{ k.moves[1], k.moves[0] }, .used = .{ k.used[1], k.used[0] } };
        std.mem.swap(LastAction, &mine, &theirs);
    }
    const snap = try know.observerSnapshot(galloc, &view, kk, false);
    const legal_items = try galloc.alloc(vs.V, legal.len);
    for (legal, legal_items) |name, *item| item.* = .{ .str = name };
    var sides: [2]vs.V = undefined;
    for ([2]LastAction{ mine, theirs }, &sides) |act, *out| {
        const f = try galloc.alloc(vs.Field, 2);
        f[0] = .{ .key = "executed", .value = .{ .str = act.executed } };
        f[1] = .{ .key = "recharged", .value = .{ .bool = act.recharged } };
        out.* = .{ .obj = f };
    }
    const actions = try galloc.alloc(vs.Field, 2);
    actions[0] = .{ .key = "user", .value = sides[0] };
    actions[1] = .{ .key = "opponent", .value = sides[1] };
    const base = snap.obj;
    const fields = try galloc.alloc(vs.Field, base.len + 2);
    @memcpy(fields[0..base.len], base);
    fields[base.len] = .{ .key = "legal", .value = .{ .list = legal_items } };
    fields[base.len + 1] = .{ .key = "last_actions", .value = .{ .obj = actions } };
    var out: std.ArrayList(u8) = .empty;
    try policy.writeJson(galloc, &out, .{ .obj = fields });
    return out.items;
}

/// An option's name as the duel sends it to the policy servers: lowercase, and "no move" for the no-op (pass, recharge).
pub fn duelName(allocator: std.mem.Allocator, battle: *const Battle, side: usize, choice: pkmn.Choice) ![]const u8 {
    const raw = try optionName(allocator, battle, side, choice);
    if (std.mem.eql(u8, raw, "pass") or std.mem.eql(u8, raw, "recharge")) return "no move";
    return std.ascii.allocLowerString(allocator, raw);
}

/// `side`'s legal options at a decision point (`result` is the update's result there) and their duel names.
pub fn legalNames(allocator: std.mem.Allocator, battle: *const Battle, result: pkmn.Result, side: usize, choices: *[pkmn.CHOICES_SIZE]pkmn.Choice) ![]const []const u8 {
    var b = battle.*;
    const n = if (side == 0) b.choices(.P1, result.p1, choices) else b.choices(.P2, result.p2, choices);
    const names = try allocator.alloc([]const u8, n);
    for (0..n) |i| names[i] = try duelName(allocator, battle, side, choices[i]);
    return names;
}
