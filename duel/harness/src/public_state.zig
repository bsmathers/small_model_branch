//! Live public frames for the opponent-team belief model (fp/belief/public_state.py), built from
//! the engine's battle state. A frame holds only what the searching player may observe: both
//! actives, revealed party members, moves that have been used, remaining counts and the previous
//! actions. Tokens follow the trainer's belief encoding (tokenizer.zig in fp_belief).

const std = @import("std");
const pkmn = @import("pkmn");
const vs = @import("value_state.zig");
const belief = @import("belief.zig");

pub const Battle = pkmn.gen1.Battle(pkmn.gen1.PRNG);
const V = vs.V;
const Field = vs.Field;

/// Lowercase alphanumeric form of an enum tag (matches the trainer's vocabulary names).
fn lowerName(allocator: std.mem.Allocator, tag: []const u8) ![]const u8 {
    return vs.norm(allocator, tag);
}

fn statusName(status: u8) []const u8 {
    if (status == 0) return "nostatus";
    if (pkmn.gen1.Status.is(status, .SLP)) return "slp";
    if (pkmn.gen1.Status.is(status, .PSN)) return "psn";
    if (pkmn.gen1.Status.is(status, .BRN)) return "brn";
    if (pkmn.gen1.Status.is(status, .FRZ)) return "frz";
    if (pkmn.gen1.Status.is(status, .PAR)) return "par";
    return "nostatus";
}

/// Per-battle observation record: what each side has revealed, and the frames so far.
pub const History = struct {
    allocator: std.mem.Allocator,
    /// Species seen on each side (active or switched in), in first-seen order.
    seen: [2]std.ArrayList([]const u8) = .{ .empty, .empty },
    /// Moves revealed for each side's party slots (persist across switches).
    revealed: [2][6]std.ArrayList([]const u8) = @splat(@splat(.empty)),
    /// Moves used by the user's active Pokémon since it switched in.
    used_since_switch: std.ArrayList([]const u8) = .empty,
    last_move: [2][]const u8 = .{ "", "" },
    frames: std.ArrayList(Frame) = .empty,

    pub const Frame = struct {
        turn: u16,
        public: V,
        own_action: ?[]const u8 = null,
        opponent_action: ?[]const u8 = null,
    };

    pub fn init(allocator: std.mem.Allocator) History {
        return .{ .allocator = allocator };
    }

    fn addUnique(self: *History, list: *std.ArrayList([]const u8), name: []const u8) !void {
        for (list.items) |x| if (std.mem.eql(u8, x, name)) return;
        try list.append(self.allocator, name);
    }

    /// The active slot (0-based party index) of a side.
    fn activeSlot(battle: *const Battle, side: usize) usize {
        return battle.sides[side].order[0] - 1;
    }

    /// Records observed actions after the transition. A blocked opponent choice is private.
    pub fn noteTransition(self: *History, battle: *const Battle, after: *const Battle, c1: pkmn.Choice, c2: pkmn.Choice) !void {
        const choices = [2]pkmn.Choice{ c1, c2 };
        var tokens: [2]?[]const u8 = .{ null, null };
        for (0..2) |side| {
            const choice = choices[side];
            const slot = activeSlot(battle, side);
            const mon = battle.sides[side].pokemon[slot];
            switch (choice.type) {
                .Move => {
                    if (choice.data == 0 or choice.data > 4) continue;
                    const move = battle.sides[side].active.moves[choice.data - 1].id;
                    // PP is consumed when a move executes, including misses. No consumption
                    // means a selected but blocked move must not reveal the opponent's intent.
                    const executed = after.sides[side].last_used_move == move and
                        (after.sides[side].pokemon[slot].moves[choice.data - 1].pp < mon.moves[choice.data - 1].pp);
                    if (side == 1 and !executed) continue;
                    const name = try lowerName(self.allocator, @tagName(move));
                    tokens[side] = try std.fmt.allocPrint(self.allocator, "MOVE:{s}", .{name});
                    if (!executed) continue;
                    try self.addUnique(&self.revealed[side][slot], name);
                    self.last_move[side] = name;
                    if (side == 0) try self.addUnique(&self.used_since_switch, name);
                },
                .Switch => {
                    const target = battle.sides[side].order[choice.data - 1] - 1;
                    const species = try lowerName(self.allocator, @tagName(battle.sides[side].pokemon[target].species));
                    tokens[side] = try std.fmt.allocPrint(self.allocator, "SWITCH:{s}", .{species});
                    self.last_move[side] = "";
                    if (side == 0) self.used_since_switch.clearRetainingCapacity();
                },
                else => {},
            }
        }
        if (self.frames.items.len > 0) {
            const previous = &self.frames.items[self.frames.items.len - 1];
            previous.own_action = tokens[0];
            previous.opponent_action = tokens[1];
        }
    }

    /// Appends the frame for the current decision.
    pub fn record(self: *History, battle: *const Battle) !void {
        var fields: std.ArrayList(Field) = .empty;
        try fields.append(self.allocator, .{ .key = "format", .value = .{ .str = "gen1ou" } });
        try fields.append(self.allocator, .{ .key = "battle_field", .value = .{ .str = "nofield" } });
        try fields.append(self.allocator, .{ .key = "weather", .value = .{ .str = "noweather" } });
        try fields.append(self.allocator, .{ .key = "frame", .value = .{ .int = battle.turn } });
        for (0..2) |side| {
            const prefix = if (side == 0) "p1" else "p2";
            const slot = activeSlot(battle, side);
            const mon = battle.sides[side].pokemon[slot];
            const active = battle.sides[side].active;

            // species seen so far, including the active one
            const species = try lowerName(self.allocator, @tagName(mon.species));
            try self.addUnique(&self.seen[side], species);

            var fainted: i64 = 0;
            for (battle.sides[side].pokemon) |pk| {
                if (pk.hp == 0) fainted += 1;
            }
            const moves = if (side == 0) self.used_since_switch.items else self.revealed[side][slot].items;
            const sorted_moves = try self.allocator.dupe([]const u8, moves);
            std.mem.sort([]const u8, sorted_moves, {}, struct {
                fn lt(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lt);
            var revealed_v: std.ArrayList(V) = .empty;
            for (sorted_moves) |m| try revealed_v.append(self.allocator, .{ .str = m });

            var party: std.ArrayList([]const u8) = .empty;
            for (self.seen[side].items) |s| try party.append(self.allocator, s);
            std.mem.sort([]const u8, party.items, {}, struct {
                fn lt(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lt);
            var party_v: std.ArrayList(V) = .empty;
            for (party.items) |s| try party_v.append(self.allocator, .{ .str = s });

            const types = try self.typesValue(mon.types);
            const boosts = try self.boostsValue(active.boosts);
            const hp_pct = @as(f64, @floatFromInt(mon.hp)) / @as(f64, @floatFromInt(@max(mon.stats.hp, 1)));

            const active_fields = try self.allocator.alloc(Field, 7);
            active_fields[0] = .{ .key = "species", .value = .{ .str = species } };
            active_fields[1] = .{ .key = "hp_pct", .value = .{ .float = hp_pct } };
            active_fields[2] = .{ .key = "types", .value = types };
            active_fields[3] = .{ .key = "status", .value = .{ .str = statusName(mon.status) } };
            active_fields[4] = .{ .key = "effect", .value = .{ .str = "noeffect" } };
            active_fields[5] = .{ .key = "boosts", .value = boosts };
            active_fields[6] = .{ .key = "revealed_moves", .value = .{ .list = revealed_v.items } };

            const prev = self.last_move[side];
            try fields.append(self.allocator, .{ .key = try std.fmt.allocPrint(self.allocator, "{s}_active", .{prefix}), .value = .{ .obj = active_fields } });
            try fields.append(self.allocator, .{ .key = try std.fmt.allocPrint(self.allocator, "{s}_revealed_party", .{prefix}), .value = .{ .list = party_v.items } });
            try fields.append(self.allocator, .{ .key = try std.fmt.allocPrint(self.allocator, "{s}_remaining", .{prefix}), .value = .{ .int = 6 - fainted } });
            try fields.append(self.allocator, .{ .key = try std.fmt.allocPrint(self.allocator, "{s}_previous_move", .{prefix}), .value = .{ .str = if (prev.len == 0) "nomove" else prev } });
            try fields.append(self.allocator, .{ .key = try std.fmt.allocPrint(self.allocator, "{s}_conditions", .{prefix}), .value = .{ .str = "noconditions" } });
            try fields.append(self.allocator, .{ .key = try std.fmt.allocPrint(self.allocator, "{s}_forced_switch", .{prefix}), .value = .{ .bool = mon.hp == 0 and side == 0 } });
        }
        try self.frames.append(self.allocator, .{ .turn = battle.turn, .public = .{ .obj = fields.items } });
    }

    fn typesValue(self: *History, types: pkmn.gen1.Types) !V {
        var list: std.ArrayList(V) = .empty;
        for ([2]pkmn.gen1.Type{ types.type1, types.type2 }) |t| {
            const name = try lowerName(self.allocator, @tagName(t));
            if (std.mem.eql(u8, name, "typeless") or std.mem.eql(u8, name, "none")) continue;
            var dup = false;
            for (list.items) |x| if (std.mem.eql(u8, x.str, name)) {
                dup = true;
            };
            if (!dup) try list.append(self.allocator, .{ .str = name });
        }
        return .{ .list = list.items };
    }

    /// Gen I has one special stat: spa and spd both report the special boost.
    fn boostsValue(self: *History, b: pkmn.gen1.Boosts) !V {
        const fields = try self.allocator.alloc(Field, 7);
        fields[0] = .{ .key = "accuracy", .value = .{ .int = b.accuracy } };
        fields[1] = .{ .key = "atk", .value = .{ .int = b.atk } };
        fields[2] = .{ .key = "def", .value = .{ .int = b.def } };
        fields[3] = .{ .key = "evasion", .value = .{ .int = 0 } };
        fields[4] = .{ .key = "spa", .value = .{ .int = b.spc } };
        fields[5] = .{ .key = "spd", .value = .{ .int = b.spc } };
        fields[6] = .{ .key = "spe", .value = .{ .int = b.spe } };
        return .{ .obj = fields };
    }
};

/// Belief-model tokens for a value (fp_belief's `public_tokens`): sorted keys, `frame` skipped,
/// hp_pct bucketed into 21 steps, and Python-style scalars.
pub fn beliefTokens(allocator: std.mem.Allocator, value: V, prefix: []const u8, out: *std.ArrayList([]const u8)) !void {
    switch (value) {
        .obj => |fields| {
            var keys: std.ArrayList(Field) = .empty;
            for (fields) |f| {
                if (std.mem.eql(u8, f.key, "frame")) continue;
                try keys.append(allocator, f);
            }
            std.mem.sort(Field, keys.items, {}, struct {
                fn lt(_: void, a: Field, b: Field) bool {
                    return std.mem.lessThan(u8, a.key, b.key);
                }
            }.lt);
            for (keys.items) |f| {
                const child = if (prefix.len == 0) f.key else try std.fmt.allocPrint(allocator, "{s}.{s}", .{ prefix, f.key });
                try beliefTokens(allocator, f.value, child, out);
            }
        },
        .list => |items| {
            for (items) |item| try beliefTokens(allocator, item, try std.fmt.allocPrint(allocator, "{s}[]", .{prefix}), out);
        },
        .float, .int => {
            if (std.mem.endsWith(u8, prefix, "hp_pct")) {
                const v: f64 = if (value == .float) value.float else @floatFromInt(value.int);
                const bucket: i64 = @intFromFloat(std.math.clamp(@round(v * 20.0), 0, 20));
                try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={d}", .{ prefix, bucket }));
                return;
            }
            try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={s}", .{ prefix, try scalarText(allocator, value) }));
        },
        else => try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={s}", .{ prefix, try scalarText(allocator, value) })),
    }
}

fn scalarText(allocator: std.mem.Allocator, value: V) ![]const u8 {
    return switch (value) {
        .null => "None",
        .bool => |b| if (b) "True" else "False",
        .int => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| try std.fmt.allocPrint(allocator, "{d}", .{f}),
        .str => |s| s,
        .list, .obj => "",
    };
}

/// Token ids for each frame, as causal_frame_tokens builds them: frame i also carries the actions
/// of frame i-1 (`previous.*`).
pub fn frameIds(allocator: std.mem.Allocator, vocab: *const belief.Vocab, frames: []const History.Frame) ![][]u32 {
    const out = try allocator.alloc([]u32, frames.len);
    for (frames, 0..) |frame, i| {
        var tokens: std.ArrayList([]const u8) = .empty;
        try beliefTokens(allocator, frame.public, "", &tokens);
        if (i > 0) {
            const prev = frames[i - 1];
            try tokens.append(allocator, try std.fmt.allocPrint(allocator, "previous.own_action={s}", .{prev.own_action orelse "None"}));
            try tokens.append(allocator, try std.fmt.allocPrint(allocator, "previous.opponent_action={s}", .{prev.opponent_action orelse "None"}));
        }
        var ids: std.ArrayList(u32) = .empty;
        for (tokens.items) |t| try ids.append(allocator, vocab.tokenId(t));
        if (ids.items.len == 0) try ids.append(allocator, belief.UNK_ID);
        out[i] = ids.items;
    }
    return out;
}
