//! Zig port of the value path's observer state (fp/value/actor_state.py, tokens.py, history.py):
//! observer snapshot -> actor state -> per-roster token groups -> value model step vectors.
//!
//! Values use a small immutable tree (`V`) with insertion-ordered objects, which keeps the same
//! ordering semantics as the Python dicts the trainer used.

const std = @import("std");
const vm = @import("value_model.zig");

pub const V = union(enum) {
    null,
    bool: bool,
    int: i64,
    float: f64,
    str: []const u8,
    list: []const V,
    obj: []const Field,
};

pub const Field = struct { key: []const u8, value: V };

/// Looks up `key` in an object; null if absent or if `v` is not an object.
pub fn get(v: V, key: []const u8) ?V {
    return switch (v) {
        .obj => |fields| for (fields) |f| {
            if (std.mem.eql(u8, f.key, key)) break f.value;
        } else null,
        else => null,
    };
}

/// Converts a parsed std.json tree into `V` (copied into `allocator`).
pub fn fromJson(allocator: std.mem.Allocator, value: std.json.Value) !V {
    return switch (value) {
        .null => .null,
        .bool => |b| .{ .bool = b },
        .integer => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .number_string => |s| .{ .str = s },
        .string => |s| .{ .str = s },
        .array => |arr| blk: {
            const items = try allocator.alloc(V, arr.items.len);
            for (arr.items, 0..) |item, i| items[i] = try fromJson(allocator, item);
            break :blk .{ .list = items };
        },
        .object => |map| blk: {
            const fields = try allocator.alloc(Field, map.count());
            var it = map.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) {
                fields[i] = .{ .key = entry.key_ptr.*, .value = try fromJson(allocator, entry.value_ptr.*) };
            }
            break :blk .{ .obj = fields };
        },
    };
}

/// Python's `norm`: lowercase, keeping only [a-z0-9].
pub fn norm(allocator: std.mem.Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s) |c| {
        const lc = std.ascii.toLower(c);
        if ((lc >= 'a' and lc <= 'z') or (lc >= '0' and lc <= '9')) try out.append(allocator, lc);
    }
    return out.items;
}

fn str(v: ?V) []const u8 {
    const x = v orelse return "";
    return switch (x) {
        .str => |s| s,
        else => "",
    };
}

fn asInt(v: ?V) i64 {
    const x = v orelse return 0;
    return switch (x) {
        .int => |i| i,
        .float => |f| @intFromFloat(f),
        .bool => |b| @intFromBool(b),
        else => 0,
    };
}

fn asFloat(v: ?V) f64 {
    const x = v orelse return 0;
    return switch (x) {
        .int => |i| @floatFromInt(i),
        .float => |f| f,
        else => 0,
    };
}

/// Python truthiness of the snapshot value used for conditions.
fn enabled(allocator: std.mem.Allocator, v: ?V) !bool {
    const x = v orelse return false;
    return switch (x) {
        .str => |s| blk: {
            const n = try norm(allocator, s);
            break :blk !(n.len == 0 or std.mem.eql(u8, n, "0") or std.mem.eql(u8, n, "false") or std.mem.eql(u8, n, "none") or std.mem.eql(u8, n, "noeffect"));
        },
        .bool => |b| b,
        .int => |i| i != 0,
        .float => |f| f != 0,
        .null => false,
        .list => |l| l.len > 0,
        .obj => |o| o.len > 0,
    };
}

const STATUS_MAP = [_][2][]const u8{
    .{ "none", "nostatus" },
    .{ "sleep", "slp" },
    .{ "poison", "psn" },
    .{ "toxic", "tox" },
    .{ "burn", "brn" },
    .{ "freeze", "frz" },
    .{ "paralyze", "par" },
    .{ "fnt", "fnt" },
};

fn statusToken(raw: []const u8) []const u8 {
    for (STATUS_MAP) |pair| {
        if (std.ascii.eqlIgnoreCase(pair[0], raw)) return pair[1];
    }
    return "nostatus";
}

const DIRECT_EFFECTS = [_][]const u8{ "bide", "confusion", "disable", "focusenergy", "leechseed", "lightscreen", "lockedmove", "mimic", "mist", "mustrecharge", "partiallytrapped", "rage", "reflect", "substitute", "typechange" };
const TRAPPING_MOVES = [_][]const u8{ "bind", "clamp", "firespin", "wrap" };
const SCREENS = [_][]const u8{ "reflect", "lightscreen", "mist" };

fn inList(list: []const []const u8, s: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

/// `active_effects(payload)`: the active effect names, in the trainer's order.
fn activeEffects(allocator: std.mem.Allocator, payload: V) ![]const []const u8 {
    var result: std.ArrayList([]const u8) = .empty;
    const explicit = try norm(allocator, str(get(payload, "effect")));
    if (!(explicit.len == 0 or std.mem.eql(u8, explicit, "none") or std.mem.eql(u8, explicit, "noeffect")))
        try result.append(allocator, explicit);

    const last_move = try norm(allocator, str(get(payload, "last_move")));
    const trapping_move = try norm(allocator, str(get(payload, "trapping_move")));
    if (get(payload, "volatiles")) |vols| {
        if (vols == .list) for (vols.list) |raw| {
            var effect = try norm(allocator, str(raw));
            if (std.mem.eql(u8, effect, "lockedmove") or std.mem.eql(u8, effect, "partiallytrapped")) {
                const source = if (std.mem.eql(u8, effect, "lockedmove")) last_move else trapping_move;
                if (inList(&TRAPPING_MOVES, source)) effect = source;
            }
            if ((inList(&DIRECT_EFFECTS, effect) or inList(&TRAPPING_MOVES, effect)) and !inList(result.items, effect))
                try result.append(allocator, effect);
        };
    }
    if (get(payload, "conditions")) |conds| {
        for (SCREENS) |screen| {
            const value: ?V = get(conds, screen);
            if (try enabled(allocator, value) and !inList(result.items, screen)) try result.append(allocator, screen);
        }
    }
    return result.items;
}

/// `SnapshotEffectMemory`: remembers when each effect was first observed, per side.
pub const EffectMemory = struct {
    clock: u64 = 0,
    sides: [2]?Side = .{ null, null },

    const Side = struct {
        identity: []const u8,
        effects: []Tracked,
    };
    const Tracked = struct { name: []const u8, clock: u64 };

    pub fn select(self: *EffectMemory, allocator: std.mem.Allocator, payload: V, side: usize) ![]const u8 {
        const active = try activeEffects(allocator, payload);
        const identity = try norm(allocator, if (get(payload, "base_species")) |b| str(b) else str(get(payload, "id")));
        var record = self.sides[side];
        if (record == null or !std.mem.eql(u8, record.?.identity, identity)) {
            record = .{ .identity = identity, .effects = &.{} };
        }
        var kept: std.ArrayList(Tracked) = .empty;
        for (record.?.effects) |t| {
            if (inList(active, t.name)) try kept.append(allocator, t);
        }
        for (active) |name| {
            var found = false;
            for (kept.items) |t| {
                if (std.mem.eql(u8, t.name, name)) found = true;
            }
            if (!found) {
                self.clock += 1;
                try kept.append(allocator, .{ .name = name, .clock = self.clock });
            }
        }
        record.?.effects = kept.items;
        self.sides[side] = record;
        if (kept.items.len == 0) return "noeffect";
        var best = kept.items[0];
        for (kept.items[1..]) |t| {
            if (t.clock < best.clock) best = t;
        }
        return best.name;
    }
};

fn boostsObj(allocator: std.mem.Allocator, raw: ?V) !V {
    const special = asInt(if (raw) |r| get(r, "spc") orelse get(r, "spa") else null);
    const get_b = struct {
        fn f(r: ?V, key: []const u8) i64 {
            return if (r) |x| asInt(get(x, key)) else 0;
        }
    }.f;
    const fields = try allocator.alloc(Field, 7);
    fields[0] = .{ .key = "atk", .value = .{ .int = get_b(raw, "atk") } };
    fields[1] = .{ .key = "spa", .value = .{ .int = special } };
    fields[2] = .{ .key = "def", .value = .{ .int = get_b(raw, "def") } };
    fields[3] = .{ .key = "spd", .value = .{ .int = if (raw != null and get(raw.?, "spd") != null) get_b(raw, "spd") else special } };
    fields[4] = .{ .key = "spe", .value = .{ .int = get_b(raw, "spe") } };
    fields[5] = .{ .key = "accuracy", .value = .{ .int = get_b(raw, "accuracy") } };
    fields[6] = .{ .key = "evasion", .value = .{ .int = get_b(raw, "evasion") } };
    return .{ .obj = fields };
}

fn typesList(allocator: std.mem.Allocator, raw: ?V) !V {
    var out: std.ArrayList(V) = .empty;
    if (raw) |r| if (r == .list) for (r.list) |t| {
        const n = try norm(allocator, str(t));
        if (n.len == 0 or std.mem.eql(u8, n, "notype")) continue;
        var seen = false;
        for (out.items) |x| if (std.mem.eql(u8, x.str, n)) {
            seen = true;
        };
        if (!seen) try out.append(allocator, .{ .str = n });
    };
    return .{ .list = out.items };
}

/// `_own_pokemon`: one roster entry from the observer's private information.
fn ownPokemon(allocator: std.mem.Allocator, raw: V, active: bool, active_boosts: ?V, effect: []const u8) !V {
    const maximum: i64 = @max(1, asInt(get(raw, "maxhp")));
    var moves: std.ArrayList(V) = .empty;
    if (get(raw, "moves")) |mlist| if (mlist == .list) for (mlist.list) |mv| {
        const name = try norm(allocator, str(get(mv, "id")));
        if (name.len == 0 or std.mem.eql(u8, name, "none")) continue;
        const pp = asInt(get(mv, "pp"));
        const max_pp = if (get(mv, "max_pp")) |m| asInt(m) else pp;
        const fields = try allocator.alloc(Field, 3);
        fields[0] = .{ .key = "name", .value = .{ .str = name } };
        fields[1] = .{ .key = "current_pp", .value = .{ .int = pp } };
        fields[2] = .{ .key = "max_pp", .value = .{ .int = max_pp } };
        try moves.append(allocator, .{ .obj = fields });
    };
    var status = statusToken(str(get(raw, "status")));
    const hp = asInt(get(raw, "hp"));
    if (hp <= 0) status = "fnt";
    const hp_pct = std.math.clamp(@as(f64, @floatFromInt(hp)) / @as(f64, @floatFromInt(maximum)), 0.0, 1.0);
    const fields = try allocator.alloc(Field, 8);
    fields[0] = .{ .key = "species", .value = .{ .str = try norm(allocator, str(get(raw, "id"))) } };
    fields[1] = .{ .key = "base_species", .value = .{ .str = try norm(allocator, if (get(raw, "base_species")) |b| str(b) else str(get(raw, "id"))) } };
    fields[2] = .{ .key = "hp_pct", .value = .{ .float = hp_pct } };
    fields[3] = .{ .key = "types", .value = try typesList(allocator, get(raw, "types")) };
    fields[4] = .{ .key = "status", .value = .{ .str = status } };
    fields[5] = .{ .key = "effect", .value = .{ .str = if (active) effect else "noeffect" } };
    fields[6] = .{ .key = "boosts", .value = try boostsObj(allocator, if (active) active_boosts else null) };
    fields[7] = .{ .key = "moves", .value = .{ .list = moves.items } };
    return .{ .obj = fields };
}

/// `actor_state_from_snapshot`: the state dictionary the token encoder reads.
pub fn actorState(allocator: std.mem.Allocator, snapshot: V, effects: *EffectMemory) !V {
    const user = get(snapshot, "user") orelse return error.NoUser;
    const roster = get(user, "pokemon") orelse return error.NoRoster;
    if (roster != .list or roster.list.len == 0) return error.NoRoster;
    const opponent = get(snapshot, "opponent") orelse .null;
    const turn = asInt(get(snapshot, "turn"));

    // own effect is selected from the active roster entry
    const own_payload = blk: {
        var fields: std.ArrayList(Field) = .empty;
        for (roster.list[0].obj) |f| try fields.append(allocator, f);
        try fields.append(allocator, .{ .key = "turn", .value = .{ .int = turn } });
        try fields.append(allocator, .{ .key = "volatiles", .value = get(user, "volatiles") orelse .{ .list = &.{} } });
        try fields.append(allocator, .{ .key = "conditions", .value = get(user, "conditions") orelse .{ .obj = &.{} } });
        try fields.append(allocator, .{ .key = "last_move", .value = get(user, "last_move") orelse .{ .str = "None" } });
        try fields.append(allocator, .{ .key = "trapping_move", .value = get(opponent, "last_move") orelse .{ .str = "None" } });
        break :blk V{ .obj = fields.items };
    };
    const own_effect = try effects.select(allocator, own_payload, 0);

    var own: std.ArrayList(V) = .empty;
    for (roster.list, 0..) |mon, i| {
        try own.append(allocator, try ownPokemon(allocator, mon, i == 0, get(user, "boosts"), if (i == 0) own_effect else "noeffect"));
    }

    // the opponent's active Pokémon, from public information
    const foe_active_species = try norm(allocator, str(get(opponent, "active_species")));
    const active_public = get(opponent, "active") orelse .null;
    const active_base = try norm(allocator, if (get(active_public, "base_species")) |b| str(b) else foe_active_species);
    const revealed = if (get(opponent, "revealed")) |r| (if (r == .list) r.list else &[_]V{}) else &[_]V{};
    var revealed_active: V = .{ .obj = &.{} };
    for (revealed) |mon| {
        if (std.mem.eql(u8, try norm(allocator, str(get(mon, "species"))), active_base)) {
            revealed_active = mon;
            break;
        }
    }
    const foe_hp = asFloat(get(active_public, "hp"));
    const foe_status = statusToken(str(get(active_public, "status")));
    var foe_moves: std.ArrayList([]const u8) = .empty;
    if (get(revealed_active, "moves")) |mv| if (mv == .list) for (mv.list) |m| {
        const n = try norm(allocator, str(get(m, "id")));
        if (n.len > 0 and !std.mem.eql(u8, n, "none")) try foe_moves.append(allocator, n);
    };
    std.mem.sort([]const u8, foe_moves.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    var foe_moves_v: std.ArrayList(V) = .empty;
    for (foe_moves.items) |m| try foe_moves_v.append(allocator, .{ .str = m });

    const foe_payload = blk: {
        var fields: std.ArrayList(Field) = .empty;
        if (active_public == .obj) for (active_public.obj) |f| {
            if (std.mem.eql(u8, f.key, "base_species") or std.mem.eql(u8, f.key, "id")) continue;
            try fields.append(allocator, f);
        };
        try fields.append(allocator, .{ .key = "id", .value = .{ .str = foe_active_species } });
        try fields.append(allocator, .{ .key = "base_species", .value = .{ .str = active_base } });
        try fields.append(allocator, .{ .key = "turn", .value = .{ .int = turn } });
        try fields.append(allocator, .{ .key = "last_move", .value = get(opponent, "last_move") orelse .{ .str = "None" } });
        try fields.append(allocator, .{ .key = "trapping_move", .value = get(user, "last_move") orelse .{ .str = "None" } });
        break :blk V{ .obj = fields.items };
    };
    const foe_effect = try effects.select(allocator, foe_payload, 1);
    const foe_boosts = try boostsObj(allocator, get(active_public, "boosts"));
    const foe_types = try typesList(allocator, get(active_public, "types"));

    const foe_fields = try allocator.alloc(Field, 8);
    foe_fields[0] = .{ .key = "species", .value = .{ .str = foe_active_species } };
    foe_fields[1] = .{ .key = "base_species", .value = .{ .str = active_base } };
    var foe_hp_clamped = std.math.clamp(foe_hp, 0.0, 1.0);
    if (foe_hp_clamped != foe_hp_clamped) foe_hp_clamped = 0;
    foe_fields[2] = .{ .key = "hp_pct", .value = .{ .float = foe_hp_clamped } };
    foe_fields[3] = .{ .key = "types", .value = foe_types };
    foe_fields[4] = .{ .key = "status", .value = .{ .str = foe_status } };
    foe_fields[5] = .{ .key = "effect", .value = .{ .str = foe_effect } };
    foe_fields[6] = .{ .key = "boosts", .value = foe_boosts };
    foe_fields[7] = .{ .key = "revealed_moves", .value = .{ .list = foe_moves_v.items } };

    // alive reserve (available switches), and party bookkeeping
    var switches: std.ArrayList(V) = .empty;
    for (own.items[1..]) |mon| {
        const st = str(get(mon, "status"));
        const hp_pct = asFloat(get(mon, "hp_pct"));
        if (!std.mem.eql(u8, st, "fnt") and hp_pct > 0) try switches.append(allocator, mon);
    }
    var party: std.ArrayList([]const u8) = .empty;
    for (revealed) |mon| {
        const n = try norm(allocator, str(get(mon, "species")));
        var dup = n.len == 0;
        for (party.items) |p| if (std.mem.eql(u8, p, n)) {
            dup = true;
        };
        if (!dup) try party.append(allocator, n);
    }
    std.mem.sort([]const u8, party.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    var party_v: std.ArrayList(V) = .empty;
    for (party.items) |p| try party_v.append(allocator, .{ .str = p });

    var actor_remaining: i64 = 0;
    for (own.items) |mon| {
        const st = str(get(mon, "status"));
        if (!std.mem.eql(u8, st, "fnt") and asFloat(get(mon, "hp_pct")) > 0) actor_remaining += 1;
    }
    var fainted: i64 = 0;
    for (revealed) |mon| {
        if (std.mem.eql(u8, str(get(mon, "status")), "fnt")) fainted += 1;
    }
    const last_user = try norm(allocator, str(get(user, "last_move")));
    const last_foe = try norm(allocator, str(get(opponent, "last_move")));
    const user_move_name = if (last_user.len == 0 or std.mem.eql(u8, last_user, "none")) "nomove" else last_user;
    const foe_move_name = if (last_foe.len == 0 or std.mem.eql(u8, last_foe, "none")) "nomove" else last_foe;

    const state = try allocator.alloc(Field, 15);
    state[0] = .{ .key = "frame", .value = .{ .int = turn } };
    state[1] = .{ .key = "format", .value = .{ .str = "gen1ou" } };
    state[2] = .{ .key = "actor_active", .value = own.items[0] };
    state[3] = .{ .key = "available_switches", .value = .{ .list = switches.items } };
    state[4] = .{ .key = "opponent_active", .value = .{ .obj = foe_fields } };
    state[5] = .{ .key = "opponent_revealed_party", .value = .{ .list = party_v.items } };
    state[6] = .{ .key = "actor_remaining", .value = .{ .int = actor_remaining } };
    state[7] = .{ .key = "opponent_remaining", .value = .{ .int = 6 - fainted } };
    state[8] = .{ .key = "actor_previous_move", .value = .{ .str = user_move_name } };
    state[9] = .{ .key = "opponent_previous_move", .value = .{ .str = foe_move_name } };
    state[10] = .{ .key = "actor_conditions", .value = .{ .str = "noconditions" } };
    state[11] = .{ .key = "opponent_conditions", .value = .{ .str = "noconditions" } };
    state[12] = .{ .key = "weather", .value = .{ .str = "noweather" } };
    state[13] = .{ .key = "battle_field", .value = .{ .str = "nofield" } };
    state[14] = .{ .key = "forced_switch", .value = .{ .bool = blk: {
        const f: V = get(snapshot, "force_switch") orelse V{ .bool = false };
        break :blk switch (f) {
            .bool => |b| b,
            else => false,
        };
    } } };
    return .{ .obj = state };
}

const GROUPS = vm.GROUPS;
const GLOBAL_GROUP = 14;
const OWN_OFFSET = 0;
const FOE_OFFSET = 6;
const GLOBAL_FIELDS = [_][]const u8{ "format", "actor_remaining", "opponent_remaining", "actor_previous_move", "opponent_previous_move", "actor_conditions", "opponent_conditions", "weather", "battle_field", "forced_switch" };

/// Python's `round(x)` for a float: round half to even.
fn roundHalfEven(x: f64) i64 {
    const f = @floor(x);
    const diff = x - f;
    var r = f;
    if (diff > 0.5) {
        r = f + 1;
    } else if (diff == 0.5) {
        r = if (@mod(f, 2) == 0) f else f + 1;
    }
    return @intFromFloat(r);
}

/// Python's `tokens(value, prefix)`: `path=value` strings, objects by sorted key, lists sorted by
/// name/species and indexed, hp_pct as whole percent.
pub fn tokens(allocator: std.mem.Allocator, value: V, prefix: []const u8, out: *std.ArrayList([]const u8)) !void {
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
                try tokens(allocator, f.value, child, out);
            }
        },
        .list => |items| {
            const keyed = try allocator.alloc(V, items.len);
            @memcpy(keyed, items);
            // stable insertion sort by name (objects) or the string itself
            var i: usize = 1;
            while (i < keyed.len) : (i += 1) {
                var j = i;
                while (j > 0 and std.mem.lessThan(u8, sortKey(keyed[j]), sortKey(keyed[j - 1]))) : (j -= 1) {
                    std.mem.swap(V, &keyed[j], &keyed[j - 1]);
                }
            }
            for (keyed, 0..) |item, idx| {
                try tokens(allocator, item, try std.fmt.allocPrint(allocator, "{s}[{d}]", .{ prefix, idx }), out);
            }
        },
        .int, .float, .bool, .str, .null => {
            if (std.mem.endsWith(u8, prefix, "hp_pct") and (value == .float or value == .int)) {
                const v: f64 = if (value == .float) value.float else @floatFromInt(value.int);
                const pct = roundHalfEven(std.math.clamp(v, 0.0, 1.0) * 100.0);
                try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={d}", .{ prefix, pct }));
                return;
            }
            try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={s}", .{ prefix, try pyText(allocator, value) }));
        },
    }
}

fn sortKey(v: V) []const u8 {
    return switch (v) {
        .obj => |fields| blk: {
            for (fields) |f| {
                if (std.mem.eql(u8, f.key, "name")) break :blk str(f.value);
            }
            for (fields) |f| {
                if (std.mem.eql(u8, f.key, "species")) break :blk str(f.value);
            }
            break :blk "";
        },
        .str => |s| s,
        else => "",
    };
}

/// Python's `str()` for the scalar values in the value path.
fn pyText(allocator: std.mem.Allocator, value: V) ![]const u8 {
    return switch (value) {
        .null => "None",
        .bool => |b| if (b) "True" else "False",
        .int => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| try std.fmt.allocPrint(allocator, "{d}", .{f}),
        .str => |s| s,
        .list, .obj => "",
    };
}

/// A roster entry as stored in memory: the first-observed (or latest) public data and when it
/// was last seen.
const Slot = struct { key: []const u8, data: V, last_seen: u64 };

/// `PlayerMemory`: stable roster slots (insertion order matters: it fixes the group positions).
pub const PlayerMemory = struct {
    allocator: std.mem.Allocator,
    own: std.ArrayList(Slot) = .empty,
    foe: std.ArrayList(Slot) = .empty,
    step: u64 = 0,

    pub fn clone(self: *const PlayerMemory) !PlayerMemory {
        return self.cloneInto(self.allocator);
    }

    pub fn cloneInto(self: *const PlayerMemory, allocator: std.mem.Allocator) !PlayerMemory {
        var copy = PlayerMemory{ .allocator = allocator, .step = self.step };
        try copy.own.appendSlice(allocator, self.own.items);
        try copy.foe.appendSlice(allocator, self.foe.items);
        return copy;
    }

    fn upsert(self: *PlayerMemory, list: *std.ArrayList(Slot), key: []const u8, data: V, step: u64) !void {
        for (list.items) |*slot| {
            if (std.mem.eql(u8, slot.key, key)) {
                slot.data = data;
                slot.last_seen = step;
                return;
            }
        }
        try list.append(self.allocator, .{ .key = key, .data = data, .last_seen = step });
    }

    fn setDefault(self: *PlayerMemory, list: *std.ArrayList(Slot), key: []const u8, data: V) !void {
        for (list.items) |slot| if (std.mem.eql(u8, slot.key, key)) return;
        try list.append(self.allocator, .{ .key = key, .data = data, .last_seen = 0 });
    }

    /// Token strings for the 15 groups of one decision.
    pub fn encode(self: *PlayerMemory, state: V) ![GROUPS][]const []const u8 {
        self.step += 1;
        const active = get(state, "actor_active") orelse return error.NoActive;
        const foe_active = get(state, "opponent_active") orelse return error.NoActive;
        var available: std.ArrayList(V) = .empty;
        try available.append(self.allocator, active);
        if (get(state, "available_switches")) |sw| if (sw == .list) try available.appendSlice(self.allocator, sw.list);

        const active_key = rosterKey(active);
        const foe_key = rosterKey(foe_active);
        // sorted by roster key (stable), first-observed slots keep their positions
        var sorted_available: std.ArrayList(V) = .empty;
        try sorted_available.appendSlice(self.allocator, available.items);
        var i: usize = 1;
        while (i < sorted_available.items.len) : (i += 1) {
            var j = i;
            while (j > 0 and std.mem.lessThan(u8, rosterKey(sorted_available.items[j]), rosterKey(sorted_available.items[j - 1]))) : (j -= 1) {
                std.mem.swap(V, &sorted_available.items[j], &sorted_available.items[j - 1]);
            }
        }
        for (sorted_available.items) |p| try self.upsert(&self.own, rosterKey(p), p, self.step);

        // public fields of the opponent's active Pokémon only
        var public: std.ArrayList(Field) = .empty;
        const allowed = [_][]const u8{ "species", "base_species", "hp_pct", "types", "status", "effect", "boosts", "revealed_moves" };
        if (foe_active == .obj) for (foe_active.obj) |f| {
            for (allowed) |a| if (std.mem.eql(u8, a, f.key)) try public.append(self.allocator, f);
        };
        try self.upsert(&self.foe, foe_key, .{ .obj = public.items }, self.step);

        if (get(state, "opponent_revealed_party")) |party| if (party == .list) {
            for (party.list) |sp| {
                const name = str(sp);
                const fields = try self.allocator.alloc(Field, 1);
                fields[0] = .{ .key = "species", .value = .{ .str = name } };
                try self.setDefault(&self.foe, name, .{ .obj = fields });
            }
        };
        if (self.own.items.len > 6 or self.foe.items.len > 6) {
            for (self.own.items) |e| std.debug.print("own key {s}\n", .{e.key});
            for (self.foe.items) |e| std.debug.print("foe key {s}\n", .{e.key});
            return error.TooManySpecies;
        }

        var groups: [GROUPS][]const []const u8 = undefined;
        for (&groups) |*g| g.* = &.{};
        try self.emitSide(&groups, self.own.items, active_key, OWN_OFFSET, true, available.items);
        try self.emitSide(&groups, self.foe.items, foe_key, FOE_OFFSET, false, available.items);

        var misc: std.ArrayList(Field) = .empty;
        for (GLOBAL_FIELDS) |name| {
            if (get(state, name)) |v| try misc.append(self.allocator, .{ .key = name, .value = v });
        }
        try misc.append(self.allocator, .{ .key = "history_step_bucket", .value = .{ .int = @intCast(@min(self.step / 8, 64)) } });
        var global_tokens: std.ArrayList([]const u8) = .empty;
        try tokens(self.allocator, .{ .obj = misc.items }, "global", &global_tokens);
        groups[GLOBAL_GROUP] = global_tokens.items;
        return groups;
    }

    fn emitSide(self: *PlayerMemory, groups: *[GROUPS][]const []const u8, slots: []const Slot, active_key: []const u8, offset: usize, own_side: bool, available: []const V) !void {
        for (slots, 0..) |slot, idx| {
            var fields: std.ArrayList(Field) = .empty;
            const is_active = std.mem.eql(u8, slot.key, active_key);
            for (slot.data.obj) |f| {
                if (std.mem.eql(u8, f.key, "base_species")) continue;
                if (own_side and std.mem.eql(u8, f.key, "revealed_moves")) continue;
                if (!is_active and (std.mem.eql(u8, f.key, "boosts") or std.mem.eql(u8, f.key, "effect"))) continue;
                try fields.append(self.allocator, f);
            }
            if (!is_active) {
                try fields.append(self.allocator, .{ .key = "boosts", .value = .{ .obj = &.{} } });
                try fields.append(self.allocator, .{ .key = "effect", .value = .{ .str = "noeffect" } });
            }
            try fields.append(self.allocator, .{ .key = "is_active", .value = .{ .bool = is_active } });
            const age: i64 = @intCast(@min(self.step - slot.last_seen, 32));
            try fields.append(self.allocator, .{ .key = "last_seen_age", .value = .{ .int = age } });
            if (own_side) {
                var available_now = false;
                for (available) |a| if (std.mem.eql(u8, rosterKey(a), slot.key)) {
                    available_now = true;
                };
                try fields.append(self.allocator, .{ .key = "currently_available", .value = .{ .bool = available_now } });
            }
            var out: std.ArrayList([]const u8) = .empty;
            try tokens(self.allocator, .{ .obj = fields.items }, "pokemon", &out);
            groups[offset + idx] = out.items;
        }
    }
};

fn rosterKey(v: V) []const u8 {
    if (get(v, "base_species")) |b| return str(b);
    return str(get(v, "species"));
}

/// Per-battle value state: the encoder memory, the effect memory and the recorded step vectors.
pub const Pipeline = struct {
    allocator: std.mem.Allocator,
    model: *const vm.Model,
    vocab: *const std.StringHashMap(u32),
    memory: PlayerMemory,
    effects: EffectMemory = .{},
    steps: std.ArrayList([]f32) = .empty,

    pub fn init(allocator: std.mem.Allocator, model: *const vm.Model, vocab: *const std.StringHashMap(u32)) Pipeline {
        return .{ .allocator = allocator, .model = model, .vocab = vocab, .memory = .{ .allocator = allocator } };
    }

    /// Token ids of the 15 groups, with unknown tokens mapped to UNK (1).
    fn groupIds(self: *const Pipeline, groups: [GROUPS][]const []const u8) ![GROUPS][]u32 {
        var out: [GROUPS][]u32 = undefined;
        for (groups, 0..) |group, g| {
            const ids = try self.allocator.alloc(u32, group.len);
            for (group, ids) |token, *dst| dst.* = self.vocab.get(token) orelse 1;
            out[g] = ids;
        }
        return out;
    }

    /// Step vector and win logit for a snapshot's encoding, given the memory it starts from.
    fn stepFor(self: *Pipeline, memory: *PlayerMemory, effects: *EffectMemory, snapshot: V) ![]f32 {
        const state = try actorState(self.allocator, snapshot, effects);
        const groups = try memory.encode(state);
        const ids = try self.groupIds(groups);
        var slices: [GROUPS][]const u32 = undefined;
        for (ids, 0..) |g, i| slices[i] = g;
        return self.model.encodeStep(self.allocator, &slices);
    }

    /// P(observer wins) if `snapshot` is the next decision. Does not change the recorded history.
    pub fn value(self: *Pipeline, snapshot: V) !f64 {
        var memory = try self.memory.clone();
        var effects = self.effects;
        const step = try self.stepFor(&memory, &effects, snapshot);
        const max_keep: usize = vm.MAX_WINDOW - 1;
        const keep: usize = @min(self.steps.items.len, max_keep);
        const start = self.steps.items.len - keep;
        const window = try self.allocator.alloc([]const f32, keep + 1);
        for (self.steps.items[start..], 0..) |s, i| window[i] = s;
        window[keep] = step;
        const logit = try self.model.winLogit(self.allocator, window);
        return 1.0 / (1.0 + @exp(-@as(f64, logit)));
    }

    /// Actor logits (nine action slots) for `snapshot` as the next decision. Does not change the history.
    pub fn actor(self: *Pipeline, snapshot: V) ![vm.ACTIONS]f32 {
        var memory = try self.memory.clone();
        var effects = self.effects;
        const step = try self.stepFor(&memory, &effects, snapshot);
        const max_keep: usize = vm.MAX_WINDOW - 1;
        const keep: usize = @min(self.steps.items.len, max_keep);
        const start = self.steps.items.len - keep;
        const window = try self.allocator.alloc([]const f32, keep + 1);
        for (self.steps.items[start..], 0..) |s_, i| window[i] = s_;
        window[keep] = step;
        const fwd = try self.model.windowForwardState(self.allocator, window, null);
        return self.model.actorLogits(&fwd.rep);
    }

    /// Records the decision for `snapshot` (one step per turn, in order).
    pub fn record(self: *Pipeline, snapshot: V) !void {
        const step = try self.stepFor(&self.memory, &self.effects, snapshot);
        try self.steps.append(self.allocator, step);
    }
};
