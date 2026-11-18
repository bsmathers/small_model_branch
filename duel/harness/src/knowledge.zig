//! Public knowledge for the observer, as in bot2's `Knowledge`, with the two changes this port
//! makes: knowledge is a small value (copied cheaply into every search node), and the observer
//! snapshot built from it contains only what the observer can see. Party slots identify species
//! (Gen I's species clause keeps them unique per side).

const std = @import("std");
const pkmn = @import("pkmn");
const vs = @import("value_state.zig");

pub const Battle = pkmn.gen1.Battle(pkmn.gen1.PRNG);
pub const Move = pkmn.gen1.Move;
const MAX_MOVES = 4;

pub const Knowledge = struct {
    /// Party slots (0-based) of each side that have been seen as active.
    seen: [2][6]bool = @splat(@splat(false)),
    /// Moves revealed by each party slot; `.None` marks an empty entry.
    moves: [2][6][MAX_MOVES]Move = @splat(@splat(@splat(.None))),

    /// Public PP consumption, indexed like moves (never copied from hidden PP).
    used: [2][6][MAX_MOVES]u8 = @splat(@splat(@splat(0))),

    fn revealMove(self: *Knowledge, side: usize, slot: usize, move: Move) void {
        for (self.moves[side][slot]) |m| if (m == move) return;
        for (&self.moves[side][slot]) |*m| {
            if (m.* == .None) {
                m.* = move;
                return;
            }
        }
    }

    /// Advances public knowledge across one simulated turn. `before` is the state the choices were
    /// made in and `after` the state they produced. A selected move is revealed only if it executed
    /// (the side's last used move is that move); full paralysis, sleep and failed replacements do
    /// not reveal it.
    pub fn afterTransition(before: *const Battle, after: *const Battle, c1: pkmn.Choice, c2: pkmn.Choice) Knowledge {
        var k = Knowledge{};
        const choices = [2]pkmn.Choice{ c1, c2 };
        for (0..2) |side| {
            const pre_slot = before.sides[side].order[0] - 1;
            k.seen[side][pre_slot] = true;
            const choice = choices[side];
            if (choice.type == .Move and choice.data > 0 and choice.data <= 4) {
                const old_move = before.sides[side].active.moves[choice.data - 1];
                const chosen = old_move.id;
                const new_pp = after.sides[side].pokemon[pre_slot].moves[choice.data - 1].pp;
                // (repo copy) PP goes down only when the move runs (Gen 1 has no other PP loss), and a move that ran is in the log;
                // the engine's last_used_move is no test, as it is cleared at every faint, so a move that knocked out its target
                // was never revealed before
                if (chosen != .None and new_pp < old_move.pp) {
                    k.revealMove(side, pre_slot, chosen);
                    k.used[side][pre_slot][0] = old_move.pp - new_pp;
                }
            }
            const post_slot = after.sides[side].order[0] - 1;
            k.seen[side][post_slot] = true;
        }
        return k;
    }

    /// Joins two knowledge values from the same battle lineage (what either one has seen).
    pub fn merge(a: Knowledge, b: Knowledge) Knowledge {
        var out = a;
        for (0..2) |side| {
            for (0..6) |slot| {
                out.seen[side][slot] = a.seen[side][slot] or b.seen[side][slot];
                for (b.moves[side][slot], 0..) |m, j| {
                    if (m == .None) continue;
                    out.revealMove(side, slot, m);
                    for (out.moves[side][slot], 0..) |existing, i| {
                        if (existing == m) out.used[side][slot][i] +|= b.used[side][slot][j];
                    }
                }
            }
        }
        return out;
    }
};

fn statusName(status: u8, hp: u16) []const u8 {
    if (hp == 0) return "fnt";
    if (status == 0) return "none";
    if (pkmn.gen1.Status.is(status, .SLP)) return "sleep";
    if (pkmn.gen1.Status.is(status, .PSN)) return "poison";
    if (pkmn.gen1.Status.is(status, .BRN)) return "burn";
    if (pkmn.gen1.Status.is(status, .FRZ)) return "freeze";
    if (pkmn.gen1.Status.is(status, .PAR)) return "paralyze";
    return "none";
}

fn lowerTag(allocator: std.mem.Allocator, tag: []const u8) ![]const u8 {
    return vs.norm(allocator, tag);
}

fn typesList(allocator: std.mem.Allocator, t: pkmn.gen1.Types) !vs.V {
    var items: std.ArrayList(vs.V) = .empty;
    for ([2]pkmn.gen1.Type{ t.type1, t.type2 }) |ty| {
        const name = try lowerTag(allocator, @tagName(ty));
        if (std.mem.eql(u8, name, "typeless") or std.mem.eql(u8, name, "none")) continue;
        try items.append(allocator, .{ .str = name });
    }
    return .{ .list = items.items };
}

fn publicHpPct(hp: u16, maximum: u16) f64 {
    if (hp == 0) return 0;
    // (repo copy) Pokémon Showdown's HP Percentage Mod: rounded up, and 99 for anything below full (was rounded down, so
    // 1 HP read as 0%)
    const top = @max(@as(u32, maximum), 1);
    var pct = (100 * @as(u32, hp) + top - 1) / top;
    if (pct >= 100 and hp < maximum) pct = 99;
    return @min(@as(f64, @floatFromInt(pct)) / 100.0, 1.0);
}

/// The trainer's name for an engine volatile, where the effect logic has one.
fn volatileName(field: []const u8) ?[]const u8 {
    const map = [_][2][]const u8{
        .{ "Bide", "bide" },
        .{ "Confusion", "confusion" },
        .{ "FocusEnergy", "focusenergy" },
        .{ "LeechSeed", "leechseed" },
        .{ "LightScreen", "lightscreen" },
        .{ "Mist", "mist" },
        .{ "Rage", "rage" },
        .{ "Reflect", "reflect" },
        .{ "Substitute", "substitute" },
        .{ "Transform", "typechange" },
        .{ "Recharging", "mustrecharge" },
        .{ "Thrashing", "lockedmove" },
        .{ "Binding", "partiallytrapped" },
    };
    for (map) |pair| if (std.mem.eql(u8, pair[0], field)) return pair[1];
    return null;
}

/// The active Pokémon's volatile statuses, by the trainer's names (only those the effect logic reads).
fn volatileList(allocator: std.mem.Allocator, volatiles: anytype) !vs.V {
    var out: std.ArrayList(vs.V) = .empty;
    inline for (@typeInfo(@TypeOf(volatiles)).@"struct".fields) |f| {
        if (f.type == bool and @field(volatiles, f.name)) {
            if (volatileName(f.name)) |name| try out.append(allocator, .{ .str = name });
        }
    }
    return .{ .list = out.items };
}

/// The opponent's public volatile statuses (each one is announced in the battle log), by the replay parser's names.
/// Kept apart from the "volatiles" list that the existing snapshot readers parse, so that their inputs do not change.
/// "binding" means this Pokémon is using a trapping move (the engine puts Binding on the user, not on the target).
fn publicVolatileList(allocator: std.mem.Allocator, volatiles: anytype) !vs.V {
    const map = [_][2][]const u8{
        .{ "Bide", "bide" },
        .{ "Confusion", "confusion" },
        .{ "FocusEnergy", "focusenergy" },
        .{ "LeechSeed", "leechseed" },
        .{ "LightScreen", "lightscreen" },
        .{ "Mist", "mist" },
        .{ "Rage", "rage" },
        .{ "Reflect", "reflect" },
        .{ "Substitute", "substitute" },
        .{ "Transform", "transform" },
        .{ "Recharging", "mustrecharge" },
        .{ "Thrashing", "lockedmove" },
        .{ "Binding", "binding" },
    };
    var out: std.ArrayList(vs.V) = .empty;
    inline for (@typeInfo(@TypeOf(volatiles)).@"struct".fields) |f| {
        if (f.type == bool and @field(volatiles, f.name)) {
            for (map) |pair| if (std.mem.eql(u8, pair[0], f.name)) try out.append(allocator, .{ .str = pair[1] });
        }
    }
    return .{ .list = out.items };
}

/// Observer snapshot (player 1) built from the battle and the observer's knowledge. Own
/// information is complete; the opponent contributes only public fields (percent HP, status,
/// types, boosts, species seen, and revealed moves).
pub fn observerSnapshot(allocator: std.mem.Allocator, battle: *const Battle, k: Knowledge, force_switch: bool) !vs.V {
    const own = battle.sides[0];
    const foe = battle.sides[1];
    const own_active = own.order[0] - 1;
    const foe_active = foe.order[0] - 1;

    // own roster, active first
    var own_list: std.ArrayList(vs.V) = .empty;
    var order: [6]usize = undefined;
    order[0] = own_active;
    var n: usize = 1;
    for (0..6) |i| if (i != own_active) {
        order[n] = i;
        n += 1;
    };
    for (order) |slot| {
        const mon = own.pokemon[slot];
        var moves: std.ArrayList(vs.V) = .empty;
        for (mon.moves) |mv| {
            if (mv.id == .None) continue;
            const name = try lowerTag(allocator, @tagName(mv.id));
            const fields = try allocator.alloc(vs.Field, 3);
            fields[0] = .{ .key = "id", .value = .{ .str = name } };
            fields[1] = .{ .key = "pp", .value = .{ .int = mv.pp } };
            fields[2] = .{ .key = "max_pp", .value = .{ .int = @divTrunc(@as(i64, Move.pp(mv.id)), 5) * 8 } };
            try moves.append(allocator, .{ .obj = fields });
        }
        const fields = try allocator.alloc(vs.Field, 7);
        const species = try lowerTag(allocator, @tagName(mon.species));
        fields[0] = .{ .key = "id", .value = .{ .str = species } };
        fields[1] = .{ .key = "base_species", .value = .{ .str = species } };
        fields[2] = .{ .key = "hp", .value = .{ .int = mon.hp } };
        fields[3] = .{ .key = "maxhp", .value = .{ .int = mon.stats.hp } };
        fields[4] = .{ .key = "status", .value = .{ .str = statusName(mon.status, mon.hp) } };
        fields[5] = .{ .key = "types", .value = try typesList(allocator, mon.types) };
        fields[6] = .{ .key = "moves", .value = .{ .list = moves.items } };
        try own_list.append(allocator, .{ .obj = fields });
    }

    const boosts = try boostsObj(allocator, own.active.boosts);
    const user_fields = try allocator.alloc(vs.Field, 6);
    user_fields[5] = .{ .key = "public_volatiles", .value = try publicVolatileList(allocator, own.active.volatiles) };
    user_fields[0] = .{ .key = "pokemon", .value = .{ .list = own_list.items } };
    user_fields[1] = .{ .key = "boosts", .value = boosts };
    user_fields[2] = .{ .key = "volatiles", .value = try volatileList(allocator, own.active.volatiles) };
    // screens are public, so the observer's own conditions come straight from the active volatiles
    const screens = try allocator.alloc(vs.Field, 3);
    screens[0] = .{ .key = "reflect", .value = .{ .int = @intFromBool(own.active.volatiles.Reflect) } };
    screens[1] = .{ .key = "lightscreen", .value = .{ .int = @intFromBool(own.active.volatiles.LightScreen) } };
    screens[2] = .{ .key = "mist", .value = .{ .int = @intFromBool(own.active.volatiles.Mist) } };
    user_fields[3] = .{ .key = "conditions", .value = .{ .obj = screens } };
    user_fields[4] = .{ .key = "last_move", .value = .{ .str = try moveName(allocator, own.last_used_move) } };

    // opponent: public fields only
    const foe_mon = foe.pokemon[foe_active];
    const foe_species = try lowerTag(allocator, @tagName(foe_mon.species));
    var revealed: std.ArrayList(vs.V) = .empty;
    for (0..6) |slot| {
        if (!k.seen[1][slot] or slot == foe_active) continue;
        try revealed.append(allocator, try revealedEntry(allocator, foe.pokemon[slot], k.moves[1][slot], k.used[1][slot]));
    }
    try revealed.append(allocator, try revealedEntry(allocator, foe_mon, k.moves[1][foe_active], k.used[1][foe_active]));

    const foe_boosts = try boostsObj(allocator, foe.active.boosts);
    const active_fields = try allocator.alloc(vs.Field, 5);
    active_fields[0] = .{ .key = "base_species", .value = .{ .str = foe_species } };
    active_fields[1] = .{ .key = "hp", .value = .{ .float = publicHpPct(foe_mon.hp, foe_mon.stats.hp) } };
    active_fields[2] = .{ .key = "status", .value = .{ .str = statusName(foe_mon.status, foe_mon.hp) } };
    active_fields[3] = .{ .key = "types", .value = try typesList(allocator, foe_mon.types) };
    active_fields[4] = .{ .key = "boosts", .value = foe_boosts };

    const opp_fields = try allocator.alloc(vs.Field, 5);
    opp_fields[0] = .{ .key = "active_species", .value = .{ .str = foe_species } };
    opp_fields[1] = .{ .key = "active", .value = .{ .obj = active_fields } };
    opp_fields[2] = .{ .key = "revealed", .value = .{ .list = revealed.items } };
    opp_fields[3] = .{ .key = "last_move", .value = .{ .str = try moveName(allocator, foe.last_used_move) } };
    opp_fields[4] = .{ .key = "public_volatiles", .value = try publicVolatileList(allocator, foe.active.volatiles) };

    const snap = try allocator.alloc(vs.Field, 6);
    snap[0] = .{ .key = "turn", .value = .{ .int = battle.turn } };
    // a fainted active Pokémon must be replaced; the caller's flag is only a hint
    snap[1] = .{ .key = "force_switch", .value = .{ .bool = force_switch or own.pokemon[own_active].hp == 0 } };
    snap[2] = .{ .key = "opponent_force_switch", .value = .{ .bool = foe.pokemon[foe_active].hp == 0 } };
    snap[3] = .{ .key = "user", .value = .{ .obj = user_fields } };
    snap[4] = .{ .key = "opponent", .value = .{ .obj = opp_fields } };
    // (repo copy) the battle's lastDamage, Gen 1 Counter's memory (public: a replay parser follows it from the log)
    snap[5] = .{ .key = "last_damage", .value = .{ .int = battle.last_damage } };
    return .{ .obj = snap[0..6] };
}

fn moveName(allocator: std.mem.Allocator, m: Move) ![]const u8 {
    if (m == .None or m == .SKIP_TURN) return "none";
    return lowerTag(allocator, @tagName(m));
}

fn revealedEntry(allocator: std.mem.Allocator, mon: pkmn.gen1.Pokemon, known: [MAX_MOVES]Move, used: [MAX_MOVES]u8) !vs.V {
    var moves: std.ArrayList(vs.V) = .empty;
    for (known, used) |m, n| {
        if (m == .None) continue;
        // (repo copy) PP from the public use count, as a replay parser tracks it (the maximum is the own side's: PP Ups applied)
        const max_pp: i64 = @divTrunc(@as(i64, Move.pp(m)), 5) * 8;
        const fields = try allocator.alloc(vs.Field, 3);
        fields[0] = .{ .key = "id", .value = .{ .str = try lowerTag(allocator, @tagName(m)) } };
        fields[1] = .{ .key = "pp", .value = .{ .int = @max(0, max_pp - @as(i64, n)) } };
        fields[2] = .{ .key = "max_pp", .value = .{ .int = max_pp } };
        try moves.append(allocator, .{ .obj = fields });
    }
    const fields = try allocator.alloc(vs.Field, 4);
    fields[0] = .{ .key = "species", .value = .{ .str = try lowerTag(allocator, @tagName(mon.species)) } };
    fields[1] = .{ .key = "hp", .value = .{ .float = publicHpPct(mon.hp, mon.stats.hp) } };
    fields[2] = .{ .key = "status", .value = .{ .str = statusName(mon.status, mon.hp) } };
    fields[3] = .{ .key = "moves", .value = .{ .list = moves.items } };
    return .{ .obj = fields };
}

fn boostsObj(allocator: std.mem.Allocator, b: pkmn.gen1.Boosts) !vs.V {
    const fields = try allocator.alloc(vs.Field, 6);
    fields[0] = .{ .key = "atk", .value = .{ .int = b.atk } };
    fields[1] = .{ .key = "def", .value = .{ .int = b.def } };
    fields[2] = .{ .key = "spc", .value = .{ .int = b.spc } };
    fields[3] = .{ .key = "spe", .value = .{ .int = b.spe } };
    fields[4] = .{ .key = "accuracy", .value = .{ .int = b.accuracy } };
    fields[5] = .{ .key = "evasion", .value = .{ .int = b.evasion } };
    return .{ .obj = fields };
}

/// The same knowledge seen from the other side (P1's and P2's entries swapped).
pub fn swapKnowledge(k: Knowledge) Knowledge {
    return .{ .seen = .{ k.seen[1], k.seen[0] }, .moves = .{ k.moves[1], k.moves[0] }, .used = .{ k.used[1], k.used[0] } };
}

/// A copy of `battle` with the two sides exchanged, so that `observerSnapshot` reads it from P2's
/// side.
pub fn swapSides(battle: *const Battle) Battle {
    var out = battle.*;
    std.mem.swap(@TypeOf(out.sides[0]), &out.sides[0], &out.sides[1]);
    return out;
}

/// Observer snapshot of `battle` from `side` (0 = P1, 1 = P2), given knowledge in that side's
/// orientation.
pub fn perspectiveSnapshot(allocator: std.mem.Allocator, battle: *const Battle, k: Knowledge, side: usize) !vs.V {
    if (side == 0) return observerSnapshot(allocator, battle, k, false);
    const swapped = swapSides(battle);
    return observerSnapshot(allocator, &swapped, k, false);
}

/// Index of `choice` in the epoch-20 actor's nine action slots, from `side`'s observer (the same
/// rule as bot2's policy server: moves are indexed by their rank in sorted normalized name order,
/// switches by 4 + their rank among the living reserves, and no-ops by 0). Null when the choice
/// has no slot.
pub fn actorIndex(battle: *const Battle, side: usize, choice: pkmn.Choice) ?usize {
    if (choice.type == .Pass) return 0;
    const active = battle.sides[side].order[0] - 1;
    if (choice.type == .Move) {
        if (choice.data == 0) return 0;
        if (choice.data > 4) return null;
        const mine = battle.sides[side].pokemon[active].moves[choice.data - 1].id;
        if (mine == .None) return null;
        var buf: [16]u8 = undefined;
        const my_name = normName(&buf, @tagName(mine));
        var rank: usize = 0;
        for (battle.sides[side].pokemon[active].moves) |m| {
            if (m.id == .None) continue;
            var other_buf: [16]u8 = undefined;
            const other = normName(&other_buf, @tagName(m.id));
            if (std.mem.lessThan(u8, other, my_name)) rank += 1;
        }
        return rank;
    }
    if (choice.type == .Switch) {
        const slot = battle.sides[side].order[choice.data - 1] - 1;
        const my_name = speciesName(battle.sides[side].pokemon[slot].species);
        var rank: usize = 0;
        for (battle.sides[side].pokemon, 0..) |pk, i| {
            if (i == active or pk.hp == 0 or pk.species == .None) continue;
            if (std.mem.lessThan(u8, speciesName(pk.species), my_name)) rank += 1;
        }
        return 4 + rank;
    }
    return null;
}

/// Lowercase alphanumeric form of an engine tag, written into `buf`.
fn normName(buf: []u8, tag: []const u8) []const u8 {
    var n: usize = 0;
    for (tag) |c| {
        const lc = std.ascii.toLower(c);
        if ((lc >= 'a' and lc <= 'z') or (lc >= '0' and lc <= '9')) {
            buf[n] = lc;
            n += 1;
        }
    }
    return buf[0..n];
}

var species_names: [256][16]u8 = undefined;
var species_lens: [256]usize = @splat(0);
var species_ready = false;

/// Normalized species name, cached per species value.
/// Fills speciesName's table now (before the search starts threads: it is filled lazily otherwise).
pub fn warmNames() void {
    _ = speciesName(.None);
}

fn speciesName(species: pkmn.gen1.Species) []const u8 {
    const i: usize = @intFromEnum(species);
    if (!species_ready) {
        species_ready = true;
        inline for (@typeInfo(pkmn.gen1.Species).@"enum".fields) |field| {
            var buf: [16]u8 = undefined;
            const n = normName(&buf, field.name);
            @memcpy(species_names[field.value][0..n.len], n);
            species_lens[field.value] = n.len;
        }
    }
    return species_names[i][0..species_lens[i]];
}

/// Knowledge at the start of a battle, after the lead switch-in (the two leads are public).
pub fn initial(battle: *const Battle) Knowledge {
    var k = Knowledge{};
    for (0..2) |side| k.seen[side][battle.sides[side].order[0] - 1] = true;
    return k;
}
