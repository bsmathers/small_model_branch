//! Expert team pool: Showdown-format team files (bot2's `metamon-cache/teams/b1_good_807/gen1ou`,
//! 807 files). Each file lists six Pokémon with their moves. (repo copy) EVs and IVs are read as Pokémon Showdown reads
//! them in Gen 1: an IV is twice the DV (DV = IV / 2, unlisted IVs 31), an EV e is stat experience e * e (the engine takes
//! ceil(sqrt(exp))), unlisted EVs are 0 when an EVs line is present and the maximum otherwise; Special comes from SpA. (The
//! original gave every Pokémon maximum DVs and stat experience: the pool's 38 minimum-Attack sets were played at full Attack.)

const std = @import("std");
const pkmn = @import("pkmn");
const belief = @import("belief.zig");
const completion = @import("completion.zig");

pub const Pokemon = pkmn.gen1.helpers.Pokemon;

pub const Team = struct {
    name: []const u8,
    mons: []Pokemon,
};

const MAX_DVS: pkmn.gen1.DVs = .{ .atk = 15, .def = 15, .spe = 15, .spc = 15 };

pub const LoadStats = struct {
    files: usize = 0,
    loaded: usize = 0,
    skipped: usize = 0,
};

/// Loads every team file in `dir` whose six Pokémon and their moves are all known to the engine.
pub fn loadDir(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, stats: *LoadStats) ![]Team {
    var out: std.ArrayList(Team) = .empty;
    var root = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer root.close(io);
    var it = root.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        stats.files += 1;
        const text = try root.readFileAlloc(io, entry.name, allocator, .unlimited);
        if (try parseTeam(allocator, text)) |mons| {
            try out.append(allocator, .{ .name = try allocator.dupe(u8, entry.name), .mons = mons });
            stats.loaded += 1;
        } else {
            stats.skipped += 1;
        }
    }
    // a stable order, so that a seed picks the same teams on every run
    std.mem.sort(Team, out.items, {}, struct {
        fn lessThan(_: void, a: Team, b: Team) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);
    return out.items;
}

/// Parses one Showdown team file; null when it does not describe six usable Pokémon.
fn parseTeam(allocator: std.mem.Allocator, text: []const u8) !?[]Pokemon {
    var mons: std.ArrayList(Pokemon) = .empty;
    var species_name: ?[]const u8 = null;
    var moves: std.ArrayList(pkmn.gen1.Move) = .empty;
    var spread = Spread{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \r\t");
        if (line.len == 0) {
            if (try finish(allocator, &mons, species_name, moves.items, spread)) |_| {}
            species_name = null;
            moves.clearRetainingCapacity();
            spread = .{};
            continue;
        }
        if (std.mem.startsWith(u8, line, "- ")) {
            const move = completion.moveFromName(try completion.normalize(allocator, line[2..])) orelse return null;
            if (moves.items.len < 4) try moves.append(allocator, move);
        } else if (std.mem.startsWith(u8, line, "EVs:")) {
            spread.readEvs(line[4..]);
        } else if (std.mem.startsWith(u8, line, "IVs:")) {
            spread.readIvs(line[4..]);
        } else if (std.mem.indexOfScalar(u8, line, ':') != null) {
            continue; // Ability, Item, Level, ...
        } else if (species_name == null) {
            // "Chansey (F)", "Tauros (M)", or a nickname "KINETICS (Chansey) (F)": drop the gender tag, then the species
            // is in the last parentheses if any, else the whole name
            var name = line;
            if (std.mem.endsWith(u8, name, " (M)") or std.mem.endsWith(u8, name, " (F)")) name = name[0 .. name.len - 4];
            if (name.len > 0 and name[name.len - 1] == ')') {
                if (std.mem.lastIndexOf(u8, name, " (")) |i| name = name[i + 2 .. name.len - 1];
            }
            species_name = name;
        }
    }
    if (try finish(allocator, &mons, species_name, moves.items, spread)) |_| {}
    if (mons.items.len != 6) return null;
    return mons.items;
}

/// Adds the Pokémon in progress; returns a marker when it was added.
const Exp = @FieldType(Pokemon, "stats");
const MAX_EXP: u16 = 0xFFFF;

/// One Pokémon's DVs and stat experience, from its EVs/IVs lines.
const Spread = struct {
    dvs: pkmn.gen1.DVs = MAX_DVS,
    exp: Exp = .{ .hp = MAX_EXP, .atk = MAX_EXP, .def = MAX_EXP, .spe = MAX_EXP, .spc = MAX_EXP },

    /// "252 HP / 252 Def / ...": listed stats get e * e, the others 0.
    fn readEvs(self: *Spread, text: []const u8) void {
        self.exp = .{ .hp = 0, .atk = 0, .def = 0, .spe = 0, .spc = 0 };
        var parts = std.mem.splitScalar(u8, text, '/');
        var spc_set = false;
        while (parts.next()) |part| {
            const item = parse(part) orelse continue;
            const e: u16 = @intCast(@min(@as(u32, item.value) * item.value, MAX_EXP));
            if (std.mem.eql(u8, item.stat, "HP")) self.exp.hp = e;
            if (std.mem.eql(u8, item.stat, "Atk")) self.exp.atk = e;
            if (std.mem.eql(u8, item.stat, "Def")) self.exp.def = e;
            if (std.mem.eql(u8, item.stat, "Spe")) self.exp.spe = e;
            if (std.mem.eql(u8, item.stat, "SpA")) {
                self.exp.spc = e;
                spc_set = true;
            }
            if (std.mem.eql(u8, item.stat, "SpD") and !spc_set) self.exp.spc = e;
        }
    }

    /// "2 Atk": listed stats get DV = IV / 2 (unlisted keep 15).
    fn readIvs(self: *Spread, text: []const u8) void {
        var parts = std.mem.splitScalar(u8, text, '/');
        var spc_set = false;
        while (parts.next()) |part| {
            const item = parse(part) orelse continue;
            const dv: u4 = @intCast(@min(item.value / 2, 15));
            if (std.mem.eql(u8, item.stat, "Atk")) self.dvs.atk = dv;
            if (std.mem.eql(u8, item.stat, "Def")) self.dvs.def = dv;
            if (std.mem.eql(u8, item.stat, "Spe")) self.dvs.spe = dv;
            if (std.mem.eql(u8, item.stat, "SpA")) {
                self.dvs.spc = dv;
                spc_set = true;
            }
            if (std.mem.eql(u8, item.stat, "SpD") and !spc_set) self.dvs.spc = dv;
        }
    }

    fn parse(part: []const u8) ?struct { value: u16, stat: []const u8 } {
        const t = std.mem.trim(u8, part, " \t\r");
        const space = std.mem.indexOfScalar(u8, t, ' ') orelse return null;
        const value = std.fmt.parseUnsigned(u16, t[0..space], 10) catch return null;
        return .{ .value = value, .stat = std.mem.trim(u8, t[space + 1 ..], " ") };
    }
};

fn finish(allocator: std.mem.Allocator, mons: *std.ArrayList(Pokemon), species_name: ?[]const u8, moves: []const pkmn.gen1.Move, spread: Spread) !?void {
    const name = species_name orelse return null;
    if (moves.len == 0) return null;
    const species = belief.gen1Species(try completion.normalize(allocator, name)) orelse return null;
    const owned = try allocator.dupe(pkmn.gen1.Move, moves);
    try mons.append(allocator, .{ .species = species, .moves = owned, .dvs = spread.dvs, .stats = spread.exp });
    return {};
}
