//! Move completion for unknown move slots, after foul-play's `sample_pokemon_moveset_with_known_pkmn_set`:
//! sample moves from Smogon usage rates, with each rate adjusted so that the chance of including a
//! move over the missing slots matches its usage. The "logical sense" filter of foul-play is not
//! ported; the engine enforces its own rules.

const std = @import("std");
const pkmn = @import("pkmn");
const belief = @import("belief.zig");

pub const Move = pkmn.gen1.Move;
const MAX_ITERATIONS = 10_000;

pub const Entry = struct { move: Move, rate: f64 };

pub const Legal = struct {
    species: std.json.ObjectMap,

    pub fn load(allocator: std.mem.Allocator, io: std.Io) !Legal {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, "models/completion/gen1ou_legal.json", allocator, .unlimited);
        return .{ .species = (try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})).object };
    }

    pub fn allows(self: *const Legal, species: []const u8, move: []const u8) bool {
        const row = self.species.get(species) orelse return false;
        for (row.array.items) |entry| if (std.mem.eql(u8, entry.string, move)) return true;
        return false;
    }
};

pub const Table = struct {
    allocator: std.mem.Allocator,
    /// Species name (lowercase alphanumeric) -> usage entries.
    by_species: std.StringHashMap([]Entry),

    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Table {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        const legal = try Legal.load(allocator, io);
        var table = Table{ .allocator = allocator, .by_species = std.StringHashMap([]Entry).init(allocator) };
        var lists = std.StringHashMap(std.ArrayList(Entry)).init(allocator);
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            var parts = std.mem.splitScalar(u8, line, '\t');
            const species = try normalize(allocator, parts.next() orelse return error.BadTable);
            const move_name = try normalize(allocator, parts.next() orelse return error.BadTable);
            const rate = try std.fmt.parseFloat(f64, parts.next() orelse return error.BadTable);
            if (!legal.allows(species, move_name)) continue;
            const move = moveFromName(move_name) orelse continue;
            const gop = try lists.getOrPut(species);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(allocator, .{ .move = move, .rate = rate });
        }
        var it = lists.iterator();
        while (it.next()) |entry| {
            try table.by_species.put(entry.key_ptr.*, try entry.value_ptr.toOwnedSlice(allocator));
        }
        return table;
    }

    /// Fills the unknown slots of a Pokémon whose `known` moves are given. Returns the moves
    /// (known first, then sampled), at most four.
    pub fn complete(self: *const Table, species: []const u8, known: []const Move, rng: *belief.Rng) ![]Move {
        var out: std.ArrayList(Move) = .empty;
        for (known) |m| {
            if (m == .None or m == .SKIP_TURN or std.mem.indexOfScalar(Move, out.items, m) != null) continue;
            if (out.items.len < 4) try out.append(self.allocator, m);
        }
        const missing = 4 - out.items.len;
        if (missing == 0) return out.items;
        const all = self.by_species.get(species) orelse return out.items;

        // candidates: usage moves not already known, with the per-slot adjusted probability
        var cands: std.ArrayList(Entry) = .empty;
        for (all) |e| {
            if (std.mem.indexOfScalar(Move, out.items, e.move) != null) continue;
            const p = 1.0 - std.math.pow(f64, 1.0 - e.rate, 1.0 / @as(f64, @floatFromInt(missing)));
            try cands.append(self.allocator, .{ .move = e.move, .rate = p });
        }
        var index: usize = 0;
        var iterations: usize = 0;
        while (out.items.len < 4 and cands.items.len > 0 and iterations < MAX_ITERATIONS) : (iterations += 1) {
            index = index % cands.items.len;
            const cand = cands.items[index];
            if (rng.nextF64() < cand.rate) {
                try out.append(self.allocator, cand.move);
                _ = cands.orderedRemove(index);
            } else {
                index += 1;
            }
        }
        return out.items;
    }
};

/// Lowercase alphanumeric form of a name.
pub fn normalize(allocator: std.mem.Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s) |c| {
        const lc = std.ascii.toLower(c);
        if ((lc >= 'a' and lc <= 'z') or (lc >= '0' and lc <= '9')) try out.append(allocator, lc);
    }
    return out.items;
}

/// Case- and punctuation-insensitive comparison of an engine tag with a normalized name.
pub fn eqlNormalized(tag: []const u8, normalized: []const u8) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (true) {
        while (i < tag.len and !std.ascii.isAlphanumeric(tag[i])) i += 1;
        while (j < normalized.len and !std.ascii.isAlphanumeric(normalized[j])) j += 1;
        if (i == tag.len or j == normalized.len) return i == tag.len and j == normalized.len;
        if (std.ascii.toLower(tag[i]) != std.ascii.toLower(normalized[j])) return false;
        i += 1;
        j += 1;
    }
}

/// The engine's move with this normalized name, if any.
pub fn moveFromName(name: []const u8) ?Move {
    inline for (@typeInfo(Move).@"enum".fields) |field| {
        if (field.value != 0 and eqlNormalized(field.name, name)) {
            return @enumFromInt(field.value);
        }
    }
    return null;
}
