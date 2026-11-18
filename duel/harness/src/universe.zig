//! Opponent universes for determinized search. A universe is a copy of the true battle in which
//! the opponent's unrevealed party slots hold belief-sampled Pokémon, and the move slots of revealed
//! Pokémon that the observer hasn't seen are filled by the completion table (foul-play's rule).
//! Revealed Pokémon keep their true species, HP, status and known moves.

const std = @import("std");
const pkmn = @import("pkmn");
const belief = @import("belief.zig");
const completion = @import("completion.zig");
const know = @import("knowledge.zig");

pub const Battle = know.Battle;
pub const Move = pkmn.gen1.Move;
const MAX_DVS: pkmn.gen1.DVs = .{ .atk = 15, .def = 15, .spe = 15, .spc = 15 };

/// A fresh opponent Pokémon (full HP, maximum DVs and stat experience) with the given moves.
fn newPokemon(allocator: std.mem.Allocator, species: pkmn.gen1.Species, moves: []const Move) !pkmn.gen1.Pokemon {
    const slots = try allocator.dupe(Move, moves);
    const side = pkmn.gen1.helpers.Side.init(&.{.{ .species = species, .moves = slots, .dvs = MAX_DVS }});
    return side.pokemon[0];
}

/// Builds one universe from the true battle. `sampled` is the belief model's team for this world,
/// given the observer's `knowledge` (its first entries are the revealed species).
pub fn build(
    allocator: std.mem.Allocator,
    truth: *const Battle,
    knowledge: know.Knowledge,
    sampled: []const belief.SampledPokemon,
    table: *const completion.Table,
    rng: *belief.Rng,
) !Battle {
    var world = truth.*;
    const foe = &world.sides[1];
    const foe_active = foe.order[0] - 1;

    // unrevealed slots take the sampled species that the observer hasn't seen
    var next: usize = 0;
    for (0..6) |slot| {
        if (knowledge.seen[1][slot]) continue;
        if (next >= sampled.len) break;
        const entry = sampled[next];
        next += 1;
        const species = belief.gen1Species(entry.species) orelse return error.UnknownSpecies;
        var moves: std.ArrayList(Move) = .empty;
        for (entry.moves) |name| {
            if (completion.moveFromName(try completion.normalize(allocator, name))) |m| try moves.append(allocator, m);
        }
        // the belief's moves come first; the usage table fills the rest (foul-play's rule)
        const completed = try table.complete(entry.species, moves.items, rng);
        moves.clearRetainingCapacity();
        try moves.appendSlice(allocator, completed);
        foe.pokemon[slot] = try newPokemon(allocator, species, moves.items);
    }

    // revealed Pokémon keep their true moves; their unknown slots are completed
    for (0..6) |slot| {
        if (!knowledge.seen[1][slot]) continue;
        const mon = &foe.pokemon[slot];
        if (mon.species == .None) continue;
        const name = try completion.normalize(allocator, @tagName(mon.species));
        const moves = try table.complete(name, knowledge.moves[1][slot][0..], rng);
        for (0..4) |j| {
            if (j < moves.len) {
                var used: u8 = 0;
                for (knowledge.moves[1][slot], knowledge.used[1][slot]) |known, count| {
                    if (known == moves[j]) used = count;
                }
                const maximum = @min(@as(u16, pkmn.gen1.Move.pp(moves[j])) * 8 / 5, 61);
                mon.moves[j] = .{ .id = moves[j], .pp = @intCast(maximum -| used) };
            } else {
                mon.moves[j] = .{};
            }
            if (slot == foe_active) foe.active.moves[j] = mon.moves[j];
        }
    }
    return world;
}

/// The observer's known opponent party, as the belief sampler takes it: seen species with the
/// moves revealed for them.
pub fn knownParty(allocator: std.mem.Allocator, truth: *const Battle, knowledge: know.Knowledge) ![]belief.SampledPokemon {
    var out: std.ArrayList(belief.SampledPokemon) = .empty;
    for (0..6) |slot| {
        if (!knowledge.seen[1][slot]) continue;
        const mon = truth.sides[1].pokemon[slot];
        var moves: std.ArrayList([]const u8) = .empty;
        for (knowledge.moves[1][slot]) |m| {
            if (m == .None) continue;
            try moves.append(allocator, try completion.normalize(allocator, @tagName(m)));
        }
        try out.append(allocator, .{ .species = try completion.normalize(allocator, @tagName(mon.species)), .moves = moves.items });
    }
    return out.items;
}

/// An earlier decision point `past` (the observer's knowledge then: `past_knowledge`) as it would have been in `world`, a
/// universe built at a later point with the observer's knowledge `knowledge_now`. The opponent's slots still unrevealed now
/// hold `world`'s sampled Pokémon (never seen, so never in play: full HP, no status, full PP); its revealed Pokémon keep their
/// real species, HP and status of that time with `world`'s movesets, at the PP left after the uses the observer knew of then.
/// The observer's side is the real one.
pub fn pastWorld(past: *const Battle, past_knowledge: know.Knowledge, world: *const Battle, knowledge_now: know.Knowledge) Battle {
    var out = past.*;
    const foe = &out.sides[1];
    const active: usize = if (foe.order[0] > 0) foe.order[0] - 1 else 6;
    for (0..6) |slot| {
        if (!knowledge_now.seen[1][slot]) {
            foe.pokemon[slot] = world.sides[1].pokemon[slot];
            continue;
        }
        const mon = &foe.pokemon[slot];
        if (mon.species == .None) continue;
        for (0..4) |j| {
            const id = world.sides[1].pokemon[slot].moves[j].id;
            if (id == .None) {
                mon.moves[j] = .{};
            } else {
                var used: u8 = 0;
                for (past_knowledge.moves[1][slot], past_knowledge.used[1][slot]) |known, count| {
                    if (known == id) used = count;
                }
                const maximum = @min(@as(u16, pkmn.gen1.Move.pp(id)) * 8 / 5, 61);
                mon.moves[j] = .{ .id = id, .pp = @intCast(maximum -| used) };
            }
            if (slot == active) foe.active.moves[j] = mon.moves[j];
        }
    }
    return out;
}
