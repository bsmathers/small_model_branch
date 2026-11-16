//! Differential damage harness for the Zig engine. For each move, runs seeded one-turn battles
//! between two fixed Pokémon (max DVs, as in the Showdown harness), records the damage dealt to
//! the defender, and skips critical hits, misses and battles with no damage. Output: one line per
//! matchup and move: `matchup move min max samples`.

const std = @import("std");
const pkmn = @import("pkmn");

const seeds_per_move = 400;
const CRIT_TAG: u8 = 26;

fn matchupTeams(attacker: pkmn.gen1.Species, defender: pkmn.gen1.Species, a_moves: []const pkmn.gen1.Move, d_moves: []const pkmn.gen1.Move) struct { a: pkmn.gen1.helpers.Pokemon, d: pkmn.gen1.helpers.Pokemon } {
    const max_dvs: pkmn.gen1.DVs = .{ .atk = 15, .def = 15, .spe = 15, .spc = 15 };
    return .{
        .a = .{ .species = attacker, .moves = a_moves, .dvs = max_dvs },
        .d = .{ .species = defender, .moves = d_moves, .dvs = max_dvs },
    };
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    const matchups = [_]struct { name: []const u8, a: pkmn.gen1.Species, d: pkmn.gen1.Species }{
        .{ .name = "tauros_to_chansey", .a = .Tauros, .d = .Chansey },
        .{ .name = "chansey_to_tauros", .a = .Chansey, .d = .Tauros },
        .{ .name = "starmie_to_exeggutor", .a = .Starmie, .d = .Exeggutor },
        .{ .name = "exeggutor_to_snorlax", .a = .Exeggutor, .d = .Snorlax },
        .{ .name = "snorlax_to_zapdos", .a = .Snorlax, .d = .Zapdos },
        .{ .name = "zapdos_to_starmie", .a = .Zapdos, .d = .Starmie },
        .{ .name = "alakazam_to_tauros", .a = .Alakazam, .d = .Tauros },
        .{ .name = "rhydon_to_zapdos", .a = .Rhydon, .d = .Zapdos },
        .{ .name = "jynx_to_snorlax", .a = .Jynx, .d = .Snorlax },
        .{ .name = "cloyster_to_alakazam", .a = .Cloyster, .d = .Alakazam },
        .{ .name = "gengar_to_chansey", .a = .Gengar, .d = .Chansey },
        .{ .name = "golem_to_starmie", .a = .Golem, .d = .Starmie },
    };
    const out = std.debug.print;
    for (matchups) |m| {
        inline for (@typeInfo(pkmn.gen1.Move).@"enum".fields) |field| {
            const move: pkmn.gen1.Move = @enumFromInt(field.value);
            if (move == .None or move == .Splash or move == .SKIP_TURN) continue;
            var lo: u32 = std.math.maxInt(u32);
            var hi: u32 = 0;
            var samples: u32 = 0;
            var max_hp: u32 = 0;
            var seed: u64 = 1;
            while (seed <= seeds_per_move) : (seed += 1) {
                                const a_moves = [_]pkmn.gen1.Move{ move, .Splash, .Splash, .Splash };
                const d_moves = [_]pkmn.gen1.Move{ .Splash, .Splash, .Splash, .Splash };
                const teams = matchupTeams(m.a, m.d, &a_moves, &d_moves);
                var battle = pkmn.gen1.helpers.Battle.init(seed, &.{teams.a}, &.{teams.d});
                const before = battle.sides[1].pokemon[0].hp;
                max_hp = battle.sides[1].pokemon[0].stats.hp;
                var buf: [pkmn.LOGS_SIZE]u8 = undefined;
                var writer: pkmn.protocol.Writer = .{ .buffer = &buf };
                var options = pkmn.battle.options(pkmn.protocol.FixedLog{ .writer = &writer }, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
                _ = battle.update(.{}, .{}, &options) catch continue;
                writer.reset();
                _ = battle.update(pkmn.Choice{ .type = .Move, .data = 1 }, pkmn.Choice{ .type = .Move, .data = 1 }, &options) catch continue;
                const logged2 = writer.buffer[0..writer.pos];
                // The protocol log is binary: each entry starts with its ArgType tag. The Crit tag is
                // the 27th ArgType in src/lib/common/protocol.zig (index 26).
                if (std.mem.indexOfScalar(u8, logged2, CRIT_TAG) != null) continue;
                const after = battle.sides[1].pokemon[0].hp;
                if (after >= before) continue;
                const d: u32 = before - after;
                lo = @min(lo, d);
                hi = @max(hi, d);
                samples += 1;
            }
            if (samples > 0) out("{s} {s} {d} {d} {d} {d}\n", .{ m.name, @tagName(move), lo, hi, samples, max_hp });
        }
    }
}
