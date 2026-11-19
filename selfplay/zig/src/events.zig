//! The pkmn engine's protocol log (Gen 1) as rows of small integer fields, from one player's side: the model's "battle log
//! reading" input. The byte layout follows the engine's own formatter (pkmn common/protocol.zig `format`) event by event.
//!
//! A row: [kind, actor side, actor species, target side, target species, move, aux move, status, reason, hp, num, flags].
//!   kind     the event's ArgType (0 = no row)
//!   side     0 none, 1 the viewer's, 2 the opponent's
//!   species  the Pokemon behind a side and slot identifier (the stored, original species; 0 none)
//!   move     the event's move (Move, Prepare, Cant from Disable, Start of Disable/Mimic...), aux move a second one (Move's
//!            "[from]" move, Status/Activate's source move)
//!   status   the status byte as a category: 0 none, 1 sleep, 2 poison, 3 burn, 4 freeze, 5 paralysis, 6 bad poison
//!   reason   the event's reason enum + 1 (0 none)
//!   hp       HP as Showdown shows it: the percentage, rounded up, 99 below full, 0 fainted (both sides: the exact own HP is
//!            in the board input)
//!   num      a number: Turn's turn (at most 255), Switch's level, Boost's amount + 128, HitCount, Magnitude, Win's side
//!   flags    0 (reserved)
//! Nothing here is private: the log is what Showdown shows a spectator, with the opponent's HP as a percentage.
const std = @import("std");
const builtin = @import("builtin");
const pkmn = @import("pkmn");

const P = @import("protocol_enums.zig");
const Battle = pkmn.gen1.Battle(pkmn.gen1.PRNG);
const endian = builtin.cpu.arch.endian();

pub const FIELDS = 12;
pub const Row = [FIELDS]u8;
pub const F = struct {
    pub const kind = 0;
    pub const actor_side = 1;
    pub const actor_species = 2;
    pub const target_side = 3;
    pub const target_species = 4;
    pub const move = 5;
    pub const aux_move = 6;
    pub const status = 7;
    pub const reason = 8;
    pub const hp = 9;
    pub const num = 10;
    pub const flags = 11;
};

pub const DecodeError = error{ UnknownEvent, Truncated };

/// The status byte as a category (Status.TOX is bad poison; Status.EXT alone, Rest's sleep, is sleep).
pub fn statusCategory(s: u8) u8 {
    const St = pkmn.gen1.Status;
    if (s == 0) return 0;
    if (s == St.TOX) return 6;
    if (St.is(s, .SLP) or St.is(s, .EXT)) return 1;
    if (St.is(s, .PSN)) return 2;
    if (St.is(s, .BRN)) return 3;
    if (St.is(s, .FRZ)) return 4;
    if (St.is(s, .PAR)) return 5;
    return 0;
}

/// Showdown's public HP percentage: rounded up, 99 for anything below full, 0 when fainted.
pub fn publicPct(hp: u16, max: u16) u8 {
    if (hp == 0) return 0;
    const top: u32 = @max(max, 1);
    var pct: u32 = (100 * @as(u32, hp) + top - 1) / top;
    if (pct >= 100 and hp < max) pct = 99;
    return @intCast(@min(pct, 100));
}

const Reader = struct {
    a: []const u8,
    i: usize = 0,
    fn byte(r: *Reader) DecodeError!u8 {
        if (r.i >= r.a.len) return error.Truncated;
        r.i += 1;
        return r.a[r.i - 1];
    }
    fn u16v(r: *Reader) DecodeError!u16 {
        if (r.i + 2 > r.a.len) return error.Truncated;
        const v = std.mem.readInt(u16, r.a[r.i..][0..2], endian);
        r.i += 2;
        return v;
    }
};

/// Fills `side` / `species` fields from an identifier byte, from `viewer`'s side.
fn ident(battle: *const Battle, viewer: usize, raw: u8, row: *Row, side_f: usize, species_f: usize) void {
    // (pkmn's ID: packed struct(u8) { id: u3 (the one-indexed team slot), player: u1, _: u4 })
    const slot: usize = raw & 0b111;
    if (slot == 0 or slot > 6) return;
    const player: usize = (raw >> 3) & 1;
    row[side_f] = if (player == viewer) 1 else 2;
    row[species_f] = @intFromEnum(battle.sides[player].pokemon[slot - 1].species);
}

/// Decodes `log` (one update's bytes) into rows for `viewer` (0 = P1), appended to `out` from `n_out`; rows past `out.len`
/// are counted in the return's `dropped` and not stored. `battle` is the state after the update (it resolves identifiers).
pub fn decode(log: []const u8, battle: *const Battle, viewer: usize, out: []Row, n_out: *usize) DecodeError!usize {
    var r = Reader{ .a = log };
    var dropped: usize = 0;
    while (r.i < r.a.len) {
        const arg: P.ArgType = @enumFromInt(try r.byte());
        if (arg == .None) continue; // (end of an update's block)
        var row: Row = @splat(0);
        row[F.kind] = @intFromEnum(arg);
        switch (arg) {
            .LastStill, .LastMiss, .Tie, .ClearAllBoost, .FieldActivate, .OHKO => {},
            .Move => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                row[F.move] = try r.byte();
                ident(battle, viewer, try r.byte(), &row, F.target_side, F.target_species);
                const reason = try r.byte();
                row[F.reason] = reason +| 1;
                if (reason == @intFromEnum(P.Move.From)) row[F.aux_move] = try r.byte();
            },
            .Switch => {
                const raw = try r.byte();
                ident(battle, viewer, raw, &row, F.actor_side, F.actor_species);
                row[F.actor_species] = try r.byte(); // (the species switched in, as the log names it)
                row[F.num] = try r.byte(); // level
                const hp = try r.u16v();
                const max = try r.u16v();
                row[F.hp] = publicPct(hp, max);
                row[F.status] = statusCategory(try r.byte());
            },
            .Cant => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                const reason = try r.byte();
                row[F.reason] = reason +| 1;
                if (reason == @intFromEnum(P.Cant.Disable)) row[F.move] = try r.byte();
            },
            .Faint, .Miss, .MustRecharge, .Crit, .SuperEffective, .Resisted => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
            },
            .Turn => {
                const turn = try r.u16v();
                row[F.num] = @intCast(@min(turn, 255));
            },
            .Win => {
                const player = try r.byte();
                row[F.num] = if (player == viewer) 1 else 2;
            },
            .Damage, .Heal => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                const hp = try r.u16v();
                const max = try r.u16v();
                row[F.hp] = publicPct(hp, max);
                row[F.status] = statusCategory(try r.byte());
                const reason = try r.byte();
                row[F.reason] = reason +| 1;
                if (arg == .Damage and reason == @intFromEnum(P.Damage.RecoilOf)) ident(battle, viewer, try r.byte(), &row, F.target_side, F.target_species);
                if (arg == .Heal and reason == @intFromEnum(P.Heal.Drain)) ident(battle, viewer, try r.byte(), &row, F.target_side, F.target_species);
            },
            .Status => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                row[F.status] = statusCategory(try r.byte());
                const reason = try r.byte();
                row[F.reason] = reason +| 1;
                if (reason == @intFromEnum(P.Status.From)) row[F.aux_move] = try r.byte();
            },
            .CureStatus => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                row[F.status] = statusCategory(try r.byte());
                row[F.reason] = (try r.byte()) +| 1;
            },
            .Boost => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                row[F.reason] = (try r.byte()) +| 1;
                // (the engine writes the amount + 6: protocol.zig `boost`)
                const amount: i16 = @as(i16, try r.byte()) - 6;
                row[F.num] = @intCast(amount + 128);
            },
            .Fail, .End, .Immune => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                row[F.reason] = (try r.byte()) +| 1;
            },
            .HitCount => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                row[F.num] = try r.byte();
            },
            .Prepare => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                row[F.move] = try r.byte();
            },
            .Activate => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                const reason = try r.byte();
                row[F.reason] = reason +| 1;
                if (reason == @intFromEnum(P.Activate.SubstituteBlock)) row[F.aux_move] = try r.byte() else if (reason == @intFromEnum(P.Activate.Magnitude)) row[F.num] = try r.byte();
            },
            .Start => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                const reason = try r.byte();
                row[F.reason] = reason +| 1;
                if (reason == @intFromEnum(P.Start.TypeChange)) row[F.num] = try r.byte() else if (reason >= @intFromEnum(P.Start.Disable)) row[F.move] = try r.byte();
            },
            .Transform => {
                ident(battle, viewer, try r.byte(), &row, F.actor_side, F.actor_species);
                ident(battle, viewer, try r.byte(), &row, F.target_side, F.target_species);
            },
            else => return error.UnknownEvent, // (Gen 2+ events: not produced by the Gen 1 engine)
        }
        if (n_out.* < out.len) {
            out[n_out.*] = row;
            n_out.* += 1;
        } else dropped += 1;
    }
    return dropped;
}
