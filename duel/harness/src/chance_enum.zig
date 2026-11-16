//! Exact chance outcomes for one turn (after pkmn_stuff's `search_leaves_chance`).
//!
//! Instead of one random draw per (own action, reply) cell, a turn is split into its important
//! random events with their probabilities: accuracy, critical hits, secondary effects, full
//! paralysis, confusion self-hits, speed ties and KO-deciding damage rolls. Each branch forces the
//! event through the engine's calc overrides (`overridden`), and the engine's chance hooks report
//! every event with its probability, so no engine change is needed.
//!
//! Expansion is best-first by path probability. A path stops splitting when its probability falls
//! below `min_path` or the leaf budget is spent; its remaining events are drawn from the seeded RNG,
//! an unbiased stand-in for the rest of that subtree. Damage rolls that cannot decide a KO (or a
//! Substitute break) use the middle roll; when they can, the roll range is split at the threshold
//! into two buckets, each represented by one roll and weighted by its share of the 39 rolls.
//!
//! The leaf weights sum to 1. Every run of one cell uses the same seed, so events before the branch
//! point are identical in parent and child (they are forced, or drawn by the same RNG calls).

const std = @import("std");
const pkmn = @import("pkmn");

pub const Battle = pkmn.gen1.Battle(pkmn.gen1.PRNG);
const Action = pkmn.gen1.chance.Action;
const Player = pkmn.Player;

pub const Kind = enum(u8) { hit, crit, damage, secondary, confused, paralyzed, speed_tie };
const KINDS = @typeInfo(Kind).@"enum".fields.len;

pub const Event = struct {
    player: u1,
    kind: Kind,
    /// Probability that the event's outcome is `true` (for damage: unused).
    p: f32,
    outcome: bool,
};

const MAX_EVENTS = 48;

/// Chance hooks: records the events of one update in order. Matches the engine's chance interface
/// (see pkmn's gen1 chance `Null`); events that are not enumerated are ignored.
pub const Recorder = struct {
    events: [MAX_EVENTS]Event = undefined,
    n: u8 = 0,

    pub const Error = error{};

    fn push(self: *Recorder, player: Player, kind: Kind, p: f32, outcome: bool) void {
        if (self.n >= MAX_EVENTS) return;
        self.events[self.n] = .{ .player = @intFromEnum(player), .kind = kind, .p = p, .outcome = outcome };
        self.n += 1;
    }

    pub fn commit(self: *Recorder, player: Player, kind: anytype) Error!void {
        _ = .{ self, player, kind };
    }
    pub fn glitch(self: *Recorder) void {
        _ = self;
    }
    pub fn clearPending(self: *Recorder) void {
        _ = self;
    }
    pub fn switched(self: *Recorder, player: Player, slot: u8) void {
        _ = .{ self, player, slot };
    }
    pub fn speedTie(self: *Recorder, p1: bool) Error!void {
        self.push(.P1, .speed_tie, 0.5, p1);
    }
    pub fn hit(self: *Recorder, player: Player, ok: bool, accuracy: anytype) Error!void {
        self.push(player, .hit, @as(f32, @floatFromInt(accuracy)) / 256, ok);
    }
    pub fn criticalHit(self: *Recorder, player: Player, crit: bool, rate: u8) Error!void {
        self.push(player, .crit, @as(f32, @floatFromInt(rate)) / 256, crit);
    }
    pub fn damage(self: *Recorder, player: Player, roll: u8) Error!void {
        _ = roll;
        self.push(player, .damage, 0, false);
    }
    pub fn confused(self: *Recorder, player: Player, ok: bool) Error!void {
        self.push(player, .confused, 0.5, ok);
    }
    pub fn paralyzed(self: *Recorder, player: Player, ok: bool) Error!void {
        // Pokémon Showdown's full paralysis chance (63/256); this engine is built with -Dshowdown
        self.push(player, .paralyzed, 63.0 / 256.0, ok);
    }
    pub fn secondaryChance(self: *Recorder, player: Player, proc: bool, rate: u8) Error!void {
        self.push(player, .secondary, @as(f32, @floatFromInt(rate)) / 256, proc);
    }
    pub fn moveSlot(self: *Recorder, player: Player, slot: u4, ms: anytype, n: u4) Error!void {
        _ = .{ self, player, slot, ms, n };
    }
    pub fn multiHit(self: *Recorder, player: Player, n: u3) Error!void {
        _ = .{ self, player, n };
    }
    pub fn duration(self: *Recorder, player: Player, turns: u4) void {
        _ = .{ self, player, turns };
    }
    pub fn observe(self: *Recorder, comptime field: Action.Field, player: Player, obs: @FieldType(Action, "confusion")) void {
        _ = .{ self, field, player, obs };
    }
    pub fn sleep(self: *Recorder, player: Player, obs: @FieldType(Action, "sleep")) Error!void {
        _ = .{ self, player, obs };
    }
    pub fn confusion(self: *Recorder, player: Player, obs: @FieldType(Action, "confusion")) Error!void {
        _ = .{ self, player, obs };
    }
    pub fn disable(self: *Recorder, player: Player, obs: @FieldType(Action, "disable")) Error!void {
        _ = .{ self, player, obs };
    }
    pub fn attacking(self: *Recorder, player: Player, obs: @FieldType(Action, "attacking")) Error!void {
        _ = .{ self, player, obs };
    }
    pub fn binding(self: *Recorder, player: Player, obs: @FieldType(Action, "binding")) Error!void {
        _ = .{ self, player, obs };
    }
    pub fn psywave(self: *Recorder, player: Player, power: u8, max: u8) Error!void {
        _ = .{ self, player, power, max };
    }
    pub fn metronome(self: *Recorder, player: Player, move: anytype) Error!void {
        _ = .{ self, player, move };
    }
    pub fn pp(self: *Recorder, player: Player, n: u4) bool {
        _ = .{ self, player, n };
        return true;
    }
};

/// Forced outcomes for one player (0 = not forced; 1 = false; 2 = true).
pub const Forced = struct {
    hit: u2 = 0,
    crit: u2 = 0,
    secondary: u2 = 0,
    confused: u2 = 0,
    paralyzed: u2 = 0,
    damage: u8 = 0,

    fn get(self: Forced, kind: Kind) u8 {
        return switch (kind) {
            .hit => self.hit,
            .crit => self.crit,
            .secondary => self.secondary,
            .confused => self.confused,
            .paralyzed => self.paralyzed,
            .damage => self.damage,
            .speed_tie => 0,
        };
    }
};

pub const Assignment = struct {
    side: [2]Forced = .{ .{}, .{} },
    /// 0 = not forced; 1 = P1 moves first; 2 = P2 moves first.
    speed_tie: u2 = 0,

    fn isForced(self: Assignment, e: Event) bool {
        if (e.kind == .speed_tie) return self.speed_tie != 0;
        return self.side[e.player].get(e.kind) != 0;
    }

    fn with(self: Assignment, e: Event, outcome: bool) Assignment {
        var a = self;
        const v: u2 = if (outcome) 2 else 1;
        const s = &a.side[e.player];
        switch (e.kind) {
            .hit => s.hit = v,
            .crit => s.crit = v,
            .secondary => s.secondary = v,
            .confused => s.confused = v,
            .paralyzed => s.paralyzed = v,
            .speed_tie => a.speed_tie = if (outcome) 1 else 2,
            .damage => unreachable,
        }
        return a;
    }

    fn withDamage(self: Assignment, player: u1, roll: u8) Assignment {
        var a = self;
        a.side[player].damage = roll;
        return a;
    }
};

fn optBool(comptime T: type, v: u2) ?T {
    return switch (v) {
        0 => null,
        1 => .false,
        else => .true,
    };
}

/// Calc hooks: forces the assignment's outcomes and records which defenders were "capped" (KOed
/// directly by the opponent's move, or had a Substitute broken).
pub const Forcer = struct {
    assignment: Assignment = .{},
    capped_flag: [2]bool = .{ false, false },

    pub fn overridden(self: Forcer, player: Player, comptime field: Action.Field) ?@FieldType(Action, @tagName(field)) {
        const T = @FieldType(Action, @tagName(field));
        const f = self.assignment.side[@intFromEnum(player)];
        if (comptime field == .hit) return optBool(T, f.hit);
        if (comptime field == .critical_hit) return optBool(T, f.crit);
        if (comptime field == .secondary_chance) return optBool(T, f.secondary);
        if (comptime field == .confused) return optBool(T, f.confused);
        if (comptime field == .paralyzed) return optBool(T, f.paralyzed);
        if (comptime field == .damage) return if (f.damage != 0) f.damage else null;
        if (comptime field == .speed_tie) return switch (self.assignment.speed_tie) {
            0 => null,
            1 => .P1,
            else => .P2,
        };
        return null;
    }
    pub fn confusion(self: *Forcer, player: Player) void {
        _ = .{ self, player };
    }
    pub fn base(self: *Forcer, player: Player, val: u16) void {
        _ = .{ self, player, val };
    }
    pub fn final(self: *Forcer, player: Player, val: u16) void {
        _ = .{ self, player, val };
    }
    pub fn capped(self: *Forcer, player: Player) void {
        self.capped_flag[@intFromEnum(player)] = true;
    }
};

pub const Run = struct {
    battle: Battle,
    result: pkmn.Result,
    events: [MAX_EVENTS]Event,
    n: u8,
    capped: [2]bool,
};

/// One update of `battle` with `c1`/`c2`, the assignment forced and other events drawn from `seed`.
pub fn run(battle: *const Battle, c1: pkmn.Choice, c2: pkmn.Choice, seed: u64, a: Assignment) !Run {
    var b = battle.*;
    b.rng = pkmn.gen1.PRNG.init(seed);
    var opts = pkmn.battle.options(pkmn.protocol.NULL, Recorder{}, Forcer{ .assignment = a });
    const r = try b.update(c1, c2, &opts);
    return .{ .battle = b, .result = r, .events = opts.chance.events, .n = opts.chance.n, .capped = opts.calc.capped_flag };
}

pub const Params = struct {
    /// Paths less likely than this are not split further.
    min_path: f32 = 0.05,
    /// Most leaves per cell.
    max_leaves: u8 = 16,
};

pub const Leaf = struct {
    battle: Battle,
    result: pkmn.Result,
    weight: f32,
};

pub const MAX_LEAVES = 64;

const Open = struct {
    a: Assignment,
    prob: f32,
    run: Run,
    /// Events of a player made irrelevant by a forced miss (crit, damage, secondary).
    skip: [2]bool = .{ false, false },
    done: bool = false,
};

fn branchable(e: Event) bool {
    return e.kind == .damage or (e.p > 0 and e.p < 1);
}

/// The event to split `o` on, or null. Events are taken in the order they happened, except that a
/// player's critical hit or damage roll waits for that player's later accuracy roll (accuracy does
/// not depend on them, and a miss makes them irrelevant).
fn branchEvent(o: *const Open) ?Event {
    const evs = o.run.events[0..o.run.n];
    for (evs, 0..) |e, i| {
        if (o.a.isForced(e) or !branchable(e)) continue;
        if (o.skip[e.player] and (e.kind == .crit or e.kind == .damage or e.kind == .secondary)) continue;
        if (e.kind == .crit or e.kind == .damage) {
            for (evs[i + 1 ..]) |later| {
                if (later.player == e.player and later.kind == .hit and !o.a.isForced(later) and branchable(later)) return later;
            }
        }
        return e;
    }
    return null;
}

const ROLL_MIN: u8 = 217;
const ROLL_MAX: u8 = 255;
const ROLLS: f32 = 39;

/// Splits the leaves of one turn. Returns the number of leaves written to `out`.
pub fn enumerate(battle: *const Battle, c1: pkmn.Choice, c2: pkmn.Choice, seed: u64, params: Params, out: []Leaf) !usize {
    const cap = @min(@as(usize, params.max_leaves), out.len, MAX_LEAVES);
    std.debug.assert(cap >= 1);
    var open: [MAX_LEAVES]Open = undefined;
    var count: usize = 1;
    open[0] = .{ .a = .{}, .prob = 1, .run = try run(battle, c1, c2, seed, .{}) };
    while (true) {
        // the most likely path that can still be split
        var pick: ?usize = null;
        for (open[0..count], 0..) |*o, i| {
            if (o.done) continue;
            if (pick == null or o.prob > open[pick.?].prob) pick = i;
        }
        const i = pick orelse break;
        const o = &open[i];
        if (o.prob < params.min_path) {
            o.done = true;
            continue;
        }
        const e = branchEvent(o) orelse {
            o.done = true;
            continue;
        };
        if (e.kind == .damage) {
            const foe = 1 - e.player;
            const lo = try run(battle, c1, c2, seed, o.a.withDamage(e.player, ROLL_MIN));
            const hi = try run(battle, c1, c2, seed, o.a.withDamage(e.player, ROLL_MAX));
            if (lo.capped[foe] or !hi.capped[foe]) {
                // the roll cannot decide a KO: the middle roll stands for all of them
                const a = o.a.withDamage(e.player, 236);
                o.* = .{ .a = a, .prob = o.prob, .run = try run(battle, c1, c2, seed, a), .skip = o.skip };
                continue;
            }
            if (count >= cap) {
                o.done = true;
                continue;
            }
            // smallest capped roll
            var l: u8 = ROLL_MIN;
            var h: u8 = ROLL_MAX;
            while (h - l > 1) {
                const mid = l + (h - l) / 2;
                const r = try run(battle, c1, c2, seed, o.a.withDamage(e.player, mid));
                if (r.capped[foe]) h = mid else l = mid;
            }
            const low_n: f32 = @floatFromInt(h - ROLL_MIN);
            const a_low = o.a.withDamage(e.player, ROLL_MIN + (h - 1 - ROLL_MIN) / 2);
            const a_high = o.a.withDamage(e.player, h);
            const prob = o.prob;
            const skip = o.skip;
            o.* = .{ .a = a_low, .prob = prob * low_n / ROLLS, .run = try run(battle, c1, c2, seed, a_low), .skip = skip };
            open[count] = .{ .a = a_high, .prob = prob * (ROLLS - low_n) / ROLLS, .run = try run(battle, c1, c2, seed, a_high), .skip = skip };
            count += 1;
            continue;
        }
        if (count >= cap) {
            o.done = true;
            continue;
        }
        const prob = o.prob;
        const a_true = o.a.with(e, true);
        const a_false = o.a.with(e, false);
        var skip_false = o.skip;
        if (e.kind == .hit) skip_false[e.player] = true;
        const skip_true = o.skip;
        o.* = .{ .a = a_true, .prob = prob * e.p, .run = try run(battle, c1, c2, seed, a_true), .skip = skip_true };
        open[count] = .{ .a = a_false, .prob = prob * (1 - e.p), .run = try run(battle, c1, c2, seed, a_false), .skip = skip_false };
        count += 1;
    }
    for (open[0..count], 0..) |*o, k| out[k] = .{ .battle = o.run.battle, .result = o.run.result, .weight = o.prob };
    return count;
}

test "leaf weights sum to one and accuracy is split" {
    const a = [_]pkmn.gen1.helpers.Pokemon{.{ .species = .Tauros, .moves = &.{ .BodySlam, .Blizzard, .HyperBeam } }};
    const b = [_]pkmn.gen1.helpers.Pokemon{.{ .species = .Chansey, .moves = &.{ .SeismicToss, .ThunderWave } }};
    var battle = pkmn.gen1.helpers.Battle.init(3, &a, &b);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try battle.update(.{}, .{}, &options);
    var leaves: [MAX_LEAVES]Leaf = undefined;
    // Blizzard (90% accuracy) against Thunder Wave
    const n = try enumerate(&battle, .{ .type = .Move, .data = 2 }, .{ .type = .Move, .data = 2 }, 7, .{ .min_path = 0.0, .max_leaves = 32 }, &leaves);
    try std.testing.expect(n >= 2);
    var total: f32 = 0;
    for (leaves[0..n]) |l| total += l.weight;
    try std.testing.expectApproxEqAbs(@as(f32, 1), total, 1e-5);
    // some leaf has Chansey untouched (Blizzard missed)
    var missed: f32 = 0;
    for (leaves[0..n]) |l| {
        if (l.battle.sides[1].pokemon[0].hp == l.battle.sides[1].pokemon[0].stats.hp) missed += l.weight;
    }
    try std.testing.expect(missed > 0.05 and missed < 0.2);
}

test "KO-deciding damage is split at the threshold" {
    const a = [_]pkmn.gen1.helpers.Pokemon{.{ .species = .Tauros, .moves = &.{.BodySlam} }};
    const b = [_]pkmn.gen1.helpers.Pokemon{ .{ .species = .Alakazam, .moves = &.{.Splash} }, .{ .species = .Chansey, .moves = &.{.SeismicToss} } };
    var battle = pkmn.gen1.helpers.Battle.init(5, &a, &b);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try battle.update(.{}, .{}, &options);
    const full = battle.sides[1].pokemon[0].stats.hp;
    // Alakazam's HP set to the damage of a non-critical roll 236: rolls 236-255 KO, 217-235 do not
    var a0 = Assignment{};
    a0.side[0].crit = 1;
    a0.side[0].hit = 2;
    a0.side[0].secondary = 1;
    const mid = try run(&battle, .{ .type = .Move, .data = 1 }, .{ .type = .Move, .data = 1 }, 1, a0.withDamage(0, 236));
    const dealt = full - mid.battle.sides[1].pokemon[0].hp;
    battle.sides[1].pokemon[0].hp = dealt;
    var leaves: [MAX_LEAVES]Leaf = undefined;
    const n = try enumerate(&battle, .{ .type = .Move, .data = 1 }, .{ .type = .Move, .data = 1 }, 9, .{ .min_path = 0.0, .max_leaves = 32 }, &leaves);
    var total: f32 = 0;
    var ko: f32 = 0;
    for (leaves[0..n]) |l| {
        total += l.weight;
        if (l.battle.sides[1].pokemon[0].hp == 0) ko += l.weight;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1), total, 1e-5);
    // KO = hit x (crit + no crit x 20/39 rolls): about 0.62 for Tauros (crit rate 55/256)
    try std.testing.expect(ko > 0.5 and ko < 0.75);
}
