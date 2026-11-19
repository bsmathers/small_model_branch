//! The self-play environment: many Gen 1 OU games on the pkmn engine at once, as a C library loaded into the training process
//! (selfplay/py/pkrl.py, ctypes). Each call writes every pending decision straight into caller-owned arrays (NumPy, pinned):
//! no processes, no serialization, no allocation per step.
//!
//! Rows: game g's player p is row 2g + p. A row is valid when that player has a decision this step; then it holds
//!   ids, floats   the board input (the 1.3M model's rmodel.Input, flattened: the native feature code rnative.zig, exact
//!                 against the Python pipeline; the "last" fields are present but the new model does not read them)
//!   events        the log since the player's previous decision (events.zig rows, at most MAX_EVENTS, the latest kept)
//!   legal         a mask over the 9 policy slots: 0-3 the active Pokemon's moves sorted by name, 4-8 the reserves sorted by
//!                 species (the order of the board's own Pokemon), the pointer head's options
//!   forced        one legal option (no policy decision; the model still steps its state)
//!   first         the player's first decision of a new game (the model starts from a zero state)
//! Per game: ended (the game finished this step; its outcome, from P1's side: 1 won, -1 lost, 0 tie or cut off) and turn.
//! A finished game starts again at once with new teams, its new rows written in the same call.
//!
//! Actions: one byte per row, the chosen slot (ignored for invalid and forced rows). An illegal slot plays the first legal
//! option and is counted (stats.invalid_actions).
const std = @import("std");
const pkmn = @import("pkmn");
const H = @import("harness");
const events = @import("events.zig");
/// The protocol enums (for tools that print event rows).
pub const P = @import("protocol_enums.zig");

const rnative = H.rnative;
const know = H.know;
const rmodel = H.rmodel;
const teamsmod = H.teams;
const Battle = pkmn.gen1.Battle(pkmn.gen1.PRNG);

pub const MAX_EVENTS = 32;
pub const SLOTS = 9;
const NONE: u8 = 255;
const LOG_BYTES = 4096;

// ---------------------------------------------------------------- the board input, flattened

pub const IDS = 6 + 6 + 24 + 6 + 6 + 24 + 6;
pub const FLOATS = 6 * 10 + 6 * 7 + @typeInfo(@FieldType(rmodel.Input, "last")).array.len + @typeInfo(@FieldType(rmodel.Input, "field")).array.len;

fn flatten(x: *const rmodel.Input, ids: []i16, floats: []f32) void {
    var i: usize = 0;
    for (x.own_species) |v| put(ids, &i, v);
    for (x.own_status) |v| put(ids, &i, v);
    for (x.own_moves) |m| for (m) |v| put(ids, &i, v);
    for (x.opp_species) |v| put(ids, &i, v);
    for (x.opp_status) |v| put(ids, &i, v);
    for (x.opp_moves) |m| for (m) |v| put(ids, &i, v);
    for ([_]u32{ x.last_own_move, x.last_opp_move, x.last_own_species, x.last_opp_species, x.last_own_status, x.last_opp_status }) |v| put(ids, &i, v);
    std.debug.assert(i == IDS);
    var j: usize = 0;
    for (x.own) |r| for (r) |v| {
        floats[j] = v;
        j += 1;
    };
    for (x.opp) |r| for (r) |v| {
        floats[j] = v;
        j += 1;
    };
    for (x.last) |v| {
        floats[j] = v;
        j += 1;
    }
    for (x.field) |v| {
        floats[j] = v;
        j += 1;
    }
    std.debug.assert(j == FLOATS);
}

fn put(ids: []i16, i: *usize, v: u32) void {
    ids[i.*] = @intCast(v);
    i.* += 1;
}

// ---------------------------------------------------------------- the C interface

pub const Layout = extern struct {
    ids: u32 = IDS,
    floats: u32 = FLOATS,
    max_events: u32 = MAX_EVENTS,
    fields: u32 = events.FIELDS,
    slots: u32 = SLOTS,
};

/// Caller-owned arrays, R = 2 x games rows.
pub const Buffers = extern struct {
    valid: [*]u8, // [R]
    forced: [*]u8, // [R]
    first: [*]u8, // [R]
    ids: [*]i16, // [R, IDS]
    floats: [*]f32, // [R, FLOATS]
    events: [*]u8, // [R, MAX_EVENTS, FIELDS]
    n_events: [*]u8, // [R]
    legal: [*]u8, // [R, SLOTS]
    ended: [*]u8, // [G]
    outcome: [*]i8, // [G]
    turn: [*]u16, // [G]
};

pub const Stats = extern struct {
    decisions: u64 = 0,
    games: u64 = 0,
    p1_wins: u64 = 0,
    ties: u64 = 0,
    cut_off: u64 = 0,
    invalid_actions: u64 = 0,
    dropped_events: u64 = 0,
    decode_errors: u64 = 0,
    engine_errors: u64 = 0,
};

const Player = struct {
    nodes: [3]rnative.Node = undefined,
    current: u8 = NONE,
    anchor: u8 = NONE,
    pending: [MAX_EVENTS]events.Row = undefined,
    n_pending: usize = 0,
    slot_choice: [SLOTS]u8 = @splat(NONE),
    choices: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined,
    n_choices: usize = 0,
    row: bool = false, // a decision row this step
    parent: u8 = NONE, // the node the latest one was made from, and what it was given (hiddenInfoCheck)
    sent: rnative.Option = .{},
    last: [2]rnative.Act = .{ .{}, .{} },
    forced: bool = false,
    first: bool = true,
};

const Game = struct {
    battle: Battle = undefined,
    result: pkmn.Result = .{},
    knowledge: know.Knowledge = .{},
    rng: std.Random.DefaultPrng = undefined,
    players: [2]Player = .{ .{}, .{} },
    log: [LOG_BYTES]u8 = undefined,
};

pub const Env = struct {
    allocator: std.mem.Allocator,
    tables: *rnative.Tables,
    pools: [2][]teamsmod.Team,
    share_a: f32,
    games: []Game,
    threads: usize,
    max_turns: u32,
    stats: Stats = .{},
};

var last_error: [256:0]u8 = @splat(0);

fn fail(comptime fmt: []const u8, args: anytype) void {
    _ = std.fmt.bufPrintZ(&last_error, fmt, args) catch {};
}

pub export fn pkrl_error() [*:0]const u8 {
    return &last_error;
}

pub export fn pkrl_layout(out: *Layout) void {
    out.* = .{};
}

/// `tables_dir`: an export's tables/ (duel/tools/export_recurrent.py); `teams_a` / `teams_b`: Showdown team folders ("" for
/// none), a game's team coming from A with probability `share_a`. `max_turns`: a game is cut off (a tie) after this turn.
pub export fn pkrl_create(n_games: u32, threads: u32, seed: u64, tables_dir: [*:0]const u8, teams_a: [*:0]const u8, teams_b: [*:0]const u8, share_a: f32, max_turns: u32) ?*Env {
    return create(n_games, threads, seed, std.mem.span(tables_dir), std.mem.span(teams_a), std.mem.span(teams_b), share_a, max_turns) catch |e| {
        fail("pkrl_create: {s}", .{@errorName(e)});
        return null;
    };
}

fn create(n_games: u32, threads: u32, seed: u64, tables_dir: []const u8, teams_a: []const u8, teams_b: []const u8, share_a: f32, max_turns: u32) !*Env {
    const allocator = std.heap.c_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const env = try allocator.create(Env);
    var pools: [2][]teamsmod.Team = .{ &.{}, &.{} };
    for ([2][]const u8{ teams_a, teams_b }, 0..) |dir, k| {
        if (dir.len == 0) continue;
        var stats = teamsmod.LoadStats{};
        pools[k] = try teamsmod.loadDir(allocator, io, dir, &stats);
        if (pools[k].len == 0) return error.EmptyTeamPool;
    }
    if (pools[0].len == 0) return error.NoTeams;
    env.* = .{
        .allocator = allocator,
        .tables = try rnative.Tables.load(allocator, io, tables_dir),
        .pools = pools,
        .share_a = if (pools[1].len == 0) 1.0 else share_a,
        .games = try allocator.alloc(Game, n_games),
        .threads = @max(1, @min(threads, n_games)),
        .max_turns = max_turns,
    };
    var root = std.Random.DefaultPrng.init(seed);
    for (env.games) |*g| {
        g.* = .{};
        g.rng = std.Random.DefaultPrng.init(root.random().int(u64));
    }
    return env;
}

pub export fn pkrl_destroy(env: *Env) void {
    env.allocator.free(env.games);
    env.allocator.destroy(env);
}

/// The number of teams loaded into each pool (A, B).
pub export fn pkrl_pool_sizes(env: *Env, out: *[2]u32) void {
    out.* = .{ @intCast(env.pools[0].len), @intCast(env.pools[1].len) };
}

pub export fn pkrl_stats(env: *Env, out: *Stats) void {
    out.* = env.stats;
}

/// Starts every game; writes the first rows.
pub export fn pkrl_reset(env: *Env, bufs: *const Buffers) i32 {
    return run(env, null, bufs);
}

/// Plays one step of every game with `actions` (one slot per row); writes the next rows.
pub export fn pkrl_step(env: *Env, actions: [*]const u8, bufs: *const Buffers) i32 {
    return run(env, actions, bufs);
}

// ---------------------------------------------------------------- parallel stepping

const Job = struct { env: *Env, actions: ?[*]const u8, bufs: *const Buffers, lo: usize, hi: usize, stats: Stats = .{}, err: ?anyerror = null };

fn run(env: *Env, actions: ?[*]const u8, bufs: *const Buffers) i32 {
    const n = env.games.len;
    const t = env.threads;
    var jobs: [64]Job = undefined;
    const used = @min(t, jobs.len);
    for (0..used) |k| jobs[k] = .{ .env = env, .actions = actions, .bufs = bufs, .lo = n * k / used, .hi = n * (k + 1) / used };
    var spawned: [64]?std.Thread = @splat(null);
    for (1..used) |k| spawned[k] = std.Thread.spawn(.{}, work, .{&jobs[k]}) catch null;
    for (1..used) |k| if (spawned[k] == null) work(&jobs[k]); // (no thread: done here)
    work(&jobs[0]);
    for (1..used) |k| if (spawned[k]) |th| th.join();
    var code: i32 = 0;
    for (jobs[0..used]) |*j| {
        inline for (@typeInfo(Stats).@"struct".fields) |f| @field(env.stats, f.name) += @field(j.stats, f.name);
        if (j.err) |e| {
            fail("pkrl: {s}", .{@errorName(e)});
            code = -1;
        }
    }
    return code;
}

fn work(job: *Job) void {
    for (job.lo..job.hi) |g| {
        stepGame(job, g) catch |e| {
            job.err = e;
            return;
        };
    }
}

// ---------------------------------------------------------------- one game

fn stepGame(job: *Job, g: usize) !void {
    const env = job.env;
    const game = &env.games[g];
    const bufs = job.bufs;
    bufs.ended[g] = 0;
    bufs.outcome[g] = 0;
    for (0..2) |p| bufs.valid[2 * g + p] = 0;
    const actions = job.actions orelse {
        try startGame(job, game, g);
        return;
    };
    // each player's choice
    var c: [2]pkmn.Choice = undefined;
    for (0..2) |p| {
        const pl = &game.players[p];
        c[p] = if (pl.n_choices > 0) pl.choices[0] else .{};
        if (pl.row and !pl.forced) {
            const a = actions[2 * g + p];
            const idx = if (a < SLOTS) pl.slot_choice[a] else NONE;
            if (idx != NONE) c[p] = pl.choices[idx] else {
                job.stats.invalid_actions += 1;
                for (pl.slot_choice) |s| if (s != NONE) {
                    c[p] = pl.choices[s];
                    break;
                };
            }
        }
    }
    const before = game.battle;
    var writer: pkmn.protocol.Writer = .{ .buffer = &game.log };
    var options = pkmn.battle.options(pkmn.protocol.FixedLog{ .writer = &writer }, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    const result = game.battle.update(c[0], c[1], &options) catch {
        job.stats.engine_errors += 1;
        try finish(job, game, g, 0, false);
        return;
    };
    game.result = result;
    game.knowledge = know.Knowledge.merge(game.knowledge, know.Knowledge.afterTransition(&before, &game.battle, c[0], c[1]));
    readLog(job, game, writer.buffer[0..writer.pos]);
    if (result.type != .None) {
        const outcome: i8 = switch (result.type) {
            .Win => 1,
            .Lose => -1,
            else => 0,
        };
        try finish(job, game, g, outcome, false);
        return;
    }
    if (game.battle.turn > env.max_turns) {
        try finish(job, game, g, 0, true);
        return;
    }
    const last = [2]rnative.Act{ rnative.actOf(&before, &game.battle, 0, c[0]), rnative.actOf(&before, &game.battle, 1, c[1]) };
    const sent = [2]rnative.Option{ rnative.optionOf(&before, 0, c[0]), rnative.optionOf(&before, 1, c[1]) };
    try observeBoth(job, game, g, last, sent);
}

fn finish(job: *Job, game: *Game, g: usize, outcome: i8, cut: bool) !void {
    const bufs = job.bufs;
    bufs.ended[g] = 1;
    bufs.outcome[g] = outcome;
    bufs.turn[g] = @intCast(@min(game.battle.turn, 65535));
    job.stats.games += 1;
    if (outcome == 1) job.stats.p1_wins += 1;
    if (outcome == 0) job.stats.ties += 1;
    if (cut) job.stats.cut_off += 1;
    try startGame(job, game, g);
}

fn pickTeam(env: *Env, r: std.Random) []const pkmn.gen1.helpers.Pokemon {
    const k: usize = if (env.pools[1].len == 0 or r.float(f32) < env.share_a) 0 else 1;
    const pool = env.pools[k];
    return pool[r.uintLessThan(usize, pool.len)].mons;
}

fn startGame(job: *Job, game: *Game, g: usize) !void {
    const env = job.env;
    const r = game.rng.random();
    const a = pickTeam(env, r);
    const b = pickTeam(env, r);
    game.battle = pkmn.gen1.helpers.Battle.init(r.int(u64), a, b);
    game.players = .{ .{}, .{} };
    var writer: pkmn.protocol.Writer = .{ .buffer = &game.log };
    var options = pkmn.battle.options(pkmn.protocol.FixedLog{ .writer = &writer }, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    game.result = try game.battle.update(.{}, .{}, &options);
    game.knowledge = know.initial(&game.battle);
    readLog(job, game, writer.buffer[0..writer.pos]);
    try observeBoth(job, game, g, .{ .{}, .{} }, .{ .{}, .{} });
}

/// The update's log, decoded for each player and appended to what it has not read yet (the oldest rows go first when full).
fn readLog(job: *Job, game: *Game, log: []const u8) void {
    for (0..2) |p| {
        const pl = &game.players[p];
        var rows: [MAX_EVENTS]events.Row = undefined;
        var n: usize = 0;
        const dropped = events.decode(log, &game.battle, p, &rows, &n) catch {
            job.stats.decode_errors += 1;
            continue;
        };
        job.stats.dropped_events += dropped;
        for (rows[0..n]) |row| {
            if (pl.n_pending == MAX_EVENTS) {
                std.mem.copyForwards(events.Row, pl.pending[0 .. MAX_EVENTS - 1], pl.pending[1..MAX_EVENTS]);
                pl.n_pending -= 1;
                job.stats.dropped_events += 1;
            }
            pl.pending[pl.n_pending] = row;
            pl.n_pending += 1;
        }
    }
}

/// Each player's observation after an update (or a game's start): its native feature node and, at a decision, its row.
fn observeBoth(job: *Job, game: *Game, g: usize, last: [2]rnative.Act, sent: [2]rnative.Option) !void {
    const env = job.env;
    const bufs = job.bufs;
    for (0..2) |p| {
        const pl = &game.players[p];
        pl.row = false;
        pl.forced = false;
        pl.slot_choice = @splat(NONE);
        const request = if (p == 0) game.result.p1 else game.result.p2;
        pl.n_choices = game.battle.choices(if (p == 0) .P1 else .P2, request, &pl.choices);
        // a free node slot: neither the latest node nor the last decision node (the next node reads both)
        var slot: u8 = 0;
        while (slot == pl.current or slot == pl.anchor) slot += 1;
        const node = &pl.nodes[slot];
        rnative.snapInto(&node.snap, env.tables, &game.battle, game.result, game.knowledge, p, last);
        const parent: ?*const rnative.Node = if (pl.current == NONE) null else &pl.nodes[pl.current];
        pl.parent = pl.current;
        pl.sent = sent[p];
        pl.last = last;
        var slots: [pkmn.CHOICES_SIZE]f32 = undefined;
        var input: rmodel.Input = .{};
        const decision = try rnative.observe(env.tables, parent, sent[p], &node.snap, node, &slots, &input);
        pl.current = slot;
        if (!decision) continue;
        pl.anchor = slot;
        const r = 2 * g + p;
        var mapped: usize = 0;
        const legal = bufs.legal[r * SLOTS ..][0..SLOTS];
        @memset(legal, 0);
        for (0..node.snap.n_legal) |i| {
            if (slots[i] < 0) continue;
            const s: usize = @intFromFloat(slots[i]);
            if (s >= SLOTS or pl.slot_choice[s] != NONE) continue;
            pl.slot_choice[s] = @intCast(i);
            legal[s] = 1;
            mapped += 1;
        }
        pl.choices = node.snap.choices; // (the snapshot's legal options, in the order its slots refer to)
        pl.n_choices = node.snap.n_legal;
        pl.forced = mapped <= 1 or node.snap.n_legal == 1;
        if (mapped == 0) {
            pl.slot_choice[0] = 0;
            legal[0] = 1;
        }
        pl.row = true;
        bufs.valid[r] = 1;
        bufs.forced[r] = @intFromBool(pl.forced);
        bufs.first[r] = @intFromBool(pl.first);
        pl.first = false;
        flatten(&input, bufs.ids[r * IDS ..][0..IDS], bufs.floats[r * FLOATS ..][0..FLOATS]);
        const ev = bufs.events[r * MAX_EVENTS * events.FIELDS ..][0 .. MAX_EVENTS * events.FIELDS];
        @memset(ev, 0);
        for (pl.pending[0..pl.n_pending], 0..) |row, k| @memcpy(ev[k * events.FIELDS ..][0..events.FIELDS], &row);
        bufs.n_events[r] = @intCast(pl.n_pending);
        pl.n_pending = 0;
        job.stats.decisions += 1;
    }
    bufs.turn[g] = @intCast(@min(game.battle.turn, 65535));
}

// ---------------------------------------------------------------- checks

/// Whether game `g`'s current observation for player `p` is unchanged when what `p` cannot know about the opponent changes:
/// the opponent's unseen Pokemon (species, moves), its seen Pokemon's exact HP within the same public percentage, their stats,
/// the moves the observer has not seen and their PP, the active's Substitute HP and a sleep counter. Recomputes the
/// observation twice from the same parent node (the true battle and the altered one) and compares the model input and the
/// legal slots. True when there is nothing to check (no decision row).
pub fn hiddenInfoCheck(env: *Env, g: usize, p: usize, r: std.Random, control: bool) !bool {
    const game = &env.games[g];
    const pl = &game.players[p];
    if (!pl.row) return true;
    const q = 1 - p;
    var altered = game.battle;
    const k = game.knowledge;
    const foe = &altered.sides[q];
    const active: usize = if (foe.order[0] > 0) foe.order[0] - 1 else 6;
    for (0..6) |slot| {
        const mon = &foe.pokemon[slot];
        if (mon.species == .None) continue;
        if (!k.seen[q][slot]) {
            // an unseen Pokemon: another species with other moves (a valid one: the engine reads them)
            mon.species = @enumFromInt(1 + r.uintLessThan(u8, 151));
            for (&mon.moves) |*m| if (m.id != .None) {
                m.id = @enumFromInt(1 + r.uintLessThan(u8, 164));
            };
            continue;
        }
        // seen: exact HP within the same public percentage, stats, unseen moves and their PP
        if (mon.hp > 1 and events.publicPct(mon.hp - 1, mon.stats.hp) == events.publicPct(mon.hp, mon.stats.hp)) mon.hp -= 1;
        mon.stats.atk +%= 7;
        mon.stats.def +%= 5;
        for (&mon.moves, 0..) |*m, j| {
            var known = false;
            for (k.moves[q][slot]) |km| known = known or (km == m.id and m.id != .None);
            if (!known and m.id != .None) {
                m.id = if (m.id == .Splash) .Teleport else .Splash;
                m.pp = 3;
                if (slot == active) foe.active.moves[j] = m.*;
            }
        }
        if (pkmn.gen1.Status.is(mon.status, .SLP) and !pkmn.gen1.Status.is(mon.status, .EXT)) mon.status = pkmn.gen1.Status.slp(1 + r.uintLessThan(u3, 7));
    }
    if (active < 6) {
        foe.active.stats.atk +%= 7;
        if (foe.active.volatiles.Substitute) foe.active.volatiles.substitute = 1 + r.uintLessThan(u8, 100);
    }
    // control: a public change as well (the opponent active's HP to a different public percentage), which must show
    if (control and active < 6 and foe.pokemon[active].hp > 0) foe.pokemon[active].hp = if (foe.pokemon[active].hp > foe.pokemon[active].stats.hp / 2) 1 else foe.pokemon[active].stats.hp;
    const parent: ?*const rnative.Node = if (pl.parent == NONE) null else &pl.nodes[pl.parent];
    var a: rnative.Node = undefined;
    var b: rnative.Node = undefined;
    var sa: [pkmn.CHOICES_SIZE]f32 = undefined;
    var sb: [pkmn.CHOICES_SIZE]f32 = undefined;
    var xa: rmodel.Input = .{};
    var xb: rmodel.Input = .{};
    rnative.snapInto(&a.snap, env.tables, &game.battle, game.result, k, p, pl.last);
    rnative.snapInto(&b.snap, env.tables, &altered, game.result, k, p, pl.last);
    const da = try rnative.observe(env.tables, parent, pl.sent, &a.snap, &a, &sa, &xa);
    const db = try rnative.observe(env.tables, parent, pl.sent, &b.snap, &b, &sb, &xb);
    if (da != db or a.snap.n_legal != b.snap.n_legal) return false;
    if (!std.mem.eql(u8, std.mem.asBytes(&xa), std.mem.asBytes(&xb))) return false;
    return std.mem.eql(u8, std.mem.sliceAsBytes(sa[0..a.snap.n_legal]), std.mem.sliceAsBytes(sb[0..b.snap.n_legal]));
}
