//! Speed and determinism of the environment library, without Python: every row plays a uniformly random legal slot.
//!   pkrl-bench speed GAMES STEPS THREADS TABLES TEAMS_A [TEAMS_B]
//!   pkrl-bench check GAMES STEPS THREADS TABLES TEAMS_A [TEAMS_B]     (two environments, same seed and actions: every buffer
//!                                                                      must agree after every step)
const std = @import("std");
const env = @import("env");
const pkmn = @import("pkmn");
const P = env.P;

const Bufs = struct {
    valid: []u8,
    forced: []u8,
    first: []u8,
    ids: []i16,
    floats: []f32,
    events: []u8,
    n_events: []u8,
    legal: []u8,
    ended: []u8,
    outcome: []i8,
    turn: []u16,

    fn init(a: std.mem.Allocator, games: usize) !Bufs {
        const r = 2 * games;
        return .{
            .valid = try a.alloc(u8, r),
            .forced = try a.alloc(u8, r),
            .first = try a.alloc(u8, r),
            .ids = try a.alloc(i16, r * env.IDS),
            .floats = try a.alloc(f32, r * env.FLOATS),
            .events = try a.alloc(u8, r * env.MAX_EVENTS * 12),
            .n_events = try a.alloc(u8, r),
            .legal = try a.alloc(u8, r * env.SLOTS),
            .ended = try a.alloc(u8, games),
            .outcome = try a.alloc(i8, games),
            .turn = try a.alloc(u16, games),
        };
    }

    fn c(self: *Bufs) env.Buffers {
        return .{ .valid = self.valid.ptr, .forced = self.forced.ptr, .first = self.first.ptr, .ids = self.ids.ptr, .floats = self.floats.ptr, .events = self.events.ptr, .n_events = self.n_events.ptr, .legal = self.legal.ptr, .ended = self.ended.ptr, .outcome = self.outcome.ptr, .turn = self.turn.ptr };
    }

    fn same(a: *const Bufs, b: *const Bufs) bool {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| {
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(@field(a, f.name)), std.mem.sliceAsBytes(@field(b, f.name)))) return false;
        }
        return true;
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn choose(b: *const Bufs, rows: usize, r: std.Random, actions: []u8) void {
    for (0..rows) |i| {
        actions[i] = 255;
        if (b.valid[i] == 0) continue;
        var legal: [9]u8 = undefined;
        var n: usize = 0;
        for (0..9) |s| if (b.legal[i * 9 + s] != 0) {
            legal[n] = @intCast(s);
            n += 1;
        };
        if (n > 0) actions[i] = legal[r.uintLessThan(usize, n)];
    }
}

pub fn main(init: std.process.Init) !void {
    const a = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 7) {
        std.debug.print("usage: pkrl-bench speed|check GAMES STEPS THREADS TABLES TEAMS_A [TEAMS_B]\n", .{});
        return error.Usage;
    }
    const mode = args[1];
    const games = try std.fmt.parseUnsigned(u32, args[2], 10);
    const steps = try std.fmt.parseUnsigned(u32, args[3], 10);
    const threads = try std.fmt.parseUnsigned(u32, args[4], 10);
    const tables = try a.dupeZ(u8, args[5]);
    const teams_a = try a.dupeZ(u8, args[6]);
    const teams_b = try a.dupeZ(u8, if (args.len > 7) args[7] else "");
    const check = std.mem.eql(u8, mode, "check");
    const trace = std.mem.eql(u8, mode, "trace");
    const leak = std.mem.eql(u8, mode, "leak") or std.mem.eql(u8, mode, "leak-control");
    const control = std.mem.eql(u8, mode, "leak-control");
    var leak_checked: u64 = 0;
    var leak_failed: u64 = 0;

    const e1 = env.pkrl_create(games, threads, 7, tables, teams_a, teams_b, 0.5, 1000) orelse {
        std.debug.print("create failed: {s}\n", .{env.pkrl_error()});
        return error.Create;
    };
    const e2 = if (check) env.pkrl_create(games, threads, 7, tables, teams_a, teams_b, 0.5, 1000) orelse return error.Create else null;
    var b1 = try Bufs.init(a, games);
    var b2 = try Bufs.init(a, games);
    var c1 = b1.c();
    var c2 = b2.c();
    if (env.pkrl_reset(e1, &c1) != 0) return error.Reset;
    if (e2) |e| if (env.pkrl_reset(e, &c2) != 0) return error.Reset;
    const rows = 2 * games;
    const actions = try a.alloc(u8, rows);
    var prng = std.Random.DefaultPrng.init(11);
    var events_total: u64 = 0;
    var rows_total: u64 = 0;
    var max_events: u8 = 0;
    var mismatches: u32 = 0;
    var t_env: u64 = 0;
    const t0 = nowNs();
    for (0..steps) |_| {
        choose(&b1, rows, prng.random(), actions);
        for (0..rows) |i| if (b1.valid[i] != 0) {
            events_total += b1.n_events[i];
            rows_total += 1;
            max_events = @max(max_events, b1.n_events[i]);
        };
        if (leak) {
            for (0..games) |g| for (0..2) |p| if (b1.valid[2 * g + p] != 0) {
                leak_checked += 1;
                if (!try env.hiddenInfoCheck(e1, g, p, prng.random(), control)) leak_failed += 1;
            };
        }
        if (trace and b1.valid[0] != 0) printRow(&b1, 0);
        if (trace and b1.ended[0] != 0) std.debug.print("=== game 0 ended: outcome {d} (P1 side), turn {d}\n", .{ b1.outcome[0], b1.turn[0] });
        const ts = nowNs();
        if (env.pkrl_step(e1, actions.ptr, &c1) != 0) {
            std.debug.print("step failed: {s}\n", .{env.pkrl_error()});
            return error.Step;
        }
        t_env += nowNs() - ts;
        if (e2) |e| {
            if (env.pkrl_step(e, actions.ptr, &c2) != 0) return error.Step;
            if (!Bufs.same(&b1, &b2)) mismatches += 1;
        }
    }
    const elapsed = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
    var stats: env.Stats = undefined;
    env.pkrl_stats(e1, &stats);
    const env_s = @as(f64, @floatFromInt(t_env)) / 1e9;
    std.debug.print("{d} games x {d} steps on {d} threads: {d} decisions, {d} games finished in {d:.2} s ({d:.2} s in the library): {d:.0} decisions/s, {d:.0} games/s\n", .{ games, steps, threads, stats.decisions, stats.games, elapsed, env_s, @as(f64, @floatFromInt(stats.decisions)) / env_s, @as(f64, @floatFromInt(stats.games)) / env_s });
    std.debug.print("P1 won {d}, ties {d} (cut off {d}); events per decision {d:.1} (max {d}, dropped {d}); invalid actions {d}, decode errors {d}, engine errors {d}\n", .{ stats.p1_wins, stats.ties, stats.cut_off, @as(f64, @floatFromInt(events_total)) / @as(f64, @floatFromInt(@max(rows_total, 1))), max_events, stats.dropped_events, stats.invalid_actions, stats.decode_errors, stats.engine_errors });
    if (check) std.debug.print("determinism: {d} of {d} steps differed\n", .{ mismatches, steps });
    if (leak) std.debug.print("hidden information: {d} of {d} observations changed when the opponent's hidden state did\n", .{ leak_failed, leak_checked });
}

fn printRow(b: *const Bufs, r: usize) void {
    const side = [_][]const u8{ "-", "own", "opp" };
    std.debug.print("--- decision (turn {d}){s}{s}; legal slots:", .{ b.turn[r / 2], if (b.first[r] != 0) " FIRST" else "", if (b.forced[r] != 0) " forced" else "" });
    for (0..9) |s| if (b.legal[r * 9 + s] != 0) std.debug.print(" {d}", .{s});
    std.debug.print("\n", .{});
    for (0..b.n_events[r]) |k| {
        const row = b.events[(r * env.MAX_EVENTS + k) * 12 ..][0..12];
        const kind: P.ArgType = @enumFromInt(row[0]);
        std.debug.print("  {s:<15}", .{@tagName(kind)});
        if (row[1] != 0) std.debug.print(" {s} {s}", .{ side[row[1]], @tagName(@as(pkmn.gen1.Species, @enumFromInt(row[2]))) });
        if (row[3] != 0) std.debug.print(" -> {s} {s}", .{ side[row[3]], @tagName(@as(pkmn.gen1.Species, @enumFromInt(row[4]))) });
        if (row[5] != 0) std.debug.print(" move {s}", .{@tagName(@as(pkmn.gen1.Move, @enumFromInt(row[5])))});
        if (row[6] != 0) std.debug.print(" from {s}", .{@tagName(@as(pkmn.gen1.Move, @enumFromInt(row[6])))});
        if (row[7] != 0) std.debug.print(" status {d}", .{row[7]});
        if (row[8] != 0) std.debug.print(" reason {d}", .{row[8] - 1});
        if (row[9] != 0 or kind == .Damage or kind == .Switch) std.debug.print(" hp {d}%", .{row[9]});
        if (row[10] != 0) std.debug.print(" num {d}", .{row[10]});
        std.debug.print("\n", .{});
    }
}
