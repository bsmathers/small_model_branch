//! Raw policy opponent: the epoch-20 Tauros actor, run through bot2's reference compact policy server
//! (selfplay/tools/compact_tauros_policy_server.py) as a subprocess. Each decision sends the observer
//! snapshot (as JSON, with the `legal` options) and reads back the chosen option's name.

const std = @import("std");
const vs = @import("value_state.zig");

// (repo copy, for metamon.recurrent.tournament) The TaurosV0 and recurrent servers' paths come from the environment:
//   DUEL_PYTHON      the python both servers run with
//   DUEL_TOOLS       the folder of metamon_tauros_server.py and recurrent_policy_server.py (also the servers' working directory)
//   TAUROS_METAMON   the patched upstream metamon checkout TaurosV0 runs from (duel/tauros_metamon.patch)
//   TAUROS_CACHE     the metamon cache holding pretrained_models/ (TaurosV0 checkpoint 62)
//   TAUROS_DEVICE    cpu (default) or cuda
// The defaults are the original Mac layout.
pub const BOT2_ROOT = "/Users/macuser/bot2";
const BOT2_PYTHON = BOT2_ROOT ++ "/.venv-metamon/bin/python";
const EXEGGCUTE_SERVER = BOT2_ROOT ++ "/selfplay/tools/metamon_exeggcute_server.py";
const SERVER = BOT2_ROOT ++ "/selfplay/tools/compact_tauros_policy_server.py";
const PYTHON = BOT2_ROOT ++ "/.venv-metamon/bin/python";

fn env(name: [*:0]const u8, default: []const u8) []const u8 {
    const raw = std.c.getenv(name) orelse return default;
    return std.mem.span(raw);
}

fn toolsDir() []const u8 {
    return env("DUEL_TOOLS", BOT2_ROOT ++ "/selfplay/tools");
}

/// Writes `v` as JSON. Floats keep a decimal point so the server reads them as floats.
pub fn writeJson(allocator: std.mem.Allocator, out: *std.ArrayList(u8), v: vs.V) !void {
    switch (v) {
        .null => try out.appendSlice(allocator, "null"),
        .bool => |b| try out.appendSlice(allocator, if (b) "true" else "false"),
        .int => |i| try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{i})),
        .float => |f| {
            const s = try std.fmt.allocPrint(allocator, "{d}", .{f});
            try out.appendSlice(allocator, s);
            if (std.mem.indexOfAny(u8, s, ".eEn") == null) try out.appendSlice(allocator, ".0");
        },
        .str => |s| try writeString(allocator, out, s),
        .list => |items| {
            try out.append(allocator, '[');
            for (items, 0..) |item, i| {
                if (i > 0) try out.append(allocator, ',');
                try writeJson(allocator, out, item);
            }
            try out.append(allocator, ']');
        },
        .obj => |fields| {
            try out.append(allocator, '{');
            for (fields, 0..) |field, i| {
                if (i > 0) try out.append(allocator, ',');
                try writeString(allocator, out, field.key);
                try out.append(allocator, ':');
                try writeJson(allocator, out, field.value);
            }
            try out.append(allocator, '}');
        },
    }
}

fn writeString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0...8, 11, 12, 14...31 => try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "\\u{x:0>4}", .{c})),
            else => try out.append(allocator, c),
        }
    }
    try out.append(allocator, '"');
}

pub const Client = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    child: std.process.Child,
    stdin: std.Io.File,
    reader: std.Io.File.Reader,
    read_buf: []u8,
    /// Battles the server has been reset for; the server keeps per-battle history.
    current_battle: ?u64 = null,
    fallbacks: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, checkpoint: []const u8) !*Client {
        return spawn(allocator, io, &.{ PYTHON, "-u", SERVER, "--checkpoint", checkpoint, "--device", "cpu", "--threads", "1", "--context-limit", "32", "--inference-mode", "full" });
    }

    /// TaurosV0 (bot2's metamon checkpoint 62) through its own snapshot server. Its option names are
    /// the normalized lowercase names (for example "bodyslam" and "switch tauros").
    pub fn initTauros(allocator: std.mem.Allocator, io: std.Io) !*Client {
        const server = try std.fmt.allocPrint(allocator, "{s}/metamon_tauros_server.py", .{toolsDir()});
        return spawn(allocator, io, &.{ env("DUEL_PYTHON", BOT2_PYTHON), "-u", server, "--metamon-root", env("TAUROS_METAMON", BOT2_ROOT ++ "/metamon"), "--cache-dir", env("TAUROS_CACHE", BOT2_ROOT ++ "/metamon-cache"), "--checkpoint", "62", "--device", env("TAUROS_DEVICE", "cpu"), "--threads", "1" });
    }

    /// The small recurrent policy (metamon.recurrent, actor head, argmax over legal options) through
    /// tools/recurrent_policy_server.py, which turns the snapshot into that model's features. Weights
    /// come from RECURRENT_WEIGHTS (default ~/model_weights/weights_best.pt). Same option names as TaurosV0.
    pub fn initRecurrent(allocator: std.mem.Allocator, io: std.Io) !*Client {
        const server = try std.fmt.allocPrint(allocator, "{s}/recurrent_policy_server.py", .{toolsDir()});
        return spawn(allocator, io, &.{ env("DUEL_PYTHON", "/Users/macuser/bot3/.venv/bin/python"), "-u", server });
    }

    /// The Exeggcute ensemble (Metamon's ensemble of gen 1 policies) through its snapshot server; same
    /// option names as TaurosV0.
    pub fn initExeggcute(allocator: std.mem.Allocator, io: std.Io) !*Client {
        return spawn(allocator, io, &.{ BOT2_PYTHON, "-u", EXEGGCUTE_SERVER, "--metamon-root", BOT2_ROOT ++ "/metamon", "--cache-dir", BOT2_ROOT ++ "/metamon-cache", "--checkpoint", "62", "--device", "cpu", "--threads", "1" });
    }

    /// Exeggcute playing the router's decision ("judge") or the argmax of its proposer blend ("blend").
    pub fn initExeggcuteVariant(allocator: std.mem.Allocator, io: std.Io, player: []const u8) !*Client {
        return spawn(allocator, io, &.{ BOT2_PYTHON, "-u", "/Users/macuser/mcts_better_eval/pkmn-search/tools/exeggcute_variant_server.py", "--player", player });
    }

    fn spawn(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !*Client {
        const self = try allocator.create(Client);
        const read_buf = try allocator.alloc(u8, 1 << 16);
        const child = try std.process.spawn(io, .{
            .argv = argv,
            .cwd = .{ .path = toolsDir() },
            .stdin = .pipe,
            .stdout = .pipe,
            .stderr = .inherit,
        });
        self.* = .{
            .allocator = allocator,
            .io = io,
            .child = child,
            .stdin = child.stdin.?,
            .reader = child.stdout.?.reader(io, read_buf),
            .read_buf = read_buf,
        };
        return self;
    }

    pub fn deinit(self: *Client) void {
        self.stdin.close(self.io);
        _ = self.child.wait(self.io) catch {};
    }

    /// Sends one JSON line and returns the reply line (copied out of the read buffer).
    fn call(self: *Client, line: []const u8) ![]u8 {
        try self.stdin.writeStreamingAll(self.io, line);
        try self.stdin.writeStreamingAll(self.io, "\n");
        const reply = (try self.reader.interface.takeDelimiter('\n')) orelse return error.PolicyServerExited;
        return self.allocator.dupe(u8, std.mem.trim(u8, reply, " \r\n"));
    }

    fn reset(self: *Client) !void {
        const reply = try self.call("{\"op\":\"reset\"}");
        defer self.allocator.free(reply);
        if (!std.mem.eql(u8, reply, "ok")) return error.PolicyResetFailed;
    }

    /// Asks the policy for its choice. `snapshot_json` is the observer snapshot with `legal` set;
    /// `battle_id` changes per game so the server's per-battle history is reset.
    /// Returns the reply, which is one of `legal` (or a random legal option on a server error).
    pub fn choose(self: *Client, battle_id: u64, snapshot_json: []const u8, legal: []const []const u8, rand: std.Random) ![]u8 {
        if (self.current_battle == null or self.current_battle.? != battle_id) {
            try self.reset();
            self.current_battle = battle_id;
        }
        const reply = try self.call(snapshot_json);
        for (legal) |option| {
            if (std.mem.eql(u8, option, reply)) return reply;
        }
        self.fallbacks += 1;
        if (self.fallbacks <= 3) std.debug.print("policy fallback: reply {s}\n  snapshot {s}\n", .{ reply, snapshot_json[0..@min(snapshot_json.len, 600)] });
        return self.allocator.dupe(u8, legal[rand.uintLessThan(usize, legal.len)]);
    }
};
