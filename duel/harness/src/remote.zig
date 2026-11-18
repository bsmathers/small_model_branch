//! The recurrent model (metamon.recurrent) for the search, through tools/recurrent_search_server.py: a subprocess that keeps
//! one node per player view (the policy server's game memory and recurrent state) and answers each new node's win logit
//! and the logits of its legal options. path.zig uses it when `active` is set (the tournament's SEARCH_MODEL=recurrent).
//!
//! Environment: DUEL_PYTHON (the python), DUEL_TOOLS (the server's folder), RECURRENT_SEARCH_WEIGHTS or RECURRENT_WEIGHTS.

const std = @import("std");
const pkmn = @import("pkmn");

pub const MAXC = pkmn.MAX_CHOICES;
/// Logits at or below this mark options without a policy slot.
pub const UNMAPPED: f32 = -1e29;

fn env(name: [*:0]const u8, default: []const u8) []const u8 {
    const raw = std.c.getenv(name) orelse return default;
    return std.mem.span(raw);
}

pub const Reply = struct {
    id: i64,
    win_logit: f32,
    /// One logit per legal option, in the order the snapshot listed them.
    logits: [MAXC]f32 = @splat(UNMAPPED),
    n: usize = 0,
};

pub const Remote = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    child: std.process.Child,
    stdin: std.Io.File,
    reader: std.Io.File.Reader,
    read_buf: []u8,
    out: std.ArrayList(u8) = .empty,
    /// Requests sent (each is one node, or one chain).
    calls: u64 = 0,
    steps: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !*Remote {
        const tools = env("DUEL_TOOLS", "../tools");
        const server = try std.fmt.allocPrint(allocator, "{s}/recurrent_search_server.py", .{tools});
        const read_buf = try allocator.alloc(u8, 1 << 16);
        const child = try std.process.spawn(io, .{
            .argv = &.{ env("DUEL_PYTHON", "python3"), "-u", server },
            .cwd = .{ .path = tools },
            .stdin = .pipe,
            .stdout = .pipe,
            .stderr = .inherit,
        });
        const self = try allocator.create(Remote);
        self.* = .{ .allocator = allocator, .io = io, .child = child, .stdin = child.stdin.?, .reader = child.stdout.?.reader(io, read_buf), .read_buf = read_buf };
        return self;
    }

    pub fn deinit(self: *Remote) void {
        self.stdin.close(self.io);
        _ = self.child.wait(self.io) catch {};
    }

    /// Sends `line` and returns the reply line (valid until the next call).
    fn call(self: *Remote, line: []const u8) ![]const u8 {
        self.calls += 1;
        try self.stdin.writeStreamingAll(self.io, line);
        try self.stdin.writeStreamingAll(self.io, "\n");
        const reply = (try self.reader.interface.takeDelimiter('\n')) orelse return error.SearchServerExited;
        const trimmed = std.mem.trim(u8, reply, " \r\n");
        if (std.mem.startsWith(u8, trimmed, "!error")) {
            std.debug.print("recurrent search server: {s}\n", .{trimmed});
            return error.SearchServerError;
        }
        return trimmed;
    }

    fn expectOk(self: *Remote, line: []const u8) !void {
        const reply = try self.call(line);
        if (!std.mem.eql(u8, reply, "ok")) return error.SearchServerError;
    }

    pub fn reset(self: *Remote) !void {
        try self.expectOk("{\"op\":\"reset\"}");
    }

    /// The id the next node will get: `release` with it forgets every node made after this call.
    pub fn mark(self: *Remote) !i64 {
        return std.fmt.parseInt(i64, try self.call("{\"op\":\"mark\"}"), 10);
    }

    pub fn release(self: *Remote, m: i64) !void {
        var buf: [64]u8 = undefined;
        try self.expectOk(try std.fmt.bufPrint(&buf, "{{\"op\":\"release\",\"mark\":{d}}}", .{m}));
    }

    /// A new node: `parent` (-1 for a game's first decision) after its player chose `sent`, then `snap_json` (a duel snapshot).
    pub fn step(self: *Remote, parent: i64, sent: []const u8, snap_json: []const u8) !Reply {
        self.out.clearRetainingCapacity();
        const w = self.allocator;
        var head: [256]u8 = undefined;
        try self.out.appendSlice(w, try std.fmt.bufPrint(&head, "{{\"op\":\"step\",\"parent\":{d},\"sent\":\"{s}\",\"snap\":", .{ parent, sent }));
        try self.out.appendSlice(w, snap_json);
        try self.out.append(w, '}');
        self.steps += 1;
        return parseReply(try self.call(self.out.items));
    }

    /// Independent steps in one request (the server runs their model steps as one batch).
    pub fn batch(self: *Remote, allocator: std.mem.Allocator, items: []const Item) ![]Reply {
        return batchReplies(self, allocator, items);
    }

    /// A step that also returns the server's model input: (the reply, the features' JSON), both valid until the next call.
    pub fn explain(self: *Remote, parent: i64, sent: []const u8, snap_json: []const u8) !struct { Reply, []const u8 } {
        self.out.clearRetainingCapacity();
        const w = self.allocator;
        var head: [256]u8 = undefined;
        try self.out.appendSlice(w, try std.fmt.bufPrint(&head, "{{\"op\":\"explain\",\"parent\":{d},\"sent\":\"{s}\",\"snap\":", .{ parent, sent }));
        try self.out.appendSlice(w, snap_json);
        try self.out.append(w, '}');
        self.steps += 1;
        const line = try self.call(self.out.items);
        const tab = std.mem.indexOfScalar(u8, line, '\t') orelse return error.BadReply;
        return .{ try parseReply(line[0..tab]), line[tab + 1 ..] };
    }

    /// Several steps in one request (`sents[i]` is what the player chose before `snaps[i]`); returns the last node.
    pub fn chain(self: *Remote, parent: i64, sents: []const []const u8, snaps: []const []const u8) !Reply {
        self.out.clearRetainingCapacity();
        const w = self.allocator;
        var head: [256]u8 = undefined;
        try self.out.appendSlice(w, try std.fmt.bufPrint(&head, "{{\"op\":\"chain\",\"parent\":{d},\"steps\":[", .{parent}));
        for (sents, snaps, 0..) |sent, snap, i| {
            if (i > 0) try self.out.append(w, ',');
            try self.out.appendSlice(w, try std.fmt.bufPrint(&head, "{{\"sent\":\"{s}\",\"snap\":", .{sent}));
            try self.out.appendSlice(w, snap);
            try self.out.append(w, '}');
        }
        try self.out.appendSlice(w, "]}");
        self.steps += snaps.len;
        return parseReply(try self.call(self.out.items));
    }
};

/// One step of a batch: a new node from `parent` (an existing node) after its player chose `sent`, then `snap`.
pub const Item = struct { parent: i64, sent: []const u8, snap: []const u8 };

pub fn batchReplies(self: *Remote, allocator: std.mem.Allocator, items: []const Item) ![]Reply {
    const out = try allocator.alloc(Reply, items.len);
    if (items.len == 0) return out;
    self.out.clearRetainingCapacity();
    const w = self.allocator;
    try self.out.appendSlice(w, "{\"op\":\"batch\",\"items\":[");
    var head: [256]u8 = undefined;
    for (items, 0..) |item, i| {
        if (i > 0) try self.out.append(w, ',');
        try self.out.appendSlice(w, try std.fmt.bufPrint(&head, "{{\"parent\":{d},\"sent\":\"{s}\",\"snap\":", .{ item.parent, item.sent }));
        try self.out.appendSlice(w, item.snap);
        try self.out.append(w, '}');
    }
    try self.out.appendSlice(w, "]}");
    self.steps += items.len;
    const line = try self.call(self.out.items);
    var parts = std.mem.splitSequence(u8, line, " ; ");
    for (out) |*r| r.* = try parseReply(parts.next() orelse return error.BadReply);
    if (parts.next() != null) return error.BadReply;
    return out;
}

fn parseReply(line: []const u8) !Reply {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    var r = Reply{ .id = try std.fmt.parseInt(i64, it.next() orelse return error.BadReply, 10), .win_logit = try std.fmt.parseFloat(f32, it.next() orelse return error.BadReply) };
    while (it.next()) |tok| {
        if (r.n >= MAXC) return error.BadReply;
        r.logits[r.n] = try std.fmt.parseFloat(f32, tok);
        r.n += 1;
    }
    return r;
}

/// The search's recurrent model, when one is in use.
pub var active: ?*Remote = null;
