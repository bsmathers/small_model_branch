//! Epoch-20 actor distributions for the search, served by tools/policy_priors.py. Each perspective
//! (p1 = our side, p2 = the opponent) has its own engine history on the server.

const std = @import("std");

const PYTHON = "/Users/macuser/bot2/.venv-metamon/bin/python";
const SCRIPT = "tools/policy_priors.py";

pub const Decision = struct {
    /// the option the actor would play (a name from the legal list)
    action: []const u8,
    /// probabilities aligned with the legal list passed to `decide`
    probs: []f32,
};

pub const Server = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    child: std.process.Child,
    stdin: std.Io.File,
    reader: std.Io.File.Reader,
    read_buf: []u8,
    errors: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !*Server {
        const self = try allocator.create(Server);
        const read_buf = try allocator.alloc(u8, 1 << 16);
        const child = try std.process.spawn(io, .{
            .argv = &.{ PYTHON, "-u", SCRIPT },
            .cwd = .{ .path = "/Users/macuser/mcts_better_eval/pkmn-search" },
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
        const ready = try self.readLine();
        defer allocator.free(ready);
        if (!std.mem.eql(u8, ready, "ready")) return error.PriorServerNotReady;
        return self;
    }

    pub fn deinit(self: *Server) void {
        self.stdin.close(self.io);
        _ = self.child.wait(self.io) catch {};
    }

    fn readLine(self: *Server) ![]u8 {
        const line = (try self.reader.interface.takeDelimiter('\n')) orelse return error.PriorServerExited;
        return self.allocator.dupe(u8, std.mem.trim(u8, line, " \r\n"));
    }

    fn send(self: *Server, line: []const u8) !void {
        try self.stdin.writeStreamingAll(self.io, line);
        try self.stdin.writeStreamingAll(self.io, "\n");
    }

    pub fn reset(self: *Server, persp: []const u8) !void {
        const req = try std.fmt.allocPrint(self.allocator, "{{\"op\":\"reset\",\"persp\":\"{s}\"}}", .{persp});
        try self.send(req);
        const reply = try self.readLine();
        if (!std.mem.eql(u8, reply, "ok")) return error.PriorResetFailed;
    }

    /// Asks the actor for `persp` (p1 or p2). `snapshot_json` is the observer snapshot with
    /// `legal` set; `legal` lists the option names in the order `probs` is returned in.
    pub fn decide(self: *Server, persp: []const u8, snapshot_json: []const u8, legal: []const []const u8, probs: []f32) !?[]const u8 {
        const req = try std.fmt.allocPrint(self.allocator, "{{\"op\":\"decide\",\"persp\":\"{s}\",\"snapshot\":{s}}}", .{ persp, snapshot_json });
        try self.send(req);
        const reply = try self.readLine();
        if (std.mem.startsWith(u8, reply, "!error")) {
            self.errors += 1;
            std.debug.print("prior server {s}: {s}\n", .{ persp, reply });
            return null;
        }
        var fields = std.mem.splitScalar(u8, reply, '\t');
        const action = fields.next() orelse return null;
        @memset(probs, 0);
        while (fields.next()) |field| {
            const colon = std.mem.lastIndexOfScalar(u8, field, ':') orelse continue;
            const name = field[0..colon];
            const p = std.fmt.parseFloat(f32, field[colon + 1 ..]) catch continue;
            for (legal, 0..) |option, i| {
                if (std.mem.eql(u8, option, name)) probs[i] = p;
            }
        }
        return action;
    }
};
