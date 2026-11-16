//! Zig port of the belief tokenizer (fp_belief `tokenizer.rs`): public-state JSON frames become
//! token strings ("path=value"), then token ids from the vocabulary.

const std = @import("std");
const belief = @import("belief.zig");

const Json = std.json.Value;

/// Python's `str()` for a scalar: None/True/False, integers, and floats written the way Rust's
/// Display prints them (whole floats without a fraction).
fn scalarText(allocator: std.mem.Allocator, value: Json) ![]const u8 {
    return switch (value) {
        .null => "None",
        .bool => |b| if (b) "True" else "False",
        .integer => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| if (f == @trunc(f) and @abs(f) < 1e15)
            try std.fmt.allocPrint(allocator, "{d}", .{@as(i64, @intFromFloat(f))})
        else
            try std.fmt.allocPrint(allocator, "{d}", .{f}),
        .string => |s| s,
        .number_string => |s| s,
        .array, .object => "",
    };
}

/// Appends the tokens for one JSON value, as `public_tokens` does: object keys in sorted order
/// (skipping "frame"), list items under "[]", and hp_pct rounded into 21 buckets.
pub fn publicTokens(allocator: std.mem.Allocator, value: Json, prefix: []const u8, out: *std.ArrayList([]const u8)) !void {
    switch (value) {
        .object => |map| {
            var keys: std.ArrayList([]const u8) = .empty;
            var it = map.iterator();
            while (it.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, "frame")) continue;
                try keys.append(allocator, entry.key_ptr.*);
            }
            std.mem.sort([]const u8, keys.items, {}, lessString);
            for (keys.items) |key| {
                const child_prefix = if (prefix.len == 0)
                    try allocator.dupe(u8, key)
                else
                    try std.fmt.allocPrint(allocator, "{s}.{s}", .{ prefix, key });
                try publicTokens(allocator, map.get(key).?, child_prefix, out);
            }
        },
        .array => |items| {
            const child_prefix = try std.fmt.allocPrint(allocator, "{s}[]", .{prefix});
            for (items.items) |item| try publicTokens(allocator, item, child_prefix, out);
        },
        .integer, .float => {
            if (std.mem.endsWith(u8, prefix, "hp_pct")) {
                const v: f64 = switch (value) {
                    .integer => |i| @floatFromInt(i),
                    .float => |f| f,
                    else => unreachable,
                };
                const bucket: i64 = @intFromFloat(std.math.clamp(@round(v * 20.0), 0.0, 20.0));
                try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={d}", .{ prefix, bucket }));
            } else {
                try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={s}", .{ prefix, try scalarText(allocator, value) }));
            }
        },
        else => try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}={s}", .{ prefix, try scalarText(allocator, value) })),
    }
}

fn lessString(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// One recorded turn as JSON: the public state plus the actions chosen from it.
pub const FrameJson = struct {
    public_state: Json,
    own_action: Json,
    opponent_action: Json,
};

/// Token ids for each frame, as `causal_frame_tokens` builds them: frame i also carries the
/// previous frame's actions.
pub fn encodeFrames(allocator: std.mem.Allocator, vocab: *const belief.Vocab, frames: []const FrameJson) ![][]u32 {
    const encoded = try allocator.alloc([]u32, frames.len);
    for (frames, 0..) |frame, index| {
        var tokens: std.ArrayList([]const u8) = .empty;
        try publicTokens(allocator, frame.public_state, "", &tokens);
        if (index > 0) {
            const previous = frames[index - 1];
            try tokens.append(allocator, try std.fmt.allocPrint(allocator, "previous.own_action={s}", .{try scalarText(allocator, previous.own_action)}));
            try tokens.append(allocator, try std.fmt.allocPrint(allocator, "previous.opponent_action={s}", .{try scalarText(allocator, previous.opponent_action)}));
        }
        var ids: std.ArrayList(u32) = .empty;
        for (tokens.items) |t| try ids.append(allocator, vocab.tokenId(t));
        if (ids.items.len == 0) try ids.append(allocator, belief.UNK_ID);
        encoded[index] = ids.items;
    }
    return encoded;
}
