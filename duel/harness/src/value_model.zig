//! Zig port of the Tauros win-probability model (the `value` path of fp_belief). Reads the same
//! `value_weights.bin` + `value_manifest.txt` written by the PyTorch export, and reproduces the
//! reference forward pass: token pooling, group fusion for the state token, five causal temporal
//! blocks, and the win head.

const std = @import("std");
const simd = @import("simd.zig");

pub const GROUPS = 15;
pub const DIM = 256;
/// Actions of the epoch-20 actor: 0 = no-op/fight, 1..3 = moves in sorted-name order, 4..8 = switches.
pub const ACTIONS = 9;

/// The last position of a forward pass: its win logit and its post-norm representation (the actor
/// and win heads both read it).
pub const Forward = struct { logit: f32, rep: [DIM]f32 };
pub const TOKEN_DIM = 64;
pub const HEADS = 4;
pub const HEAD_DIM = DIM / HEADS;
pub const LN_EPS: f32 = 1e-5;
/// Longest window the value model attends over (the trained max_seq_len).
pub const MAX_WINDOW = 32;
/// Temporal layers in the trained model; a KV cache row holds one key and value per layer.
pub const LAYERS = 5;

/// Cached keys and values of one window position, for every temporal layer.
pub const Kv = struct {
    k: [LAYERS][DIM]f32,
    v: [LAYERS][DIM]f32,
};

const Block = struct {
    attn_norm_w: []const f32,
    attn_norm_b: []const f32,
    qkv_w: []const f32,
    qkv_b: []const f32,
    out_w: []const f32,
    out_b: []const f32,
    ff_norm_w: []const f32,
    ff_norm_b: []const f32,
    ff0_w: []const f32,
    ff0_b: []const f32,
    ff3_w: []const f32,
    ff3_b: []const f32,
    hidden: usize,
};

pub const Model = struct {
    vocab_size: usize,
    max_seq_len: usize,
    token_embedding: []f32,
    pool_query: []f32,
    group_projection_w: []f32,
    group_projection_b: []f32,
    group_roles: []f32,
    state_token: []f32,
    position: []f32,
    group_fusion: Block,
    temporal: []Block,
    group_norm_w: []f32,
    group_norm_b: []f32,
    temporal_norm_w: []f32,
    temporal_norm_b: []f32,
    win_norm_w: []f32,
    win_norm_b: []f32,
    win1_w: []f32,
    win1_b: []f32,
    win4_w: []f32,
    win4_b: []f32,
    /// Actor head (Linear DIM -> ACTIONS) from the same checkpoint: `actor_weights.bin`.
    actor_w: []f32,
    actor_b: []f32,

    /// Loads `value_weights.bin` and `value_manifest.txt` from `dir`. All tensors live in `allocator`.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !Model {
        const manifest_path = try std.fmt.allocPrint(allocator, "{s}/value_manifest.txt", .{dir});
        const weights_path = try std.fmt.allocPrint(allocator, "{s}/value_weights.bin", .{dir});
        const manifest = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .unlimited);
        const blob = try std.Io.Dir.cwd().readFileAlloc(io, weights_path, allocator, .unlimited);

        var config = std.StringHashMap(usize).init(allocator);
        var tensors = std.StringHashMap([3]usize).init(allocator);
        var lines = std.mem.splitScalar(u8, manifest, '\n');
        while (lines.next()) |line| {
            var parts = std.mem.tokenizeScalar(u8, line, ' ');
            const kind = parts.next() orelse continue;
            if (std.mem.eql(u8, kind, "config")) {
                const key = parts.next() orelse return error.BadManifest;
                const value = try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10);
                try config.put(key, value);
            } else if (std.mem.eql(u8, kind, "tensor")) {
                const name = parts.next() orelse return error.BadManifest;
                const rows = try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10);
                const cols = try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10);
                const offset = try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10);
                try tensors.put(name, .{ rows, cols, offset });
            } else return error.BadManifest;
        }

        const vocab_size = config.get("vocab_size") orelse return error.BadManifest;
        const max_seq_len = config.get("max_seq_len") orelse return error.BadManifest;
        const temporal_blocks = config.get("temporal_blocks") orelse return error.BadManifest;
        if (temporal_blocks != LAYERS) return error.DimensionMismatch;
        if (config.get("model_dim") != DIM or config.get("token_dim") != TOKEN_DIM or
            config.get("heads") != HEADS or config.get("groups") != GROUPS)
            return error.DimensionMismatch;

        var loader = Loader{ .allocator = allocator, .blob = blob, .tensors = &tensors };
        const temporal = try allocator.alloc(Block, temporal_blocks);
        for (temporal, 0..) |*block, i| {
            const prefix = try std.fmt.allocPrint(allocator, "temporal.{d}", .{i});
            block.* = try loader.block(prefix, 4);
        }
        return .{
            .vocab_size = vocab_size,
            .max_seq_len = max_seq_len,
            .token_embedding = try loader.read("token_embedding.weight", vocab_size * TOKEN_DIM),
            .pool_query = try loader.read("pool_query", TOKEN_DIM),
            .group_projection_w = try loader.read("group_projection.weight", DIM * TOKEN_DIM),
            .group_projection_b = try loader.read("group_projection.bias", DIM),
            .group_roles = try loader.read("group_roles.weight", GROUPS * DIM),
            .state_token = try loader.read("state_token", DIM),
            .position = try loader.read("position_embedding.weight", max_seq_len * DIM),
            .group_fusion = try loader.block("group_fusion", 2),
            .temporal = temporal,
            .group_norm_w = try loader.read("group_norm.weight", DIM),
            .group_norm_b = try loader.read("group_norm.bias", DIM),
            .temporal_norm_w = try loader.read("temporal_norm.weight", DIM),
            .temporal_norm_b = try loader.read("temporal_norm.bias", DIM),
            .win_norm_w = try loader.read("win_head.0.weight", DIM),
            .win_norm_b = try loader.read("win_head.0.bias", DIM),
            .win1_w = try loader.read("win_head.1.weight", (DIM / 2) * DIM),
            .win1_b = try loader.read("win_head.1.bias", DIM / 2),
            .win4_w = try loader.read("win_head.4.weight", DIM / 2),
            .win4_b = try loader.read("win_head.4.bias", 1),
            .actor_w = try readRawF32(allocator, io, dir, "actor_weights.bin", ACTIONS * DIM, 0),
            .actor_b = try readRawF32(allocator, io, dir, "actor_weights.bin", ACTIONS, ACTIONS * DIM * 4),
        };
    }

    /// Actor logits from a representation (the actor head has no norm of its own).
    pub fn actorLogits(self: *const Model, rep: *const [DIM]f32) [ACTIONS]f32 {
        var out: [ACTIONS]f32 = undefined;
        linear(&out, rep, 1, DIM, self.actor_w, ACTIONS, self.actor_b);
        return out;
    }

    /// Step vector (256 floats) for one decision. `groups[g]` holds the token ids of group g;
    /// id 0 is padding.
    pub fn encodeStep(self: *const Model, allocator: std.mem.Allocator, groups: []const []const u32) ![]f32 {
        if (groups.len != GROUPS) return error.BadGroupCount;
        const n = GROUPS + 1;
        const fused = try allocator.alloc(f32, n * DIM);
        defer allocator.free(fused);
        const key_valid = try allocator.alloc(bool, n);
        defer allocator.free(key_valid);

        @memcpy(fused[0..DIM], self.state_token);
        key_valid[0] = true;
        const pooled = try allocator.alloc(f32, TOKEN_DIM);
        defer allocator.free(pooled);
        var widest: usize = 0;
        for (groups) |group| widest = @max(widest, group.len);
        const scores = try allocator.alloc(f32, widest);
        defer allocator.free(scores);
        for (groups, 0..) |group, g| {
            var count: usize = 0;
            var max: f32 = -std.math.inf(f32);
            for (group) |id| {
                if (id == 0) continue;
                if (id >= self.vocab_size) return error.TokenOutOfVocabulary;
                const row = self.token_embedding[id * TOKEN_DIM ..][0..TOKEN_DIM];
                scores[count] = simd.dot(row, self.pool_query);
                max = @max(max, scores[count]);
                count += 1;
            }
            @memset(pooled, 0);
            key_valid[g + 1] = count > 0;
            if (count > 0) {
                var total: f32 = 0;
                for (scores[0..count]) |*s| s.* -= max;
                    simd.expInPlace(scores[0..count]);
                    for (scores[0..count]) |s| total += s;
                var k: usize = 0;
                for (group) |id| {
                    if (id == 0) continue;
                    const w = scores[k] / total;
                    const row = self.token_embedding[id * TOKEN_DIM ..][0..TOKEN_DIM];
                    simd.axpy(pooled, w, row);
                    k += 1;
                }
            }
            // group projection plus the learned role for this group
            const out_row = fused[(g + 1) * DIM ..][0..DIM];
            linear(out_row, pooled, 1, TOKEN_DIM, self.group_projection_w, DIM, self.group_projection_b);
            for (out_row, self.group_roles[g * DIM ..][0..DIM]) |*o, r| o.* += r;
        }

        // group fusion: only the state token (row 0) queries; it attends to valid rows
        const block = self.group_fusion;
        const normed = try allocator.alloc(f32, n * DIM);
        defer allocator.free(normed);
        layerNorm(normed, fused, n, DIM, block.attn_norm_w, block.attn_norm_b);
        const qkv = try allocator.alloc(f32, n * 3 * DIM);
        defer allocator.free(qkv);
        linear(qkv, normed, n, DIM, block.qkv_w, 3 * DIM, block.qkv_b);
        const attended = try allocator.alloc(f32, DIM);
        defer allocator.free(attended);
        @memset(attended, 0);
        const scale = 1.0 / @sqrt(@as(f32, HEAD_DIM));
        for (0..HEADS) |head| {
            const lo = head * HEAD_DIM;
            var logits: [GROUPS + 1]f32 = undefined;
            var keys: [GROUPS + 1]usize = undefined;
            var count: usize = 0;
            var max: f32 = -std.math.inf(f32);
            for (0..n) |j| {
                if (!key_valid[j]) continue;
                const k = qkv[j * 3 * DIM + DIM + lo ..][0..HEAD_DIM];
                logits[count] = simd.dot(qkv[lo..][0..HEAD_DIM], k) * scale;
                max = @max(max, logits[count]);
                keys[count] = j;
                count += 1;
            }
            var total: f32 = 0;
            for (logits[0..count]) |*l| l.* -= max;
                simd.expInPlace(logits[0..count]);
                for (logits[0..count]) |l| total += l;
            for (0..count) |c| {
                const w = logits[c] / total;
                const v = qkv[keys[c] * 3 * DIM + 2 * DIM + lo ..][0..HEAD_DIM];
                for (attended[lo..][0..HEAD_DIM], v) |*a, x| a.* += w * x;
            }
        }
        const projected = try allocator.alloc(f32, DIM);
        defer allocator.free(projected);
        linear(projected, attended, 1, DIM, block.out_w, DIM, block.out_b);
        const x = try allocator.alloc(f32, DIM);
        defer allocator.free(x);
        for (x, fused[0..DIM], projected) |*xi, f, p| xi.* = f + p;
        try feedForward(allocator, block, x);
        const out = try allocator.alloc(f32, DIM);
        layerNorm(out, x, 1, DIM, self.group_norm_w, self.group_norm_b);
        return out;
    }

    /// Win logit of the last step of a window of step vectors (oldest first, at most
    /// `max_seq_len` of them): causal attention over the whole window.
    pub fn winLogit(self: *const Model, allocator: std.mem.Allocator, steps: []const []const f32) !f32 {
        return (try self.windowForwardState(allocator, steps, null)).logit;
    }

    /// As `winLogit`; when `rows` is given (one entry per step) it also stores each position's
    /// keys and values for every layer, which later positions read through `tokenForward`.
    pub fn windowForward(self: *const Model, allocator: std.mem.Allocator, steps: []const []const f32, rows: ?[]Kv) !f32 {
        return (try self.windowForwardState(allocator, steps, rows)).logit;
    }

    /// As `windowForward`, also returning the last position's representation.
    pub fn windowForwardState(self: *const Model, allocator: std.mem.Allocator, steps: []const []const f32, rows: ?[]Kv) !Forward {
        const t_len = steps.len;
        if (t_len == 0 or t_len > self.max_seq_len) return error.BadWindowLength;
        if (rows) |r| if (r.len != t_len) return error.BadWindowLength;
        const x = try allocator.alloc(f32, t_len * DIM);
        defer allocator.free(x);
        for (steps, 0..) |step, t| {
            if (step.len != DIM) return error.BadStepLength;
            for (x[t * DIM ..][0..DIM], step, self.position[t * DIM ..][0..DIM]) |*xi, s_, p| xi.* = s_ + p;
        }
        const h = try allocator.alloc(f32, t_len * DIM);
        defer allocator.free(h);
        const qkv = try allocator.alloc(f32, t_len * 3 * DIM);
        defer allocator.free(qkv);
        const attended = try allocator.alloc(f32, t_len * DIM);
        defer allocator.free(attended);
        const projected = try allocator.alloc(f32, t_len * DIM);
        defer allocator.free(projected);
        const scale = 1.0 / @sqrt(@as(f32, HEAD_DIM));
        const logits = try allocator.alloc(f32, t_len);
        defer allocator.free(logits);

        for (self.temporal, 0..) |block, layer| {
            layerNorm(h, x, t_len, DIM, block.attn_norm_w, block.attn_norm_b);
            linear(qkv, h, t_len, DIM, block.qkv_w, 3 * DIM, block.qkv_b);
            if (rows) |r| for (0..t_len) |t| {
                @memcpy(r[t].k[layer][0..DIM], qkv[t * 3 * DIM + DIM ..][0..DIM]);
                @memcpy(r[t].v[layer][0..DIM], qkv[t * 3 * DIM + 2 * DIM ..][0..DIM]);
            };
            @memset(attended, 0);
            for (0..t_len) |t| {
                for (0..HEADS) |head| {
                    const lo = head * HEAD_DIM;
                    const q = qkv[t * 3 * DIM + lo ..][0..HEAD_DIM];
                    var max: f32 = -std.math.inf(f32);
                    for (0..t + 1) |j| {
                        const k = qkv[j * 3 * DIM + DIM + lo ..][0..HEAD_DIM];
                        logits[j] = simd.dot(q, k) * scale;
                        max = @max(max, logits[j]);
                    }
                    var total: f32 = 0;
                    for (logits[0 .. t + 1]) |*l| l.* -= max;
                        simd.expInPlace(logits[0 .. t + 1]);
                        for (logits[0 .. t + 1]) |l| total += l;
                    for (0..t + 1) |j| {
                        const w = logits[j] / total;
                        const v = qkv[j * 3 * DIM + 2 * DIM + lo ..][0..HEAD_DIM];
                        for (attended[t * DIM + lo ..][0..HEAD_DIM], v) |*a, vv| a.* += w * vv;
                    }
                }
            }
            linear(projected, attended, t_len, DIM, block.out_w, DIM, block.out_b);
            for (x, projected) |*xi, p| xi.* += p;
            try feedForwardRows(allocator, block, x, t_len);
        }
        return self.winHead(allocator, x, t_len);
    }

    /// The win head on the last of `t_len` rows of `x` (after the temporal stack).
    fn winHead(self: *const Model, allocator: std.mem.Allocator, x: []const f32, t_len: usize) !Forward {
        const rep = try allocator.alloc(f32, t_len * DIM);
        defer allocator.free(rep);
        layerNorm(rep, x, t_len, DIM, self.temporal_norm_w, self.temporal_norm_b);
        const last = rep[(t_len - 1) * DIM ..][0..DIM];
        var rep_last: [DIM]f32 = undefined;
        @memcpy(&rep_last, last);
        const head_in = try allocator.alloc(f32, DIM);
        defer allocator.free(head_in);
        layerNorm(head_in, last, 1, DIM, self.win_norm_w, self.win_norm_b);
        const hidden = try allocator.alloc(f32, DIM / 2);
        defer allocator.free(hidden);
        linear(hidden, head_in, 1, DIM, self.win1_w, DIM / 2, self.win1_b);
        simd.geluInPlace(hidden);
        var out: [1]f32 = undefined;
        linear(&out, hidden, 1, DIM / 2, self.win4_w, 1, self.win4_b);
        return .{ .logit = out[0], .rep = rep_last };
    }

    /// One new position at index `pos`, attending to `prefix` (cached rows of the earlier
    /// positions, oldest first) and to itself. Writes this position's keys and values to `out`
    /// and returns the win logit. Equal to `windowForward` over the whole window.
    pub fn tokenForward(self: *const Model, allocator: std.mem.Allocator, step: []const f32, pos: usize, prefix: []const *const Kv, out: *Kv) !f32 {
        return (try self.tokenForwardState(allocator, step, pos, prefix, out)).logit;
    }

    /// As `tokenForward`, also returning the new position's representation.
    pub fn tokenForwardState(self: *const Model, allocator: std.mem.Allocator, step: []const f32, pos: usize, prefix: []const *const Kv, out: *Kv) !Forward {
        if (pos >= self.max_seq_len or pos != prefix.len) return error.BadWindowLength;
        const x = try allocator.alloc(f32, DIM);
        defer allocator.free(x);
        for (x, step, self.position[pos * DIM ..][0..DIM]) |*xi, s_, p| xi.* = s_ + p;
        const h = try allocator.alloc(f32, DIM);
        defer allocator.free(h);
        const qkv = try allocator.alloc(f32, 3 * DIM);
        defer allocator.free(qkv);
        const attended = try allocator.alloc(f32, DIM);
        defer allocator.free(attended);
        const projected = try allocator.alloc(f32, DIM);
        defer allocator.free(projected);
        const logits = try allocator.alloc(f32, prefix.len + 1);
        defer allocator.free(logits);
        const scale = 1.0 / @sqrt(@as(f32, HEAD_DIM));

        for (self.temporal, 0..) |block, layer| {
            layerNorm(h, x, 1, DIM, block.attn_norm_w, block.attn_norm_b);
            linear(qkv, h, 1, DIM, block.qkv_w, 3 * DIM, block.qkv_b);
            @memcpy(out.k[layer][0..DIM], qkv[DIM .. 2 * DIM]);
            @memcpy(out.v[layer][0..DIM], qkv[2 * DIM .. 3 * DIM]);
            @memset(attended, 0);
            for (0..HEADS) |head| {
                const lo = head * HEAD_DIM;
                const q = qkv[lo..][0..HEAD_DIM];
                var max: f32 = -std.math.inf(f32);
                for (prefix, 0..) |row, j| {
                    logits[j] = simd.dot(q, row.k[layer][lo..][0..HEAD_DIM]) * scale;
                    max = @max(max, logits[j]);
                }
                logits[prefix.len] = simd.dot(q, out.k[layer][lo..][0..HEAD_DIM]) * scale;
                max = @max(max, logits[prefix.len]);
                var total: f32 = 0;
                for (logits) |*l| l.* -= max;
                    simd.expInPlace(logits);
                    for (logits) |l| total += l;
                for (prefix, 0..) |row, j| {
                    const w = logits[j] / total;
                    for (attended[lo..][0..HEAD_DIM], row.v[layer][lo..][0..HEAD_DIM]) |*a, v| a.* += w * v;
                }
                const w_self = logits[prefix.len] / total;
                for (attended[lo..][0..HEAD_DIM], out.v[layer][lo..][0..HEAD_DIM]) |*a, v| a.* += w_self * v;
            }
            linear(projected, attended, 1, DIM, block.out_w, DIM, block.out_b);
            for (x, projected) |*xi, p| xi.* += p;
            try feedForwardRows(allocator, block, x, 1);
        }
        return self.winHead(allocator, x, 1);
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        allocator.free(self.temporal);
        self.* = undefined;
    }
};

/// Residual feed-forward applied in place to the `DIM`-wide row `x` (group fusion, one row).
fn feedForward(allocator: std.mem.Allocator, block: Block, x: []f32) !void {
    try feedForwardRows(allocator, block, x, 1);
}

/// Pre-norm feed-forward residual on `rows` rows of `x` in place.
fn feedForwardRows(allocator: std.mem.Allocator, block: Block, x: []f32, rows: usize) !void {
    const ff_in = try allocator.alloc(f32, rows * DIM);
    defer allocator.free(ff_in);
    layerNorm(ff_in, x, rows, DIM, block.ff_norm_w, block.ff_norm_b);
    const hidden = try allocator.alloc(f32, rows * block.hidden);
    defer allocator.free(hidden);
    linear(hidden, ff_in, rows, DIM, block.ff0_w, block.hidden, block.ff0_b);
    simd.geluInPlace(hidden);
    const out = try allocator.alloc(f32, rows * DIM);
    defer allocator.free(out);
    linear(out, hidden, rows, block.hidden, block.ff3_w, DIM, block.ff3_b);
    for (x, out) |*xi, o| xi.* += o;
}

const cblas = struct {
    const RowMajor = 101;
    const NoTrans = 111;
    const Trans = 112;
    extern fn cblas_sgemm(order: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) void;
};

/// out[r, o] = sum_i x[r, i] * w[o, i] + b[o]  (weights stored as [out, in], like PyTorch).
/// Uses the platform BLAS (Accelerate on macOS) for the product.
fn linear(out: []f32, x: []const f32, rows: usize, in_dim: usize, w: []const f32, out_dim: usize, b: []const f32) void {
    // C (rows x out) = X (rows x in) * W^T, with W stored as (out x in) row-major
    if (@import("builtin").os.tag == .macos) {
        cblas.cblas_sgemm(cblas.RowMajor, cblas.NoTrans, cblas.Trans, @intCast(rows), @intCast(out_dim), @intCast(in_dim), 1.0, x.ptr, @intCast(in_dim), w.ptr, @intCast(in_dim), 0.0, out.ptr, @intCast(out_dim));
    } else {
        // (repo copy) no system BLAS is linked elsewhere (Linux): a plain product. Only the search's value model uses it, not the duel
        for (0..rows) |r| {
            const xr = x[r * in_dim ..][0..in_dim];
            for (0..out_dim) |o| {
                const wo = w[o * in_dim ..][0..in_dim];
                var acc: f32 = 0;
                for (xr, wo) |xi, wi| acc += xi * wi;
                out[r * out_dim + o] = acc;
            }
        }
    }
    for (0..rows) |r| {
        for (out[r * out_dim ..][0..out_dim], b[0..out_dim]) |*o, bias| o.* += bias;
    }
}

fn layerNorm(out: []f32, x: []const f32, rows: usize, width: usize, w: []const f32, b: []const f32) void {
    if (width % simd.W == 0) return simd.layerNorm(out, x, rows, width, w, b, LN_EPS);
    for (0..rows) |r| {
        const row = x[r * width ..][0..width];
        var mean: f32 = 0;
        for (row) |v| mean += v;
        mean /= @floatFromInt(width);
        var variance: f32 = 0;
        for (row) |v| variance += (v - mean) * (v - mean);
        variance /= @floatFromInt(width);
        const inv = 1.0 / @sqrt(variance + LN_EPS);
        for (0..width) |i| {
            out[r * width + i] = (row[i] - mean) * inv * w[i] + b[i];
        }
    }
}

fn dot(a: []const f32, b: []const f32) f32 {
    var s: f32 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}

/// Exact GELU, as torch.nn.GELU() (erf form).
pub fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x * std.math.sqrt1_2));
}

/// Abramowitz & Stegun 7.1.26, the same approximation the Rust implementation uses.
fn erf(x: f32) f32 {
    const sign: f32 = if (x < 0) -1 else 1;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign * (1.0 - poly * @exp(-ax * ax));
}

/// Reads `count` float32 values at byte `offset` of a raw weights file in `dir`.
fn readRawF32(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, file: []const u8, count: usize, offset: usize) ![]f32 {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, file });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    if (bytes.len < offset + count * 4) return error.BadWeightsFile;
    const out = try allocator.alloc(f32, count);
    @memcpy(std.mem.sliceAsBytes(out), bytes[offset..][0 .. count * 4]);
    return out;
}

/// Loads tensors by name from the flat float32 blob, using the manifest's element offsets.
const Loader = struct {
    allocator: std.mem.Allocator,
    blob: []const u8,
    tensors: *std.StringHashMap([3]usize),

    fn read(self: *Loader, name: []const u8, expected: usize) ![]f32 {
        const entry = self.tensors.get(name) orelse return error.MissingTensor;
        const len = entry[0] * entry[1];
        if (len != expected) return error.TensorShapeMismatch;
        const start = entry[2] * 4;
        if (start + len * 4 > self.blob.len) return error.WeightsTooShort;
        const out = try self.allocator.alloc(f32, len);
        for (out, 0..) |*v, i| {
            const bytes = self.blob[start + i * 4 ..][0..4];
            v.* = @bitCast(std.mem.readInt(u32, bytes, .little));
        }
        return out;
    }

    fn block(self: *Loader, prefix: []const u8, hidden_factor: usize) !Block {
        const hidden = hidden_factor * DIM;
        const name = struct {
            fn get(a: std.mem.Allocator, p: []const u8, s: []const u8) ![]u8 {
                return std.fmt.allocPrint(a, "{s}.{s}", .{ p, s });
            }
        }.get;
        return .{
            .attn_norm_w = try self.read(try name(self.allocator, prefix, "attn_norm.weight"), DIM),
            .attn_norm_b = try self.read(try name(self.allocator, prefix, "attn_norm.bias"), DIM),
            .qkv_w = try self.read(try name(self.allocator, prefix, "qkv.weight"), 3 * DIM * DIM),
            .qkv_b = try self.read(try name(self.allocator, prefix, "qkv.bias"), 3 * DIM),
            .out_w = try self.read(try name(self.allocator, prefix, "attn_out.weight"), DIM * DIM),
            .out_b = try self.read(try name(self.allocator, prefix, "attn_out.bias"), DIM),
            .ff_norm_w = try self.read(try name(self.allocator, prefix, "ff_norm.weight"), DIM),
            .ff_norm_b = try self.read(try name(self.allocator, prefix, "ff_norm.bias"), DIM),
            .ff0_w = try self.read(try name(self.allocator, prefix, "ff.0.weight"), hidden * DIM),
            .ff0_b = try self.read(try name(self.allocator, prefix, "ff.0.bias"), hidden),
            .ff3_w = try self.read(try name(self.allocator, prefix, "ff.3.weight"), DIM * hidden),
            .ff3_b = try self.read(try name(self.allocator, prefix, "ff.3.bias"), DIM),
            .hidden = hidden,
        };
    }
};
