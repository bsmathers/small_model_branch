//! The recurrent model (metamon/recurrent/model.py, PolicyValueModel) in Zig, for the search's native backend. Weights come
//! from tools/export_recurrent.py: model.bin (float32 tensors) and model.json (their names, shapes and offsets).
//!
//! One step reads 15 tokens: the carried state `h`, the last turn, six own slots, six opponent slots and the field, through four
//! pre-norm transformer blocks (6-head unmasked attention, GELU MLP). The state token's output is the next `h`, and the input of
//! the actor head (Linear 192 -> 9, plus the optional deep actor) and the win head (LayerNorm -> 192x64 -> GELU -> 64x1).

const std = @import("std");
const simd = @import("simd.zig");
const build_options = @import("build_options");
/// A cblas_sgemm is linked: Accelerate on macOS, OpenBLAS with -Dopenblas=true.
const have_blas = build_options.have_blas;

pub const D = 192;
pub const HEADS = 6;
pub const HD = D / HEADS;
pub const FF = 384;
pub const LAYERS = 4;
pub const TOKENS = 15;
pub const ACTIONS = 9;
pub const EMB = 16;
pub const SEMB = 4;
pub const N_SPECIES = 152;
pub const N_MOVES = 168;
pub const N_STATUS = 8;
const OWN_W = 94;
const OPP_W = 91;
const LAST_W = 106;
const FIELD_W = 39;
const LN_EPS: f32 = 1e-5;

/// One decision's input (features.encode_observer's arrays).
pub const Input = struct {
    own_species: [6]u32 = @splat(0),
    own_status: [6]u32 = @splat(0),
    own_moves: [6][4]u32 = @splat(@splat(0)),
    opp_species: [6]u32 = @splat(0),
    opp_status: [6]u32 = @splat(0),
    opp_moves: [6][4]u32 = @splat(@splat(0)),
    last_own_move: u32 = 0,
    last_opp_move: u32 = 0,
    last_own_species: u32 = 0,
    last_opp_species: u32 = 0,
    last_own_status: u32 = 0,
    last_opp_status: u32 = 0,
    own: [6][10]f32 = @splat(@splat(0)),
    opp: [6][7]f32 = @splat(@splat(0)),
    last: [34]f32 = @splat(0),
    field: [FIELD_W]f32 = @splat(0),
};

const Linear = struct { w: []const f32, b: []const f32, in: usize, out: usize };
const Norm = struct { w: []const f32, b: []const f32 };
const Block = struct { norm1: Norm, qkv: Linear, proj: Linear, norm2: Norm, ff1: Linear, ff2: Linear };

pub const Model = struct {
    species: []const f32,
    moves: []const f32,
    status: []const f32,
    role: []const f32,
    state_in: Linear,
    last_in: Linear,
    own_in: Linear,
    opp_in: Linear,
    field_in: Linear,
    blocks: [LAYERS]Block,
    final_norm: Norm,
    policy: Linear,
    win_norm: Norm,
    win1: Linear,
    win2: Linear,
    /// The deep actor head (model.DeepActor: LayerNorm, 192 -> w -> w -> 9), when the weights have one.
    deep: ?struct { norm: Norm, l1: Linear, l2: Linear, l3: Linear } = null,
    weights_sha256: []const u8 = "",

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !Model {
        const json_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(allocator, "{s}/model.json", .{dir}), allocator, .unlimited);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(allocator, "{s}/model.bin", .{dir}), allocator, .unlimited);
        const meta = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_text, .{});
        const floats = try allocator.alloc(f32, bytes.len / 4);
        @memcpy(std.mem.sliceAsBytes(floats), bytes[0 .. floats.len * 4]);
        var map = std.StringHashMap([]const f32).init(allocator);
        var shapes = std.StringHashMap([2]usize).init(allocator);
        for (meta.object.get("tensors").?.array.items) |t| {
            const name = t.object.get("name").?.string;
            const shape = t.object.get("shape").?.array.items;
            var n: usize = 1;
            for (shape) |s| n *= @intCast(s.integer);
            const off: usize = @intCast(t.object.get("offset").?.integer);
            if (off + n > floats.len) return error.BadRecurrentWeights;
            try map.put(name, floats[off..][0..n]);
            const rows: usize = if (shape.len > 0) @intCast(shape[0].integer) else 1;
            const cols: usize = if (shape.len > 1) @intCast(shape[1].integer) else 1;
            try shapes.put(name, .{ rows, cols });
        }
        const cfg = meta.object.get("config").?.object;
        if (cfg.get("d").?.integer != D or cfg.get("layers").?.integer != LAYERS or cfg.get("heads").?.integer != HEADS or cfg.get("ff").?.integer != FF)
            return error.UnsupportedRecurrentConfig;
        const L = struct {
            fn get(m: *std.StringHashMap([]const f32), name: []const u8) ![]const f32 {
                return m.get(name) orelse {
                    std.debug.print("recurrent model: no tensor {s}\n", .{name});
                    return error.MissingTensor;
                };
            }
            fn lin(m: *std.StringHashMap([]const f32), s: *std.StringHashMap([2]usize), a: std.mem.Allocator, prefix: []const u8) !Linear {
                const w_name = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
                const shape = s.get(w_name) orelse return error.MissingTensor;
                return .{ .w = try get(m, w_name), .b = try get(m, try std.fmt.allocPrint(a, "{s}.bias", .{prefix})), .out = shape[0], .in = shape[1] };
            }
            fn norm(m: *std.StringHashMap([]const f32), a: std.mem.Allocator, prefix: []const u8) !Norm {
                return .{ .w = try get(m, try std.fmt.allocPrint(a, "{s}.weight", .{prefix})), .b = try get(m, try std.fmt.allocPrint(a, "{s}.bias", .{prefix})) };
            }
        };
        var model = Model{
            .species = try L.get(&map, "species.weight"),
            .moves = try L.get(&map, "moves.weight"),
            .status = try L.get(&map, "status.weight"),
            .role = try L.get(&map, "role"),
            .state_in = try L.lin(&map, &shapes, allocator, "state_in"),
            .last_in = try L.lin(&map, &shapes, allocator, "last_in"),
            .own_in = try L.lin(&map, &shapes, allocator, "own_in"),
            .opp_in = try L.lin(&map, &shapes, allocator, "opp_in"),
            .field_in = try L.lin(&map, &shapes, allocator, "field_in"),
            .blocks = undefined,
            .final_norm = try L.norm(&map, allocator, "final_norm"),
            .policy = try L.lin(&map, &shapes, allocator, "policy"),
            .win_norm = try L.norm(&map, allocator, "win.0"),
            .win1 = try L.lin(&map, &shapes, allocator, "win.1"),
            .win2 = try L.lin(&map, &shapes, allocator, "win.3"),
            .weights_sha256 = if (meta.object.get("weights_sha256")) |v| v.string else "",
        };
        for (&model.blocks, 0..) |*b, i| {
            const p = try std.fmt.allocPrint(allocator, "blocks.{d}", .{i});
            b.* = .{
                .norm1 = try L.norm(&map, allocator, try std.fmt.allocPrint(allocator, "{s}.norm1", .{p})),
                .qkv = try L.lin(&map, &shapes, allocator, try std.fmt.allocPrint(allocator, "{s}.qkv", .{p})),
                .proj = try L.lin(&map, &shapes, allocator, try std.fmt.allocPrint(allocator, "{s}.proj", .{p})),
                .norm2 = try L.norm(&map, allocator, try std.fmt.allocPrint(allocator, "{s}.norm2", .{p})),
                .ff1 = try L.lin(&map, &shapes, allocator, try std.fmt.allocPrint(allocator, "{s}.ff1", .{p})),
                .ff2 = try L.lin(&map, &shapes, allocator, try std.fmt.allocPrint(allocator, "{s}.ff2", .{p})),
            };
        }
        if (map.get("policy_deep.net.0.weight") != null) {
            model.deep = .{
                .norm = try L.norm(&map, allocator, "policy_deep.net.0"),
                .l1 = try L.lin(&map, &shapes, allocator, "policy_deep.net.1"),
                .l2 = try L.lin(&map, &shapes, allocator, "policy_deep.net.3"),
                .l3 = try L.lin(&map, &shapes, allocator, "policy_deep.net.5"),
            };
        }
        if (model.own_in.in != OWN_W or model.opp_in.in != OPP_W or model.last_in.in != LAST_W or model.field_in.in != FIELD_W) return error.UnsupportedRecurrentConfig;
        return model;
    }

    fn emb(table: []const f32, width: usize, id: u32) []const f32 {
        return table[@as(usize, id) * width ..][0..width];
    }

    /// Floats of scratch memory `stepBatch` needs for `n` positions.
    pub fn scratchLen(n: usize) usize {
        const R = n * TOKENS;
        return R * D * 4 + R * 3 * D + R * FF + n * 6 * (OWN_W + OPP_W) + n * (LAST_W + FIELD_W) + n * 6 * D * 2 + n * D * 4 + n * 64 * 3;
    }

    /// `step` for `xs.len` independent positions at once: every matrix product runs over all of their tokens (the same results
    /// up to the products' rounding).
    pub fn stepBatch(self: *const Model, xs: []const Input, h_prev: []const [D]f32, h_out: [][D]f32, logits: [][ACTIONS]f32, wins: []f32, scratch: []f32) void {
        const n = xs.len;
        if (n == 0) return;
        const R = n * TOKENS;
        var off: usize = 0;
        const take = struct {
            fn f(buf: []f32, o: *usize, len: usize) []f32 {
                const out = buf[o.*..][0..len];
                o.* += len;
                return out;
            }
        }.f;
        const tok = take(scratch, &off, R * D);
        const y = take(scratch, &off, R * D);
        const qkv = take(scratch, &off, R * 3 * D);
        const att = take(scratch, &off, R * D);
        const tmp = take(scratch, &off, R * D);
        const hidden = take(scratch, &off, R * FF);
        const own_rows = take(scratch, &off, n * 6 * OWN_W);
        const opp_rows = take(scratch, &off, n * 6 * OPP_W);
        const last_rows = take(scratch, &off, n * LAST_W);
        const field_rows = take(scratch, &off, n * FIELD_W);
        const own_proj = take(scratch, &off, n * 6 * D);
        const opp_proj = take(scratch, &off, n * 6 * D);
        const s_proj = take(scratch, &off, n * D);
        const l_proj = take(scratch, &off, n * D);
        const f_proj = take(scratch, &off, n * D);
        const norm_rows = take(scratch, &off, n * D);
        const small1 = take(scratch, &off, n * 64);
        const small2 = take(scratch, &off, n * 64);
        const small3 = take(scratch, &off, n * 64);
        for (xs, 0..) |*x, b| {
            const ids = [2][3]u32{ .{ x.last_own_move, x.last_own_species, x.last_own_status }, .{ x.last_opp_move, x.last_opp_species, x.last_opp_status } };
            var o: usize = 0;
            const last = last_rows[b * LAST_W ..][0..LAST_W];
            for (0..2) |side| {
                @memcpy(last[o..][0..17], x.last[side * 17 ..][0..17]);
                o += 17;
                @memcpy(last[o..][0..EMB], emb(self.moves, EMB, ids[side][0]));
                o += EMB;
                @memcpy(last[o..][0..EMB], emb(self.species, EMB, ids[side][1]));
                o += EMB;
                @memcpy(last[o..][0..SEMB], emb(self.status, SEMB, ids[side][2]));
                o += SEMB;
            }
            for (0..6) |sl| {
                const row = own_rows[(b * 6 + sl) * OWN_W ..][0..OWN_W];
                o = 0;
                @memcpy(row[o..][0..EMB], emb(self.species, EMB, x.own_species[sl]));
                o += EMB;
                row[o] = x.own[sl][0];
                o += 1;
                @memcpy(row[o..][0..SEMB], emb(self.status, SEMB, x.own_status[sl]));
                o += SEMB;
                row[o] = x.own[sl][1];
                o += 1;
                @memcpy(row[o..][0..4], x.own[sl][2..6]);
                o += 4;
                for (0..4) |j| {
                    @memcpy(row[o..][0..EMB], emb(self.moves, EMB, x.own_moves[sl][j]));
                    o += EMB;
                    row[o] = x.own[sl][6 + j];
                    o += 1;
                }
                const orow = opp_rows[(b * 6 + sl) * OPP_W ..][0..OPP_W];
                o = 0;
                orow[o] = x.opp[sl][0];
                o += 1;
                @memcpy(orow[o..][0..EMB], emb(self.species, EMB, x.opp_species[sl]));
                o += EMB;
                orow[o] = x.opp[sl][1];
                o += 1;
                @memcpy(orow[o..][0..SEMB], emb(self.status, SEMB, x.opp_status[sl]));
                o += SEMB;
                orow[o] = x.opp[sl][2];
                o += 1;
                @memcpy(orow[o..][0..4], x.opp[sl][3..7]);
                o += 4;
                for (0..4) |j| {
                    @memcpy(orow[o..][0..EMB], emb(self.moves, EMB, x.opp_moves[sl][j]));
                    o += EMB;
                }
            }
            @memcpy(field_rows[b * FIELD_W ..][0..FIELD_W], &x.field);
        }
        const h_flat: []const f32 = @as([*]const f32, @ptrCast(h_prev.ptr))[0 .. n * D];
        linear(s_proj, h_flat, n, self.state_in);
        linear(l_proj, last_rows, n, self.last_in);
        linear(own_proj, own_rows, n * 6, self.own_in);
        linear(opp_proj, opp_rows, n * 6, self.opp_in);
        linear(f_proj, field_rows, n, self.field_in);
        for (0..n) |b| {
            const base = b * TOKENS * D;
            @memcpy(tok[base..][0..D], s_proj[b * D ..][0..D]);
            @memcpy(tok[base + D ..][0..D], l_proj[b * D ..][0..D]);
            @memcpy(tok[base + 2 * D ..][0 .. 6 * D], own_proj[b * 6 * D ..][0 .. 6 * D]);
            @memcpy(tok[base + 8 * D ..][0 .. 6 * D], opp_proj[b * 6 * D ..][0 .. 6 * D]);
            @memcpy(tok[base + 14 * D ..][0..D], f_proj[b * D ..][0..D]);
            for (tok[base..][0 .. TOKENS * D], self.role[0 .. TOKENS * D]) |*t, r| t.* += r;
        }
        for (self.blocks) |blk| {
            simd.layerNorm(y, tok, R, D, blk.norm1.w, blk.norm1.b, LN_EPS);
            linear(qkv, y, R, blk.qkv);
            for (0..n) |b| attend(att[b * TOKENS * D ..][0 .. TOKENS * D], qkv[b * TOKENS * 3 * D ..][0 .. TOKENS * 3 * D]);
            linear(tmp, att, R, blk.proj);
            for (tok, tmp) |*t, v| t.* += v;
            simd.layerNorm(y, tok, R, D, blk.norm2.w, blk.norm2.b, LN_EPS);
            linear(hidden, y, R, blk.ff1);
            simd.geluInPlace(hidden);
            linear(tmp, hidden, R, blk.ff2);
            for (tok, tmp) |*t, v| t.* += v;
        }
        const h_out_flat: []f32 = @as([*]f32, @ptrCast(h_out.ptr))[0 .. n * D];
        for (0..n) |b| simd.layerNorm(h_out_flat[b * D ..][0..D], tok[b * TOKENS * D ..][0..D], 1, D, self.final_norm.w, self.final_norm.b, LN_EPS);
        const logits_flat: []f32 = @as([*]f32, @ptrCast(logits.ptr))[0 .. n * ACTIONS];
        linear(logits_flat, h_out_flat, n, self.policy);
        if (self.deep) |deep| {
            simd.layerNorm(norm_rows, h_out_flat, n, D, deep.norm.w, deep.norm.b, LN_EPS);
            const w1 = deep.l1.out;
            linear(small1[0 .. n * w1], norm_rows, n, deep.l1);
            simd.geluInPlace(small1[0 .. n * w1]);
            linear(small2[0 .. n * deep.l2.out], small1[0 .. n * w1], n, deep.l2);
            simd.geluInPlace(small2[0 .. n * deep.l2.out]);
            var extra: [ACTIONS]f32 = undefined;
            for (0..n) |b| {
                linear(&extra, small2[b * deep.l2.out ..][0..deep.l2.out], 1, deep.l3);
                for (logits[b][0..], extra) |*l, e| l.* += e;
            }
        }
        simd.layerNorm(norm_rows, h_out_flat, n, D, self.win_norm.w, self.win_norm.b, LN_EPS);
        const wd = self.win1.out;
        linear(small3[0 .. n * wd], norm_rows, n, self.win1);
        simd.geluInPlace(small3[0 .. n * wd]);
        linear(wins[0..n], small3[0 .. n * wd], n, self.win2);
    }

    /// One recurrent step: the policy logits over the nine slots and the win logit; `h_out` gets the next state.
    pub fn step(self: *const Model, x: *const Input, h_prev: *const [D]f32, h_out: *[D]f32, logits: *[ACTIONS]f32) f32 {
        var tok: [TOKENS * D]f32 = undefined;
        // token 0: the carried state
        linear(tok[0..D], h_prev, 1, self.state_in);
        // token 1: the last turn
        var last: [LAST_W]f32 = undefined;
        {
            var o: usize = 0;
            const ids = [2][3]u32{ .{ x.last_own_move, x.last_own_species, x.last_own_status }, .{ x.last_opp_move, x.last_opp_species, x.last_opp_status } };
            for (0..2) |side| {
                @memcpy(last[o..][0..17], x.last[side * 17 ..][0..17]);
                o += 17;
                @memcpy(last[o..][0..EMB], emb(self.moves, EMB, ids[side][0]));
                o += EMB;
                @memcpy(last[o..][0..EMB], emb(self.species, EMB, ids[side][1]));
                o += EMB;
                @memcpy(last[o..][0..SEMB], emb(self.status, SEMB, ids[side][2]));
                o += SEMB;
            }
        }
        linear(tok[D..][0..D], &last, 1, self.last_in);
        // tokens 2-7: own slots
        var own: [6 * OWN_W]f32 = undefined;
        for (0..6) |s| {
            const row = own[s * OWN_W ..][0..OWN_W];
            var o: usize = 0;
            @memcpy(row[o..][0..EMB], emb(self.species, EMB, x.own_species[s]));
            o += EMB;
            row[o] = x.own[s][0];
            o += 1;
            @memcpy(row[o..][0..SEMB], emb(self.status, SEMB, x.own_status[s]));
            o += SEMB;
            row[o] = x.own[s][1];
            o += 1;
            @memcpy(row[o..][0..4], x.own[s][2..6]);
            o += 4;
            for (0..4) |j| {
                @memcpy(row[o..][0..EMB], emb(self.moves, EMB, x.own_moves[s][j]));
                o += EMB;
                row[o] = x.own[s][6 + j];
                o += 1;
            }
        }
        linear(tok[2 * D ..][0 .. 6 * D], &own, 6, self.own_in);
        // tokens 8-13: opponent slots
        var opp: [6 * OPP_W]f32 = undefined;
        for (0..6) |s| {
            const row = opp[s * OPP_W ..][0..OPP_W];
            var o: usize = 0;
            row[o] = x.opp[s][0];
            o += 1;
            @memcpy(row[o..][0..EMB], emb(self.species, EMB, x.opp_species[s]));
            o += EMB;
            row[o] = x.opp[s][1];
            o += 1;
            @memcpy(row[o..][0..SEMB], emb(self.status, SEMB, x.opp_status[s]));
            o += SEMB;
            row[o] = x.opp[s][2];
            o += 1;
            @memcpy(row[o..][0..4], x.opp[s][3..7]);
            o += 4;
            for (0..4) |j| {
                @memcpy(row[o..][0..EMB], emb(self.moves, EMB, x.opp_moves[s][j]));
                o += EMB;
            }
        }
        linear(tok[8 * D ..][0 .. 6 * D], &opp, 6, self.opp_in);
        // token 14: the field
        linear(tok[14 * D ..][0..D], &x.field, 1, self.field_in);
        for (0..TOKENS * D) |i| tok[i] += self.role[i];

        var y: [TOKENS * D]f32 = undefined;
        var qkv: [TOKENS * 3 * D]f32 = undefined;
        var att: [TOKENS * D]f32 = undefined;
        var tmp: [TOKENS * D]f32 = undefined;
        var hidden: [TOKENS * FF]f32 = undefined;
        const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
        for (self.blocks) |b| {
            simd.layerNorm(&y, &tok, TOKENS, D, b.norm1.w, b.norm1.b, LN_EPS);
            linear(&qkv, &y, TOKENS, b.qkv);
            for (0..HEADS) |hh| {
                for (0..TOKENS) |i| {
                    const q = qkv[i * 3 * D + hh * HD ..][0..HD];
                    var scores: [TOKENS]f32 = undefined;
                    var max: f32 = -std.math.inf(f32);
                    for (0..TOKENS) |j| {
                        const k = qkv[j * 3 * D + D + hh * HD ..][0..HD];
                        scores[j] = simd.dot(q, k) * scale;
                        max = @max(max, scores[j]);
                    }
                    var total: f32 = 0;
                    for (&scores) |*s| {
                        s.* = @exp(s.* - max);
                        total += s.*;
                    }
                    const out = att[i * D + hh * HD ..][0..HD];
                    @memset(out, 0);
                    for (0..TOKENS) |j| simd.axpy(out, scores[j] / total, qkv[j * 3 * D + 2 * D + hh * HD ..][0..HD]);
                }
            }
            linear(&tmp, &att, TOKENS, b.proj);
            for (&tok, tmp) |*t, v| t.* += v;
            simd.layerNorm(&y, &tok, TOKENS, D, b.norm2.w, b.norm2.b, LN_EPS);
            linear(&hidden, &y, TOKENS, b.ff1);
            simd.geluInPlace(&hidden);
            linear(&tmp, &hidden, TOKENS, b.ff2);
            for (&tok, tmp) |*t, v| t.* += v;
        }
        simd.layerNorm(h_out, tok[0..D], 1, D, self.final_norm.w, self.final_norm.b, LN_EPS);
        linear(logits, h_out, 1, self.policy);
        if (self.deep) |deep| {
            var n: [D]f32 = undefined;
            simd.layerNorm(&n, h_out, 1, D, deep.norm.w, deep.norm.b, LN_EPS);
            var a1: [64]f32 = undefined;
            var a2: [64]f32 = undefined;
            var extra: [ACTIONS]f32 = undefined;
            linear(a1[0..deep.l1.out], &n, 1, deep.l1);
            simd.geluInPlace(a1[0..deep.l1.out]);
            linear(a2[0..deep.l2.out], a1[0..deep.l1.out], 1, deep.l2);
            simd.geluInPlace(a2[0..deep.l2.out]);
            linear(&extra, a2[0..deep.l2.out], 1, deep.l3);
            for (logits, extra) |*l, e| l.* += e;
        }
        var wn: [D]f32 = undefined;
        simd.layerNorm(&wn, h_out, 1, D, self.win_norm.w, self.win_norm.b, LN_EPS);
        var w1: [64]f32 = undefined;
        linear(w1[0..self.win1.out], &wn, 1, self.win1);
        simd.geluInPlace(w1[0..self.win1.out]);
        var w2: [1]f32 = undefined;
        linear(&w2, w1[0..self.win1.out], 1, self.win2);
        return w2[0];
    }
};

/// Multi-head attention of one position's 15 tokens: `qkv` [15, 3D] (q, k, v, each heads x HD), `out` [15, D]. Per head the
/// keys are laid out by dimension, so a query's 15 scores come from 32 multiply-adds of 16-wide vectors (no horizontal sums).
fn attend(out: []f32, qkv: []const f32) void {
    const V16 = @Vector(16, f32);
    const V8 = @Vector(8, f32);
    const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
    for (0..HEADS) |hh| {
        var kt: [HD]V16 = undefined; // kt[d][j]: key j's dimension d (j = 15 padded)
        for (0..HD) |d| {
            var col: [16]f32 = @splat(0);
            for (0..TOKENS) |j| col[j] = qkv[j * 3 * D + D + hh * HD + d];
            kt[d] = col;
        }
        for (0..TOKENS) |i| {
            const q = qkv[i * 3 * D + hh * HD ..][0..HD];
            var acc: V16 = @splat(0);
            for (0..HD) |d| acc += @as(V16, @splat(q[d] * scale)) * kt[d];
            var sc: [16]f32 = acc;
            sc[TOKENS] = -std.math.inf(f32);
            var max: f32 = sc[0];
            for (sc[1..TOKENS]) |v| max = @max(max, v);
            for (sc[0..TOKENS]) |*v| v.* -= max;
            simd.expInPlace(sc[0..16]);
            var total: f32 = 0;
            for (sc[0..TOKENS]) |v| total += v;
            const inv = 1.0 / total;
            var o: [HD / 8]V8 = @splat(@splat(0));
            for (0..TOKENS) |j| {
                const p: V8 = @splat(sc[j] * inv);
                const v = qkv[j * 3 * D + 2 * D + hh * HD ..][0..HD];
                inline for (0..HD / 8) |c| o[c] += p * @as(V8, v[c * 8 ..][0..8].*);
            }
            const dst = out[i * D + hh * HD ..][0..HD];
            inline for (0..HD / 8) |c| dst[c * 8 ..][0..8].* = o[c];
        }
    }
}

const cblas = struct {
    const RowMajor = 101;
    const NoTrans = 111;
    const Trans = 112;
    extern fn cblas_sgemm(order: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c: [*]f32, ldc: c_int) void;
};

/// out[r, o] = sum_i x[r, i] * w[o, i] + b[o] (PyTorch's [out, in] weights): the linked BLAS (Accelerate on macOS, OpenBLAS
/// with -Dopenblas=true), else a portable vector kernel.
fn linear(out: []f32, x: []const f32, rows: usize, l: Linear) void {
    std.debug.assert(out.len >= rows * l.out and x.len >= rows * l.in);
    if (have_blas and rows * l.out * l.in >= 4096 and !portable()) {
        // (the bias first, then the product added to it: no second pass over the output)
        for (0..rows) |r| @memcpy(out[r * l.out ..][0..l.out], l.b[0..l.out]);
        cblas.cblas_sgemm(cblas.RowMajor, cblas.NoTrans, cblas.Trans, @intCast(rows), @intCast(l.out), @intCast(l.in), 1.0, x.ptr, @intCast(l.in), l.w.ptr, @intCast(l.in), 1.0, out.ptr, @intCast(l.out));
        return;
    }
    linearPortable(out, x, rows, l);
}

/// Reads the environment switches now (before the search starts threads).
pub fn warm() void {
    _ = portable();
}

var portable_flag: ?bool = null;
/// RMODEL_PORTABLE=1: the portable kernel on macOS too (for checking it there).
fn portable() bool {
    if (portable_flag == null) portable_flag = if (std.c.getenv("RMODEL_PORTABLE")) |v| v[0] == '1' else false;
    return portable_flag.?;
}

/// The product without a BLAS: blocks of 4 rows x 2 outputs, 8 accumulators of 8 lanes (each weight row loaded once for 4
/// rows, each input row for 2 outputs), then the lengths' tails and the leftover rows and outputs one by one.
fn linearPortable(out: []f32, x: []const f32, rows: usize, l: Linear) void {
    const V8 = @Vector(8, f32);
    const in = l.in;
    const vec_end = in - in % 8;
    var r: usize = 0;
    while (r + 4 <= rows) : (r += 4) {
        const xr = [4][]const f32{ x[r * in ..][0..in], x[(r + 1) * in ..][0..in], x[(r + 2) * in ..][0..in], x[(r + 3) * in ..][0..in] };
        var o: usize = 0;
        while (o + 2 <= l.out) : (o += 2) {
            const w0 = l.w[o * in ..][0..in];
            const w1 = l.w[(o + 1) * in ..][0..in];
            var acc: [4][2]V8 = @splat(@splat(@splat(0)));
            var i: usize = 0;
            while (i < vec_end) : (i += 8) {
                const a: V8 = w0[i..][0..8].*;
                const b: V8 = w1[i..][0..8].*;
                inline for (0..4) |k| {
                    const xv: V8 = xr[k][i..][0..8].*;
                    acc[k][0] += xv * a;
                    acc[k][1] += xv * b;
                }
            }
            inline for (0..4) |k| {
                var s0 = @reduce(.Add, acc[k][0]);
                var s1 = @reduce(.Add, acc[k][1]);
                for (vec_end..in) |t| {
                    s0 += xr[k][t] * w0[t];
                    s1 += xr[k][t] * w1[t];
                }
                out[(r + k) * l.out + o] = s0 + l.b[o];
                out[(r + k) * l.out + o + 1] = s1 + l.b[o + 1];
            }
        }
        while (o < l.out) : (o += 1) {
            const wo = l.w[o * in ..][0..in];
            for (0..4) |k| out[(r + k) * l.out + o] = simd.dot(xr[k], wo) + l.b[o];
        }
    }
    while (r < rows) : (r += 1) {
        for (0..l.out) |o| out[r * l.out + o] = simd.dot(x[r * in ..][0..in], l.w[o * in ..][0..in]) + l.b[o];
    }
}
