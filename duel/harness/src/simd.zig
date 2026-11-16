//! Vectorized helpers for the value model's scalar hot loops: dot products, axpy, exp, and GELU.
//! Lengths that are not a multiple of the vector width fall back to scalar code for the tail.

const std = @import("std");

pub const W = 8;
pub const V = @Vector(W, f32);
const VI = @Vector(W, i32);

/// Sum of a[i] * b[i].
pub fn dot(a: []const f32, b: []const f32) f32 {
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + W <= a.len) : (i += W) {
        const va: V = a[i..][0..W].*;
        const vb: V = b[i..][0..W].*;
        acc += va * vb;
    }
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

/// out[i] += w * row[i].
pub fn axpy(out: []f32, w: f32, row: []const f32) void {
    const vw: V = @splat(w);
    var i: usize = 0;
    while (i + W <= out.len) : (i += W) {
        const vo: V = out[i..][0..W].*;
        const vr: V = row[i..][0..W].*;
        out[i..][0..W].* = vo + vw * vr;
    }
    while (i < out.len) : (i += 1) out[i] += w * row[i];
}

/// exp(x) for a vector of inputs: range reduction to 2^n * exp(r) with |r| <= ln2/2, and a degree-7
/// Taylor polynomial for exp(r) (relative error about 2e-8 in that range).
pub fn expV(x0: V) V {
    const lo: V = @splat(-87.0);
    const hi: V = @splat(88.0);
    const x = @min(@max(x0, lo), hi);
    const n = @floor(x * @as(V, @splat(1.44269504088896341)) + @as(V, @splat(0.5)));
    const r = (x - n * @as(V, @splat(0.693145751953125))) - n * @as(V, @splat(1.428606765330187e-6));
    var p: V = @splat(1.0 / 5040.0);
    p = p * r + @as(V, @splat(1.0 / 720.0));
    p = p * r + @as(V, @splat(1.0 / 120.0));
    p = p * r + @as(V, @splat(1.0 / 24.0));
    p = p * r + @as(V, @splat(1.0 / 6.0));
    p = p * r + @as(V, @splat(0.5));
    p = p * r + @as(V, @splat(1.0));
    p = p * r + @as(V, @splat(1.0));
    const ni: VI = @intFromFloat(n);
    const bits: VI = (ni + @as(VI, @splat(127))) << @as(VI, @splat(23));
    return p * @as(V, @bitCast(bits));
}

/// In-place exp over a slice (vector body, scalar tail).
pub fn expInPlace(x: []f32) void {
    var i: usize = 0;
    while (i + W <= x.len) : (i += W) {
        const v: V = x[i..][0..W].*;
        x[i..][0..W].* = expV(v);
    }
    while (i < x.len) : (i += 1) x[i] = @exp(x[i]);
}

/// Abramowitz & Stegun 7.1.26 erf for a vector (same coefficients as the scalar erf).
fn erfV(x: V) V {
    const zero: V = @splat(0);
    const one: V = @splat(1);
    const neg = x < zero;
    const ax = @abs(x);
    const t = one / (one + @as(V, @splat(0.3275911)) * ax);
    const poly = t * (@as(V, @splat(0.254829592)) + t * (@as(V, @splat(-0.284496736)) + t * (@as(V, @splat(1.421413741)) + t * (@as(V, @splat(-1.453152027)) + t * @as(V, @splat(1.061405429))))));
    const e = one - poly * expV(-ax * ax);
    return @select(f32, neg, -e, e);
}

/// GELU (erf form) applied in place, vector body with scalar tail.
pub fn geluInPlace(x: []f32) void {
    const half: V = @splat(0.5);
    const one: V = @splat(1);
    const s2: V = @splat(std.math.sqrt1_2);
    var i: usize = 0;
    while (i + W <= x.len) : (i += W) {
        const v: V = x[i..][0..W].*;
        x[i..][0..W].* = half * v * (one + erfV(v * s2));
    }
    while (i < x.len) : (i += 1) {
        const v = x[i];
        const e = blk: {
            const ax = @abs(v * std.math.sqrt1_2);
            const t = 1.0 / (1.0 + 0.3275911 * ax);
            const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
            const r = 1.0 - poly * @exp(-ax * ax);
            break :blk if (v * std.math.sqrt1_2 < 0) -r else r;
        };
        x[i] = 0.5 * v * (1.0 + e);
    }
}

/// LayerNorm over `rows` rows of `width` values (width a multiple of the vector width), written to
/// `out`: (x - mean) / sqrt(var + eps) * w + b, with the same two-pass statistics as the scalar code.
pub fn layerNorm(out: []f32, x: []const f32, rows: usize, width: usize, w: []const f32, b: []const f32, eps: f32) void {
    std.debug.assert(width % W == 0);
    for (0..rows) |r| {
        const row = x[r * width ..][0..width];
        var sum: V = @splat(0);
        var i: usize = 0;
        while (i < width) : (i += W) sum += @as(V, row[i..][0..W].*);
        const mean = @reduce(.Add, sum) / @as(f32, @floatFromInt(width));
        const vmean: V = @splat(mean);
        var sq: V = @splat(0);
        i = 0;
        while (i < width) : (i += W) {
            const d = @as(V, row[i..][0..W].*) - vmean;
            sq += d * d;
        }
        const variance = @reduce(.Add, sq) / @as(f32, @floatFromInt(width));
        const inv: V = @splat(1.0 / @sqrt(variance + eps));
        i = 0;
        while (i < width) : (i += W) {
            const d = @as(V, row[i..][0..W].*) - vmean;
            const gw: V = w[i..][0..W].*;
            const gb: V = b[i..][0..W].*;
            out[r * width + i ..][0..W].* = d * inv * gw + gb;
        }
    }
}

test "dot and axpy match scalar code" {
    var a: [70]f32 = undefined;
    var b: [70]f32 = undefined;
    for (&a, &b, 0..) |*x, *y, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)));
        y.* = @cos(@as(f32, @floatFromInt(i)) * 0.5);
    }
    var scalar: f32 = 0;
    for (a, b) |x, y| scalar += x * y;
    try std.testing.expectApproxEqAbs(scalar, dot(&a, &b), 1e-4);
    var out = a;
    axpy(&out, 0.25, &b);
    for (out, a, b) |o, x, y| try std.testing.expectApproxEqAbs(x + 0.25 * y, o, 1e-6);
}

test "layernorm matches the scalar form" {
    var x: [48]f32 = undefined;
    var w: [16]f32 = undefined;
    var b: [16]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.7) * 3.0;
    for (&w, &b, 0..) |*wi, *bi, i| {
        wi.* = 1.0 + 0.05 * @as(f32, @floatFromInt(i));
        bi.* = 0.01 * @as(f32, @floatFromInt(i));
    }
    var out: [48]f32 = undefined;
    layerNorm(&out, &x, 3, 16, &w, &b, 1e-5);
    for (0..3) |r| {
        const row = x[r * 16 ..][0..16];
        var mean: f32 = 0;
        for (row) |v| mean += v;
        mean /= 16;
        var variance: f32 = 0;
        for (row) |v| variance += (v - mean) * (v - mean);
        variance /= 16;
        const inv = 1.0 / @sqrt(variance + 1e-5);
        for (0..16) |i| try std.testing.expectApproxEqAbs((row[i] - mean) * inv * w[i] + b[i], out[r * 16 + i], 1e-5);
    }
}

test "exp matches @exp to float precision" {
    var x: [64]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = -20.0 + 0.6 * @as(f32, @floatFromInt(i));
    var y = x;
    expInPlace(&y);
    for (x, y) |v, e| {
        const want = @exp(v);
        try std.testing.expect(@abs(e - want) <= 3e-7 * want + 1e-38);
    }
}

test "gelu matches the scalar erf form" {
    var x: [37]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = -4.0 + 0.23 * @as(f32, @floatFromInt(i));
    var y = x;
    geluInPlace(&y);
    for (x, y) |v, g| {
        const e = v * std.math.sqrt1_2;
        const ax = @abs(e);
        const t = 1.0 / (1.0 + 0.3275911 * ax);
        const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        const r = 1.0 - poly * @exp(-ax * ax);
        const want = 0.5 * v * (1.0 + if (e < 0) -r else r);
        try std.testing.expectApproxEqAbs(want, g, 2e-6);
    }
}
