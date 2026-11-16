//! Zig port of the opponent-team belief model (fp_belief `Model::infer` and `sampler.rs`).
//! Reads `weights.bin`, `manifest.txt` and `training_metadata.json` from the model directory.
//! Frame encoding (public-state tokens) lives in `tokenizer.zig`.

const std = @import("std");
const pkmn = @import("pkmn");
const completion = @import("completion.zig");

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const TEAM_SIZE = 6;
pub const MOVE_SLOTS = 4;
const LN_EPS: f32 = 1e-5;

pub const Vocab = struct {
    token_to_id: std.StringHashMap(u32),
    species: [][]const u8,
    moves: [][]const u8,
    legal: completion.Legal,

    /// Reads `training_metadata.json`: `input_vocab` (token -> id), `species` and `moves` lists.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Vocab {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{});
        const root = parsed.object;
        var token_to_id = std.StringHashMap(u32).init(allocator);
        var vocab_it = root.get("input_vocab").?.object.iterator();
        while (vocab_it.next()) |entry| {
            try token_to_id.put(entry.key_ptr.*, @intCast(entry.value_ptr.integer));
        }
        return .{
            .token_to_id = token_to_id,
            .legal = try completion.Legal.load(allocator, io),
            .species = try stringList(allocator, root.get("species").?),
            .moves = try stringList(allocator, root.get("moves").?),
        };
    }

    pub fn speciesIndex(self: *const Vocab, name: []const u8) ?usize {
        for (self.species, 0..) |s, i| if (std.mem.eql(u8, s, name)) return i;
        return null;
    }

    /// Token id for a token string; unknown tokens map to UNK.
    pub fn tokenId(self: *const Vocab, token: []const u8) u32 {
        return self.token_to_id.get(token) orelse UNK_ID;
    }
};

fn stringList(allocator: std.mem.Allocator, value: std.json.Value) ![][]const u8 {
    const items = value.array.items;
    const out = try allocator.alloc([]const u8, items.len);
    for (items, 0..) |item, i| out[i] = item.string;
    return out;
}

pub const Marginals = struct {
    species: []f32,
    /// species_count * move_count, row-major by species.
    moves: []f32,
};

pub const Model = struct {
    vocab_size: usize,
    species_count: usize,
    move_count: usize,
    embedding_dim: usize,
    hidden_dim: usize,
    token_embedding: []f32,
    frame_norm_w: []f32,
    frame_norm_b: []f32,
    gru_w_ih: []f32,
    gru_w_hh: []f32,
    gru_b_ih: []f32,
    gru_b_hh: []f32,
    species_w: []f32,
    species_b: []f32,
    species_embedding: []f32,
    move0_w: []f32,
    move0_b: []f32,
    move2_w: []f32,
    move2_b: []f32,
    move3_w: []f32,
    move3_b: []f32,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !Model {
        const manifest_path = try std.fmt.allocPrint(allocator, "{s}/manifest.txt", .{dir});
        const weights_path = try std.fmt.allocPrint(allocator, "{s}/weights.bin", .{dir});
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
                try config.put(key, try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10));
            } else if (std.mem.eql(u8, kind, "tensor")) {
                const name = parts.next() orelse return error.BadManifest;
                const rows = try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10);
                const cols = try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10);
                const offset = try std.fmt.parseUnsigned(usize, parts.next() orelse return error.BadManifest, 10);
                try tensors.put(name, .{ rows, cols, offset });
            } else return error.BadManifest;
        }

        const v = config.get("input_vocab_size") orelse return error.BadManifest;
        const s = config.get("species_count") orelse return error.BadManifest;
        const m = config.get("move_count") orelse return error.BadManifest;
        const e = config.get("embedding_dim") orelse return error.BadManifest;
        const h = config.get("hidden_dim") orelse return error.BadManifest;
        var loader = Loader{ .allocator = allocator, .blob = blob, .tensors = &tensors };
        return .{
            .vocab_size = v,
            .species_count = s,
            .move_count = m,
            .embedding_dim = e,
            .hidden_dim = h,
            .token_embedding = try loader.read("token_embedding.weight", v * e),
            .frame_norm_w = try loader.read("frame_norm.weight", e),
            .frame_norm_b = try loader.read("frame_norm.bias", e),
            .gru_w_ih = try loader.read("encoder.weight_ih_l0", 3 * h * e),
            .gru_w_hh = try loader.read("encoder.weight_hh_l0", 3 * h * h),
            .gru_b_ih = try loader.read("encoder.bias_ih_l0", 3 * h),
            .gru_b_hh = try loader.read("encoder.bias_hh_l0", 3 * h),
            .species_w = try loader.read("species_head.weight", s * h),
            .species_b = try loader.read("species_head.bias", s),
            .species_embedding = try loader.read("species_embedding.weight", s * h),
            .move0_w = try loader.read("move_head.0.weight", h * h),
            .move0_b = try loader.read("move_head.0.bias", h),
            .move2_w = try loader.read("move_head.2.weight", h),
            .move2_b = try loader.read("move_head.2.bias", h),
            .move3_w = try loader.read("move_head.3.weight", m * h),
            .move3_b = try loader.read("move_head.3.bias", m),
        };
    }

    /// Runs the GRU over the frames (each frame is a bag of token ids; 0 is padding) and returns
    /// the species and move marginals.
    pub fn infer(self: *const Model, allocator: std.mem.Allocator, frames: []const []const u32) !Marginals {
        if (frames.len == 0) return error.NoFrames;
        const e = self.embedding_dim;
        const hd = self.hidden_dim;
        const h = try allocator.alloc(f32, hd);
        @memset(h, 0);
        const x = try allocator.alloc(f32, e);
        const gi = try allocator.alloc(f32, 3 * hd);
        const gh = try allocator.alloc(f32, 3 * hd);
        const next = try allocator.alloc(f32, hd);
        for (frames) |frame| {
            @memset(x, 0);
            var count: usize = 0;
            for (frame) |tok| {
                if (tok == PAD_ID) continue;
                if (tok >= self.vocab_size) return error.TokenOutOfVocabulary;
                for (x, self.token_embedding[tok * e ..][0..e]) |*acc, v| acc.* += v;
                count += 1;
            }
            if (count == 0) return error.EmptyFrame;
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(count));
            for (x) |*v| v.* *= inv;
            layerNorm(x, self.frame_norm_w, self.frame_norm_b);
            // GRU with PyTorch's gate order: reset, update, new
            matvecBias(gi, self.gru_w_ih, self.gru_b_ih, x, 3 * hd);
            matvecBias(gh, self.gru_w_hh, self.gru_b_hh, h, 3 * hd);
            for (0..hd) |i| {
                const r = sigmoid(gi[i] + gh[i]);
                const z = sigmoid(gi[hd + i] + gh[hd + i]);
                const n = std.math.tanh(gi[2 * hd + i] + r * gh[2 * hd + i]);
                next[i] = (1.0 - z) * n + z * h[i];
            }
            @memcpy(h, next);
        }

        const species = try allocator.alloc(f32, self.species_count);
        matvecBias(species, self.species_w, self.species_b, h, self.species_count);
        for (species) |*v| v.* = sigmoid(v.*);

        const m = self.move_count;
        const moves = try allocator.alloc(f32, self.species_count * m);
        const state = try allocator.alloc(f32, hd);
        const a = try allocator.alloc(f32, hd);
        const logits = try allocator.alloc(f32, m);
        for (0..self.species_count) |s| {
            for (state, h, self.species_embedding[s * hd ..][0..hd]) |*dst, hv, ev| dst.* = hv + ev;
            matvecBias(a, self.move0_w, self.move0_b, state, hd);
            for (a) |*v| v.* = gelu(v.*);
            layerNorm(a, self.move2_w, self.move2_b);
            matvecBias(logits, self.move3_w, self.move3_b, a, m);
            for (moves[s * m ..][0..m], logits) |*dst, l| dst.* = sigmoid(l);
        }
        return .{ .species = species, .moves = moves };
    }
};

/// y = W x + b, with W stored row-major as (rows, x.len).
fn matvecBias(y: []f32, w: []const f32, b: []const f32, x: []const f32, rows: usize) void {
    const cols = x.len;
    for (0..rows) |r| {
        var acc: f32 = 0;
        for (w[r * cols ..][0..cols], x) |wv, xv| acc += wv * xv;
        y[r] = acc + b[r];
    }
}

fn layerNorm(x: []f32, weight: []const f32, bias: []const f32) void {
    const n: f32 = @floatFromInt(x.len);
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= n;
    var variance: f32 = 0;
    for (x) |v| variance += (v - mean) * (v - mean);
    variance /= n;
    const inv = 1.0 / @sqrt(variance + LN_EPS);
    for (x, weight, bias) |*v, w, b| v.* = (v.* - mean) * inv * w + b;
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x * std.math.sqrt1_2));
}

/// Abramowitz & Stegun 7.1.26, as in the Rust implementation.
fn erf(x: f32) f32 {
    const sign: f32 = if (x < 0) -1 else 1;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.327_591_1 * ax);
    const poly = t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    return sign * (1.0 - poly * @exp(-ax * ax));
}

const Loader = struct {
    allocator: std.mem.Allocator,
    blob: []const u8,
    tensors: *std.StringHashMap([3]usize),

    fn read(self: *Loader, name: []const u8, expected: usize) ![]f32 {
        const entry = self.tensors.get(name) orelse return error.MissingTensor;
        if (entry[0] * entry[1] != expected) return error.TensorShapeMismatch;
        const start = entry[2] * 4;
        if (start + expected * 4 > self.blob.len) return error.WeightsTooShort;
        const out = try self.allocator.alloc(f32, expected);
        for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, self.blob[start + i * 4 ..][0..4], .little));
        return out;
    }
};

/// Gumbel top-k over log-probabilities, drawing from splitmix64 like `sampler.rs`.
pub const Rng = struct {
    state: u64,

    pub fn init(seed: u64) Rng {
        return .{ .state = seed };
    }

    fn nextU64(self: *Rng) u64 {
        self.state +%= 0x9E37_79B9_7F4A_7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58_476D_1CE4_E5B9;
        z = (z ^ (z >> 27)) *% 0x94D0_49BB_1331_11EB;
        return z ^ (z >> 31);
    }

    /// Uniform in [0, 1).
    pub fn nextF64(self: *Rng) f64 {
        return @as(f64, @floatFromInt(self.nextU64() >> 11)) / @as(f64, @floatFromInt(@as(u64, 1) << 53));
    }
};

const EPS: f64 = 1e-9;

/// Indices of the k largest Gumbel-perturbed log-probabilities (ties broken by lower index).
fn gumbelTopK(allocator: std.mem.Allocator, probs: []const f64, indices: []const usize, k: usize, rng: *Rng) ![]usize {
    const Keyed = struct { key: f64, index: usize };
    const keyed = try allocator.alloc(Keyed, indices.len);
    defer allocator.free(keyed);
    for (indices, probs, 0..) |index, p, i| {
        const uniform = std.math.clamp(rng.nextF64(), EPS, 1.0 - EPS);
        const gumbel = -@log(-@log(uniform));
        keyed[i] = .{ .key = @log(@max(p, EPS)) + gumbel, .index = index };
    }
    std.mem.sort(Keyed, keyed, {}, struct {
        fn lessThan(_: void, a: Keyed, b: Keyed) bool {
            if (a.key != b.key) return a.key > b.key;
            return a.index < b.index;
        }
    }.lessThan);
    const out = try allocator.alloc(usize, @min(k, keyed.len));
    for (out, keyed[0..out.len]) |*dst, src| dst.* = src.index;
    return out;
}

/// The gen1 species with this (normalized) name, or null if the engine has no such species.
pub fn gen1Species(name: []const u8) ?pkmn.gen1.Species {
    inline for (@typeInfo(pkmn.gen1.Species).@"enum".fields) |field| {
        if (field.value != 0 and completion.eqlNormalized(field.name, name)) {
            return @enumFromInt(field.value);
        }
    }
    return null;
}

pub const SampledPokemon = struct {
    species: []const u8,
    moves: [][]const u8,
};

/// Draws one complete opponent team. `known` lists revealed species with their revealed moves;
/// those entries are kept and the remaining slots are filled from the marginals.
pub fn sampleCompletion(
    allocator: std.mem.Allocator,
    vocab: *const Vocab,
    marginals: Marginals,
    move_count: usize,
    known: []const SampledPokemon,
    rng: *Rng,
) ![]SampledPokemon {
    if (known.len > TEAM_SIZE) return error.TooManyRevealed;
    const needed = TEAM_SIZE - known.len;
    var team: std.ArrayList(SampledPokemon) = .empty;
    for (known) |k| try team.append(allocator, k);
    if (needed == 0) return team.toOwnedSlice(allocator);

    var candidate_indices: std.ArrayList(usize) = .empty;
    var candidate_probs: std.ArrayList(f64) = .empty;
    for (vocab.species, 0..) |name, i| {
        // species the engine cannot represent are excluded from the draw, not resampled afterwards
        if (gen1Species(name) == null or !vocab.legal.species.contains(name)) continue;
        var is_known = false;
        for (known) |k| {
            if (std.mem.eql(u8, k.species, name)) is_known = true;
        }
        if (is_known) continue;
        try candidate_indices.append(allocator, i);
        try candidate_probs.append(allocator, marginals.species[i]);
    }
    const chosen = try gumbelTopK(allocator, candidate_probs.items, candidate_indices.items, needed, rng);
    for (chosen) |species_index| {
        const row = marginals.moves[species_index * move_count ..][0..move_count];
        var move_indices: std.ArrayList(usize) = .empty;
        var move_probs: std.ArrayList(f64) = .empty;
        for (row, 0..) |prob, i| {
            if (!vocab.legal.allows(vocab.species[species_index], vocab.moves[i])) continue;
            try move_indices.append(allocator, i);
            try move_probs.append(allocator, prob);
        }
        const picked = try gumbelTopK(allocator, move_probs.items, move_indices.items, MOVE_SLOTS, rng);
        const moves = try allocator.alloc([]const u8, picked.len);
        for (moves, picked) |*dst, idx| dst.* = vocab.moves[idx];
        try team.append(allocator, .{ .species = vocab.species[species_index], .moves = moves });
    }
    return team.toOwnedSlice(allocator);
}
