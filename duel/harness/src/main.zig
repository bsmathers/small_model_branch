const std = @import("std");
const pkmn = @import("pkmn");
const search = @import("search.zig");
const vm = @import("value_model.zig");
const belief = @import("belief.zig");
const vs = @import("value_state.zig");
const pubstate = @import("public_state.zig");
const know = @import("knowledge.zig");
const pathmod = @import("path.zig");
const completion = @import("completion.zig");
const universe = @import("universe.zig");
const bel = @import("belief.zig");
const pubhist = @import("public_state.zig");
const tokenizer = @import("tokenizer.zig");
const policy = @import("policy.zig");
const teams = @import("teams.zig");
const priors = @import("priors.zig");
const snapmod = @import("snapshot.zig");
const remote = @import("remote.zig");
const rmodel = @import("rmodel.zig");
const rnative = @import("rnative.zig");
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern fn openblas_set_num_threads(n: c_int) void;
const optionName = snapmod.optionName;
const LastAction = snapmod.LastAction;
const lastAction = snapmod.lastAction;
const serializeDuelSnapshot = snapmod.serializeDuelSnapshot;

const team1 = [_]pkmn.gen1.helpers.Pokemon{
    .{ .species = .Tauros, .moves = &.{ .BodySlam, .HyperBeam, .Blizzard, .Earthquake } },
    .{ .species = .Chansey, .moves = &.{ .Reflect, .SeismicToss, .SoftBoiled, .ThunderWave } },
    .{ .species = .Snorlax, .moves = &.{ .BodySlam, .Reflect, .Rest, .IceBeam } },
    .{ .species = .Exeggutor, .moves = &.{ .SleepPowder, .Psychic, .Explosion, .DoubleEdge } },
    .{ .species = .Starmie, .moves = &.{ .Recover, .ThunderWave, .Blizzard, .Thunderbolt } },
    .{ .species = .Alakazam, .moves = &.{ .Psychic, .SeismicToss, .ThunderWave, .Recover } },
};
const team2 = [_]pkmn.gen1.helpers.Pokemon{
    .{ .species = .Tauros, .moves = &.{ .BodySlam, .HyperBeam, .Blizzard, .Earthquake } },
    .{ .species = .Chansey, .moves = &.{ .Reflect, .SeismicToss, .SoftBoiled, .ThunderWave } },
    .{ .species = .Snorlax, .moves = &.{ .BodySlam, .Reflect, .Rest, .IceBeam } },
    .{ .species = .Exeggutor, .moves = &.{ .SleepPowder, .Psychic, .Explosion, .DoubleEdge } },
    .{ .species = .Starmie, .moves = &.{ .Recover, .ThunderWave, .Blizzard, .Thunderbolt } },
    .{ .species = .Alakazam, .moves = &.{ .Psychic, .SeismicToss, .ThunderWave, .Recover } },
};

/// Plays one game. P1 uses the determinized search; P2 plays uniformly random moves. Returns +1
/// if P1 wins, -1 if P2 wins, 0 otherwise. `turns` receives the battle length.
fn playGame(allocator: std.mem.Allocator, seed: u64, iterations: u32, turns: *u32) !i32 {
    var prng = std.Random.DefaultPrng.init(seed);
    var rand = prng.random();
    var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var c1: pkmn.Choice = .{};
    var c2: pkmn.Choice = .{};
    var result = try battle.update(c1, c2, &options);
    const cfg = search.Config{ .iterations = iterations };
    while (result.type == .None) {
        const outcome = try search.search(allocator, &battle, result, cfg, &rand);
        const n1 = battle.choices(.P1, result.p1, &choices1);
        if (n1 == 0) break;
        c1 = choices1[@min(outcome.choice, n1 - 1)];
        const n2 = battle.choices(.P2, result.p2, &choices2);
        if (n2 == 0) break;
        c2 = choices2[rand.uintLessThan(u8, n2)];
        result = try battle.update(c1, c2, &options);
    }
    turns.* = battle.turn;
    return switch (result.type) {
        .Win => 1,
        .Lose => -1,
        else => 0,
    };
}

pub fn main(init: std.process.Init) !void {
    // (repo copy) Accelerate's matrix products on one thread unless asked otherwise: the harness runs many games in parallel,
    // and the products here are small, so more threads cost CPU (measured: 2.3 against 1.0 CPU s per klnash decision)
    if (@import("builtin").os.tag == .macos) _ = setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
    if (comptime @import("build_options").openblas) {
        if (std.c.getenv("OPENBLAS_NUM_THREADS") == null) openblas_set_num_threads(1);
    }
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len > 1 and std.mem.eql(u8, args[1], "value-check")) return valueCheck(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "belief-check")) return beliefCheck(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "tokenizer-check")) return tokenizerCheck(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "observer-check")) return observerCheck(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "live-belief")) return liveBelief(init, args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "frame-schema")) return frameSchema(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "knowledge-check")) return knowledgeCheck(init, args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "path-check")) return pathCheck(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "value-game")) return valueGame(init, args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "engine-fuzz")) return engineFuzz(args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "status-check")) return statusCheck();
    if (args.len > 1 and std.mem.eql(u8, args[1], "tournament")) return tournament(init, args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "duel")) return duel(init, args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "rmodel-check")) return rmodelCheck(init, args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "mechanics-check")) return mechanicsCheck();
    if (args.len > 1 and std.mem.eql(u8, args[1], "value-sanity")) return valueSanity(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "actor-check")) return actorCheck(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "index-check")) return indexCheck(init);
    if (args.len > 1 and std.mem.eql(u8, args[1], "leak-check")) return leakCheck(init, args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "team-check")) {
        var stats = teams.LoadStats{};
        const pool = try teams.loadDir(init.arena.allocator(), init.io, "../teamset", &stats);
        std.debug.print("expert teams: {d} files, {d} loaded, {d} skipped; first {s}\n", .{ stats.files, stats.loaded, stats.skipped, if (pool.len > 0) pool[0].name else "none" });
        return;
    }
    if (args.len > 1 and std.mem.eql(u8, args[1], "size-check")) {
        std.debug.print("node {d} bytes, path {d} bytes\n", .{ search.nodeBytes(), search.pathBytes() });
        return;
    }
    if (args.len > 1 and std.mem.eql(u8, args[1], "pick-check")) return pickCheck(args);
    if (args.len > 1 and std.mem.eql(u8, args[1], "sample-teams")) return sampleTeams(init, args);
    const first_seed: u64 = if (args.len > 1) try std.fmt.parseUnsigned(u64, args[1], 10) else 1;
    const games: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 1;
    const iterations: u32 = if (args.len > 3) try std.fmt.parseUnsigned(u32, args[3], 10) else 1200;

    var wins: u32 = 0;
    var losses: u32 = 0;
    var g: u64 = 0;
    while (g < games) : (g += 1) {
        var turns: u32 = 0;
        const outcome = try playGame(std.heap.page_allocator, first_seed + g, iterations, &turns);
        if (outcome > 0) wins += 1;
        if (outcome < 0) losses += 1;
        std.debug.print("seed {d}: {s} after {d} turns\n", .{ first_seed + g, if (outcome > 0) "P1 (search) wins" else if (outcome < 0) "P2 (random) wins" else "draw/unfinished", turns });
    }
    std.debug.print("search {d} wins, {d} losses, {d} games\n", .{ wins, losses, games });
}

/// Checks the Zig value model against the PyTorch logits stored in golden_value.txt.
fn valueCheck(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    var model = try vm.Model.load(allocator, init.io, "models/value");
    defer model.deinit(allocator);

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/value/golden_value.txt", allocator, .unlimited);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var worst: f32 = 0;
    var windows: u32 = 0;
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "window ")) continue;
        var header = std.mem.tokenizeScalar(u8, line, ' ');
        _ = header.next();
        const n_steps = try std.fmt.parseUnsigned(usize, header.next().?, 10);
        _ = try std.fmt.parseUnsigned(usize, header.next().?, 10);

        const steps = try allocator.alloc([]const f32, n_steps);
        for (0..n_steps) |s| {
            const step_line = lines.next() orelse return error.TruncatedGolden;
            var groups = try allocator.alloc([]const u32, vm.GROUPS);
            var g: usize = 0;
            var parts = std.mem.splitSequence(u8, step_line, " | ");
            while (parts.next()) |part| : (g += 1) {
                if (g >= vm.GROUPS) return error.BadGroupCount;
                var ids: std.ArrayList(u32) = .empty;
                var toks = std.mem.tokenizeScalar(u8, part, ' ');
                while (toks.next()) |tok| try ids.append(allocator, try std.fmt.parseUnsigned(u32, tok, 10));
                groups[g] = ids.items;
            }
            if (g != vm.GROUPS) return error.BadGroupCount;
            steps[s] = try model.encodeStep(allocator, groups);
        }
        const logits_line = lines.next() orelse return error.TruncatedGolden;
        var logits = std.mem.tokenizeScalar(u8, logits_line["win_logits ".len..], ' ');
        var last: f32 = 0;
        while (logits.next()) |tok| last = try std.fmt.parseFloat(f32, tok);

        const logit = try model.winLogit(allocator, steps);
        worst = @max(worst, @abs(logit - last));
        windows += 1;
    }
    std.debug.print("zig value model: checked {d} windows; max |zig - pytorch| logit difference = {e}\n", .{ windows, worst });
    if (windows == 0 or !std.math.isFinite(worst) or worst > 1e-4) return error.ValueMismatch;
}

/// Checks the Zig belief model against the PyTorch marginals in golden.txt.
fn beliefCheck(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const model = try belief.Model.load(allocator, init.io, "models/belief");
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/belief/golden.txt", allocator, .unlimited);
    var lines = std.mem.splitScalar(u8, text, '\n');
    const header = lines.next() orelse return error.TruncatedGolden;
    const frame_count = try std.fmt.parseUnsigned(usize, header["frames ".len..], 10);
    const frames = try allocator.alloc([]const u32, frame_count);
    for (frames) |*frame| {
        var parts = std.mem.tokenizeScalar(u8, lines.next() orelse return error.TruncatedGolden, ' ');
        _ = parts.next();
        _ = parts.next();
        var ids: std.ArrayList(u32) = .empty;
        while (parts.next()) |tok| try ids.append(allocator, try std.fmt.parseUnsigned(u32, tok, 10));
        frame.* = ids.items;
    }
    const expected_species = try parseFloats(allocator, lines.next() orelse return error.TruncatedGolden, "species ");
    const expected_moves = try parseFloats(allocator, lines.next() orelse return error.TruncatedGolden, "moves ");
    const out = try model.infer(allocator, frames);
    var worst: f32 = 0;
    for (out.species, expected_species) |a, b| worst = @max(worst, @abs(a - b));
    for (out.moves, expected_moves) |a, b| worst = @max(worst, @abs(a - b));
    std.debug.print("zig belief model: {d} frames; max |zig - pytorch| marginal difference = {e}\n", .{ frame_count, worst });
    if (frame_count == 0 or !std.math.isFinite(worst) or worst > 1e-4) return error.BeliefMismatch;
}

fn parseFloats(allocator: std.mem.Allocator, line: []const u8, prefix: []const u8) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    var parts = std.mem.tokenizeScalar(u8, line[prefix.len..], ' ');
    while (parts.next()) |tok| try out.append(allocator, try std.fmt.parseFloat(f32, tok));
    return out.items;
}

/// Checks the Zig tokenizer against the token ids the trainer produced for golden_states.json.
fn tokenizerCheck(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const vocab = try belief.Vocab.load(allocator, init.io, "models/belief/training_metadata.json");
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/belief/golden_states.json", allocator, .unlimited);
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})).object;
    const frames_json = root.get("frames").?.array.items;
    const expected_rows = root.get("token_ids").?.array.items;

    const frames = try allocator.alloc(tokenizer.FrameJson, frames_json.len);
    for (frames, frames_json) |*dst, src| {
        dst.* = .{
            .public_state = src.object.get("public_state").?,
            .own_action = src.object.get("own_action").?,
            .opponent_action = src.object.get("opponent_action").?,
        };
    }
    const encoded = try tokenizer.encodeFrames(allocator, &vocab, frames);
    var mismatched: usize = 0;
    for (encoded, expected_rows, 0..) |ids, expected_row, index| {
        const expected = expected_row.array.items;
        var same = ids.len == expected.len;
        if (same) for (ids, expected) |a, b| {
            if (a != b.integer) same = false;
        };
        if (!same) {
            mismatched += 1;
            if (mismatched <= 3) std.debug.print("frame {d}: zig len {d}, expected len {d}\n", .{ index, ids.len, expected.len });
        }
    }
    std.debug.print("tokenizer: {d} of {d} frames match the trainer's token ids\n", .{ encoded.len - mismatched, encoded.len });
    if (encoded.len == 0 or mismatched != 0) return error.TokenizerMismatch;
}

/// Samples opponent teams from the golden frames with the given seed (for comparison with the
/// Rust implementation). Prints one team per line.
fn sampleTeams(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const seed: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 7;
    const samples: usize = if (args.len > 3) try std.fmt.parseUnsigned(usize, args[3], 10) else 2;
    const vocab = try belief.Vocab.load(allocator, init.io, "models/belief/training_metadata.json");
    const model = try belief.Model.load(allocator, init.io, "models/belief");
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/belief/golden_states.json", allocator, .unlimited);
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})).object;
    const frames_json = root.get("frames").?.array.items;
    const frames_in = try allocator.alloc(tokenizer.FrameJson, frames_json.len);
    for (frames_in, frames_json) |*dst, src| {
        dst.* = .{
            .public_state = src.object.get("public_state").?,
            .own_action = src.object.get("own_action").?,
            .opponent_action = src.object.get("opponent_action").?,
        };
    }
    const frames = try tokenizer.encodeFrames(allocator, &vocab, frames_in);
    const marginals = try model.infer(allocator, frames);
    var rng = belief.Rng.init(seed);
    for (0..samples) |_| {
        const team = try belief.sampleCompletion(allocator, &vocab, marginals, model.move_count, &.{}, &rng);
        for (team, 0..) |member, i| {
            if (i > 0) std.debug.print(" ", .{});
            std.debug.print("{s}[", .{member.species});
            for (member.moves, 0..) |m, j| std.debug.print("{s}{s}", .{ if (j > 0) "," else "", m });
            std.debug.print("]", .{});
        }
        std.debug.print("\n", .{});
    }
}

/// Replays the 43-turn reference game through the Zig observer pipeline and compares each turn's
/// win probability with bot2's reference values.
fn observerCheck(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const model = try vm.Model.load(allocator, init.io, "models/value");
    const vocab_text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/value/token_vocab.json", allocator, .unlimited);
    const vocab_json = try std.json.parseFromSliceLeaky(std.json.Value, allocator, vocab_text, .{});
    var vocab = std.StringHashMap(u32).init(allocator);
    var it = vocab_json.object.iterator();
    while (it.next()) |entry| try vocab.put(entry.key_ptr.*, @intCast(entry.value_ptr.integer));

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/value/golden_reference_sequence.json", allocator, .unlimited);
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})).object;
    const snapshots = root.get("snapshots").?.array.items;
    const reference = root.get("win_probabilities").?.array.items;

    var pipeline = vs.Pipeline.init(allocator, &model, &vocab);
    var worst: f64 = 0;
    for (snapshots, reference, 0..) |snap_json, ref, turn_index| {
        const snapshot = try vs.fromJson(allocator, snap_json);
        const ours = try pipeline.value(snapshot);
        const expected = ref.float;
        const diff = @abs(ours - expected);
        worst = @max(worst, diff);
        if (diff > 1e-3) std.debug.print("turn index {d}: zig {d:.6} reference {d:.6}\n", .{ turn_index, ours, expected });
        try pipeline.record(snapshot);
    }
    std.debug.print("observer pipeline: {d} turns; max |zig - reference| win probability = {e}\n", .{ snapshots.len, worst });
}

/// Plays a game with the search for P1 and random play for P2, recording live public frames. At
/// turn `probe_turn` the belief model is run on the frames so far and its samples are compared with
/// the opponent's actual species.
fn liveBelief(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const seed: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 1;
    const iterations: u32 = if (args.len > 3) try std.fmt.parseUnsigned(u32, args[3], 10) else 600;
    const probe_turn: u16 = 12;

    const vocab = try belief.Vocab.load(allocator, init.io, "models/belief/training_metadata.json");
    const model = try belief.Model.load(allocator, init.io, "models/belief");

    var prng = std.Random.DefaultPrng.init(seed);
    var rand = prng.random();
    var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var history = pubstate.History.init(allocator);
    var result = try battle.update(.{}, .{}, &options);
    var probed = false;
    while (result.type == .None) {
        try history.record(&battle);
        if (!probed and battle.turn >= probe_turn) {
            probed = true;
            try probeBelief(allocator, &vocab, &model, &history, &battle, rand);
        }
        const outcome = try search.search(allocator, &battle, result, .{ .iterations = iterations }, &rand);
        const n1 = battle.choices(.P1, result.p1, &choices1);
        const n2 = battle.choices(.P2, result.p2, &choices2);
        if (n1 == 0 or n2 == 0) break;
        const c1 = choices1[@min(outcome.choice, n1 - 1)];
        const c2 = choices2[rand.uintLessThan(u8, n2)];
        const before = battle;
        result = try battle.update(c1, c2, &options);
        try history.noteTransition(&before, &battle, c1, c2);
    }
    std.debug.print("game finished at turn {d}, {d} frames recorded\n", .{ battle.turn, history.frames.items.len });
}

fn probeBelief(allocator: std.mem.Allocator, vocab: *const belief.Vocab, model: *const belief.Model, history: *const pubstate.History, battle: *const pubstate.Battle, rand: std.Random) !void {
    const ids = try pubstate.frameIds(allocator, vocab, history.frames.items);
    var total: usize = 0;
    var unknown: usize = 0;
    for (ids) |frame| {
        for (frame) |id| {
            total += 1;
            if (id == belief.UNK_ID) unknown += 1;
        }
    }
    const marginals = try model.infer(allocator, ids);
    // the opponent's real party, for comparison with the samples
    var actual: std.ArrayList([]const u8) = .empty;
    for (battle.sides[1].pokemon) |pk| {
        if (pk.species == .None) continue;
        try actual.append(allocator, try vs.norm(allocator, @tagName(pk.species)));
    }
    var rng = belief.Rng.init(rand.int(u64));
    var hit: usize = 0;
    var seen_total: usize = 0;
    for (0..3) |_| {
        const team = try belief.sampleCompletion(allocator, vocab, marginals, model.move_count, &.{}, &rng);
        for (team) |member| {
            seen_total += 1;
            for (actual.items) |a| if (std.mem.eql(u8, a, member.species)) {
                hit += 1;
                break;
            };
        }
    }
    std.debug.print("probe at turn {d}: {d} frames, UNK tokens {d}/{d}; sampled species matching the true party: {d}/{d}\n", .{ battle.turn, ids.len, unknown, total, hit, seen_total });
}

fn flattenKeys(allocator: std.mem.Allocator, value: vs.V, prefix: []const u8, out: *std.ArrayList([]const u8)) !void {
    if (value != .obj) return;
    for (value.obj) |f| {
        const path = if (prefix.len == 0) f.key else try std.fmt.allocPrint(allocator, "{s}.{s}", .{ prefix, f.key });
        try out.append(allocator, path);
        try flattenKeys(allocator, f.value, path, out);
    }
}

fn flattenJson(allocator: std.mem.Allocator, value: std.json.Value, prefix: []const u8, out: *std.ArrayList([]const u8)) !void {
    if (value != .object) return;
    var it = value.object.iterator();
    while (it.next()) |entry| {
        const path = if (prefix.len == 0) entry.key_ptr.* else try std.fmt.allocPrint(allocator, "{s}.{s}", .{ prefix, entry.key_ptr.* });
        try out.append(allocator, path);
        try flattenJson(allocator, entry.value_ptr.*, path, out);
    }
}

/// Compares the key structure of a live frame with the golden frames from the trainer.
fn frameSchema(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/belief/golden_states.json", allocator, .unlimited);
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})).object;
    var golden: std.ArrayList([]const u8) = .empty;
    try flattenJson(allocator, root.get("frames").?.array.items[0].object.get("public_state").?, "", &golden);

    var prng = std.Random.DefaultPrng.init(1);
    var rand = prng.random();
    var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try battle.update(.{}, .{}, &options);
    var history = pubstate.History.init(allocator);
    try history.record(&battle);
    var ours: std.ArrayList([]const u8) = .empty;
    try flattenKeys(allocator, history.frames.items[0].public, "", &ours);

    var missing: usize = 0;
    for (golden.items) |g| {
        var found = false;
        for (ours.items) |o| if (std.mem.eql(u8, o, g)) {
            found = true;
        };
        if (!found) {
            missing += 1;
            std.debug.print("golden key not produced: {s}\n", .{g});
        }
    }
    var extra: usize = 0;
    for (ours.items) |o| {
        var found = false;
        for (golden.items) |g| if (std.mem.eql(u8, o, g)) {
            found = true;
        };
        if (!found) {
            extra += 1;
            std.debug.print("extra key produced: {s}\n", .{o});
        }
    }
    std.debug.print("schema: {d} golden keys, {d} produced keys, {d} missing, {d} extra\n", .{ golden.items.len, ours.items.len, missing, extra });
}

/// Plays a game and, every turn, builds the public snapshot from accumulated knowledge. Checks
/// that the opponent part carries no hidden fields, that knowledge only grows, and that the value
/// pipeline accepts every snapshot.
fn knowledgeCheck(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const seed: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 1;
    const vocab_text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/value/token_vocab.json", allocator, .unlimited);
    const vocab_json = try std.json.parseFromSliceLeaky(std.json.Value, allocator, vocab_text, .{});
    var vocab = std.StringHashMap(u32).init(allocator);
    var it = vocab_json.object.iterator();
    while (it.next()) |entry| try vocab.put(entry.key_ptr.*, @intCast(entry.value_ptr.integer));
    const model = try vm.Model.load(allocator, init.io, "models/value");
    var pipeline = vs.Pipeline.init(allocator, &model, &vocab);

    var prng = std.Random.DefaultPrng.init(seed);
    var rand = prng.random();
    var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var result = try battle.update(.{}, .{}, &options);
    var knowledge = know.Knowledge{};
    knowledge = know.initial(&battle);
    var turns: usize = 0;
    var hidden_violations: usize = 0;
    var worst_prob: f64 = 0;
    while (result.type == .None) : (turns += 1) {
        const snapshot = try know.observerSnapshot(allocator, &battle, knowledge, false);
        var keys: std.ArrayList([]const u8) = .empty;
        try flattenKeys(allocator, snapshot, "", &keys);
        for (keys.items) |k| {
            if (std.mem.startsWith(u8, k, "opponent.")) {
                for (HIDDEN) |h| if (std.mem.indexOf(u8, k, h) != null) {
                    hidden_violations += 1;
                    std.debug.print("hidden key in opponent view: {s}\n", .{k});
                };
            }
        }
        const p = try pipeline.value(snapshot);
        worst_prob = @max(worst_prob, p);
        try pipeline.record(snapshot);

        const outcome = try search.search(allocator, &battle, result, .{ .iterations = 200 }, &rand);
        const n1 = battle.choices(.P1, result.p1, &choices1);
        const n2 = battle.choices(.P2, result.p2, &choices2);
        if (n1 == 0 or n2 == 0) break;
        const c1 = choices1[@min(outcome.choice, n1 - 1)];
        const c2 = choices2[rand.uintLessThan(u8, n2)];
        const before = battle;
        result = try battle.update(c1, c2, &options);
        const step = know.Knowledge.afterTransition(&before, &battle, c1, c2);
        const merged = know.Knowledge.merge(knowledge, step);
        knowledge = merged;
    }
    std.debug.print("knowledge check: {d} turns, hidden-key violations {d}, max value {d:.3}\n", .{ turns, hidden_violations, worst_prob });
}

/// A float from the environment, or `default` when unset or unparsable.
fn envFloat(name: [*:0]const u8, default: f32) f32 {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseFloat(f32, std.mem.span(raw)) catch default;
}

const HIDDEN = [_][]const u8{ "speed", "sleep_elapsed", "rest_turns", "attack", "defense", "special_attack", "special_defense", "level", "binding", "trapping", "force_trapped", "stats", "dvs" };

fn loadVocab(allocator: std.mem.Allocator, io: std.Io) !std.StringHashMap(u32) {
    const vocab_text = try std.Io.Dir.cwd().readFileAlloc(io, "models/value/token_vocab.json", allocator, .unlimited);
    const vocab_json = try std.json.parseFromSliceLeaky(std.json.Value, allocator, vocab_text, .{});
    var vocab = std.StringHashMap(u32).init(allocator);
    var it = vocab_json.object.iterator();
    while (it.next()) |entry| try vocab.put(entry.key_ptr.*, @intCast(entry.value_ptr.integer));
    return vocab;
}

/// (a) Replays the golden reference game as a chain of path nodes; (b) checks that sibling nodes
/// built from one parent don't disturb it and that a node is reproducible.
fn pathCheck(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const model = try vm.Model.load(allocator, init.io, "models/value");
    const vocab = try loadVocab(allocator, init.io);

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/value/golden_reference_sequence.json", allocator, .unlimited);
    const root_json = (try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})).object;
    const snapshots = root_json.get("snapshots").?.array.items;
    const reference = root_json.get("win_probabilities").?.array.items;
    const nodes = try allocator.alloc(pathmod.Path, snapshots.len);
    var worst: f64 = 0;
    for (snapshots, reference, 0..) |snap_json, ref, i| {
        const snapshot = try vs.fromJson(allocator, snap_json);
        const parent: ?*const pathmod.Path = if (i == 0) null else &nodes[i - 1];
        nodes[i] = try pathmod.Path.extend(allocator, &model, &vocab, parent, .{}, snapshot);
        worst = @max(worst, @abs(nodes[i].value() - ref.float));
    }
    std.debug.print("path chain: {d} turns; max |path - reference| = {e}\n", .{ snapshots.len, worst });

    // Verify the block16 schedule against independent full-window recomputation,
    // including rollover at decisions 32, 48 and 64 and the following KV appends.
    const block_nodes = try allocator.alloc(pathmod.Path, 70);
    var block_error: f32 = 0;
    for (0..70) |i| {
        const snap = try vs.fromJson(allocator, snapshots[i % snapshots.len]);
        block_nodes[i] = try pathmod.Path.extend(allocator, &model, &vocab, if (i == 0) null else &block_nodes[i - 1], .{}, snap);
        block_nodes[i].block_window = true;
        const length = @min(i + 1, 16 + (i + 1) % 16);
        if (block_nodes[i].window_len != length) return error.BlockWindowLengthMismatch;
        const steps = try allocator.alloc([]const f32, length);
        for (0..length) |j| steps[j] = block_nodes[i + 1 - length + j].step;
        const exact = try model.winLogit(allocator, steps);
        block_error = @max(block_error, @abs(exact - block_nodes[i].logit));
    }
    std.debug.print("block16 history: 70 decisions; max logit error {e}\n", .{block_error});
    if (!std.math.isFinite(block_error) or block_error > 1e-4 or worst > 1e-4) return error.PathMismatch;

    // (c) a 16-step root view, extended incrementally, equals the direct window
    const view = try pathmod.Path.rootView(allocator, &model, &nodes[40], 16);
    const next = try vs.fromJson(allocator, snapshots[41]);
    const extended = try pathmod.Path.extend(allocator, &model, &vocab, &view, .{}, next);
    var direct: [17][]const f32 = undefined;
    for (0..17) |k| direct[k] = nodes[25 + k].step;
    const direct_logit = try model.winLogit(allocator, &direct);
    std.debug.print("16-step root view + 1 step: logit {d:.7} vs direct window {d:.7}\n", .{ extended.logit, direct_logit });

    // (b) siblings from the same parent
    var prng = std.Random.DefaultPrng.init(3);
    var rand = prng.random();
    var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    _ = try battle.update(.{}, .{}, &options);
    const root_snap = try know.observerSnapshot(allocator, &battle, .{}, false);
    const root = try pathmod.Path.extend(allocator, &model, &vocab, null, .{}, root_snap);
    const parent_own = root.memory.own.items.len;
    const parent_foe = root.memory.foe.items.len;
    const parent_step = root.memory.step;
    var sibling_bad: usize = 0;
    var repeat_bad: usize = 0;
    for (1..5) |move| {
        var b1 = battle;
        var b2 = battle;
        const c = pkmn.Choice{ .type = .Move, .data = @intCast(move) };
        _ = try b1.update(c, pkmn.Choice{ .type = .Move, .data = 1 }, &options);
        _ = try b2.update(c, pkmn.Choice{ .type = .Move, .data = 1 }, &options);
        const first = try pathmod.Path.child(allocator, &model, &vocab, &root, &battle, &b1, c, pkmn.Choice{ .type = .Move, .data = 1 });
        const second = try pathmod.Path.child(allocator, &model, &vocab, &root, &battle, &b2, c, pkmn.Choice{ .type = .Move, .data = 1 });
        if (root.memory.own.items.len != parent_own or root.memory.foe.items.len != parent_foe or root.memory.step != parent_step) sibling_bad += 1;
        if (!std.mem.eql(f32, first.step, second.step)) repeat_bad += 1;
    }
    // screens and volatiles must reach the snapshot the model reads
    battle.sides[0].active.volatiles.Reflect = true;
    const screened = try know.observerSnapshot(allocator, &battle, .{}, false);
    var keys: std.ArrayList([]const u8) = .empty;
    try flattenKeys(allocator, screened, "", &keys);
    var has_reflect_condition = false;
    for (keys.items) |k| if (std.mem.eql(u8, k, "user.conditions.reflect")) {
        has_reflect_condition = true;
    };
    const user = vs.get(screened, "user").?;
    const volatiles = vs.get(user, "volatiles").?;
    std.debug.print("reflect in conditions: {}; volatiles listed: {d}\n", .{ has_reflect_condition, volatiles.list.len });
    std.debug.print("siblings: parent memory disturbed {d} times; repeated child differs {d} times\n", .{ sibling_bad, repeat_bad });
}

const universe_count = 4;
const Battle = search.Battle;

/// Plays full games with the value-model search for P1 (random P2). The real game is kept as a chain
/// of path nodes, and each search starts from the latest one, so the root sees the game so far.
fn valueGame(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const first_seed: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 1;
    const games: u64 = if (args.len > 3) try std.fmt.parseUnsigned(u64, args[3], 10) else 1;
    const iterations: u32 = if (args.len > 4) try std.fmt.parseUnsigned(u32, args[4], 10) else 200;
    const model = try vm.Model.load(allocator, init.io, "models/value");
    const vocab = try loadVocab(allocator, init.io);
    const belief_vocab = try bel.Vocab.load(allocator, init.io, "models/belief/training_metadata.json");
    const belief_model = try bel.Model.load(allocator, init.io, "models/belief");
    const move_table = try completion.Table.load(allocator, init.io, "models/completion/gen1ou_moves.txt");

    var wins: u32 = 0;
    var losses: u32 = 0;
    var g: u64 = 0;
    while (g < games) : (g += 1) {
        var game_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer game_arena.deinit();
        const galloc = game_arena.allocator();
        var prng = std.Random.DefaultPrng.init(first_seed + g);
        var rand = prng.random();
        var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var result = try battle.update(.{}, .{}, &options);
        var knowledge = know.Knowledge{};
        knowledge = know.initial(&battle);
        var real = try galloc.create(pathmod.Path);
        real.* = try pathmod.Path.extend(galloc, &model, &vocab, null, knowledge, try know.observerSnapshot(galloc, &battle, knowledge, false));
        var history = pubhist.History.init(galloc);
        var belief_rng = bel.Rng.init(first_seed + g + 1000);
        while (result.type == .None) {
            try history.record(&battle);
            const frame_ids = try pubhist.frameIds(galloc, &belief_vocab, history.frames.items);
            const marginals = try belief_model.infer(galloc, frame_ids);
            const known = try universe.knownParty(galloc, &battle, knowledge);
            const universes = try galloc.alloc(Battle, universe_count);
            for (universes) |*u| {
                const sampled = try bel.sampleCompletion(galloc, &belief_vocab, marginals, belief_model.move_count, known, &belief_rng);
                u.* = try universe.build(galloc, &battle, knowledge, sampled[known.len..], &move_table, &belief_rng);
            }
            const cfg = search.Config{ .iterations = iterations, .model = &model, .vocab = &vocab, .root_knowledge = knowledge, .root_path = real, .universes = universes };
            const outcome = try search.search(std.heap.page_allocator, &battle, result, cfg, &rand);
            const n1 = battle.choices(.P1, result.p1, &choices1);
            const n2 = battle.choices(.P2, result.p2, &choices2);
            if (n1 == 0 or n2 == 0) break;
            const c1 = choices1[@min(outcome.choice, n1 - 1)];
            const c2 = choices2[rand.uintLessThan(u8, n2)];
            const before = battle;
            result = try battle.update(c1, c2, &options);
            try history.noteTransition(&before, &battle, c1, c2);
            knowledge = know.Knowledge.merge(knowledge, know.Knowledge.afterTransition(&before, &battle, c1, c2));
            if (result.type == .None) {
                const next = try galloc.create(pathmod.Path);
                next.* = try pathmod.Path.extend(galloc, &model, &vocab, real, knowledge, try know.observerSnapshot(galloc, &battle, knowledge, false));
                real = next;
            }
        }
        if (result.type == .Win) wins += 1;
        if (result.type == .Lose) losses += 1;
        std.debug.print("seed {d}: {s} after {d} turns\n", .{ first_seed + g, @tagName(result.type), battle.turn });
    }
    std.debug.print("value search {d} wins, {d} losses, {d} games\n", .{ wins, losses, games });
}

/// Random-vs-random engine games with invariant checks: every update succeeds, HP stays within
/// [0, max], and the game ends within the turn cap (a stall is reported).
fn engineFuzz(args: []const []const u8) !void {
    const first_seed: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 1;
    const games: u64 = if (args.len > 3) try std.fmt.parseUnsigned(u64, args[3], 10) else 100;
    const turn_cap: u32 = 2000;
    var counts = [_]u32{ 0, 0, 0, 0 };
    var stalls: u32 = 0;
    var errors: u32 = 0;
    var bad_hp: u32 = 0;
    var total_turns: u64 = 0;
    var g: u64 = 0;
    while (g < games) : (g += 1) {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        var prng = std.Random.DefaultPrng.init(first_seed + g);
        var rand = prng.random();
        var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var result = battle.update(.{}, .{}, &options) catch {
            errors += 1;
            continue;
        };
        var ok = true;
        while (result.type == .None and battle.turn < turn_cap) {
            const n1 = battle.choices(.P1, result.p1, &choices1);
            const n2 = battle.choices(.P2, result.p2, &choices2);
            if (n1 == 0 or n2 == 0) break;
            const c1 = choices1[rand.uintLessThan(u8, n1)];
            const c2 = choices2[rand.uintLessThan(u8, n2)];
            result = battle.update(c1, c2, &options) catch {
                errors += 1;
                ok = false;
                break;
            };
            for (0..2) |side| {
                for (battle.sides[side].pokemon) |mon| {
                    if (mon.hp > mon.stats.hp) {
                        bad_hp += 1;
                        ok = false;
                    }
                }
            }
        }
        if (result.type == .None and battle.turn >= turn_cap) stalls += 1;
        if (!ok) continue;
        total_turns += battle.turn;
        switch (result.type) {
            .Win => counts[0] += 1,
            .Lose => counts[1] += 1,
            .Tie => counts[2] += 1,
            else => counts[3] += 1,
        }
    }
    std.debug.print("engine-fuzz {d} games: win {d} lose {d} tie {d} unfinished {d}; stalls {d}, update errors {d}, hp violations {d}, mean turns {d}\n", .{
        games, counts[0], counts[1], counts[2], counts[3], stalls, errors, bad_hp, if (games > 0) total_turns / games else 0,
    });
}

/// Status mechanics against the Gen 1 rules: residual damage (burn and poison lose max/16, at least 1),
/// and the stat penalties (burn halves Attack, paralysis quarters Speed) on the active Pokémon.
fn statusCheck() !void {
    const max_dvs: pkmn.gen1.DVs = .{ .atk = 15, .def = 15, .spe = 15, .spc = 15 };
    const splash = [_]pkmn.gen1.Move{ .Splash, .Splash, .Splash, .Splash };
    const statuses = [_]pkmn.gen1.Status{ .BRN, .PSN };
    for (statuses) |st| {
        var failures: u32 = 0;
        var seed: u64 = 1;
        while (seed <= 20) : (seed += 1) {
            const p2 = pkmn.gen1.helpers.Pokemon{ .species = .Chansey, .moves = &splash, .dvs = max_dvs, .status = pkmn.gen1.Status.init(st) };
            const p1 = pkmn.gen1.helpers.Pokemon{ .species = .Tauros, .moves = &splash, .dvs = max_dvs };
            var battle = pkmn.gen1.helpers.Battle.init(seed, &.{p1}, &.{p2});
            var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
            _ = try battle.update(.{}, .{}, &options);
            const max = battle.sides[1].pokemon[0].stats.hp;
            const before = battle.sides[1].pokemon[0].hp;
            _ = try battle.update(.{ .type = .Move, .data = 1 }, .{ .type = .Move, .data = 1 }, &options);
            const after = battle.sides[1].pokemon[0].hp;
            const expected: u16 = @max(1, max / 16);
            if (before - after != expected) failures += 1;
        }
        std.debug.print("{s} residual: expected max(1, maxhp/16) each turn, failures {d}/20\n", .{ @tagName(st), failures });
    }
    // burn halves the attacker's Attack in the damage formula: compare one Body Slam, both ways
    const slam = [_]pkmn.gen1.Move{ .BodySlam, .Splash, .Splash, .Splash };
    var dmg = [_]u16{ 0, 0 };
    for ([_]?pkmn.gen1.Status{ null, .BRN }, 0..) |st, i| {
        var attacker = pkmn.gen1.helpers.Pokemon{ .species = .Tauros, .moves = &slam, .dvs = max_dvs };
        if (st) |s2| attacker.status = pkmn.gen1.Status.init(s2);
        const defender = pkmn.gen1.helpers.Pokemon{ .species = .Chansey, .moves = &splash, .dvs = max_dvs };
        var battle = pkmn.gen1.helpers.Battle.init(7, &.{attacker}, &.{defender});
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        _ = try battle.update(.{}, .{}, &options);
        const before = battle.sides[1].pokemon[0].hp;
        _ = try battle.update(.{ .type = .Move, .data = 1 }, .{ .type = .Move, .data = 1 }, &options);
        dmg[i] = before - battle.sides[1].pokemon[0].hp;
    }
    std.debug.print("burned attacker Body Slam damage {d} vs healthy {d} (expect roughly half; a 2.0 ratio is correct)\n", .{ dmg[1], dmg[0] });
}

/// Name of a legal option as the policy server reads it: "switch <species>" or a move name.

/// The observer snapshot with the legal options, from P1's side or (`swap`) from P2's side.
fn serializeSnapshot(galloc: std.mem.Allocator, battle: *const Battle, k: know.Knowledge, legal: []const []const u8, swap: bool) ![]u8 {
    var view = battle.*;
    var kk = k;
    if (swap) {
        std.mem.swap(@TypeOf(view.sides[0]), &view.sides[0], &view.sides[1]);
        kk = know.Knowledge{ .seen = .{ k.seen[1], k.seen[0] }, .moves = .{ k.moves[1], k.moves[0] }, .used = .{ k.used[1], k.used[0] } };
    }
    const snap = try know.observerSnapshot(galloc, &view, kk, false);
    const legal_items = try galloc.alloc(vs.V, legal.len);
    for (legal, legal_items) |name, *item| item.* = .{ .str = name };
    const base = snap.obj;
    const fields = try galloc.alloc(vs.Field, base.len + 1);
    @memcpy(fields[0..base.len], base);
    fields[base.len] = .{ .key = "legal", .value = .{ .list = legal_items } };
    var out: std.ArrayList(u8) = .empty;
    try policy.writeJson(galloc, &out, .{ .obj = fields });
    return out.items;
}


fn duelPlayer(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !*policy.Client {
    if (std.mem.eql(u8, name, "tauros")) return policy.Client.initTauros(allocator, io);
    if (std.mem.eql(u8, name, "recurrent")) return policy.Client.initRecurrent(allocator, io);
    if (std.mem.eql(u8, name, "epoch20")) return policy.Client.initEpoch20(allocator, io);
    return policy.Client.initExeggcuteVariant(allocator, io, name);
}

/// Two Exeggcute policies play each other: the blend player (argmax of the proposer blend) against the
/// judge player (the router's own decision). Games come in pairs on the same expert teams and seed with
/// the sides swapped. `duel <seed> <games>`
fn duel(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const first_seed: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 1;
    const games: u64 = if (args.len > 3) try std.fmt.parseUnsigned(u64, args[3], 10) else 250;
    // players: blend | judge (Exeggcute variants) | tauros (TaurosV0, argmax); `blend` below is player A, `judge` player B
    const name_a = if (args.len > 4) args[4] else "blend";
    const name_b = if (args.len > 5) args[5] else "judge";
    const blend = try duelPlayer(allocator, init.io, name_a);
    defer blend.deinit();
    const judge = try duelPlayer(allocator, init.io, name_b);
    defer judge.deinit();
    var pool_stats = teams.LoadStats{};
    const pool = try teams.loadDir(allocator, init.io, "../teamset", &pool_stats);
    std.debug.print("duel: {s} (A) vs {s} (B), {d} games, pool {d} of {d} files\n", .{ name_a, name_b, games, pool_stats.loaded, pool_stats.files });
    var blend_wins: u32 = 0;
    var judge_wins: u32 = 0;
    var draws: u32 = 0;
    var g: u64 = 0;
    while (g < games) : (g += 1) {
        var game_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer game_arena.deinit();
        const galloc = game_arena.allocator();
        const pair = g / 2;
        const blend_is_p1 = g % 2 == 0;
        // (repo copy) reciprocal pairs: both games of a pair are the same battle (P1 has team_p1, P2 team_p2, the same battle
        // seed) and the two players swap seats, so each plays both teams from the same position (the original kept each
        // player's team and swapped only the seats)
        var team_rng = std.Random.DefaultPrng.init(first_seed + pair + 5000);
        const entry_p1 = pool[team_rng.random().uintLessThan(usize, pool.len)];
        const entry_p2 = pool[team_rng.random().uintLessThan(usize, pool.len)];
        const team_p1 = entry_p1.mons;
        const team_p2 = entry_p2.mons;
        var prng = std.Random.DefaultPrng.init(first_seed + pair);
        var rand = prng.random();
        const engine_seed = rand.int(u64);
        var battle = pkmn.gen1.helpers.Battle.init(engine_seed, team_p1, team_p2);
        // (repo copy) DUEL_REPLAY=1: what a replay in another simulator needs. The engine's seed is Pokémon Showdown's Gen 5
        // RNG seed (as "gen5,<16 hex digits>"); every update's choices by name and the engine's state after it
        const replay = envFloat("DUEL_REPLAY", 0) > 0;
        if (replay) std.debug.print("REPLAY start game {d} seed {x:0>16} p1 {s} p2 {s}\n", .{ g + 1, engine_seed, entry_p1.name, entry_p2.name });
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var result = try battle.update(.{}, .{}, &options);
        var knowledge = know.initial(&battle);
        var last_actions = [2]LastAction{ .{}, .{} };
        while (result.type == .None) {
            const n1 = battle.choices(.P1, result.p1, &choices1);
            const n2 = battle.choices(.P2, result.p2, &choices2);
            if (n1 == 0 or n2 == 0) break;
            // (repo copy) TaurosV0 gets Showdown's names for Struggle, Recharge and locked moves (snapshot.taurosName)
            const p1_tauros = std.mem.eql(u8, if (blend_is_p1) name_a else name_b, "tauros");
            const p2_tauros = std.mem.eql(u8, if (blend_is_p1) name_b else name_a, "tauros");
            const names1 = try galloc.alloc([]const u8, n1);
            for (0..n1) |i| names1[i] = if (p1_tauros) try snapmod.taurosName(galloc, &battle, 0, choices1[i]) else try snapmod.duelName(galloc, &battle, 0, choices1[i]);
            const names2 = try galloc.alloc([]const u8, n2);
            for (0..n2) |i| names2[i] = if (p2_tauros) try snapmod.taurosName(galloc, &battle, 1, choices2[i]) else try snapmod.duelName(galloc, &battle, 1, choices2[i]);
            const snap1 = try serializeDuelSnapshot(galloc, &battle, knowledge, names1, false, last_actions);
            const snap2 = try serializeDuelSnapshot(galloc, &battle, knowledge, names2, true, last_actions);
            const client1 = if (blend_is_p1) blend else judge;
            const client2 = if (blend_is_p1) judge else blend;
            const reply1 = try client1.choose(first_seed + g, snap1, names1, rand);
            const reply2 = try client2.choose(first_seed + g, snap2, names2, rand);
            // (repo copy) DUEL_TRACE=1: every decision of both players, for auditing a game afterwards
            if (envFloat("DUEL_TRACE", 0) > 0) std.debug.print("  trace game {d} turn {d}: P1 ({s}) {s} | P2 ({s}) {s}\n", .{ g + 1, battle.turn, if (blend_is_p1) name_a else name_b, reply1, if (blend_is_p1) name_b else name_a, reply2 });
            var c1 = choices1[0];
            for (0..n1) |i| {
                if (std.mem.eql(u8, names1[i], reply1)) c1 = choices1[i];
            }
            var c2 = choices2[0];
            for (0..n2) |i| {
                if (std.mem.eql(u8, names2[i], reply2)) c2 = choices2[i];
            }
            const before = battle;
            result = try battle.update(c1, c2, &options);
            if (replay) {
                var line: std.ArrayList(u8) = .empty;
                for (0..2) |side| {
                    const sd = battle.sides[side];
                    try line.appendSlice(galloc, try std.fmt.allocPrint(galloc, " p{d}={s}", .{ side + 1, @tagName(sd.pokemon[sd.order[0] - 1].species) }));
                    for (sd.pokemon) |mon| try line.appendSlice(galloc, try std.fmt.allocPrint(galloc, ",{s}:{d}:{d}", .{ @tagName(mon.species), mon.hp, mon.status }));
                }
                std.debug.print("REPLAY step game {d} turn {d} c1 {s} c2 {s} result {s} state{s}\n", .{ g + 1, battle.turn, try optionName(galloc, &before, 0, c1), try optionName(galloc, &before, 1, c2), @tagName(result.type), line.items });
            }
            knowledge = know.Knowledge.merge(knowledge, know.Knowledge.afterTransition(&before, &battle, c1, c2));
            last_actions = .{ try lastAction(galloc, &before, &battle, 0, c1), try lastAction(galloc, &before, &battle, 1, c2) };
        }
        var winner: []const u8 = "draw";
        if (result.type == .Win) {
            winner = if (blend_is_p1) "A" else "B";
        } else if (result.type == .Lose) {
            winner = if (blend_is_p1) "B" else "A";
        }
        if (std.mem.eql(u8, winner, "A")) blend_wins += 1 else if (std.mem.eql(u8, winner, "B")) judge_wins += 1 else draws += 1;
        std.debug.print("game {d} seed {d} A is {s}: {s} after {d} turns (A {d}, B {d}, draws {d})\n", .{ g + 1, first_seed + pair, if (blend_is_p1) "P1" else "P2", winner, battle.turn, blend_wins, judge_wins, draws });
    }
    std.debug.print("duel: A {d}, B {d}, draws {d} of {d} games; fallbacks A {d} B {d}\n", .{ blend_wins, judge_wins, draws, games, blend.fallbacks, judge.fallbacks });
}

/// Search (as P1) against the raw epoch-20 policy (as P2). Teams alternate between games so each
/// side plays both rosters. Options after the game count and search budget:
///   args[5] = model | hp | priors   (default model: value model + belief universes)
///   args[6] = visits | prior        (final root rule; prior = log(actor prior) + scale * (Q - V))
fn tournament(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const first_seed: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 1;
    const games: u64 = if (args.len > 3) try std.fmt.parseUnsigned(u64, args[3], 10) else 100;
    const iterations: u32 = if (args.len > 4) try std.fmt.parseUnsigned(u32, args[4], 10) else 200;
    const mode = if (args.len > 5) args[5] else "model";
    const use_model = std.mem.eql(u8, mode, "model") or std.mem.eql(u8, mode, "priors");
    const use_priors = std.mem.eql(u8, mode, "priors");
    const final_arg = if (args.len > 6) args[6] else "visits";
    const improved_fpu = std.mem.eql(u8, final_arg, "adaptive-fpu") or std.mem.eql(u8, final_arg, "adaptive-v2");
    const improved_history = std.mem.eql(u8, final_arg, "adaptive-history") or std.mem.eql(u8, final_arg, "adaptive-v2");
    const diag = std.mem.eql(u8, final_arg, "widediag");
    // widexact: widecont with exact chance outcomes per cell (chance_enum.zig) instead of 8 draws.
    // klnash / klnash-max: the same cells (exact chance, every reply) solved as a KL-anchored
    // Bayesian game at the root; the move is drawn from (or is the argmax of) its strategy.
    // Env: KL_TAU, KL_TAU_OPP (anchor temperatures), CHANCE_MIN_PATH, CHANCE_MAX_LEAVES.
    const exact = std.mem.eql(u8, final_arg, "widexact");
    const kl_argmax = std.mem.eql(u8, final_arg, "klnash-max");
    const kl = kl_argmax or std.mem.eql(u8, final_arg, "klnash");
    const cont = std.mem.eql(u8, final_arg, "widecont") or exact or kl;
    const vr = cont or std.mem.eql(u8, final_arg, "widevr");
    const wide = diag or vr or std.mem.eql(u8, final_arg, "wide");
    var diag_count: u32 = 0;
    const n_worlds: usize = if (vr) 8 else if (wide) 16 else universe_count;
    const n_confirm: usize = if (kl) 0 else if (vr) 4 else if (wide) 8 else 0;
    const kl_tau = envFloat("KL_TAU", 0.02);
    const kl_tau_opp = envFloat("KL_TAU_OPP", 0.02);
    const chance_min_path = envFloat("CHANCE_MIN_PATH", 0.05);
    const chance_max_leaves: u8 = @intFromFloat(envFloat("CHANCE_MAX_LEAVES", 16));
    if (exact or kl) std.debug.print("exact chance: min path {d}, max leaves {d}\n", .{ chance_min_path, chance_max_leaves });
    if (kl) std.debug.print("kl root game: tau {d}, opponent tau {d}, {s}\n", .{ kl_tau, kl_tau_opp, if (kl_argmax) "argmax" else "sampled" });
    const prior_final: search.FinalRule = if (kl) .kl_nash else if (std.mem.eql(u8, final_arg, "prior")) .prior_value else if (std.mem.eql(u8, final_arg, "nash")) .nash else if (std.mem.eql(u8, final_arg, "adaptive") or std.mem.eql(u8, final_arg, "wide") or std.mem.eql(u8, final_arg, "widediag") or vr or improved_fpu or improved_history) .adaptive else .visits;
    // Mean-value racing can eliminate actions required by a mixed equilibrium.
    const racing = false;
    if ((prior_final == .adaptive or prior_final == .kl_nash) and !use_priors) return error.AdaptiveRequiresPriors;
    // SEARCH_MODEL=recurrent: the recurrent model (tools/recurrent_search_server.py, weights from RECURRENT_SEARCH_WEIGHTS or
    // RECURRENT_WEIGHTS) gives the search its values and both sides' actor priors, instead of the windowed value model
    // SEARCH_MODEL=native: the same recurrent model computed in-process (rnative.zig, exported by tools/export_recurrent.py into
    // RECURRENT_NATIVE_DIR, default models/recurrent); NATIVE_CHECK=1 also runs the search server and compares every node
    const model_name = if (std.c.getenv("SEARCH_MODEL")) |v| std.mem.span(v) else "";
    const native = std.mem.eql(u8, model_name, "native");
    const recurrent = native or std.mem.eql(u8, model_name, "recurrent");
    if (recurrent and !use_model) return error.RecurrentNeedsModelMode;
    var model_storage: vm.Model = undefined;
    var vocab_storage: std.StringHashMap(u32) = undefined;
    if (!recurrent) {
        model_storage = try vm.Model.load(allocator, init.io, "models/value");
        vocab_storage = try loadVocab(allocator, init.io);
    }
    const model_ptr: ?*const vm.Model = if (recurrent) null else &model_storage;
    const vocab_ptr: ?*const std.StringHashMap(u32) = if (recurrent) null else &vocab_storage;
    if (native) {
        const dir = if (std.c.getenv("RECURRENT_NATIVE_DIR")) |v| std.mem.span(v) else "models/recurrent";
        rnative.active = try rnative.Native.load(allocator, init.io, dir);
        std.debug.print("search model: native recurrent ({s}, weights {s})\n", .{ dir, rnative.active.?.model.weights_sha256 });
        if (envFloat("NATIVE_CHECK", 0) > 0) {
            remote.active = try remote.Remote.init(allocator, init.io);
            std.debug.print("NATIVE_CHECK: every node is also made by the search server and compared\n", .{});
        }
    } else if (recurrent) {
        remote.active = try remote.Remote.init(allocator, init.io);
        std.debug.print("search model: recurrent\n", .{});
    }
    defer if (native and remote.active != null) std.debug.print("NATIVE_CHECK: {d} nodes compared, {d} mismatches; largest logit difference otherwise {e:.2}\n", .{ pathmod.Path.check_nodes, pathmod.Path.check_mismatches, pathmod.Path.check_worst });
    defer if (remote.active) |server| server.deinit();
    const belief_vocab = try bel.Vocab.load(allocator, init.io, "models/belief/training_metadata.json");
    const belief_model = try bel.Model.load(allocator, init.io, "models/belief");
    const move_table = try completion.Table.load(allocator, init.io, "models/completion/gen1ou_moves.txt");
    // args[7] = epoch20 (default: the epoch-20 actor plays P2) or tauros (TaurosV0 plays P2)
    // args[7] = epoch20 (the epoch-20 actor plays P2), tauros (TaurosV0) or exeggcute (Metamon's ensemble)
    const opponent = if (args.len > 7) args[7] else "epoch20";
    const vs_tauros = std.mem.eql(u8, opponent, "tauros");
    const vs_exeggcute = std.mem.eql(u8, opponent, "exeggcute");
    const lowered_names = vs_tauros or vs_exeggcute;
    const opp_client = if (vs_tauros) try policy.Client.initTauros(allocator, init.io) else if (vs_exeggcute) try policy.Client.initExeggcute(allocator, init.io) else null;
    defer if (opp_client) |client| client.deinit();
    // args[8] = fixed (the two built-in teams, alternating sides) or expert (random files from the 807 pool, both sides)
    const expert_teams = args.len > 8 and std.mem.eql(u8, args[8], "expert");
    var pool_stats = teams.LoadStats{};
    const pool: []teams.Team = if (expert_teams) try teams.loadDir(allocator, init.io, "../teamset", &pool_stats) else &.{};
    if (!vs_tauros and !vs_exeggcute and !std.mem.eql(u8, opponent, "epoch20")) return error.UnknownOpponent;
    if (recurrent and !vs_tauros and !vs_exeggcute) return error.RecurrentNeedsTaurosOrExeggcute; // (epoch20 plays from the windowed model)
    if (expert_teams and pool.len == 0) return error.EmptyTeamPool;
    std.debug.print("opponent {s}, teams {s}", .{ opponent, if (expert_teams) "expert" else "fixed" });
    if (expert_teams) std.debug.print(" (pool {d} of {d} files)", .{ pool_stats.loaded, pool_stats.files });
    std.debug.print("\n", .{});

    var search_wins: u32 = 0;
    var policy_wins: u32 = 0;
    var draws: u32 = 0;
    var g: u64 = 0;
    while (g < games) : (g += 1) {
        var game_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer game_arena.deinit();
        const galloc = game_arena.allocator();
        const paired_teams = args.len > 9 and std.mem.eql(u8, args[9], "paired-teams");
        const fixture_seed = first_seed + (if (paired_teams) g / 2 else g);
        var team_rng = std.Random.DefaultPrng.init(fixture_seed + 5000);
        var search_index = if (expert_teams) team_rng.random().uintLessThan(usize, pool.len) else 0;
        var policy_index = if (expert_teams) team_rng.random().uintLessThan(usize, pool.len) else 0;
        if (paired_teams and g % 2 == 1) std.mem.swap(usize, &search_index, &policy_index);
        const search_team: []const pkmn.gen1.helpers.Pokemon = if (expert_teams) pool[search_index].mons else if (g % 2 == 0) &team1 else &team2;
        const policy_team: []const pkmn.gen1.helpers.Pokemon = if (expert_teams) pool[policy_index].mons else if (g % 2 == 0) &team2 else &team1;
        std.debug.print("game seed {d}: search team {s}; policy team {s}; mode {s}; root {s}; {d} simulations x {d} universes\n", .{fixture_seed, if (expert_teams) pool[search_index].name else "fixed", if (expert_teams) pool[policy_index].name else "fixed", mode, final_arg, iterations, n_worlds});
        var prng = std.Random.DefaultPrng.init(fixture_seed);
        var rand = prng.random();
        var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), search_team, policy_team);
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var result = try battle.update(.{}, .{}, &options);
        var knowledge = know.Knowledge{};
        knowledge = know.initial(&battle);
        var real = try galloc.create(pathmod.Path);
        if (remote.active) |server| try server.reset();
        real.* = try pathmod.Path.turn(galloc, model_ptr, vocab_ptr, null, &battle, &battle, result, .{}, .{}, knowledge, 0);
        // the epoch-20 opponent's own chain (windowed model only)
        var opponent_path = try galloc.create(pathmod.Path);
        if (!recurrent) opponent_path.* = try pathmod.Path.extend(galloc, model_ptr.?, vocab_ptr.?, null, know.swapKnowledge(knowledge), try know.perspectiveSnapshot(galloc, &battle, know.swapKnowledge(knowledge), 1));
        // the real game's decision points, for the opponent's history in the search (search.Config.history)
        var points: std.ArrayList(search.HistoryPoint) = .empty;
        // HISTORY_CHECK=1 (recurrent model): the real opponent's own chain along the game, against which the search's P2 root
        // chain rebuilt in the true world (the opponent's real team) is compared at every decision; they must agree
        const history_check = recurrent and remote.active != null and envFloat("HISTORY_CHECK", 0) > 0; // (it compares on the search server)
        var p2_real = try galloc.create(pathmod.Path);
        if (history_check) p2_real.* = try pathmod.Path.turn(galloc, null, null, null, &battle, &battle, result, .{}, .{}, knowledge, 1);
        var history = pubhist.History.init(galloc);
        var belief_rng = bel.Rng.init(fixture_seed + 1000);
        var decisions: u32 = 0;
        while (result.type == .None and decisions < 1000) : (decisions += 1) {
            try history.record(&battle);
            try points.append(galloc, .{ .battle = battle, .result = result, .knowledge = knowledge });
            const frame_ids = try pubhist.frameIds(galloc, &belief_vocab, history.frames.items);
            const n1 = battle.choices(.P1, result.p1, &choices1);
            const n2 = battle.choices(.P2, result.p2, &choices2);
            // universes are only needed when we have a real choice to make (speedup: a forced move
            // skips the belief model, the universe builds and the search)
            const universes: []const Battle = if (n1 > 1) blk: {
                const marginals = try belief_model.infer(galloc, frame_ids);
                const known = try universe.knownParty(galloc, &battle, knowledge);
                const built = try galloc.alloc(Battle, n_worlds + n_confirm);
                for (built) |*u| {
                    const sampled = try bel.sampleCompletion(galloc, &belief_vocab, marginals, belief_model.move_count, known, &belief_rng);
                    u.* = try universe.build(galloc, &battle, knowledge, sampled[known.len..], &move_table, &belief_rng);
                }
                break :blk built;
            } else &.{};
            if (n1 == 0 or n2 == 0) break;

            // The opponent's move: TaurosV0 when it is the opponent, otherwise the epoch-20 policy server.
            // TaurosV0 names options as lowercase normalized strings ("bodyslam", "switch tauros"); the
            // epoch-20 server matches names normalized too, so the same list works for both.
            const names2 = try galloc.alloc([]const u8, n2);
            for (0..n2) |i| {
                const raw = try optionName(galloc, &battle, 1, choices2[i]);
                // TaurosV0 spells the no-op "no move" (index 0) where our names say "pass" or "recharge"
                names2[i] = if (!lowered_names) raw else if (std.mem.eql(u8, raw, "pass") or std.mem.eql(u8, raw, "recharge")) "no move" else try std.ascii.allocLowerString(galloc, raw);
            }
            var c2 = choices2[0];
            if (opp_client) |client| {
                const snapshot = try serializeSnapshot(galloc, &battle, knowledge, names2, true);
                const reply = try client.choose(first_seed + g, snapshot, names2, rand);
                for (0..n2) |i| if (std.mem.eql(u8, names2[i], reply)) { c2 = choices2[i]; };
            } else {
                // Same exported epoch20 actor, full 32-step history, legal argmax, no search.
                var best: f32 = -std.math.inf(f32);
                for (choices2[0..n2]) |choice| {
                    const index = know.actorIndex(&battle, 1, choice) orelse return error.InvalidActorAction;
                    const score = opponent_path.actor[index];
                    if (score > best) { best = score; c2 = choice; }
                }
            }

            if (history_check and points.items.len >= 2) {
                const server = remote.active.?;
                const m = try server.mark();
                const check_cfg = search.Config{ .recurrent = true, .history = points.items, .root_knowledge = knowledge };
                const rebuilt = try search.opponentRoot(galloc, check_cfg, &battle, result);
                var worst: f32 = 0;
                for (choices2[0..n2]) |choice| {
                    const idx = know.actorIndex(&battle, 1, choice) orelse continue;
                    const a = rebuilt.actor[idx];
                    const b = p2_real.actor[idx];
                    if (std.math.isInf(a) and std.math.isInf(b)) continue;
                    worst = @max(worst, @abs(a - b));
                }
                std.debug.print("HISTORY_CHECK decision {d}: largest opponent logit difference {d:.6}, win logit {d:.6} vs {d:.6}\n", .{ decisions + 1, worst, rebuilt.logit, p2_real.logit });
                // which rebuilt past states differ from the real ones (with the true world they should not)
                for (points.items[0 .. points.items.len - 1], 0..) |point, pi| {
                    const w = universe.pastWorld(&point.battle, point.knowledge, &battle, knowledge);
                    for (0..6) |slot| {
                        const x = w.sides[1].pokemon[slot];
                        const y = point.battle.sides[1].pokemon[slot];
                        for (0..4) |j| {
                            if (x.moves[j].id != y.moves[j].id or x.moves[j].pp != y.moves[j].pp)
                                std.debug.print("  point {d} slot {d} move {d}: rebuilt {s} pp {d}, real {s} pp {d} (known uses {any})\n", .{ pi, slot, j, @tagName(x.moves[j].id), x.moves[j].pp, @tagName(y.moves[j].id), y.moves[j].pp, point.knowledge.used[1][slot] });
                        }
                        if (x.hp != y.hp or x.status != y.status or x.species != y.species) std.debug.print("  point {d} slot {d}: hp/status/species differ\n", .{ pi, slot });
                    }
                    const xa = w.sides[1].active.moves;
                    const ya = point.battle.sides[1].active.moves;
                    for (0..4) |j| if (xa[j].id != ya[j].id or xa[j].pp != ya[j].pp) std.debug.print("  point {d} active move {d}: rebuilt pp {d}, real pp {d}\n", .{ pi, j, xa[j].pp, ya[j].pp });
                }
                try server.release(m);
            }
            var cfg = if (use_model) search.Config{ .iterations = iterations, .model = model_ptr, .vocab = vocab_ptr, .recurrent = recurrent, .history = points.items, .root_knowledge = knowledge, .root_path = real, .universes = universes, .actor_priors = use_priors, .root_final = prior_final, .root_racing = racing, .first_play_value = improved_fpu, .block_history = improved_history, .exploration = if (improved_fpu) 1.25 else 1.4, .max_depth = if (prior_final == .adaptive) 6 else 24, .switch_penalty = if (prior_final == .adaptive) 0 else 0.01 } else search.Config{ .iterations = iterations, .universes = universes, .evaluator = search.hpValue, .root_final = prior_final, .root_racing = racing, .first_play_value = improved_fpu, .block_history = improved_history, .exploration = if (improved_fpu) 1.25 else 1.4, .max_depth = if (prior_final == .adaptive) 6 else 24, .switch_penalty = if (prior_final == .adaptive) 0 else 0.01 };
            if (wide and universes.len > 0) {
                cfg.universes = universes[0..n_worlds];
                cfg.confirm_universes = universes[n_worlds..];
                cfg.wide = true;
                cfg.adaptive_candidates = pkmn.MAX_CHOICES;
                if (vr) {
                    // one ply, chance averaged over 8 paired draws, likely replies + one tail draw
                    cfg.max_depth = 1;
                    cfg.leaf_only_value = true;
                    cfg.chance_samples = 8;
                    cfg.reply_mass = 0.95;
                    if (cont) {
                        // two greedy actor turns from each leaf before the value read
                        cfg.leaf_only_value = false;
                        cfg.continuation = 2;
                    }
                }
                if (exact or kl) {
                    cfg.exact_chance = true;
                    cfg.chance_min_path = chance_min_path;
                    cfg.chance_max_leaves = chance_max_leaves;
                }
                if (kl) {
                    // SEARCH_THREADS: threads of the kl_nash root (native recurrent model)
                    cfg.threads = @intFromFloat(envFloat("SEARCH_THREADS", 1));
                    cfg.switch_penalty = 0;
                    cfg.kl_tau = kl_tau;
                    cfg.kl_tau_opp = kl_tau_opp;
                    cfg.kl_argmax = kl_argmax;
                }
            }
            if (diag and n1 > 1) {
                diag_count += 1;
                // every third searched decision; the diagnostic uses its own RNG so play is unchanged
                if (diag_count % 3 == 1) {
                    var drng = std.Random.DefaultPrng.init(fixture_seed * 100003 + diag_count);
                    var dr = drng.random();
                    var dcfg = cfg;
                    dcfg.confirm_universes = &.{};
                    std.debug.print("DIAG game {d} decision {d} tag {d}\n", .{ fixture_seed, decisions + 1, fixture_seed * 10000 + diag_count });
                    try search.diagnose(std.heap.page_allocator, &battle, result, dcfg, &dr, 4, @intCast(fixture_seed * 10000 + diag_count));
                }
            }
            const outcome = try search.search(std.heap.page_allocator, &battle, result, cfg, &rand);
            const c1 = choices1[@min(outcome.choice, n1 - 1)];
            if (wide and n1 > 1) std.debug.print("confirm: ran {any}, lower {d:.4}\n", .{ outcome.confirm_ran, outcome.confirm_lower });
            if (prior_final == .adaptive and n1 > 1) std.debug.print("tree: mean depth {d:.2}, max depth {d}, reused {d}\n", .{if (outcome.simulations > 0) @as(f64, @floatFromInt(outcome.depth_total)) / @as(f64, @floatFromInt(outcome.simulations)) else 0, outcome.max_depth, outcome.reused});
            if (prior_final == .adaptive and n1 > 1) std.debug.print("adaptive: simulations {d}, sweeps {d}, actor {d}, choice {d}, lower {d:.4}, early {any}\n", .{outcome.simulations, outcome.sweeps, outcome.actor_choice, outcome.choice, outcome.override_lower, outcome.early_stop});
            if ((exact or kl) and n1 > 1) std.debug.print("exact: leaves {d}, cells {d}\n", .{ outcome.leaves, outcome.simulations });
            if (kl and n1 > 1) {
                std.debug.print("klnash: actor {d}, choice {d}, value {d:.4}, iterations {d}, residual {e:.1}, x", .{ outcome.actor_choice, outcome.choice, outcome.game_value, outcome.sweeps, outcome.residual });
                for (0..n1) |a| std.debug.print(" {d:.3}", .{outcome.strategy[a]});
                std.debug.print(" q", .{});
                for (0..n1) |a| std.debug.print(" {d:.3}", .{outcome.value[a]});
                std.debug.print("\n", .{});
            }
            std.debug.print("turn {d} decision {d}: search {s}; policy {s}; root win {d:.4}\n", .{ battle.turn, decisions + 1, try optionName(galloc, &battle, 0, c1), try optionName(galloc, &battle, 1, c2), real.value() });

            const before = battle;
            points.items[points.items.len - 1].c1 = c1;
            points.items[points.items.len - 1].c2 = c2;
            result = try battle.update(c1, c2, &options);
            try history.noteTransition(&before, &battle, c1, c2);
            knowledge = know.Knowledge.merge(knowledge, know.Knowledge.afterTransition(&before, &battle, c1, c2));
            if (result.type == .None) {
                const next = try galloc.create(pathmod.Path);
                if (recurrent) {
                    next.* = try pathmod.Path.turn(galloc, null, null, real, &before, &battle, result, c1, c2, .{}, 0);
                } else {
                    next.* = try pathmod.Path.extend(galloc, model_ptr.?, vocab_ptr.?, real, knowledge, try know.observerSnapshot(galloc, &battle, knowledge, false));
                }
                real = next;
                if (history_check) {
                    const next2 = try galloc.create(pathmod.Path);
                    next2.* = try pathmod.Path.turn(galloc, null, null, p2_real, &before, &battle, result, c1, c2, .{}, 1);
                    p2_real = next2;
                }
                if (!recurrent) {
                    const next_opponent = try galloc.create(pathmod.Path);
                    next_opponent.* = try pathmod.Path.childSide(galloc, model_ptr.?, vocab_ptr.?, opponent_path, &before, &battle, c1, c2, 1);
                    opponent_path = next_opponent;
                }
            }
        }
        var winner: []const u8 = "draw";
        if (result.type == .Win) {
            winner = "search";
            search_wins += 1;
        } else if (result.type == .Lose) {
            winner = "policy";
            policy_wins += 1;
        } else {
            draws += 1;
        }
        std.debug.print("game {d} seed {d} search team {s}: {s} after {d} turns\n", .{ g + 1, fixture_seed, if (g % 2 == 0) "A" else "B", winner, battle.turn });
    }
    std.debug.print("tournament: search {d}, policy {d}, draws {d} of {d} games; opponent server fallbacks {d}\n", .{ search_wins, policy_wins, draws, games, if (opp_client) |client| client.fallbacks else @as(u32, 0) });
}

/// Root decision on a simple position with perfect information: Tauros (attacker) against a
/// Chansey that can only Splash. Any attacking move is obviously better than switching.
fn pickCheck(args: []const []const u8) !void {
    const iterations: u32 = if (args.len > 2) try std.fmt.parseUnsigned(u32, args[2], 10) else 1000;
    const max_dvs: pkmn.gen1.DVs = .{ .atk = 15, .def = 15, .spe = 15, .spc = 15 };
    const attacker_moves = [_]pkmn.gen1.Move{ .BodySlam, .HyperBeam, .Blizzard, .Earthquake };
    const splash = [_]pkmn.gen1.Move{ .Splash, .Splash, .Splash, .Splash };
    const p1 = [_]pkmn.gen1.helpers.Pokemon{ .{ .species = .Tauros, .moves = &attacker_moves, .dvs = max_dvs }, .{ .species = .Snorlax, .moves = &splash, .dvs = max_dvs } };
    const p2 = [_]pkmn.gen1.helpers.Pokemon{ .{ .species = .Chansey, .moves = &splash, .dvs = max_dvs }, .{ .species = .Snorlax, .moves = &splash, .dvs = max_dvs } };
    var battle = pkmn.gen1.helpers.Battle.init(1, &p1, &p2);
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
    const result = try battle.update(.{}, .{}, &options);
    var rand_state = std.Random.DefaultPrng.init(7);
    var rand = rand_state.random();
    const worlds = [_]Battle{battle};
    const cfg = search.Config{ .iterations = iterations, .universes = &worlds, .evaluator = search.hpValue };
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    const n1 = battle.choices(.P1, result.p1, &choices1);
    const out = try search.search(arena.allocator(), &battle, result, cfg, &rand);
    for (0..n1) |k| std.debug.print("choice {d} {s}:{d} visits {d:.0} value {d:.3}\n", .{ k, @tagName(choices1[k].type), choices1[k].data, out.visits[k], out.value[k] });
    std.debug.print("chosen {d}\n", .{out.choice});
}

/// Sleep and paralysis against the Gen 1 rules:
///  - a Pokémon that starts a turn asleep deals no damage that turn, and it is asleep for as many
///    turns as its sleep counter (1..7);
///  - a Pokémon that starts a turn fully paralysed (not asleep) loses the turn with probability
///    63/256 (about 24.6%); paralysis also quarters Speed on the active Pokémon.
fn mechanicsCheck() !void {
    const max_dvs: pkmn.gen1.DVs = .{ .atk = 15, .def = 15, .spe = 15, .spc = 15 };
    const hypnosis = [_]pkmn.gen1.Move{ .Hypnosis, .Splash, .Splash, .Splash };
    const slam = [_]pkmn.gen1.Move{ .BodySlam, .Splash, .Splash, .Splash };
    const thunder = [_]pkmn.gen1.Move{ .ThunderWave, .Splash, .Splash, .Splash };
    var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);

    // Sleep: Chansey (huge HP) puts Tauros to sleep with Hypnosis; Tauros attacks every turn it can.
    var sleep_hist = [_]u32{0} ** 8;
    var sleep_turns_total: u32 = 0;
    var sleep_damage_violations: u32 = 0;
    var sleep_length_mismatch: u32 = 0;
    var episodes: u32 = 0;
    var seed: u64 = 1;
    while (seed <= 300) : (seed += 1) {
        const p1 = pkmn.gen1.helpers.Pokemon{ .species = .Chansey, .moves = &hypnosis, .dvs = max_dvs };
        const p2 = pkmn.gen1.helpers.Pokemon{ .species = .Tauros, .moves = &slam, .dvs = max_dvs };
        var battle = pkmn.gen1.helpers.Battle.init(seed, &.{p1}, &.{p2});
        _ = try battle.update(.{}, .{}, &options);
        var counter: ?u8 = null;
        var asleep_turns: u32 = 0;
        var turn: u32 = 0;
        while (turn < 40) : (turn += 1) {
            const foe = &battle.sides[1].pokemon[0];
            const asleep_at_start = pkmn.gen1.Status.is(foe.status, .SLP);
            const own_before = battle.sides[0].pokemon[0].hp;
            _ = try battle.update(.{ .type = .Move, .data = if (counter == null) @as(u8, 1) else @as(u8, 2) }, .{ .type = .Move, .data = 1 }, &options);
            const damage = own_before - battle.sides[0].pokemon[0].hp;
            if (asleep_at_start) {
                asleep_turns += 1;
                if (damage != 0) sleep_damage_violations += 1;
            }
            if (counter != null and !pkmn.gen1.Status.is(battle.sides[1].pokemon[0].status, .SLP)) break;
            if (battle.sides[0].pokemon[0].hp == 0) break;
            battle.sides[0].pokemon[0].hp = battle.sides[0].pokemon[0].stats.hp;
            if (!asleep_at_start and pkmn.gen1.Status.is(battle.sides[1].pokemon[0].status, .SLP) and counter == null) {
                counter = battle.sides[1].pokemon[0].status & 7;
            }
        }
        if (counter) |c| {
            episodes += 1;
            sleep_hist[c] += 1;
            sleep_turns_total += asleep_turns;
            if (asleep_turns != c) sleep_length_mismatch += 1;
        }
    }
    std.debug.print("sleep: {d} episodes; counter histogram 1..7 = ", .{episodes});
    for (1..8) |c| std.debug.print("{d} ", .{sleep_hist[c]});
    std.debug.print("\n  asleep-at-start turns {d}, damage dealt while asleep {d} (expect 0), episodes whose asleep turns != counter {d}\n", .{ sleep_turns_total, sleep_damage_violations, sleep_length_mismatch });

    // Paralysis: Chansey lands Thunder Wave, then Tauros attacks each turn.
    var par_turns: u32 = 0;
    var par_no_damage: u32 = 0;
    var spe_before: u16 = 0;
    var spe_after: u16 = 0;
    seed = 1;
    while (seed <= 200) : (seed += 1) {
        const p1 = pkmn.gen1.helpers.Pokemon{ .species = .Chansey, .moves = &thunder, .dvs = max_dvs };
        const p2 = pkmn.gen1.helpers.Pokemon{ .species = .Tauros, .moves = &slam, .dvs = max_dvs };
        var battle = pkmn.gen1.helpers.Battle.init(seed, &.{p1}, &.{p2});
        _ = try battle.update(.{}, .{}, &options);
        if (seed == 1) spe_before = battle.sides[1].active.stats.spe;
        _ = try battle.update(.{ .type = .Move, .data = 1 }, .{ .type = .Move, .data = 2 }, &options);
        const par_base = battle;
        var turn: u32 = 0;
        while (turn < 60) : (turn += 1) {
            battle = par_base;
            battle.rng = pkmn.gen1.PRNG.init(seed * 1000 + turn);
            const foe = &battle.sides[1].pokemon[0];
            const par_at_start = pkmn.gen1.Status.is(foe.status, .PAR) and !pkmn.gen1.Status.is(foe.status, .SLP);
            const own_before = battle.sides[0].pokemon[0].hp;
            _ = try battle.update(.{ .type = .Move, .data = 2 }, .{ .type = .Move, .data = 1 }, &options);
            if (par_at_start) {
                par_turns += 1;
                if (battle.sides[0].pokemon[0].hp == own_before) par_no_damage += 1;
            }
            if (seed == 1 and turn == 0) spe_after = battle.sides[1].active.stats.spe;
        }
    }
    const frac: f64 = @as(f64, @floatFromInt(par_no_damage)) / @as(f64, @floatFromInt(@max(par_turns, 1)));
    std.debug.print("paralysis: {d} paralysed turns, {d} lost ({d:.3}); expect about 63/256 = 0.246 (Body Slam misses are rare)\n", .{ par_turns, par_no_damage, frac });
    std.debug.print("  Tauros active Speed before paralysis {d}, after {d} (expect about a quarter)\n", .{ spe_before, spe_after });
    if (sleep_damage_violations != 0 or sleep_length_mismatch != 0 or frac < 0.20 or frac > 0.30 or spe_after != spe_before / 4) return error.MechanicsMismatch;
}

/// Value model on hand-made states: it should rate a healthier P1 higher, and a damaged P2 higher.
fn valueSanity(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const model = try vm.Model.load(allocator, init.io, "models/value");
    const vocab = try loadVocab(allocator, init.io);
    const names = [_][]const u8{ "full vs full", "P2 active 10% HP", "P1 active 10% HP", "P2 whole team 10%", "P1 whole team 10%" };
    for (names, 0..) |name, state| {
        var battle = pkmn.gen1.helpers.Battle.init(1, &team1, &team2);
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        _ = try battle.update(.{}, .{}, &options);
        switch (state) {
            1 => battle.sides[1].pokemon[battle.sides[1].order[0] - 1].hp = battle.sides[1].pokemon[battle.sides[1].order[0] - 1].stats.hp / 10,
            2 => battle.sides[0].pokemon[battle.sides[0].order[0] - 1].hp = battle.sides[0].pokemon[battle.sides[0].order[0] - 1].stats.hp / 10,
            3 => for (&battle.sides[1].pokemon) |*pk| {
                pk.hp = pk.stats.hp / 10;
            },
            4 => for (&battle.sides[0].pokemon) |*pk| {
                pk.hp = pk.stats.hp / 10;
            },
            else => {},
        }
        const knowledge = know.Knowledge{};
        const snap = try know.observerSnapshot(allocator, &battle, knowledge, false);
        const path = try pathmod.Path.extend(allocator, &model, &vocab, null, knowledge, snap);
        std.debug.print("{s}: P1 win probability {d:.3}\n", .{ name, path.value() });
    }
}

/// Actor logits along the golden snapshot chain, one plain-text line per turn, for comparison with
/// the PyTorch policy engine (tools/actor_reference.py).
fn actorCheck(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const model = try vm.Model.load(allocator, init.io, "models/value");
    const vocab_text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/value/token_vocab.json", allocator, .unlimited);
    const vocab_json = try std.json.parseFromSliceLeaky(std.json.Value, allocator, vocab_text, .{});
    var vocab = std.StringHashMap(u32).init(allocator);
    var it = vocab_json.object.iterator();
    while (it.next()) |entry| try vocab.put(entry.key_ptr.*, @intCast(entry.value_ptr.integer));
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, "models/value/golden_reference_sequence.json", allocator, .unlimited);
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})).object;
    const snapshots = root.get("snapshots").?.array.items;
    var pipeline = vs.Pipeline.init(allocator, &model, &vocab);
    for (snapshots, 0..) |snap_json, turn| {
        const snapshot = try vs.fromJson(allocator, snap_json);
        const logits = try pipeline.actor(snapshot);
        std.debug.print("turn {d}", .{turn});
        for (logits) |l| std.debug.print(" {d:.6}", .{l});
        std.debug.print("\n", .{});
        try pipeline.record(snapshot);
    }
}

/// Random games for checking the action-slot mapping: at every decision of either side, writes
/// one line per legal option: `side option zig_index` plus the observer snapshot JSON, for
/// tools/index_reference.py to compare with the policy server's own legal_actions mapping.
fn indexCheck(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    var out_file = try std.Io.Dir.cwd().createFile(init.io, "logs/index_check.txt", .{});
    defer out_file.close(init.io);
    var w_buf: [4096]u8 = undefined;
    var writer = out_file.writer(init.io, &w_buf);
    var g: u64 = 0;
    while (g < 20) : (g += 1) {
        var prng = std.Random.DefaultPrng.init(100 + g);
        var rand = prng.random();
        var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), &team1, &team2);
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var result = try battle.update(.{}, .{}, &options);
        var knowledge = know.Knowledge{};
        knowledge = know.initial(&battle);
        var turn: u32 = 0;
        while (result.type == .None and turn < 200) : (turn += 1) {
            const n1 = battle.choices(.P1, result.p1, &choices1);
            const n2 = battle.choices(.P2, result.p2, &choices2);
            if (n1 == 0 or n2 == 0) break;
            for ([_]usize{ 0, 1 }, [_]u32{ n1, n2 }) |side, n| {
                const choices = if (side == 0) &choices1 else &choices2;
                const names = try allocator.alloc([]const u8, n);
                for (0..n) |i| names[i] = try optionName(allocator, &battle, side, choices[i]);
                const snapshot = try serializeSnapshot(allocator, &battle, knowledge, names, side == 1);
                try writer.interface.print("snapshot {d} {s}\n", .{ g, snapshot });
                for (0..n) |i| {
                    const idx = know.actorIndex(&battle, side, choices[i]) orelse 99;
                    try writer.interface.print("option {d} {d} {s} {d}\n", .{ g, side, names[i], idx });
                }
            }
            const c1 = choices1[rand.uintLessThan(u8, n1)];
            const c2 = choices2[rand.uintLessThan(u8, n2)];
            try history_placeholder(c1, c2);
            const before = battle;
            result = try battle.update(c1, c2, &options);
            knowledge = know.Knowledge.merge(knowledge, know.Knowledge.afterTransition(&before, &battle, c1, c2));
        }
    }
    try writer.interface.flush();
}

fn history_placeholder(c1: pkmn.Choice, c2: pkmn.Choice) !void {
    _ = c1;
    _ = c2;
}


/// Perturbs what `side` keeps hidden from `knowledge`'s observer: unseen party members get different
/// HP and status, and unrevealed move slots get a different move and PP. Public facts (species and
/// HP of seen members, revealed moves, boosts) are left alone.
fn perturbHidden(battle: *Battle, side: usize, k: know.Knowledge, rand: std.Random) void {
    const statuses = [_]u8{ 0, pkmn.gen1.Status.init(.BRN), pkmn.gen1.Status.init(.PSN), pkmn.gen1.Status.init(.PAR) };
    const replacements = [_]pkmn.gen1.Move{ .Splash, .Tackle, .Psychic, .Thunderbolt };
    for (0..6) |slot| {
        const mon = &battle.sides[side].pokemon[slot];
        if (mon.species == .None) continue;
        // the active Pokémon is always public (its HP and status are shown), so it keeps its state
        const is_active = slot == battle.sides[side].order[0] - 1;
        if (!k.seen[side][slot] and !is_active) {
            mon.hp = @max(1, mon.stats.hp / (2 + rand.uintLessThan(u16, 4)));
            mon.status = statuses[rand.uintLessThan(usize, statuses.len)];
        }
        for (0..4) |j| {
            if (j >= mon.moves.len) break;
            if (std.mem.indexOfScalar(pkmn.gen1.Move, &k.moves[side][slot], mon.moves[j].id) == null) {
                mon.moves[j].id = replacements[rand.uintLessThan(usize, replacements.len)];
                mon.moves[j].pp = 1;
            }
        }
    }
}

/// Information-leakage check on random games between random expert teams. Each sampled decision is
/// compared with a copy in which only the side the observer must not see is changed:
///   - the observer's snapshot (what the epoch-20 policy and the value model read) must not change;
///   - the public history frames (the belief model's input) must not change;
///   - the sampled universes must be identical, given the same belief random stream;
///   - on every 10th decision, the full search must choose the same action with the same visits.
fn leakCheck(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const games: u64 = if (args.len > 2) try std.fmt.parseUnsigned(u64, args[2], 10) else 10;
    const iterations: u32 = if (args.len > 3) try std.fmt.parseUnsigned(u32, args[3], 10) else 200;
    var stats = teams.LoadStats{};
    const pool = try teams.loadDir(allocator, init.io, "../teamset", &stats);
    const model = try vm.Model.load(allocator, init.io, "models/value");
    const vocab = try loadVocab(allocator, init.io);
    const belief_vocab = try bel.Vocab.load(allocator, init.io, "models/belief/training_metadata.json");
    const belief_model = try bel.Model.load(allocator, init.io, "models/belief");
    const move_table = try completion.Table.load(allocator, init.io, "models/completion/gen1ou_moves.txt");

    var decisions: u32 = 0;
    var searched: u32 = 0;
    var snapshot_leaks: u32 = 0;
    var history_leaks: u32 = 0;
    var universe_leaks: u32 = 0;
    var lead_mismatch: u32 = 0;
    var control_misses: u32 = 0;
    var public_controls: u32 = 0;
    var search_leaks: u32 = 0;
    var g: u64 = 0;
    while (g < games) : (g += 1) {
        var game_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer game_arena.deinit();
        const galloc = game_arena.allocator();
        var prng = std.Random.DefaultPrng.init(first_seed_base + g);
        var rand = prng.random();
        const a = pool[rand.uintLessThan(usize, pool.len)].mons;
        const b = pool[rand.uintLessThan(usize, pool.len)].mons;
        var battle = pkmn.gen1.helpers.Battle.init(rand.int(u64), a, b);
        var options = pkmn.battle.options(pkmn.protocol.NULL, pkmn.gen1.chance.NULL, pkmn.gen1.calc.NULL);
        var choices1: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var choices2: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
        var result = try battle.update(.{}, .{}, &options);
        var knowledge = know.Knowledge{};
        knowledge = know.initial(&battle);
        var history = pubhist.History.init(galloc);
        // the real game's observer path (the production search's root path)
        var real = try galloc.create(pathmod.Path);
        real.* = try pathmod.Path.extend(galloc, &model, &vocab, null, knowledge, try know.observerSnapshot(galloc, &battle, knowledge, false));
        var turn: u32 = 0;
        while (result.type == .None and turn < 300) : (turn += 1) {
            const n1 = battle.choices(.P1, result.p1, &choices1);
            const n2 = battle.choices(.P2, result.p2, &choices2);
            if (n1 == 0 or n2 == 0) break;
            decisions += 1;

            // 1. the observer snapshots: P2's hidden state must not reach P1's snapshot, and P1's
            // hidden state must not reach P2's
            const names1 = try galloc.alloc([]const u8, n1);
            for (0..n1) |i| names1[i] = try optionName(galloc, &battle, 0, choices1[i]);
            const names2 = try galloc.alloc([]const u8, n2);
            for (0..n2) |i| names2[i] = try optionName(galloc, &battle, 1, choices2[i]);
            var perturbed = battle;
            perturbHidden(&perturbed, 1, knowledge, rand);
            const snap1 = try serializeSnapshot(galloc, &battle, knowledge, names1, false);
            const snap1_p = try serializeSnapshot(galloc, &perturbed, knowledge, names1, false);
            if (!std.mem.eql(u8, snap1, snap1_p)) snapshot_leaks += 1;
            var perturbed2 = battle;
            perturbHidden(&perturbed2, 0, knowledge, rand);
            const snap2 = try serializeSnapshot(galloc, &battle, knowledge, names2, true);
            const snap2_p = try serializeSnapshot(galloc, &perturbed2, knowledge, names2, true);
            if (!std.mem.eql(u8, snap2, snap2_p)) snapshot_leaks += 1;
            // control: a public change (the opponent's active HP) must move the snapshot
            var public_change = battle;
            const foe_active = battle.sides[1].order[0] - 1;
            // Force a visible change even when low nonzero HP rounds to 0%.
            public_change.sides[1].pokemon[foe_active].hp = 0;
            const snap_c = try serializeSnapshot(galloc, &public_change, knowledge, names1, false);
            if (battle.sides[1].pokemon[foe_active].hp > 0) {
                public_controls += 1;
                if (std.mem.eql(u8, snap1, snap_c)) control_misses += 1;
            }

            // 2. the public history and the belief input, from P1's side (P2 perturbed)
            // both histories see the same turn: the same chosen actions, recorded from each copy
            const c1 = choices1[rand.uintLessThan(u8, n1)];
            const c2 = choices2[rand.uintLessThan(u8, n2)];
            // the frame recorded from the true battle and from the perturbed copy, from the same public
            // history: only the true battle's frame is kept
            const before_len = history.frames.items.len;
            try history.record(&battle);
            try history.record(&perturbed);
            const ids = try pubhist.frameIds(galloc, &belief_vocab, history.frames.items[before_len .. before_len + 1]);
            const ids_p = try pubhist.frameIds(galloc, &belief_vocab, history.frames.items[before_len + 1 .. before_len + 2]);
            history.frames.shrinkRetainingCapacity(before_len + 1);
            if (!sameIds(ids, ids_p)) history_leaks += 1;

            // 3. universes, from the same belief stream
            const known = try universe.knownParty(galloc, &battle, knowledge);
            const known_p = try universe.knownParty(galloc, &perturbed, knowledge);
            if (known.len != known_p.len) universe_leaks += 1;
            const marginals = try belief_model.infer(galloc, ids);
            var rng_s = bel_rng_for(g, turn);
            var rng_sp = bel_rng_for(g, turn);
            const sampled = try bel.sampleCompletion(galloc, &belief_vocab, marginals, belief_model.move_count, known, &rng_s);
            const sampled_p = try bel.sampleCompletion(galloc, &belief_vocab, marginals, belief_model.move_count, known_p, &rng_sp);
            var rng_u = bel_rng_for(g, turn);
            var rng_up = bel_rng_for(g, turn);
            const u = try universe.build(galloc, &battle, knowledge, sampled[known.len..], &move_table, &rng_u);
            const u_p = try universe.build(galloc, &perturbed, knowledge, sampled_p[known_p.len..], &move_table, &rng_up);
            if (!sameUniverse(&u, &u_p)) universe_leaks += 1;
            // the opponent's active Pokémon is public: the universe must keep it
            const lead = battle.sides[1].order[0] - 1;
            if (u.sides[1].pokemon[lead].species != battle.sides[1].pokemon[lead].species) lead_mismatch += 1;

            // 4. the full search on every 10th decision
            if (decisions % 10 == 0) {
                searched += 1;
                const worlds = [_]Battle{u};
                const worlds_p = [_]Battle{u_p};
                var p1 = std.Random.DefaultPrng.init(first_seed_base + g * 1000 + turn + 99);
                var p2 = std.Random.DefaultPrng.init(first_seed_base + g * 1000 + turn + 99);
                var r1 = p1.random();
                var r2 = p2.random();
                // production configuration: value model on the observer paths, actor priors for both sides
                // in the tree, Nash root with racing
                const cfg = search.Config{ .iterations = iterations, .model = &model, .vocab = &vocab, .root_knowledge = knowledge, .root_path = real, .universes = &worlds, .actor_priors = true, .root_final = .nash, .root_racing = true };
                const cfg_p = search.Config{ .iterations = iterations, .model = &model, .vocab = &vocab, .root_knowledge = knowledge, .root_path = real, .universes = &worlds_p, .actor_priors = true, .root_final = .nash, .root_racing = true };
                const o1 = try search.search(std.heap.page_allocator, &battle, result, cfg, &r1);
                const o2 = try search.search(std.heap.page_allocator, &perturbed, result, cfg_p, &r2);
                if (o1.choice != o2.choice or !std.mem.eql(f32, &o1.visits, &o2.visits)) search_leaks += 1;
            }

            // play the turn and update the observer's knowledge and path
            const before = battle;
            result = try battle.update(c1, c2, &options);
            try history.noteTransition(&before, &battle, c1, c2);
            knowledge = know.Knowledge.merge(knowledge, know.Knowledge.afterTransition(&before, &battle, c1, c2));
            if (result.type == .None) {
                const next = try galloc.create(pathmod.Path);
                next.* = try pathmod.Path.extend(galloc, &model, &vocab, real, knowledge, try know.observerSnapshot(galloc, &battle, knowledge, false));
                real = next;
            }
        }
    }
    std.debug.print("leak-check: {d} games, {d} decisions, {d} searched\n", .{ games, decisions, searched });
    std.debug.print("  observer snapshots changed by hidden state: {d}\n", .{snapshot_leaks});
    std.debug.print("  public history changed by hidden state: {d}\n", .{history_leaks});
    std.debug.print("  sampled universes changed by hidden state: {d}\n", .{universe_leaks});
    std.debug.print("  search decisions changed by hidden state: {d} of {d}\n", .{ search_leaks, searched });
    std.debug.print("  universes whose opponent active differs from the public one: {d}\n", .{lead_mismatch});
    std.debug.print("  control: public-HP changes that failed to move the snapshot: {d} of {d}\n", .{ control_misses, public_controls });
    if (snapshot_leaks + history_leaks + universe_leaks + search_leaks + lead_mismatch + control_misses != 0)
        return error.InformationLeak;
}

const first_seed_base: u64 = 9000;

fn bel_rng_for(g: u64, turn: u32) bel.Rng {
    return bel.Rng.init(first_seed_base + g * 1000 + turn + 77);
}

fn sameIds(a: [][]u32, b: [][]u32) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!std.mem.eql(u32, x, y)) return false;
    }
    return true;
}

fn sameUniverse(a: *const Battle, b: *const Battle) bool {
    for (0..6) |slot| {
        const x = a.sides[1].pokemon[slot];
        const y = b.sides[1].pokemon[slot];
        if (x.species != y.species or x.hp != y.hp or x.status != y.status) return false;
        for (0..4) |j| {
            if (x.moves[j].id != y.moves[j].id or x.moves[j].pp != y.moves[j].pp) return false;
        }
    }
    return true;
}

/// `rmodel-check DIR CASES`: the native recurrent model (rmodel.zig, weights in DIR) on the cases of a JSON file (a list of
/// {"ids": {...}, "floats": {...}, "h": [...]}, as tools/check_rmodel.py writes it); one line per case: the win logit, the nine
/// logits and the new state, for comparison with PyTorch.
fn rmodelCheck(init: std.process.Init, args: []const []const u8) !void {
    const allocator = init.arena.allocator();
    const model = try rmodel.Model.load(allocator, init.io, args[2]);
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], allocator, .unlimited);
    const cases = try std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{});
    var out: std.ArrayList(u8) = .empty;
    const all_x = try allocator.alloc(rmodel.Input, cases.array.items.len);
    const all_h = try allocator.alloc([rmodel.D]f32, cases.array.items.len);
    for (cases.array.items, 0..) |c, case_index| {
        var x = rmodel.Input{};
        const ids = c.object.get("ids").?.object;
        const fl = c.object.get("floats").?.object;
        const I = struct {
            fn v(o: std.json.ObjectMap, k: []const u8) []std.json.Value {
                return o.get(k).?.array.items;
            }
            fn f(val: std.json.Value) f32 {
                return switch (val) {
                    .float => |q| @floatCast(q),
                    .integer => |q| @floatFromInt(q),
                    else => 0,
                };
            }
        };
        for (0..6) |s| {
            x.own_species[s] = @intCast(I.v(ids, "own_species")[s].integer);
            x.own_status[s] = @intCast(I.v(ids, "own_status")[s].integer);
            x.opp_species[s] = @intCast(I.v(ids, "opp_species")[s].integer);
            x.opp_status[s] = @intCast(I.v(ids, "opp_status")[s].integer);
            for (0..4) |j| {
                x.own_moves[s][j] = @intCast(I.v(ids, "own_moves")[s].array.items[j].integer);
                x.opp_moves[s][j] = @intCast(I.v(ids, "opp_moves")[s].array.items[j].integer);
            }
            for (0..10) |j| x.own[s][j] = I.f(I.v(fl, "own")[s].array.items[j]);
            for (0..7) |j| x.opp[s][j] = I.f(I.v(fl, "opp")[s].array.items[j]);
        }
        x.last_own_move = @intCast(I.v(ids, "last_own_move")[0].integer);
        x.last_opp_move = @intCast(I.v(ids, "last_opp_move")[0].integer);
        x.last_own_species = @intCast(I.v(ids, "last_own_species")[0].integer);
        x.last_opp_species = @intCast(I.v(ids, "last_opp_species")[0].integer);
        x.last_own_status = @intCast(I.v(ids, "last_own_status")[0].integer);
        x.last_opp_status = @intCast(I.v(ids, "last_opp_status")[0].integer);
        for (0..34) |j| x.last[j] = I.f(I.v(fl, "last")[j]);
        for (0..39) |j| x.field[j] = I.f(I.v(fl, "field")[j]);
        var h: [rmodel.D]f32 = undefined;
        for (0..rmodel.D) |j| h[j] = I.f(c.object.get("h").?.array.items[j]);
        all_x[case_index] = x;
        all_h[case_index] = h;
        var h2: [rmodel.D]f32 = undefined;
        var logits: [rmodel.ACTIONS]f32 = undefined;
        const win = model.step(&x, &h, &h2, &logits);
        try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{win}));
        for (logits) |l| try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, " {d}", .{l}));
        for (h2) |v| try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, " {d}", .{v}));
        try out.append(allocator, '\n');
    }
    // the same cases as one batch (stepBatch), prefixed "B"
    const n = all_x.len;
    const h_out = try allocator.alloc([rmodel.D]f32, n);
    const logits_all = try allocator.alloc([rmodel.ACTIONS]f32, n);
    const wins = try allocator.alloc(f32, n);
    const scratch = try allocator.alloc(f32, rmodel.Model.scratchLen(n));
    model.stepBatch(all_x, all_h, h_out, logits_all, wins, scratch);
    for (0..n) |i| {
        try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "B {d}", .{wins[i]}));
        for (logits_all[i]) |l| try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, " {d}", .{l}));
        for (h_out[i]) |v| try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, " {d}", .{v}));
        try out.append(allocator, '\n');
    }
    std.debug.print("{s}", .{out.items});
}
