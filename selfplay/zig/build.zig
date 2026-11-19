//! The self-play environment library (libpkrl) and its benchmark.
//!   zig build -Doptimize=ReleaseFast            -> zig-out/lib/libpkrl.{so,dylib}, zig-out/bin/pkrl-bench
//! The pkmn engine is expected at ./engine (pkmn/engine 78dc891, the same commit as the duel harness: a checkout or a link).
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // the engine with protocol logging on: the model reads the battle log
    const pkmn = b.dependency("pkmn", .{ .optimize = optimize, .target = target, .showdown = true, .log = true, .chance = false, .calc = false });
    const build_options = b.addOptions();
    build_options.addOption(bool, "have_blas", false); // (the harness's model code is not used here)
    build_options.addOption(bool, "openblas", false);
    // the harness's feature code (rnative.zig and what it needs), as one module
    const harness = b.createModule(.{ .root_source_file = b.path("../../duel/harness/src/rl_exports.zig"), .target = target, .optimize = optimize });
    harness.addImport("pkmn", pkmn.module("pkmn"));
    harness.addOptions("build_options", build_options);

    const env = b.createModule(.{ .root_source_file = b.path("src/env.zig"), .target = target, .optimize = optimize, .link_libc = true });
    env.addImport("pkmn", pkmn.module("pkmn"));
    env.addImport("harness", harness);
    const lib = b.addLibrary(.{ .name = "pkrl", .linkage = .dynamic, .root_module = env });
    b.installArtifact(lib);

    const bench_mod = b.createModule(.{ .root_source_file = b.path("src/bench.zig"), .target = target, .optimize = optimize, .link_libc = true });
    bench_mod.addImport("pkmn", pkmn.module("pkmn"));
    bench_mod.addImport("env", env);
    const bench = b.addExecutable(.{ .name = "pkrl-bench", .root_module = bench_mod });
    b.installArtifact(bench);
}
