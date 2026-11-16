const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "pkmn-search",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const pkmn = b.dependency("pkmn", .{ .showdown = true, .log = false, .chance = false, .calc = false });
    exe.root_module.addImport("pkmn", pkmn.module("pkmn"));
    exe.root_module.link_libc = true;
    if (target.result.os.tag == .macos) exe.root_module.linkFramework("Accelerate", .{}); // (elsewhere value_model.zig has its own matrix product)
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    const harness = b.addExecutable(.{
        .name = "pkmn-harness",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/harness.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const harness_pkmn = b.dependency("pkmn", .{ .showdown = true, .log = true, .chance = false, .calc = false });
    harness.root_module.addImport("pkmn", harness_pkmn.module("pkmn"));
    b.installArtifact(harness);
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/search.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    tests.root_module.addImport("pkmn", pkmn.module("pkmn"));
    tests.root_module.link_libc = true;
    if (target.result.os.tag == .macos) tests.root_module.linkFramework("Accelerate", .{});
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run search regression tests").dependOn(&run_tests.step);
    b.step("run", "Play a game: search (P1) vs random (P2)").dependOn(&run.step);
}
