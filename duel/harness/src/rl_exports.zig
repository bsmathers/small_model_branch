//! The harness code the self-play environment (selfplay/zig) builds on, as one module: the native feature code (exact against
//! the Python pipeline), the observer's knowledge, the team files and the model input layout.
pub const rnative = @import("rnative.zig");
pub const know = @import("knowledge.zig");
pub const teams = @import("teams.zig");
pub const rmodel = @import("rmodel.zig");
