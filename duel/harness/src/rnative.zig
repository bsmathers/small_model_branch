//! The recurrent model's inputs, computed natively: a port of the policy server's per-snapshot code (tools/recurrent_policy_server.py
//! prepare / finish / legal_slots, tools/metamon_connector.py SnapshotEncoder.state, tools/snapshot_effects.py, and
//! metamon/recurrent/features.py encode_observer) that reads the battle directly instead of a JSON snapshot.
//!
//! A `Snap` holds exactly what the duel's snapshot (snapshot.zig serializeDuelSnapshot) carries for one side, as engine values;
//! every function below follows its Python counterpart line by line (named in its comment), so a chain of `Node`s equals the
//! search server's (tools/recurrent_search_server.py). Only the branches the duel's snapshots reach are ported: they always
//! carry `last_actions`, so the policy server's fallbacks for snapshots without them (and the opponent's remembered last
//! move, which only those read) are left out. SEARCH_MODEL=native NATIVE_CHECK=1 checks every node against the Python server.

const std = @import("std");
const pkmn = @import("pkmn");
const know = @import("knowledge.zig");
const rmodel = @import("rmodel.zig");

const Battle = know.Battle;
const Species = pkmn.gen1.Species;
const Move = pkmn.gen1.Move;
const Type = pkmn.gen1.Type;

fn enumSize(comptime E: type) usize {
    var top: usize = 0;
    for (@typeInfo(E).@"enum".fields) |f| top = @max(top, f.value);
    return top + 1;
}
const N_SPECIES_ENUM = enumSize(Species);
const N_MOVES_ENUM = enumSize(Move); // (SKIP_TURN is 255)

// ---------------------------------------------------------------- names and tables

fn normInto(buf: []u8, tag: []const u8, letters_only: bool) []const u8 {
    var n: usize = 0;
    for (tag) |c| {
        const lc = std.ascii.toLower(c);
        if ((lc >= 'a' and lc <= 'z') or (!letters_only and lc >= '0' and lc <= '9')) {
            buf[n] = lc;
            n += 1;
        }
    }
    return buf[0..n];
}

/// metamon/recurrent/tables.py Tables, loaded from the JSON files tools/export_recurrent.py writes, indexed by engine values.
pub const Tables = struct {
    species_id: [N_SPECIES_ENUM]u32 = @splat(0),
    has_base: [N_SPECIES_ENUM]bool = @splat(false),
    base: [N_SPECIES_ENUM][5]i32 = @splat(@splat(0)), // hp, atk, def, spe, spa
    move_id: [N_MOVES_ENUM]u32 = @splat(0),
    has_move: [N_MOVES_ENUM]bool = @splat(false), // (`name in tables.moves`)
    move_power: [N_MOVES_ENUM]i32 = @splat(0),
    move_type: [N_MOVES_ENUM]u8 = @splat(0), // index into type names
    type_names: [32][16]u8 = undefined,
    type_lens: [32]usize = @splat(0),
    n_types: usize = 0,
    engine_type: [16]u8 = @splat(0), // engine Type -> type index
    chart: [32][32]f32 = @splat(@splat(1.0)), // [attacker][defender]
    /// Sort keys: metamon's pokemon_name (letters and digits) and move_name (letters only, with its renames).
    species_key: [N_SPECIES_ENUM][16]u8 = undefined,
    species_key_len: [N_SPECIES_ENUM]usize = @splat(0),
    move_key: [N_MOVES_ENUM][20]u8 = undefined,
    move_key_len: [N_MOVES_ENUM]usize = @splat(0),
    /// tables.normalize of the engine move's name, for the moves with special damage rules
    move_norm: [N_MOVES_ENUM][20]u8 = undefined,
    move_norm_len: [N_MOVES_ENUM]usize = @splat(0),

    fn typeIndex(self: *Tables, name: []const u8) !u8 {
        for (0..self.n_types) |i| {
            if (std.mem.eql(u8, self.type_names[i][0..self.type_lens[i]], name)) return @intCast(i);
        }
        if (self.n_types >= 32) return error.TooManyTypes;
        @memcpy(self.type_names[self.n_types][0..name.len], name);
        self.type_lens[self.n_types] = name.len;
        self.n_types += 1;
        return @intCast(self.n_types - 1);
    }

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !*Tables {
        const t = try allocator.create(Tables);
        t.* = .{};
        const read = struct {
            fn f(a: std.mem.Allocator, i: std.Io, d: []const u8, file: []const u8) !std.json.Value {
                const text = try std.Io.Dir.cwd().readFileAlloc(i, try std.fmt.allocPrint(a, "{s}/tables/{s}", .{ d, file }), a, .unlimited);
                return std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
            }
        }.f;
        const species_ids = try read(allocator, io, dir, "species_ids.json");
        const move_ids = try read(allocator, io, dir, "move_ids.json");
        const base = try read(allocator, io, dir, "base_stats.json");
        const moves = try read(allocator, io, dir, "moves.json");
        const chart = try read(allocator, io, dir, "type_chart.json");
        var it = chart.object.iterator();
        while (it.next()) |row| {
            const a = try t.typeIndex(row.key_ptr.*);
            var jt = row.value_ptr.object.iterator();
            while (jt.next()) |cell| {
                const b = try t.typeIndex(cell.key_ptr.*);
                t.chart[a][b] = switch (cell.value_ptr.*) {
                    .float => |v| @floatCast(v),
                    .integer => |v| @floatFromInt(v),
                    else => 1.0,
                };
            }
        }
        inline for (@typeInfo(Type).@"enum".fields) |f| {
            var buf: [16]u8 = undefined;
            t.engine_type[f.value] = try t.typeIndex(normInto(&buf, f.name, false));
        }
        inline for (@typeInfo(Species).@"enum".fields) |f| {
            var buf: [16]u8 = undefined;
            const name = normInto(&buf, f.name, false);
            @memcpy(t.species_key[f.value][0..name.len], name);
            t.species_key_len[f.value] = name.len;
            if (species_ids.object.get(name)) |v| t.species_id[f.value] = @intCast(v.integer);
            if (base.object.get(name)) |v| {
                t.has_base[f.value] = true;
                const o = v.object;
                t.base[f.value] = .{ @intCast(o.get("hp").?.integer), @intCast(o.get("atk").?.integer), @intCast(o.get("def").?.integer), @intCast(o.get("spe").?.integer), @intCast(o.get("spa").?.integer) };
            }
        }
        inline for (@typeInfo(Move).@"enum".fields) |f| {
            var buf: [20]u8 = undefined;
            const name = normInto(&buf, f.name, false);
            @memcpy(t.move_norm[f.value][0..name.len], name);
            t.move_norm_len[f.value] = name.len;
            if (move_ids.object.get(name)) |v| t.move_id[f.value] = @intCast(v.integer);
            if (moves.object.get(name)) |v| {
                t.has_move[f.value] = true;
                t.move_power[f.value] = @intCast(v.array.items[0].integer);
                var tb: [16]u8 = undefined;
                t.move_type[f.value] = try t.typeIndex(normInto(&tb, v.array.items[1].string, false));
            }
            // metamon move_name: letters only, then its renames
            var kb: [20]u8 = undefined;
            var key = normInto(&kb, f.name, true);
            if (std.mem.startsWith(u8, key, "hiddenpower")) key = "hiddenpower" else if (std.mem.eql(u8, key, "vicegrip")) key = "visegrip" else if (std.mem.startsWith(u8, key, "return")) key = "return" else if (std.mem.startsWith(u8, key, "frustration")) key = "frustration";
            @memcpy(t.move_key[f.value][0..key.len], key);
            t.move_key_len[f.value] = key.len;
        }
        return t;
    }

    fn speciesKey(self: *const Tables, s: Species) []const u8 {
        return self.species_key[@intFromEnum(s)][0..self.species_key_len[@intFromEnum(s)]];
    }
    fn moveKey(self: *const Tables, m: Move) []const u8 {
        return self.move_key[@intFromEnum(m)][0..self.move_key_len[@intFromEnum(m)]];
    }
    fn moveNorm(self: *const Tables, m: Move) []const u8 {
        return self.move_norm[@intFromEnum(m)][0..self.move_norm_len[@intFromEnum(m)]];
    }
    fn typeName(self: *const Tables, ty: u8) []const u8 {
        return self.type_names[ty][0..self.type_lens[ty]];
    }
    /// tables.effectiveness (defender types here are never empty or typeless)
    fn effectiveness(self: *const Tables, attack: u8, defend: u8) f64 {
        return self.chart[attack][defend];
    }
};

// ---------------------------------------------------------------- the snapshot

/// features.STATUS_IDS order; the snapshot's names map to it as metamon_connector._STATUS does.
pub const Status = enum(u8) { nostatus = 0, slp = 1, brn = 2, frz = 3, par = 4, psn = 5, tox = 6, fnt = 7 };

/// knowledge.zig statusName, then metamon_connector._status
fn snapStatus(status: u8, hp: u16) Status {
    if (hp == 0) return .fnt;
    if (status == 0) return .nostatus;
    if (pkmn.gen1.Status.is(status, .SLP)) return .slp;
    if (pkmn.gen1.Status.is(status, .PSN)) return .psn;
    if (pkmn.gen1.Status.is(status, .BRN)) return .brn;
    if (pkmn.gen1.Status.is(status, .FRZ)) return .frz;
    if (pkmn.gen1.Status.is(status, .PAR)) return .par;
    return .nostatus;
}

/// knowledge.zig publicVolatileList's names
pub const Pv = enum(u8) { bide, confusion, focusenergy, leechseed, lightscreen, mist, rage, reflect, substitute, transform, mustrecharge, lockedmove, binding };

pub const PvList = struct {
    items: [16]Pv = undefined,
    len: u8 = 0,
    fn has(self: *const PvList, v: Pv) bool {
        for (self.items[0..self.len]) |x| if (x == v) return true;
        return false;
    }
};

/// knowledge.zig publicVolatileList: the engine's volatiles in field order
fn publicVolatiles(v: anytype) PvList {
    var out = PvList{};
    inline for (@typeInfo(@TypeOf(v)).@"struct".fields) |f| {
        if (f.type == bool and @field(v, f.name)) {
            const name: ?Pv = if (comptime std.mem.eql(u8, f.name, "Bide")) .bide else if (comptime std.mem.eql(u8, f.name, "Confusion")) .confusion else if (comptime std.mem.eql(u8, f.name, "FocusEnergy")) .focusenergy else if (comptime std.mem.eql(u8, f.name, "LeechSeed")) .leechseed else if (comptime std.mem.eql(u8, f.name, "LightScreen")) .lightscreen else if (comptime std.mem.eql(u8, f.name, "Mist")) .mist else if (comptime std.mem.eql(u8, f.name, "Rage")) .rage else if (comptime std.mem.eql(u8, f.name, "Reflect")) .reflect else if (comptime std.mem.eql(u8, f.name, "Substitute")) .substitute else if (comptime std.mem.eql(u8, f.name, "Transform")) .transform else if (comptime std.mem.eql(u8, f.name, "Recharging")) .mustrecharge else if (comptime std.mem.eql(u8, f.name, "Thrashing")) .lockedmove else if (comptime std.mem.eql(u8, f.name, "Binding")) .binding else null;
            if (name) |n| {
                out.items[out.len] = n;
                out.len += 1;
            }
        }
    }
    return out;
}

pub const TypeSet = struct {
    items: [2]u8 = undefined, // type indices, deduplicated
    len: u8 = 0,
    fn has(self: *const TypeSet, t: u8) bool {
        for (self.items[0..self.len]) |x| if (x == t) return true;
        return false;
    }
};

fn typeSet(tables: *const Tables, t: pkmn.gen1.Types) TypeSet {
    var out = TypeSet{};
    for ([2]Type{ t.type1, t.type2 }) |ty| {
        const idx = tables.engine_type[@intFromEnum(ty)];
        if (!out.has(idx)) {
            out.items[out.len] = idx;
            out.len += 1;
        }
    }
    return out;
}

pub const MoveSlot = struct { id: Move = .None, pp: i64 = 0, max_pp: i64 = 0 };

pub const OwnMon = struct {
    species: Species = .None,
    hp: u16 = 0,
    maxhp: u16 = 0,
    status: Status = .nostatus,
    types: TypeSet = .{},
    moves: [4]MoveSlot = @splat(.{}),
    nmoves: u8 = 0,
};

pub const OppEntry = struct {
    species: Species = .None,
    hp_pct: f64 = 0,
    status: Status = .nostatus,
    moves: [4]MoveSlot = @splat(.{}),
    nmoves: u8 = 0,
};

pub const Boosts = struct { atk: i64 = 0, def: i64 = 0, spc: i64 = 0, spe: i64 = 0 };

/// A legal option as the duel names it: "no move", a move, or "switch <species>".
pub const Option = struct {
    kind: enum(u8) { nomove, move, switch_ } = .nomove,
    move: Move = .None,
    species: Species = .None,
};

pub const Act = struct { executed: Move = .None, recharged: bool = false };

pub const Snap = struct {
    turn: u32 = 0,
    force_switch: bool = false,
    opp_force_switch: bool = false,
    last_damage: u16 = 0,
    party: [6]OwnMon = @splat(.{}),
    n_party: u8 = 0,
    boosts: Boosts = .{},
    pv: PvList = .{},
    last_move: Move = .None, // (.None is "none")
    opp_species: Species = .None,
    opp_hp: f64 = 0,
    opp_status: Status = .nostatus,
    opp_types: TypeSet = .{},
    opp_boosts: Boosts = .{},
    revealed: [6]OppEntry = @splat(.{}),
    n_revealed: u8 = 0,
    opp_last_move: Move = .None,
    opp_pv: PvList = .{},
    legal: [pkmn.CHOICES_SIZE]Option = @splat(.{}),
    choices: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined,
    n_legal: u8 = 0,
    /// [user, opponent], from this side's view
    acts: [2]Act = @splat(.{}),
};

/// knowledge.zig publicHpPct (Showdown's HP Percentage Mod), as a fraction
fn publicHpPct(hp: u16, maximum: u16) f64 {
    if (hp == 0) return 0;
    const top = @max(@as(u32, maximum), 1);
    var pct = (100 * @as(u32, hp) + top - 1) / top;
    if (pct >= 100 and hp < maximum) pct = 99;
    return @min(@as(f64, @floatFromInt(pct)) / 100.0, 1.0);
}

fn lastMove(m: Move) Move {
    return if (m == .SKIP_TURN) .None else m;
}

fn maxPp(m: Move) i64 {
    return @divTrunc(@as(i64, Move.pp(m)), 5) * 8;
}

/// The duel's snapshot for `side` (snapshot.serializeDuelSnapshot: knowledge.observerSnapshot of the side's view, its legal
/// options and the update's last actions) at the decision point `battle` (`result` is the update's result there). `k` is the
/// observer's knowledge in P1's orientation; `last` is [P1's, P2's] last action.
pub fn snapOf(tables: *const Tables, battle: *const Battle, result: pkmn.Result, k: know.Knowledge, side: usize, last: [2]Act) Snap {
    var s: Snap = undefined;
    snapInto(&s, tables, battle, result, k, side, last);
    return s;
}

/// snapOf, written into `out` (a node's own snapshot: no copy).
pub fn snapInto(out: *Snap, tables: *const Tables, battle: *const Battle, result: pkmn.Result, k: know.Knowledge, side: usize, last: [2]Act) void {
    out.* = .{};
    const s = out;
    // legal options from the unswapped battle (snapshot.legalNames / duelName)
    {
        var b = battle.*;
        const n = if (side == 0) b.choices(.P1, result.p1, &s.choices) else b.choices(.P2, result.p2, &s.choices);
        s.n_legal = @intCast(n);
        const me = battle.sides[side];
        for (0..n) |i| {
            const c = s.choices[i];
            if (c.type == .Move and c.data > 0) {
                s.legal[i] = .{ .kind = .move, .move = me.pokemon[me.order[0] - 1].moves[c.data - 1].id };
            } else if (c.type == .Switch) {
                s.legal[i] = .{ .kind = .switch_, .species = me.pokemon[me.order[0 + c.data - 1] - 1].species };
            } else s.legal[i] = .{ .kind = .nomove };
        }
    }
    var view = battle.*;
    var kk = k;
    s.acts = .{ last[0], last[1] };
    if (side == 1) {
        std.mem.swap(@TypeOf(view.sides[0]), &view.sides[0], &view.sides[1]);
        kk = know.Knowledge{ .seen = .{ k.seen[1], k.seen[0] }, .moves = .{ k.moves[1], k.moves[0] }, .used = .{ k.used[1], k.used[0] } };
        s.acts = .{ last[1], last[0] };
    }
    const own = view.sides[0];
    const foe = view.sides[1];
    const own_active = own.order[0] - 1;
    const foe_active = foe.order[0] - 1;
    s.turn = view.turn;
    s.force_switch = own.pokemon[own_active].hp == 0;
    s.opp_force_switch = foe.pokemon[foe_active].hp == 0;
    s.last_damage = view.last_damage;
    var order: [6]usize = undefined;
    order[0] = own_active;
    var n: usize = 1;
    for (0..6) |i| if (i != own_active) {
        order[n] = i;
        n += 1;
    };
    for (order) |slot| {
        const mon = own.pokemon[slot];
        if (mon.species == .None) continue;
        var m = OwnMon{ .species = mon.species, .hp = mon.hp, .maxhp = mon.stats.hp, .status = snapStatus(mon.status, mon.hp), .types = typeSet(tables, mon.types) };
        for (mon.moves) |mv| {
            if (mv.id == .None) continue;
            m.moves[m.nmoves] = .{ .id = mv.id, .pp = mv.pp, .max_pp = maxPp(mv.id) };
            m.nmoves += 1;
        }
        s.party[s.n_party] = m;
        s.n_party += 1;
    }
    s.boosts = .{ .atk = own.active.boosts.atk, .def = own.active.boosts.def, .spc = own.active.boosts.spc, .spe = own.active.boosts.spe };
    s.pv = publicVolatiles(own.active.volatiles);
    s.last_move = lastMove(own.last_used_move);
    const foe_mon = foe.pokemon[foe_active];
    s.opp_species = foe_mon.species;
    s.opp_hp = publicHpPct(foe_mon.hp, foe_mon.stats.hp);
    s.opp_status = snapStatus(foe_mon.status, foe_mon.hp);
    s.opp_types = typeSet(tables, foe_mon.types);
    s.opp_boosts = .{ .atk = foe.active.boosts.atk, .def = foe.active.boosts.def, .spc = foe.active.boosts.spc, .spe = foe.active.boosts.spe };
    for (0..7) |step| {
        // the seen reserves in slot order, then the active (knowledge.observerSnapshot)
        const slot = if (step < 6) step else foe_active;
        if (step < 6 and (!kk.seen[1][slot] or slot == foe_active)) continue;
        const mon = foe.pokemon[slot];
        var e = OppEntry{ .species = mon.species, .hp_pct = publicHpPct(mon.hp, mon.stats.hp), .status = snapStatus(mon.status, mon.hp) };
        for (kk.moves[1][slot], kk.used[1][slot]) |m, used| {
            if (m == .None) continue;
            const mx = maxPp(m);
            e.moves[e.nmoves] = .{ .id = m, .pp = @max(0, mx - @as(i64, used)), .max_pp = mx };
            e.nmoves += 1;
        }
        s.revealed[s.n_revealed] = e;
        s.n_revealed += 1;
    }
    s.opp_last_move = lastMove(foe.last_used_move);
    s.opp_pv = publicVolatiles(foe.active.volatiles);
}

/// snapshot.lastAction, as an Act
pub fn actOf(before: *const Battle, after: *const Battle, side: usize, c: pkmn.Choice) Act {
    const b = before.sides[side];
    const a = after.sides[side];
    if (c.type != .Move or b.order[0] == 0) return .{};
    if (b.active.volatiles.Recharging) return .{ .recharged = true };
    const slot = b.order[0] - 1;
    const forced = b.active.volatiles.Thrashing or b.active.volatiles.Rage or b.active.volatiles.Charging;
    if (!forced and c.data >= 1 and c.data <= 4) {
        const mv = b.active.moves[c.data - 1];
        if (mv.id == .None) return .{};
        if (a.pokemon[slot].moves[c.data - 1].pp < mv.pp) return .{ .executed = mv.id };
        if (b.active.volatiles.Binding and a.last_used_move == mv.id) return .{ .executed = mv.id };
        return .{};
    }
    const locked = b.last_selected_move;
    if (locked != .None and locked != .SKIP_TURN and a.last_used_move == locked) return .{ .executed = locked };
    return .{};
}

/// The option `side` chose (snapshot.duelName), as the server records it in "sent".
pub fn optionOf(battle: *const Battle, side: usize, c: pkmn.Choice) Option {
    const me = battle.sides[side];
    if (c.type == .Move and c.data > 0) return .{ .kind = .move, .move = me.pokemon[me.order[0] - 1].moves[c.data - 1].id };
    if (c.type == .Switch) return .{ .kind = .switch_, .species = me.pokemon[me.order[c.data - 1] - 1].species };
    return .{ .kind = .nomove };
}

// ---------------------------------------------------------------- metamon's state

/// A UniversalPokemon, with what the features read.
pub const UMon = struct {
    species: Species = .None,
    hp_pct: f64 = 0,
    status: Status = .nostatus,
    types: TypeSet = .{},
    moves: [4]MoveSlot = @splat(.{}), // current pp, max pp (clamped as metamon_connector._move does)
    nmoves: u8 = 0,
    atk: i64 = 0,
    def: i64 = 0,
    spe: i64 = 0,
    spa: i64 = 0,
    spd: i64 = 0,
    effect: ?Eff = null,

    fn alive(self: *const UMon) bool {
        return self.hp_pct > 0 and self.status != .fnt;
    }
};

/// recurrent_policy_server.showdown_pct
fn showdownPct(hp: u16, top: u16) i64 {
    if (hp == 0) return 0;
    const t: i64 = @max(@as(i64, top), 1);
    const pct = @divFloor(100 * @as(i64, hp) + t - 1, t);
    return if (pct >= 100 and hp < top) 99 else @min(pct, 100);
}

fn uMoves(out: *UMon, moves: []const MoveSlot) void {
    out.nmoves = 0;
    for (moves) |m| {
        const mx = @max(m.max_pp, 0);
        out.moves[out.nmoves] = .{ .id = m.id, .pp = @max(0, @min(m.pp, mx)), .max_pp = mx };
        out.nmoves += 1;
    }
}

/// A party member as prepare builds it: metamon_connector._pokemon(exact_hp=True) of the snapshot entry with its HP as Showdown's
/// percentage (the active also carries the side's boosts).
fn ownUMon(m: *const OwnMon, boosts: ?Boosts) UMon {
    var u = UMon{ .species = m.species, .status = m.status, .types = m.types };
    u.hp_pct = @max(0.0, @min(1.0, @as(f64, @floatFromInt(showdownPct(m.hp, m.maxhp))) / 100.0));
    if (u.hp_pct <= 0) u.status = .fnt;
    uMoves(&u, m.moves[0..m.nmoves]);
    if (boosts) |b| {
        u.atk = b.atk;
        u.def = b.def;
        u.spe = b.spe;
        u.spa = b.spc;
        u.spd = b.spc;
    }
    return u;
}

// ---------------------------------------------------------------- the game memory

pub const Kind = enum(u8) { move, switch_, recharge, none }; // features.KINDS
pub const Cant = enum(u8) { par, slp, frz, recharge, partiallytrapped, flinch, fainted, wokeup, opponent_fainted, confusion, game_ended }; // CANT_REASONS

/// features.LastTurn
pub const LastTurn = struct {
    own_kind: Kind = .none,
    own_move: Move = .None,
    own_species: Species = .None,
    own_cant: ?Cant = null,
    opp_kind: Kind = .none,
    opp_move: Move = .None,
    opp_species: Species = .None,
    opp_cant: ?Cant = null,
    own_hp_change: f64 = 0,
    opp_hp_change: f64 = 0,
    own_status: Status = .nostatus,
    opp_status: Status = .nostatus,
    own_fainted: bool = false,
    opp_fainted: bool = false,
};

/// snapshot_effects: the effects active_effects can return
pub const Eff = enum(u8) { bide, confusion, disable, focusenergy, leechseed, lightscreen, lockedmove, mimic, mist, mustrecharge, partiallytrapped, rage, reflect, substitute, typechange, bind, clamp, firespin, wrap };

const EffSide = struct {
    identity: Species = .None,
    turn: u32 = 0,
    effects: [12]Eff = undefined,
    clocks: [12]u32 = undefined,
    n: u8 = 0,
};

/// snapshot_effects.SnapshotEffectMemory
pub const EffectMemory = struct {
    clock: u32 = 0,
    sides: [2]?EffSide = .{ null, null },

    fn select(self: *EffectMemory, current: []const Eff, identity: Species, turn: u32, side: usize) ?Eff {
        if (self.sides[side] == null or self.sides[side].?.identity != identity) self.sides[side] = .{ .identity = identity, .turn = turn };
        var r = &self.sides[side].?;
        r.turn = turn;
        var keep: u8 = 0;
        for (0..r.n) |i| {
            var present = false;
            for (current) |a| present = present or a == r.effects[i];
            if (present) {
                r.effects[keep] = r.effects[i];
                r.clocks[keep] = r.clocks[i];
                keep += 1;
            }
        }
        r.n = keep;
        for (current) |a| {
            var tracked = false;
            for (r.effects[0..r.n]) |e| tracked = tracked or e == a;
            if (!tracked) {
                self.clock += 1;
                r.effects[r.n] = a;
                r.clocks[r.n] = self.clock;
                r.n += 1;
            }
        }
        if (r.n == 0) return null;
        var best: usize = 0;
        for (1..r.n) |i| if (r.clocks[i] < r.clocks[best]) {
            best = i;
        };
        return r.effects[best];
    }
};

const Trap = struct { trapper: Species, uses: u32, turn: u32 };

/// recurrent_policy_server.Game (what the features read)
pub const Game = struct {
    h: [rmodel.D]f32 = @splat(0),
    /// The last decision node (its snapshot and state are the previous ones; a pointer, not a copy: nodes outlive their children)
    prev_node: ?*const Node = null,
    sent: Option = .{},
    last: LastTurn = .{},
    decisions: u32 = 0,
    counter_selected: [2]bool = .{ false, false },
    opp_faint_switch: bool = false,
    effects: EffectMemory = .{},
    trap: [2]?Trap = .{ null, null },
    acts: [2]?Act = .{ null, null },
};

/// A search node: the game memory after reading its snapshot, the snapshot and the state it read (for the next node's
/// last-turn record), and its outputs.
pub const Node = struct {
    game: Game = .{},
    snap: Snap = .{},
    own: UMon = .{},
    opp: UMon = .{},
    decision: bool = false,
    win: f32 = 0,
};

// ---------------------------------------------------------------- the last-turn record

const FLINCH = [_]Move{ .Bite, .Stomp, .Headbutt, .RollingKick, .HyperFang, .LowKick, .BoneClub };
const TRAPPING = [_]Move{ .Wrap, .Bind, .FireSpin, .Clamp };

fn isAny(m: Move, set: []const Move) bool {
    for (set) |x| if (x == m) return true;
    return false;
}

const Rec = struct { kind: Kind, move: Move = .None, species: Species = .None, cant: ?Cant = null };

/// recurrent_policy_server.cant_reason (side 0 = "user")
fn cantReason(prev: *const Snap, prev_own: *const UMon, prev_opp: *const UMon, snap: *const Snap, own: *const UMon, opp: *const UMon, side: usize) ?Cant {
    const mine_before = if (side == 0) prev_own else prev_opp;
    const mine_now = if (side == 0) own else opp;
    const theirs_now = if (side == 0) opp else own;
    if (mine_now.hp_pct <= 0 or (if (side == 0) snap.force_switch else snap.opp_force_switch)) return .fainted;
    const before = mine_before.status;
    const now = mine_now.status;
    if (before == .slp) return if (now == .slp) .slp else .wokeup;
    if (before == .frz) return .frz;
    if (now == .slp) return .slp;
    if (now == .frz) return .frz;
    if (theirs_now.hp_pct <= 0 or (if (side == 0) snap.opp_force_switch else snap.force_switch)) return .opponent_fainted;
    const other_now = if (side == 0) &snap.opp_pv else &snap.pv;
    const other_before = if (side == 0) &prev.opp_pv else &prev.pv;
    if (other_now.has(.binding) or other_before.has(.binding)) return .partiallytrapped;
    const other_act = snap.acts[1 - side];
    const mine_pv_before = if (side == 0) &prev.pv else &prev.opp_pv;
    if (isAny(other_act.executed, &FLINCH) and !mine_pv_before.has(.substitute)) return .flinch;
    if (before == .par) return .par;
    if (mine_pv_before.has(.confusion)) return .confusion;
    return null;
}

/// recurrent_policy_server.own_cant (with the engine's record of the update: `action` is always there)
fn ownCant(g: *const Game, snap: *const Snap, own: *const UMon, opp: *const UMon, action: Act) Rec {
    const pn = g.prev_node.?;
    if (g.sent.kind == .switch_) return .{ .kind = .switch_, .species = g.sent.species, .cant = if (pn.snap.force_switch) g.last.own_cant else null };
    if (action.recharged) return .{ .kind = .recharge, .cant = .recharge };
    if (action.executed != .None) return .{ .kind = .move, .move = action.executed };
    if (g.sent.kind == .nomove) return .{ .kind = .none };
    return .{ .kind = .none, .cant = cantReason(&pn.snap, &pn.own, &pn.opp, snap, own, opp, 0) };
}

/// recurrent_policy_server.opp_cant (action branch)
fn oppCant(g: *const Game, snap: *const Snap, own: *const UMon, opp: *const UMon, action: Act) Rec {
    const pn = g.prev_node.?;
    if (pn.snap.force_switch and !pn.snap.opp_force_switch) return .{ .kind = g.last.opp_kind, .move = g.last.opp_move, .species = g.last.opp_species, .cant = g.last.opp_cant };
    if (g.opp_faint_switch) return if (action.executed != .None) .{ .kind = .move, .move = action.executed } else .{ .kind = .none, .cant = .fainted };
    if (pn.snap.opp_species != snap.opp_species) return .{ .kind = .switch_, .species = snap.opp_species };
    if (action.recharged) return .{ .kind = .recharge, .cant = .recharge };
    if (action.executed != .None) return .{ .kind = .move, .move = action.executed };
    return .{ .kind = .none, .cant = cantReason(&pn.snap, &pn.own, &pn.opp, snap, own, opp, 1) };
}

/// recurrent_policy_server.change: (hp change, status inflicted, fainted)
fn change(prev: *const UMon, now: *const UMon) struct { f64, Status, bool } {
    if (prev.species != now.species) return .{ 0.0, .nostatus, now.status == .fnt };
    const inflicted: Status = if (prev.status == .nostatus and now.status != .fnt) now.status else .nostatus;
    return .{ now.hp_pct - prev.hp_pct, inflicted, now.status == .fnt };
}

/// snapshot_effects.active_effects of recurrent_policy_server.effect_payload(snap, side)
fn activeEffects(snap: *const Snap, side: usize, out: *[16]Eff) usize {
    const mine = if (side == 0) &snap.pv else &snap.opp_pv;
    const theirs = if (side == 0) &snap.opp_pv else &snap.pv;
    const last_move = if (side == 0) snap.last_move else snap.opp_last_move;
    const trapping_move = if (side == 0) snap.opp_last_move else snap.last_move;
    var raw: [17]?Eff = undefined;
    var nraw: usize = 0;
    for (mine.items[0..mine.len]) |v| {
        if (v == .binding) continue;
        raw[nraw] = switch (v) {
            .bide => .bide,
            .confusion => .confusion,
            .focusenergy => .focusenergy,
            .leechseed => .leechseed,
            .lightscreen => .lightscreen,
            .mist => .mist,
            .rage => .rage,
            .reflect => .reflect,
            .substitute => .substitute,
            .mustrecharge => .mustrecharge,
            .lockedmove => .lockedmove,
            .transform, .binding => null, // (not a metamon effect)
        };
        nraw += 1;
    }
    if (theirs.has(.binding)) {
        raw[nraw] = .partiallytrapped;
        nraw += 1;
    }
    var n: usize = 0;
    for (raw[0..nraw]) |maybe| {
        var e = maybe orelse continue;
        if (e == .lockedmove or e == .partiallytrapped) {
            const source = if (e == .lockedmove) last_move else trapping_move;
            if (isAny(source, &TRAPPING)) e = switch (source) {
                .Wrap => .wrap,
                .Bind => .bind,
                .FireSpin => .firespin,
                .Clamp => .clamp,
                else => unreachable,
            };
        }
        var dup = false;
        for (out[0..n]) |x| dup = dup or x == e;
        if (!dup) {
            out[n] = e;
            n += 1;
        }
    }
    return n;
}

/// Game.trap_probability (parse.TRAP_PROBABILITY)
fn trapProbability(g: *Game, snap: *const Snap, trapper_side: usize) f64 {
    const pv = if (trapper_side == 0) &snap.pv else &snap.opp_pv;
    if (!pv.has(.binding)) {
        g.trap[trapper_side] = null;
        return 0.0;
    }
    const trapper = if (trapper_side == 0) snap.party[0].species else snap.opp_species;
    const prev = g.trap[trapper_side];
    const uses: u32 = if (prev == null or prev.?.trapper != trapper) 1 else prev.?.uses + @as(u32, if (snap.turn != prev.?.turn) 1 else 0);
    g.trap[trapper_side] = .{ .trapper = trapper, .uses = uses, .turn = snap.turn };
    return switch (uses) {
        1 => 1.0,
        2 => 5.0 / 8.0,
        3 => 2.0 / 5.0,
        4 => 1.0 / 2.0,
        else => 0.0,
    };
}

fn statOf(base: i32, hp: bool) f64 {
    return if (hp) @floatFromInt((base + 15) * 2 + 63 + 100 + 10) else @floatFromInt((base + 15) * 2 + 63 + 5);
}

const CounterDamage = struct { player: bool, species: Species, pct: f64 };

/// recurrent_policy_server.counter_damage (`g.last` is already the new record)
fn counterDamage(tables: *const Tables, g: *const Game, snap: *const Snap, own: *const UMon, opp: *const UMon) ?CounterDamage {
    const damage: i64 = snap.last_damage;
    if (damage <= 0) return null;
    var player = false;
    if (g.prev_node) |pn| {
        const before = pn.snap.party[0];
        const after = snap.party[0];
        if (before.species == after.species and @as(i64, before.hp) - @as(i64, after.hp) == damage) {
            player = true;
        } else if (before.species == after.species and before.hp > after.hp and g.last.opp_hp_change >= 0) player = true;
    }
    const mon = if (player) own else opp;
    if (!tables.has_base[@intFromEnum(mon.species)]) return null;
    const estimate = statOf(tables.base[@intFromEnum(mon.species)][0], true);
    return .{ .player = player, .species = mon.species, .pct = 100.0 * @as(f64, @floatFromInt(damage)) / @max(estimate, 1.0) };
}

// ---------------------------------------------------------------- features

const Stats = struct { hp: f64, atk: f64, def: f64, spe: f64, spa: f64, spd: f64 };

/// features._own_stats / _opp_stats (no actual stats: the max-DV estimate either way)
fn statsOf(tables: *const Tables, s: Species) ?Stats {
    if (!tables.has_base[@intFromEnum(s)]) return null;
    const b = tables.base[@intFromEnum(s)];
    const spa = statOf(b[4], false);
    return .{ .hp = statOf(b[0], true), .atk = statOf(b[1], false), .def = statOf(b[2], false), .spe = statOf(b[3], false), .spa = spa, .spd = spa };
}

fn boosted(stat: f64, boost0: i64) f64 {
    const boost = @max(-6, @min(6, boost0));
    if (boost >= 0) return stat * @as(f64, @floatFromInt(2 + boost)) / 2.0;
    return stat * 2.0 / @as(f64, @floatFromInt(2 - boost));
}

fn boostFeature(v: i64) f32 {
    return @floatCast(@as(f64, @floatFromInt(@max(-6, @min(6, v)))) / 6.0);
}

const PHYSICAL = [_][]const u8{ "normal", "fighting", "poison", "ground", "flying", "bug", "rock", "ghost", "steel" };

fn isPhysical(tables: *const Tables, t: u8) bool {
    for (PHYSICAL) |p| if (std.mem.eql(u8, tables.typeName(t), p)) return true;
    return false;
}

/// features._damage_fraction
fn damageFraction(tables: *const Tables, attacker: *const UMon, defender: *const UMon, m: Move, as: Stats, ds: Stats) f64 {
    const power: i64 = tables.move_power[@intFromEnum(m)];
    const mtype = tables.move_type[@intFromEnum(m)];
    const fixed: ?f64 = switch (m) {
        .SeismicToss, .NightShade => 100,
        .SonicBoom => 20,
        .DragonRage => 40,
        else => null,
    };
    const ohko = m == .Fissure or m == .Guillotine or m == .HornDrill;
    if (fixed != null or ohko) {
        var immune = false;
        if (m != .NightShade) {
            for (defender.types.items[0..defender.types.len]) |t| immune = immune or tables.effectiveness(mtype, t) == 0;
        }
        if (immune) return 0.0;
        if (ohko) return 1.0;
        return @min(1.0, fixed.? / @max(1.0, ds.hp));
    }
    if (power <= 0) return 0.0;
    var atk: f64 = undefined;
    var dfn: f64 = undefined;
    if (!isPhysical(tables, mtype)) {
        atk = boosted(as.spa, attacker.spa);
        dfn = boosted(ds.spd, defender.spd);
    } else {
        atk = boosted(as.atk, attacker.atk);
        dfn = boosted(ds.def, defender.def);
    }
    if (atk >= 256 or dfn >= 256) {
        atk = @max(1.0, @floor(atk / 4.0));
        dfn = @max(1.0, @floor(dfn / 4.0));
    }
    if (m == .Explosion or m == .SelfDestruct) dfn = @max(1.0, @floor(dfn / 2.0));
    const level_term: i64 = 2 * 100 / 5 + 2;
    var damage: i64 = @divFloor(level_term * power * @as(i64, @intFromFloat(atk)), @max(1, @as(i64, @intFromFloat(dfn))));
    damage = @divFloor(damage, 50) + 2;
    if (attacker.types.has(mtype)) damage = @divFloor(damage * 3, 2);
    var multiplier: f64 = 1.0;
    for (defender.types.items[0..defender.types.len]) |t| multiplier *= tables.effectiveness(mtype, t);
    damage = @intFromFloat(@as(f64, @floatFromInt(damage)) * multiplier);
    if (multiplier > 0) damage = @max(1, damage);
    return @min(1.0, @as(f64, @floatFromInt(damage)) / @max(1.0, ds.hp));
}

const Extras = struct {
    last_damage: ?CounterDamage,
    last_move: [2]Move, // player, opponent
    counter_selected: [2]bool,
    cant: [2]?Cant,
    effects: [2]PvList,
    trapped: [2]f64,
};

/// features.counter_fraction (user 0 = "player")
fn counterFraction(tables: *const Tables, user: usize, ex: *const Extras, own_hp: f64, opp_hp: f64) f64 {
    const d = ex.last_damage orelse return 0.0;
    const target = 1 - user;
    const moved = ex.last_move[target];
    if (moved == .None or !tables.has_move[@intFromEnum(moved)]) return 0.0;
    const power = tables.move_power[@intFromEnum(moved)];
    const tname = tables.typeName(tables.move_type[@intFromEnum(moved)]);
    if (power <= 0 or !(std.mem.eql(u8, tname, "normal") or std.mem.eql(u8, tname, "fighting"))) return 0.0;
    if (ex.counter_selected[target] and ex.cant[target] == null) return 0.0;
    const victim_hp = if (tables.has_base[@intFromEnum(d.species)]) statOf(tables.base[@intFromEnum(d.species)][0], true) else (if (d.player) own_hp else opp_hp);
    const target_hp = if (user == 0) opp_hp else own_hp;
    return @min(1.0, 2.0 * d.pct / 100.0 * victim_hp / @max(1.0, target_hp));
}

/// features._move_damage
fn moveDamage(tables: *const Tables, user: usize, attacker: *const UMon, defender: *const UMon, m: Move, as: Stats, ds: Stats, ex: *const Extras, own_hp: f64, opp_hp: f64) f64 {
    switch (m) {
        .Counter => return counterFraction(tables, user, ex, own_hp, opp_hp),
        .SuperFang => {
            const hp = ds.hp;
            if (defender.hp_pct <= 0) return 0.0;
            const half: i64 = @divFloor(@as(i64, @intFromFloat(defender.hp_pct * hp)), 2);
            return @max(1.0, @as(f64, @floatFromInt(half))) / @max(1.0, hp);
        },
        .Psywave => return @min(1.0, 149.0 / @max(1.0, ds.hp)),
        .Bide => return 0.0,
        else => return damageFraction(tables, attacker, defender, m, as, ds),
    }
}

fn lessSpecies(tables: *const Tables, a: Species, b: Species) bool {
    return std.mem.lessThan(u8, tables.speciesKey(a), tables.speciesKey(b));
}

fn sortMoves(tables: *const Tables, u: *UMon) void {
    // (insertion sort by metamon's move_name: stable, as sorted())
    var i: usize = 1;
    while (i < u.nmoves) : (i += 1) {
        var j = i;
        while (j > 0 and std.mem.lessThan(u8, tables.moveKey(u.moves[j].id), tables.moveKey(u.moves[j - 1].id))) : (j -= 1) {
            std.mem.swap(MoveSlot, &u.moves[j], &u.moves[j - 1]);
        }
    }
}

fn sortBySpecies(tables: *const Tables, mons: []UMon) void {
    var i: usize = 1;
    while (i < mons.len) : (i += 1) {
        var j = i;
        while (j > 0 and lessSpecies(tables, mons[j].species, mons[j - 1].species)) : (j -= 1) std.mem.swap(UMon, &mons[j], &mons[j - 1]);
    }
}

/// features.canonical_own: the party in policy order (the active, the living reserves, the fainted ones; moves sorted)
fn canonicalOwn(tables: *const Tables, own: *const UMon, party: []const UMon, out: *[6]UMon) usize {
    var n: usize = 0;
    for (party) |p| if (p.species == own.species) {
        out[n] = p;
        n += 1;
    };
    if (n == 0) {
        out[0] = own.*;
        n = 1;
    }
    const start_living = n;
    for (party) |p| if (p.species != own.species and p.alive()) {
        out[n] = p;
        n += 1;
    };
    sortBySpecies(tables, out[start_living..n]);
    const start_fainted = n;
    for (party) |p| if (p.species != own.species and !p.alive()) {
        out[n] = p;
        n += 1;
    };
    sortBySpecies(tables, out[start_fainted..n]);
    for (out[0..n]) |*p| sortMoves(tables, p);
    return n;
}

/// features.encode_observer
fn encode(tables: *const Tables, snap: *const Snap, own_active_state: *const UMon, opp_active_state: *const UMon, party: []const UMon, revealed_in: []const UMon, last: *const LastTurn, turn: u32, ex: *const Extras, x: *rmodel.Input, ordered: *[6]UMon) !usize {
    const n_own = canonicalOwn(tables, own_active_state, party, ordered);
    var own_active = own_active_state.*;
    sortMoves(tables, &own_active);
    var revealed: [6]UMon = undefined;
    @memcpy(revealed[0..revealed_in.len], revealed_in);
    for (revealed[0..revealed_in.len]) |*p| sortMoves(tables, p);
    sortBySpecies(tables, revealed[0..revealed_in.len]);
    var opp_active = opp_active_state.*;
    sortMoves(tables, &opp_active);
    x.* = .{};
    for (ordered[0..n_own], 0..) |*p, slot| {
        const live = p.hp_pct > 0;
        x.own_species[slot] = tables.species_id[@intFromEnum(p.species)];
        x.own_status[slot] = @intFromEnum(p.status);
        for (p.moves[0..@min(p.nmoves, 4)], 0..) |m, j| {
            x.own_moves[slot][j] = tables.move_id[@intFromEnum(m.id)];
            const mx: f64 = @floatFromInt(@max(1, m.max_pp));
            x.own[slot][6 + j] = if (m.max_pp != 0) @floatCast(@min(1.0, @as(f64, @floatFromInt(m.pp)) / mx)) else 0.0;
        }
        x.own[slot][0] = if (live) @floatCast(p.hp_pct) else 0.0;
        x.own[slot][1] = if (p.species == own_active.species) 1.0 else 0.0;
        x.own[slot][2] = boostFeature(p.atk);
        x.own[slot][3] = boostFeature(p.def);
        x.own[slot][4] = boostFeature(p.spe);
        x.own[slot][5] = boostFeature(p.spa);
    }
    for (revealed[0..revealed_in.len], 0..) |*p, slot| {
        x.opp_species[slot] = tables.species_id[@intFromEnum(p.species)];
        x.opp_status[slot] = @intFromEnum(p.status);
        for (p.moves[0..@min(p.nmoves, 4)], 0..) |m, j| x.opp_moves[slot][j] = tables.move_id[@intFromEnum(m.id)];
        x.opp[slot][0] = 1.0;
        x.opp[slot][1] = @floatCast(p.hp_pct);
        x.opp[slot][2] = if (p.species == opp_active.species) 1.0 else 0.0;
        x.opp[slot][3] = boostFeature(p.atk);
        x.opp[slot][4] = boostFeature(p.def);
        x.opp[slot][5] = boostFeature(p.spe);
        x.opp[slot][6] = boostFeature(p.spa);
    }
    const sides = [2]struct { Kind, Move, Species, ?Cant, f64, bool, Status }{
        .{ last.own_kind, last.own_move, last.own_species, last.own_cant, last.own_hp_change, last.own_fainted, last.own_status },
        .{ last.opp_kind, last.opp_move, last.opp_species, last.opp_cant, last.opp_hp_change, last.opp_fainted, last.opp_status },
    };
    for (sides, 0..) |sd, side| {
        const base = side * 17;
        x.last[base + @intFromEnum(sd[0])] = 1.0;
        x.last[base + 4] = @floatCast(@max(-1.0, @min(1.0, sd[4])));
        x.last[base + 5] = if (sd[5]) 1.0 else 0.0;
        if (sd[3]) |c| x.last[base + 6 + @intFromEnum(c)] = 1.0;
        const move_id = if (sd[0] == .move) tables.move_id[@intFromEnum(sd[1])] else 0;
        const species_id = if (sd[0] == .switch_) tables.species_id[@intFromEnum(sd[2])] else 0;
        if (side == 0) {
            x.last_own_move = move_id;
            x.last_own_species = species_id;
            x.last_own_status = @intFromEnum(sd[6]);
        } else {
            x.last_opp_move = move_id;
            x.last_opp_species = species_id;
            x.last_opp_status = @intFromEnum(sd[6]);
        }
    }
    const own_stats = statsOf(tables, own_active.species) orelse return error.NoBaseStats;
    const opp_stats = statsOf(tables, opp_active.species) orelse return error.NoBaseStats;
    var f: usize = 0;
    for (0..4) |j| {
        x.field[f + j] = if (j < own_active.nmoves) @floatCast(moveDamage(tables, 0, &own_active, &opp_active, own_active.moves[j].id, own_stats, opp_stats, ex, own_stats.hp, opp_stats.hp)) else 0.0;
    }
    for (0..4) |j| {
        x.field[f + 4 + j] = if (j < opp_active.nmoves) @floatCast(moveDamage(tables, 1, &opp_active, &own_active, opp_active.moves[j].id, opp_stats, own_stats, ex, own_stats.hp, opp_stats.hp)) else 0.0;
    }
    f += 8;
    x.field[f] = @floatCast(own_stats.spe / 500.0);
    x.field[f + 1] = @floatCast(opp_stats.spe / 500.0);
    f += 2;
    const edges = [_]u32{ 2, 4, 8, 16, 32, 64, 128 };
    var bucket: usize = edges.len;
    for (edges, 0..) |e, i| if (turn < e) {
        bucket = i;
        break;
    };
    x.field[f + bucket] = 1.0;
    f += 8;
    var alive: usize = 0;
    for (ordered[0..n_own]) |p| alive += @intFromBool(p.hp_pct > 0);
    var fainted_opp: usize = 0;
    for (snap.revealed[0..snap.n_revealed]) |r| fainted_opp += @intFromBool(r.status == .fnt);
    x.field[f] = @floatCast(@as(f64, @floatFromInt(alive)) / 6.0);
    x.field[f + 1] = @floatCast(@as(f64, @floatFromInt(6 - fainted_opp)) / 6.0);
    f += 2;
    x.field[f] = if (snap.force_switch) 1.0 else 0.0;
    x.field[f + 1] = 0.0;
    f += 2;
    const screens = [3]Eff{ .reflect, .lightscreen, .mist };
    for (screens, 0..) |sc, i| x.field[f + i] = if (own_active.effect != null and own_active.effect.? == sc) 1.0 else 0.0;
    for (screens, 0..) |sc, i| x.field[f + 3 + i] = if (opp_active.effect != null and opp_active.effect.? == sc) 1.0 else 0.0;
    f += 6;
    x.field[f] = @floatCast(ex.trapped[0]);
    x.field[f + 1] = @floatCast(ex.trapped[1]);
    f += 2;
    const fx = [4]Pv{ .substitute, .confusion, .leechseed, .focusenergy };
    for (0..2) |side| {
        for (fx, 0..) |e, i| x.field[f + side * 4 + i] = if (ex.effects[side].has(e)) 1.0 else 0.0;
    }
    f += 8;
    x.field[f] = 0.0; // derived
    f += 1;
    std.debug.assert(f == 39);
    return n_own;
}

// ---------------------------------------------------------------- one node

pub const Native = struct {
    model: rmodel.Model,
    tables: *Tables,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !*Native {
        const n = try allocator.create(Native);
        n.* = .{ .model = try rmodel.Model.load(allocator, io, dir), .tables = try Tables.load(allocator, io, dir) };
        return n;
    }

    /// The child of `parent` (null: a game's first decision) after its player chose `sent`, reading `snap`
    /// (recurrent_search_server.Server.step). `logits[i]` is legal option i's logit (UNMAPPED without a policy slot).
    pub fn step(self: *const Native, parent: ?*const Node, sent: Option, snap: *const Snap, out: *Node, logits: *[pkmn.CHOICES_SIZE]f32, input_out: ?*rmodel.Input) !void {
        var input: rmodel.Input = undefined;
        if (!try self.begin(parent, sent, snap, out, logits, &input)) return;
        if (input_out) |io| io.* = input;
        var h2: [rmodel.D]f32 = undefined;
        var slot_logits: [rmodel.ACTIONS]f32 = undefined;
        const win = self.model.step(&input, &out.game.h, &h2, &slot_logits);
        end(out, snap, win, &h2, &slot_logits, logits);
    }

    /// Several independent children at once (each parent an existing node): the model steps all the decisions as one batch.
    /// `scratch` grows as needed.
    pub fn stepMany(self: *const Native, allocator: std.mem.Allocator, parents: []const ?*const Node, sents: []const Option, outs: []*Node, logits: [][pkmn.CHOICES_SIZE]f32) !void {
        const n = parents.len;
        const inputs = try allocator.alloc(rmodel.Input, n);
        const which = try allocator.alloc(usize, n);
        var m: usize = 0;
        for (0..n) |i| {
            if (try self.begin(parents[i], sents[i], &outs[i].snap, outs[i], &logits[i], &inputs[m])) {
                which[m] = i;
                m += 1;
            }
        }
        if (m == 0) return;
        const h_prev = try allocator.alloc([rmodel.D]f32, m);
        for (0..m) |j| h_prev[j] = outs[which[j]].game.h;
        const h_out = try allocator.alloc([rmodel.D]f32, m);
        const slot_logits = try allocator.alloc([rmodel.ACTIONS]f32, m);
        const wins = try allocator.alloc(f32, m);
        const scratch = try allocator.alloc(f32, rmodel.Model.scratchLen(m));
        self.model.stepBatch(inputs[0..m], h_prev, h_out, slot_logits, wins, scratch);
        for (0..m) |j| end(outs[which[j]], &outs[which[j]].snap, wins[j], &h_out[j], &slot_logits[j], &logits[which[j]]);
    }

    /// Everything of a step before the model: the child's memory and its model input. False for a wait state (then the node and
    /// its logits are complete).
    fn begin(self: *const Native, parent: ?*const Node, sent: Option, snap: *const Snap, out: *Node, logits: *[pkmn.CHOICES_SIZE]f32, input: *rmodel.Input) !bool {
        // (out.snap is the snapshot: the caller writes it there; everything else of the node starts fresh)
        if (snap != &out.snap) out.snap = snap.*;
        out.game = .{};
        out.own = .{};
        out.opp = .{};
        out.decision = false;
        out.win = 0;
        if (parent) |p| {
            out.game = p.game;
            if (p.decision) {
                // finish(): what the parent's player answered there, and the parent's snapshot as the previous one
                out.game.prev_node = p;
                out.game.sent = sent;
                out.game.opp_faint_switch = false;
                out.game.acts = .{ null, null };
            }
        }
        const decision = try self.prepare(&out.game, snap, input, &out.own, &out.opp, logits);
        out.decision = decision;
        if (!decision) {
            out.win = if (parent) |p| p.win else 0.0;
            if (parent) |p| {
                out.own = p.own;
                out.opp = p.opp;
            }
            for (0..snap.n_legal) |i| logits[i] = 0.0;
        }
        return decision;
    }

    /// Everything of a step after the model: the new state and win logit, and each legal option's logit (logits[] held each
    /// option's slot, from prepare).
    fn end(out: *Node, snap: *const Snap, win: f32, h2: *const [rmodel.D]f32, slot_logits: *const [rmodel.ACTIONS]f32, logits: *[pkmn.CHOICES_SIZE]f32) void {
        out.win = win;
        out.game.h = h2.*;
        for (0..snap.n_legal) |i| {
            const slot = logits[i];
            logits[i] = if (slot < 0) UNMAPPED else slot_logits[@intFromFloat(slot)];
        }
    }

    /// recurrent_policy_server.prepare (and legal_slots: `slots[i]` gets legal option i's policy slot, or -1)
    fn prepare(self: *const Native, g: *Game, snap: *const Snap, x: *rmodel.Input, own_out: *UMon, opp_out: *UMon, slots: *[pkmn.CHOICES_SIZE]f32) !bool {
        const t = self.tables;
        // note_actions
        for (0..2) |side| {
            if (g.acts[side] == null or (g.acts[side].?.executed == .None and !g.acts[side].?.recharged)) g.acts[side] = snap.acts[side];
        }
        if (snap.n_legal == 1 and snap.legal[0].kind == .nomove and snap.opp_force_switch and !snap.force_switch) {
            g.opp_faint_switch = true;
            return false;
        }
        if (snap.n_party == 0) return error.EmptyParty;
        // metamon_connector.state: the actives
        var own = ownUMon(&snap.party[0], snap.boosts);
        var opp = UMon{ .species = snap.opp_species, .hp_pct = @max(0.0, @min(1.0, snap.opp_hp)), .status = snap.opp_status, .types = snap.opp_types };
        opp.atk = snap.opp_boosts.atk;
        opp.def = snap.opp_boosts.def;
        opp.spe = snap.opp_boosts.spe;
        opp.spa = snap.opp_boosts.spc;
        opp.spd = snap.opp_boosts.spc;
        for (snap.revealed[0..snap.n_revealed]) |r| {
            if (r.species == snap.opp_species) {
                uMoves(&opp, r.moves[0..r.nmoves]);
                break;
            }
        }
        // the full party (the active from the state, the others from the snapshot)
        var party: [6]UMon = undefined;
        party[0] = own;
        for (1..snap.n_party) |i| party[i] = ownUMon(&snap.party[i], null);
        // the last-turn record
        if (g.prev_node) |pn| {
            const o = ownCant(g, snap, &own, &opp, g.acts[0].?);
            const p = oppCant(g, snap, &own, &opp, g.acts[1].?);
            const oc = change(&pn.own, &own);
            const pc = change(&pn.opp, &opp);
            g.last = .{
                .own_kind = o.kind,
                .own_move = o.move,
                .own_species = o.species,
                .own_cant = o.cant,
                .opp_kind = p.kind,
                .opp_move = p.move,
                .opp_species = p.species,
                .opp_cant = p.cant,
                .own_hp_change = oc[0],
                .opp_hp_change = pc[0],
                .own_status = oc[1],
                .opp_status = pc[1],
                .own_fainted = oc[2],
                .opp_fainted = pc[2],
            };
            for ([2]Rec{ o, p }, 0..) |rec, who| {
                if (rec.kind == .move) {
                    g.counter_selected[who] = rec.move == .Counter;
                } else if (rec.cant != null) g.counter_selected[who] = false;
            }
        }
        // effects
        var buf: [16]Eff = undefined;
        var n = activeEffects(snap, 0, &buf);
        own.effect = g.effects.select(buf[0..n], snap.party[0].species, snap.turn, 0);
        n = activeEffects(snap, 1, &buf);
        opp.effect = g.effects.select(buf[0..n], snap.opp_species, snap.turn, 1);
        party[0].effect = own.effect;
        // the revealed opponents (the active is the state's object)
        var revealed: [6]UMon = undefined;
        for (snap.revealed[0..snap.n_revealed], 0..) |r, i| {
            if (r.species == opp.species) {
                revealed[i] = opp;
            } else {
                var u = UMon{ .species = r.species, .hp_pct = @max(0.0, @min(1.0, r.hp_pct)), .status = r.status };
                uMoves(&u, r.moves[0..r.nmoves]);
                revealed[i] = u;
            }
        }
        // extras
        var ex = Extras{
            .last_damage = counterDamage(t, g, snap, &own, &opp),
            .last_move = .{ snap.last_move, snap.opp_last_move },
            .counter_selected = g.counter_selected,
            .cant = .{ g.last.own_cant, g.last.opp_cant },
            .effects = .{ snap.pv, snap.opp_pv },
            .trapped = .{ 0, 0 },
        };
        ex.trapped[0] = trapProbability(g, snap, 1); // player_trapped: caught in the opponent's trap
        ex.trapped[1] = trapProbability(g, snap, 0);
        g.decisions += 1;
        var ordered: [6]UMon = undefined;
        const n_own = try encode(t, snap, &own, &opp, party[0..snap.n_party], revealed[0..snap.n_revealed], &g.last, g.decisions, &ex, x, &ordered);
        own_out.* = own;
        opp_out.* = opp;
        // legal_slots: each option's policy slot
        var reserves: [5]Species = undefined;
        var n_res: usize = 0;
        for (ordered[1..n_own]) |p| if (p.alive() and n_res < 5) {
            reserves[n_res] = p.species;
            n_res += 1;
        };
        for (0..snap.n_legal) |i| {
            const opt = snap.legal[i];
            slots[i] = -1;
            switch (opt.kind) {
                .switch_ => for (reserves[0..n_res], 0..) |r, k| if (r == opt.species) {
                    slots[i] = @floatFromInt(4 + k);
                    break;
                },
                .move => for (ordered[0].moves[0..@min(ordered[0].nmoves, 4)], 0..) |m, k| if (m.id == opt.move) {
                    slots[i] = @floatFromInt(k);
                    break;
                },
                .nomove => {},
            }
        }
        return true;
    }
};

pub const UNMAPPED: f32 = -1e30;

/// The search's native recurrent model, when SEARCH_MODEL=native.
pub var active: ?*Native = null;
