//! The pkmn engine's protocol enums (pkmn/engine 78dc891, src/lib/common/protocol.zig, copied verbatim: the engine's public
//! API does not export them). The decoder (events.zig) reads the log's bytes with these values.

pub const ArgType = enum(u8) {
    None,

    // Special
    LastStill,
    LastMiss,

    // Gen 1
    Move,
    Switch,
    Cant,
    Faint,
    Turn,
    Win,
    Tie,
    Damage,
    Heal,
    Status,
    CureStatus,
    Boost,
    ClearAllBoost,
    Fail,
    Miss,
    HitCount,
    Prepare,
    MustRecharge,
    Activate,
    FieldActivate,
    Start,
    End,
    OHKO,
    Crit,
    SuperEffective,
    Resisted,
    Immune,
    Transform,

    // Gen 2
    Drag,
    Item,
    EndItem,
    CureTeam,
    SetHP,
    SetBoost,
    CopyBoost,
    SideStart,
    SideEnd,
    SingleMove,
    SingleTurn,
    Weather,

    // Gen 3
    Block,
    Ability,
    EndAbility,
    ClearNegativeBoost,
    FormeChange,
    LastNoTarget, // FIXME

    // Gen 4
    SwapBoost,
    FieldStart,
    FieldEnd,
    DetailsChange,

    // Gen 5
    ClearPoke,
    Poke,
    TeamPreview,
    Center,
    Swap,
    Combine,
    Waiting,
    Replace,
    ClearBoost,

    // Gen 6
    Mega,
    Primal,
    InvertBoost,

    // Gen 7
    ZBroken,
    ZPower,
    Burst,
    ClearPositiveBoost,

    // Gen 8
    CanDynamax,
    SwapSideConditions,
    UpdatePoke,

    // Gen 9
    Terastallize,
};

pub const Move = enum(u8) {
    None,
    From,
};

pub const Cant = enum(u8) {
    Sleep,
    Freeze,
    Paralysis,
    Bound,
    Flinch,
    Disable,
    Recharge,
    PP,

    Attract,
};

pub const Heal = enum(u8) {
    None,
    Silent,
    Drain,

    Leftovers,
    Berry,
    BerryJuice,
    GoldBerry,
};

pub const Damage = enum(u8) {
    None,
    Poison,
    Burn,
    Confusion,
    LeechSeed,
    RecoilOf,

    Bind,
    Wrap,
    FireSpin,
    Clamp,
    Whirlpool,
    Spikes,

    Nightmare,
    Curse,
    Sandstorm,
};

pub const Status = enum(u8) {
    None,
    Silent,
    From,
};

pub const CureStatus = enum(u8) {
    Message,
    Silent,
};

pub const Boost = enum(u8) {
    Rage,
    Attack,
    Defense,
    Speed,
    SpecialAttack,
    SpecialDefense,
    Accuracy,
    Evasion,
};

pub const Fail = enum(u8) {
    None,
    Sleep,
    Poison,
    Burn,
    Freeze,
    Paralysis,
    Toxic,
    Substitute,
    Weak,
};

pub const Activate = enum(u8) {
    Bide,
    Confusion,
    Haze,
    Mist,
    Struggle,
    Substitute, // FIXME SubstituteDamage
    Splash,

    Rage, // TODO

    Attract,
    LockOn,
    Bind,
    Wrap,
    FireSpin,
    Clamp,
    Whirlpool,
    SubstituteBlock,

    DestinyBond,

    BeatUp, // FIXME of

    Magnitude,
    Spite,
};

pub const Start = enum(u8) {
    Bide,
    Confusion,
    ConfusionSilent,
    FocusEnergy,
    LeechSeed,
    LightScreen,
    Mist,
    Reflect,
    Substitute,

    TypeChange,

    Disable,
    Mimic,

    Attract,
    Nightmare,
    Curse,
    Foresight,
    Encore,
    FutureSight,

    PerishSong0,
    PerishSong1,
    PerishSong2,
    PerishSong3,
    PerishSong3Silent,
};

pub const End = enum(u8) {
    Disable,
    Confusion,
    Bide,
    Substitute,

    Nightmare,
    Curse,
    Foresight,
    Encore,
    FutureSight,
    LeechSeed,

    Bind,
    Wrap,
    FireSpin,
    Clamp,
    Whirlpool,

    DisableSilent,
    ConfusionSilent,
    MistSilent,
    FocusEnergySilent,
    LeechSeedSilent,
    ToxicSilent,
    LightScreenSilent,
    ReflectSilent,
    BideSilent,

    LeechSeedFrom,
};

pub const Immune = enum(u8) {
    None,
    OHKO,
};

pub const EndItem = enum(u8) {
    None,
    Eat,
};

pub const SetHP = enum(u8) {
    None,
    Silent,
};

pub const Side = enum(u8) {
    Safeguard,
    Reflect,
    LightScreen,
    Spikes,
};

pub const Weather = enum(u8) {
    None,
    Upkeep,
};

