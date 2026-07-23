HIDDEN_POWER_IVS = {
    "Bug": "31/0/30/31/30/31",
    "Dark": "31/1/31/31/31/31",
    "Dragon": "31/0/31/31/31/31",
    "Electric": "31/1/31/30/31/31",
    "Fighting": "31/1/30/30/30/30",
    "Fire": "31/0/31/30/31/30",
    "Flying": "30/0/30/30/30/31",
    "Ghost": "31/1/30/31/30/31",
    "Grass": "31/0/31/30/31/31",
    "Ground": "31/1/31/30/30/31",
    "Ice": "31/0/30/31/31/31",
    "Poison": "31/1/30/30/30/31",
    "Psychic": "31/0/31/31/31/30",
    "Rock": "31/1/30/31/30/30",
    "Steel": "31/1/31/31/30/31",
    "Water": "31/0/30/30/31/31",
}


HIDDEN_POWER_DVS = {
    "Bug": "30/26/26/30/30/30",
    "Dark": "30/30/30/30/30/30",
    "Dragon": "22/30/28/30/30/30",
    "Electric": "14/28/30/30/30/30",
    "Fighting": "6/24/24/30/30/30",
    "Fire": "6/28/24/30/30/30",
    "Flying": "14/24/26/30/30/30",
    "Ghost": "22/26/28/30/30/30",
    "Grass": "6/28/28/30/30/30",
    "Ground": "14/24/30/30/30/30",
    "Ice": "30/30/26/30/30/30",
    "Poison": "6/24/28/30/30/30",
    "Psychic": "22/30/24/30/30/30",
    "Rock": "22/26/24/30/30/30",
    "Steel": "30/26/30/30/30/30",
    "Water": "14/28/26/30/30/30",
}

INCOMPATIBLE_MOVES = {
    "gen1": {
        "GENERALBANLIST": [],
    },
    "gen2": {
        "GENERALBANLIST": [
            ("Hypnosis", "Mean Look"),
            ("Hypnosis", "Spider Web"),
            ("Lovely Kiss", "Mean Look"),
            ("Lovely Kiss", "Spider Web"),
            ("Sing", "Mean Look"),
            ("Sing", "Spider Web"),
            ("Sleep Powder", "Mean Look"),
            ("Sleep Powder", "Spider Web"),
            ("Spore", "Mean Look"),
            ("Spore", "Spider Web"),
        ],
        "Golem": [
            ("Explosion", "Rapid Spin"),
            ("Rock Slide", "Rapid Spin"),
            ("Body Slam", "Rapid Spin"),
            ("Take Down", "Rapid Spin"),
        ],
        "Gengar": [
            ("Explosion", "Perish Song"),
        ],
        "Cloyster": [
            ("Explosion", "Rapid Spin"),
        ],
        "Umbreon": [
            ("Mean Look", "Baton Pass"),
        ],
        "Blissey": [
            ("Heal Bell", "Thunder Wave"),
        ],
        "Charizard": [
            ("Belly Drum", "Crunch"),
            ("Belly Drum", "Seismic Toss"),
        ],
        "Scizor": [
            ("Baton Pass", "Double-Edge"),
        ],
        "Tentacruel": [("Rapid Spin", "Substitute")],
        "Exeggutor": [
            ("Explosion", "Synthesis"),
        ],
    },
    "gen3": {
        "GENERALBANLIST": [],
        "Charizard": [
            ("Belly Drum", "Beat Up"),
        ],
        "Blissey": [
            ("Wish", "Aromatherapy"),
        ],
        "Skarmory": [
            ("Drill Peck", "Whirlwind"),
        ],
        "Tyranitar": [
            ("Pursuit", "Dragon Dance"),
        ],
        "Salamence": [
            ("Dragon Dance", "Wish"),
        ],
        "Scizor": [
            ("Reversal", "Silver Wind"),
        ],
        "Gengar": [
            ("Will-O-Wisp", "Perish Song"),
        ],
        "Smeargle": [
            ("Spore", "Baton Pass"),
            # hack for gen2 one boost passer clause
            (
                "Aility",
                "Amnesia",
                "Acid Armor",
                "Belly Drum",
                "Bulk Up",
                "Calm Mind",
                "Dragon Dance",
                "Swords Dance",
                "Tail Glow",
            ),
        ],
    },
    "gen4": {
        "GENERALBANLIST": [],
        "Dragonite": [
            ("Extreme Speed", "Heal Bell"),
        ],
        "Blissey": [
            ("Aromatherapy", "Seismic Toss"),
        ],
        "Tyranitar": [
            ("Pursuit", "Dragon Dance"),
        ],
        "Roserade": [
            ("Sleep Powder", "Spikes"),
        ],
        "Skarmory": [
            ("Drill Peck", "Brave Bird"),
            ("Counter", "Brave Bird"),
        ],
    },
    "gen5": {
        "GENERALBANLIST": [],
    },
    "gen6": {
        "GENERALBANLIST": [],
    },
    "gen7": {
        "GENERALBANLIST": [],
    },
    "gen8": {
        "GENERALBANLIST": [],
    },
    "gen9": {
        "GENERALBANLIST": [],
    },
}

# Add general banlist to all pokemons
for format in INCOMPATIBLE_MOVES.keys():
    for mon in INCOMPATIBLE_MOVES[format].keys():
        if mon != "GENERALBANLIST":
            INCOMPATIBLE_MOVES[format][mon].extend(
                INCOMPATIBLE_MOVES[format]["GENERALBANLIST"]
            )
