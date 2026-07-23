import os
import re
import json
from enum import Enum


class Tier(Enum):
    UBERS = 0
    OU = 1
    UU = 2
    RU = 3
    NU = 4
    PU = 5
    ZU = 6
    NFE = 7
    LC = 8

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented


CAP_TIER_MAP = {
    "(OU)": Tier.OU,
    "OU": Tier.OU,
    "Uber": Tier.UBERS,
    "UU": Tier.UU,
    "RU": Tier.RU,
    "NU": Tier.NU,
    "PU": Tier.PU,
    "ZU": Tier.ZU,
    "NFE": Tier.NFE,
    "LC": Tier.LC,
    "OUBL": Tier.UBERS,
    "UUBL": Tier.OU,
    "RUBL": Tier.UU,
    "NUBL": Tier.RU,
    "PUBL": Tier.NU,
    "ZUBL": Tier.PU,
}


def ts_to_dict(ts_file):
    # 1) Read file and wrap in braces
    with open(ts_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return {}
    lines[0] = "{"
    lines[-1] = "}"
    content = "".join(lines)

    # 2) Remove any literal ellipses
    content = content.replace("...", "").replace("…", "")

    # 3) Temporarily escape the colon in 'Type: Null'
    content = content.replace("Type: Null", "Type Null")

    # 4) Quote unquoted keys (foo: → "foo":)
    content = re.sub(r"(\w+)\s*:", r'"\1":', content)

    # 5) Escape any unescaped inner quotes
    #    (e.g. Dragon"s → Dragon\"s, Oricorio-Pa"u → Oricorio-Pa\"u)
    content = re.sub(r'([A-Za-z0-9\-])"(?=[A-Za-z0-9])', r'\1\\"', content)

    # 6) Convert single-quoted values to double-quoted
    #    (e.g. 'nidoranf' → "nidoranf")
    content = re.sub(
        r"(:\s*)\'([^']*)\'", lambda m: f'{m.group(1)}"{m.group(2)}"', content
    )

    # 7) Remove JavaScript comments
    content = re.sub(r"//.*", "", content)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

    # 8) Fill missing values with null ( "key":, → "key": null, )
    content = re.sub(r'(".*?")\s*:\s*(?=[,\}])', r"\1: null", content)

    # 9) Remove trailing commas before } or ]
    content = re.sub(r",\s*(?=[\}\]])", "", content)

    # 10) Restore the 'Type: Null' string
    content = content.replace("Type Null", "Type: Null")

    # 11) Strip out newlines and tabs
    content = content.replace("\n", "").replace("\t", "")
    # 11) Parse the cleaned JSON
    data = json.loads(content)
    return data


def get_valid_pokemon(format):
    ps_path = os.path.join(
        os.path.abspath(__file__).split("metamon")[0],
        "metamon",
        "server",
        "pokemon-showdown",
    )
    if not os.path.exists(ps_path):
        raise ImportError("Cannot find path to pokemon-showdown")

    if "gen9" in format:
        formats_data = "formats-data.ts"
    else:
        generation = format[:4]
        formats_data = f"mods/{generation}/formats-data.ts"

    dex_data = "pokedex.ts"
    data_file = os.path.join(ps_path, "data", formats_data)
    dex_file = os.path.join(ps_path, "data", dex_data)
    formats_data = ts_to_dict(data_file)
    dex_data = ts_to_dict(dex_file)

    valid_pokemon = {}
    for pm, tiers in formats_data.items():
        if "tier" not in tiers:
            continue
        if pm not in dex_data:
            continue
        tier = tiers["tier"]
        if tier not in CAP_TIER_MAP:
            continue
        tier = CAP_TIER_MAP[tier]
        name = dex_data[pm]["name"]
        name = re.sub(r"[^a-zA-Z0-9]", "", name)
        if tier not in valid_pokemon:
            valid_pokemon[tier] = []
        valid_pokemon[tier].append(name)
    return valid_pokemon
