"""Compares the encoder's max-roll damage with the Zig engine's observed maximum.

    pkmn-harness 2> harness_out.txt      (in pkmn-search; lines: `<atk>_to_<def> <Move> lo hi samples max_hp`)
    python -m metamon.recurrent.check_damage harness_out.txt

The harness measures the defender's total HP loss in a turn, which includes a burn/poison tick (max HP / 16)
and confusion self-hits when the move inflicts them, so Burn/Poison/Confusion-inflicting moves read high;
those are reported separately. Multi-hit moves and Bide are not modeled and are skipped; Counter, Psywave and Super Fang
are skipped here too, because Counter depends on battle memory (see check_oracle.py, which grades it) and the other two are
checked against this harness's maxima in test form.
"""
import re
import sys
from types import SimpleNamespace as NS

from metamon.backend.showdown_dex.dex import Dex
from metamon.recurrent import features as F
from metamon.recurrent.tables import Tables

SKIP = {"doubleslap", "cometpunch", "furyattack", "furyswipes", "pinmissile", "spikecannon", "barrage",
        "bonemerang", "doublekick", "twineedle", "triattack", "counter", "bide", "psywave", "superfang",
        "submission", "metronome", "mirrormove", "mimic", "transform", "disable", "conversion"}


def main(path: str) -> None:
    tables = Tables.from_dex(1)
    dex = Dex(1).load_pokedex(1)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    types = {norm(k): " ".join(norm(x) for x in v["types"]) for k, v in dex.items() if norm(k) in tables.base}

    def mon(n):
        return NS(name=n, types=types[n], atk_boost=0, def_boost=0, spa_boost=0, spd_boost=0, spe_boost=0,
                  atk_stat=-1, def_stat=-1, spa_stat=-1, spd_stat=-1, spe_stat=-1, hp_stat=-1)

    exact = near = side_effect = bad = 0
    bad_rows = []
    for line in open(path):
        parts = line.split()
        if len(parts) != 6:
            continue
        name, move, _, hi, _, hp = parts
        hi, hp = int(hi), int(hp)
        a, d = name.split("_to_")
        mv = norm(move)
        if mv in SKIP:
            continue
        A, D = mon(a), mon(d)
        sa, sd = F._own_stats(A, tables), F._opp_stats(D, tables)
        assert int(sd["hp"]) == hp, (name, sd["hp"], hp)
        pred = round(F._damage_fraction(A, D, NS(name=mv), tables, sa, sd) * sd["hp"])
        if mv in F.OHKO_MOVES and pred >= hp - 1 and hi >= hp - 1:
            exact += 1
        elif pred == hi:
            exact += 1
        elif abs(pred - hi) <= 1:
            near += 1
        elif pred > 0 and hi > pred and (hi - pred in (hp // 16, hp // 16 - 1, hp // 16 + 1) or mv in ("psybeam", "confusion")):
            side_effect += 1
        elif pred == 0 and hi > 0:
            side_effect += 1  # status moves whose effect (poison, confusion) hurt the target
        else:
            bad += 1
            bad_rows.append((name, move, pred, hi))
    print(f"exact {exact}, within 1 {near}, read high from a status tick or confusion hit {side_effect}, unexplained {bad}")
    for r in bad_rows[:20]:
        print("  ", r)


if __name__ == "__main__":
    main(sys.argv[1])
