"""
Anything in this file was originally copied from poke-env.

https://github.com/hsahovic/poke-env

This contains the low-level data types that were originally used by our replay parser
to cut sim2sim gap with poke-env (weather and effects are always spelled the same,
etc.). 1:1 compatibility with poke-env is now deprecated, and we use a frozen fork of poke-env
to maintain the performance of our original models.

Therefore any future changes to the list of effects, weather, conditions, etc., would prompt
the full dtype to be copied from the version here https://github.com/UT-Austin-RPL/poke-env and
pasted into this file.

This is the last piece that sets up a clean split from poke-env below the "Player/Battle" layer.
"""

from poke_env.environment import (
    Effect,
    Field,
    Move,
    SideCondition,
    Status,
    Weather,
    STACKABLE_CONDITIONS,
)

PEEffect = Effect
PEField = Field
PEMove = Move
PESideCondition = SideCondition
PEStatus = Status
PEWeather = Weather
PEStackableConditions = STACKABLE_CONDITIONS
