import tqdm
import uuid
from dataclasses import dataclass
from poke_env import AccountConfiguration
import asyncio

from metamon.baselines import get_baseline
from metamon.env import get_metamon_teams


@dataclass
class MatchupResult:
    player_name: str
    opponent_name: str
    battle_format: str
    team_split: str
    win_pct: float
    lose_pct: float
    tie_pct: float


async def _run_battles(player, opponent, n_battles=5):
    player.reset_battles()
    opponent.reset_battles()
    await player.battle_against(opponent, n_battles=n_battles)
    return player.win_rate, player.lose_rate, player.tie_rate


def head2head(
    player_name: str,
    opponent_name: str,
    battle_format: str,
    team_split: str,
    n_battles: int,
    max_concurrent: int = 5,
    verbose: bool = False,
) -> MatchupResult:
    iter_ = range(n_battles)
    if verbose:
        iter_ = tqdm.tqdm(iter_)
    all_win_rates, all_lose_rates, all_tie_rates = [], [], []
    for _ in iter_:
        player = get_baseline(player_name)(
            battle_format=battle_format,
            team=get_metamon_teams(battle_format, team_split),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(
                username=f"p{str(uuid.uuid4())[:15]}", password=None
            ),
        )
        opponent = get_baseline(opponent_name)(
            battle_format=battle_format,
            team=get_metamon_teams(battle_format, team_split),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(
                username=f"p{str(uuid.uuid4())[:15]}", password=None
            ),
        )
        player.randomize()
        opponent.randomize()
        win_rate, lose_rate, tie_rate = asyncio.get_event_loop().run_until_complete(
            _run_battles(player, opponent)
        )
        all_win_rates.append(win_rate)
        all_lose_rates.append(lose_rate)
        all_tie_rates.append(tie_rate)

    avg = lambda x: 100 * (sum(x) / len(x))
    return MatchupResult(
        player_name=player_name,
        opponent_name=opponent_name,
        battle_format=battle_format,
        team_split=team_split,
        win_pct=avg(all_win_rates),
        lose_pct=avg(all_lose_rates),
        tie_pct=avg(all_tie_rates),
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()
    parser.add_argument("--battle_format", required=True)
    parser.add_argument("--team_split", type=str, default="paper_replays")
    parser.add_argument("--player", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--battles", type=int, default=10)
    parser.add_argument("--concurrent", type=int, default=10)
    args = parser.parse_args()
    result = head2head(
        player_name=args.player,
        opponent_name=args.opponent,
        battle_format=args.battle_format,
        team_split=args.team_split,
        n_battles=args.battles,
        max_concurrent=args.concurrent,
        verbose=True,
    )

    print(
        f"{result.player_name} vs {result.opponent_name} on {result.battle_format} with {result.team_split} teams: Wins {result.win_pct: .2f}% / Loses {result.lose_pct : .2f}% / Ties {result.tie_pct : .2f}%"
    )
