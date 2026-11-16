"""Student vs TaurosV0 on a real Pokémon Showdown server, Gen 1 OU, both sides drawing teams from the same team set.

Starts the server (node pokemon-showdown start --no-security PORT), TaurosV0 as the acceptor (tauros_acceptor.py, its own
metamon client), and the student as the challenger (student_player.py); plays --games battles one after another; stops all.

    python run_showdown.py --weights W --showdown ~/showdown --tauros-metamon UPSTREAM --cache CACHE --teams DIR --games 100 --out OUT
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--showdown", required=True, help="a built pokemon-showdown checkout")
ap.add_argument("--tauros-metamon", required=True, help="upstream metamon with duel/tauros_metamon.patch applied")
ap.add_argument("--cache", required=True, help="metamon cache with TaurosV0's checkpoint, usage stats and teams/<team-set>")
ap.add_argument("--teams", required=True, help="the student's team files (the same set)")
ap.add_argument("--team-set", default="b1_good_807", help="TaurosV0's team set name in the cache")
ap.add_argument("--games", type=int, default=100)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "METAMON_CACHE_DIR": args.cache, "HF_HUB_OFFLINE": "1", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"}
os.environ.update({k: env[k] for k in ("METAMON_CACHE_DIR", "HF_HUB_OFFLINE")})  # (the student runs in this process; metamon reads these at import)
procs = []


def start(cmd, log, cwd=None, extra_env=None):
    handle = open(out / log, "w")
    procs.append(subprocess.Popen(cmd, cwd=cwd, env={**env, **(extra_env or {})}, stdout=handle, stderr=subprocess.STDOUT))
    return procs[-1]


try:
    server = start(["node", "pokemon-showdown", "start", "--no-security", str(args.port)], "server.txt", cwd=args.showdown)
    for _ in range(120):
        try:
            import urllib.request

            urllib.request.urlopen(f"http://localhost:{args.port}", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    print(f"server up on {args.port}", flush=True)
    here = Path(__file__).resolve().parent
    tauros = start([sys.executable, str(here / "tauros_acceptor.py"), "--metamon-root", args.tauros_metamon, "--cache", args.cache,
                    "--username", "TaurosVZeroEval", "--opponent", "StudentEval", "--battles", str(args.games), "--team-set", args.team_set,
                    "--team-dir", args.teams,
                    "--temperature", str(args.temperature), "--results", str(out / "tauros_results"),
                    "--trajectories", str(out / "tauros_trajectories")], "tauros.txt")

    sys.path.insert(0, str(here))
    from poke_env import AccountConfiguration, ServerConfiguration

    from student_player import StudentPlayer

    config = ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")
    student = StudentPlayer(args.weights, args.teams, temperature=args.temperature, seed=args.seed, log_dir=str(out / "battles"),
                            account_configuration=AccountConfiguration("StudentEval", None), server_configuration=config,
                            max_concurrent_battles=1, start_timer_on_battle_start=False)

    async def main():
        # TaurosV0 needs time to load before it can accept
        for _ in range(600):
            if "Made Challenge Env" in (out / "tauros.txt").read_text(errors="replace") or tauros.poll() is not None:
                break
            await asyncio.sleep(1)
        await asyncio.sleep(5)
        from poke_env.concurrency import handle_threaded_coroutines

        async def challenge_and_wait_start():
            # poke-env's send_challenges, with a timeout on the battle starting (a challenge TaurosV0 does not take is
            # cancelled and sent again)
            for attempt in range(5):
                await student.ps_client.challenge("TaurosVZeroEval", student._format, student.next_team)
                try:
                    await asyncio.wait_for(student._battle_semaphore.acquire(), timeout=90)
                    return attempt
                except asyncio.TimeoutError:
                    print(f"  challenge not accepted in 90 s (attempt {attempt + 1}); cancelling and sending again", flush=True)
                    await student.ps_client.send_message("/cancelchallenge TaurosVZeroEval")
                    await asyncio.sleep(5)
            raise RuntimeError("TaurosV0 does not accept challenges")

        start_time = time.time()
        for g in range(args.games):
            if tauros.poll() is not None:
                print("TaurosV0 process exited", flush=True)
                break
            started = time.time()
            await handle_threaded_coroutines(challenge_and_wait_start())
            while student.n_finished_battles <= g:  # (the battle in progress)
                if tauros.poll() is not None or time.time() - started > 900:
                    raise RuntimeError(f"game {g + 1} did not finish (TaurosV0 {'exited' if tauros.poll() is not None else 'alive'})")
                await asyncio.sleep(1)
            w, f = student.n_won_battles, student.n_finished_battles
            print(f"{f} games: student {w}, TaurosV0 {student.n_lost_battles}, ties {student.n_tied_battles} "
                  f"({time.time() - start_time:.0f}s; this one {time.time() - started:.0f}s) {dict(student.stats)}", flush=True)

    asyncio.run(main())
    n, w, l, t = student.n_finished_battles, student.n_won_battles, student.n_lost_battles, student.n_tied_battles
    score = (w + 0.5 * t) / max(1, n)
    se = (score * (1 - score) / max(1, n)) ** 0.5
    result = {"games": n, "student": w, "tauros": l, "ties": t, "score": score, "se": se, "stats": dict(student.stats),
              "temperature": args.temperature, "weights": args.weights}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    for k, v in sorted(student.unmapped.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  unmapped {v}x: {k}", flush=True)
    result["unmapped"] = dict(student.unmapped)
    valid = not student.stats.get("errors") and not student.stats.get("fallbacks")
    result["valid"] = valid
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"RESULT{'' if valid else ' (INVALID: the student failed to decide at some decisions)'} student {w} - TaurosV0 {l} ({t} ties) of {n}: "
          f"student scores {score:.1%} ± {se:.1%}; {dict(student.stats)}", flush=True)
finally:
    for p in reversed(procs):
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
