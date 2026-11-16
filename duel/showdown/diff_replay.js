// Replays Zig duel games (DUEL_REPLAY=1 logs) in Pokémon Showdown's simulator with the same seed (the engine's PSRNG seed is
// Showdown's "gen5,<16 hex digits>") and the same choices, and compares the state after every update: the turn, each side's
// active Pokémon, every Pokémon's HP and status, and the result. Reports each game's first divergence.
//
//   node diff_replay.js SHOWDOWN_DIR TEAM_DIR LOG_FILE... [--verbose]
const fs = require('fs');
const path = require('path');
const [showdown, teamDir, ...rest] = process.argv.slice(2);
const verbose = rest.includes('--verbose');
const logs = rest.filter(a => !a.startsWith('--'));
// SHOWDOWN_DIR: a built pokemon-showdown checkout (dist/sim), or a folder with @pkmn/sim installed (the packaged simulator the
// engine's integration tests use)
const local = fs.existsSync(path.join(showdown, 'dist/sim'));
const sim = require(local ? path.join(showdown, 'dist/sim') : path.join(showdown, 'node_modules/@pkmn/sim'));
const {Battle, Teams, toID} = sim;
const PRNG = local ? require(path.join(showdown, 'dist/sim/prng')).PRNG : sim.PRNG;
const Dex = local ? require(path.join(showdown, 'dist/sim/dex')).Dex : sim.Dex;

// The pkmn engine reproduces Pokémon Showdown *as patched by its integration tests* (src/test/showdown.ts, `patch`): before
// Gen 3, event handlers and queued actions keep "host" order instead of being speed-sorted (a speed tie there makes Showdown
// shuffle with the battle's PRNG: draws the engine never makes), and a few no-op handlers get priorities. PATCH=0 replays
// on stock Showdown instead. FORMAT: gen1ou (default) or "test" (the tests' gen1customgame with their three mods).
const PATCH = process.env.PATCH !== '0';
const FORMAT = process.env.FORMAT === 'test' ? 'gen1customgame@@@Endless Battle Clause,Sleep Clause Mod,Freeze Clause Mod' : 'gen1ou';
if (PATCH) {
  const dex = Dex.mod('gen1');
  dex.data.Rulesets['sleepclausemod'].onSetStatusPriority = -999;
  dex.data.Rulesets['freezeclausemod'].onSetStatusPriority = -998;
  dex.conditions.get('disable').onDisableMovePriority = 7;
}
function insertChoice(choices, midTurn = false) {
  if (Array.isArray(choices)) { for (const c of choices) this.insertChoice(c); return; }
  const choice = choices;
  if (choice.pokemon) choice.pokemon.updateSpeed();
  const actions = this.resolveAction(choice, midTurn);
  let firstIndex = null, lastIndex = null;
  for (const [i, cur] of this.list.entries()) {
    const compared = this.battle.comparePriority(actions[0], cur);
    if (compared <= 0 && firstIndex === null) firstIndex = i;
    if (compared < 0) { lastIndex = i; break; }
  }
  if (firstIndex === null) { this.list.push(...actions); return; }
  if (lastIndex === null) lastIndex = this.list.length;
  const index = firstIndex === lastIndex ? firstIndex : (this.battle.gen > 3) ? this.battle.random(firstIndex, lastIndex + 1) : lastIndex;
  this.list.splice(index, 0, ...actions);
}
function eachEvent(eventid, effect, relayVar) {
  const actives = this.getAllActive();
  if (!effect && this.effect) effect = this.effect;
  if (this.gen >= 3) this.speedSort(actives, (a, b) => b.speed - a.speed);
  for (const pokemon of actives) this.runEvent(eventid, pokemon, null, effect, relayVar);
  if (eventid === 'Weather' && this.gen >= 7) this.eachEvent('Update');
}
function fieldEvent(eventid, targets) {
  const callbackName = `on${eventid}`;
  const getKey = eventid === 'Residual' ? 'duration' : undefined;
  let handlers = this.findFieldEventHandlers(this.field, `onField${eventid}`, getKey);
  for (const side of this.sides) {
    if (side.n < 2 || !side.allySide) handlers = handlers.concat(this.findSideEventHandlers(side, `onSide${eventid}`, getKey));
    for (const active of side.active) {
      if (!active) continue;
      if (eventid === 'SwitchIn') handlers = handlers.concat(this.findPokemonEventHandlers(active, `onAny${eventid}`));
      if (targets && !targets.includes(active)) continue;
      handlers = handlers.concat(this.findPokemonEventHandlers(active, callbackName, getKey));
      handlers = handlers.concat(this.findSideEventHandlers(side, callbackName, undefined, active));
      handlers = handlers.concat(this.findFieldEventHandlers(this.field, callbackName, undefined, active));
      handlers = handlers.concat(this.findBattleEventHandlers(callbackName, getKey, active));
    }
  }
  if (this.gen >= 3) this.speedSort(handlers);
  while (handlers.length) {
    const handler = handlers.shift();
    const effect = handler.effect;
    if (handler.effectHolder.fainted && !(handler.state && handler.state.isSlotCondition)) continue;
    if (eventid === 'Residual' && handler.end && handler.state && handler.state.duration) {
      handler.state.duration--;
      if (!handler.state.duration) {
        const endCallArgs = handler.endCallArgs || [handler.effectHolder, effect.id];
        handler.end.call(...endCallArgs);
        if (this.ended) return;
        continue;
      }
    }
    let handlerEventid = eventid;
    if (handler.effectHolder.sideConditions) handlerEventid = `Side${eventid}`;
    if (handler.effectHolder.pseudoWeather) handlerEventid = `Field${eventid}`;
    if (handler.callback) this.singleEvent(handlerEventid, effect, handler.state, handler.effectHolder, null, null, undefined, handler.callback);
    this.faintMessages();
    if (this.ended) return;
  }
}
function patchBattle(battle) {
  battle.trunc = battle.dex.trunc.bind(battle.dex);
  battle.queue.insertChoice = insertChoice.bind(battle.queue);
  battle.eachEvent = eachEvent.bind(battle);
  battle.fieldEvent = fieldEvent.bind(battle);
}
// helpers.Battle.init(seed) in Showdown mode seeds the battle with newSeed() drawn from a PSRNG seeded with `seed`: four
// 16-bit draws, the first one the most significant
function battleSeed(hex) {
  const p = new PRNG('gen5,' + hex);
  return 'gen5,' + [0, 1, 2, 3].map(() => p.random(0, 0x10000).toString(16).padStart(4, '0')).join('');
}

const STATUS = s => (s & 7) || s === 0x80 ? 'slp' : s & 8 ? 'psn' : s & 16 ? 'brn' : s & 32 ? 'frz' : s & 64 ? 'par' : '';
const psStatus = m => (m.status === 'tox' ? 'psn' : m.status === 'fnt' ? '' : m.status || '');
const packed = {};
// A set without an EVs line gets the maximum in every stat (as the engine's loader and the server's validator do; Teams.import
// alone would leave them at 0)
function team(name) {
  if (packed[name]) return packed[name];
  const text = fs.readFileSync(path.join(teamDir, name), 'utf8');
  const blocks = text.split(/\n\s*\n/).filter(b => b.trim());
  const sets = Teams.import(text);
  sets.forEach((set, i) => {
    if (!/^\s*EVs:/m.test(blocks[i] || '')) set.evs = {hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252};
  });
  return (packed[name] = Teams.pack(sets));
}

// the games: per log file, "REPLAY start" then its "REPLAY step" lines
const games = [];
for (const file of logs) {
  let g = null;
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    let m = line.match(/^REPLAY start game (\d+) seed ([0-9a-f]{16}) p1 (\S+) p2 (\S+)/);
    if (m) { g = {id: `${path.basename(file)}#${m[1]}`, seed: m[2], p1: m[3], p2: m[4], steps: []}; games.push(g); continue; }
    m = line.match(/^REPLAY step game (\d+) turn (\d+) c1 (.+?) c2 (.+?) result (\w+) state p1=(\w+)((?:,\w+:\d+:\d+)+) p2=(\w+)((?:,\w+:\d+:\d+)+)/);
    if (m && g) {
      const mons = s => s.slice(1).split(',').map(x => { const [sp, hp, st] = x.split(':'); return {id: toID(sp), hp: +hp, st: STATUS(+st)}; });
      g.steps.push({turn: +m[2], c: [m[3], m[4]], result: m[5], active: [toID(m[6]), toID(m[8])], mons: [mons(m[7]), mons(m[9])]});
    }
  }
}

function choice(raw) {
  if (raw === 'recharge') return 'move 1';
  if (raw.startsWith('switch ')) return 'switch ' + toID(raw.slice(7));
  return 'move ' + toID(raw);
}

const summary = {games: 0, inSync: 0, steps: 0, adapted: 0, divergences: {}};
const firsts = [];
for (const g of games) {
  summary.games++;
  const battle = new Battle({formatid: FORMAT, seed: battleSeed(g.seed), strictChoices: false});
  if (PATCH) patchBattle(battle);  // (before the players join: the battle starts then)
  battle.setPlayer('p1', {name: 'P1', team: team(g.p1)});
  battle.setPlayer('p2', {name: 'P2', team: team(g.p2)});
  let diverged = null, at = 0;
  for (let i = 0; i < g.steps.length && !diverged; i++) {
    const st = g.steps[i];
    at = i;
    // who must choose, read for both sides before either chooses (the last choice runs the update and the next requests)
    const askedOf = ['p1', 'p2'].map(sid => !!(battle[sid].requestState && battle[sid].requestState !== 'wait' && !battle[sid].isChoiceDone()));
    for (const [k, sid] of [[0, 'p1'], [1, 'p2']]) {
      const side = battle[sid], raw = st.c[k];
      const asked = askedOf[k];
      if (raw === 'pass') {
        if (asked) diverged = {kind: 'request', detail: `Showdown asks ${sid} for a ${side.requestState} choice; the engine had it pass`};
        continue;
      }
      if (!asked) { diverged = {kind: 'request', detail: `the engine has ${sid} choose ${raw}; Showdown asks nothing`}; break; }
      let input = choice(raw);
      if (!battle.choose(sid, input)) {
        // Showdown's one "Fight" option (Gen 1 states where every move does the same) stands for any move
        const req = side.activeRequest && side.activeRequest.active && side.activeRequest.active[0];
        if (req && req.moves.length === 1 && battle.choose(sid, 'move 1')) { summary.adapted++; input = 'move 1 (fight)'; continue; }
        diverged = {kind: 'choice', detail: `Showdown refuses ${sid} "${input}" (${raw}): ${(battle.log.slice(-1)[0] || '').slice(0, 160)}`};
      }
    }
    if (diverged) break;
    summary.steps++;
    // the state after the update
    const ended = battle.ended;
    const psResult = !ended ? 'None' : !battle.winner ? 'Tie' : battle.winner === 'P1' ? 'Win' : 'Lose';
    const engineResult = st.result;
    if (psResult !== engineResult) { diverged = {kind: 'result', detail: `result engine ${engineResult}, Showdown ${psResult}`}; break; }
    if (!ended && battle.turn !== st.turn) { diverged = {kind: 'turn', detail: `turn engine ${st.turn}, Showdown ${battle.turn}`}; break; }
    for (const [k, sid] of [[0, 'p1'], [1, 'p2']]) {
      const side = battle[sid];
      if (!ended && side.active[0] && side.active[0].species.id !== st.active[k] && !(side.active[0].fainted && st.mons[k].find(m => m.id === st.active[k])?.hp === 0)) {
        diverged = {kind: 'active', detail: `${sid} active engine ${st.active[k]}, Showdown ${side.active[0].species.id}`}; break;
      }
      for (const em of st.mons[k]) {
        const pm = side.pokemon.find(p => p.species.id === em.id);
        if (!pm) { diverged = {kind: 'team', detail: `${sid} has no ${em.id} in Showdown`}; break; }
        if (pm.hp !== em.hp) { diverged = {kind: 'hp', detail: `${sid} ${em.id} HP engine ${em.hp}, Showdown ${pm.hp}`}; break; }
        if (em.hp > 0 && psStatus(pm) !== em.st) { diverged = {kind: 'status', detail: `${sid} ${em.id} status engine '${em.st}', Showdown '${psStatus(pm)}'`}; break; }
      }
      if (diverged) break;
    }
  }
  if (diverged) {  // (wherever it was found: while choosing or after the update)
    const st = g.steps[at];
    // who is asleep or frozen now on each side, and who caused it (Sleep / Freeze Clause decisions depend on it)
    diverged.statuses = ['p1', 'p2'].map(sid => battle[sid].pokemon.filter(p => p.status === 'slp' || p.status === 'frz').map(p =>
      `${sid} ${p.species.id} ${p.status} (hp ${p.hp}, from ${p.statusState.source ? p.statusState.source.side.id + ' ' + p.statusState.source.species.id : 'unknown'})`)).flat();
    const prev = g.steps[at - 1];
    if (prev) diverged.engineBefore = prev.mons.map((side, k) => side.filter(m => m.st).map(m => `p${k + 1} ${m.id} ${m.st} hp ${m.hp}`)).flat();
    Object.assign(diverged, {step: at + 1, turn: st.turn, choices: st.c, log: battle.log.slice(-14)});
  }
  if (!diverged) { summary.inSync++; continue; }
  summary.divergences[diverged.kind] = (summary.divergences[diverged.kind] || 0) + 1;
  firsts.push({game: g.id, seed: g.seed, steps: g.steps.length, ...diverged});
}
console.log(`${summary.games} games replayed: ${summary.inSync} identical to the end, ${summary.games - summary.inSync} diverged; ` +
            `${summary.steps} updates matched; Fight substitutions ${summary.adapted}`);
console.log('first divergence by kind:', JSON.stringify(summary.divergences));
for (const f of firsts.slice(0, verbose ? firsts.length : 8)) {
  console.log(`\n${f.game} (seed ${f.seed}) at update ${f.step}/${f.steps}, turn ${f.turn}, choices ${f.choices.join(' / ')}: ${f.kind}: ${f.detail}`);
  if (verbose || firsts.indexOf(f) < 3) for (const l of f.log) console.log('    ' + l);
}
fs.writeFileSync(process.env.DIFF_OUT || '/dev/null', JSON.stringify({summary, firsts}, null, 1));
