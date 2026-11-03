#!/usr/bin/env node
'use strict';
/*
 * Ground truth from the real Pokemon Showdown simulator.
 *
 *   node showdown_oracle.js SHOWDOWN_DIR TEAMS_DIR OUT_DIR N_BATTLES [FIRST_SEED] [SWITCH_PROB]
 *
 * Plays seeded random Gen 1 OU battles. For every battle it writes OUT_DIR/<id>.log (the public replay log, the
 * input of metamon.recurrent.parse) and OUT_DIR/<id>.json with, for every decision of both players, the exact
 * request Showdown sent, the hidden engine state at that moment (statuses, volatiles, locks, PP) and the choice made.
 */
const fs = require('fs');
const path = require('path');

const [showdownDir, teamsDir, outDir, nBattles, firstSeed = '1', switchProb = '0.25'] = process.argv.slice(2);
const sim = require(path.join(showdownDir, 'dist/sim'));
const {BattlePlayer} = require(path.join(showdownDir, 'dist/sim/battle-stream'));
const {BattleStream, getPlayerStreams, Teams} = sim;

fs.mkdirSync(outDir, {recursive: true});
const teamFiles = fs.readdirSync(teamsDir).filter(f => f.endsWith('_team')).sort();

function mulberry32(a) {
	return function () {
		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul(a ^ a >>> 15, 1 | a);
		t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
		return ((t ^ t >>> 14) >>> 0) / 4294967296;
	};
}

function pokemonState(p) {
	return {
		species: p.species.id, name: p.name, hp: p.hp, maxhp: p.maxhp, status: p.status, fainted: p.fainted,
		active: p.isActive, position: p.position,
		statusState: p.status ? {...p.statusState, source: undefined, sourceEffect: undefined} : null,
		volatiles: Object.keys(p.volatiles),
		boosts: {...p.boosts},
		moves: p.moveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled})),
		trapped: p.trapped, maybeTrapped: p.maybeTrapped, maybeLocked: p.maybeLocked,
	};
}

/*
 * What Counter would do if `user` used it on `target` right now, from the engine's own damageCallback (its hint and
 * -fail messages are muted so that the battle log is untouched). `lastDamage` and the move memory it reads are
 * returned as well for debugging.
 */
function counterTruth(battle, userSide, targetSide) {
	const user = userSide.active[0], target = targetSide.active[0];
	if (!user || !target || user.fainted || target.fainted) return null;
	const counter = battle.dex.moves.get('counter');
	const add = battle.add, hint = battle.hint;
	battle.add = () => {};
	battle.hint = () => {};
	let result;
	try {
		result = counter.damageCallback.call(battle, user, target);
	} finally {
		battle.add = add;
		battle.hint = hint;
	}
	return {
		damage: result === false ? 0 : result, succeeds: result !== false, lastDamage: battle.lastDamage,
		targetSide: {
			lastMove: targetSide.lastMove ? targetSide.lastMove.id : null,
			lastEnemyMove: targetSide.lastEnemyMove ? targetSide.lastEnemyMove.id : null,
			lastSelectedMove: targetSide.lastSelectedMove, lastEnemySelectedMove: targetSide.lastEnemySelectedMove,
		},
		targetMaxHp: target.maxhp, userMaxHp: user.maxhp,
	};
}

function sideState(side) {
	return {
		pokemon: side.pokemon.map(pokemonState),
		conditions: Object.keys(side.sideConditions),
	};
}

class RandomRecorder extends BattlePlayer {
	constructor(stream, id, rng, battle, decisions, switchP) {
		super(stream);
		this.id = id; this.rng = rng; this.battleRef = battle; this.decisions = decisions; this.switchP = switchP;
	}
	receiveRequest(request) {
		if (request.wait) return;
		const battle = this.battleRef();
		const me = battle.sides[this.id === 'p1' ? 0 : 1];
		const opp = battle.sides[this.id === 'p1' ? 1 : 0];
		let choice;
		const reserves = request.side.pokemon.map((p, i) => ({p, i})).filter(({p}) => !p.active && !p.condition.endsWith(' fnt'));
		if (request.forceSwitch) {
			const pick = reserves[Math.floor(this.rng() * reserves.length)];
			choice = `switch ${pick.i + 1}`;
		} else {
			const act = request.active[0];
			const moves = act.moves.map((m, i) => ({m, i})).filter(({m}) => !m.disabled);
			const canSwitch = !act.trapped && reserves.length > 0;
			if (canSwitch && (moves.length === 0 || this.rng() < this.switchP)) {
				const pick = reserves[Math.floor(this.rng() * reserves.length)];
				choice = `switch ${pick.i + 1}`;
			} else {
				const pick = moves[Math.floor(this.rng() * moves.length)];
				choice = `move ${pick.i + 1}`;
			}
		}
		this.decisions.push({
			side: this.id, turn: battle.turn, request, choice,
			me: sideState(me), opp: sideState(opp),
			counterMe: counterTruth(battle, me, opp), counterOpp: counterTruth(battle, opp, me),
		});
		this.choose(choice);
	}
	receiveError(error) {
		throw error;
	}
}

async function playBattle(index) {
	const seed = Number(firstSeed) + index;
	const rng = mulberry32(seed * 7919 + 13);
	const pick = () => teamFiles[Math.floor(rng() * teamFiles.length)];
	// LOCK_MODE=1 swaps random moves for ones that lock, trap, disable or put to sleep, so that every request shape occurs
	const lockMoves = ['Fly', 'Dig', 'Solar Beam', 'Razor Wind', 'Skull Bash', 'Sky Attack', 'Thrash', 'Petal Dance', 'Rage',
		'Bide', 'Wrap', 'Bind', 'Fire Spin', 'Clamp', 'Hyper Beam', 'Disable', 'Mimic', 'Counter', 'Mist', 'Hypnosis',
		'Sing', 'Spore', 'Sleep Powder', 'Lovely Kiss', 'Confuse Ray', 'Supersonic', 'Blizzard', 'Ice Beam', 'Rest', 'Substitute'];
	const pack = f => {
		const sets = Teams.import(fs.readFileSync(path.join(teamsDir, f), 'utf8'));
		// Gen 1 has no abilities, but some pool files list real ones (Intimidate...): the real server has "No Ability"
		for (const set of sets) set.ability = 'No Ability';
		if (process.env.FUZZ_MODE) {
			// every Pokemon is a random Gen 1 species with four random moves from the whole Gen 1 move list
			const gen1 = sim.Dex.mod('gen1');
			const species = gen1.species.all().filter(s => s.num >= 1 && s.num <= 151 && !s.forme);
			const moves = gen1.moves.all().filter(m => !m.isNonstandard && m.num >= 1 && m.num <= 165 && m.id !== 'struggle');
			for (const set of sets) {
				set.species = species[Math.floor(rng() * species.length)].name;
				set.name = '';
				const chosen = new Set();
				while (chosen.size < 4) chosen.add(moves[Math.floor(rng() * moves.length)].name);
				set.moves = [...chosen];
			}
			const names = new Set();
			for (const set of sets) {  // species clause
				while (names.has(set.species)) set.species = species[Math.floor(rng() * species.length)].name;
				names.add(set.species);
			}
		}
		if (process.env.COUNTER_MODE) {
			// most Pokemon carry Counter, so that its odd rules come up often
			for (const set of sets) {
				if (rng() < 0.7) {
					const j = Math.floor(rng() * set.moves.length);
					if (!set.moves.includes('Counter')) set.moves[j] = 'Counter';
				}
			}
		}
		if (process.env.SUB_MODE) {
			// Substitute everywhere, with the moves it interacts with: status that it blocks, trapping, Counter, Leech Seed...
			const subMoves = ['Substitute', 'Substitute', 'Confuse Ray', 'Supersonic', 'Toxic', 'Poison Powder', 'Stun Spore', 'Thunder Wave',
				'Leech Seed', 'Wrap', 'Bind', 'Fire Spin', 'Clamp', 'Counter', 'Swords Dance', 'Growl', 'Screech', 'Reflect', 'Light Screen',
				'Mist', 'Focus Energy', 'Hypnosis', 'Seismic Toss', 'Night Shade', 'Super Fang', 'Double-Edge', 'Body Slam', 'Hyper Beam',
				'Drain Punch', 'Mega Drain', 'Absorb', 'Dream Eater', 'Rest', 'Recover', 'Haze', 'Disable', 'Explosion', 'Self-Destruct', 'Thrash'];
			for (const set of sets) {
				const unique = ['Substitute'];
				while (unique.length < 4) {
					const m = rng() < 0.75 ? subMoves[Math.floor(rng() * subMoves.length)] : set.moves[Math.floor(rng() * set.moves.length)];
					if (!unique.includes(m)) unique.push(m);
				}
				set.moves = unique;
			}
		}
		if (process.env.LOCK_MODE) {
			for (const set of sets) {
				for (let j = 0; j < set.moves.length; j++) {
					if (rng() < 0.45) set.moves[j] = lockMoves[Math.floor(rng() * lockMoves.length)];
				}
				// keep four distinct moves: a team with fewer would make Struggle impossible to tell from a short list
				const unique = [...new Set(set.moves)];
				while (unique.length < 4) {
					const m = lockMoves[Math.floor(rng() * lockMoves.length)];
					if (!unique.includes(m)) unique.push(m);
				}
				set.moves = unique;
			}
		}
		// The real server validates teams: in a format with no EV limit (every Gen 1 and 2 format) a set without EVs gets 252
		// in every stat, while the bare simulator would give it none. Reproduce that, and pass the sets themselves.
		const team = Teams.unpack(Teams.pack(sets));
		for (const set of team) {
			if (!set.evs) set.evs = {hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252};
		}
		return team;
	};
	const t1 = pick(), t2 = pick();
	const stream = new BattleStream();
	const streams = getPlayerStreams(stream);
	const decisions = [];
	const battleRef = () => stream.battle;
	const p1 = new RandomRecorder(streams.p1, 'p1', rng, battleRef, decisions, Number(switchProb));
	const p2 = new RandomRecorder(streams.p2, 'p2', rng, battleRef, decisions, Number(switchProb));
	void p1.start(); void p2.start();
	void streams.omniscient.write(`>start ${JSON.stringify({formatid: 'gen1ou', seed: [seed, seed + 1, seed + 2, seed + 3]})}\n` +
		`>player p1 ${JSON.stringify({name: 'Alpha', team: pack(t1)})}\n` +
		`>player p2 ${JSON.stringify({name: 'Beta', team: pack(t2)})}`);
	let log = '';
	for await (const chunk of streams.spectator) log += chunk + '\n';
	const id = `oracle-${seed}`;
	const header = `|t:|${1700000000 + seed}\n`;
	fs.writeFileSync(path.join(outDir, `${id}.log`), header + log);
	fs.writeFileSync(path.join(outDir, `${id}.json`), JSON.stringify({id, seed, teams: [t1, t2], decisions}));
	return decisions.length;
}

(async () => {
	let total = 0;
	for (let i = 0; i < Number(nBattles); i++) {
		try {
			total += await playBattle(i);
		} catch (e) {
			console.error(`battle ${i} failed: ${e.message}`);
		}
	}
	console.log(`wrote ${nBattles} battles, ${total} decisions to ${outDir}`);
})();
