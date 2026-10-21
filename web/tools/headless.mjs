// Headless closed-loop run (Node >= 22, no browser): the same World and lockstep protocol as the
// browser simulator's 航法誘導 mode, for batch evaluation of a Python GNC (`lunarsim gnc`).
//
//   node web/tools/headless.mjs --url ws://127.0.0.1:8765/ --seconds 60 --waypoints "5,0;5,5"
//   options: --preset mare_plain --size "128 m / 0.25 m" --seed 1 --period 0.1 --no-lidar
// Prints one JSON line with the final truth / estimate.

import { parseArgs } from 'node:util';

import { helloMessage, observationMessage, truthPose } from '../js/gnc/protocol.js';
import { SCENARIOS, SIZES } from '../js/sim/scenarios.js';
import { World } from '../js/sim/world.js';

const { values: o } = parseArgs({ options: {
  url: { type: 'string', default: 'ws://127.0.0.1:8765/' }, preset: { type: 'string', default: 'mare_plain' },
  size: { type: 'string', default: '128 m / 0.25 m' }, n: { type: 'string' }, res: { type: 'string' },
  seed: { type: 'string' }, seconds: { type: 'string', default: '30' }, period: { type: 'string', default: '0.1' },
  waypoints: { type: 'string', default: '' }, 'no-lidar': { type: 'boolean', default: false },
} });

const sc = structuredClone(SCENARIOS[o.preset]);
if (!sc) throw new Error(`unknown preset ${o.preset}`);
Object.assign(sc.terrain, SIZES[o.size] || {});
if (o.n) sc.terrain.n = +o.n;
if (o.res) sc.terrain.res = +o.res;
if (o.seed) sc.seed = +o.seed;
if (o['no-lidar']) sc.lidars = [];
const waypoints = o.waypoints ? o.waypoints.split(';').map((p) => p.split(',').map(Number)) : [];
const period = +o.period, steps = Math.round(+o.seconds / period);

const world = new World(sc);
const ws = new WebSocket(o.url);
ws.binaryType = 'arraybuffer';
const inbox = [];
let wake = null;
ws.onmessage = (ev) => { inbox.push(JSON.parse(ev.data)); if (wake) { wake(); wake = null; } };
const closed = new Promise((resolve) => { ws.onclose = resolve; });
const next = async () => {
  while (!inbox.length) await Promise.race([new Promise((r) => { wake = r; }), closed.then(() => { throw new Error('connection closed'); })]);
  const m = inbox.shift();
  if (m.type === 'error') throw new Error(`GNC error: ${m.message}`);
  return m;
};
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = () => reject(new Error(`cannot connect to ${o.url}`)); });

ws.send(JSON.stringify(helloMessage(world, period, waypoints)));
const ready = await next();
let est = null, status = '';
const t0 = performance.now();
for (let seq = 1; seq <= steps; seq++) {
  const out = world.advance(period, { scans: 'all' });
  ws.send(observationMessage(world, out, seq, period, waypoints));
  const m = await next();
  if (m.type !== 'cmd' || m.seq !== seq) throw new Error(`unexpected reply ${JSON.stringify(m)}`);
  world.rover.setCommand(m.v, m.omega);
  est = m.estimate || null; status = m.status || '';
}
ws.close();
const truth = truthPose(world);
console.log(JSON.stringify({ gnc: ready.name, time: world.time, steps, wallSeconds: (performance.now() - t0) / 1000,
  truth, estimate: est, status, positionError: est ? Math.hypot(est.x - truth.x, est.y - truth.y) : null }));
