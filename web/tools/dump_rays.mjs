// Dump a browser-simulator terrain and the SPAD / spinning LiDAR rays with their hit distances,
// for the cross-check against the Python ray caster (tests/test_web_sim.py).
//   node web/tools/dump_rays.mjs OUT_DIR [n]
// Writes z.f64 (n x n), rays.f64 (N x 7: origin, unit direction, t or Infinity) and meta.json.

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { matVec } from '../js/sim/frames.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { World } from '../js/sim/world.js';

const [out, nArg] = process.argv.slice(2);
const sc = structuredClone(SCENARIOS.south_pole_lowsun);
sc.terrain.n = +(nArg || 256); sc.cameras = [];
const w = new World(sc), hm = w.terrain, rows = [];
for (let k = 0; k < 3; k++) {
  w.rover.reset(-10 + 9 * k, 6 - 7 * k, 0.3 + 2.2 * k);
  for (const l of w.lidars) {
    const pose = w.sensorPose(l.cfg.mount), o = pose.t;
    for (let b = 0; b < l.pattern.n; b++) {
      const d = matVec(pose.R, [l.pattern.dirs[3 * b], l.pattern.dirs[3 * b + 1], l.pattern.dirs[3 * b + 2]]);
      rows.push(o[0], o[1], o[2], d[0], d[1], d[2], w.caster.terrain.trace(o[0], o[1], o[2], d[0], d[1], d[2], 0, l.cfg.rangeMax * 1.05));
    }
  }
}
mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'z.f64'), Buffer.from(Float64Array.from(hm.z).buffer));
writeFileSync(join(out, 'rays.f64'), Buffer.from(Float64Array.from(rows).buffer));
writeFileSync(join(out, 'meta.json'), JSON.stringify({ n: hm.n, res: hm.res, x0: hm.x0, y0: hm.y0, tMax: 105 }));
