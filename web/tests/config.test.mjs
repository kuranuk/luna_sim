import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import test from 'node:test';

import { CONFIG_FORMAT, deepMerge, resolveConfig, scenarioToConfig } from '../js/sim/config.js';
import { compose, matVec, mountPose } from '../js/sim/frames.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { World } from '../js/sim/world.js';

test('presets carry front, side and rear cameras looking the right way', () => {
  const sc = SCENARIOS.south_pole_lowsun;
  assert.deepEqual(sc.cameras.map((c) => c.name), ['navcam', 'side_left', 'side_right', 'rear']);
  const body = { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] };
  const axis = (name) => matVec(compose(body, mountPose(sc.cameras.find((c) => c.name === name).mount)).R, [1, 0, 0]);
  assert.ok(axis('navcam')[0] > 0.9); // forward (FLU +x)
  assert.ok(axis('side_left')[1] > 0.9); // left (+y)
  assert.ok(axis('side_right')[1] < -0.9); // right (-y)
  assert.ok(axis('rear')[0] < -0.9); // backward
  for (const n of ['side_left', 'side_right', 'rear']) assert.ok(axis(n)[2] < 0, `${n} looks slightly down`);
});

test('config: merge onto the base preset, sensors by name, templates for new sensors', () => {
  const sc = resolveConfig({
    base: 'mare_plain', seed: 5, terrain: { res: 0.25 }, sun: { azDeg: 200 },
    cameras: [{ name: 'navcam', hfovDeg: 100 }, { name: 'rear', enabled: false }, { name: 'mast', mount: { z: 1.7 } }],
    lidars: [{ name: 'spad_flash', vfovDeg: 30 }, { name: 'tilted', kind: 'spinning', channels: 16 }],
    lights: [],
  });
  assert.equal(sc.seed, 5); assert.equal(sc.terrain.res, 0.25); assert.equal(sc.terrain.n, 1024);
  assert.equal(sc.sun.azDeg, 200); assert.equal(sc.sun.mode, 'manual');
  assert.deepEqual(sc.cameras.map((c) => c.name), ['navcam', 'rear', 'mast']); // side cameras removed
  assert.equal(sc.cameras[0].hfovDeg, 100); assert.equal(sc.cameras[0].mount.pitchDeg, 12); // rest kept
  assert.equal(sc.cameras[2].mount.z, 1.7); assert.equal(sc.cameras[2].mount.x, 0.5); // from the template
  assert.equal(sc.lidars[1].kind, 'spinning'); assert.equal(sc.lidars[1].channels, 16); assert.equal(sc.lidars[1].hResDeg, 0.2);
  assert.equal(sc.lights.length, 0);
  const w = new World(sc);
  assert.equal(w.cameras.find((c) => c.cfg.name === 'rear').enabled, false);
  assert.equal(w.lidars[0].pattern.n, 160 * 120);
  assert.ok(Math.abs(w.cameras[0].K.fx - 320 / Math.tan(50 * Math.PI / 180)) < 1e-9);
  // omitted arrays keep the base sensors; the base itself is not modified
  assert.equal(resolveConfig({}).cameras.length, 4);
  assert.equal(SCENARIOS.mare_plain.cameras[0].hfovDeg, 70);
});

test('config: every problem is reported', () => {
  assert.throws(() => resolveConfig({ base: 'nope' }), /base "nope"/);
  try {
    resolveConfig({ format: 'x/9', terrain: { n: 1000, res: -1 }, sun: { mode: 'astronomical', utc: 'later' },
      cameras: [{ name: 'navcam', hfovDeg: 500, width: 10.5 }, { hfovDeg: 3 }], lidars: [{ name: 'l', kind: 'rosette' }] });
    assert.fail('no error');
  } catch (e) {
    for (const s of ['format', 'cameras[1]', 'rosette']) assert.match(e.message, new RegExp(s.replace(/[[\]]/g, '\\$&')));
  }
  try {
    resolveConfig({ terrain: { n: 1000, res: -1 }, sun: { mode: 'astronomical', utc: 'later' }, cameras: [{ name: 'navcam', hfovDeg: 500, width: 10.5 }] });
    assert.fail('no error');
  } catch (e) {
    for (const s of ['terrain.n', 'terrain.res', 'sun.utc', 'hfovDeg', 'width']) assert.match(e.message, new RegExp(s.replace('.', '\\.')));
  }
});

test('config: export -> import round trip and the shipped examples', () => {
  const sc = resolveConfig({ base: 'crater_field', cameras: [{ name: 'navcam', hfovDeg: 33 }] });
  const txt = scenarioToConfig(sc, 'crater_field'), back = resolveConfig(JSON.parse(txt));
  assert.equal(JSON.parse(txt).format, CONFIG_FORMAT);
  assert.deepEqual(back, sc);
  const dir = new URL('../configs/', import.meta.url);
  const files = readdirSync(dir).filter((f) => f.endsWith('.json'));
  assert.ok(files.length >= 2);
  for (const f of files) resolveConfig(JSON.parse(readFileSync(new URL(f, dir), 'utf8')));
  assert.deepEqual(deepMerge({ a: { b: 1, c: [1, 2] } }, { a: { c: [3] } }), { a: { b: 1, c: [3] } });
});
