import assert from 'node:assert/strict';
import test from 'node:test';

import { decode, observationMessage } from '../js/gnc/protocol.js';
import { hapke } from '../js/sim/brdf.js';
import { autoExposure, electronsPerRadiance, expose, profileSolidAngle, spotIntensity } from '../js/sim/camera.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { World } from '../js/sim/world.js';

const SENSOR = SCENARIOS.south_pole_lowsun.cameras[0].sensor;

test('Hapke reflectance equals the Python implementation (brdf.reflectance, default HapkeConfig)', () => {
  const cases = [ // [n, l, v, r] from lunarsim.core.illumination.brdf.reflectance
    [[0, 0, 1], [0.5000110003630134, 0, 0.8660190526287391], [0, 0, 1], 0.031229329205313003],
    [[0, 0, 1], [0.9900094051340234, 0, 0.14100133951908816], [0.300006600217808, 0.4000088002904107, 0.8660190526287391], 0.004809643035168846],
    [[0.09999875002343701, 0, 0.9949875627331982], [-0.19979033017224906, 0.8990564857751208, 0.38959114383588567],
      [0.7026752586689714, -0.20076435961970612, 0.6825988227070009], 0.009037002481579898],
    [[0, 0, 1], [0.6, 0, 0.8], [0.6, 0, 0.8], 0.07219797900893336],
  ];
  for (const [n, l, v, r] of cases) assert.ok(Math.abs(hapke(n, l, v, SCENARIOS.mare_plain.reflectance) / r - 1) < 1e-12);
  assert.equal(hapke([0, 0, 1], [0, 0.1, -1], [0, 0, 1]), 0); // sun below the surface
});

test('sensor and spotlight radiometry equal the Python values', () => {
  assert.ok(Math.abs(electronsPerRadiance(SENSOR) / 2018284.1494467831 - 1) < 1e-14);
  assert.ok(Math.abs(profileSolidAngle(15 * Math.PI / 180, 30 * Math.PI / 180) / 0.4882194659681351 - 1) < 1e-12);
  const l = spotIntensity(SCENARIOS.south_pole_lowsun.lights[0]);
  assert.ok(Math.abs(l.i0 - 3000 / 300 * 0.8 / 0.4882194659681351) < 1e-9);
});

test('expose: auto exposure puts the percentile at the target; shot + read noise statistics', () => {
  const n = 200000, rad = new Float32Array(n).fill(2.0);
  const t = autoExposure(rad, SENSOR);
  const mean = 2.0 * electronsPerRadiance(SENSOR) * t + SENSOR.darkCurrent * t;
  assert.ok(Math.abs(mean / (SENSOR.aeTarget * SENSOR.fullWell) - 1) < 1e-6);
  const img = expose(rad, SENSOR, 'navcam', 0, 1), g = SENSOR.fullWell / 4095;
  let s = 0, s2 = 0;
  for (const d of img.dn) { s += d; s2 += d * d; }
  const m = s / n, v = s2 / n - m * m;
  assert.ok(Math.abs(m * g / mean - 1) < 2e-3, `mean ${m * g} vs ${mean}`);
  const expVar = (mean + SENSOR.readNoise ** 2) / g ** 2 + 1 / 12; // + quantisation
  assert.ok(Math.abs(v / expVar - 1) < 0.05, `var ${v} vs ${expVar}`);
  assert.deepEqual(expose(rad, SENSOR, 'navcam', 0, 1).dn, img.dn); // seeded
  const bright = expose(new Float32Array(10).fill(1e6), { ...SENSOR, autoExposure: false }, 'navcam', 0, 1);
  assert.equal(bright.saturated, 10);
  assert.ok(bright.dn.every((d) => d === 4095));
});

test('World schedules camera frames with an injected renderer and sends them to Python', () => {
  const sc = structuredClone(SCENARIOS.mare_plain);
  sc.terrain.n = 128; sc.terrain.res = 0.25; sc.lidars = [];
  sc.cameras[0].width = 32; sc.cameras[0].height = 24;
  const w = new World(sc);
  assert.equal(w.advance(1.0).camera.length, 0); // no renderer (Node): skipped
  w.reset();
  const calls = [];
  w.cameraRenderer = { render: (world, cam, pose) => { calls.push(pose.t); return new Float32Array(cam.K.width * cam.K.height).fill(1); } };
  const out = w.advance(1.0, { scans: 'all' });
  assert.equal(out.camera.length, 3); // t = 0.01 (due at 0), 0.5, 1.0 at 2 Hz
  const f = out.camera[1];
  assert.equal(f.image.dn.length, 32 * 24);
  assert.ok(Math.abs(f.image.K.fx - 0.5 * 32 / Math.tan(35 * Math.PI / 180)) < 1e-12);
  const { header, blobs } = decode(observationMessage(w, out, 1, 1.0, []));
  assert.equal(header.cameras.length, 3);
  const c = header.cameras[2];
  assert.deepEqual([...blobs[`camera/${c.name}/${c.frame}/image`]], [...out.camera[2].image.dn]);
});
