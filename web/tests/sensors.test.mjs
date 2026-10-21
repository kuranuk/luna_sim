import assert from 'node:assert/strict';
import test from 'node:test';

import { eulerZYX, rotvec } from '../js/sim/frames.js';
import { NOISE_LABEL, renderLidar, scanPattern } from '../js/sim/lidar.js';
import { HeightfieldCaster } from '../js/sim/raycast.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { sunManual } from '../js/sim/sun.js';
import { World } from '../js/sim/world.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

function flatScenario() {
  const sc = structuredClone(SCENARIOS.mare_plain);
  sc.terrain.n = 256; sc.terrain.res = 0.25;
  sc.terrain.fractal.enabled = false; sc.terrain.craters.enabled = false; sc.terrain.rocks.enabled = false;
  return sc;
}

test('rotvec inverts small and large rotations', () => {
  for (const [r, p, y] of [[0.01, -0.02, 0.03], [0.5, 0.2, -1.0], [0, 0, 3.0]]) {
    const R = eulerZYX(r, p, y), v = rotvec(R), ang = Math.hypot(...v);
    close(Math.cos(ang), (R[0] + R[4] + R[8] - 1) / 2, 1e-12, 'angle');
  }
});

test('rover drives straight and turns on flat ground; IMU sees gravity and centripetal force', () => {
  const sc = flatScenario();
  sc.imu.gyroNoise = sc.imu.accelNoise = sc.imu.gyroRandomWalk = sc.imu.accelRandomWalk = 0;
  sc.imu.gyroBias = sc.imu.accelBias = 0; sc.imu.bandwidthHz = null;
  sc.lidars = [];
  const w = new World(sc);
  w.setCommand(0.2, 0);
  const out = w.advance(5.0);
  close(w.rover.x, 0.2 * 5 * (1 - 0.02), 1e-9, 'x after 5 s (2 % flat slip)');
  close(w.rover.contact.t[2], 0.5, 1e-9, 'body height');
  assert.equal(out.imu.length, 500);
  close(out.imu[300].accel[2], 1.6243, 1e-9, 'gravity');
  w.setCommand(0.2, 0.1);
  const turn = w.advance(3.0).imu.slice(100);
  const ay = turn.reduce((s, x) => s + x.accel[1], 0) / turn.length;
  close(ay, 0.2 * 0.98 * 0.1, 2e-4, 'centripetal (left)');
  const wz = turn.reduce((s, x) => s + x.gyro[2], 0) / turn.length;
  close(wz, 0.1, 1e-5, 'yaw rate');
});

test('IMU noise statistics follow the configured densities', () => {
  const sc = flatScenario();
  sc.lidars = []; sc.imu.bandwidthHz = null;
  const w = new World(sc);
  const s = w.advance(60).imu;
  const sd = (f) => { const v = s.map(f), m = v.reduce((a, b) => a + b) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };
  close(sd((x) => x.gyro[0] - x.gyroTrue[0] - x.gyroBias[0]), sc.imu.gyroNoise * 10, sc.imu.gyroNoise * 0.5, 'gyro white');
  close(sd((x) => x.accel[2] - x.accelTrue[2] - x.accelBias[2]), sc.imu.accelNoise * 10, sc.imu.accelNoise * 0.5, 'accel white');
});

test('LiDAR on a flat plane: ranges, noise and detection', () => {
  const sc = flatScenario();
  const w = new World(sc);
  const pose = { R: eulerZYX(0, 30 * Math.PI / 180, 0), t: [0, 0, 1.0] };
  const spin = sc.lidars[1], spinPat = scanPattern(spin);
  assert.equal(spinPat.n, 32 * 1800);
  const scan = renderLidar(spin, spinPat, pose, w.terrain, w.caster, null, 0, 1);
  assert.ok(scan.count > 1000);
  let maxErr = 0;
  for (let k = 0; k < scan.count; k++) maxErr = Math.max(maxErr, Math.abs(scan.world[3 * k + 2]));
  assert.ok(maxErr < 0.15, `ground points near z = 0 (max ${maxErr})`);
  const flash = sc.lidars[0], pat = scanPattern(flash);
  const dark = renderLidar(flash, pat, pose, w.terrain, w.caster, null, 0, 1);
  const lit = renderLidar(flash, pat, pose, w.terrain, w.caster, sunManual(0, 30), 0, 1);
  const noise = (s) => Array.from(s.labels).filter((l) => l === NOISE_LABEL).length;
  assert.ok(dark.count > 0.9 * pat.n, `dark: most pixels detected (${dark.count}/${pat.n})`);
  assert.ok(noise(dark) < 0.01 * pat.n);
  assert.ok(lit.count <= dark.count, 'sunlight background lowers detection');
  let e = 0, c = 0;
  for (let k = 0; k < dark.count; k++) if (dark.labels[k] !== NOISE_LABEL) { e += Math.abs(dark.world[3 * k + 2]); c++; }
  assert.ok(e / c < 0.05, `mean |z| ${e / c}`);
});

test('world: LiDAR frame rate and sun shading', () => {
  const sc = flatScenario();
  const w = new World(sc);
  const out = w.advance(2.0);
  const n = (name) => out.lidar.filter((f) => f.name === name).length;
  assert.equal(n('spad_flash'), 5); // t = 0, 0.5, 1.0, 1.5, 2.0 at 2 Hz (end inclusive)
  assert.equal(n('spin32'), 21);
  const { lit, m } = w.sunlit(4);
  assert.equal(lit.length, m * m);
  assert.ok(lit.every((v) => v === 1), 'flat terrain fully lit at 30° elevation');
  const caster = new HeightfieldCaster(w.terrain);
  assert.equal(caster.occluded(0, 0, 0.01, 0, 0, 1), false);
});
