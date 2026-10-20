import assert from 'node:assert/strict';
import test from 'node:test';

import { decode, encode, helloMessage, observationMessage } from '../js/gnc/protocol.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { World } from '../js/sim/world.js';

test('GNC message encode/decode round trip (alignment, dtypes, shapes)', () => {
  const blobs = [
    { name: 'a', data: Float64Array.from([1.5, -2, 3]), shape: [3] },
    { name: 'b', data: Uint8Array.from([1, 2, 3, 4, 5]), shape: [5] },
    { name: 'c', data: Float32Array.from([1, 2, 3, 4, 5, 6]), shape: [2, 3] },
    { name: 'd', data: new Uint16Array(0), shape: [0] },
  ];
  const { header, blobs: got } = decode(encode({ type: 'x', seq: 7, nested: { k: [1, 2] } }, blobs));
  assert.deepEqual(header, { type: 'x', seq: 7, nested: { k: [1, 2] } });
  for (const b of blobs) { assert.equal(got[b.name].constructor, b.data.constructor); assert.deepEqual([...got[b.name]], [...b.data]); }
});

test('observation message carries IMU, LiDAR and 10 Hz truth for one control period', () => {
  const sc = structuredClone(SCENARIOS.mare_plain);
  sc.terrain.n = 256; sc.terrain.res = 0.25;
  const w = new World(sc);
  const hello = helloMessage(w, 0.5, [[1, 2]]);
  assert.equal(hello.info.controlPeriod, 0.5);
  assert.equal(hello.info.scenario.imu.rateHz, 100);
  w.setCommand(0.1, 0.05);
  const out = w.advance(0.5, { scans: 'all' });
  const { header, blobs } = decode(observationMessage(w, out, 3, 0.5, [[1, 2]]));
  assert.equal(header.seq, 3); assert.equal(header.dt, 0.5);
  assert.ok(Math.abs(header.t - 0.5) < 1e-9);
  assert.equal(blobs['imu/t'].length, 50);
  assert.equal(blobs['imu/gyro'].length, 150);
  assert.equal(blobs['truth/poses'].length, 8 * out.poses.length);
  const names = header.lidar.map((l) => l.name);
  assert.ok(names.includes('spad_flash') && names.includes('spin32'), names.join());
  for (const l of header.lidar) {
    assert.equal(blobs[`lidar/${l.name}/${l.frame}/points`].length, 4 * l.count);
    assert.equal(blobs[`lidar/${l.name}/${l.frame}/labels`].length, l.count);
  }
  assert.equal(header.wheel.v, 0.1);
});
