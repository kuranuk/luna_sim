// Binary message format shared with the Python GNC side (lunarsim/gnc/protocol.py):
//   "LSG1" | uint32 header length | header JSON (UTF-8) | zero pad to 8 | blobs (each 8-aligned)
// header.blobs = [{ name, dtype, shape, offset }] with offset from the start of the blob section.

export const PROTOCOL_VERSION = 1;
const MAGIC = [0x4c, 0x53, 0x47, 0x31];
const DTYPES = { f4: Float32Array, f8: Float64Array, u1: Uint8Array, u2: Uint16Array, i4: Int32Array, u4: Uint32Array };
const align8 = (n) => (n + 7) & ~7;

function dtypeOf(a) {
  for (const [k, C] of Object.entries(DTYPES)) if (a instanceof C) return k;
  throw new Error(`unsupported array type ${a && a.constructor && a.constructor.name}`);
}

/** blobs: [{ name, data: TypedArray, shape }] -> ArrayBuffer */
export function encode(header, blobs) {
  const descs = [];
  let off = 0;
  for (const b of blobs) {
    descs.push({ name: b.name, dtype: dtypeOf(b.data), shape: b.shape, offset: off });
    off += align8(b.data.byteLength);
  }
  const head = new TextEncoder().encode(JSON.stringify({ ...header, blobs: descs }));
  const base = align8(8 + head.length), buf = new ArrayBuffer(base + off), u8 = new Uint8Array(buf);
  u8.set(MAGIC, 0);
  new DataView(buf).setUint32(4, head.length, true);
  u8.set(head, 8);
  blobs.forEach((b, k) => u8.set(new Uint8Array(b.data.buffer, b.data.byteOffset, b.data.byteLength), base + descs[k].offset));
  return buf;
}

/** ArrayBuffer -> { header, blobs: { name: TypedArray } } */
export function decode(buf) {
  const u8 = new Uint8Array(buf);
  if (u8.length < 8 || MAGIC.some((m, k) => u8[k] !== m)) throw new Error('bad magic');
  const hlen = new DataView(buf).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hlen)));
  const base = align8(8 + hlen), blobs = {};
  for (const d of header.blobs || []) {
    const C = DTYPES[d.dtype], n = d.shape.reduce((a, b) => a * b, 1);
    blobs[d.name] = new C(buf.slice(base + d.offset, base + d.offset + n * C.BYTES_PER_ELEMENT));
  }
  delete header.blobs;
  return { header, blobs };
}

/** Observation message for one control period of World.advance output (see lunarsim/gnc/session.py). */
export function observationMessage(world, out, seq, dt, waypoints) {
  const imu = out.imu, n = imu.length, t = new Float64Array(n), gyro = new Float64Array(3 * n), accel = new Float64Array(3 * n);
  imu.forEach((s, k) => { t[k] = s.t; gyro.set(s.gyro, 3 * k); accel.set(s.accel, 3 * k); });
  const blobs = [
    { name: 'imu/t', data: t, shape: [n] },
    { name: 'imu/gyro', data: gyro, shape: [n, 3] },
    { name: 'imu/accel', data: accel, shape: [n, 3] },
    { name: 'truth/poses', data: Float64Array.from(out.poses.flat()), shape: [out.poses.length, 8] },
  ];
  const lidar = [];
  for (const f of out.lidar) {
    const p = `lidar/${f.name}/${f.frame}/`, c = f.scan.count;
    const lc = (world.lidars.find((l) => l.cfg.name === f.name) || {}).cfg || {};
    const pattern = lc.kind === 'flash' ? { kind: 'flash', projection: lc.projection || 'equiangular', width: lc.width, height: lc.height, hfovDeg: lc.hfovDeg, vfovDeg: lc.vfovDeg }
      : { kind: lc.kind, channels: lc.channels, vMinDeg: lc.vMinDeg, vMaxDeg: lc.vMaxDeg, hResDeg: lc.hResDeg };
    lidar.push({ name: f.name, frame: f.frame, t: f.time, count: c, beams: f.scan.beams, pattern, mount: lc.mount });
    blobs.push({ name: p + 'points', data: f.scan.points, shape: [c, 4] }, { name: p + 'labels', data: f.scan.labels, shape: [c] },
      { name: p + 'time', data: f.scan.time, shape: [c] }, { name: p + 'ring', data: f.scan.ring, shape: [c] },
      { name: p + 'beam', data: f.scan.beam, shape: [c] });
  }
  const cameras = [];
  for (const f of out.camera || []) {
    const im = f.image, K = im.K;
    cameras.push({ name: f.name, frame: f.frame, t: f.time, width: im.width, height: im.height, bits: im.bits, exposure: im.exposure,
      saturated: im.saturated, fx: K.fx, fy: K.fy, cx: K.cx, cy: K.cy,
      mount: ((world.cameras.find((c) => c.cfg.name === f.name) || {}).cfg || {}).mount });
    blobs.push({ name: `camera/${f.name}/${f.frame}/image`, data: im.dn, shape: [im.height, im.width] });
  }
  const r = world.rover;
  const header = { type: 'obs', seq, t: world.time, dt, wheel: { v: r.v, omega: r.omega }, waypoints, lidar, cameras, truth: truthPose(world) };
  return encode(header, blobs);
}

export function truthPose(world) {
  const r = world.rover, c = r.contact, [x, y, z] = c.t;
  return { x, y, z, yaw: r.yaw, roll: c.roll, pitch: c.pitch, groundX: r.x, groundY: r.y };
}

/** Session info for the hello message. */
export function helloMessage(world, controlPeriod, waypoints) {
  return { type: 'hello', protocol: PROTOCOL_VERSION, info: {
    controlPeriod, waypoints, initialPose: truthPose(world), time: world.time,
    camerasAvailable: !!world.cameraRenderer, // false in Node / without WebGL2: no camera images in obs
    scenario: world.scenario, terrain: { n: world.terrain.n, res: world.terrain.res, x0: world.terrain.x0, y0: world.terrain.y0 },
  } };
}
