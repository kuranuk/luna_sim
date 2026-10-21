// Simulation worker: owns the World, runs the real-time loop and serves sensor data.
// Main thread -> worker: build, run, speed, cmd, waypoints, sun, lidar, reset, sunlit, export,
// gnc (connect / disconnect). Worker -> main: built, state, lidar (transferable arrays), sunlit,
// export, gnc, error.
// GNC mode: the worker is the WebSocket client of a Python `lunarsim gnc` server and runs in
// lockstep: advance one control period (all LiDAR frames), send the observation, wait for the
// command. Sim time advances at most at the real-time x speed rate, slower if Python is slower.

import { helloMessage, observationMessage, truthPose } from './gnc/protocol.js';
import { World } from './sim/world.js';

let world = null, running = false, speed = 1, last = 0, timer = null, stateEvery = 0;
const log = { traj: [], imu: [], lastScan: {} };
const DISPLAY_MAX = 512; // display mesh nodes per side
const LIDAR_POST_MS = 80; // throttle point-cloud messages to the main thread (per LiDAR)
const lastLidarPost = {};
const gnc = { ws: null, url: '', ready: false, waiting: false, budget: 0, period: 0.1, seq: 0, name: '',
  status: '', waypoints: [], estimate: null, estTrail: [], errors: [], sentAt: 0, rttMs: 0 };

function post(msg, transfer = []) { self.postMessage(msg, transfer); }

function terrainPayload(w) {
  const t = w.terrain, stride = t.n > DISPLAY_MAX ? Math.ceil((t.n - 1) / (DISPLAY_MAX - 1)) : 1;
  const m = Math.floor((t.n - 1) / stride) + 1, z = new Float32Array(m * m), lab = new Uint8Array(m * m);
  for (let a = 0; a < m; a++) {
    for (let b = 0; b < m; b++) { z[a * m + b] = t.z[a * stride * t.n + b * stride]; lab[a * m + b] = t.labels[a * stride * t.n + b * stride]; }
  }
  return { z, labels: lab, m, stride, n: t.n, res: t.res, x0: t.x0, y0: t.y0, extent: t.extent,
    craters: t.craters.length, rocks: t.rocks.length };
}

function sendSunlit() {
  const st = world.terrain.n > DISPLAY_MAX ? Math.ceil((world.terrain.n - 1) / (DISPLAY_MAX - 1)) : 1;
  const t0 = performance.now(), s = world.sunlit(st);
  post({ type: 'sunlit', lit: s.lit, m: s.m, ms: performance.now() - t0, sun: world.state().sun }, [s.lit.buffer]);
}

function handleOutput(out) {
  for (const s of out.imu) log.imu.push(s);
  for (const p of out.poses || []) log.traj.push(p);
  if (log.imu.length > 200000) log.imu.splice(0, log.imu.length - 200000);
  const now = performance.now(), latest = {};
  for (const f of out.lidar) { log.lastScan[f.name] = f; latest[f.name] = f; }
  for (const f of Object.values(latest)) {
    if (now - (lastLidarPost[f.name] || -1e9) < LIDAR_POST_MS) continue;
    lastLidarPost[f.name] = now;
    const world3 = f.scan.world.slice(), inten = new Float32Array(f.scan.count), lab = f.scan.labels.slice();
    for (let k = 0; k < f.scan.count; k++) inten[k] = f.scan.points[4 * k + 3];
    post({ type: 'lidar', name: f.name, frame: f.frame, time: f.time, count: f.scan.count, beams: f.scan.beams,
      world: world3, intensity: inten, labels: lab, ms: f.ms }, [world3.buffer, inten.buffer, lab.buffer]);
  }
}

function sendState() {
  const st = world.state();
  post({ type: 'state', state: st, trail: world.trail.length > 1 ? world.trail.slice(-2000) : null, gnc: gncInfo() });
}

// ------------------------------------------------------------------ GNC (Python over WebSocket)
function gncInfo() {
  if (!gnc.ws) return null;
  return { state: gnc.ready ? 'connected' : 'connecting', url: gnc.url, name: gnc.name, status: gnc.status, seq: gnc.seq,
    rttMs: gnc.rttMs, estimate: gnc.estimate, estTrail: gnc.estTrail.slice(-2000), truth: truthPose(world) };
}

function gncConnect(url, period) {
  gncDisconnect();
  gnc.url = url; gnc.period = period;
  const ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  gnc.ws = ws;
  ws.onopen = () => { if (gnc.ws === ws) gncHello(); };
  ws.onmessage = (ev) => { if (gnc.ws === ws) gncMessage(ev.data); };
  ws.onerror = () => { if (gnc.ws === ws) post({ type: 'gnc', event: 'error', message: `WebSocket に接続できません: ${url}` }); };
  ws.onclose = (ev) => {
    if (gnc.ws !== ws) return;
    gnc.ws = null; gnc.ready = false; gnc.waiting = false;
    if (world) world.rover.setCommand(0, 0);
    post({ type: 'gnc', event: 'closed', code: ev.code, reason: ev.reason });
    if (world) sendState();
  };
  post({ type: 'gnc', event: 'connecting', url });
}

function gncDisconnect() {
  if (!gnc.ws) return;
  const ws = gnc.ws;
  gnc.ws = null; gnc.ready = false; gnc.waiting = false;
  ws.close();
  if (world) world.rover.setCommand(0, 0);
  post({ type: 'gnc', event: 'closed', code: 1000, reason: 'disconnected' });
}

/** (Re)start the GNC session: on connect, reset and new terrain. */
function gncHello() {
  if (!gnc.ws || gnc.ws.readyState !== WebSocket.OPEN || !world) return;
  Object.assign(gnc, { ready: false, waiting: false, budget: 0, seq: 0, estimate: null, estTrail: [], status: '' });
  world.follower = null;
  world.rover.setCommand(0, 0);
  gnc.ws.send(JSON.stringify(helloMessage(world, gnc.period, gnc.waypoints)));
}

function gncMessage(data) {
  if (typeof data !== 'string') return;
  const m = JSON.parse(data);
  if (m.type === 'ready') {
    gnc.ready = true; gnc.name = m.name || '';
    post({ type: 'gnc', event: 'ready', name: gnc.name });
    gncPump();
  } else if (m.type === 'cmd') {
    if (m.seq !== gnc.seq) return; // stale reply from before a reset
    gnc.rttMs = performance.now() - gnc.sentAt;
    world.rover.setCommand(m.v, m.omega);
    gnc.status = m.status || '';
    gnc.estimate = m.estimate || null;
    if (gnc.estimate) {
      const e = gnc.estimate, last = gnc.estTrail[gnc.estTrail.length - 1];
      if (!last || Math.hypot(e.x - last[0], e.y - last[1]) > 0.05) gnc.estTrail.push([e.x, e.y, e.z]);
    }
    gnc.waiting = false;
    gncPump();
  } else if (m.type === 'error') {
    gnc.waiting = false; running = false;
    world.rover.setCommand(0, 0);
    post({ type: 'gnc', event: 'error', message: `Python 側でエラー: ${m.message}`, paused: true });
  }
}

/** Advance one control period and send the observation, if the time budget allows. */
function gncPump() {
  if (!gnc.ready || gnc.waiting || !running || gnc.budget < gnc.period - 1e-9) return;
  gnc.budget -= gnc.period;
  const t0 = performance.now(), out = world.advance(gnc.period, { scans: 'all' });
  for (const f of out.lidar) f.ms = (performance.now() - t0) / out.lidar.length;
  gnc.seq++;
  gnc.ws.send(observationMessage(world, out, gnc.seq, gnc.period, gnc.waypoints));
  gnc.sentAt = performance.now();
  gnc.waiting = true;
  handleOutput(out);
}

function tick() {
  if (!world || !running) return;
  const now = performance.now(), dt = Math.min((now - last) / 1000, 0.25) * speed;
  last = now;
  if (gnc.ws) { // lockstep with Python
    gnc.budget = Math.min(gnc.budget + dt, Math.max(0.5, 2 * gnc.period));
    gncPump();
    if (++stateEvery % 2 === 0) sendState();
    return;
  }
  const t0 = performance.now(), out = world.advance(dt, { scans: 'latest' });
  for (const f of out.lidar) f.ms = (performance.now() - t0) / Math.max(out.lidar.length, 1);
  handleOutput(out);
  if (++stateEvery % 2 === 0) sendState();
}

function startLoop() {
  if (timer) clearInterval(timer);
  last = performance.now();
  timer = setInterval(tick, 25);
}

function csv(rows, header) { return header + '\n' + rows.map((r) => r.join(',')).join('\n') + '\n'; }

function exportData(what, name) {
  if (what === 'trajectory') { // TUM: t tx ty tz qx qy qz qw (world ENU, body frame)
    return { filename: 'trajectory_tum.txt', mime: 'text/plain',
      data: log.traj.map(([t, x, y, z, qw, qx, qy, qz]) => [t, x, y, z, qx, qy, qz, qw].map((v) => v.toFixed(6)).join(' ')).join('\n') + '\n' };
  }
  if (what === 'imu') { // EuRoC-style
    return { filename: 'imu.csv', mime: 'text/csv', data: csv(log.imu.map((s) => [Math.round(s.t * 1e9), ...s.gyro, ...s.accel]),
      '#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]') };
  }
  if (what === 'ply') { // last scan of `name`, sensor frame (FLU)
    const f = log.lastScan[name];
    if (!f) throw new Error(`no scan of ${name} yet`);
    const n = f.scan.count, hdr = `ply\nformat binary_little_endian 1.0\ncomment lunarsim-web ${name} frame ${f.frame} t=${f.time.toFixed(3)} sensor frame FLU\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nproperty uchar label\nend_header\n`;
    const head = new TextEncoder().encode(hdr), buf = new ArrayBuffer(head.length + n * 17), dv = new DataView(buf);
    new Uint8Array(buf).set(head);
    for (let k = 0, o = head.length; k < n; k++, o += 17) {
      for (let c = 0; c < 4; c++) dv.setFloat32(o + 4 * c, f.scan.points[4 * k + c], true);
      dv.setUint8(o + 16, f.scan.labels[k]);
    }
    return { filename: `${name}_${String(f.frame).padStart(6, '0')}.ply`, mime: 'application/octet-stream', data: buf };
  }
  if (what === 'heightmap') { // .npy float64 (rows south -> north) + geometry in the header comment-free JSON sidecar
    const t = world.terrain, header = `{'descr': '<f8', 'fortran_order': False, 'shape': (${t.n}, ${t.n}), }`;
    let h = `\x93NUMPY\x01\x00`, pad = 64 - ((10 + header.length + 1) % 64);
    const hdr = header + ' '.repeat(pad % 64) + '\n';
    h += String.fromCharCode(hdr.length & 255, hdr.length >> 8) + hdr;
    const head = Uint8Array.from(h, (ch) => ch.charCodeAt(0)), buf = new ArrayBuffer(head.length + t.z.byteLength);
    new Uint8Array(buf).set(head);
    new Float64Array(buf, head.length).set(t.z);
    return { filename: `heightmap_${t.n}x${t.n}_res${t.res}m_x0${t.x0}_y0${t.y0}.npy`, mime: 'application/octet-stream', data: buf };
  }
  throw new Error(`unknown export ${what}`);
}

self.onmessage = (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'build') {
      running = false;
      world = new World(m.scenario);
      log.traj = []; log.imu = world.imuSamples ? world.imuSamples.slice() : []; log.lastScan = {};
      const tp = terrainPayload(world);
      post({ type: 'built', terrain: tp, buildMs: world.buildMs, lidars: world.lidars.map((l) => ({ name: l.cfg.name, beams: l.pattern.n, rateHz: l.cfg.rateHz })),
        state: world.state() }, [tp.z.buffer, tp.labels.buffer]);
      sendSunlit();
      handleOutput({ imu: [], lidar: world.lidars.map((l) => world.scan(l)) });
      for (const l of world.lidars) l.next = 1 / l.cfg.rateHz;
      startLoop();
      gncHello();
    } else if (m.type === 'gnc') {
      if (m.connect) gncConnect(m.url, m.period); else gncDisconnect();
    } else if (!world) {
      return;
    } else if (m.type === 'run') { running = m.running; last = performance.now(); }
    else if (m.type === 'speed') speed = m.speed;
    else if (m.type === 'cmd') { if (!gnc.ws) world.setCommand(m.v, m.omega); }
    else if (m.type === 'waypoints') { gnc.waypoints = m.points; if (!gnc.ws && m.points.length) world.setWaypoints(m.points, m.speed); }
    else if (m.type === 'sun') { if (m.utc) world.setSunUtc(m.utc); else world.setSunManual(m.azDeg, m.elDeg); sendSunlit(); sendState(); }
    else if (m.type === 'lidar') { const l = world.lidars.find((x) => x.cfg.name === m.name); if (l) l.enabled = m.on; }
    else if (m.type === 'reset') { world.reset(); log.traj = []; log.imu = world.imuSamples ? world.imuSamples.slice() : []; gncHello(); sendState(); }
    else if (m.type === 'export') { const r = exportData(m.what, m.name); post({ type: 'export', ...r }, r.data instanceof ArrayBuffer ? [r.data] : []); }
    else if (m.type === 'state') sendState();
  } catch (e) {
    post({ type: 'error', message: String(e && e.stack || e) });
  }
};
