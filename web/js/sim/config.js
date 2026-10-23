// Scenario config files (JSON) for the browser simulator.
//
// A config is a preset plus overrides:
//   { "format": "lunarsim-web/1", "base": "south_pole_lowsun", "seed": 7,
//     "terrain": { "n": 1024, "res": 0.2 }, "sun": { "mode": "manual", "azDeg": 90, "elDeg": 5 },
//     "cameras": [ { "name": "navcam", "hfovDeg": 90 }, { "name": "mast", "mount": { "z": 1.5 } } ],
//     "lidars": [ { "name": "spad_flash", "hfovDeg": 80, "vfovDeg": 60 } ] }
// Objects are merged key by key onto the base. The arrays cameras / lights / lidars define the sensor
// list: each entry is merged onto the base sensor of the same name, or onto a template for a new name
// (lidars: by "kind", default flash), so a config only needs the values it changes. Omitting an
// array keeps the base sensors; "enabled": false keeps a sensor but switches it off; [] removes all.
// Units as in scenarios.js: metres, seconds, degrees in fields ending in Deg.

import { SCENARIOS, TEMPLATES } from './scenarios.js';

export const CONFIG_FORMAT = 'lunarsim-web/1';
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/** Recursive merge: objects key by key, everything else (numbers, strings, arrays) replaced. */
export function deepMerge(base, over) {
  if (!isObj(base) || !isObj(over)) return structuredClone(over);
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(over)) out[k] = isObj(v) && isObj(out[k]) ? deepMerge(out[k], v) : structuredClone(v);
  return out;
}

function mergeList(kind, baseList, entries, errors) {
  if (!Array.isArray(entries)) { errors.push(`${kind} は配列で指定してください`); return baseList; }
  const seen = new Set();
  return entries.map((e, k) => {
    if (!isObj(e) || typeof e.name !== 'string' || !e.name) { errors.push(`${kind}[${k}]: "name" が必要です`); return null; }
    if (seen.has(e.name)) errors.push(`${kind}: 名前 "${e.name}" が重複しています`);
    seen.add(e.name);
    let base = baseList.find((b) => b.name === e.name);
    if (!base) {
      if (kind === 'lidars') {
        const lk = e.kind || 'flash';
        base = TEMPLATES.lidar[lk];
        if (!base) { errors.push(`lidars[${k}]: kind "${lk}" は flash / spinning のどちらかです`); return null; }
      } else base = kind === 'cameras' ? TEMPLATES.camera : TEMPLATES.light;
    }
    return deepMerge(base, e);
  }).filter(Boolean);
}

function check(sc, errors) {
  const num = (v, path, lo, hi, { int = false, openLo = false } = {}) => {
    if (typeof v !== 'number' || !Number.isFinite(v) || (int && !Number.isInteger(v)) || (openLo ? v <= lo : v < lo) || v > hi) {
      errors.push(`${path} = ${JSON.stringify(v)}: ${int ? '整数 ' : ''}${openLo ? '(' : '['}${lo}, ${hi}] の範囲で指定してください`);
    }
  };
  const t = sc.terrain;
  num(t.n, 'terrain.n', 64, 2048, { int: true });
  if (Number.isInteger(t.n) && (t.n & (t.n - 1))) errors.push(`terrain.n = ${t.n}: 2 のべき乗にしてください（FFT）`);
  num(t.res, 'terrain.res', 0.01, 10);
  num(sc.seed, 'seed', 0, 2 ** 32 - 1, { int: true });
  if (!['manual', 'astronomical'].includes(sc.sun.mode)) errors.push('sun.mode は manual / astronomical');
  if (sc.sun.mode === 'astronomical' && Number.isNaN(Date.parse(sc.sun.utc))) errors.push(`sun.utc = ${JSON.stringify(sc.sun.utc)}: ISO 8601 の日時にしてください`);
  const mount = (m, path) => { for (const k of ['x', 'y', 'z', 'rollDeg', 'pitchDeg', 'yawDeg']) num(m[k], `${path}.mount.${k}`, -1e3, 1e3); };
  sc.cameras.forEach((c, k) => {
    const p = `cameras[${k}] (${c.name})`;
    num(c.width, `${p}.width`, 8, 4096, { int: true }); num(c.height, `${p}.height`, 8, 4096, { int: true });
    num(c.hfovDeg, `${p}.hfovDeg`, 5, 170); num(c.rateHz, `${p}.rateHz`, 0, 100, { openLo: true });
    if (![8, 10, 12, 14, 16].includes(c.sensor.bits)) errors.push(`${p}.sensor.bits は 8/10/12/14/16`);
    mount(c.mount, p);
  });
  sc.lidars.forEach((l, k) => {
    const p = `lidars[${k}] (${l.name})`;
    num(l.rateHz, `${p}.rateHz`, 0, 100, { openLo: true }); num(l.rangeMax, `${p}.rangeMax`, 0.1, 1e4);
    if (l.kind === 'flash') {
      num(l.hfovDeg, `${p}.hfovDeg`, 1, 170); num(l.vfovDeg, `${p}.vfovDeg`, 1, 170);
      num(l.width, `${p}.width`, 1, 2048, { int: true }); num(l.height, `${p}.height`, 1, 2048, { int: true });
    } else {
      num(l.channels, `${p}.channels`, 1, 512, { int: true }); num(l.hResDeg, `${p}.hResDeg`, 0.01, 90);
      num(l.vMinDeg, `${p}.vMinDeg`, -90, 90); num(l.vMaxDeg, `${p}.vMaxDeg`, -90, 90);
      if (l.vMaxDeg < l.vMinDeg) errors.push(`${p}: vMaxDeg < vMinDeg`);
    }
    mount(l.mount, p);
  });
  sc.lights.forEach((l, k) => {
    const p = `lights[${k}] (${l.name})`;
    num(l.fluxLm, `${p}.fluxLm`, 0, 1e7);
    num(l.beamHalfAngleDeg, `${p}.beamHalfAngleDeg`, 0, 89, { openLo: true });
    num(l.fieldHalfAngleDeg, `${p}.fieldHalfAngleDeg`, l.beamHalfAngleDeg, 89.9, { openLo: true });
    mount(l.mount, p);
  });
}

/** Config object -> full scenario. Throws an Error listing every problem found. */
export function resolveConfig(cfg) {
  const errors = [];
  if (!isObj(cfg)) throw new Error('設定は JSON オブジェクトにしてください');
  if (cfg.format !== undefined && cfg.format !== CONFIG_FORMAT) errors.push(`format "${cfg.format}" は未対応です（${CONFIG_FORMAT}）`);
  const baseName = cfg.base || 'south_pole_lowsun';
  const base = SCENARIOS[baseName];
  if (!base) throw new Error(`base "${baseName}" はありません（${Object.keys(SCENARIOS).join(', ')}）`);
  const { format, base: _b, cameras, lights, lidars, ...rest } = cfg;
  const sc = deepMerge(base, rest);
  for (const [key, list] of [['cameras', cameras], ['lights', lights], ['lidars', lidars]]) {
    if (list !== undefined) sc[key] = mergeList(key, base[key] || [], list, errors);
  }
  if (cfg.imu === null) sc.imu = null;
  if (sc.name === base.name && cfg.name === undefined) sc.name = `${baseName} (config)`;
  if (!errors.length) check(sc, errors);
  if (errors.length) throw new Error(`設定エラー:\n- ${errors.join('\n- ')}`);
  return sc;
}

/** Full scenario -> config JSON text (complete, so it also documents every parameter). */
export function scenarioToConfig(sc, baseName) {
  return JSON.stringify({ format: CONFIG_FORMAT, base: baseName, ...structuredClone(sc) }, null, 2) + '\n';
}
