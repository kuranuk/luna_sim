// Real-DEM base terrain (pure JS, no DOM).
// A DEM file pair made by scripts/fetch_lola_dem.py: <name>.json (meta) + <name>.f32 (float32, little
// endian, row-major, row 0 = south). Heights [m] are in the local ENU frame of the site, relative to
// the site, on a regular grid x = x0 + j*res (east), y = y0 + i*res (north).
// Source of the shipped file: LRO LOLA RDR laser shots (USGS Astrogeology tiles on AWS Open Data).

export const DEM_FORMAT = 'lunarsim-dem/1';

/** Build a DEM object from the meta JSON and the raw float32 buffer. Throws on any mismatch. */
export function parseDem(meta, buffer) {
  if (!meta || meta.format !== DEM_FORMAT) throw new Error(`DEM: format は "${DEM_FORMAT}" のはずです（${JSON.stringify(meta && meta.format)}）`);
  const { n, res, x0, y0 } = meta;
  if (!Number.isInteger(n) || n < 4 || !(res > 0) || !Number.isFinite(x0) || !Number.isFinite(y0)) throw new Error('DEM: n / res / x0 / y0 が不正です');
  if (buffer.byteLength !== 4 * n * n) throw new Error(`DEM: データ長 ${buffer.byteLength} バイトは ${n} x ${n} の float32 と合いません`);
  const view = new DataView(buffer instanceof ArrayBuffer ? buffer : buffer.buffer, buffer.byteOffset || 0, buffer.byteLength);
  const z = new Float32Array(n * n);
  for (let k = 0; k < n * n; k++) { // explicit little endian (independent of the host)
    z[k] = view.getFloat32(4 * k, true);
    if (!Number.isFinite(z[k])) throw new Error('DEM: 有限でない値が含まれています');
  }
  return { z, n, res, x0, y0, site: { lat: meta.site.lat, lon: meta.site.lon }, source: meta.source || '', stats: meta.stats || {} };
}

const cr = (p0, p1, p2, p3, t) => p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));

/** Height [m] at (x, y) by Catmull-Rom bicubic interpolation (C1, exact at the nodes).
 *  Outside the grid the coordinates are clamped to the edge (flat continuation). */
export function demHeight(dem, x, y) {
  const { z, n, res } = dem, m = n - 1;
  const fx = Math.min(m, Math.max(0, (x - dem.x0) / res)), fy = Math.min(m, Math.max(0, (y - dem.y0) / res));
  const j = Math.min(m - 1, Math.floor(fx)), i = Math.min(m - 1, Math.floor(fy)), tx = fx - j, ty = fy - i;
  const row = (a) => {
    const o = Math.min(m, Math.max(0, a)) * n;
    return cr(z[o + Math.max(0, j - 1)], z[o + j], z[o + j + 1], z[o + Math.min(m, j + 2)], tx);
  };
  return cr(row(i - 1), row(i), row(i + 1), row(i + 2), ty);
}

/** Problems that make the DEM unusable for this scenario ([] = fine). */
export function demProblems(dem, sc) {
  const out = [], t = sc.terrain, half = 0.5 * (t.n - 1) * t.res;
  if (Math.abs(dem.site.lat - sc.site.lat) > 1e-6 || Math.abs(dem.site.lon - sc.site.lon) > 1e-6) {
    out.push(`DEM の地点（緯度 ${dem.site.lat}°, 経度 ${dem.site.lon}°）がシナリオの地点（${sc.site.lat}°, ${sc.site.lon}°）と違います`);
  }
  if (-half < dem.x0 || half > dem.x0 + (dem.n - 1) * dem.res || -half < dem.y0 || half > dem.y0 + (dem.n - 1) * dem.res) {
    out.push(`地形（±${half.toFixed(0)} m）が DEM の範囲（${dem.x0} 〜 ${dem.x0 + (dem.n - 1) * dem.res} m）をはみ出します`);
  }
  return out;
}

/** True if the scenario asks for a DEM base. */
export const demWanted = (sc) => !!(sc.terrain.dem && sc.terrain.dem.enabled && sc.terrain.dem.url);
