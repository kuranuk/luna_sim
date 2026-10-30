// Moon-wide constants and frames (no DOM, testable in Node).

export const MOON_RADIUS = 1737400; // [m] IAU mean radius

/** 3x3 rotation (row-major) Moon-fixed (X lon 0, Y lon 90E, Z north) -> local ENU at (lat, lon) [deg]. */
export function enuFromMoonFixed(latDeg, lonDeg) {
  const la = latDeg * Math.PI / 180, lo = lonDeg * Math.PI / 180;
  const sl = Math.sin(la), cl = Math.cos(la), so = Math.sin(lo), co = Math.cos(lo);
  return [-so, co, 0, -sl * co, -sl * so, cl, cl * co, cl * so, sl];
}
