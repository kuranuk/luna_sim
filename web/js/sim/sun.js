// Sun (and Earth) direction at a lunar site: port of the analytic model of
// lunarsim/core/illumination/sun.py (validated there against SPICE: Sun ≤ 0.07°, Earth ≤ 0.11°).
// Azimuth from North clockwise, elevation above the local horizon, ENU unit vectors.

const RAD = Math.PI / 180;
export const SOLAR_IRRADIANCE_1AU = 1361; // W/m² (Kopp & Lean 2011: 1360.8 ± 0.5)
const AU_M = 149597870700, MOON_R = 1737.4e3, INC = 1.54242; // lunar equator to ecliptic [deg]

const mod360 = (x) => ((x % 360) + 360) % 360;

export const azelToEnu = (az, el) => [Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el)];
export function enuToAzel(v) {
  const az = Math.atan2(v[0], v[1]);
  return [az < 0 ? az + 2 * Math.PI : az, Math.atan2(v[2], Math.hypot(v[0], v[1]))];
}

function elements(jd) { // low-precision solar and lunar mean elements [deg, AU] (Meeus)
  const d = jd - 2451545.0, T = d / 36525;
  const g = mod360(357.528 + 0.9856003 * d) * RAD;
  return {
    lamSun: mod360(280.460 + 0.9856474 * d + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)),
    rAu: 1.00014 - 0.01671 * Math.cos(g) - 0.00014 * Math.cos(2 * g),
    Lp: mod360(218.3164477 + 481267.88123421 * T), D: mod360(297.8501921 + 445267.1114034 * T),
    M: mod360(357.5291092 + 35999.0502909 * T), Mp: mod360(134.9633964 + 477198.8675055 * T),
    F: mod360(93.2720950 + 483202.0175233 * T), Om: mod360(125.0445479 - 1934.1362891 * T),
  };
}

const selUnit = (lat, lon) => [Math.cos(lat * RAD) * Math.cos(lon * RAD), Math.cos(lat * RAD) * Math.sin(lon * RAD), Math.sin(lat * RAD)];

function earthSelenographic(e) { // sub-Earth point (Meeus ch. 47 truncated + ch. 53 libration)
  const D = e.D * RAD, M = e.M * RAD, Mp = e.Mp * RAD, F = e.F * RAD, s = Math.sin;
  const lam = e.Lp + 6.288774 * s(Mp) + 1.274027 * s(2 * D - Mp) + 0.658314 * s(2 * D) + 0.213618 * s(2 * Mp)
    - 0.185116 * s(M) - 0.114332 * s(2 * F) + 0.058793 * s(2 * D - 2 * Mp) + 0.057066 * s(2 * D - M - Mp)
    + 0.053322 * s(2 * D + Mp) + 0.045758 * s(2 * D - M) - 0.040923 * s(M - Mp) - 0.034720 * s(D) - 0.030383 * s(M + Mp);
  const beta = 5.128122 * s(F) + 0.280602 * s(Mp + F) + 0.277693 * s(Mp - F) + 0.173237 * s(2 * D - F)
    + 0.055413 * s(2 * D - Mp + F) + 0.046271 * s(2 * D - Mp - F) + 0.032573 * s(2 * D + F) + 0.017198 * s(2 * Mp + F);
  const distKm = 385000.56 - 20905.355 * Math.cos(Mp) - 3699.111 * Math.cos(2 * D - Mp)
    - 2955.968 * Math.cos(2 * D) - 569.925 * Math.cos(2 * Mp);
  const inc = INC * RAD, W = (lam - e.Om) * RAD, b = beta * RAD;
  const A = Math.atan2(Math.sin(W) * Math.cos(b) * Math.cos(inc) - Math.sin(b) * Math.sin(inc), Math.cos(W) * Math.cos(b));
  const lon = mod360(A / RAD - e.F);
  const lat = Math.asin(-Math.sin(W) * Math.cos(b) * Math.sin(inc) - Math.sin(b) * Math.cos(inc)) / RAD;
  return [lat, lon > 180 ? lon - 360 : lon, distKm * 1e3];
}

/** Sun at a manual azimuth / elevation [deg]. */
export function sunManual(azDeg, elDeg) {
  const az = azDeg * RAD, el = elDeg * RAD;
  return { dir: azelToEnu(az, el), az, el, irradiance: SOLAR_IRRADIANCE_1AU, distanceAu: 1, earthDir: null, earthDistance: null };
}

/** Sun and Earth at selenographic latitude / east longitude [deg] at a Date (UTC). */
export function sunAstronomical(latDeg, lonDeg, date) {
  const e = elements(2440587.5 + date.getTime() / 86400000);
  let slon = mod360(180 - (e.Lp - e.lamSun));
  if (slon > 180) slon -= 360;
  const slat = Math.asin(Math.sin(INC * RAD) * Math.sin((e.lamSun - e.Om) * RAD)) / RAD;
  const [elat, elon, edist] = earthSelenographic(e);
  const la = latDeg * RAD, lo = lonDeg * RAD;
  const E = [-Math.sin(lo), Math.cos(lo), 0];
  const N = [-Math.sin(la) * Math.cos(lo), -Math.sin(la) * Math.sin(lo), Math.cos(la)];
  const U = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const toEnu = (v) => [dot(E, v), dot(N, v), dot(U, v)];
  const site = selUnit(latDeg, lonDeg).map((c) => c * MOON_R);
  const m2e = selUnit(elat, elon).map((c) => c * edist);
  // Geocentric Sun + Moon->Earth vector = selenocentric Sun (removes the ≤ 0.15° parallax).
  const s = toEnu(selUnit(slat, slon).map((c, k) => c * e.rAu * AU_M + m2e[k] - site[k]));
  const ev = toEnu(m2e.map((c, k) => c - site[k]));
  const ns = Math.hypot(...s), ne = Math.hypot(...ev);
  const dir = s.map((c) => c / ns), [az, el] = enuToAzel(dir), rSun = ns / AU_M;
  return { dir, az, el, irradiance: SOLAR_IRRADIANCE_1AU / (rSun * rSun), distanceAu: rSun,
    earthDir: ev.map((c) => c / ne), earthDistance: ne };
}
