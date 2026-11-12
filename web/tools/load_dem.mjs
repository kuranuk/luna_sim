// Node helper: read the DEM named by scenario.terrain.dem.url (relative to web/) from disk.
// Returns null when the scenario does not ask for one.
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { demWanted, parseDem } from '../js/sim/dem.js';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');

export function loadDem(sc) {
  if (!demWanted(sc)) return null;
  const path = resolve(WEB, sc.terrain.dem.url);
  if (!path.startsWith(WEB)) throw new Error(`DEM path outside web/: ${sc.terrain.dem.url}`);
  const meta = JSON.parse(readFileSync(path, 'utf8'));
  const raw = readFileSync(resolve(dirname(path), meta.data));
  return parseDem(meta, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}
