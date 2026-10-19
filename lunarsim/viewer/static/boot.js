// Entry point: load the viewer; if the CDN copy of Three.js cannot be loaded (offline,
// blocked, SRI mismatch), switch to the vendored copy once and reload.
const KEY = 'lunarsim.viewer.local';
import('./app.js').catch((err) => {
  const cfg = window.LUNARSIM_VIEWER || {};
  let retried = false;
  try { retried = sessionStorage.getItem(KEY) === '1'; } catch (e) { /* storage blocked */ }
  if (!cfg.local && !retried) {
    console.warn('CDN load failed, falling back to vendored Three.js:', err);
    try { sessionStorage.setItem(KEY, '1'); } catch (e) { /* ignore */ }
    location.reload();
    return;
  }
  const box = document.getElementById('error');
  if (box) box.textContent = `viewer failed to load: ${err}`;
  console.error(err);
});
