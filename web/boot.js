// Entry point: load the app; if the CDN copy of Three.js cannot be loaded (offline, blocked,
// SRI mismatch) switch to the vendored copy once and reload.
const KEY = 'lunarsim.web.local';
import('./js/main.js').catch((err) => {
  let retried = false;
  try { retried = sessionStorage.getItem(KEY) === '1'; } catch (e) { /* storage blocked */ }
  if (!(window.LUNARSIM_WEB || {}).local && !retried) {
    console.warn('CDN load failed, falling back to vendored Three.js:', err);
    try { sessionStorage.setItem(KEY, '1'); } catch (e) { /* ignore */ }
    location.reload();
    return;
  }
  document.getElementById('error').textContent = `読み込みに失敗しました: ${err}`;
  console.error(err);
});
