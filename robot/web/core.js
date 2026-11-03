// What every tab shares: the robot's state (pushed by /api/events), the action catalog, calling actions.
export const $ = id => document.getElementById(id);
export const n1 = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '?' : Number(v).toFixed(d);
export const has = v => v !== null && v !== undefined;
export const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

export const R = {S: null, catalog: {}, lastEvent: 0, listeners: []};

export function onState(fn) { R.listeners.push(fn); }

let toastText = '', toastUntil = 0;
export function toast(text) {
  if (text === toastText && Date.now() < toastUntil - 2000) return;  // the same error again: once is enough
  toastText = text; toastUntil = Date.now() + 4000;
}
export function message() {
  return Date.now() < toastUntil ? toastText : (R.S && R.S.message) || '';
}

async function post(name, params) {
  const r = await fetch('/api/' + name, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify(params)});
  return r.json();
}

// Do an action: asks its confirmation first (from /api/actions) unless confirm: false; shows errors.
export async function call(name, params = {}, {confirm: ask = true, quiet = false} = {}) {
  const a = R.catalog[name];
  if (ask && a && a.confirm && !window.confirm(a.confirm)) return {error: 'cancelled'};
  try {
    const j = await post(name, params);
    if (j.error && !quiet) toast(j.error);
    return j;
  } catch (e) {
    if (!quiet) toast("Couldn't reach the robot");
    return {error: 'offline'};
  }
}

function update(data, full) {
  if (!full && R.S && data.joints)  // fast updates leave out each servo's settings: keep the last ones
    for (const [n, j] of Object.entries(data.joints)) j.info = R.S.joints[n] && R.S.joints[n].info;
  R.S = full || !R.S ? data : {...R.S, ...data};
  R.lastEvent = Date.now();
  if (!R.S.pi) return;  // until the first full state
  for (const fn of R.listeners) {
    try { fn(R.S, full); } catch (e) { console.error(e); }
  }
}

export async function connect() {
  try { R.catalog = Object.fromEntries((await (await fetch('/api/actions')).json()).map(a => [a.name, a])); }
  catch (e) { setTimeout(connect, 1000); return; }
  const es = new EventSource('/api/events');
  es.addEventListener('fast', e => update(JSON.parse(e.data), false));
  es.addEventListener('full', e => update(JSON.parse(e.data), true));
  // The page is there: without this the walking policy stands still after 2 s and stops after 6 s.
  setInterval(() => { if (!document.hidden) post('heartbeat', {}).catch(() => {}); }, 500);
}
export const connected = () => Date.now() - R.lastEvent < 1500;

// ---- battery (3S LiPo, from the servos' supply readings) ----
const CURVE = [[3.27,0],[3.61,5],[3.69,10],[3.71,15],[3.73,20],[3.75,25],[3.77,30],[3.79,35],[3.80,40],[3.82,45],
  [3.84,50],[3.85,55],[3.87,60],[3.91,65],[3.95,70],[3.98,75],[4.02,80],[4.08,85],[4.11,90],[4.15,95],[4.20,100]];
function percent(v) {
  if (v <= CURVE[0][0]) return 0;
  for (let k = 1; k < CURVE.length; k++) {
    const [v0, p0] = CURVE[k - 1], [v1, p1] = CURVE[k];
    if (v <= v1) return p0 + (p1 - p0) * (v - v0) / (v1 - v0);
  }
  return 100;
}
let battV = null;
export function battery(S) {
  const vins = Object.values(S.joints).map(j => j.vin).filter(v => v);
  if (!vins.length) return null;
  const now = Math.max(...vins);
  battV = battV === null ? now : battV + 0.3 * (now - battV);
  return {volts: battV, pct: Math.round(percent(battV / 3)), level: battV < 10.2 ? 'bad' : battV < 10.5 ? 'warn' : 'ok'};
}
