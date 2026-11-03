// The shell: top bar (connection, mode, battery, messages, STOP), alerts, tabs. Each tab is its own module.
import {$, R, battery, call, connect, connected, message, onState} from './core.js';
import * as drive from './drive.js';
import * as actions from './actions.js';
import * as status from './status.js';
import * as setup from './setup.js';

const TABS = {drive, actions, status, setup};
for (const [name, tab] of Object.entries(TABS)) tab.build($('tab-' + name));

function showTab() {
  const name = (location.hash || '#drive').slice(1);
  const current = TABS[name] ? name : 'drive';
  for (const n of Object.keys(TABS)) $('tab-' + n).classList.toggle('on', n === current);
  document.querySelectorAll('#tabs a').forEach(a => a.classList.toggle('on', a.hash === '#' + current));
}
window.addEventListener('hashchange', showTab);
showTab();

$('stopAll').onclick = () => call('stop');

function alerts(S) {
  const bad = [], warn = [];
  if (S.pi_hot) bad.push(`The Pi reached ${S.limits.pi_temp}°C: all servos are relaxed until it cools.`);
  for (const [id, t] of Object.entries(S.hot)) bad.push(`Servo ${id} is at ${t}°C: relaxed and locked until it cools.`);
  if (!S.zero_set) warn.push('Not calibrated: angles are relative to count 500. Set zero (Setup) before trusting any angle.');
  const missing = Object.values(S.joints).filter(j => !j.found).map(j => j.id);
  if (missing.length) warn.push(`Servo${missing.length > 1 ? 's' : ''} ${missing.join(', ')} not found: check power and cables, then Rescan (Setup).`);
  for (const j of Object.values(S.joints)) {
    const v = j.info && j.info.vin_limits;
    if (v && v[1] < 12.6) { warn.push(`Servo ${j.id}'s voltage limit is ${v[1]} V: a full 3S pack (12.6 V) can make it go limp.`); break; }
  }
  const b = battery(S);
  if (b && b.level === 'bad') bad.push(`Battery ${b.volts.toFixed(2)} V: stop and charge.`);
  $('alerts').innerHTML = (bad.length ? `<div class="alert">${bad.join('<br>')}</div>` : '') +
    (warn.length ? `<div class="alert warn">${warn.join('<br>')}</div>` : '');
}

let lastAlerts = 0;
onState((S, full) => {
  const w = S.walker;
  $('mode').textContent = S.mode === 'walking' ? `walking: ${w.policy}` : S.mode === 'gesture' ? `gesture: ${S.gesture}` : S.mode;
  $('mode').className = 'pill ' + (S.mode === 'walking' ? 'ok' : S.mode === 'idle' ? '' : 'warn');
  const b = battery(S);
  $('battery').textContent = b ? `${b.volts.toFixed(1)} V ~${b.pct}%` : 'battery ?';
  $('battery').className = 'pill ' + (b ? b.level : '');
  if (full || Date.now() - lastAlerts > 1000) { alerts(S); lastAlerts = Date.now(); }
});

setInterval(() => {
  $('conn').classList.toggle('on', connected());
  $('msg').textContent = connected() ? message() : 'Lost connection to the robot';
  $('msg').className = 'msg' + (connected() ? '' : ' bad');
}, 250);

document.addEventListener('visibilitychange', () => {
  document.title = document.hidden && R.S && R.S.walker.running ? 'Juan: keep this tab open!' : 'Juan';
});
connect();
