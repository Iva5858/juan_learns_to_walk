// Status: battery, the Pi, the servo buses, the IMU, the voice and camera, as read on the robot.
import {$, battery, call, esc, has, n1, onState} from './core.js';

let built = false;

export function build(root) {
  root.innerHTML = `
  <div class="sub" style="margin-bottom:8px">The driving keys work here too (0–6 speed, arrows, Space, Esc).</div>
  <div class="grid">
    <div class="card"><h3>Battery (servo pack)</h3>
      <div class="big" id="sBatt">-</div><div class="meter"><div id="sBattBar"></div></div>
      <div class="sub" id="sBattNote">Read from the servos</div></div>
    <div class="card"><h3>Raspberry Pi</h3>
      <div class="big" id="sPiTemp">-</div>
      <div class="sub">CPU temperature · servos relax at <span id="sPiLim"></span>°C · power <b id="sPower">-</b></div>
      <div class="kv" id="sPi"></div><div class="cores" id="sCores"></div></div>
    <div class="card"><h3>Servo buses</h3><div class="kv" id="sBuses"></div>
      <table style="margin-top:8px"><thead><tr><th>Servo</th><th>Temp</th><th>Volt</th><th>Torque</th></tr></thead>
      <tbody id="sServos"></tbody></table></div>
    <div class="card"><h3>IMU</h3>
      <div class="big" id="sLean">-</div><svg class="horizon" id="sHorizon" viewBox="0 0 220 100"></svg>
      <div class="kv" id="sImu"></div><div class="sub" id="sImuNote"></div>
      <div class="row"><button class="small" id="sZeroGyro" title="Measure the gyro's bias over 2 s (also done at start-up)">Zero gyro (robot still, 2 s)</button>
        <button class="small" id="sLevel" title="The lean shown here becomes relative to now; the policy always gets the raw tilt">Level here (display only)</button></div>
      <details><summary class="sub">Axes and bus (set once, when mounting the IMU)</summary>
        <div class="row">Torso x, y, z from sensor <select id="sAx0"></select> <select id="sAx1"></select> <select id="sAx2"></select>
          <button class="small" id="sAxes">Apply</button></div>
        <div class="row">I2C bus <input id="sBus" type="number" min="0" max="20" style="width:60px"> <button class="small" id="sBusBtn">Apply</button></div>
        <div class="sub">Check: tip it forward and "fwd" goes up; tip it to its right and "right" goes up. Gyro: rolling right gives
          x &gt; 0, nose down y &gt; 0, turning left z &gt; 0.</div></details></div>
    <div class="card"><h3>Voice, faces, camera</h3><div class="kv" id="sExtras"></div></div>
    <div class="card"><h3>System</h3>
      <div class="row"><button class="danger" id="sShutdown">Shut down dashboard</button></div>
      <div class="sub">Stops the walking policy and saves its log, stops gestures, faces and voice, squats down
        (~4 s) and relaxes every servo, then exits the dashboard. Relaxed, it can still tip over: hold it or keep the
        harness on. Fallen, it relaxes at once. Start it again on the Pi: <kbd>python3 -m robot.dashboard</kbd>.</div>
      <div class="sub" id="sShutNote"></div></div>
  </div>`;
  $('sShutdown').onclick = async () => {
    $('sShutNote').textContent = 'Squatting, then relaxing…';
    const r = await call('shutdown');
    if (!r.error) $('sShutNote').textContent = 'Dashboard shut down' + (r.log ? `; walk log saved: ${r.log}` : '') + '.';
  };
  $('sZeroGyro').onclick = () => call('cal/zero_gyro');
  $('sLevel').onclick = () => call('cal/imu_level');
  $('sAxes').onclick = () => call('cal/imu', {axes: ['sAx0', 'sAx1', 'sAx2'].map(id => $(id).value).join(',')});
  $('sBusBtn').onclick = () => call('cal/imu', {bus: +$('sBus').value});
}

const cls = (v, w, b) => !has(v) ? '' : v >= b ? 'bad' : v >= w ? 'warn' : '';

onState((S, full) => {
  const b = battery(S);
  $('sBatt').textContent = b ? `${b.volts.toFixed(2)} V · ~${b.pct}%` : '-';
  $('sBattBar').style.width = (b ? b.pct : 0) + '%';
  $('sBattBar').style.background = b ? `var(--${b.level})` : '';
  $('sBattNote').textContent = b ? `3S pack, ${(b.volts / 3).toFixed(2)} V per cell` + (b.level === 'bad' ? ' · stop and charge' : b.level === 'warn' ? ' · charge soon' : '') : 'No reading';
  imu(S);
  if (!built && S.imu_stats) {
    const opts = ['x', '-x', 'y', '-y', 'z', '-z'].map(a => `<option>${a}</option>`).join('');
    ['sAx0', 'sAx1', 'sAx2'].forEach((id, k) => { $(id).innerHTML = opts; $(id).value = S.imu_stats.axes[k]; });
    $('sBus').value = S.imu_stats.bus;
    built = true;
  }
  if (!full) return;
  const p = S.pi;
  $('sPiTemp').textContent = has(p.temp) ? p.temp.toFixed(1) + '°C' : '?';
  $('sPiTemp').className = 'big ' + cls(p.temp, S.limits.pi_temp - 10, S.limits.pi_temp);
  $('sPiLim').textContent = S.limits.pi_temp;
  const w = $('sPower');
  [w.textContent, w.className] = !has(p.undervolt) ? ['?', ''] : p.undervolt ? ['Too low!', 'bad'] : p.throttled ? ['Throttled', 'bad']
    : p.undervolt_before ? ['Dipped earlier', 'warn'] : ['OK', 'ok'];
  const pct = v => has(v) ? v.toFixed(0) + '%' : '?';
  $('sPi').innerHTML = `
    <div>CPU<b class="${cls(p.cpu, 70, 90)}">${pct(p.cpu)}</b>load ${n1(p.load, 2)}</div>
    <div>Dashboard<b>${pct(p.self_cpu)}</b>of one core</div>
    <div>Memory<b>${p.mem_used_mb ? (p.mem_used_mb / 1024).toFixed(1) : '?'} GB</b>of ${p.mem_total_mb ? (p.mem_total_mb / 1024).toFixed(1) : '?'} GB</div>
    <div>CPU clock<b>${p.arm_mhz ? p.arm_mhz.toFixed(0) : '?'} MHz</b></div>
    <div>GPU clock<b>${p.gpu_mhz ? p.gpu_mhz.toFixed(0) : '?'} MHz</b></div>
    <div style="grid-column:1/-1">Addresses (ssh iva585@…)<b style="font-size:13px;word-break:break-all">${(p.addresses || []).map(esc).join(' · ') || '?'}</b></div>
    <div>Wi-Fi<b class="${!has(p.wifi_dbm) ? '' : p.wifi_dbm < -75 ? 'bad' : p.wifi_dbm < -65 ? 'warn' : ''}">${has(p.wifi_dbm) ? p.wifi_dbm.toFixed(0) + ' dBm' : '?'}</b></div>`;
  $('sCores').innerHTML = (p.cores || []).map((c, i) => `<div title="core ${i}: ${c.toFixed(0)}%"><i style="height:${Math.max(2, c).toFixed(0)}%"></i></div>`).join('');
  $('sBuses').innerHTML = S.buses.map(bus => {
    const c = bus.pass_ms > 18 ? 'bad' : bus.pass_ms > 15 ? 'warn' : '';
    return `<div>${esc(bus.port)} (servos ${bus.ids.join(', ')})<b class="${c}">${n1(bus.pass_ms)} ms</b>max ${n1(bus.pass_max)} · fail ${n1(bus.fail_pct)}%</div>`;
  }).join('') || '<div>No servo bus found</div>';
  $('sServos').innerHTML = Object.values(S.joints).map(j => `<tr><td>${j.id}</td>
    <td class="${cls(j.temp, S.limits.servo_temp - 10, S.limits.servo_temp)}">${has(j.temp) ? j.temp + '°C' : '?'}</td>
    <td>${j.vin ? j.vin.toFixed(2) + ' V' : '?'}</td><td>${j.torque ? 'on' : j.torque === false ? 'off' : '?'}</td></tr>`).join('');
  const P = S.personality, V = P.voice;
  $('sExtras').innerHTML = `
    <div>Audio<b>${V.backend ? esc(V.backend) + (V.tts ? ' + ' + esc(V.tts) : '') : '<span class="bad">none</span>'}</b>${V.error ? `<span class="warn">${esc(V.error)}</span>` : ''}</div>
    <div>Faces<b>${P.greet_faces ? 'on' : 'off'}</b>${P.faces_error ? `<span class="warn">${esc(P.faces_error)}</span>` : P.faces && P.faces.faces ? 'sees ' + P.faces.faces : ''}</div>
    <div>Voice commands<b>${P.listening ? 'on' : 'off'}</b>${P.listener_error ? `<span class="warn">${esc(P.listener_error)}</span>` : ''}</div>
    <div>Camera<b>${S.camera.available ? 'available' : 'no'}</b>${S.camera.error ? esc(S.camera.error) : ''}</div>
    <div>Stabilization<b>${S.camera.stabilization_configured ? 'configured' : 'off'}</b>${S.camera.stabilization_error ? esc(S.camera.stabilization_error) : ''}</div>`;
});

function imu(S) {
  const m = S.imu, st = S.imu_stats || {};
  if (!m) {
    $('sLean').textContent = '-';
    $('sImuNote').innerHTML = `<span class="bad">${esc(S.imu_error || 'No reading')}</span>`;
    $('sHorizon').innerHTML = ''; $('sImu').innerHTML = '';
    return;
  }
  $('sLean').textContent = `${n1(m.lean_forward)}° fwd, ${n1(m.lean_right)}° right`;
  const tilt = Math.hypot(m.lean_forward, m.lean_right);
  $('sLean').className = 'big ' + (tilt > 30 ? 'bad' : tilt > 10 ? 'warn' : '');
  const roll = Math.max(-60, Math.min(60, m.lean_right)), pitch = Math.max(-45, Math.min(45, m.lean_forward));
  $('sHorizon').innerHTML = `<g transform="translate(110 50) rotate(${-roll}) translate(0 ${-pitch * 1.2})">
      <rect x="-200" y="0" width="400" height="200" fill="var(--band)"/>
      <line x1="-200" x2="200" y1="0" y2="0" stroke="var(--accent)" stroke-width="2"/></g>
    <line x1="80" x2="140" y1="50" y2="50" stroke="var(--text)" stroke-width="3"/><circle cx="110" cy="50" r="3" fill="var(--text)"/>`;
  $('sImu').innerHTML = `
    <div>Gyro x<b>${n1(m.gyro[0], 2)}</b>rad/s</div><div>Gyro y<b>${n1(m.gyro[1], 2)}</b>rad/s</div><div>Gyro z<b>${n1(m.gyro[2], 2)}</b>rad/s</div>
    <div>Accel<b>${n1(m.accel_g, 2)} g</b>${m.biasing ? '<span class="warn">zeroing gyro...</span>' : ''}</div>
    <div>Reads<b class="${st.fail_pct > 1 ? 'bad' : ''}">${n1(st.fail_pct)}% bad</b>${n1(st.read_ms)} ms each</div>
    <div>Vibration<b class="${m.gyro_raw_rms > 0.3 ? 'warn' : ''}">${n1(m.gyro_raw_rms, 2)}</b>rad/s raw</div>`;
  $('sImuNote').textContent = `${m.device}, I2C bus ${st.bus}, axes ${(st.axes || []).join(', ')}, heading ${n1(m.yaw * 180 / Math.PI, 0)}°` + (m.age > 0.5 ? ' · reading is stale' : '');
}
