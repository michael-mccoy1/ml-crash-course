"use strict";
/* Shared by the 3D shape steps. The shape is a DNA-style double helix: two spiral strands joined by rungs.
   The surface a model has learned is found by walking out from the true shape along its normals until the model's prediction changes sign. */
var SH = { R: 0.5, r: 0.16, c: 0.27, tm: 6, rr: 0.07, rz0: -1.5, rdz: 0.2, nr: 16, box: [0.8, 0.8, 1.8], clip: 0.25, NU: 64, NV: 5, tlo: -0.9, tspan: 4.8 };
/* offsets from shMesh are saved as bytes: 0 to 254 cover tlo to tlo + tspan, 255 means no surface */
function shQuant(t){ var q = new Uint8Array(t.length); for (var i = 0; i < t.length; i++) q[i] = isNaN(t[i]) ? 255 : Math.round(Math.max(0, Math.min(1, (t[i] - SH.tlo) / SH.tspan)) * 254); return q; }
function shDecode(b64, n){ var bin = atob(b64), t = new Float32Array(n); for (var i = 0; i < n; i++) { var q = bin.charCodeAt(i); t[i] = q === 255 ? NaN : q / 254 * SH.tspan + SH.tlo; } return t; }
function shHelixDist(x, y, z, ph){
  var c = SH.c, R = SH.R, th = Math.atan2(y, x) - ph, t = th + 6.28318530718 * Math.round((z / c - th) / 6.28318530718); t = Math.max(-SH.tm, Math.min(SH.tm, t));
  for (var k = 0; k < 4; k++) { var a = t + ph, ca = Math.cos(a), sa = Math.sin(a), g = R * (x * sa - y * ca) - c * z + c * c * t, h = R * (x * ca + y * sa) + c * c; if (h > 0.01) t -= g / h; t = Math.max(-SH.tm, Math.min(SH.tm, t)); }
  var a2 = t + ph, dx = x - R * Math.cos(a2), dy = y - R * Math.sin(a2), dz = z - c * t; return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
function shRungDist(x, y, z){
  var best = 1e9, k0 = Math.round((z - SH.rz0) / SH.rdz);
  for (var k = k0 - 1; k <= k0 + 1; k++) { if (k < 0 || k >= SH.nr) continue; var zk = SH.rz0 + k * SH.rdz, a = zk / SH.c, ux = Math.cos(a), uy = Math.sin(a), s = Math.max(-SH.R, Math.min(SH.R, x * ux + y * uy)), dx = x - s * ux, dy = y - s * uy, dz = z - zk, d = Math.sqrt(dx * dx + dy * dy + dz * dz); if (d < best) best = d; }
  return best;
}
/* signed distance to the surface of the whole double helix: negative inside, positive outside */
function shSD(x, y, z){ return Math.min(shHelixDist(x, y, z, 0) - SH.r, shHelixDist(x, y, z, Math.PI) - SH.r, shRungDist(x, y, z) - SH.rr); }
/* a third of the points anywhere in the box, a third inside the helix, a third within 0.3 outside it */
function shTrain(n, seed){
  var rnd = rng(seed), X = new Float32Array(n * 3), T = new Float32Array(n), k = 0, b = SH.box;
  while (k < n) { var x = (rnd() * 2 - 1) * b[0], y = (rnd() * 2 - 1) * b[1], z = (rnd() * 2 - 1) * b[2], d = shSD(x, y, z), m3 = k % 3; if (m3 === 0 || (m3 === 1 && d < 0) || (m3 === 2 && d >= 0 && d < 0.3)) { X[k * 3] = x; X[k * 3 + 1] = y; X[k * 3 + 2] = z; T[k] = clamp(d, -SH.clip, SH.clip); k++; } }
  return { X: X, T: T, n: n };
}
/* held-out points within 0.25 of the surface, half inside and half outside */
function shTest(n, seed){
  var rnd = rng(seed), X = new Float32Array(n * 3), inside = new Uint8Array(n), k = 0, ni = 0, no = 0, b = SH.box;
  while (k < n) { var x = (rnd() * 2 - 1) * b[0], y = (rnd() * 2 - 1) * b[1], z = (rnd() * 2 - 1) * b[2], d = shSD(x, y, z); if (Math.abs(d) > 0.25) continue; var ins = d < 0 ? 1 : 0; if (ins ? ni >= n / 2 : no >= n / 2) continue; X[k * 3] = x; X[k * 3 + 1] = y; X[k * 3 + 2] = z; inside[k] = ins; if (ins) ni++; else no++; k++; }
  return { X: X, inside: inside, n: n };
}
/* balanced accuracy: the average of the share of inside points called inside and outside points called outside */
function shAccuracy(test, predict){
  var tp = 0, tn = 0, pos = 0, neg = 0;
  for (var i = 0; i < test.n; i++) { var p = predict(test.X[i * 3], test.X[i * 3 + 1], test.X[i * 3 + 2]) < 0 ? 1 : 0; if (test.inside[i]) { pos++; tp += p; } else { neg++; tn += 1 - p; } }
  return (tp / pos + tn / neg) / 2;
}
/* the true surface as patches: two strands and the rungs. Each patch is a grid of points with outward normals. */
function shGeom(){
  if (shGeom.g) return shGeom.g; var patches = [], nu = SH.NU, nv = SH.NV, i, j;
  [0, Math.PI].forEach(function(phase){
    var S = new Float32Array(nu * nv * 3), N = new Float32Array(nu * nv * 3);
    for (i = 0; i < nu; i++) { var t = -SH.tm + 2 * SH.tm * i / (nu - 1), a = t + phase, ca = Math.cos(a), sa = Math.sin(a), L = Math.sqrt(SH.R * SH.R + SH.c * SH.c), Tx = -SH.R * sa / L, Ty = SH.R * ca / L, Tz = SH.c / L, nx = -ca, ny = -sa, nz = 0, bx = Ty * nz - Tz * ny, by = Tz * nx - Tx * nz, bz = Tx * ny - Ty * nx;
      for (j = 0; j < nv; j++) { var v = j / nv * 6.28318530718, cv_ = Math.cos(v), sv = Math.sin(v), k = (i * nv + j) * 3, ox = cv_ * nx + sv * bx, oy = cv_ * ny + sv * by, oz = cv_ * nz + sv * bz; S[k] = SH.R * ca + SH.r * ox; S[k + 1] = SH.R * sa + SH.r * oy; S[k + 2] = SH.c * t + SH.r * oz; N[k] = ox; N[k + 1] = oy; N[k + 2] = oz; } }
    patches.push({ S: S, N: N, nu: nu, nv: nv, r: SH.r });
  });
  var ru = 6, rv = 3;
  for (var q = 0; q < SH.nr; q++) {
    var zk = SH.rz0 + q * SH.rdz, a3 = zk / SH.c, ux = Math.cos(a3), uy = Math.sin(a3), S2 = new Float32Array(ru * rv * 3), N2 = new Float32Array(ru * rv * 3), e1x = -uy, e1y = ux;
    for (i = 0; i < ru; i++) { var s = -(SH.R - SH.r) + 2 * (SH.R - SH.r) * i / (ru - 1);
      for (j = 0; j < rv; j++) { var v2 = j / rv * 6.28318530718, c2 = Math.cos(v2), s2 = Math.sin(v2), k2 = (i * rv + j) * 3; N2[k2] = c2 * e1x; N2[k2 + 1] = c2 * e1y; N2[k2 + 2] = s2; S2[k2] = s * ux + SH.rr * N2[k2]; S2[k2 + 1] = s * uy + SH.rr * N2[k2 + 1]; S2[k2 + 2] = zk + SH.rr * s2; } }
    patches.push({ S: S2, N: N2, nu: ru, nv: rv, r: SH.rr });
  }
  var total = 0; patches.forEach(function(p){ p.off = total; total += p.nu * p.nv; });
  return (shGeom.g = { patches: patches, total: total });
}
/* for every grid point of the true shape, how far along the normal (in units of that patch's radius) the model's surface sits. NaN where the model has no inside there */
function shMesh(predict){
  var g = shGeom(), out = new Float32Array(g.total);
  g.patches.forEach(function(p){
    var n = p.nu * p.nv, r = p.r;
    for (var k = 0; k < n; k++) {
      var sx = p.S[k * 3], sy = p.S[k * 3 + 1], sz = p.S[k * 3 + 2], nx = p.N[k * 3], ny = p.N[k * 3 + 1], nz = p.N[k * 3 + 2], t0 = SH.tlo * r, dt = 0.4 * r;
      if (predict(sx + nx * t0, sy + ny * t0, sz + nz * t0) >= 0) { out[p.off + k] = NaN; continue; }
      var found = false;
      for (var s = 1; s <= 12; s++) { var t2 = t0 + s * dt; if (predict(sx + nx * t2, sy + ny * t2, sz + nz * t2) >= 0) { var lo = t2 - dt, hi = t2; for (var it = 0; it < 5; it++) { var mid = (lo + hi) / 2; if (predict(sx + nx * mid, sy + ny * mid, sz + nz * mid) < 0) lo = mid; else hi = mid; } out[p.off + k] = (lo + hi) / 2 / r; found = true; break; } }
      if (!found) out[p.off + k] = NaN;
    }
  });
  return out;
}
/* Animated view of several models drawn as nodes. cfg: { canvas, chart, panels: [{ short, color, ck: [{ t, acc, x }], titleAt }], x0, x1, xl, dur }.
   Each node starts at a scattered position. At a checkpoint it sits on the model's surface (offset t, see shMesh) or, where the model has no surface yet, stays scattered.
   The grey dots are the true helix. prog runs from 0 to 1. */
function shAnim(cfg){
  var G = shGeom(), N = G.total, st = { prog: 1, playing: false, yaw: 0.5, pitch: 0.35, drag: false }, alive = true, last = 0, dur = cfg.dur || 14, canvas = cfg.canvas;
  var P = new Float32Array(N * 3), AL = new Float32Array(N), SX = new Float32Array(N), SY = new Float32Array(N), SD = new Float32Array(N), RX = new Float32Array(N), RY = new Float32Array(N), HS = new Float32Array(N), SC = new Float32Array(N * 3), BK = new Uint8Array(N), lx = 0, ly = 0, listeners = [];
  var rnd = rng(7), b = SH.box, q; for (q = 0; q < N; q++) { HS[q] = rnd(); SC[q * 3] = (rnd() * 2 - 1) * b[0]; SC[q * 3 + 1] = (rnd() * 2 - 1) * b[1]; SC[q * 3 + 2] = (rnd() * 2 - 1) * b[2]; }
  var TP = {};
  /* the training points inside the helix: a third of the training set, at most 6,000 of them drawn */
  function trainIn(n){
    if (TP[n]) return TP[n]; var tr = shTrain(Math.min(n, 20000), 31), want = Math.min(Math.round(n / 3), 6000), o = [];
    for (var i = 0; i < tr.n && o.length < want * 3; i++) if (tr.T[i] < 0) { o.push(tr.X[i * 3], tr.X[i * 3 + 1], tr.X[i * 3 + 2]); }
    return (TP[n] = new Float32Array(o.slice(0, want * 3)));
  }
  function ease(a){ return a * a * (3 - 2 * a); }
  function at(panel, prog){ var K = panel.ck.length, f = clamp(prog, 0, 1) * (K - 1), i0 = Math.min(K - 1, Math.floor(f)), i1 = Math.min(K - 1, i0 + 1), a = ease(f - i0); return { i0: i0, i1: i1, a: a }; }
  function build(panel, ps){
    var c0 = panel.ck[ps.i0], c1 = panel.ck[ps.i1], a = ps.a;
    G.patches.forEach(function(p){
      var n = p.nu * p.nv, k;
      for (k = 0; k < n; k++) {
        var gi = p.off + k, t0 = c0.t[gi], t1 = c1.t[gi], n0 = isNaN(t0), n1 = isNaN(t1), x0, y0, z0, x1, y1, z1;
        if (n0) { x0 = SC[gi * 3]; y0 = SC[gi * 3 + 1]; z0 = SC[gi * 3 + 2]; } else { x0 = p.S[k * 3] + p.N[k * 3] * t0 * p.r; y0 = p.S[k * 3 + 1] + p.N[k * 3 + 1] * t0 * p.r; z0 = p.S[k * 3 + 2] + p.N[k * 3 + 2] * t0 * p.r; }
        if (n1) { x1 = SC[gi * 3]; y1 = SC[gi * 3 + 1]; z1 = SC[gi * 3 + 2]; } else { x1 = p.S[k * 3] + p.N[k * 3] * t1 * p.r; y1 = p.S[k * 3 + 1] + p.N[k * 3 + 1] * t1 * p.r; z1 = p.S[k * 3 + 2] + p.N[k * 3 + 2] * t1 * p.r; }
        var ai = clamp(a * 1.6 - 0.6 * HS[gi], 0, 1); P[gi * 3] = x0 + (x1 - x0) * ai; P[gi * 3 + 1] = y0 + (y1 - y0) * ai; P[gi * 3 + 2] = z0 + (z1 - z0) * ai; AL[gi] = (n0 ? 0 : 1) + ((n1 ? 0 : 1) - (n0 ? 0 : 1)) * ai;
      }
    });
  }
  var FILL = ['#7A0E08', '#BD120A', '#D9534D', '#EE9993'];
  function drawPanel(x, ox, oy, S, panel, ps){
    build(panel, ps);
    var cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch), k;
    if (panel.n) { var tp = trainIn(panel.n), tn = tp.length / 3; x.fillStyle = 'rgba(35,42,52,0.55)'; for (k = 0; k < tn; k++) { var qa = tp[k * 3] * cy - tp[k * 3 + 1] * sy, qb = tp[k * 3] * sy + tp[k * 3 + 1] * cy; x.fillRect(ox + qa * S - 0.6, oy - (qb * sp + tp[k * 3 + 2] * cp) * S - 0.6, 1.2, 1.2); } }
    for (k = 0; k < N; k++) {
      var px2 = P[k * 3], py2 = P[k * 3 + 1], pz2 = P[k * 3 + 2], a2 = px2 * cy - py2 * sy, b2 = px2 * sy + py2 * cy, d = b2 * cp - pz2 * sp, e2 = b2 * sp + pz2 * cp; SX[k] = ox + a2 * S; SY[k] = oy - e2 * S; SD[k] = d;
      BK[k] = AL[k] < 0.5 ? 4 : clamp(Math.floor((d + 1.2) / 2.4 * 4), 0, 3);
    }
    var rad = Math.max(1.5, S / 40);
    for (var bk = 0; bk < 5; bk++) {
      x.fillStyle = bk === 4 ? 'rgba(189,18,10,0.3)' : FILL[bk]; var r2 = bk === 4 ? rad * 0.8 : rad; x.beginPath();
      for (k = 0; k < N; k++) if (BK[k] === bk) { x.moveTo(SX[k] + r2, SY[k]); x.arc(SX[k], SY[k], r2, 0, 6.2832); }
      x.fill();
    }
    function pr(X, Y, Z){ var a = X * cy - Y * sy, bb = X * sy + Y * cy; return [ox + a * S, oy - (bb * sp + Z * cp) * S]; }
    x.strokeStyle = 'rgba(70,86,104,0.85)'; x.lineWidth = 1.6; x.lineJoin = 'round';
    [0, Math.PI].forEach(function(ph){ x.beginPath(); for (var i = 0; i <= 90; i++) { var t = -SH.tm + 2 * SH.tm * i / 90, q = pr(SH.R * Math.cos(t + ph), SH.R * Math.sin(t + ph), SH.c * t); if (i) x.lineTo(q[0], q[1]); else x.moveTo(q[0], q[1]); } x.stroke(); });
    x.lineWidth = 1.2; x.beginPath(); for (var rk = 0; rk < SH.nr; rk++) { var zk = SH.rz0 + rk * SH.rdz, ak = zk / SH.c, c1 = Math.cos(ak) * SH.R, s1 = Math.sin(ak) * SH.R, q1 = pr(c1, s1, zk), q2 = pr(-c1, -s1, zk); x.moveTo(q1[0], q1[1]); x.lineTo(q2[0], q2[1]); } x.stroke();
  }
  function drawChart(){
    if (!cfg.chart) return; var series = cfg.panels.map(function(p){ var ps = at(p, st.prog), pts = []; for (var i = 0; i <= ps.i0; i++) pts.push([p.ck[i].x, p.ck[i].acc]); if (ps.a > 0 && ps.i1 !== ps.i0) pts.push([p.ck[ps.i0].x + (p.ck[ps.i1].x - p.ck[ps.i0].x) * ps.a, p.ck[ps.i0].acc + (p.ck[ps.i1].acc - p.ck[ps.i0].acc) * ps.a]); return { pts: pts, color: p.color, w: 2.8, label: p.short }; });
    lineChart(cfg.chart, series, { xl: cfg.xl, yl: 'accuracy', y0: 0.5, y1: 1, x0: cfg.x0, x1: cfg.x1, xticks: cfg.xticks, xdec: cfg.xdec });
  }
  function draw(){
    var g = cv(canvas), x = g.x, n = cfg.panels.length, w = g.w / n, S = Math.min(w / 1.5, (g.h - 64) / 2 / 1.9);
    cfg.panels.forEach(function(p, i){
      var ps = at(p, st.prog); x.save(); x.beginPath(); x.rect(i * w, 0, w, g.h); x.clip(); drawPanel(x, i * w + w / 2, g.h / 2 + 12, S, p, ps); x.restore();
      var acc = p.ck[ps.i0].acc + (p.ck[ps.i1].acc - p.ck[ps.i0].acc) * ps.a; x.fillStyle = C.char; x.textAlign = 'left'; x.font = '700 13px Aptos, Segoe UI, sans-serif'; x.fillText(p.titleAt(ps.a >= 0.5 ? ps.i1 : ps.i0), i * w + 10, 18);
      var trn = p.ck[ps.i0].tr !== undefined ? p.ck[ps.i0].tr + (p.ck[ps.i1].tr - p.ck[ps.i0].tr) * ps.a : null; x.font = '12px Aptos, Segoe UI, sans-serif'; x.fillStyle = p.color; x.fillText(trn === null ? 'accuracy ' + acc.toFixed(2) : 'test ' + acc.toFixed(2) + ', train ' + trn.toFixed(2), i * w + 10, 35);
    });
    x.font = '11px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; var l1 = 'grey lines: the true helix', l2 = 'red dots: nodes placed on the surface the model has learned', l3 = 'black specks: training points inside the helix'; x.fillStyle = '#6F8196'; x.fillText(l1, 10, g.h - 6); x.fillStyle = '#BD120A'; x.fillText(l2, 24 + x.measureText(l1).width, g.h - 6); x.fillStyle = '#232A34'; x.textAlign = 'right'; x.fillText(l3, g.w - 8, g.h - 6);
    drawChart();
  }
  function visible(){ var r = canvas.getBoundingClientRect(); return r.bottom > 0 && r.top < window.innerHeight && r.width > 0; }
  function frame(now){
    if (!alive) return; requestAnimationFrame(frame); if (!visible()) { last = now; return; } var dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (st.playing) { st.prog = Math.min(1, st.prog + dt / dur); if (st.prog >= 1) st.playing = false; emit(); }
    if (!st.drag) st.yaw += dt * 0.5; draw();
  }
  function emit(){ listeners.forEach(function(f){ f(st); }); }
  canvas.addEventListener('pointerdown', function(e){ st.drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', function(e){ if (!st.drag) return; st.yaw += (e.clientX - lx) * 0.01; st.pitch = clamp(st.pitch + (e.clientY - ly) * 0.01, -1.5, 1.5); lx = e.clientX; ly = e.clientY; });
  canvas.addEventListener('pointerup', function(){ st.drag = false; });
  requestAnimationFrame(function(t){ last = t; frame(t); });
  return {
    st: st, draw: draw, onChange: function(f){ listeners.push(f); }, setPanels: function(p, x0, x1){ cfg.panels = p; if (x0 !== undefined) cfg.x0 = x0; if (x1 !== undefined) cfg.x1 = x1; draw(); },
    play: function(){ if (st.prog >= 1) st.prog = 0; st.playing = true; emit(); }, pause: function(){ st.playing = false; emit(); },
    restart: function(){ st.prog = 0; st.playing = true; emit(); }, setProg: function(p){ st.prog = clamp(p, 0, 1); st.playing = false; emit(); }, stop: function(){ alive = false; }
  };
}
/* Play, restart and a progress slider for an animation made by shAnim */
function shControls(api, anim){
  var w = api.addControl('<button type="button" class="btn sm" data-pp>Play</button> <button type="button" class="btn sm ghost" data-rs>Restart</button> Training progress <input class="pg" type="range" min="0" max="1000" step="1" value="1000" style="width:260px">');
  var pb = qs('[data-pp]', w), rs = qs('[data-rs]', w), pg = qs('.pg', w);
  function sync(st){ pb.textContent = st.playing ? 'Pause' : 'Play'; pg.value = Math.round(st.prog * 1000); }
  anim.onChange(sync); pb.addEventListener('click', function(){ if (anim.st.playing) anim.pause(); else anim.play(); }); rs.addEventListener('click', function(){ anim.restart(); }); pg.addEventListener('input', function(){ anim.setProg(pg.value / 1000); }); sync(anim.st);
}
