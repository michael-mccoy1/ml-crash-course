"use strict";
/* Shared by the torus steps. A torus, held-out points, and a smooth shaded surface that morphs as a model trains.
   The learned surface is found by walking out from the true torus along its normal until the model's prediction changes sign. */
var SH = { R: 1, r: 0.4, L: 1.7, NU: 64, NV: 24 };
/* signed distance to the torus surface: negative inside, positive outside */
function shSD(x, y, z){ var q = Math.sqrt(x * x + y * y) - SH.R; return Math.sqrt(q * q + z * z) - SH.r; }
/* half of the points anywhere in the cube, half within 0.6 of the surface */
function shTrain(n, seed){
  var rnd = rng(seed), X = new Float32Array(n * 3), T = new Float32Array(n), k = 0;
  while (k < n) { var x = (rnd() * 2 - 1) * SH.L, y = (rnd() * 2 - 1) * SH.L, z = (rnd() * 2 - 1) * SH.L, d = shSD(x, y, z); if (k % 2 === 0 || Math.abs(d) < 0.6) { X[k * 3] = x; X[k * 3 + 1] = y; X[k * 3 + 2] = z; T[k] = clamp(d, -0.4, 0.4); k++; } }
  return { X: X, T: T, n: n };
}
/* held-out points within 0.5 of the surface, half inside and half outside */
function shTest(n, seed){
  var rnd = rng(seed), X = new Float32Array(n * 3), inside = new Uint8Array(n), k = 0, ni = 0, no = 0;
  while (k < n) { var x = (rnd() * 2 - 1) * SH.L, y = (rnd() * 2 - 1) * SH.L, z = (rnd() * 2 - 1) * SH.L, d = shSD(x, y, z); if (Math.abs(d) > 0.5) continue; var ins = d < 0 ? 1 : 0; if (ins ? ni >= n / 2 : no >= n / 2) continue; X[k * 3] = x; X[k * 3 + 1] = y; X[k * 3 + 2] = z; inside[k] = ins; if (ins) ni++; else no++; k++; }
  return { X: X, inside: inside, n: n };
}
/* balanced accuracy: the average of the share of inside points called inside and outside points called outside */
function shAccuracy(test, predict){
  var tp = 0, tn = 0, pos = 0, neg = 0;
  for (var i = 0; i < test.n; i++) { var p = predict(test.X[i * 3], test.X[i * 3 + 1], test.X[i * 3 + 2]) < 0 ? 1 : 0; if (test.inside[i]) { pos++; tp += p; } else { neg++; tn += 1 - p; } }
  return (tp / pos + tn / neg) / 2;
}
/* a grid over the torus: positions and outward normals */
function shGeom(){
  if (shGeom.g) return shGeom.g; var nu = SH.NU, nv = SH.NV, N = nu * nv, S = new Float32Array(N * 3), Nn = new Float32Array(N * 3);
  for (var i = 0; i < nu; i++) for (var j = 0; j < nv; j++) { var a = i / nu * 6.28318530718, b = j / nv * 6.28318530718, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), k = (i * nv + j) * 3; S[k] = (SH.R + SH.r * cb) * ca; S[k + 1] = (SH.R + SH.r * cb) * sa; S[k + 2] = SH.r * sb; Nn[k] = cb * ca; Nn[k + 1] = cb * sa; Nn[k + 2] = sb; }
  return (shGeom.g = { S: S, N: Nn, nu: nu, nv: nv });
}
/* for each grid point, how far along the normal the model's surface sits. NaN where the model has no inside there */
function shMesh(predict){
  var g = shGeom(), n = g.nu * g.nv, out = new Float32Array(n);
  for (var k = 0; k < n; k++) {
    var sx = g.S[k * 3], sy = g.S[k * 3 + 1], sz = g.S[k * 3 + 2], nx = g.N[k * 3], ny = g.N[k * 3 + 1], nz = g.N[k * 3 + 2];
    if (predict(sx - 0.3 * nx, sy - 0.3 * ny, sz - 0.3 * nz) >= 0) { out[k] = NaN; continue; }
    var found = false, dt = 0.08;
    for (var s = 1; s <= 10; s++) { var t2 = -0.3 + s * dt; if (predict(sx + nx * t2, sy + ny * t2, sz + nz * t2) >= 0) { var lo = t2 - dt, hi = t2; for (var it = 0; it < 5; it++) { var mid = (lo + hi) / 2; if (predict(sx + nx * mid, sy + ny * mid, sz + nz * mid) < 0) lo = mid; else hi = mid; } out[k] = (lo + hi) / 2; found = true; break; } }
    if (!found) out[k] = NaN;
  }
  return out;
}
/* Animated view of several models. cfg: { canvas, chart, panels: [{ title, color, ck: [{ t, acc, x }] }], x0, x1, xl, dur }.
   Each ck is a checkpoint: the surface offsets t, the accuracy, and the training time x. prog runs from 0 to 1. */
function shAnim(cfg){
  var G = shGeom(), nu = G.nu, nv = G.nv, N = nu * nv, st = { prog: 1, playing: false, yaw: 0.5, pitch: 1.0, drag: false }, alive = true, last = 0, dur = cfg.dur || 14, canvas = cfg.canvas;
  var P = new Float32Array(N * 3), V = new Float32Array(N * 3), AL = new Float32Array(N), SX = new Float32Array(N), SY = new Float32Array(N), SD = new Float32Array(N), SH_ = new Float32Array(N), order = [], lx = 0, ly = 0, listeners = [];
  for (var q = 0; q < N; q++) order.push(q);
  function ease(a){ return a * a * (3 - 2 * a); }
  function at(panel, prog){ var K = panel.ck.length, f = clamp(prog, 0, 1) * (K - 1), i0 = Math.min(K - 1, Math.floor(f)), i1 = Math.min(K - 1, i0 + 1), a = ease(f - i0); return { i0: i0, i1: i1, a: a }; }
  function build(panel, ps){
    var c0 = panel.ck[ps.i0], c1 = panel.ck[ps.i1], a = ps.a, k;
    for (k = 0; k < N; k++) { var t0 = c0.t[k], t1 = c1.t[k], a0 = isNaN(t0) ? 0.25 : 1, a1 = isNaN(t1) ? 0.25 : 1; if (isNaN(t0)) t0 = -0.36; if (isNaN(t1)) t1 = -0.36; var t = t0 + (t1 - t0) * a; AL[k] = a0 + (a1 - a0) * a; P[k * 3] = G.S[k * 3] + G.N[k * 3] * t; P[k * 3 + 1] = G.S[k * 3 + 1] + G.N[k * 3 + 1] * t; P[k * 3 + 2] = G.S[k * 3 + 2] + G.N[k * 3 + 2] * t; }
    for (var i = 0; i < nu; i++) for (var j = 0; j < nv; j++) {
      var ip = ((i + 1) % nu) * nv + j, im = ((i + nu - 1) % nu) * nv + j, jp = i * nv + (j + 1) % nv, jm = i * nv + (j + nv - 1) % nv, o = (i * nv + j) * 3;
      var ux = P[ip * 3] - P[im * 3], uy = P[ip * 3 + 1] - P[im * 3 + 1], uz = P[ip * 3 + 2] - P[im * 3 + 2], vx = P[jp * 3] - P[jm * 3], vy = P[jp * 3 + 1] - P[jm * 3 + 1], vz = P[jp * 3 + 2] - P[jm * 3 + 2];
      var nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, ln = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1; V[o] = nx / ln; V[o + 1] = ny / ln; V[o + 2] = nz / ln;
    }
  }
  function drawPanel(x, ox, oy, S, panel, ps){
    build(panel, ps);
    var cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch), k;
    for (k = 0; k < N; k++) {
      var px = P[k * 3], py = P[k * 3 + 1], pz = P[k * 3 + 2], a = px * cy - py * sy, b = px * sy + py * cy, d = b * cp - pz * sp, e = b * sp + pz * cp; SX[k] = ox + a * S; SY[k] = oy - e * S; SD[k] = d;
      var nx = V[k * 3], ny = V[k * 3 + 1], nz = V[k * 3 + 2], na = nx * cy - ny * sy, nb = nx * sy + ny * cy, nd = nb * cp - nz * sp, ne = nb * sp + nz * cp;
      var lam = Math.abs(na * -0.35 + ne * 0.6 + nd * 0.72); SH_[k] = 0.3 + 0.7 * lam;
    }
    /* faint rings of the true torus for reference */
    x.strokeStyle = 'rgba(52,55,63,.22)'; x.lineWidth = 0.8;
    for (var j0 = 0; j0 < nv; j0 += 6) { x.beginPath(); for (var i0 = 0; i0 <= nu; i0++) { var gk = (i0 % nu) * nv + j0, gx = G.S[gk * 3], gy = G.S[gk * 3 + 1], gz = G.S[gk * 3 + 2], ga = gx * cy - gy * sy, gb = gx * sy + gy * cy, ge = gb * sp + gz * cp; var X0 = ox + ga * S, Y0 = oy - ge * S; i0 ? x.lineTo(X0, Y0) : x.moveTo(X0, Y0); } x.stroke(); }
    var keys = new Float32Array(N); for (var qd = 0; qd < N; qd++) { var ii = Math.floor(qd / nv), jj = qd % nv, a1 = ((ii + 1) % nu) * nv + jj, a2 = ((ii + 1) % nu) * nv + (jj + 1) % nv, a3 = ii * nv + (jj + 1) % nv; keys[qd] = SD[qd] + SD[a1] + SD[a2] + SD[a3]; }
    order.sort(function(p1, p2){ return keys[p2] - keys[p1]; });
    for (var oi = 0; oi < N; oi++) {
      var qq = order[oi], i = Math.floor(qq / nv), j = qq % nv, v0 = qq, v1 = ((i + 1) % nu) * nv + j, v2 = ((i + 1) % nu) * nv + (j + 1) % nv, v3 = i * nv + (j + 1) % nv, sh = (SH_[v0] + SH_[v1] + SH_[v2] + SH_[v3]) / 4, al = (AL[v0] + AL[v1] + AL[v2] + AL[v3]) / 4;
      var r = Math.round(137 + (247 - 137) * sh * sh), gg = Math.round(21 + (140 - 21) * sh * sh), bb = Math.round(21 + (133 - 21) * sh * sh);
      x.fillStyle = 'rgba(' + r + ',' + gg + ',' + bb + ',' + al.toFixed(2) + ')'; x.strokeStyle = x.fillStyle; x.lineWidth = 0.7;
      x.beginPath(); x.moveTo(SX[v0], SY[v0]); x.lineTo(SX[v1], SY[v1]); x.lineTo(SX[v2], SY[v2]); x.lineTo(SX[v3], SY[v3]); x.closePath(); x.fill(); x.stroke();
    }
  }
  function drawChart(){
    if (!cfg.chart) return; var series = cfg.panels.map(function(p){ var ps = at(p, st.prog), pts = []; for (var i = 0; i <= ps.i0; i++) pts.push([p.ck[i].x, p.ck[i].acc]); if (ps.a > 0 && ps.i1 !== ps.i0) pts.push([p.ck[ps.i0].x + (p.ck[ps.i1].x - p.ck[ps.i0].x) * ps.a, p.ck[ps.i0].acc + (p.ck[ps.i1].acc - p.ck[ps.i0].acc) * ps.a]); return { pts: pts, color: p.color, w: 2.8, label: p.short }; });
    lineChart(cfg.chart, series, { xl: cfg.xl, yl: 'accuracy', y0: 0.5, y1: 1, x0: cfg.x0, x1: cfg.x1, ydec: 2, xdec: cfg.xdec === undefined ? 1 : cfg.xdec });
  }
  function draw(){
    var g = cv(canvas), x = g.x, n = cfg.panels.length, w = g.w / n, S = Math.min(w, g.h) / 3.9;
    cfg.panels.forEach(function(p, i){
      var ps = at(p, st.prog); x.save(); x.beginPath(); x.rect(i * w, 0, w, g.h); x.clip(); drawPanel(x, i * w + w / 2, g.h / 2 + 16, S, p, ps); x.restore();
      var acc = p.ck[ps.i0].acc + (p.ck[ps.i1].acc - p.ck[ps.i0].acc) * ps.a; x.fillStyle = C.char; x.textAlign = 'left'; x.font = '700 13px Aptos, Segoe UI, sans-serif'; x.fillText(p.titleAt(ps.a >= 0.5 ? ps.i1 : ps.i0), i * w + 10, 18); x.font = '12px Aptos, Segoe UI, sans-serif'; x.fillStyle = p.color; x.fillText('accuracy ' + acc.toFixed(2), i * w + 10, 35);
    });
    drawChart();
  }
  function visible(){ var r = canvas.getBoundingClientRect(); return r.bottom > 0 && r.top < window.innerHeight && r.width > 0; }
  function frame(now){
    if (!alive) return; requestAnimationFrame(frame); if (!visible()) { last = now; return; } var dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (st.playing) { st.prog = Math.min(1, st.prog + dt / dur); if (st.prog >= 1) st.playing = false; emit(); }
    if (!st.drag) st.yaw += dt * 0.32; draw();
  }
  function emit(){ listeners.forEach(function(f){ f(st); }); }
  canvas.addEventListener('pointerdown', function(e){ st.drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', function(e){ if (!st.drag) return; st.yaw += (e.clientX - lx) * 0.01; st.pitch = clamp(st.pitch + (e.clientY - ly) * 0.01, -1.5, 1.5); lx = e.clientX; ly = e.clientY; });
  canvas.addEventListener('pointerup', function(){ st.drag = false; });
  requestAnimationFrame(function(t){ last = t; frame(t); });
  return {
    st: st, draw: draw, onChange: function(f){ listeners.push(f); },
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
