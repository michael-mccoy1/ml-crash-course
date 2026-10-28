"use strict";
/* Lesson 11, Advanced steps. Written for PK-literate scientists who are new to machine learning.
   Steps: Hybrid ODE, Sequence models, Transfer to in vivo, Graph networks to parameters.
   Every experiment on this page uses simulated data with a known truth, or the public Biogen HLM set through loadDataset('HLM').
   Helpers carry the prefix mc. The shared HLM model (MP, mpReady) comes from mlpk_a.js. */

/* ---------- helpers: random numbers, a small network, boosting, charts ---------- */
function mcNormal(r){ var u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
function mcMean(a){ var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }
function mcR2(y, p){ var m = mcMean(y), a = 0, b = 0, i; for (i = 0; i < y.length; i++) { a += (y[i] - p[i]) * (y[i] - p[i]); b += (y[i] - m) * (y[i] - m); } return b ? 1 - a / b : 0; }
function mcTrapz(arr, dt){ var s = 0; for (var i = 1; i < arr.length; i++) s += 0.5 * (arr[i] + arr[i - 1]) * dt; return s; }
function mcMax(a){ var m = -Infinity; for (var i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
function mcFold(v){ return v >= 1 ? v : 1 / v; }
function mcSig(v){ return Math.abs(v) >= 100 ? v.toFixed(0) : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)); }
/* Dense network, tanh hidden layers, linear output, Adam. */
function mcNetMake(sizes, seed){ var r = rng(seed), W = [], b = []; for (var l = 0; l < sizes.length - 1; l++) { var w = new Float64Array(sizes[l] * sizes[l + 1]), sd = Math.sqrt(1 / sizes[l]); for (var i = 0; i < w.length; i++) w[i] = (r() * 2 - 1) * sd * 1.7; W.push(w); b.push(new Float64Array(sizes[l + 1])); } return { sizes: sizes, W: W, b: b }; }
function mcNetRun(m, x){ var a = x, L = m.sizes.length - 1; for (var l = 0; l < L; l++) { var nin = m.sizes[l], nout = m.sizes[l + 1], o = new Float64Array(nout); for (var j = 0; j < nout; j++) { var s = m.b[l][j]; for (var i = 0; i < nin; i++) s += a[i] * m.W[l][i * nout + j]; o[j] = l < L - 1 ? Math.tanh(s) : s; } a = o; } return a; }
function mcNetTrain(m, X, Y, epochs, lr, batch, seed){
  var r = rng(seed), L = m.sizes.length - 1, n = X.length, mW = m.W.map(function(w){ return new Float64Array(w.length); }), vW = m.W.map(function(w){ return new Float64Array(w.length); }), mB = m.b.map(function(b){ return new Float64Array(b.length); }), vB = m.b.map(function(b){ return new Float64Array(b.length); }), step = 0, perm = [], i, ep;
  for (i = 0; i < n; i++) perm.push(i);
  for (ep = 0; ep < epochs; ep++) {
    for (i = n - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (var s0 = 0; s0 < n; s0 += batch) {
      var nb = Math.min(batch, n - s0), gW = m.W.map(function(w){ return new Float64Array(w.length); }), gB = m.b.map(function(b){ return new Float64Array(b.length); });
      for (var q = 0; q < nb; q++) {
        var idx = perm[s0 + q], A = [X[idx]], l;
        for (l = 0; l < L; l++) { var nin = m.sizes[l], nout = m.sizes[l + 1], o = new Float64Array(nout); for (var jj = 0; jj < nout; jj++) { var s = m.b[l][jj]; for (var ii = 0; ii < nin; ii++) s += A[l][ii] * m.W[l][ii * nout + jj]; o[jj] = l < L - 1 ? Math.tanh(s) : s; } A.push(o); }
        var d = new Float64Array([2 * (A[L][0] - Y[idx]) / nb]);
        for (l = L - 1; l >= 0; l--) { var nin2 = m.sizes[l], nout2 = m.sizes[l + 1], dprev = new Float64Array(nin2); for (var j2 = 0; j2 < nout2; j2++) { gB[l][j2] += d[j2]; for (var i2 = 0; i2 < nin2; i2++) { gW[l][i2 * nout2 + j2] += A[l][i2] * d[j2]; dprev[i2] += m.W[l][i2 * nout2 + j2] * d[j2]; } } if (l > 0) for (var i3 = 0; i3 < nin2; i3++) dprev[i3] *= (1 - A[l][i3] * A[l][i3]); d = dprev; }
      }
      step++; var c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
      for (var l2 = 0; l2 < L; l2++) {
        for (var k = 0; k < m.W[l2].length; k++) { var g = gW[l2][k]; mW[l2][k] = 0.9 * mW[l2][k] + 0.1 * g; vW[l2][k] = 0.999 * vW[l2][k] + 0.001 * g * g; m.W[l2][k] -= lr * (mW[l2][k] / c1) / (Math.sqrt(vW[l2][k] / c2) + 1e-8); }
        for (var k2 = 0; k2 < m.b[l2].length; k2++) { var g2 = gB[l2][k2]; mB[l2][k2] = 0.9 * mB[l2][k2] + 0.1 * g2; vB[l2][k2] = 0.999 * vB[l2][k2] + 0.001 * g2 * g2; m.b[l2][k2] -= lr * (mB[l2][k2] / c1) / (Math.sqrt(vB[l2][k2] / c2) + 1e-8); }
      }
    }
  }
}
/* Gradient boosting on a numeric table, 32 quantile bins per column, squared error. Returns a predictor for one row. */
function mcBoostFit(X, y, rounds, lr, depth, minLeaf){
  var n = X.length, nf = X[0].length, edges = [], codes = [], f, i, trees = [];
  for (f = 0; f < nf; f++) { var v = X.map(function(r){ return r[f]; }).sort(function(a, b){ return a - b; }), e = []; for (var q = 1; q < 32; q++) { var xv = v[Math.floor(q / 32 * n)]; if (!e.length || xv > e[e.length - 1]) e.push(xv); } edges.push(e); }
  function code(f2, val){ var e = edges[f2], k = 0; while (k < e.length && val > e[k]) k++; return k; }
  for (f = 0; f < nf; f++) { var c = new Uint8Array(n); for (i = 0; i < n; i++) c[i] = code(f, X[i][f]); codes.push(c); }
  function build(res, idx, d){
    var m = idx.length, sum = 0, a; for (a = 0; a < m; a++) sum += res[idx[a]]; var leaf = { v: sum / m };
    if (d === 0 || m < 2 * minLeaf) return leaf; var best = 1e-12, bf = -1, bb = -1, base = sum * sum / m;
    for (var ff = 0; ff < nf; ff++) { var nb = edges[ff].length + 1, cnt = new Float64Array(nb), sm = new Float64Array(nb); for (a = 0; a < m; a++) { var k = codes[ff][idx[a]]; cnt[k]++; sm[k] += res[idx[a]]; } var cl = 0, sl = 0; for (var b = 0; b < nb - 1; b++) { cl += cnt[b]; sl += sm[b]; var cr = m - cl; if (cl < minLeaf || cr < minLeaf) continue; var sr = sum - sl, gain = sl * sl / cl + sr * sr / cr - base; if (gain > best) { best = gain; bf = ff; bb = b; } } }
    if (bf < 0) return leaf; var Lf = [], Rt = []; for (a = 0; a < m; a++) (codes[bf][idx[a]] <= bb ? Lf : Rt).push(idx[a]);
    return { f: bf, b: bb, v: leaf.v, l: build(res, Lf, d - 1), r: build(res, Rt, d - 1) };
  }
  var mean = mcMean(y), F = new Float64Array(n).fill(mean), all = []; for (i = 0; i < n; i++) all.push(i);
  for (var rd = 0; rd < rounds; rd++) { var res = new Float64Array(n); for (i = 0; i < n; i++) res[i] = y[i] - F[i]; var t = build(res, all, depth); trees.push(t); for (i = 0; i < n; i++) { var nd = t; while (nd.f !== undefined) nd = codes[nd.f][i] <= nd.b ? nd.l : nd.r; F[i] += lr * nd.v; } }
  return function(row){ var p = mean; for (var qq = 0; qq < trees.length; qq++) { var nd = trees[qq]; while (nd.f !== undefined) nd = code(nd.f, row[nd.f]) <= nd.b ? nd.l : nd.r; p += lr * nd.v; } return p; };
}
/* A line chart on a canvas. o: { xr, yr, xlog, ylog, xt, yt: [[value, label]], xl, yl, series: [{ pts, c, w, dash }], dots: [{ x, y, c, r }], hl: [{ y, c, dash, label }], band: [[x, lo, hi]] } */
function mcChart(canvas, o){
  var g = cv(canvas), x = g.x, L = o.L || 46, R = g.w - 12, T = o.T || 12, B = g.h - (o.bot || 40), fx = o.xlog ? Math.log10 : function(v){ return v; }, fy = o.ylog ? Math.log10 : function(v){ return v; };
  var px = function(v){ return L + (fx(v) - fx(o.xr[0])) / (fx(o.xr[1]) - fx(o.xr[0])) * (R - L); }, py = function(v){ return B - (fy(v) - fy(o.yr[0])) / (fy(o.yr[1]) - fy(o.yr[0])) * (B - T); };
  x.font = '11px Aptos, "Segoe UI", sans-serif'; x.lineWidth = 1; x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right';
  (o.yt || []).forEach(function(t){ var yy = py(t[0]); x.beginPath(); x.moveTo(L, yy); x.lineTo(R, yy); x.stroke(); x.fillText(t[1], L - 6, yy + 4); });
  x.textAlign = 'center'; (o.xt || []).forEach(function(t){ var xx = px(t[0]); x.beginPath(); x.moveTo(xx, T); x.lineTo(xx, B); x.stroke(); x.fillText(t[1], xx, B + 15); });
  (o.band || []).length && (function(){ x.fillStyle = 'rgba(225,36,42,.10)'; x.beginPath(); o.band.forEach(function(p, i){ i ? x.lineTo(px(p[0]), py(p[2])) : x.moveTo(px(p[0]), py(p[2])); }); for (var i = o.band.length - 1; i >= 0; i--) x.lineTo(px(o.band[i][0]), py(o.band[i][1])); x.closePath(); x.fill(); })();
  (o.hl || []).forEach(function(h){ x.strokeStyle = h.c || C.bg; x.lineWidth = 1.4; x.setLineDash(h.dash || [5, 4]); x.beginPath(); x.moveTo(L, py(h.y)); x.lineTo(R, py(h.y)); x.stroke(); x.setLineDash([]); if (h.label) { x.fillStyle = h.c || C.mid; x.textAlign = 'right'; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.fillText(h.label, R - 4, py(h.y) - 5); x.font = '11px Aptos, "Segoe UI", sans-serif'; } });
  x.save(); x.beginPath(); x.rect(L, T - 2, R - L, B - T + 4); x.clip();
  (o.series || []).forEach(function(s){ x.strokeStyle = s.c; x.lineWidth = s.w || 2.2; x.setLineDash(s.dash || []); x.beginPath(); var st = false; s.pts.forEach(function(p){ if (o.ylog && p[1] <= 0) return; var X = px(p[0]), Y = py(p[1]); st ? x.lineTo(X, Y) : x.moveTo(X, Y); st = true; }); x.stroke(); x.setLineDash([]); });
  (o.dots || []).forEach(function(d){ if (o.ylog && d.y <= 0) return; x.fillStyle = d.c || C.char; x.beginPath(); x.arc(px(d.x), py(d.y), d.r || 3.6, 0, 6.3); x.fill(); if (d.ring) { x.strokeStyle = '#fff'; x.lineWidth = 1; x.stroke(); } });
  x.restore();
  x.strokeStyle = C.char; x.lineWidth = 1.2; x.beginPath(); x.moveTo(L, T); x.lineTo(L, B); x.lineTo(R, B); x.stroke();
  x.fillStyle = C.char; x.font = '600 11.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText(o.xl || '', (L + R) / 2, g.h - 5);
  x.save(); x.translate(12, (T + B) / 2); x.rotate(-Math.PI / 2); x.fillText(o.yl || '', 0, 0); x.restore();
  return { px: px, py: py, g: g };
}
function mcPanelText(html, size){ return '<div style="font-size:' + (size || 13) + 'px;line-height:1.5;color:' + C.char + '">' + html + '</div>'; }

/* ============================================================
   1. Hybrid ODE
   Simulated truth: one-compartment oral dosing with saturable (Michaelis-Menten) elimination.
   dC/dt = ka Ag / V - Vmax C / (Km + C), Ag = dose exp(-ka t). All constants are simulation settings, not drug data.
   ============================================================ */
var MCH = { V: 50, KA: 1.0, VM: 0.5, KM: 1.5, DT: 0.05, TEND: 36, TRAIN: [20, 100], TEST: 500, TIMES: [0.5, 1, 2, 4, 8, 12, 24], CV: 0.08, H: 6, S: 2, cache: {}, seed: 7 };
function mchTruth(dose){
  var f = function(t, y){ return [-MCH.KA * y[0], MCH.KA * y[0] / MCH.V - MCH.VM * y[1] / (MCH.KM + y[1])]; }, r = PKE.rk4(f, [dose, 0], 0, MCH.TEND, 0.01), out = [], i;
  for (i = 0; i < r.t.length; i += 5) out.push(r.y[i][1]); return out; /* one value per 0.05 h */
}
function mchLinConc(dose, ke, t){ if (Math.abs(MCH.KA - ke) < 1e-9) ke += 1e-6; return dose * MCH.KA / (MCH.V * (MCH.KA - ke)) * (Math.exp(-ke * t) - Math.exp(-MCH.KA * t)); }
function mchSim(seed){
  var r = rng(seed), obs = [], sd = Math.sqrt(Math.log(1 + MCH.CV * MCH.CV));
  MCH.TRAIN.forEach(function(d){ var tr = MCH.cache['t' + d] || (MCH.cache['t' + d] = mchTruth(d)); MCH.TIMES.forEach(function(t){ obs.push({ dose: d, t: t, c: tr[Math.round(t / MCH.DT)] * Math.exp(sd * mcNormal(r)) }); }); });
  return obs;
}
function mchFitLinear(obs){
  var sse = function(ke){ var s = 0; obs.forEach(function(o){ var e = Math.log(Math.max(1e-12, mchLinConc(o.dose, ke, o.t))) - Math.log(o.c); s += e * e; }); return s; };
  var a = 0.01, b = 0.95, g = (Math.sqrt(5) - 1) / 2, c = b - g * (b - a), d = a + g * (b - a);
  for (var i = 0; i < 80; i++) { if (sse(c) < sse(d)) b = d; else a = c; c = b - g * (b - a); d = a + g * (b - a); }
  return (a + b) / 2;
}
/* Hybrid model: dC/dt = ka Ag(t) / V - C k(C). k(C) = softplus(b2 + sum_j w2_j tanh(w1_j C / S + b1_j)) is the learned elimination rate constant.
   Fitted by Adam. The gradient is obtained by backpropagation through an explicit Euler solver (step 0.05 h). ka and V are fixed. */
function mchSoftplus(u){ return u > 30 ? u : Math.log(1 + Math.exp(u)); }
function mchForward(P, dose, nSteps, store){
  var H = MCH.H, C = 0, out = new Float64Array(nSteps + 1), st = store ? { a: new Float64Array(nSteps * H), u: new Float64Array(nSteps), c: new Float64Array(nSteps) } : null;
  for (var n = 0; n < nSteps; n++) {
    var x = C / MCH.S, u = P.b2, j, a, s = MCH.KA * dose * Math.exp(-MCH.KA * n * MCH.DT) / MCH.V;
    for (j = 0; j < H; j++) { a = Math.tanh(P.w1[j] * x + P.b1[j]); u += P.w2[j] * a; if (st) st.a[n * H + j] = a; }
    var k = mchSoftplus(u); if (st) { st.u[n] = u; st.c[n] = C; }
    C = C + MCH.DT * (s - C * k); if (C < 0) C = 0; out[n + 1] = C;
  }
  return { C: out, st: st };
}
function mchLossGrad(P, obs){
  var H = MCH.H, G = { w1: new Float64Array(H), b1: new Float64Array(H), w2: new Float64Array(H), b2: 0 }, loss = 0, by = {};
  obs.forEach(function(o){ (by[o.dose] = by[o.dose] || []).push(o); });
  Object.keys(by).forEach(function(dk){
    var dose = +dk, list = by[dk], nSteps = Math.round(24 / MCH.DT), fp = mchForward(P, dose, nSteps, true), inj = new Float64Array(nSteps + 1);
    list.forEach(function(o){ var n = Math.round(o.t / MCH.DT), c = Math.max(fp.C[n], 1e-9), e = Math.log(c) - Math.log(o.c); loss += e * e; inj[n] += 2 * e / c; });
    var adj = 0;
    for (var n = nSteps - 1; n >= 0; n--) {
      adj += inj[n + 1]; var C = fp.st.c[n], u = fp.st.u[n], sg = 1 / (1 + Math.exp(-u)), k = mchSoftplus(u), x = C / MCH.S, dkdC = 0, j, a, dep;
      for (j = 0; j < H; j++) { a = fp.st.a[n * H + j]; dep = P.w2[j] * (1 - a * a); dkdC += sg * dep * P.w1[j] / MCH.S; }
      var coef = -MCH.DT * C * adj; G.b2 += coef * sg;
      for (j = 0; j < H; j++) { a = fp.st.a[n * H + j]; dep = P.w2[j] * (1 - a * a); G.w2[j] += coef * sg * a; G.b1[j] += coef * sg * dep; G.w1[j] += coef * sg * dep * x; }
      adj = adj * (1 - MCH.DT * (k + C * dkdC));
    }
  });
  return { loss: loss, G: G };
}
function mchFitHybrid(obs, ke0, seed, iters){
  var H = MCH.H, r = rng(seed), P = { w1: new Float64Array(H), b1: new Float64Array(H), w2: new Float64Array(H), b2: Math.log(Math.exp(ke0) - 1) }, j, keys = ['w1', 'b1', 'w2'];
  for (j = 0; j < H; j++) { P.w1[j] = (r() - 0.5) * 2; P.b1[j] = (r() - 0.5) * 0.5; P.w2[j] = (r() - 0.5) * 0.2; }
  var m = { w1: new Float64Array(H), b1: new Float64Array(H), w2: new Float64Array(H), b2: 0 }, v = { w1: new Float64Array(H), b1: new Float64Array(H), w2: new Float64Array(H), b2: 0 }, lr = 0.03, first = 0, last = 0;
  for (var it = 1; it <= iters; it++) {
    var lg = mchLossGrad(P, obs), c1 = 1 - Math.pow(0.9, it), c2 = 1 - Math.pow(0.999, it); if (it === 1) first = lg.loss / obs.length; last = lg.loss / obs.length;
    keys.forEach(function(k){ for (var q = 0; q < H; q++) { var gq = lg.G[k][q]; m[k][q] = 0.9 * m[k][q] + 0.1 * gq; v[k][q] = 0.999 * v[k][q] + 0.001 * gq * gq; P[k][q] -= lr * (m[k][q] / c1) / (Math.sqrt(v[k][q] / c2) + 1e-8); } });
    var g2 = lg.G.b2; m.b2 = 0.9 * m.b2 + 0.1 * g2; v.b2 = 0.999 * v.b2 + 0.001 * g2 * g2; P.b2 -= lr * (m.b2 / c1) / (Math.sqrt(v.b2 / c2) + 1e-8);
  }
  return { P: P, first: first, last: last };
}
/* Plain network: inputs time and log dose, output ln concentration. */
function mchNetIn(t, dose){ return [t / MCH.TEND, (Math.log10(dose) - Math.log10(MCH.TRAIN[0])) / (Math.log10(MCH.TRAIN[1]) - Math.log10(MCH.TRAIN[0]))]; }
function mchFitNet(obs, seed){
  var ys = obs.map(function(o){ return Math.log(o.c); }), mu = mcMean(ys), net = mcNetMake([2, 16, 16, 1], seed);
  mcNetTrain(net, obs.map(function(o){ return mchNetIn(o.t, o.dose); }), ys.map(function(y){ return y - mu; }), 3000, 0.01, 14, seed + 1);
  return function(t, dose){ var lc = mcNetRun(net, mchNetIn(t, dose))[0] + mu; return Math.exp(Math.max(-12, Math.min(6, lc))); };
}
function mchRun(seed){
  var obs = mchSim(seed), ke = mchFitLinear(obs), hy = mchFitHybrid(obs, ke, 3, 600), nf = mchFitNet(obs, 5), n = Math.round(MCH.TEND / MCH.DT), doses = MCH.TRAIN.concat([MCH.TEST]), out = { obs: obs, ke: ke, hybLoss: [hy.first, hy.last], curves: {}, metrics: {} };
  doses.forEach(function(d){
    var tr = MCH.cache['t' + d] || (MCH.cache['t' + d] = mchTruth(d)), lin = [], hyb = Array.from(mchForward(hy.P, d, n, false).C), nn = [], i;
    for (i = 0; i <= n; i++) { lin.push(mchLinConc(d, ke, i * MCH.DT)); nn.push(i === 0 ? 0 : nf(i * MCH.DT, d)); }
    out.curves[d] = { truth: tr, lin: lin, hyb: hyb, nn: nn };
    var A = mcTrapz(tr, MCH.DT), M = mcMax(tr), q = out.curves[d];
    out.metrics[d] = { lin: [mcTrapz(lin, MCH.DT) / A, mcMax(lin) / M], hyb: [mcTrapz(hyb, MCH.DT) / A, mcMax(hyb) / M], nn: [mcTrapz(nn, MCH.DT) / A, mcMax(nn) / M] };
  });
  var fe = function(fn){ var s = 0; obs.forEach(function(o){ s += Math.abs(Math.log(fn(o) / o.c)); }); return Math.exp(s / obs.length); };
  out.train = { lin: fe(function(o){ return mchLinConc(o.dose, ke, o.t); }), hyb: fe(function(o){ return out.curves[o.dose].hyb[Math.round(o.t / MCH.DT)]; }), nn: fe(function(o){ return out.curves[o.dose].nn[Math.round(o.t / MCH.DT)]; }) };
  return out;
}
LESSONS.mlpk.add({ level: 'advanced', short: 'Hybrid ODE', title: 'Hybrid mechanistic-ML models',
  change: 'Compare the three fits at the training doses and at the extrapolated dose. Switch the concentration axis. Select New noise draw to refit.',
  watch: 'A hybrid model keeps the mechanistic structure of an ODE and replaces one unknown term with a learned function. The data here are simulated from a one-compartment oral model in which elimination saturates (Michaelis-Menten), at two training doses with seven sampling times each and lognormal error (CV 8%). Three models are fitted to the same 14 observations. (a) A mechanistic model with linear elimination, one parameter. (b) A hybrid model with the same structure in which the elimination rate constant k(C) is a network with 6 tanh units, fitted by backpropagation through an Euler solver with the absorption rate constant and volume fixed at their true values. (c) A network that takes time and log dose as inputs and returns ln concentration, with no mechanistic structure. The right-hand panel is a 500 mg dose, 5 times the highest training dose, and no model saw data there.',
  build: function(api){
    var res = null, logY = false, seed = 7, busy = false;
    var lp = api.el('pn', 'left:28px;top:18px;width:904px;height:30px;padding:5px 12px', '<div class="lg" style="display:flex;gap:20px;font-size:12px;color:' + C.char + '"></div>');
    var cps = [0, 1, 2].map(function(k){ return api.el('pn', 'left:' + (28 + k * 306) + 'px;top:54px;width:292px;height:236px', '<h4 class="ch"></h4><canvas width="264" height="190"></canvas>'); });
    var tp = api.el('pn', 'left:28px;top:300px;width:904px;height:222px;overflow:auto', '<div class="tb"></div>');
    var ctl = api.addControl('<span class="seg" data-ax><button type="button" data-v="0" class="on">Linear axis</button><button type="button" data-v="1">Log axis</button></span> <button type="button" class="btn sm ghost" data-new>New noise draw</button>');
    var NAMES = [['truth', 'Simulated truth', C.char, [6, 4], 2], ['lin', 'Mechanistic, linear elimination', C.bg, [], 2.4], ['hyb', 'Hybrid, learned elimination', C.red, [], 2.8], ['nn', 'Network on time and dose', '#6F8196', [2, 3], 2.4]];
    qs('.lg', lp).innerHTML = NAMES.map(function(n){ return '<span><svg width="26" height="10" style="vertical-align:middle;margin-right:5px"><line x1="0" y1="5" x2="26" y2="5" stroke="' + n[2] + '" stroke-width="3"' + (n[3].length ? ' stroke-dasharray="' + n[3].join(' ') + '"' : '') + '/></svg>' + n[1] + '</span>'; }).join('') + '<span><svg width="12" height="10" style="vertical-align:middle;margin-right:5px"><circle cx="6" cy="5" r="3.6" fill="' + C.char + '"/></svg>Observed</span>';
    function draw(){
      if (!res) return; var doses = MCH.TRAIN.concat([MCH.TEST]);
      doses.forEach(function(d, k){
        var cv_ = res.curves[d], pts = function(arr, from){ var o = []; for (var i = from || 0; i < arr.length; i++) o.push([i * MCH.DT, arr[i]]); return o; }, ymax = Math.ceil(mcMax(cv_.truth) * 1.35 * 10) / 10, series = NAMES.map(function(n){ return { pts: pts(cv_[n[0]], n[0] === 'nn' ? 5 : 0), c: n[2], dash: n[3], w: n[4] }; }), yt;
        if (logY) yt = [[0.01, '0.01'], [0.1, '0.1'], [1, '1'], [10, '10']]; else { var st = ymax <= 0.5 ? 0.1 : (ymax <= 2 ? 0.5 : (ymax <= 5 ? 1 : 2)); yt = []; for (var v = 0; v <= ymax + 1e-9; v += st) yt.push([v, v.toFixed(st < 1 ? 1 : 0)]); }
        qs('.ch', cps[k]).textContent = d + ' mg' + (d === MCH.TEST ? ', extrapolation' : ', training dose');
        mcChart(qs('canvas', cps[k]), { xr: [0, MCH.TEND], yr: logY ? [0.01, 20] : [0, yt[yt.length - 1][0]], ylog: logY, xt: [[0, '0'], [12, '12'], [24, '24'], [36, '36']], yt: yt, xl: 'time (h)', yl: 'concentration (mg/L)', series: series, dots: res.obs.filter(function(o){ return o.dose === d; }).map(function(o){ return { x: o.t, y: o.c, c: C.char, r: 3.6, ring: true }; }) });
      });
      var row = function(label, key, col){ var m = res.metrics; return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:4px 6px;font-weight:700;color:' + col + '">' + label + '</td><td style="text-align:right">' + res.train[key].toFixed(2) + '</td>' + doses.map(function(d){ return '<td style="text-align:right">' + m[d][key][0].toFixed(2) + '</td>'; }).join('') + '<td style="text-align:right">' + m[MCH.TEST][key][1].toFixed(2) + '</td></tr>'; };
      qs('.tb', tp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12.5px;color:' + C.char + '"><tr style="font-size:11px;color:' + C.mid + ';letter-spacing:.04em"><td></td><td style="text-align:right;padding:2px 6px">Training points, geometric mean fold error</td><td style="text-align:right;padding:2px 6px">AUC ratio, 20 mg</td><td style="text-align:right;padding:2px 6px">AUC ratio, 100 mg</td><td style="text-align:right;padding:2px 6px">AUC ratio, 500 mg</td><td style="text-align:right;padding:2px 6px">Cmax ratio, 500 mg</td></tr>' + row('Mechanistic, linear', 'lin', C.mid) + row('Hybrid', 'hyb', C.red) + row('Network on time and dose', 'nn', '#6F8196') + '</table>' +
        mcPanelText('<p style="margin:8px 0 0">AUC is over 0 to 36 h. Ratios are predicted over true, and 1.00 is exact. All three models fit the training doses closely, so the training columns do not separate them. At 500 mg the linear model under-predicts exposure by about ' + mcSig(1 / res.metrics[MCH.TEST].lin[0]) + ' fold because it cannot represent saturation, and the plain network has no structure that constrains it beyond 100 mg. The hybrid learns k(C) from the two training doses. Its learned rate constant falls with concentration, which the 500 mg dose requires.</p><p style="margin:6px 0 0;color:' + C.mid + '">Simulation settings, not drug data: V 50 L, ka 1 per h, Vmax 0.5 mg/L per h, Km 1.5 mg/L. Doses 20, 100 and 500 mg. Sampling at 0.5, 1, 2, 4, 8, 12 and 24 h. The linear model has ke fitted by golden-section search on the log residuals (fitted ke ' + res.ke.toFixed(2) + ' per h, against the true low-concentration value ' + (MCH.VM / MCH.KM).toFixed(2) + ' per h). Hybrid loss: ' + res.hybLoss[0].toFixed(3) + ' to ' + res.hybLoss[1].toFixed(4) + ' mean squared log residual in 600 Adam steps.</p>', 12);
    }
    async function fit(s){ if (busy) return; busy = true; seed = s; api.busy('Fitting the three models'); await tick(); await tick(); try { res = mchRun(seed); } finally { api.idle(); busy = false; } draw(); }
    async function render(){ if (!res) await fit(seed); else draw(); }
    qsa('[data-ax] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ logY = +b.dataset.v === 1; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    qs('[data-new]', ctl).addEventListener('click', function(){ fit(seed + 1); });
    async function play(tok){
      await render(); api.cap('Two training doses, seven samples each. Elimination saturates in the simulated truth.'); await api.sleep(3400); api.checkTok(tok);
      api.cap('At 20 and 100 mg all three models follow the observations. The training error does not separate them.'); await api.sleep(3600); api.checkTok(tok);
      api.cap('At 500 mg, a dose no model was fitted to, the models diverge. The hybrid keeps the mechanistic structure and follows the truth most closely.'); await api.sleep(4200); api.checkTok(tok);
      logY = true; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 1); }); draw(); api.cap('On a log axis the terminal slopes show the elimination rate each model assumes.'); await api.sleep(3600);
      logY = false; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 0); }); draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ============================================================
   2. Sequence models
   Simulated subjects, one-compartment oral dose of 100 mg, lognormal between-subject variability, 4 to 6 irregular samples, lognormal residual error (CV 10%).
   Target: log10 CL, truth known.
   ============================================================ */
var MCS = { GRID: [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24], SIZES: [20, 50, 100, 200, 400], REP: 4, DOSE: 100, res: null, seed: 5 };
function mcsSubjects(n, seed){
  var r = rng(seed), pop = PKE.population(n, seed + 1, { CL: { typical: 5, cv: 0.35 }, V: { typical: 45, cv: 0.25 }, ka: { typical: 1.2, cv: 0.4 } }), out = [], sd = Math.sqrt(Math.log(1.1 * 1.1));
  for (var i = 0; i < n; i++) {
    var p = pop[i], k = 4 + Math.floor(r() * 3), times = [], oc = PKE.oneComp({ dose: MCS.DOSE, V: p.V, CL: p.CL, ka: p.ka, F: 1, route: 'oral' });
    for (var j = 0; j < k; j++) times.push(0.25 + r() * 23.75); times.sort(function(a, b){ return a - b; });
    out.push({ times: times, conc: times.map(function(t){ return oc.conc(t) * Math.exp(sd * mcNormal(r)); }), logCL: Math.log10(p.CL), p: p });
  }
  return out;
}
function mcsSummary(s){
  var t = s.times, c = s.conc, n = t.length, cm = 0, tm = 0, i, auc = 0;
  for (i = 0; i < n; i++) { if (c[i] > cm) { cm = c[i]; tm = t[i]; } if (i) auc += 0.5 * (c[i] + c[i - 1]) * (t[i] - t[i - 1]); }
  var tail = Math.min(3, n), xs = [], ys = []; for (i = n - tail; i < n; i++) { xs.push(t[i]); ys.push(Math.log(c[i])); }
  var mx = mcMean(xs), my = mcMean(ys), sxy = 0, sxx = 0; for (i = 0; i < tail; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) * (xs[i] - mx); }
  return [Math.log(cm), tm, Math.log(c[n - 1]), t[n - 1], Math.log(auc + 1e-9), sxx > 0 ? sxy / sxx : 0, n];
}
function mcsGrid(s){
  return MCS.GRID.map(function(g){ var t = s.times, c = s.conc, k = 0; while (k < t.length && t[k] < g) k++; if (k === 0) return Math.log(c[0]); if (k === t.length) return Math.log(c[t.length - 1]); var w = (g - t[k - 1]) / (t[k] - t[k - 1]); return (1 - w) * Math.log(c[k - 1]) + w * Math.log(c[k]); });
}
function mcsRun(seed){
  var test = mcsSubjects(500, 99), yt = test.map(function(s){ return s.logCL; }), Xst = test.map(mcsSummary), Xgt = test.map(mcsGrid), res = { sizes: MCS.SIZES, boost: [], mlp: [], example: mcsSubjects(4, seed + 500), last: null };
  MCS.SIZES.forEach(function(n){
    var bs = [], ms = [];
    for (var rp = 0; rp < MCS.REP; rp++) {
      var tr = mcsSubjects(n, seed * 1000 + n * 7 + rp), y = tr.map(function(s){ return s.logCL; }), Xs = tr.map(mcsSummary), Xg = tr.map(mcsGrid);
      var bfit = mcBoostFit(Xs, y, 100, 0.1, 3, 2), pb = Xst.map(bfit); bs.push(mcR2(yt, pb));
      var mu = [], sd = []; for (var f = 0; f < MCS.GRID.length; f++) { var col = Xg.map(function(r){ return r[f]; }), m = mcMean(col), v = Math.sqrt(mcMean(col.map(function(q){ return (q - m) * (q - m); }))) || 1; mu.push(m); sd.push(v); }
      var norm = function(r){ return r.map(function(q, f){ return (q - mu[f]) / sd[f]; }); }, ym = mcMean(y), net = mcNetMake([MCS.GRID.length, 8, 1], 3 + rp);
      mcNetTrain(net, Xg.map(norm), y.map(function(q){ return q - ym; }), 150, 0.005, 16, 4 + rp);
      var pm = Xgt.map(function(r){ return mcNetRun(net, norm(r))[0] + ym; }); ms.push(mcR2(yt, pm));
      if (n === MCS.SIZES[MCS.SIZES.length - 1] && rp === 0) res.last = { y: yt, boost: pb, mlp: pm };
    }
    res.boost.push(bs); res.mlp.push(ms);
  });
  return res;
}
LESSONS.mlpk.add({ level: 'advanced', short: 'Sequence models', title: 'Sequence models for profiles',
  change: 'Compare the learning curves of the two trained models. Select New subjects to repeat the experiment with a new simulated population.',
  watch: 'Sparse, irregularly sampled concentration-time profiles do not fit a fixed-width table. Three ways to give a profile to a model are compared for predicting log10 CL in simulated subjects (one-compartment oral dose, lognormal between-subject variability, 4 to 6 samples at random times, lognormal error with CV 10%). Summary features are Cmax, tmax, the last concentration and time, a trapezoid AUC, a terminal slope and the number of samples; they feed gradient boosting. The profile can also be interpolated in log concentration onto a fixed grid of 12 times; that vector feeds a small MLP. A recurrent or attention model instead reads the profile directly as an ordered sequence of (time, concentration) pairs of any length. Only the first two are trained here, on 20 to 400 simulated subjects with 4 repeats per size, and scored on 500 further subjects. Sequence architectures are not trained in this page.',
  build: function(api){
    var busy = false, seed = 5;
    var ep = api.el('pn', 'left:28px;top:20px;width:430px;height:248px', '<h4>Four simulated subjects: sparse samples and the true profile</h4><canvas width="402" height="204"></canvas>');
    var sp = api.el('pn', 'left:474px;top:20px;width:458px;height:248px', '<h4>Three ways to give a profile to a model</h4><canvas width="430" height="204"></canvas>');
    var lp = api.el('pn', 'left:28px;top:280px;width:520px;height:242px', '<h4>Test R2 for log10 CL against training subjects</h4><canvas width="492" height="198"></canvas>');
    var tp = api.el('pn', 'left:564px;top:280px;width:368px;height:242px;overflow:auto', '<div class="tb"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-new>New subjects</button>');
    var COL = [C.red, C.char, C.bg, '#6F8196'];
    function drawExamples(res){
      var ex = res.example, ymax = 0, k, i; ex.forEach(function(s){ var oc = PKE.oneComp({ dose: MCS.DOSE, V: s.p.V, CL: s.p.CL, ka: s.p.ka, F: 1, route: 'oral' }); s._oc = oc; ymax = Math.max(ymax, oc.summary.cmax, mcMax(s.conc)); });
      ymax = Math.ceil(ymax * 1.1 * 10) / 10; var series = ex.map(function(s, q){ var pts = []; for (var t = 0; t <= 24.01; t += 0.25) pts.push([t, s._oc.conc(t)]); return { pts: pts, c: COL[q], w: 1.8 }; }), dots = []; ex.forEach(function(s, q){ s.times.forEach(function(t, j){ dots.push({ x: t, y: s.conc[j], c: COL[q], r: 4, ring: true }); }); });
      var yt = []; var st = ymax > 2.5 ? 1 : 0.5; for (var v = 0; v <= ymax + 1e-9; v += st) yt.push([v, v.toFixed(1)]);
      mcChart(qs('canvas', ep), { xr: [0, 24], yr: [0, yt[yt.length - 1][0] || ymax], xt: [[0, '0'], [6, '6'], [12, '12'], [18, '18'], [24, '24']], yt: yt, xl: 'time after dose (h)', yl: 'concentration (mg/L)', series: series, dots: dots });
    }
    function drawSchematic(){
      var g = cv(qs('canvas', sp)), x = g.x, W = g.w, bx = function(X, Y, w, h, label, o){ o = o || {}; x.fillStyle = o.f || '#fff'; x.strokeStyle = o.s || C.char; x.lineWidth = o.w || 1.4; if (o.d) x.setLineDash(o.d); x.beginPath(); x.rect(X, Y, w, h); x.fill(); x.stroke(); x.setLineDash([]); x.fillStyle = o.c || C.char; x.font = (o.b ? '700 ' : '') + (o.sz || 11) + 'px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; (Array.isArray(label) ? label : [label]).forEach(function(t, i, a){ x.fillText(t, X + w / 2, Y + h / 2 + 4 - (a.length - 1) * 7 + i * 14); }); },
          ar = function(x1, y1, x2, y2){ x.strokeStyle = C.char; x.lineWidth = 1.3; x.beginPath(); x.moveTo(x1, y1); x.lineTo(x2, y2); x.stroke(); var a = Math.atan2(y2 - y1, x2 - x1); x.fillStyle = C.char; x.beginPath(); x.moveTo(x2, y2); x.lineTo(x2 - 7 * Math.cos(a - 0.4), y2 - 7 * Math.sin(a - 0.4)); x.lineTo(x2 - 7 * Math.cos(a + 0.4), y2 - 7 * Math.sin(a + 0.4)); x.closePath(); x.fill(); };
      var rows = [[['Summary features', 'Cmax, tmax, last value,', 'AUC, slope, n'], 'Gradient boosting', 'trained here'], [['Fixed grid', '12 interpolated log', 'concentrations'], 'MLP', 'trained here'], [['Ordered sequence', '(t1, c1), (t2, c2), ...', 'any length'], 'Recurrent or attention layers', 'not trained here']];
      x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillStyle = C.mid; x.fillText('Profile representation', 70, 12); x.fillText('Model', 305, 12); x.fillText('Output', 405, 12);
      rows.forEach(function(r, i){ var Y = 22 + i * 60, off = i === 2; bx(2, Y, 140, 46, r[0], { f: C.light, sz: 10.5 }); ar(144, Y + 23, 218, Y + 23); bx(220, Y + 4, 168, 38, [r[1], r[2]], { s: off ? C.mid : C.red, d: off ? [4, 3] : null, c: off ? C.mid : C.char, sz: 11 }); ar(390, Y + 23, 412, Y + 23); bx(414, Y + 10, 28, 26, 'CL', { f: '#FBE9E9', sz: 11 }); });
      x.fillStyle = C.char; x.font = '600 10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText('Each row ends in log10 CL.', 2, 22 + 3 * 60 + 2);
      x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.fillText('A sequence model accepts a different number of samples per subject without resampling.', 2, 22 + 3 * 60 + 18);
    }
    function drawCurve(res){
      var pts = function(arr){ return arr.map(function(a, i){ return [res.sizes[i], mcMean(a)]; }); }, dots = [];
      [['boost', C.red], ['mlp', '#6F8196']].forEach(function(p){ res[p[0]].forEach(function(a, i){ a.forEach(function(v){ dots.push({ x: res.sizes[i], y: v, c: p[1], r: 2.4 }); }); }); });
      var hc = mcChart(qs('canvas', lp), { xr: [15, 500], yr: [-0.3, 0.8], xlog: true, xt: res.sizes.map(function(n){ return [n, String(n)]; }), yt: [[-0.2, '-0.2'], [0, '0'], [0.2, '0.2'], [0.4, '0.4'], [0.6, '0.6'], [0.8, '0.8']], xl: 'training subjects (log scale)', yl: 'test R2', series: [{ pts: pts(res.boost), c: C.red, w: 2.8 }, { pts: pts(res.mlp), c: '#6F8196', w: 2.8 }], dots: dots, hl: [{ y: 0, c: C.bg, dash: [3, 3] }] });
      var g = hc.g, x = g.x; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillStyle = C.red; x.fillText('Boosting on summary features', 58, 24); x.fillStyle = '#6F8196'; x.fillText('MLP on the fixed grid', 58, 40);
    }
    function table(res){
      var rows = res.sizes.map(function(n, i){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:3px 4px">' + n + '</td><td style="text-align:right">' + mcMean(res.boost[i]).toFixed(2) + '</td><td style="text-align:right">' + mcMean(res.mlp[i]).toFixed(2) + '</td></tr>'; }).join('');
      qs('.tb', tp).innerHTML = mcPanelText('<table style="width:100%;border-collapse:collapse"><tr style="font-size:11px;color:' + C.mid + '"><td style="padding:2px 4px">Subjects</td><td style="text-align:right">Boosting</td><td style="text-align:right">MLP</td></tr>' + rows + '</table><p style="margin:8px 0 0">Mean test R2 over ' + MCS.REP + ' training sets of each size, on 500 test subjects. Dots in the chart are the individual runs.</p><p style="margin:6px 0 0">With 20 subjects the MLP on the interpolated grid is not reliable, since its 12 inputs and 113 parameters exceed the number of subjects. Gradient boosting on 7 summary features degrades more slowly. The MLP closes the gap as subjects are added. The result is specific to this simulation: summary features carry most of the information about CL here, and interpolating a sparse profile discards some of it.</p>', 12);
    }
    async function run(s){ if (busy) return; busy = true; seed = s; api.busy('Training the two models on simulated subjects'); await tick(); await tick(); try { MCS.res = mcsRun(seed); } finally { api.idle(); busy = false; } draw(); }
    function draw(){ if (!MCS.res) return; drawExamples(MCS.res); drawSchematic(); drawCurve(MCS.res); table(MCS.res); }
    async function render(){ if (!MCS.res) await run(seed); else draw(); }
    qs('[data-new]', ctl).addEventListener('click', function(){ run(seed + 1); });
    async function play(tok){
      await render(); api.cap('Each subject has 4 to 6 samples at different times, so the profiles cannot be stacked into a fixed table directly.'); await api.sleep(3600); api.checkTok(tok);
      api.cap('A profile can be summarised, interpolated onto a fixed grid, or read as an ordered sequence. Only the first two are trained here.'); await api.sleep(4000); api.checkTok(tok);
      api.cap('Boosting on summary features is ahead at small numbers of subjects. The MLP on the grid closes the gap as subjects are added.'); await api.sleep(4000); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ============================================================
   3. Transfer to in vivo
   The in vivo target is SIMULATED from the in vitro HLM data: log10 CLint (measured) plus a fixed offset and lognormal scatter.
   The experiment shows the mechanism of transfer, not real in vivo performance.
   ============================================================ */
var MCT = { SIZES: [20, 40, 80, 160, 280], REP: 10, sd: 0.25, off: Math.log10(3), res: null, pool: null };
function mctFeat(ds, i){ var r = []; for (var f = 0; f < ds.BASE.length; f++) r.push(ds.COL[ds.BASE[f]][i]); return r; }
function mctRun(sdNoise){
  var ds = MP.ds, P = MP.F, rr = rng(11), pool = ds.TEST.filter(function(i){ return !ds.cens[i]; }), yv = {}, i;
  pool.forEach(function(q){ yv[q] = ds.Y[q] + MCT.off + sdNoise * mcNormal(rr); });
  var sh = pool.slice(); for (i = sh.length - 1; i > 0; i--) { var j = Math.floor(rr() * (i + 1)), t = sh[i]; sh[i] = sh[j]; sh[j] = t; }
  var ev = sh.slice(0, 150), rest = sh.slice(150), yev = ev.map(function(q){ return yv[q]; }), out = { sizes: MCT.SIZES, scratch: [], stack: [], frozen: [], ceiling: mcR2(yev, ev.map(function(q){ return ds.Y[q] + MCT.off; })), raw: mcR2(yev, ev.map(function(q){ return P[q]; })), pool: pool.length };
  MCT.SIZES.forEach(function(n){
    var ra = [], rb = [], rc = [];
    for (var rp = 0; rp < MCT.REP; rp++) {
      var r3 = rng(100 + rp * 7 + n), s2 = rest.slice(); for (var q = s2.length - 1; q > 0; q--) { var jj = Math.floor(r3() * (q + 1)), tt = s2[q]; s2[q] = s2[jj]; s2[jj] = tt; }
      var tr = s2.slice(0, Math.min(n, s2.length)), ty = tr.map(function(z){ return yv[z]; }), Xtr = tr.map(function(z){ return mctFeat(ds, z); });
      var fa = mcBoostFit(Xtr, ty, 100, 0.1, 2, 3), fb = mcBoostFit(Xtr.map(function(r, k){ return r.concat([P[tr[k]]]); }), ty, 100, 0.1, 2, 3);
      var px = tr.map(function(z){ return P[z]; }), mx = mcMean(px), my = mcMean(ty), sxy = 0, sxx = 0; for (var k2 = 0; k2 < px.length; k2++) { sxy += (px[k2] - mx) * (ty[k2] - my); sxx += (px[k2] - mx) * (px[k2] - mx); } var b = sxx > 0 ? sxy / sxx : 0;
      ra.push(mcR2(yev, ev.map(function(z){ return fa(mctFeat(ds, z)); }))); rb.push(mcR2(yev, ev.map(function(z){ return fb(mctFeat(ds, z).concat([P[z]])); }))); rc.push(mcR2(yev, ev.map(function(z){ return my + b * (P[z] - mx); })));
    }
    out.scratch.push(ra); out.stack.push(rb); out.frozen.push(rc);
  });
  return out;
}
LESSONS.mlpk.add({ level: 'advanced', short: 'Transfer to in vivo', title: 'Transfer from in vitro to in vivo',
  change: 'Read the three learning curves. Drag the scatter of the simulated in vivo values to change how noisy the target is.',
  watch: 'In vitro data are plentiful and in vivo clearance data are scarce, so a model trained on the in vitro endpoint can supply a feature or a starting point for the in vivo one. The in vivo target here is simulated from the public HLM set: the measured log10 CLint plus a fixed offset of log10 3 and lognormal scatter, for the 431 quantified compounds of the scaffold test set. The experiment therefore shows the mechanism of transfer, not real in vivo performance. The HLM boosted-trees model of lesson 02 is trained on the HLM training compounds only. Three models are trained on n of the simulated in vivo compounds and scored on 150 others. Scratch: boosting on the eight descriptors. Stacked: boosting on the descriptors plus the HLM model prediction as a ninth feature, the same stacking idea as in lesson 01. Frozen predictor with a linear head: the HLM prediction is the only input and a straight line is fitted, which is a two-parameter model.',
  build: function(api){
    var sdv = MCT.sd, busy = false;
    var cp = api.el('pn', 'left:28px;top:20px;width:560px;height:330px', '<h4>Test R2 against the number of simulated in vivo compounds</h4><canvas width="532" height="286"></canvas>');
    var rp = api.el('pn', 'left:604px;top:20px;width:328px;height:330px;overflow:auto', '<div class="rs"></div>');
    var tp = api.el('pn', 'left:28px;top:362px;width:904px;height:160px;overflow:auto', '<div class="tb"></div>');
    var ctl = api.addControl('Scatter of the simulated target, log10 units <input class="sd" type="range" min="0.05" max="0.5" step="0.05" style="width:180px"> <b class="sdv"></b>');
    var sI = qs('.sd', ctl), sV = qs('.sdv', ctl); sI.value = sdv; sV.textContent = sdv.toFixed(2);
    function draw(){
      var R = MCT.res; if (!R) return; var pts = function(a){ return a.map(function(v, i){ return [R.sizes[i], mcMean(v)]; }); }, dots = [];
      [['scratch', C.char], ['stack', '#6F8196'], ['frozen', C.red]].forEach(function(p){ R[p[0]].forEach(function(a, i){ a.forEach(function(v){ dots.push({ x: R.sizes[i], y: v, c: p[1], r: 2 }); }); }); });
      var hc = mcChart(qs('canvas', cp), { xr: [15, 350], yr: [-0.6, 0.9], xlog: true, xt: R.sizes.map(function(n){ return [n, String(n)]; }), yt: [[-0.4, '-0.4'], [-0.2, '-0.2'], [0, '0'], [0.2, '0.2'], [0.4, '0.4'], [0.6, '0.6'], [0.8, '0.8']], xl: 'simulated in vivo compounds used for training (log scale)', yl: 'test R2', series: [{ pts: pts(R.scratch), c: C.char, w: 2.6 }, { pts: pts(R.stack), c: '#6F8196', w: 2.6, dash: [6, 4] }, { pts: pts(R.frozen), c: C.red, w: 2.8 }], dots: dots, hl: [{ y: R.ceiling, c: '#2E7D4F', label: 'ceiling set by the scatter of the target, ' + R.ceiling.toFixed(2) }, { y: 0, c: C.bg, dash: [3, 3] }] });
      var g = hc.g, x = g.x; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; [['Scratch: descriptors only', C.char], ['Stacked: descriptors and HLM prediction', '#6F8196'], ['Frozen predictor and linear head', C.red]].forEach(function(l, i){ x.fillStyle = l[1]; x.fillText(l[0], 54, 76 + i * 15); });
      var rows = R.sizes.map(function(n, i){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:3px 6px">' + n + '</td><td style="text-align:right">' + mcMean(R.scratch[i]).toFixed(2) + '</td><td style="text-align:right">' + mcMean(R.stack[i]).toFixed(2) + '</td><td style="text-align:right">' + mcMean(R.frozen[i]).toFixed(2) + '</td></tr>'; }).join('');
      qs('.rs', rp).innerHTML = mcPanelText('<div class="ol">Simulation settings</div><p style="margin:2px 0 6px">Offset log10 3, scatter SD ' + sdv.toFixed(2) + ' log10 units. These are settings of the simulation, not measurements. Pool: ' + R.pool + ' quantified test compounds, 150 held out for scoring, the rest drawn for training. Mean over ' + MCT.REP + ' draws of each size.</p><div class="ol">Reference values</div><p style="margin:2px 0 0">The simulated target explains at most R2 ' + R.ceiling.toFixed(2) + ' of its own variance. The HLM prediction used directly, with no offset correction, scores R2 ' + R.raw.toFixed(2) + ' because it does not carry the offset.</p>', 12);
      qs('.tb', tp).innerHTML = mcPanelText('<table style="width:60%;border-collapse:collapse;float:left;margin-right:20px"><tr style="font-size:11px;color:' + C.mid + '"><td style="padding:2px 6px">Training compounds</td><td style="text-align:right">Scratch</td><td style="text-align:right">Stacked</td><td style="text-align:right">Frozen and linear</td></tr>' + rows + '</table><p style="margin:0">The frozen predictor with a linear head is ahead at small n. Its two parameters are estimated from a few compounds, and it inherits what the HLM model learned from ' + MP.ds.TRAIN.length.toLocaleString('en-US') + ' training compounds. Boosting on descriptors alone needs many compounds. Adding the HLM prediction as a feature gives almost no gain here, because a tree ensemble with 9 inputs and a few dozen rows cannot learn a smooth linear relation to one of them. A low-capacity model on the transferred prediction can.</p>', 12);
    }
    async function run(){ if (busy) return; busy = true; api.busy('Fitting the HLM model and the three transfer models'); await tick(); await tick(); try { await mpReady(api); MCT.res = mctRun(sdv); } finally { api.idle(); busy = false; } draw(); }
    async function render(){ if (!MCT.res) await run(); else draw(); }
    sI.addEventListener('change', function(){ sdv = +sI.value; sV.textContent = sdv.toFixed(2); run(); });
    sI.addEventListener('input', function(){ sV.textContent = (+sI.value).toFixed(2); });
    async function play(tok){
      await render(); api.cap('The HLM model was fitted on ' + MP.ds.TRAIN.length.toLocaleString('en-US') + ' compounds. The simulated in vivo set has at most a few hundred.'); await api.sleep(3600); api.checkTok(tok);
      api.cap('With 20 compounds, boosting on descriptors alone is worse than predicting the mean of the training compounds.'); await api.sleep(3800); api.checkTok(tok);
      api.cap('A straight line fitted to the frozen HLM prediction is positive already at 20 compounds and rises with little data.'); await api.sleep(4000); api.checkTok(tok);
      api.cap('Adding the prediction as a feature to a tree ensemble does not help at these sample sizes.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ============================================================
   4. Graph networks to parameters
   ============================================================ */
var MCG = { list: null, cur: 0, V: 49 };
function mcgGraphSchematic(canvas){
  var g = cv(canvas), x = g.x, W = g.w, H = g.h;
  x.save(); x.scale(0.84, 0.84);
  var node = function(X, Y, r, f, s){ x.fillStyle = f || '#fff'; x.strokeStyle = s || C.char; x.lineWidth = 1.5; x.beginPath(); x.arc(X, Y, r, 0, 6.3); x.fill(); x.stroke(); }, edge = function(a, b, c){ x.strokeStyle = c || C.bg; x.lineWidth = 1.6; x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke(); }, txt = function(t, X, Y, o){ o = o || {}; x.fillStyle = o.c || C.char; x.font = (o.b ? '700 ' : '') + (o.s || 11.5) + 'px Aptos, "Segoe UI", sans-serif'; x.textAlign = o.a || 'center'; x.fillText(t, X, Y); },
      box = function(X, Y, w, h, f, s, d){ x.fillStyle = f || '#fff'; x.strokeStyle = s || C.char; x.lineWidth = 1.4; if (d) x.setLineDash(d); x.beginPath(); x.rect(X, Y, w, h); x.fill(); x.stroke(); x.setLineDash([]); },
      arrow = function(x1, y1, x2, y2){ x.strokeStyle = C.char; x.lineWidth = 1.3; x.beginPath(); x.moveTo(x1, y1); x.lineTo(x2, y2); x.stroke(); var a = Math.atan2(y2 - y1, x2 - x1); x.fillStyle = C.char; x.beginPath(); x.moveTo(x2, y2); x.lineTo(x2 - 7 * Math.cos(a - 0.4), y2 - 7 * Math.sin(a - 0.4)); x.lineTo(x2 - 7 * Math.cos(a + 0.4), y2 - 7 * Math.sin(a + 0.4)); x.closePath(); x.fill(); };
  /* molecular graph */
  var P = [[44, 78], [84, 56], [124, 78], [124, 122], [84, 144], [44, 122], [164, 56]];
  [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [2, 6]].forEach(function(e){ edge(P[e[0]], P[e[1]]); }); P.forEach(function(p, i){ node(p[0], p[1], 8, i === 6 ? '#FBE9E9' : '#fff', i === 6 ? C.red : C.char); });
  txt('Molecular graph', 100, 22, { b: true }); txt('atoms are nodes, bonds are edges', 100, 38, { s: 10.5, c: C.mid });
  arrow(176, 100, 214, 100);
  /* message passing */
  box(216, 40, 130, 120, C.light, C.bg); txt('Message passing', 281, 58, { b: true, s: 11 }); txt('rounds: each atom updates', 281, 74, { s: 10.5, c: C.mid }); txt('from its bonded neighbours', 281, 88, { s: 10.5, c: C.mid });
  var Q = [[250, 116], [281, 104], [312, 116], [312, 142], [281, 154], [250, 142]]; Q.forEach(function(p, i){ edge(p, Q[(i + 1) % 6], C.bg); }); Q.forEach(function(p, i){ node(p[0], p[1], 6, i === 1 ? '#FBE9E9' : '#fff', i === 1 ? C.red : C.char); });
  arrow(348, 100, 380, 100);
  /* pooled vector */
  txt('Pooled vector', 412, 58, { b: true, s: 11 }); for (var i = 0; i < 6; i++) { box(396, 66 + i * 16, 32, 13, i % 2 ? '#fff' : C.light, C.char); } txt('one vector', 412, 172, { s: 10.5, c: C.mid });
  arrow(432, 100, 462, 100);
  /* heads */
  var heads = [['logD', true], ['CLint', true], ['fup', false], ['Vss', false]]; txt('Output heads', 520, 22, { b: true }); txt('shared representation, one head per endpoint', 520, 38, { s: 10.5, c: C.mid });
  heads.forEach(function(h, k){ var Y = 52 + k * 36, on = h[1]; box(464, Y, 112, 26, on ? '#FBE9E9' : '#fff', on ? C.red : C.bg, on ? null : [4, 3]); txt(h[0] + (on ? '' : ' (no data here)'), 520, Y + 17, { s: 11.5, c: on ? C.char : C.mid }); arrow(436, 100, 464, Y + 13); });
  txt('Red: endpoints with data here (logD, CLint)', 460, 206, { s: 10.5, c: C.dred, a: 'center' }); txt('Dashed: endpoints with no data here', 460, 222, { s: 10.5, c: C.mid, a: 'center' }); x.restore();
}
LESSONS.mlpk.add({ level: 'advanced', short: 'Graph networks to parameters', title: 'Graph networks for PK parameters',
  change: 'Select a test compound. The panel shows the predictions of the course models and the clearance and half-life derived from them.',
  watch: 'A message-passing network (Chemprop, lesson 03; Yang et al., J. Chem. Inf. Model. 2019, 59, 3370) turns a molecular graph into one vector. Several output heads can share that vector, one per endpoint, which is multi-task learning: an endpoint with little data can use what the encoder learned from endpoints with more. For PK, plausible heads are logD, CLint, fraction unbound and Vss. This course has data for logD (AstraZeneca) and HLM CLint (Biogen) only, so fraction unbound and Vss are shown as heads with no data. No graph network runs in this page. The prediction below comes from the descriptor boosted-trees model of lesson 02, which stands in for the CLint head. A predicted CLint is converted to hepatic clearance by IVIVE, and to a half-life with a volume of distribution that is an assumption.',
  build: function(api){
    var gp = api.el('pn', 'left:28px;top:20px;width:560px;height:262px', '<h4>A multi-task graph network</h4><canvas width="532" height="228"></canvas>');
    var rp = api.el('pn', 'left:604px;top:20px;width:328px;height:262px;overflow:auto', '<div class="rs"></div>');
    var bp = api.el('pn', 'left:28px;top:294px;width:904px;height:228px;overflow:auto', '<div class="bt"></div>');
    var ctl = api.addControl('Compound <select class="sel" style="width:300px"></select> <button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var sel = qs('.sel', ctl), ready = false;
    async function load(){
      await mpReady(api); await loadStructs();
      if (!MCG.list) { MCG.list = (STRUCTS.HLM || []).filter(function(e){ return MP.ds.test[e.i] && !MP.ds.cens[e.i]; }); MCG.cur = Math.floor(MCG.list.length / 2); }
      if (!sel.options.length) MCG.list.forEach(function(e, k){ var o = document.createElement('option'); o.value = k; o.textContent = 'Compound ' + (e.i + 1) + ', measured log10 CLint ' + MP.ds.Y[e.i].toFixed(2); sel.appendChild(o); });
      sel.value = String(MCG.cur); ready = true;
    }
    function table(){
      var ex = EXTRAS && EXTRAS.chemprop_ft, refs = (EXTRAS && EXTRAS.refs && EXTRAS.refs.HLM) || [], cp = refs.filter(function(r){ return /^Chemprop$/.test(r[0]); })[0], lg = refs.filter(function(r){ return /LightGBM/.test(r[0]); })[0];
      var avg = function(mode, n){ var r = ex ? ex.runs.filter(function(q){ return q.mode === mode && q.n === n; }) : []; return r.length ? mcMean(r.map(function(q){ return q.r2; })) : null; }, f = function(v){ return v === null ? 'n/a' : v.toFixed(2); };
      qs('.bt', bp).innerHTML = mcPanelText('<div style="display:grid;grid-template-columns:420px 1fr;gap:26px"><div><div class="ol">Saved Chemprop results on HLM clearance, scaffold-split test compounds, R2</div><table style="width:100%;border-collapse:collapse;font-size:12.5px;margin-top:4px"><tr style="font-size:11px;color:' + C.mid + '"><td></td><td style="text-align:right">50 labelled</td><td style="text-align:right">1,000 labelled</td></tr>' +
        ['scratch|From scratch', 'frozen|Frozen encoder pre-trained on logD', 'full|Full fine-tuning from logD'].map(function(s){ var p = s.split('|'); return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:3px 4px">' + p[1] + '</td><td style="text-align:right">' + f(avg(p[0], 50)) + '</td><td style="text-align:right">' + f(avg(p[0], 1000)) + '</td></tr>'; }).join('') +
        '</table><p style="margin:6px 0 0;color:' + C.mid + '">Means of the saved runs in extras.json. Reference run with the full HLM training set: Chemprop ' + (cp ? cp[1].toFixed(2) : 'n/a') + (lg ? ', LightGBM on fingerprints and descriptors ' + lg[1].toFixed(2) : '') + '.</p></div><div><div class="ol">What carries over to PK parameters</div><p style="margin:4px 0 0">The architecture does: a graph encoder with one head per endpoint. The data do not: the numbers above are for HLM CLint, which the course has. A PK parameter such as Vss or fraction unbound needs its own measurements, and the amount of data for those endpoints is usually much smaller than for HLM clearance. The saved fine-tuning results show what that costs: with 50 labelled compounds a network trained from scratch scores close to zero. Full fine-tuning from the logD encoder scores higher than training from scratch at both sizes, while a frozen encoder scores lower at 50 compounds (lesson 05).</p></div></div>', 12);
    }
    function draw(){
      if (!ready) return; var e = MCG.list[MCG.cur], ds = MP.ds, i = e.i, predC = MP.F[i], meas = ds.Y[i];
      var iv = PKE.fromHLM(predC, { bw: 70 }), ivm = PKE.fromHLM(meas, { bw: 70 }), th = function(cl){ return Math.LN2 * MCG.V / cl; }, row = function(l, a, b){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:3px 4px">' + l + '</td><td style="text-align:right;padding:3px 4px">' + a + '</td><td style="text-align:right;padding:3px 4px;color:' + C.red + ';font-weight:700">' + b + '</td></tr>'; };
      qs('.rs', rp).innerHTML = mcPanelText('<div style="height:92px">' + (e.svg ? '<div class="struct" style="height:92px">' + e.svg + '</div>' : '') + '</div><table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:4px"><tr style="font-size:11px;color:' + C.mid + '"><td></td><td style="text-align:right">Measured</td><td style="text-align:right">Predicted</td></tr>' + row('log10 CLint (mL/min/kg)', meas.toFixed(2), predC.toFixed(2)) + row('Hepatic CL, plasma (L/h)', ivm.clhPlasmaLh.toFixed(1), iv.clhPlasmaLh.toFixed(1)) + row('Half-life (h), V ' + MCG.V + ' L', th(ivm.clhPlasmaLh).toFixed(1), th(iv.clhPlasmaLh).toFixed(1)) + row('Fraction unbound', 'no data', 'no data') + row('Vss', 'no data', 'assumed ' + MCG.V + ' L') + '</table><p style="margin:6px 0 0;font-size:11.5px;color:' + C.mid + '">A descriptor model (lesson 02) stands in for the CLint head. IVIVE uses fup 1, blood to plasma ratio 1 and hepatic blood flow 20.7 mL/min/kg as assumptions. V is 0.7 L/kg times 70 kg, an assumption: the course has no Vss data.</p>', 12);
    }
    async function render(){ await load(); await loadExtras(); mcgGraphSchematic(qs('canvas', gp)); table(); draw(); }
    sel.addEventListener('change', function(){ MCG.cur = +sel.value; draw(); });
    qs('[data-next]', ctl).addEventListener('click', function(){ if (!ready) return; MCG.cur = (MCG.cur + 1) % MCG.list.length; sel.value = String(MCG.cur); draw(); });
    async function play(tok){
      await render(); api.cap('A graph encoder turns the molecule into one vector. Each endpoint has its own output head on that vector.'); await api.sleep(3800); api.checkTok(tok);
      api.cap('The course has data for logD and HLM CLint, so those two heads can be trained. Fraction unbound and Vss have no data here.'); await api.sleep(4000); api.checkTok(tok);
      api.cap('A predicted CLint becomes a hepatic clearance by IVIVE, and a half-life once a volume of distribution is assumed.'); await api.sleep(3800); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- guide entries ---------- */
GUIDE.steps['mlpk/Hybrid ODE'] = { notice: 'At the training doses the three fits are close. At 500 mg they diverge.', q: 'Why can the hybrid model extrapolate to a higher dose when the plain network cannot?', a: 'The ODE fixes how absorption, volume and elimination combine, so only the elimination rate as a function of concentration is learned. The plain network has no such structure, and nothing constrains it beyond the training doses.' };
GUIDE.steps['mlpk/Sequence models'] = { notice: 'Boosting on summary features is ahead at small numbers of subjects, and the MLP closes the gap as subjects are added.', q: 'What does a recurrent or attention model accept that a table-based model does not?', a: 'A profile of any length, as an ordered sequence of (time, concentration) pairs, without summarising it or resampling it to a fixed grid.' };
GUIDE.steps['mlpk/Transfer to in vivo'] = { notice: 'At 20 compounds the frozen predictor with a linear head is ahead of both boosting models.', q: 'Why does a two-parameter head on the transferred prediction beat boosting at small n?', a: 'It has few parameters to estimate and inherits what the HLM model learned from far more compounds. Boosting on the descriptors has to learn the relation from the few in vivo compounds.' };
GUIDE.steps['mlpk/Graph networks to parameters'] = { notice: 'Two of the four output heads have data in this course.', q: 'Why use one shared encoder with several output heads?', a: 'The endpoints share one learned representation of the molecule, so an endpoint with little data can use what the encoder learned from endpoints with more data.' };
