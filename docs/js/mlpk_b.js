"use strict";
/* Lesson 11, Intermediate steps. Written for PK-literate scientists who are new to machine learning.
   Steps: Surrogate models, Covariate screening, Ensembles for uncertainty.
   Data: simulations in data/pbpk_sims.json (generated offline with pkengine.js), simulated population data made in the page, and the public Biogen HLM set through loadDataset('HLM'). */

/* ---------- shared helpers (prefix mlb) ---------- */
function mlbSeg(attr, items, cur){ return '<span class="seg" ' + attr + '>' + items.map(function(it, i){ return '<button type="button" data-v="' + i + '"' + (i === cur ? ' class="on"' : '') + '>' + esc(it) + '</button>'; }).join('') + '</span>'; }
function mlbNormal(seed){ var r = rng(seed); return function(){ var u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }; }
function mlbBins(X){
  var n = X.length, p = X[0].length, B = { edges: {}, codes: {} }, feats = [], f, i;
  for (f = 0; f < p; f++) {
    var name = 'f' + f, v = X.map(function(r){ return r[f]; }).sort(function(a, b){ return a - b; }), e = [];
    for (var q = 1; q < 32; q++) { var x = v[Math.floor(q / 32 * n)]; if (!e.length || x > e[e.length - 1]) e.push(x); }
    B.edges[name] = e; var c = new Uint8Array(n); for (i = 0; i < n; i++) c[i] = codeOf(B, name, X[i][f]); B.codes[name] = c; feats.push(name);
  }
  return { B: B, feats: feats };
}
function mlbRow(feats, row){ var o = {}; for (var j = 0; j < feats.length; j++) o[feats[j]] = row[j]; return o; }
/* gradient boosting on a numeric table: squared error, trees from core.js buildTree */
function mlbBoost(X, y, o){
  o = o || {}; var n = y.length, bb = mlbBins(X), B = bb.B, feats = bb.feats, idx = new Int32Array(n), i, m, rounds = o.rounds || 200, lr = o.lr || 0.1, depth = o.depth || 3, minLeaf = o.minLeaf || Math.max(2, Math.min(5, Math.floor(n / 12)));
  for (i = 0; i < n; i++) idx[i] = i; var base = mean(y), F = new Float64Array(n).fill(base), res = new Float32Array(n), rand = rng(3), trees = [];
  for (m = 0; m < rounds; m++) { for (i = 0; i < n; i++) res[i] = y[i] - F[i]; var t = buildTree(B, feats, res, idx, depth, minLeaf, feats.length, rand); trees.push(t); for (i = 0; i < n; i++) F[i] += lr * predictTree(t, B, i); }
  return { predict: function(row){ var v = mlbRow(feats, row), s = base; for (var k = 0; k < trees.length; k++) s += lr * predictVals(trees[k], B, v); return s; },
    gain: function(){ var acc = {}; trees.forEach(function(t){ addGain(t, acc); }); return feats.map(function(f){ return acc[f] || 0; }); } };
}
function mlbForest(X, y, o){
  o = o || {}; var n = y.length, bb = mlbBins(X), B = bb.B, feats = bb.feats, k, i, nt = o.trees || 100, depth = o.depth || 6, mtry = Math.max(1, Math.round(Math.sqrt(feats.length))), rand = rng(o.seed || 7), trees = [];
  for (k = 0; k < nt; k++) { var boot = new Int32Array(n); for (i = 0; i < n; i++) boot[i] = Math.floor(rand() * n); trees.push(buildTree(B, feats, y, boot, depth, 3, mtry, rand)); }
  return { gain: function(){ var acc = {}; trees.forEach(function(t){ addGain(t, acc); }); return feats.map(function(f){ return acc[f] || 0; }); } };
}
/* small ReLU network with two hidden layers, full-batch Adam, inputs and outputs standardised */
function mlbMLP(X, Y, o){
  o = o || {}; var n = X.length, d = X[0].length, k = Y[0].length, h = o.hidden || 24, steps = o.steps || 500, lr = o.lr || 0.01, r = rng(o.seed || 5), j, i, s, t;
  var xm = [], xs = [], ym = [], ys = [];
  for (j = 0; j < d; j++) { var c = X.map(function(row){ return row[j]; }); xm.push(mean(c)); xs.push(Math.sqrt(mean(c.map(function(v){ return (v - xm[j]) * (v - xm[j]); }))) || 1); }
  for (j = 0; j < k; j++) { var cy = Y.map(function(row){ return row[j]; }); ym.push(mean(cy)); ys.push(Math.sqrt(mean(cy.map(function(v){ return (v - ym[j]) * (v - ym[j]); }))) || 1); }
  var Xn = X.map(function(row){ return row.map(function(v, q){ return (v - xm[q]) / xs[q]; }); }), Yn = Y.map(function(row){ return row.map(function(v, q){ return (v - ym[q]) / ys[q]; }); });
  function init(a, b){ var w = new Float64Array(a * b), sd = Math.sqrt(2 / a); for (var q = 0; q < w.length; q++) { var u = 0; while (u === 0) u = r(); w[q] = sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); } return w; }
  var W = [init(d, h), init(h, h), init(h, k)], Bi = [new Float64Array(h), new Float64Array(h), new Float64Array(k)], dims = [[d, h], [h, h], [h, k]];
  var m1 = W.map(function(w){ return new Float64Array(w.length); }), v1 = W.map(function(w){ return new Float64Array(w.length); }), mb = Bi.map(function(b){ return new Float64Array(b.length); }), vb = Bi.map(function(b){ return new Float64Array(b.length); });
  var a1 = new Float64Array(n * h), a2 = new Float64Array(n * h), out = new Float64Array(n * k);
  for (s = 1; s <= steps; s++) {
    var gW = W.map(function(w){ return new Float64Array(w.length); }), gB = Bi.map(function(b){ return new Float64Array(b.length); }), g2 = new Float64Array(n * h), g1 = new Float64Array(n * h), go = new Float64Array(n * k);
    for (i = 0; i < n; i++) {
      var xi = Xn[i], q, z;
      for (q = 0; q < h; q++) { z = Bi[0][q]; for (j = 0; j < d; j++) z += xi[j] * W[0][j * h + q]; a1[i * h + q] = z > 0 ? z : 0; }
      for (q = 0; q < h; q++) { z = Bi[1][q]; for (j = 0; j < h; j++) z += a1[i * h + j] * W[1][j * h + q]; a2[i * h + q] = z > 0 ? z : 0; }
      for (q = 0; q < k; q++) { z = Bi[2][q]; for (j = 0; j < h; j++) z += a2[i * h + j] * W[2][j * k + q]; out[i * k + q] = z; go[i * k + q] = 2 * (z - Yn[i][q]) / (n * k); }
    }
    for (i = 0; i < n; i++) {
      var q2;
      for (q2 = 0; q2 < k; q2++) { var gq = go[i * k + q2]; gB[2][q2] += gq; for (j = 0; j < h; j++) { gW[2][j * k + q2] += a2[i * h + j] * gq; g2[i * h + j] += W[2][j * k + q2] * gq; } }
      for (q2 = 0; q2 < h; q2++) { var gz = a2[i * h + q2] > 0 ? g2[i * h + q2] : 0; if (!gz) continue; gB[1][q2] += gz; for (j = 0; j < h; j++) { gW[1][j * h + q2] += a1[i * h + j] * gz; g1[i * h + j] += W[1][j * h + q2] * gz; } }
      for (q2 = 0; q2 < h; q2++) { var gz1 = a1[i * h + q2] > 0 ? g1[i * h + q2] : 0; if (!gz1) continue; gB[0][q2] += gz1; for (j = 0; j < d; j++) gW[0][j * h + q2] += Xn[i][j] * gz1; }
    }
    var c1 = 1 - Math.pow(0.9, s), c2 = 1 - Math.pow(0.999, s);
    for (t = 0; t < 3; t++) {
      for (j = 0; j < W[t].length; j++) { var g = gW[t][j]; m1[t][j] = 0.9 * m1[t][j] + 0.1 * g; v1[t][j] = 0.999 * v1[t][j] + 0.001 * g * g; W[t][j] -= lr * (m1[t][j] / c1) / (Math.sqrt(v1[t][j] / c2) + 1e-8); }
      for (j = 0; j < Bi[t].length; j++) { var gb = gB[t][j]; mb[t][j] = 0.9 * mb[t][j] + 0.1 * gb; vb[t][j] = 0.999 * vb[t][j] + 0.001 * gb * gb; Bi[t][j] -= lr * (mb[t][j] / c1) / (Math.sqrt(vb[t][j] / c2) + 1e-8); }
    }
  }
  return { predict: function(row){
    var xi = row.map(function(v, q){ return (v - xm[q]) / xs[q]; }), A = new Float64Array(h), A2 = new Float64Array(h), res = [], q, jj, z;
    for (q = 0; q < h; q++) { z = Bi[0][q]; for (jj = 0; jj < d; jj++) z += xi[jj] * W[0][jj * h + q]; A[q] = z > 0 ? z : 0; }
    for (q = 0; q < h; q++) { z = Bi[1][q]; for (jj = 0; jj < h; jj++) z += A[jj] * W[1][jj * h + q]; A2[q] = z > 0 ? z : 0; }
    for (q = 0; q < k; q++) { z = Bi[2][q]; for (jj = 0; jj < h; jj++) z += A2[jj] * W[2][jj * k + q]; res.push(z * ys[q] + ym[q]); }
    return res; } };
}
function mlbR2(y, p){ var m = mean(y), a = 0, b = 0; for (var i = 0; i < y.length; i++) { a += (y[i] - p[i]) * (y[i] - p[i]); b += (y[i] - m) * (y[i] - m); } return b ? 1 - a / b : 0; }
function mlbFold(y, p){ var s = 0, w = 0, L = Math.log10(2); for (var i = 0; i < y.length; i++) { var e = Math.abs(y[i] - p[i]); s += e; if (e <= L) w++; } return { gmfe: Math.pow(10, s / y.length), within2: w / y.length }; }
function mlbAxes(g, L, T, R, B, xr, yr, xt, yt, xl, yl){
  var x = g.x; x.strokeStyle = C.silver; x.fillStyle = C.mid; x.lineWidth = 1; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right';
  yt.forEach(function(v){ var yy = B - (v - yr[0]) / (yr[1] - yr[0]) * (B - T); x.beginPath(); x.moveTo(L, yy); x.lineTo(R, yy); x.stroke(); x.fillText(String(+v.toFixed(2)), L - 6, yy + 4); });
  x.textAlign = 'center'; xt.forEach(function(v){ var xx = L + (v - xr[0]) / (xr[1] - xr[0]) * (R - L); x.fillText(String(+v.toFixed(2)), xx, B + 15); });
  x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText(xl, (L + R) / 2, g.h - 6); x.save(); x.translate(13, (T + B) / 2); x.rotate(-Math.PI / 2); x.fillText(yl, 0, 0); x.restore();
}

/* ---------- 1. Surrogate models ---------- */
var MB1 = { data: null, p: null, res: null, tsim: null };
async function mb1Ready(api){
  if (!MB1.p) MB1.p = (async function(){
    await tick(); var J = await loadJSON('pbpk_sims'), tr = J.train, oo = J.outOfRange;
    function feat(r){ return [r[0], r[1], Math.log10(r[2]), Math.log10(r[3]), r[4], Math.log10(r[5])]; }
    var NPOOL = 450, pool = tr.slice(0, NPOOL), test = tr.slice(NPOOL), sizes = [25, 50, 100, 200, 450];
    var Xt = test.map(feat), Yt = test.map(function(r){ return [Math.log10(r[6]), Math.log10(r[7])]; }), Xo = oo.map(feat), Yo = oo.map(function(r){ return [Math.log10(r[6]), Math.log10(r[7])]; });
    var R = { J: J, sizes: sizes, test: { X: Xt, Y: Yt }, oor: { X: Xo, Y: Yo }, models: {} }, si, tg;
    for (si = 0; si < sizes.length; si++) {
      var n = sizes[si], Xs = pool.slice(0, n).map(feat), Ys = pool.slice(0, n).map(function(r){ return [Math.log10(r[6]), Math.log10(r[7])]; }), out = {};
      out.boost = [0, 1].map(function(c){ return mlbBoost(Xs, Ys.map(function(y){ return y[c]; }), { rounds: 200, lr: 0.1, depth: 3 }); });
      await tick();
      out.mlp = mlbMLP(Xs, Ys, { hidden: 24, steps: 500, lr: 0.01, seed: 5 });
      await tick();
      R.models[n] = out;
    }
    R.pred = function(n, kind, X){ var m = R.models[n]; return kind === 'boost' ? X.map(function(r){ return [m.boost[0].predict(r), m.boost[1].predict(r)]; }) : X.map(function(r){ return m.mlp.predict(r); }); };
    R.cache = {};
    R.get = function(n, kind){ var key = n + kind; if (!R.cache[key]) R.cache[key] = { test: R.pred(n, kind, Xt), oor: R.pred(n, kind, Xo) }; return R.cache[key]; };
    /* time of one simulation in this page, and of one surrogate prediction */
    var t0 = performance.now(), m0 = PKE.pbpk.model({ logP: 2, fup: 0.3, bp: 1, clintU: PKE.pbpk.clintUFromHLM(2, 70, 1), clr: 0, ka: 1, F: 1, bw: 70 });
    PKE.pbpk.simulate(m0, { route: 'oral', dose: 10 }, 48, null, 50); R.tsim = performance.now() - t0;
    var x0 = Xt[0], reps = 2000, t1 = performance.now(), q; for (q = 0; q < reps; q++) { var mm = R.models[450]; mm.boost[0].predict(x0); mm.boost[1].predict(x0); } R.tboost = (performance.now() - t1) / reps;
    t1 = performance.now(); for (q = 0; q < reps; q++) R.models[450].mlp.predict(x0); R.tmlp = (performance.now() - t1) / reps;
    MB1.res = R;
  })();
  api.busy('Training the surrogates on the stored simulations'); try { await MB1.p; } finally { api.idle(); }
}
LESSONS.mlpk.add({ level: 'intermediate', short: 'Surrogate models', title: 'Surrogate models',
  change: 'Select the model, the output and the number of training simulations. Compare the held-out points with the out-of-range points.',
  watch: 'A surrogate model is a machine learning model trained on the inputs and outputs of a simulator, so that a fast prediction replaces a slow simulation. The simulator here is the whole-body PBPK model of lesson 10. Six hundred oral-dose simulations were generated offline with sampled values of CLint, logP, fup, ka, F and dose (ranges in the table). The surrogates predict log10 AUC0-48 h and log10 Cmax from those six inputs: boosted trees, and a neural network with two hidden layers of 24 units. They are trained on a subset of 450 simulations and scored on 150 held-out simulations. Another 60 simulations have inputs outside the sampled ranges and show what a surrogate does where it has no training data. Blood to plasma ratio 1 and zero renal clearance are assumptions held constant.',
  build: function(api){
    var kind = 'boost', tg = 0, nsel = 4, showOut = true;
    var sp = api.el('pn', 'left:28px;top:20px;width:444px;height:330px', '<h4 class="st">Predicted against simulated</h4><canvas width="416" height="268"></canvas>');
    var cp = api.el('pn', 'left:488px;top:20px;width:444px;height:330px', '<h4>Test R2 against training simulations</h4><canvas width="416" height="268"></canvas>');
    var tp = api.el('pn', 'left:28px;top:362px;width:904px;height:158px;overflow:auto', '<div class="tb" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Model ' + mlbSeg('data-k', ['Boosted trees', 'Neural network'], 0) + ' Output ' + mlbSeg('data-t', ['AUC', 'Cmax'], 0) + ' Training simulations ' + mlbSeg('data-n', ['25', '50', '100', '200', '450'], 4) + ' <label style="font-size:12.5px"><input type="checkbox" data-o checked> Show out-of-range</label>');
    function draw(){
      var R = MB1.res; if (!R) return; var n = R.sizes[nsel], P = R.get(n, kind), name = tg ? 'Cmax' : 'AUC', yt = R.test.Y.map(function(r){ return r[tg]; }), pt = P.test.map(function(r){ return r[tg]; }), yo = R.oor.Y.map(function(r){ return r[tg]; }), po = P.oor.map(function(r){ return r[tg]; });
      var all = yt.concat(pt, showOut ? yo.concat(po) : []), lo = Math.floor(Math.min.apply(null, all) * 2) / 2, hi = Math.ceil(Math.max.apply(null, all) * 2) / 2;
      qs('.st', sp).textContent = 'Predicted against simulated log10 ' + name + ', ' + (kind === 'boost' ? 'boosted trees' : 'neural network') + ', ' + n + ' training simulations';
      var g = cv(qs('canvas', sp)), x = g.x, L = 48, Rr = g.w - 14, T = 14, B = g.h - 44, px = function(v){ return L + (v - lo) / (hi - lo) * (Rr - L); }, py = function(v){ return B - (v - lo) / (hi - lo) * (B - T); }, ticks = []; for (var tck = Math.ceil(lo); tck <= hi; tck++) ticks.push(tck);
      mlbAxes(g, L, T, Rr, B, [lo, hi], [lo, hi], ticks, ticks, 'simulated log10 ' + name + (tg ? ' (mg/L)' : ' (mg h/L)'), 'predicted');
      x.strokeStyle = C.char; x.setLineDash([5, 5]); x.beginPath(); x.moveTo(px(lo), py(lo)); x.lineTo(px(hi), py(hi)); x.stroke(); x.setLineDash([]);
      x.fillStyle = 'rgba(52,55,63,.55)'; yt.forEach(function(v, i){ x.beginPath(); x.arc(px(v), py(pt[i]), 3, 0, 6.3); x.fill(); });
      if (showOut) { x.strokeStyle = C.red; x.lineWidth = 1.8; yo.forEach(function(v, i){ x.beginPath(); x.arc(px(v), py(po[i]), 3.6, 0, 6.3); x.stroke(); }); }
      x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillStyle = C.char; x.fillText('Dark: 150 held-out simulations.', L + 8, T + 12); if (showOut) { x.fillStyle = C.red; x.fillText('Red rings: 60 out-of-range simulations.', L + 8, T + 26); }
      /* learning curve */
      var g2 = cv(qs('canvas', cp)), x2 = g2.x, L2 = 48, R2 = g2.w - 14, T2 = 18, B2 = g2.h - 44, lx = function(v){ return L2 + (Math.log10(v) - Math.log10(25)) / (Math.log10(450) - Math.log10(25)) * (R2 - L2); }, ly = function(v){ return B2 - clamp(v, 0, 1) * (B2 - T2); };
      mlbAxes(g2, L2, T2, R2, B2, [0, 1], [0, 1], [], [0, 0.25, 0.5, 0.75, 1], 'training simulations (log scale)', 'test R2 of log10 ' + name);
      x2.textAlign = 'center'; x2.fillStyle = C.mid; x2.font = '11px Aptos, "Segoe UI", sans-serif'; R.sizes.forEach(function(s){ x2.fillText(String(s), lx(s), B2 + 15); });
      ['boost', 'mlp'].forEach(function(kk, ki){ var col = ki ? C.red : C.char; x2.strokeStyle = col; x2.lineWidth = 2.4; x2.setLineDash(ki ? [6, 4] : []); x2.beginPath(); R.sizes.forEach(function(s, i){ var PP = R.get(s, kk), r2 = mlbR2(R.test.Y.map(function(r){ return r[tg]; }), PP.test.map(function(r){ return r[tg]; })); i ? x2.lineTo(lx(s), ly(r2)) : x2.moveTo(lx(s), ly(r2)); }); x2.stroke(); x2.setLineDash([]);
        R.sizes.forEach(function(s){ var PP = R.get(s, kk), r2 = mlbR2(R.test.Y.map(function(r){ return r[tg]; }), PP.test.map(function(r){ return r[tg]; })); x2.fillStyle = col; x2.beginPath(); x2.arc(lx(s), ly(r2), s === n && kk === kind ? 6 : 3.5, 0, 6.3); x2.fill(); }); });
      x2.textAlign = 'left'; x2.font = '600 11px Aptos, "Segoe UI", sans-serif'; x2.fillStyle = C.char; x2.fillText('Boosted trees', L2 + 8, T2 + 12); x2.fillStyle = C.red; x2.fillText('Neural network (dashed)', L2 + 100, T2 + 12);
      /* table */
      var mt = mlbR2(yt, pt), ft = mlbFold(yt, pt), mo = mlbR2(yo, po), fo = mlbFold(yo, po), spd = R.tsim / (kind === 'boost' ? R.tboost : R.tmlp), st = kind === 'boost' ? R.tboost : R.tmlp;
      qs('.tb', tp).innerHTML = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700;text-align:right"><td style="text-align:left">Set for log10 ' + name + '</td><td>Simulations</td><td>R2</td><td>Geometric mean fold error</td><td>Within 2-fold</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Held-out, inside the sampled ranges</td><td>150</td><td><b>' + mt.toFixed(3) + '</b></td><td>' + ft.gmfe.toFixed(2) + '</td><td>' + (100 * ft.within2).toFixed(0) + '%</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + ';color:' + C.dred + '"><td style="text-align:left">Outside the sampled ranges</td><td>60</td><td><b>' + mo.toFixed(3) + '</b></td><td>' + fo.gmfe.toFixed(2) + '</td><td>' + (100 * fo.within2).toFixed(0) + '%</td></tr></table>' +
        '<p style="margin:8px 0 2px">One PBPK simulation took <b>' + (R.tsim / 1000).toFixed(2) + ' s</b> in this page. One surrogate prediction of both outputs took <b>' + (st * 1000).toFixed(0) + ' microseconds</b>, about <b>' + (spd >= 100 ? Math.round(spd).toLocaleString() : spd.toFixed(0)) + ' times</b> faster. Sampled ranges: log10 CLint 0.7 to 3.2, logP 0 to 4, fup 0.01 to 1, ka 0.2 to 3 per h, F 0.3 to 1, dose 1 to 100 mg. Out-of-range inputs: log10 CLint 3.4 to 4.0, logP 4.5 to 5.5, fup 0.002 to 0.008, ka 4 to 8 per h.</p>' +
        '<p style="margin:2px 0;color:' + C.mid + '">A surrogate returns a value for any input. Outside the sampled ranges the trees return the value of the nearest leaf and the network extends its fitted function, so neither is reliable there.</p>';
    }
    qsa('[data-k] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ kind = +b.dataset.v ? 'mlp' : 'boost'; qsa('[data-k] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); draw(); }); });
    qsa('[data-t] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ tg = +b.dataset.v; qsa('[data-t] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); draw(); }); });
    qsa('[data-n] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ nsel = +b.dataset.v; qsa('[data-n] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); draw(); }); });
    qs('[data-o]', ctl).addEventListener('change', function(){ showOut = this.checked; draw(); });
    function setSeg(attr, v){ qsa('[' + attr + '] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === v); }); }
    async function render(){ await mb1Ready(api); draw(); }
    async function play(tok){
      await render(); kind = 'boost'; tg = 0; nsel = 4; showOut = true; qs('[data-o]', ctl).checked = true; setSeg('data-k', 0); setSeg('data-t', 0); setSeg('data-n', 4); draw();
      api.cap('The simulator has six inputs. Each simulation gives AUC0-48 h and Cmax.'); await api.sleep(3000);
      api.cap('Boosted trees trained on 450 simulations reproduce the held-out simulations closely.'); await api.sleep(3200);
      for (var k = 0; k < 4; k++) { api.checkTok(tok); nsel = k; setSeg('data-n', k); draw(); api.cap('Training on ' + MB1.res.sizes[k] + ' simulations.'); await api.sleep(1700); }
      nsel = 4; setSeg('data-n', 4); kind = 'mlp'; setSeg('data-k', 1); draw(); api.cap('The neural network reaches a similar accuracy. Outside the sampled ranges both models lose accuracy.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/Surrogate models'] = { notice: 'The red rings, outside the sampled ranges, fall away from the diagonal for both models.', q: 'Why does a surrogate need a check on its input range?', a: 'It was trained only on the sampled inputs. Outside them it returns a value without any information from the simulator, and the error is not signalled.' };

/* ---------- 2. Covariate screening ---------- */
var MB2 = { cache: {} };
function mlbCov(n, seed, nNoise, sigma){
  var z = mlbNormal(seed), r = rng(seed * 7 + 1), cols = { WT: [], AGE: [], SEX: [], CRCL: [], ALB: [] }, i, k, noise = [];
  for (k = 0; k < nNoise; k++) noise.push([]);
  var logCL = [], logV = [];
  for (i = 0; i < n; i++) {
    var wt = clamp(75 * Math.exp(0.2 * z()), 40, 150), age = clamp(50 + 15 * z(), 18, 85), sex = r() < 0.5 ? 0 : 1, scr = 0.9 * Math.exp(0.2 * z()), alb = clamp(4.2 + 0.4 * z(), 2.5, 5.5);
    var crcl = (140 - age) * wt / (72 * scr) * (sex ? 0.85 : 1);
    cols.WT.push(wt); cols.AGE.push(age); cols.SEX.push(sex); cols.CRCL.push(crcl); cols.ALB.push(alb); for (k = 0; k < nNoise; k++) noise[k].push(z());
    var cl = 5 * Math.pow(wt / 70, 0.75) * Math.pow(crcl / 100, 0.3) * Math.exp(0.25 * z()), v = 40 * (wt / 70) * Math.exp(0.2 * z());
    logCL.push(Math.log(cl) + sigma * z()); logV.push(Math.log(v) + sigma * z());
  }
  var names = ['WT', 'AGE', 'SEX', 'CRCL', 'ALB'], M = [];
  for (k = 0; k < nNoise; k++) names.push('N' + (k + 1));
  for (i = 0; i < n; i++) { var row = [cols.WT[i], cols.AGE[i], cols.SEX[i], cols.CRCL[i], cols.ALB[i]]; for (k = 0; k < nNoise; k++) row.push(noise[k][i]); M.push(row); }
  return { names: names, X: M, logCL: logCL, logV: logV };
}
function mlbOLS(cols, y){
  var n = y.length, p = cols.length + 1, A = [], bvec = new Array(p).fill(0), i, a, b;
  for (a = 0; a < p; a++) A.push(new Array(p).fill(0));
  for (i = 0; i < n; i++) { var row = [1]; for (a = 0; a < cols.length; a++) row.push(cols[a][i]); for (a = 0; a < p; a++) { bvec[a] += row[a] * y[i]; for (b = 0; b < p; b++) A[a][b] += row[a] * row[b]; } }
  for (a = 0; a < p; a++) A[a][a] += 1e-9 * (1 + A[a][a]);
  var M = A.map(function(r, q){ return r.concat([bvec[q]]); }), c, rr, kk;
  for (c = 0; c < p; c++) { var pv = c; for (rr = c + 1; rr < p; rr++) if (Math.abs(M[rr][c]) > Math.abs(M[pv][c])) pv = rr; var tmp = M[c]; M[c] = M[pv]; M[pv] = tmp; for (rr = 0; rr < p; rr++) if (rr !== c) { var f = M[rr][c] / M[c][c]; for (kk = c; kk <= p; kk++) M[rr][kk] -= f * M[c][kk]; } }
  var beta = M.map(function(r, q){ return r[p] / r[q]; }), rss = 0;
  for (i = 0; i < n; i++) { var pr = beta[0]; for (a = 0; a < cols.length; a++) pr += beta[a + 1] * cols[a][i]; rss += (y[i] - pr) * (y[i] - pr); }
  return { rss: rss, beta: beta };
}
/* forward stepwise selection on standardised covariates with AIC or BIC */
function mlbStepwise(X, names, y, crit){
  var n = y.length, p = names.length, cols = [], j, i;
  for (j = 0; j < p; j++) { var c = X.map(function(r){ return r[j]; }), m = mean(c), s = Math.sqrt(mean(c.map(function(v){ return (v - m) * (v - m); }))) || 1; cols.push(c.map(function(v){ return (v - m) / s; })); }
  var sel = [], order = [], pen = crit === 'AIC' ? 2 : Math.log(n), cur = mlbOLS([], y), curScore = n * Math.log(cur.rss / n) + pen * 1;
  for (;;) {
    var best = -1, bs = curScore, bres = null;
    for (j = 0; j < p; j++) { if (sel.indexOf(j) >= 0) continue; var rs = mlbOLS(sel.concat([j]).map(function(q){ return cols[q]; }), y), sc = n * Math.log(rs.rss / n) + pen * (sel.length + 2); if (sc < bs - 1e-9) { bs = sc; best = j; bres = rs; } }
    if (best < 0) break; sel.push(best); order.push({ j: best, gain: curScore - bs }); curScore = bs; cur = bres;
  }
  return { sel: sel, order: order };
}
function mlbShadow(X, rngf){
  var n = X.length, p = X[0].length, perm = [], j, i, S = X.map(function(r){ return r.slice(); });
  for (j = 0; j < p; j++) { var idx = []; for (i = 0; i < n; i++) idx.push(i); for (i = n - 1; i > 0; i--) { var k = Math.floor(rngf() * (i + 1)), t = idx[i]; idx[i] = idx[k]; idx[k] = t; } for (i = 0; i < n; i++) S[i].push(X[idx[i]][j]); }
  return S;
}
/* tree-ensemble screening: split-gain importance of each covariate against the largest importance of a permuted (shadow) copy */
function mlbTreeScreen(X, y, kind, seed, light){
  var p = X[0].length, S = mlbShadow(X, rng(seed + 99)), gain, tot, imp, shMax, sel = [], j;
  if (kind === 'rf') gain = mlbForest(S, y, { trees: light ? 60 : 200, depth: 6, seed: seed }).gain(); else gain = mlbBoost(S, y, { rounds: light ? 80 : 150, lr: 0.08, depth: 3, minLeaf: 5 }).gain();
  tot = gain.reduce(function(a, b){ return a + b; }, 0) || 1; imp = gain.slice(0, p).map(function(v){ return v / tot; }); shMax = Math.max.apply(null, gain.slice(p).map(function(v){ return v / tot; }));
  for (j = 0; j < p; j++) if (imp[j] > shMax) sel.push(j);
  return { imp: imp, shadow: shMax, sel: sel };
}
LESSONS.mlpk.add({ level: 'intermediate', short: 'Covariate screening', title: 'Covariate screening',
  change: 'Set the number of subjects, the noise level and the number of noise covariates. Select Repeat over 30 seeds to see how stable each method is.',
  watch: 'Covariate screening asks which subject characteristics explain between-subject variability in a PK parameter. The data here are simulated, so the true model is known. Clearance is CL = 5 L/h times (WT/70)^0.75 times (CRCL/100)^0.3 times exp(eta), with eta normal with SD 0.25. Volume is V = 40 L times (WT/70) times exp(eta), with eta normal with SD 0.2. The covariates are weight (WT), age (AGE), sex (SEX), creatinine clearance (CRCL), albumin (ALB) and a chosen number of pure noise variables. CRCL is computed from age, weight, sex and serum creatinine with the Cockcroft-Gault equation, so it is correlated with WT, AGE and SEX. Only WT and CRCL affect CL, and only WT affects V. The screened quantity is the log of an individual estimate with added estimation noise. Three methods are compared: forward stepwise linear regression with AIC or BIC, a random forest, and gradient boosting. For the two tree ensembles a covariate is selected if its split-gain importance exceeds that of the most important permuted copy of any covariate, a simplified form of the shadow-feature rule of the Boruta method.',
  build: function(api){
    var N = 200, sigma = 0.15, NN = 4, seedI = 1, tg = 0, crit = 'BIC', rep = null;
    var lp = api.el('pn', 'left:28px;top:20px;width:904px;height:326px', '<h4 class="ct">Importance and selection</h4><div class="cb"></div>');
    var rp = api.el('pn', 'left:28px;top:358px;width:904px;height:162px;overflow:auto', '<div class="cr" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Subjects <input class="n" type="range" min="50" max="1000" step="50" value="200" style="width:110px"> <b class="nv">200</b> Noise SD <input class="s" type="range" min="0.05" max="0.5" step="0.05" value="0.15" style="width:90px"> <b class="sv">0.15</b> Noise covariates <input class="k" type="range" min="0" max="8" step="1" value="4" style="width:80px"> <b class="kv">4</b> Parameter ' + mlbSeg('data-t', ['CL', 'V'], 0) + ' Criterion ' + mlbSeg('data-c', ['BIC', 'AIC'], 0) + ' <button type="button" class="btn sm ghost" data-seed>New draw</button> <button type="button" class="btn sm" data-rep>Repeat over 30 seeds</button>');
    var nI = qs('.n', ctl), sI = qs('.s', ctl), kI = qs('.k', ctl);
    var TRUE = function(){ return tg ? ['WT'] : ['WT', 'CRCL']; };
    function run(){
      var key = [N, sigma, NN, seedI, tg, crit].join('|'); if (MB2.cache[key]) return MB2.cache[key];
      var D0 = mlbCov(N, seedI, NN, sigma), y = tg ? D0.logV : D0.logCL;
      var sw = mlbStepwise(D0.X, D0.names, y, crit), rf = mlbTreeScreen(D0.X, y, 'rf', seedI), gb = mlbTreeScreen(D0.X, y, 'gb', seedI);
      return (MB2.cache[key] = { D: D0, sw: sw, rf: rf, gb: gb });
    }
    function draw(){
      var R = run(), names = R.D.names, p = names.length, tr = TRUE(), g = cv(qs('canvas', lp) || (function(){ var c = document.createElement('canvas'); c.width = 876; c.height = 280; qs('.cb', lp).appendChild(c); return c; })()), x = g.x, W = g.w;
      var colsW = (W - 40) / 3, mx = 0.01;
      var methods = [{ n: 'Forward stepwise regression, ' + crit, imp: names.map(function(_, j){ var o = R.sw.order.filter(function(q){ return q.j === j; })[0]; return o ? o.gain : 0; }), sel: R.sw.sel, ref: 0, lab: 'Improvement in ' + crit + ' when added' }, { n: 'Random forest', imp: R.rf.imp, sel: R.rf.sel, ref: R.rf.shadow, lab: 'Split-gain importance (share)' }, { n: 'Gradient boosting', imp: R.gb.imp, sel: R.gb.sel, ref: R.gb.shadow, lab: 'Split-gain importance (share)' }];
      x.clearRect(0, 0, g.w, g.h);
      methods.forEach(function(m, mi){
        var x0 = 10 + mi * (colsW + 10), top = 30, bot = g.h - 24, rowH = (bot - top) / p, maxv = Math.max.apply(null, m.imp.concat([m.ref, 1e-9])) * 1.15, barL = x0 + 52, barW = colsW - 66;
        x.fillStyle = C.char; x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText(m.n, x0, 14);
        names.forEach(function(nm, j){
          var y0 = top + j * rowH, isTrue = tr.indexOf(nm) >= 0, sel = m.sel.indexOf(j) >= 0, w = m.imp[j] / maxv * barW;
          x.fillStyle = isTrue ? C.dred : C.char; x.font = (isTrue ? '700 ' : '') + '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right'; x.fillText(nm, x0 + 44, y0 + rowH * 0.62);
          x.fillStyle = sel ? (isTrue ? '#2E7D4F' : C.red) : C.bg; x.fillRect(barL, y0 + 2, Math.max(0, w), rowH - 5);
          x.fillStyle = C.char; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText(sel ? (isTrue ? 'true' : 'false positive') : (isTrue ? 'missed' : ''), barL + Math.max(0, w) + 4, y0 + rowH * 0.62);
        });
        if (m.ref) { var xr = barL + m.ref / maxv * barW; x.strokeStyle = C.red; x.setLineDash([4, 3]); x.lineWidth = 1.4; x.beginPath(); x.moveTo(xr, top - 4); x.lineTo(xr, bot); x.stroke(); x.setLineDash([]); x.fillStyle = C.red; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText('largest permuted copy', Math.min(xr + 3, barL + barW - 90), bot + 14); }
        else { x.fillStyle = C.mid; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText(m.lab, barL, bot + 14); }
      });
      var ev = methods.map(function(m){ var tp = tr.filter(function(t){ return m.sel.indexOf(names.indexOf(t)) >= 0; }).length, fp = m.sel.filter(function(j){ return tr.indexOf(names[j]) < 0; }).length; return { n: m.n, tp: tp, fp: fp, fn: tr.length - tp, sel: m.sel.map(function(j){ return names[j]; }) }; });
      var html = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700;text-align:right"><td style="text-align:left">Method, ' + (tg ? 'log V' : 'log CL') + ', true covariates ' + tr.join(' and ') + ', seed ' + seedI + '</td><td>True found</td><td>False positives</td><td>Selected</td></tr>' + ev.map(function(e){ return '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">' + esc(e.n) + '</td><td>' + e.tp + ' of ' + tr.length + '</td><td>' + e.fp + '</td><td style="text-align:left;padding-left:14px">' + (e.sel.join(', ') || 'none') + '</td></tr>'; }).join('') + '</table>';
      if (rep) html += '<p style="margin:8px 0 2px"><b>Over ' + rep.n + ' seeds</b> at these settings (tree ensembles with fewer trees and rounds for speed): ' + rep.rows.map(function(r){ return esc(r.n.split(',')[0]) + ': all true found in ' + Math.round(100 * r.all) + '% of seeds, mean false positives ' + r.fp.toFixed(2); }).join('; ') + '.</p>';
      else html += '<p style="margin:8px 0 2px;color:' + C.mid + '">All data are simulated. The true covariates are marked in dark red. CRCL is correlated with AGE, WT and SEX, so a covariate with no effect of its own can look important.</p>';
      qs('.cr', rp).innerHTML = html; qs('.ct', lp).textContent = 'Importance and selection of covariates for ' + (tg ? 'log volume of distribution' : 'log clearance') + ', ' + N + ' simulated subjects';
    }
    async function repeat(){
      var btn = qs('[data-rep]', ctl); btn.disabled = true; var n = 30, acc = [{ n: 'Forward stepwise regression', all: 0, fp: 0 }, { n: 'Random forest', all: 0, fp: 0 }, { n: 'Gradient boosting', all: 0, fp: 0 }], tr = TRUE(), s;
      for (s = 1; s <= n; s++) {
        api.busy('Screening, seed ' + s + ' of ' + n); await tick();
        var D0 = mlbCov(N, 1000 + s, NN, sigma), y = tg ? D0.logV : D0.logCL, names = D0.names, sets = [mlbStepwise(D0.X, names, y, crit).sel, mlbTreeScreen(D0.X, y, 'rf', 1000 + s, true).sel, mlbTreeScreen(D0.X, y, 'gb', 1000 + s, true).sel];
        sets.forEach(function(sel, i){ var nm = sel.map(function(j){ return names[j]; }); if (tr.every(function(t){ return nm.indexOf(t) >= 0; })) acc[i].all++; acc[i].fp += nm.filter(function(q){ return tr.indexOf(q) < 0; }).length; });
      }
      api.idle(); rep = { n: n, rows: acc.map(function(a){ return { n: a.n, all: a.all / n, fp: a.fp / n }; }) }; btn.disabled = false; draw();
    }
    function upd(){ N = +nI.value; sigma = +sI.value; NN = +kI.value; qs('.nv', ctl).textContent = N; qs('.sv', ctl).textContent = sigma.toFixed(2); qs('.kv', ctl).textContent = NN; rep = null; draw(); }
    [nI, sI, kI].forEach(function(e){ e.addEventListener('change', upd); e.addEventListener('input', function(){ qs('.nv', ctl).textContent = nI.value; qs('.sv', ctl).textContent = (+sI.value).toFixed(2); qs('.kv', ctl).textContent = kI.value; }); });
    qsa('[data-t] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ tg = +b.dataset.v; qsa('[data-t] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); rep = null; draw(); }); });
    qsa('[data-c] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ crit = +b.dataset.v ? 'AIC' : 'BIC'; qsa('[data-c] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); rep = null; draw(); }); });
    qs('[data-seed]', ctl).addEventListener('click', function(){ seedI++; rep = null; draw(); });
    qs('[data-rep]', ctl).addEventListener('click', repeat);
    async function render(){ api.busy('Simulating subjects and screening covariates'); await tick(); try { draw(); } finally { api.idle(); } }
    async function play(tok){
      N = 200; sigma = 0.15; NN = 4; tg = 0; crit = 'BIC'; rep = null; nI.value = 200; sI.value = 0.15; kI.value = 4; qs('.nv', ctl).textContent = 200; qs('.sv', ctl).textContent = '0.15'; qs('.kv', ctl).textContent = 4; qsa('[data-t] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === 0); }); draw();
      api.cap('Weight and creatinine clearance are the covariates that act on clearance in the simulated model.'); await api.sleep(3200);
      N = 50; nI.value = 50; qs('.nv', ctl).textContent = 50; draw(); api.cap('With 50 subjects a method can miss a true covariate or select a covariate with no effect.'); await api.sleep(3600);
      N = 600; nI.value = 600; qs('.nv', ctl).textContent = 600; draw(); api.cap('With 600 subjects the true covariates are recovered more reliably. CRCL, which is correlated with WT and AGE, remains the hard case.'); await api.sleep(3800); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/Covariate screening'] = { notice: 'The permuted-copy line is the importance a covariate with no effect can reach by chance.', q: 'Why is split-gain importance compared with permuted copies of the covariates?', a: 'Split gain is nonzero for a covariate with no effect, because a tree always finds some split. The largest importance reached by permuted copies estimates how large a chance value can be.' };

/* ---------- 3. Ensembles for uncertainty ---------- */
var MB3 = { p: null, R: null };
function mlbZ(p){ /* normal quantile, Abramowitz and Stegun 26.2.23, absolute error below 4.5e-4 */
  var q = p > 0.5 ? 1 - p : p, t = Math.sqrt(-2 * Math.log(q)), z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t); return p > 0.5 ? z : -z;
}
async function mb3Ready(api){
  if (!MB3.p) MB3.p = (async function(){
    await tick(); var ds = await loadDataset('HLM'); await loadStructs(); await tick();
    var feats = ds.BASE.slice(), r = rng(11), sh = ds.TRAIN.slice(), i, a;
    for (a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
    var nCal = Math.round(sh.length * 0.15), cal = sh.slice(0, nCal), fit = sh.slice(nCal), NB = 20, preds = [], rb = rng(21);
    for (var b = 0; b < NB; b++) { var boot = new Array(fit.length); for (i = 0; i < fit.length; i++) boot[i] = fit[Math.floor(rb() * fit.length)]; preds.push(fitBoost(feats, boot, 150, 0.08, 3, 0, ds).F); await tick(); }
    var N = ds.N, mu = new Float64Array(N), sd = new Float64Array(N);
    for (i = 0; i < N; i++) { var s = 0, q; for (q = 0; q < NB; q++) s += preds[q][i]; mu[i] = s / NB; var v = 0; for (q = 0; q < NB; q++) v += (preds[q][i] - mu[i]) * (preds[q][i] - mu[i]); sd[i] = Math.sqrt(v / (NB - 1)); }
    var cres = cal.map(function(k){ return ds.Y[k] - mu[k]; }), sdRes = Math.sqrt(mean(cres.map(function(v){ return v * v; })));
    var q2 = ds.TEST.filter(function(k){ return !ds.cens[k]; });
    MB3.R = { ds: ds, mu: mu, sd: sd, sdRes: sdRes, test: q2, NB: NB, nCal: nCal, nFit: fit.length, rmse: Math.sqrt(mean(q2.map(function(k){ return (ds.Y[k] - mu[k]) * (ds.Y[k] - mu[k]); }))), meanSd: mean(q2.map(function(k){ return sd[k]; })),
      list: (STRUCTS.HLM || []).filter(function(e){ return ds.test[e.i] && !ds.cens[e.i]; }) };
  })();
  api.busy('Fitting 20 bootstrap boosted-trees models on the HLM training compounds'); try { await MB3.p; } finally { api.idle(); }
}
LESSONS.mlpk.add({ level: 'intermediate', short: 'Ensembles for uncertainty', title: 'Ensembles for parameter uncertainty',
  change: 'Select the nominal level and a test compound. Compare the interval from the ensemble alone with the interval that includes residual error.',
  watch: 'An ensemble of models trained on bootstrap samples gives a spread of predictions for each compound. The spread measures how much the prediction depends on the training sample. It does not include assay noise or the bias of the model, so it understates the total predictive error. Here 20 boosted-trees models are fitted to bootstrap samples of the HLM training compounds (85% of them). The interval from the ensemble alone is the mean plus and minus a normal quantile times the ensemble standard deviation. The second interval adds the root mean squared error of the ensemble mean on 15% of the training compounds that no model was fitted to. Coverage is the fraction of measured values that fall inside the interval, on the test compounds with a quantified value. A calibrated interval has coverage equal to the nominal level. The chosen interval is carried through IVIVE (lesson 10) and a one-compartment intravenous model, with fup 1, blood to plasma ratio 1 and a 100 mg dose as assumptions, to give a range of AUC.',
  build: function(api){
    var lev = 2, cur = 0, LV = [0.5, 0.8, 0.9, 0.95], DOSE = 100;
    var cp = api.el('pn', 'left:28px;top:20px;width:400px;height:330px', '<h4>Coverage against nominal level</h4><canvas width="372" height="290"></canvas>');
    var ap = api.el('pn', 'left:440px;top:20px;width:492px;height:330px', '<h4 class="at">Selected test compound</h4><canvas width="464" height="150"></canvas><div class="ai" style="font-size:12.5px;line-height:1.5;color:' + C.char + ';margin-top:4px"></div>');
    var tp = api.el('pn', 'left:28px;top:362px;width:904px;height:158px;overflow:auto', '<div class="tt" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Nominal level ' + mlbSeg('data-l', ['50%', '80%', '90%', '95%'], 2) + ' <button type="button" class="btn sm ghost" data-next>Next compound</button>');
    function cover(p, withRes){ var R = MB3.R, z = mlbZ((1 + p) / 2), c = 0; R.test.forEach(function(k){ var h = z * (withRes ? Math.sqrt(R.sd[k] * R.sd[k] + R.sdRes * R.sdRes) : R.sd[k]); if (Math.abs(R.ds.Y[k] - R.mu[k]) <= h) c++; }); return c / R.test.length; }
    function auc(l10){ return DOSE / PKE.fromHLM(l10, { bw: 70 }).clhPlasmaLh; }
    function draw(){
      var R = MB3.R; if (!R) return; var g = cv(qs('canvas', cp)), x = g.x, L = 46, Rr = g.w - 14, T = 14, B = g.h - 44, px = function(v){ return L + v * (Rr - L); }, py = function(v){ return B - v * (B - T); };
      mlbAxes(g, L, T, Rr, B, [0, 1], [0, 1], [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], 'nominal level', 'observed coverage');
      x.strokeStyle = C.bg; x.setLineDash([5, 4]); x.lineWidth = 1.4; x.beginPath(); x.moveTo(px(0), py(0)); x.lineTo(px(1), py(1)); x.stroke(); x.setLineDash([]);
      var grid = []; for (var q = 0.05; q < 0.995; q += 0.05) grid.push(q);
      [[false, C.char, 'Ensemble spread only'], [true, C.red, 'Spread plus residual error']].forEach(function(m, mi){ x.strokeStyle = m[1]; x.lineWidth = 2.4; x.beginPath(); grid.forEach(function(p, i){ var c = cover(p, m[0]); i ? x.lineTo(px(p), py(c)) : x.moveTo(px(p), py(c)); }); x.stroke(); x.fillStyle = m[1]; x.beginPath(); x.arc(px(LV[lev]), py(cover(LV[lev], m[0])), 5, 0, 6.3); x.fill(); x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText(m[2], L + 8, T + 12 + mi * 14); });
      x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillText('Dashed: a calibrated interval.', L + 8, T + 44);
      /* compound */
      var e = R.list[cur % R.list.length], k = e.i, y = R.ds.Y[k], mu = R.mu[k], sd = R.sd[k], z = mlbZ((1 + LV[lev]) / 2), hA = z * sd, hB = z * Math.sqrt(sd * sd + R.sdRes * R.sdRes);
      var g2 = cv(qs('canvas', ap)), x2 = g2.x, L2 = 20, R2 = g2.w - 20, lo = Math.min(y, mu - hB) - 0.2, hi = Math.max(y, mu + hB) + 0.2, p2 = function(v){ return L2 + (v - lo) / (hi - lo) * (R2 - L2); };
      x2.clearRect(0, 0, g2.w, g2.h); x2.strokeStyle = C.silver; x2.lineWidth = 1; x2.beginPath(); x2.moveTo(L2, 112); x2.lineTo(R2, 112); x2.stroke();
      x2.fillStyle = C.mid; x2.font = '11px Aptos, "Segoe UI", sans-serif'; x2.textAlign = 'center'; for (var tk = Math.ceil(lo * 2) / 2; tk <= hi; tk += 0.5) { x2.fillText(tk.toFixed(1), p2(tk), 126); x2.beginPath(); x2.moveTo(p2(tk), 108); x2.lineTo(p2(tk), 112); x2.stroke(); } x2.fillText('log10 CLint (mL/min/kg)', (L2 + R2) / 2, 146);
      [[hB, 'rgba(225,36,42,.22)', 'spread plus residual', 60, 44], [hA, 'rgba(52,55,63,.45)', 'ensemble spread only', 84, 22]].forEach(function(b){ x2.fillStyle = b[1]; x2.fillRect(p2(mu - b[0]), b[3], p2(mu + b[0]) - p2(mu - b[0]), 16); x2.fillStyle = C.char; x2.font = '10.5px Aptos, "Segoe UI", sans-serif'; x2.textAlign = 'left'; x2.fillText(b[2], p2(mu + b[0]) + 6, b[3] + 12); });
      x2.fillStyle = C.char; x2.beginPath(); x2.arc(p2(mu), 80, 4, 0, 6.3); x2.fill(); x2.strokeStyle = C.red; x2.lineWidth = 2.4; x2.beginPath(); x2.moveTo(p2(y), 40); x2.lineTo(p2(y), 108); x2.stroke(); x2.fillStyle = C.red; x2.font = '600 11px Aptos, "Segoe UI", sans-serif'; x2.textAlign = 'center'; x2.fillText('measured', p2(y), 34);
      var aM = auc(y), aP = auc(mu), aA = [auc(mu + hA), auc(mu - hA)], aB = [auc(mu + hB), auc(mu - hB)];
      qs('.at', ap).textContent = 'Compound ' + (k + 1) + ', ' + Math.round(LV[lev] * 100) + '% intervals for log10 CLint';
      qs('.ai', ap).innerHTML = 'Measured ' + y.toFixed(2) + ', ensemble mean ' + mu.toFixed(2) + ', ensemble SD ' + sd.toFixed(2) + '. Predicted AUC ' + aP.toFixed(2) + ' mg h/L; AUC at the measured CLint ' + aM.toFixed(2) + '. Range of AUC: <b>' + aA[0].toFixed(2) + ' to ' + aA[1].toFixed(2) + '</b> from the spread alone (' + (aA[1] / aA[0]).toFixed(1) + '-fold), <b>' + aB[0].toFixed(2) + ' to ' + aB[1].toFixed(2) + '</b> with residual error (' + (aB[1] / aB[0]).toFixed(1) + '-fold). Assumptions: IV bolus of ' + DOSE + ' mg, one compartment, AUC = dose divided by plasma CL from the well-stirred model, fup 1, blood to plasma ratio 1, 70 kg.';
      var cA = cover(LV[lev], false), cB = cover(LV[lev], true);
      qs('.tt', tp).innerHTML = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700;text-align:right"><td style="text-align:left">Quantity, ' + R.test.length + ' test compounds with a quantified value</td><td>Value</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Mean ensemble SD</td><td>' + R.meanSd.toFixed(2) + ' log10 units</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">RMSE of the ensemble mean on the test compounds</td><td>' + R.rmse.toFixed(2) + ' log10 units</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Residual SD on the ' + R.nCal + ' calibration compounds</td><td>' + R.sdRes.toFixed(2) + ' log10 units</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Coverage at the ' + Math.round(LV[lev] * 100) + '% nominal level: ensemble spread only</td><td><b style="color:' + C.dred + '">' + Math.round(100 * cA) + '%</b></td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Coverage at the ' + Math.round(LV[lev] * 100) + '% nominal level: spread plus residual error</td><td><b>' + Math.round(100 * cB) + '%</b></td></tr></table>' +
        '<p style="margin:8px 0 2px">The ensemble spread is about ' + (R.rmse / R.meanSd).toFixed(0) + ' times smaller than the error of its mean on new compounds. It reflects the sensitivity of the model to the training sample, not assay noise or model bias, so intervals built from it alone are too narrow. Adding the residual error from compounds not used for fitting brings the coverage close to the nominal level, on the assumption that the residual error is similar for new compounds and roughly normal. Compounds outside the training chemistry can violate that assumption.</p>';
    }
    qsa('[data-l] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ lev = +b.dataset.v; qsa('[data-l] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); draw(); }); });
    qs('[data-next]', ctl).addEventListener('click', function(){ cur++; draw(); });
    async function render(){ await mb3Ready(api); draw(); }
    async function play(tok){
      await render(); lev = 2; cur = 0; qsa('[data-l] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === 2); }); draw();
      api.cap('Twenty models fitted to bootstrap samples disagree a little for each compound.'); await api.sleep(3000);
      api.cap('At the 90% nominal level the ensemble spread alone covers far fewer measured values than 90%.'); await api.sleep(3600);
      api.cap('Adding the residual error from compounds not used for fitting gives coverage near the nominal level.'); await api.sleep(3600);
      cur = 1; draw(); api.cap('For one compound the two intervals give very different ranges of AUC.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/Ensembles for uncertainty'] = { notice: 'The ensemble-only coverage curve lies far below the dashed line.', q: 'Why does the spread of a bootstrap ensemble understate the error on new compounds?', a: 'The spread measures sensitivity to the training sample. It does not include assay noise or the systematic error of the model, which together dominate the error on new compounds.' };
