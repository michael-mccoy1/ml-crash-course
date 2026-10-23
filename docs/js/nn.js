"use strict";
/* Lesson 4: Neural networks (engine and steps 1 to 5) */
var L3 = new Lesson('nn', 'Neural networks');
L3.optsHTML = function(){
  return sharedOptions() +
    optRange('nlayers', 'Hidden layers', 1, 4, 1, 2, 'Stages between inputs and output.') +
    optRange('nunits', 'Neurons per layer', 4, 128, 4, 32, 'Width of each hidden layer.') +
    optRange('ndrop', 'Dropout', 0, 0.6, 0.05, 0.1, 'Share of neurons switched off each step.') +
    optRange('nlr', 'Learning rate', 0.0005, 0.02, 0.0005, 0.003, 'Size of each weight update.') +
    optRange('nepochs', 'Epochs', 10, 120, 10, 40, 'Passes through the training set.') +
    '<div class="og"><label style="display:flex;gap:8px;align-items:center;font-size:.86rem;font-weight:600"><input type="checkbox" id="nearly" checked style="accent-color:#E1242A"> Keep the best epoch</label><div class="hint">Restore the epoch with the lowest validation loss.</div></div>' +
    '<div class="og"><div class="ol">Last training</div><div class="rbox" id="nres"><div class="hint">Press Train in the workbench.</div></div></div>' + filterGroup();
};
function NP(){ return { layers: +$('nlayers').value, units: +$('nunits').value, drop: +$('ndrop').value, lr: +$('nlr').value, epochs: +$('nepochs').value, early: $('nearly').checked }; }
var NN = { m: null, params: null, feats: null, prep: null, prepKey: null, hist: null, pred: null, best: 0, secs: null, sel: 0, act: null, r2: null, busy: false, sig: null };

function nnSig(){ var p = NP(); return JSON.stringify([p.layers, p.units, p.drop, p.lr, p.epochs, p.early, activeFeats(), EP, filtKey()]); }
function nnStale(){ return !NN.m || nnSig() !== NN.sig; }
function nnResultsBox(){ var el = $('nres'); if (!el) return; if (!NN.m) { el.innerHTML = '<div class="hint">Press Train in the workbench.</div>'; return; } el.innerHTML = '<div class="rrow">Unseen R2<b>' + NN.r2.toFixed(2) + '</b></div><div class="rrow">Best epoch<b>' + NN.best + '</b></div><div class="rrow">Training time<b>' + fmtT(NN.secs) + '</b></div>' + (nnStale() ? '<div class="hint" style="color:#E1242A">Options changed since this training.</div>' : ''); }
var nTimer = null;
L3.onOpts = function(root){
  var fire = function(){ clearTimeout(nTimer); nTimer = setTimeout(function(){ nnResultsBox(); L3.refresh(); }, 250); };
  ['nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs'].forEach(function(id){ $(id).addEventListener('input', function(){ if (L3.live) L3.live(); fire(); }); }); $('nearly').addEventListener('change', function(){ if (L3.live) L3.live(); fire(); });
};
L3.onData = function(){ NN.m = null; NN.prep = null; NN.pc = null; NN.prepKey = null; nnResultsBox(); };
featListeners.push(function(){ if (L3.built) { nnResultsBox(); L3.refresh(); } });

/* ---------- engine: a small multilayer perceptron written by hand (fast for these sizes) ---------- */
function gauss(r){ var u = r() || 1e-9, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function netSizes(nin, layers, units){ var s = [nin]; for (var i = 0; i < layers; i++) s.push(units); s.push(1); return s; }
function mlpInit(sizes, seed){ var r = rng(seed), W = [], b = []; for (var l = 0; l < sizes.length - 1; l++) { var nin = sizes[l], nout = sizes[l + 1], w = new Float32Array(nin * nout), sd = Math.sqrt(2 / nin); for (var i = 0; i < w.length; i++) w[i] = gauss(r) * sd; W.push(w); b.push(new Float32Array(nout)); } return { sizes: sizes, W: W, b: b }; }
function mlpBuf(sizes){ return sizes.map(function(n){ return new Float32Array(n); }); }
function mlpForward(m, X, off, A, masks, drop, r){
  var S = m.sizes, L = S.length - 1, a0 = A[0], F = S[0]; for (var i = 0; i < F; i++) a0[i] = X[off + i];
  for (var l = 0; l < L; l++) { var nin = S[l], nout = S[l + 1], W = m.W[l], bb = m.b[l], inp = A[l], out = A[l + 1];
    for (var j = 0; j < nout; j++) out[j] = bb[j];
    for (var i2 = 0; i2 < nin; i2++) { var v = inp[i2]; if (v === 0) continue; var base = i2 * nout; for (var j2 = 0; j2 < nout; j2++) out[j2] += v * W[base + j2]; }
    if (l < L - 1) { var mk = masks ? masks[l] : null; for (var j3 = 0; j3 < nout; j3++) { var z = out[j3] > 0 ? out[j3] : 0; if (mk) { var keep = !(drop > 0) || r() >= drop; mk[j3] = keep ? 1 / (1 - drop) : 0; z *= mk[j3]; } out[j3] = z; } } }
  return A[L][0];
}
function mlpMSE(m, X, y, n, A){ var F = m.sizes[0], s = 0; for (var k = 0; k < n; k++) { var e = mlpForward(m, X, k * F, A, null, 0, null) - y[k]; s += e * e; } return s / n; }
function mlpCopy(m){ return { sizes: m.sizes, W: m.W.map(function(w){ return new Float32Array(w); }), b: m.b.map(function(b){ return new Float32Array(b); }) }; }
function nnPrep(f){
  var feats = activeFeats(), idx = activeTrain(f), key = JSON.stringify([EP, feats, idx.length, filtKey(f)]); if (NN.prepKey === key && NN.pc) return NN.pc;
  var mu = {}, sd = {}; feats.forEach(function(f){ var v = idx.map(function(i){ return D.COL[f][i]; }); mu[f] = mean(v); sd[f] = Math.sqrt(mean(v.map(function(x){ return (x - mu[f]) * (x - mu[f]); }))) || 1; });
  var ymu = mean(idx.map(function(i){ return D.Y[i]; })), F = feats.length, rowOf = function(i){ return feats.map(function(f){ return clamp((D.COL[f][i] - mu[f]) / sd[f], -5, 5); }); };
  var sh = idx.slice(), r = rng(11); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
  var nv = Math.round(sh.length * 0.15), va = sh.slice(0, nv), tn = sh.slice(nv);
  function pack(ix){ var X = new Float32Array(ix.length * F), y = new Float32Array(ix.length); ix.forEach(function(i, k){ var rw = rowOf(i); for (var q = 0; q < F; q++) X[k * F + q] = rw[q]; y[k] = D.Y[i] - ymu; }); return { X: X, y: y, n: ix.length }; }
  var all = new Float32Array(D.N * F); for (var i = 0; i < D.N; i++) { var rw = rowOf(i); for (var q = 0; q < F; q++) all[i * F + q] = rw[q]; }
  NN.pc = { feats: feats, mu: mu, sd: sd, ymu: ymu, rowOf: rowOf, tn: tn, va: va, T: pack(tn), V: pack(va), all: all, F: F }; NN.prepKey = key; return NN.pc;
}
function mlpPredictAll(m, P){ var A = mlpBuf(m.sizes), out = new Float64Array(D.N); for (var i = 0; i < D.N; i++) out[i] = mlpForward(m, P.all, i * P.F, A, null, 0, null) + P.ymu; return out; }
async function nnTrain(o){
  var P = nnPrep(o.filt), p = Object.assign(NP(), o.params || {}), sizes = netSizes(P.F, p.layers, p.units), m = mlpInit(sizes, 7 + p.layers * 13 + p.units), S = sizes, L = S.length - 1, r = rng(5 + p.units), tN = performance.now(), over = 0, sig = nnSig();
  if (o.keep) NN.busy = true;
  try {
    var A = mlpBuf(S), masks = S.slice(1, -1).map(function(n){ return new Float32Array(n); }), gW = m.W.map(function(w){ return new Float32Array(w.length); }), gB = m.b.map(function(b){ return new Float32Array(b.length); });
    var mW = gW.map(function(w){ return new Float32Array(w.length); }), vW = gW.map(function(w){ return new Float32Array(w.length); }), mB = gB.map(function(b){ return new Float32Array(b.length); }), vB = gB.map(function(b){ return new Float32Array(b.length); });
    var delta = S.slice(1).map(function(n){ return new Float32Array(n); }), dprev = S.map(function(n){ return new Float32Array(n); }), order = P.T.n, perm = Array.from({ length: order }, function(_, i){ return i; }), BS = 64, tr = [], va = [], bestV = Infinity, bestM = null, bestEp = 0, step = 0, b1 = 0.9, b2 = 0.999, eps = 1e-8, F = P.F;
    for (var ep = 0; ep < p.epochs; ep++) {
      for (var a = perm.length - 1; a > 0; a--) { var jj = Math.floor(r() * (a + 1)), tt = perm[a]; perm[a] = perm[jj]; perm[jj] = tt; }
      for (var s0 = 0; s0 < order; s0 += BS) {
        var nb = Math.min(BS, order - s0); gW.forEach(function(g){ g.fill(0); }); gB.forEach(function(g){ g.fill(0); });
        for (var q = 0; q < nb; q++) {
          var idx = perm[s0 + q], pred = mlpForward(m, P.T.X, idx * F, A, masks, p.drop, r), d = 2 * (pred - P.T.y[idx]) / nb; delta[L - 1][0] = d;
          for (var l = L - 1; l >= 0; l--) { var nin = S[l], nout = S[l + 1], inp = A[l], dl = delta[l], gw = gW[l], gb = gB[l], W = m.W[l];
            for (var j = 0; j < nout; j++) gb[j] += dl[j];
            var dp = l > 0 ? dprev[l] : null; if (dp) dp.fill(0);
            for (var i = 0; i < nin; i++) { var ai = inp[i], base = i * nout; if (ai !== 0) for (var j2 = 0; j2 < nout; j2++) gw[base + j2] += ai * dl[j2]; if (dp && ai !== 0) { var sacc = 0; for (var j3 = 0; j3 < nout; j3++) sacc += W[base + j3] * dl[j3]; dp[i] = sacc * masks[l - 1][i]; } }
            if (l > 0) { var dst = delta[l - 1]; for (var i2 = 0; i2 < nin; i2++) dst[i2] = dp[i2]; } }
        }
        step++; var c1 = 1 - Math.pow(b1, step), c2 = 1 - Math.pow(b2, step), lr = p.lr;
        for (var l2 = 0; l2 < L; l2++) { var w2 = m.W[l2], g2 = gW[l2], mw = mW[l2], vw = vW[l2]; for (var k = 0; k < w2.length; k++) { var g = g2[k]; mw[k] = b1 * mw[k] + (1 - b1) * g; vw[k] = b2 * vw[k] + (1 - b2) * g * g; w2[k] -= lr * (mw[k] / c1) / (Math.sqrt(vw[k] / c2) + eps); }
          var bb = m.b[l2], gb2 = gB[l2], mb = mB[l2], vb = vB[l2]; for (var k2 = 0; k2 < bb.length; k2++) { var g3 = gb2[k2]; mb[k2] = b1 * mb[k2] + (1 - b1) * g3; vb[k2] = b2 * vb[k2] + (1 - b2) * g3 * g3; bb[k2] -= lr * (mb[k2] / c1) / (Math.sqrt(vb[k2] / c2) + eps); } }
      }
      var ltr = mlpMSE(m, P.T.X, P.T.y, P.T.n, A), lva = mlpMSE(m, P.V.X, P.V.y, P.V.n, A); tr.push(ltr); va.push(lva);
      if (lva < bestV) { bestV = lva; bestEp = ep + 1; bestM = mlpCopy(m); }
      var cb0 = performance.now(), sObj = { tr: tr, va: va, best: bestEp, model: m, prep: P, epochs: p.epochs, params: p };
      if (o.onEpoch) await o.onEpoch(ep, sObj); else if (ep % 2 === 1) await tick(); over += performance.now() - cb0;
    }
    var secs = (performance.now() - tN - over) / 1000, fin = p.early && bestM ? bestM : m, pred = mlpPredictAll(fin, P);
    var res = { model: fin, hist: { tr: tr, va: va }, best: bestEp, secs: secs, pred: pred, r2: r2(D.TEST, pred), params: p, prep: P };
    if (o.keep) { NN.m = fin; NN.prep = P; NN.hist = res.hist; NN.best = bestEp; NN.secs = secs; NN.pred = pred; NN.r2 = res.r2; NN.params = p; NN.feats = P.feats; NN.sig = sig; var ms = record('Neural network', P.feats, pred, secs); ms.params = p.layers + ' layers of ' + p.units + ', dropout ' + p.drop; NN.act = null; NN.sel = 0; nnResultsBox(); }
    return res;
  } finally { if (o.keep) NN.busy = false; }
}
async function nnReady(api){
  if (NN.m) { if (api) { api.idle(); api.stale(nnStale() ? 'Options changed since the last Train.' : ''); } return true; }
  if (api) api.gate('Train the network in the workbench first.'); return false;
}
function nnPredictRows(rows){ var m = NN.m, A = mlpBuf(m.sizes), F = m.sizes[0], flat = new Float32Array(rows.length * F); rows.forEach(function(rw, k){ for (var q = 0; q < F; q++) flat[k * F + q] = rw[q]; }); return rows.map(function(_, k){ return mlpForward(m, flat, k * F, A, null, 0, null) + NN.prep.ymu; }); }
function nnActivations(i){ var m = NN.m, P = NN.prep, A = mlpBuf(m.sizes); mlpForward(m, P.all, i * P.F, A, null, 0, null); return A; }

/* ---------- network drawing (real weights, activations, selectable neurons) ---------- */
function drawNetSVG(host, o){
  var sizes = o.model ? o.model.sizes : o.sizes, W = o.W, H = o.H, shown = sizes.map(function(s, i){ return Math.min(s, i === sizes.length - 1 ? 1 : (o.maxShow || 9)); }), padL = o.padL === undefined ? 86 : o.padL, padR = o.padR === undefined ? 60 : o.padR;
  var xs = shown.map(function(_, i){ return padL + i * (W - padL - padR) / (shown.length - 1); }), pos = shown.map(function(n){ return Array.from({ length: n }, function(_, k){ return 30 + (k + 0.5) * (H - 60) / n; }); }), svg = '', ST = o.act ? (o.stage === undefined ? 99 : o.stage) : -1;
  var getW; if (o.model) { getW = function(l, a, b){ return o.model.W[l][a * o.model.sizes[l + 1] + b]; }; } else { var rnd = rng(o.seed || 3), memo = {}; getW = function(l, a, b){ var k = l + '_' + a + '_' + b; if (!(k in memo)) memo[k] = rnd() + rnd() + rnd() - 1.5; return memo[k]; }; }
  for (var l = 0; l < sizes.length - 1; l++) { var mx = 1e-6; for (var a = 0; a < shown[l]; a++) for (var b = 0; b < shown[l + 1]; b++) mx = Math.max(mx, Math.abs(getW(l, a, b)));
    for (var a2 = 0; a2 < shown[l]; a2++) for (var b2 = 0; b2 < shown[l + 1]; b2++) { var w = getW(l, a2, b2), s = Math.abs(w) / mx, op = o.act ? (l < ST ? 0.2 + 0.5 * s : 0.05) : 0.18 + 0.6 * s, anim = o.training || (o.act && l === ST - 1 && ST <= sizes.length - 1) || o.flow;
      svg += '<line class="flowline' + (anim ? '' : ' off') + '" x1="' + xs[l].toFixed(1) + '" y1="' + pos[l][a2].toFixed(1) + '" x2="' + xs[l + 1].toFixed(1) + '" y2="' + pos[l + 1][b2].toFixed(1) + '" stroke="' + (w > 0 ? C.red : C.bg) + '" stroke-opacity="' + op.toFixed(2) + '" stroke-width="' + (0.4 + 2.6 * s).toFixed(2) + '"/>'; } }
  var rm = rng((o.seed || 3) + 17);
  for (var i = 0; i < shown.length; i++) { var out = i === shown.length - 1; for (var k = 0; k < shown[i]; k++) {
      var fill = out ? C.red : '#fff', fo = 1, val = null; if (o.act && i <= ST) val = i === 0 ? o.act.x[k] : o.act.layers[i - 1][k];
      if (val !== null && !out) { if (i === 0) { fill = val >= 0 ? C.red : C.char; fo = 0.15 + 0.85 * Math.min(1, Math.abs(val) / 2.5); } else if (val > 0) { fill = C.red; fo = 0.15 + 0.85 * Math.min(1, val / (o.act.max[i - 1] || 1)); } }
      if (out && o.act && i > ST) fill = '#fff';
      var sel = o.sel !== undefined && i === 1 && o.sel === k && o.act, dropped = o.drop > 0 && i > 0 && !out && rm() < o.drop;
      svg += '<circle ' + (o.onPick && i === 1 && !out ? 'data-pick="' + k + '" style="cursor:pointer" ' : '') + 'cx="' + xs[i].toFixed(1) + '" cy="' + pos[i][k].toFixed(1) + '" r="' + (out ? 14 : 9) + '" fill="' + fill + '" fill-opacity="' + fo + '" stroke="' + (sel ? C.red : (out ? C.dred : C.char)) + '" stroke-width="' + (sel ? 4 : 2) + '"' + (dropped ? ' stroke-dasharray="3 3" opacity=".3"' : '') + '/>';
      if (i === 0 && o.feats) svg += '<text x="' + (xs[0] - 15) + '" y="' + (pos[0][k] + 4).toFixed(1) + '" text-anchor="end" font-size="11.5" fill="' + C.char + '">' + esc(o.feats[k]) + (o.act ? ' ' + fmtFeat(o.feats[k], o.act.raw[k]) : '') + '</text>'; }
    if (!o.noLabels) svg += '<text x="' + xs[i].toFixed(1) + '" y="' + (H - 6) + '" text-anchor="middle" font-size="12" font-weight="700" fill="' + C.char + '">' + (i === 0 ? 'inputs' : (out ? 'prediction' : 'hidden ' + i)) + '</text>';
    if (i > 0 && !out && sizes[i] > shown[i]) svg += '<text x="' + xs[i].toFixed(1) + '" y="14" text-anchor="middle" font-size="10.5" fill="' + C.mid + '">' + sizes[i] + ' neurons, ' + shown[i] + ' drawn</text>'; else if (i > 0 && !out && !o.noLabels) svg += '<text x="' + xs[i].toFixed(1) + '" y="14" text-anchor="middle" font-size="10.5" fill="' + C.mid + '">' + sizes[i] + ' neurons</text>'; }
  if (o.act && ST >= sizes.length - 1) svg += '<text x="' + xs[xs.length - 1].toFixed(1) + '" y="' + (pos[pos.length - 1][0] - 24) + '" text-anchor="middle" font-size="13" font-weight="700" fill="' + C.red + '">' + o.act.pred.toFixed(2) + '</text>';
  host.innerHTML = svg;
  if (o.onPick) qsa('circle[data-pick]', host).forEach(function(c){ c.addEventListener('click', function(){ o.onPick(+c.dataset.pick); }); });
  return { xs: xs, pos: pos, shown: shown };
}
function lossChart(canvas, hist, best, upto, xmax){
  var n = upto === undefined ? hist.tr.length : upto, pts1 = [], pts2 = []; for (var i = 0; i < n; i++) { pts1.push([i + 1, hist.tr[i]]); pts2.push([i + 1, hist.va[i]]); }
  var g = lineChart(canvas, [{ pts: pts1, color: C.char }, { pts: pts2, color: C.red }], { xl: 'epoch', yl: 'loss', y0: 0, x0: 1, x1: Math.max(2, xmax || n), xdec: 0, ydec: 2 }); if (best && g.sx && n >= best) { var x = g.x, X = g.sx(best); x.strokeStyle = C.dred; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(X, g.box.t); x.lineTo(X, g.box.b); x.stroke(); x.setLineDash([]); x.fillStyle = C.dred; x.font = '600 11px Aptos, Segoe UI, sans-serif'; x.textAlign = X > g.w / 2 ? 'right' : 'left'; x.fillText('best epoch ' + best, X + (X > g.w / 2 ? -6 : 6), g.box.t + 12); } return g;
}

/* ---------- workbench: set up a network, press Train ---------- */
L3.add({ short: 'Workbench', bench: true, uses: ['endpoint', 'features', 'filters', 'nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs', 'nearly'],
  title: 'Set up a network, press Train, and watch it learn',
  watch: 'The diagram follows your options as you drag them. Train starts from random weights and replays the training slowly: the weights change on the left while the error curve and the predictions on the right update every epoch.',
  change: 'Drag the options, then press Train. Slow shows every epoch, Fast skips ahead.',
  build: function(api){
    var bb = benchBar(api);
    var np = api.el('pn', 'left:28px;top:80px;width:456px;height:372px', '<h4>The network</h4><svg viewBox="0 0 432 318" width="432" height="318"></svg>');
    var lp = api.el('pn', 'left:504px;top:80px;width:428px;height:178px', '<h4>Error per epoch (dark: training, red: validation)</h4><canvas width="404" height="140"></canvas>');
    var sp = api.el('pn', 'left:504px;top:270px;width:428px;height:182px', '<h4>Predictions on unseen compounds</h4><canvas width="404" height="144"></canvas>');
    var mn = api.el('pn', 'left:28px;top:462px;width:904px;height:56px;display:flex;align-items:center;font-size:14px;line-height:1.4', '<div class="mn"></div>');
    var svg = qs('svg', np), running = false;
    function fresh(){ return !!(NN.m && !nnStale()); }
    function msg(t){ svg.innerHTML = '<text x="20" y="40" font-size="13" fill="' + C.red + '">' + t + '</text>'; }
    function emptyCharts(p){ lineChart(qs('canvas', lp), [{ pts: [[1, 0], [Math.max(2, p.epochs), 1]], color: 'rgba(0,0,0,0)', nodot: true }], { xl: 'epoch', yl: 'error', x0: 1, x1: Math.max(2, p.epochs), y0: 0, xdec: 0, ydec: 2 }); scatterPV(qs('canvas', sp), new Float32Array(D.N), [], null); }
    function picture(){
      var f = activeFeats(), p = NP(); if (!f.length) { msg('Switch on at least one feature.'); return; }
      if (fresh()) drawNetSVG(svg, { W: 432, H: 318, model: NN.m, feats: NN.feats, maxShow: 8, padL: 78, padR: 36 });
      else drawNetSVG(svg, { W: 432, H: 318, sizes: netSizes(f.length, p.layers, p.units), feats: f, drop: p.drop, seed: 3, maxShow: 8, padL: 78, padR: 36 });
    }
    function note(){
      var f = activeFeats(), p = NP(), sizes = netSizes(Math.max(1, f.length), p.layers, p.units), tot = 0; for (var q = 0; q < sizes.length - 1; q++) tot += sizes[q] * sizes[q + 1] + sizes[q + 1];
      qs('.mn', mn).textContent = p.layers + ' hidden layer' + (p.layers > 1 ? 's' : '') + ' of ' + p.units + ' neurons, ' + tot.toLocaleString() + ' numbers to learn. Each epoch is one pass over the training compounds, nudging every weight by about ' + p.lr + ' per step. Dropout ' + p.drop + ' switches off that share of neurons at each step. ' + (p.early ? 'The best epoch on held-out training compounds is kept.' : 'The last epoch is kept.');
    }
    function render(){
      var p = NP(); picture(); note();
      if (fresh()) { lossChart(qs('canvas', lp), NN.hist, NN.best, undefined, NN.hist.tr.length); scatterPV(qs('canvas', sp), NN.pred, D.TEST, 'unseen R2 ' + fix(NN.r2, 2), { max: 320 }); api.stale(''); bb.say('Unseen R2 ' + fix(NN.r2, 2) + '. Best epoch ' + NN.best + '. Training took ' + fmtT(NN.secs) + '.'); }
      else { emptyCharts(p); api.stale(NN.m ? 'Options changed. Press Train.' : ''); bb.say(NN.m ? '' : 'Ready. Press Train.'); }
    }
    async function train(){
      if (running || NN.busy) return; if (!activeFeats().length) { bb.say('Switch on at least one feature.'); return; }
      running = true; bb.btn.disabled = true; api.tok++; var tok = api.tok, p = NP(), per = 450 * speedMul(); api.stale('');
      try {
        api.busy('Preparing data'); await delay(30); nnPrep(); api.idle(); emptyCharts(p);
        await nnTrain({ keep: true, onEpoch: async function(ep, s){
          api.checkTok(tok); var pred = mlpPredictAll(s.model, s.prep);
          drawNetSVG(svg, { W: 432, H: 318, model: s.model, feats: s.prep.feats, training: true, maxShow: 8, padL: 78, padR: 36 });
          lossChart(qs('canvas', lp), { tr: s.tr, va: s.va }, 0, undefined, p.epochs); scatterPV(qs('canvas', sp), pred, D.TEST, 'epoch ' + (ep + 1) + ', R2 ' + fix(r2(D.TEST, pred), 2), { max: 320 });
          bb.say('Epoch ' + (ep + 1) + ' of ' + p.epochs + '. Validation error ' + fix(s.va[ep], 3)); progress((ep + 1) / p.epochs);
          if (per > 1) await api.sleep(per); else if (ep % 2) await tick();
        } });
      } catch (e) { if (!(e instanceof Cancel)) console.error(e); }
      api.idle(); running = false; bb.btn.disabled = false; progress(1); nnResultsBox(); L3.refresh();
    }
    bb.btn.addEventListener('click', train);
    L3.live = function(){ if (running) api.tok++; else render(); };
    return { render: render };
  } });

/* ---------- N1: a neuron is a weighted sum ---------- */
L3.add({ short: 'One neuron', title: 'A neuron is a weighted sum followed by a switch',
  change: 'Drag the weight and bias sliders inside the picture. Untick features in the options to change the inputs. Next compound picks another molecule.',
  watch: 'Each input is multiplied by a weight, the products and a bias are added, and ReLU turns a negative total into zero. Red lines add to the total, grey lines subtract.',
  build: function(api){
    var cur = 0, feats = [], x = [0, 0, 0], w = [0.9, -0.6, 0.4], bias = 0.2, built = false;
    var svg = api.el('abs', 'left:0;top:0;width:960px;height:540px;pointer-events:none', '<svg viewBox="0 0 960 540" width="960" height="540"></svg>');
    var rows = [0, 1, 2].map(function(k){ var y = 56 + k * 112; return api.el('pn fd', 'left:28px;top:' + y + 'px;width:210px;height:92px', '<h4 class="fn">feature</h4><div class="xv big" style="font-size:24px">0</div><div class="xr" style="font-size:11.5px;color:#6B7280"></div>'); });
    var sl = [0, 1, 2].map(function(k){ var y = 56 + k * 112; return api.el('pn fd', 'left:300px;top:' + y + 'px;width:176px;height:92px', '<h4>Weight</h4><input type="range" min="-2" max="2" step="0.1" value="0" style="width:100%"><div class="wv big" style="font-size:22px;color:' + C.red + '">0</div>'); });
    var bp = api.el('pn fd', 'left:300px;top:394px;width:176px;height:50px', '<h4 style="margin-bottom:2px">Bias <span class="bv" style="float:right;color:' + C.red + '">0.2</span></h4><input type="range" min="-2" max="2" step="0.1" value="0.2" style="width:100%">');
    var sum = api.el('pn fd', 'left:536px;top:176px;width:110px;height:110px;border-radius:55px;text-align:center;padding-top:26px', '<div class="lbl">total</div><div class="zv big" style="font-size:26px">0</div>');
    var rp = api.el('pn fd', 'left:690px;top:56px;width:242px;height:250px', '<h4>ReLU switch</h4><canvas width="218" height="190"></canvas>');
    var op = api.el('pn fd', 'left:690px;top:320px;width:242px;height:124px', '<h4>Output of the neuron</h4><div class="av big" style="color:' + C.red + '">0</div><div class="am note" style="font-size:12px;color:#6B7280"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    function stdOf(ft, i){ var v = D.TRAIN.map(function(k){ return D.COL[ft][k]; }), mu = mean(v), sd = Math.sqrt(mean(v.map(function(q){ return (q - mu) * (q - mu); }))) || 1; return clamp((D.COL[ft][i] - mu) / sd, -3, 3); }
    function pickCmp(keep){ var f = activeFeats().slice(0, 3); while (f.length < 3) { var extra = D.BASE.filter(function(b){ return f.indexOf(b) < 0; })[0]; f.push(extra); } feats = f; if (!keep || !cur) cur = D.TEST[Math.floor(Math.random() * D.TEST.length)]; x = f.map(function(ft){ return stdOf(ft, cur); }); }
    function calc(){ var p = x.map(function(v, k){ return v * w[k]; }), z = p[0] + p[1] + p[2] + bias; return { p: p, z: z, a: Math.max(0, z) }; }
    function draw(){
      var c = calc(), s = '', ys = [102, 214, 326], cx = 591, cy = 231, mx = Math.max(1, Math.abs(c.p[0]), Math.abs(c.p[1]), Math.abs(c.p[2]));
      c.p.forEach(function(pv, k){ s += '<line x1="476" y1="' + ys[k] + '" x2="' + (cx - 54) + '" y2="' + (cy + (k - 1) * 18) + '" stroke="' + (pv >= 0 ? C.red : C.bg) + '" stroke-width="' + (1 + 7 * Math.abs(pv) / mx).toFixed(1) + '" stroke-opacity=".85" stroke-linecap="round"/>'; s += '<line x1="238" y1="' + ys[k] + '" x2="300" y2="' + ys[k] + '" stroke="' + C.bg + '" stroke-width="2"/>'; s += '<text x="482" y="' + (ys[k] - 6) + '" font-size="12" font-weight="700" fill="' + (pv >= 0 ? C.red : C.mid) + '">' + (pv >= 0 ? '+' : '') + pv.toFixed(2) + '</text>'; });
      s += '<line x1="476" y1="418" x2="' + (cx - 40) + '" y2="' + (cy + 38) + '" stroke="' + (bias >= 0 ? C.red : C.bg) + '" stroke-width="2" stroke-opacity=".8"/>'; s += '<line x1="' + (cx + 55) + '" y1="' + cy + '" x2="690" y2="' + cy + '" stroke="' + C.red + '" stroke-width="2.5"/>'; qs('svg', svg).innerHTML = s;
      qs('.zv', sum).textContent = c.z.toFixed(2); qs('.av', op).textContent = c.a.toFixed(2); qs('.am', op).textContent = c.z > 0 ? 'The total is positive, so the neuron passes it on.' : 'The total is negative, so the neuron outputs zero.';
      rows.forEach(function(r, k){ qs('.fn', r).textContent = feats[k]; qs('.xv', r).textContent = x[k].toFixed(2); qs('.xr', r).textContent = 'raw value ' + fmtFeat(feats[k], D.COL[feats[k]][cur]) + ', rescaled'; }); sl.forEach(function(sx, k){ qs('.wv', sx).textContent = (w[k] >= 0 ? '+' : '') + w[k].toFixed(1); qs('input', sx).value = w[k]; }); qs('.bv', bp).textContent = bias.toFixed(1); qs('input', bp).value = bias;
      var g = cv(qs('canvas', rp)), cx2 = g.x, b = { l: 30, r: g.w - 10, t: 10, b: g.h - 28 }, lo = -3, hi = 3, sxp = function(v){ return b.l + (v - lo) / (hi - lo) * (b.r - b.l); }, syp = function(v){ return b.b - (clamp(v, 0, 3) / 3) * (b.b - b.t); };
      cx2.strokeStyle = C.silver; cx2.beginPath(); cx2.moveTo(b.l, b.b); cx2.lineTo(b.r, b.b); cx2.moveTo(sxp(0), b.t); cx2.lineTo(sxp(0), b.b); cx2.stroke(); cx2.strokeStyle = C.char; cx2.lineWidth = 3; cx2.beginPath(); cx2.moveTo(sxp(lo), syp(0)); cx2.lineTo(sxp(0), syp(0)); cx2.lineTo(sxp(hi), syp(hi)); cx2.stroke();
      cx2.fillStyle = C.red; cx2.beginPath(); cx2.arc(sxp(clamp(c.z, lo, hi)), syp(c.a), 6, 0, 6.3); cx2.fill(); cx2.fillStyle = C.mid; cx2.font = '11px Aptos, Segoe UI, sans-serif'; cx2.textAlign = 'center'; cx2.fillText('total', (b.l + b.r) / 2, g.h - 8); cx2.fillText('0', sxp(0), b.b + 14); }
    sl.forEach(function(sx, k){ qs('input', sx).addEventListener('input', function(){ api.tok++; w[k] = +this.value; draw(); }); }); qs('input', bp).addEventListener('input', function(){ api.tok++; bias = +this.value; draw(); });
    qs('[data-next]', api.sbar).addEventListener('click', function(){ api.tok++; pickCmp(false); draw(); });
    function show(list, on){ list.forEach(function(e){ e.classList.toggle('on', on); }); }
    function render(){ pickCmp(built); built = true; show(rows, true); show(sl, true); show([bp, sum, rp, op], true); draw(); }
    return { render: render };
  } });

/* ---------- N2: layers and units ---------- */
L3.add({ short: 'Layers and neurons', title: 'Layers and neurons set how many numbers the network must learn',
  change: 'Drag Hidden layers, Neurons per layer and Dropout in the options.',
  watch: 'The diagram follows your settings. The table on the right counts every weight and bias. Dashed circles are neurons switched off by dropout.',
  build: function(api){
    var dp = api.el('pn', 'left:28px;top:28px;width:640px;height:408px', '<svg viewBox="0 0 620 380" width="620" height="380"></svg>');
    var cp = api.el('pn', 'left:688px;top:28px;width:244px;height:408px', '<h4>Parameters to learn</h4><div class="big pc" style="color:' + C.red + '">0</div><div class="note" style="font-size:12px;color:#6B7280;margin-bottom:8px">weights and biases</div><div class="pl" style="font-size:12px"></div>');
    var seed = 3;
    function draw(p){ p = p || NP(); var f = activeFeats(); if (!f.length) return; var sizes = netSizes(f.length, p.layers, p.units); drawNetSVG(qs('svg', dp), { W: 620, H: 380, sizes: sizes, feats: f, drop: p.drop, seed: seed, flow: true, maxShow: 8 });
      var rows = '', tot = 0; for (var q = 0; q < sizes.length - 1; q++) { var wts = sizes[q] * sizes[q + 1], bs = sizes[q + 1]; tot += wts + bs; rows += '<div style="display:flex;justify-content:space-between;border-bottom:1px solid #D1D8E0;padding:4px 0"><span>' + (q === 0 ? 'inputs' : 'hidden ' + q) + ' to ' + (q === sizes.length - 2 ? 'output' : 'hidden ' + (q + 1)) + '</span><b>' + (wts + bs).toLocaleString() + '</b></div><div style="font-size:11px;color:#6B7280;margin-bottom:4px">' + sizes[q] + ' by ' + sizes[q + 1] + ' weights plus ' + bs + ' biases</div>'; }
      qs('.pl', cp).innerHTML = rows; qs('.pc', cp).textContent = tot.toLocaleString(); }
    function render(){ draw(); }
    return { render: render };
  } });

/* ---------- N3: learning rate ---------- */
L3.add({ short: 'Learning rate', title: 'The learning rate sets the size of each step downhill',
  change: 'Drag Learning rate and Epochs in the options, or drag Steps taken under the stage to watch the paths grow.',
  watch: 'Colour shows the error of a one-neuron model for each pair of weights. Training walks downhill. The red path uses your learning rate.',
  build: function(api){
    var mp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Error for every pair of weights, one neuron on one feature</h4><canvas width="540" height="350"></canvas>');
    var sp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px', '<h4>Three step sizes</h4><div class="lr"></div>');
    var tr = null, DATAX = null, DATAY = null, paths = [], heat = null, NSTEP = 45;
    var gx = [-2.2, 2.2], gy = [-2.2, 2.2];
    function prep(){ var f = activeFeats()[0] || D.BASE[0], idx = D.TRAIN.filter(function(_, k){ return k % 3 === 0; }), xv = idx.map(function(i){ return D.COL[f][i]; }), yv = idx.map(function(i){ return D.Y[i]; }), mx = mean(xv), my = mean(yv), sx = Math.sqrt(mean(xv.map(function(v){ return (v - mx) * (v - mx); }))) || 1, sy = Math.sqrt(mean(yv.map(function(v){ return (v - my) * (v - my); }))) || 1; DATAX = xv.map(function(v){ return (v - mx) / sx; }); DATAY = yv.map(function(v){ return (v - my) / sy; }); tr = { f: f }; heat = null; }
    function loss(w, b){ var s = 0, n = DATAX.length; for (var i = 0; i < n; i++) { var e = DATAY[i] - (w * DATAX[i] + b); s += e * e; } return s / n; }
    function grad(w, b){ var gw = 0, gb = 0, n = DATAX.length; for (var i = 0; i < n; i++) { var e = DATAY[i] - (w * DATAX[i] + b); gw += -2 * e * DATAX[i]; gb += -2 * e; } return [gw / n, gb / n]; }
    function lrs(){ var mine = +$('nlr').value * 50; return [{ n: 'Small', lr: 0.02, c: C.bg }, { n: 'Yours', lr: mine, c: C.red }, { n: 'Too large', lr: 1.04, c: C.char }]; }
    function runPaths(){ NSTEP = clamp(+$('nepochs').value, 10, 120); paths = lrs().map(function(s){ var w = -1.9, b = 1.6, pts = [[w, b, loss(w, b)]]; for (var k = 0; k < NSTEP; k++) { var g = grad(w, b); w -= s.lr * g[0]; b -= s.lr * g[1]; pts.push([w, b, loss(w, b)]); } return { s: s, pts: pts }; }); }
    function drawHeat(){ var g = cv(qs('canvas', mp)), x = g.x, W = g.w, H = g.h, cell = 5;
      if (!heat || heat.key !== tr.f) { var cols = Math.ceil(W / cell), rws = Math.ceil(H / cell), vals = [], lmin = 1e9, opt = [0, 0]; for (var r = 0; r < rws; r++) for (var c = 0; c < cols; c++) { var w = gx[0] + (c + 0.5) / cols * (gx[1] - gx[0]), b = gy[1] - (r + 0.5) / rws * (gy[1] - gy[0]), l = loss(w, b); vals.push(l); if (l < lmin) { lmin = l; opt = [w, b]; } } heat = { key: tr.f, vals: vals, cols: cols, rws: rws, lmin: lmin, opt: opt }; }
      for (var r2 = 0; r2 < heat.rws; r2++) for (var c2 = 0; c2 < heat.cols; c2++) { var v = clamp((heat.vals[r2 * heat.cols + c2] - heat.lmin) / 7, 0, 1), band = Math.floor(v * 10) / 10, t = Math.pow(band, 0.9); x.fillStyle = 'rgb(' + Math.round(lerp(255, 225, t)) + ',' + Math.round(lerp(255, 36, t)) + ',' + Math.round(lerp(255, 42, t)) + ')'; x.fillRect(c2 * cell, r2 * cell, cell, cell); }
      x.strokeStyle = 'rgba(52,55,63,.25)'; x.lineWidth = 1; x.beginPath(); x.moveTo(W / 2, 0); x.lineTo(W / 2, H); x.moveTo(0, H / 2); x.lineTo(W, H / 2); x.stroke(); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText('weight on ' + tr.f, W / 2, H - 6); x.save(); x.translate(12, H / 2); x.rotate(-Math.PI / 2); x.fillText('bias', 0, 0); x.restore(); x.textAlign = 'left'; x.fillText('white = lowest error, red = high error. Each band is a step up in error.', 8, 14); return g; }
    function toPx(g, w, b){ return [(w - gx[0]) / (gx[1] - gx[0]) * g.w, (gy[1] - b) / (gy[1] - gy[0]) * g.h]; }
    function setup(){ if (!tr || tr.f !== (activeFeats()[0] || D.BASE[0])) prep(); runPaths(); }
    function draw(upto){ setup(); var g = drawHeat(), x = g.x, opt = toPx(g, heat.opt[0], heat.opt[1]); x.fillStyle = '#fff'; x.strokeStyle = C.char; x.lineWidth = 2.4; x.beginPath(); x.arc(opt[0], opt[1], 7, 0, 6.3); x.fill(); x.stroke(); x.fillStyle = C.char; x.font = '700 12px Aptos, Segoe UI, sans-serif'; x.fillText('best weights', opt[0] + 11, opt[1] + 4); var st0 = toPx(g, -1.9, 1.6); x.fillStyle = C.char; x.beginPath(); x.arc(st0[0], st0[1], 5, 0, 6.3); x.fill(); x.font = '600 11px Aptos, Segoe UI, sans-serif'; x.fillText('start', st0[0] + 8, st0[1] - 6);
      paths.forEach(function(p){ var n = Math.min(upto, p.pts.length - 1); x.strokeStyle = p.s.c; x.lineWidth = 2.6; x.lineJoin = 'round'; x.beginPath(); for (var k = 0; k <= n; k++) { var q = toPx(g, clamp(p.pts[k][0], -4, 4), clamp(p.pts[k][1], -4, 4)); k ? x.lineTo(q[0], q[1]) : x.moveTo(q[0], q[1]); } x.stroke(); var e = toPx(g, clamp(p.pts[n][0], -4, 4), clamp(p.pts[n][1], -4, 4)); x.fillStyle = p.s.c; x.strokeStyle = '#fff'; x.lineWidth = 2; x.beginPath(); x.arc(e[0], e[1], 6, 0, 6.3); x.fill(); x.stroke(); });
      qs('.lr', sp).innerHTML = paths.map(function(p){ var n = Math.min(upto, p.pts.length - 1), L = p.pts[n][2]; return '<div style="margin:12px 0;border-left:4px solid ' + p.s.c + ';padding-left:10px"><b>' + p.s.n + '</b> <span class="note" style="font-size:12px;color:#6B7280">step size ' + p.s.lr.toFixed(2) + (p.s.n === 'Yours' ? ' (learning rate ' + $('nlr').value + ')' : '') + '</span><div style="font-size:12.5px">after ' + n + ' of ' + NSTEP + ' steps, error <b>' + (L > 99 ? 'exploding' : L.toFixed(2)) + '</b></div></div>'; }).join('') + '<p style="font-size:12px;color:#6B7280;margin-top:10px">A simplified picture: one neuron with two weights and plain gradient descent. Real networks have thousands of weights and use Adam, but the step-size trade-off is the same.</p>'; }
    var sc = api.addControl('Steps taken <input type="range" min="0" max="120" step="1" value="120" style="width:240px"><b class="sv"></b>'), ss = qs('input', sc), atEnd = true;
    ss.addEventListener('input', function(){ atEnd = +ss.value >= NSTEP; draw(+ss.value); qs('.sv', sc).textContent = ss.value; });
    function render(){ setup(); ss.max = NSTEP; if (atEnd) ss.value = NSTEP; draw(+ss.value); qs('.sv', sc).textContent = ss.value; }
    return { render: render };
  } });

/* ---------- N5: dropout and early stopping ---------- */
L3.add({ short: 'Dropout and early stopping', title: 'Dropout and the best epoch guard against overfitting',
  change: 'Set Dropout and the other options, then press Run comparison under the stage.',
  watch: 'Two networks train side by side, one without dropout and one with yours. The curves are validation error. Dashed lines mark each network\'s best epoch.',
  auto: false,
  build: function(api){
    var np = api.el('pn', 'left:28px;top:28px;width:360px;height:408px', '<h4>Neurons switched off in each training step</h4><svg viewBox="0 0 340 330" width="340" height="330"></svg><div class="dn note" style="font-size:12px;color:#6B7280"></div>');
    var cp = api.el('pn', 'left:408px;top:28px;width:524px;height:290px', '<h4>Validation error per epoch</h4><canvas width="502" height="240"></canvas>');
    var rp = api.el('pn', 'left:408px;top:330px;width:524px;height:106px', '<div class="cr" style="font-size:13px;color:#4a4f5a">Preparing the comparison.</div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-run>Run comparison</button> <button type="button" class="btn sm ghost" data-mask>New dropout pattern</button>'); var cmp = null, running = false, seedN = 5;
    function sig(){ var p = NP(); return JSON.stringify([p.layers, p.units, p.drop, p.lr, Math.min(p.epochs, 60), activeFeats(), EP]); }
    function maskDraw(seed){ var p = NP(), f = activeFeats(); if (!f.length) return; drawNetSVG(qs('svg', np), { W: 340, H: 330, sizes: netSizes(f.length, p.layers, p.units), feats: f, drop: p.drop, seed: seed, flow: true, maxShow: 8, padL: 70, padR: 30, noLabels: true }); qs('.dn', np).textContent = 'Dropout ' + p.drop + ': about ' + Math.round(p.drop * 100) + '% of hidden neurons are switched off at each step. Dashed circles are off.'; }
    function drawCmp(){ if (!cmp) { cv(qs('canvas', cp)); return; } var a = cmp.a, b = cmp.b, n = Math.max(a.hist.va.length, b.hist.va.length), g = lineChart(qs('canvas', cp), [{ pts: a.hist.va.map(function(v, i){ return [i + 1, v]; }), color: C.char, nodot: true, label: 'no dropout' }, { pts: b.hist.va.map(function(v, i){ return [i + 1, v]; }), color: C.red, nodot: true, label: 'dropout ' + cmp.p.drop }], { xl: 'epoch', yl: 'validation error', x0: 1, x1: n, y0: 0, xdec: 0, ydec: 2 }), x = g.x;
      [[a, C.char], [b, C.red]].forEach(function(r){ var X = g.sx(r[0].best); x.strokeStyle = r[1]; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(X, g.box.t); x.lineTo(X, g.box.b); x.stroke(); x.setLineDash([]); }); }
    function summary(){ if (!cmp) { qs('.cr', rp).innerHTML = '<span class="note">Press Run comparison. It trains two networks for up to 60 epochs.</span>'; return; } var a = cmp.a, b = cmp.b; qs('.cr', rp).innerHTML = '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px"><div><div class="lbl">No dropout</div><div style="font-size:13px">best epoch <b>' + a.best + '</b>, unseen R2 <b>' + a.r2.toFixed(2) + '</b></div></div><div><div class="lbl" style="color:' + C.red + '">Dropout ' + cmp.p.drop + '</div><div style="font-size:13px">best epoch <b>' + b.best + '</b>, unseen R2 <b>' + b.r2.toFixed(2) + '</b></div></div></div><p style="font-size:12px;color:#6B7280;margin-top:6px">Both networks restore their best epoch. Without that, the last epoch would score lower once validation error has turned upward.</p>'; }
    async function run(){ if (running) return; running = true; var btn = qs('[data-run]', ctl); btn.disabled = true; var p = NP(), st = qs('.cr', rp); if (!activeFeats().length) { st.textContent = 'Tick at least one feature.'; btn.disabled = false; running = false; return; }
      var want = sig(); p.epochs = Math.min(p.epochs, 60); api.busy('Training a network without dropout'); progress(0.2); await tick(); var a = await nnTrain({ params: Object.assign({}, p, { drop: 0 }) }); api.busy('Training a network with dropout ' + p.drop); progress(0.6); await tick(); var b = await nnTrain({ params: p }); cmp = { a: a, b: b, p: p, sig: want }; api.idle(); progress(1); drawCmp(); summary(); btn.disabled = false; running = false; render(); }
    qs('[data-run]', ctl).addEventListener('click', function(){ api.tok++; cmp = null; run(); }); qs('[data-mask]', ctl).addEventListener('click', function(){ seedN++; maskDraw(seedN); });
    function render(){ maskDraw(seedN); drawCmp(); summary(); api.stale(cmp && cmp.sig !== sig() ? 'Options changed. Run the comparison again.' : ''); }
    return { render: render };
  } });

L3.add(filterStepDef('nn'));
