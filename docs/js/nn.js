"use strict";
/* Lesson 4: Neural networks (engine and steps 1 to 5) */
var L3 = new Lesson('nn', 'Neural networks');
L3.optsHTML = function(){
  return sharedOptions() +
    optRange('nlayers', 'Hidden layers', 1, 4, 1, 2, 'Stages between inputs and output.') +
    optRange('nunits', 'Neurons per layer', 4, 128, 4, 32, 'Width of each hidden layer.') +
    optRange('ndrop', 'Dropout', 0, 0.6, 0.05, 0.1, 'Share of neurons switched off each step.') +
    optRange('nlr', 'Learning rate', 0.0005, 0.02, 0.0005, 0.003, 'Size of each weight update.') +
    optRange('nepochs', 'Epochs', 10, 120, 10, 50, 'Passes through the training set.') +
    '<div class="og"><label style="display:flex;gap:8px;align-items:center;font-size:.86rem;font-weight:600"><input type="checkbox" id="nearly" checked style="accent-color:#E1242A"> Keep the best epoch</label><div class="hint">Restore the epoch with the lowest validation loss.</div></div>' +
    '<div class="og"><button type="button" class="btn wide" id="ntrain">Train network</button><div class="status" id="nst"></div><div class="rbox" id="nres"></div></div>';
};
function NP(){ return { layers: +$('nlayers').value, units: +$('nunits').value, drop: +$('ndrop').value, lr: +$('nlr').value, epochs: +$('nepochs').value, early: $('nearly').checked }; }
var NN = { m: null, params: null, feats: null, prep: null, prepKey: null, hist: null, pred: null, best: 0, secs: null, sel: 0, act: null, r2: null, busy: false };
var nTimer = null;
function nnStale(){ if (!NN.m) return false; var p = NP(); return JSON.stringify([p.layers, p.units, p.drop, p.lr, p.epochs, p.early, activeFeats(), EP]) !== NN.sig; }
function nnResultsBox(){ var el = $('nres'); if (!el) return; if (!NN.m) { el.innerHTML = ''; return; } el.innerHTML = '<div class="rrow">Unseen R2<b>' + NN.r2.toFixed(2) + '</b></div><div class="rrow">Best epoch<b>' + NN.best + '</b></div><div class="rrow">Training time<b>' + fmtT(NN.secs) + '</b></div>' + (nnStale() ? '<div class="hint" style="color:#E1242A">Settings changed since this network was trained.</div>' : ''); }
L3.onOpts = function(root){
  var fire = function(){ clearTimeout(nTimer); nTimer = setTimeout(function(){ nnResultsBox(); L3.refresh(); }, 200); };
  ['nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs'].forEach(function(id){ $(id).addEventListener('input', fire); }); $('nearly').addEventListener('change', fire);
  $('ntrain').addEventListener('click', function(){ var s = L3.steps[3]; s.sec.scrollIntoView({ behavior: ANIM ? 'smooth' : 'auto', block: 'start' }); setTimeout(function(){ s.play(); }, ANIM ? 500 : 0); });
};
L3.onData = function(){ NN.m = null; NN.prepKey = null; nnResultsBox(); };
featListeners.push(function(){ if (L3.built) { nnResultsBox(); L3.refresh(); } });

/* ---------- engine ---------- */
function nnPrep(){
  var feats = activeFeats(), idx = activeTrain(), key = JSON.stringify([EP, feats, idx.length, filterOn() ? FILT : 0]); if (NN.prepKey === key && NN.prep) return NN.prep;
  if (NN.prep) NN.prep.tens.forEach(function(t){ t.dispose(); });
  var mu = {}, sd = {}; feats.forEach(function(f){ var v = idx.map(function(i){ return D.COL[f][i]; }); mu[f] = mean(v); sd[f] = Math.sqrt(mean(v.map(function(x){ return (x - mu[f]) * (x - mu[f]); }))) || 1; });
  var ymu = mean(idx.map(function(i){ return D.Y[i]; })), rowOf = function(i){ return feats.map(function(f){ return clamp((D.COL[f][i] - mu[f]) / sd[f], -5, 5); }); };
  var sh = idx.slice(), r = rng(11); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
  var nv = Math.round(sh.length * 0.15), va = sh.slice(0, nv), tn = sh.slice(nv), mk = function(ix){ return [tf.tensor2d(ix.map(rowOf)), tf.tensor2d(ix.map(function(i){ return [D.Y[i] - ymu]; }))]; };
  var tT = mk(tn), vT = mk(va), allX = tf.tensor2d(Array.from({ length: D.N }, function(_, i){ return rowOf(i); }));
  NN.prep = { feats: feats, mu: mu, sd: sd, ymu: ymu, rowOf: rowOf, tn: tn, va: va, tT: tT, vT: vT, allX: allX, tens: [tT[0], tT[1], vT[0], vT[1], allX] }; NN.prepKey = key; return NN.prep;
}
function buildModel(layers, units, drop, lr, nin){
  var m = tf.sequential(); for (var l = 0; l < layers; l++) { m.add(tf.layers.dense({ units: units, activation: 'relu', inputShape: l === 0 ? [nin] : undefined })); if (drop > 0) m.add(tf.layers.dropout({ rate: drop })); }
  m.add(tf.layers.dense({ units: 1 })); m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' }); return m;
}
async function nnTrain(o){
  var P = nnPrep(), p = Object.assign(NP(), o.params || {}), m = buildModel(p.layers, p.units, p.drop, p.lr, P.feats.length), tr = [], va = [], bestV = Infinity, bestW = null, bestEp = 0, over = 0, tN = performance.now();
  await m.fit(P.tT[0], P.tT[1], { epochs: p.epochs, batchSize: 64, shuffle: true, yieldEvery: 'never', validationData: [P.vT[0], P.vT[1]], callbacks: { onEpochEnd: async function(ep, logs){
    var cb0 = performance.now(); tr.push(logs.loss); va.push(logs.val_loss);
    if (logs.val_loss < bestV) { bestV = logs.val_loss; bestEp = ep + 1; if (bestW) bestW.forEach(function(w){ w.dispose(); }); bestW = m.getWeights().map(function(w){ return w.clone(); }); }
    if (o.onEpoch) await o.onEpoch(ep, { tr: tr, va: va, best: bestEp, model: m, prep: P }); over += performance.now() - cb0; } } });
  var secs = (performance.now() - tN - over) / 1000; if (p.early && bestW) m.setWeights(bestW);
  var out = tf.tidy(function(){ return m.predict(P.allX).dataSync(); }), pred = new Float64Array(D.N); for (var i = 0; i < D.N; i++) pred[i] = out[i] + P.ymu;
  var res = { model: m, hist: { tr: tr, va: va }, best: bestEp, secs: secs, pred: pred, r2: r2(D.TEST, pred), params: p, prep: P, bestW: bestW };
  if (o.keep) { if (NN.m && NN.m !== m) NN.m.dispose(); NN.m = m; NN.hist = res.hist; NN.best = bestEp; NN.secs = secs; NN.pred = pred; NN.r2 = res.r2; NN.params = p; NN.feats = P.feats; NN.sig = JSON.stringify([p.layers, p.units, p.drop, p.lr, p.epochs, p.early, P.feats, EP]); var ms = record('Neural network', P.feats, pred, secs); ms.params = p.layers + ' layers of ' + p.units + ', dropout ' + p.drop; NN.act = null; nnResultsBox(); }
  else { if (bestW) bestW.forEach(function(w){ w.dispose(); }); m.dispose(); }
  return res;
}
async function ensureNet(){ if (NN.m && !nnStale()) return NN; var st = $('nst'); if (st) st.textContent = 'Training a network'; progress(0.3); await tick(); await nnTrain({ keep: true }); progress(1); if (st) st.textContent = 'Done'; return NN; }
function denseLayers(m){ return m.layers.filter(function(l){ return l.getClassName() === 'Dense'; }); }
function nnPredictRows(rows){ return tf.tidy(function(){ return Array.from(NN.m.predict(tf.tensor2d(rows)).dataSync()); }).map(function(v){ return v + NN.prep.ymu; }); }

/* ---------- network drawing (real weights, activations, selectable neurons) ---------- */
function netSizes(nin, layers, units){ var s = [nin]; for (var i = 0; i < layers; i++) s.push(units); s.push(1); return s; }
function drawNetSVG(host, o){
  var W = o.W, H = o.H, sizes = o.sizes, shown = sizes.map(function(s, i){ return Math.min(s, i === sizes.length - 1 ? 1 : (o.maxShow || 9)); }), padL = o.padL === undefined ? 86 : o.padL, padR = o.padR === undefined ? 60 : o.padR;
  var xs = shown.map(function(_, i){ return padL + i * (W - padL - padR) / (shown.length - 1); }), pos = shown.map(function(n){ return Array.from({ length: n }, function(_, k){ return 30 + (k + 0.5) * (H - 60) / n; }); }), svg = '', ST = o.act ? (o.stage === undefined ? 99 : o.stage) : -1;
  var getW; if (o.model) { var dl = denseLayers(o.model), cache = dl.map(function(l){ var ker = l.getWeights()[0]; return { d: ker.dataSync(), cols: ker.shape[1] }; }); getW = function(l, a, b){ return cache[l].d[a * cache[l].cols + b]; }; } else { var rnd = rng(o.seed || 3), memo = {}; getW = function(l, a, b){ var k = l + '_' + a + '_' + b; if (!(k in memo)) memo[k] = rnd() + rnd() + rnd() - 1.5; return memo[k]; }; }
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
function nparams(sizes){ var t = 0; for (var q = 0; q < sizes.length - 1; q++) t += (sizes[q] + 1) * sizes[q + 1]; return t; }
function lossChart(canvas, hist, best, upto){
  var n = upto === undefined ? hist.tr.length : upto, pts1 = [], pts2 = []; for (var i = 0; i < n; i++) { pts1.push([i + 1, hist.tr[i]]); pts2.push([i + 1, hist.va[i]]); }
  var g = lineChart(canvas, [{ pts: pts1, color: C.char }, { pts: pts2, color: C.red }], { xl: 'epoch', yl: 'loss', y0: 0, x0: 1, x1: Math.max(2, n), xdec: 0, ydec: 2 }); if (best && g.sx && n >= best) { var x = g.x, X = g.sx(best); x.strokeStyle = C.dred; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(X, g.box.t); x.lineTo(X, g.box.b); x.stroke(); x.setLineDash([]); x.fillStyle = C.dred; x.font = '600 11px Aptos, Segoe UI, sans-serif'; x.textAlign = X > g.w / 2 ? 'right' : 'left'; x.fillText('best epoch ' + best, X + (X > g.w / 2 ? -6 : 6), g.box.t + 12); } return g;
}

/* ---------- N1: a neuron is a weighted sum ---------- */
L3.add({ short: 'One neuron', title: 'A neuron is a weighted sum followed by a switch',
  change: 'Press Play, then drag the weight and bias sliders inside the picture.',
  watch: 'Inputs times weights, added to a bias, then ReLU turns negative totals into zero. Lines are red for positive contributions and grey for negative.',
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
    function pickCmp(){ var f = activeFeats().slice(0, 3); while (f.length < 3) f.push(D.BASE[f.length]); feats = f; var i = D.TEST[Math.floor(Math.random() * D.TEST.length)]; cur = i; var P = nnPrep(); x = f.map(function(ft){ var mu = mean(D.TRAIN.map(function(k){ return D.COL[ft][k]; })), v = D.TRAIN.map(function(k){ return D.COL[ft][k]; }), sd = Math.sqrt(mean(v.map(function(q){ return (q - mu) * (q - mu); }))) || 1; return clamp((D.COL[ft][i] - mu) / sd, -3, 3); }); }
    function calc(){ var p = x.map(function(v, k){ return v * w[k]; }), z = p[0] + p[1] + p[2] + bias; return { p: p, z: z, a: Math.max(0, z) }; }
    function draw(){
      var c = calc(), s = '', ys = [102, 214, 326], cx = 591, cy = 231; var mx = Math.max(1, Math.abs(c.p[0]), Math.abs(c.p[1]), Math.abs(c.p[2]));
      c.p.forEach(function(pv, k){ s += '<line x1="476" y1="' + ys[k] + '" x2="' + (cx - 54) + '" y2="' + (cy + (k - 1) * 18) + '" stroke="' + (pv >= 0 ? C.red : C.bg) + '" stroke-width="' + (1 + 7 * Math.abs(pv) / mx).toFixed(1) + '" stroke-opacity=".85" stroke-linecap="round"/>'; s += '<line x1="238" y1="' + ys[k] + '" x2="300" y2="' + ys[k] + '" stroke="' + C.bg + '" stroke-width="2"/>'; s += '<text x="396" y="' + (ys[k] - 52) + '" font-size="11" fill="#6B7280" text-anchor="middle"></text>'; s += '<text x="482" y="' + (ys[k] - 6) + '" font-size="12" font-weight="700" fill="' + (pv >= 0 ? C.red : C.mid) + '">' + (pv >= 0 ? '+' : '') + pv.toFixed(2) + '</text>'; });
      s += '<line x1="476" y1="418" x2="' + (cx - 40) + '" y2="' + (cy + 38) + '" stroke="' + (bias >= 0 ? C.red : C.bg) + '" stroke-width="2" stroke-opacity=".8"/>'; s += '<line x1="' + (cx + 55) + '" y1="' + cy + '" x2="690" y2="' + cy + '" stroke="' + C.red + '" stroke-width="2.5"/>'; qs('svg', svg).innerHTML = s;
      qs('.zv', sum).textContent = c.z.toFixed(2); qs('.av', op).textContent = c.a.toFixed(2); qs('.am', op).textContent = c.z > 0 ? 'The total is positive, so the neuron passes it on.' : 'The total is negative, so the neuron outputs zero.';
      rows.forEach(function(r, k){ qs('.fn', r).textContent = feats[k]; qs('.xv', r).textContent = x[k].toFixed(2); qs('.xr', r).textContent = 'raw value ' + fmtFeat(feats[k], D.COL[feats[k]][cur]) + ', rescaled'; }); sl.forEach(function(sx, k){ qs('.wv', sx).textContent = (w[k] >= 0 ? '+' : '') + w[k].toFixed(1); qs('input', sx).value = w[k]; }); qs('.bv', bp).textContent = bias.toFixed(1); qs('input', bp).value = bias;
      var g = cv(qs('canvas', rp)), cx2 = g.x, b = { l: 30, r: g.w - 10, t: 10, b: g.h - 28 }, lo = -3, hi = 3, sxp = function(v){ return b.l + (v - lo) / (hi - lo) * (b.r - b.l); }, syp = function(v){ return b.b - (clamp(v, 0, 3) / 3) * (b.b - b.t); };
      cx2.strokeStyle = C.silver; cx2.beginPath(); cx2.moveTo(b.l, b.b); cx2.lineTo(b.r, b.b); cx2.moveTo(sxp(0), b.t); cx2.lineTo(sxp(0), b.b); cx2.stroke(); cx2.strokeStyle = C.char; cx2.lineWidth = 3; cx2.beginPath(); cx2.moveTo(sxp(lo), syp(0)); cx2.lineTo(sxp(0), syp(0)); cx2.lineTo(sxp(hi), syp(hi)); cx2.stroke();
      cx2.fillStyle = C.red; cx2.beginPath(); cx2.arc(sxp(clamp(c.z, lo, hi)), syp(c.a), 6, 0, 6.3); cx2.fill(); cx2.fillStyle = C.mid; cx2.font = '11px Aptos, Segoe UI, sans-serif'; cx2.textAlign = 'center'; cx2.fillText('total', (b.l + b.r) / 2, g.h - 8); cx2.fillText('0', sxp(0), b.b + 14); }
    sl.forEach(function(sx, k){ qs('input', sx).addEventListener('input', function(){ api.tok++; w[k] = +this.value; draw(); }); }); qs('input', bp).addEventListener('input', function(){ api.tok++; bias = +this.value; draw(); });
    qs('[data-next]', api.sbar).addEventListener('click', function(){ api.tok++; pickCmp(); draw(); });
    function show(list, on){ list.forEach(function(e){ e.classList.toggle('on', on); }); }
    function render(){ if (!built) { pickCmp(); built = true; } show(rows, true); show(sl, true); show([bp, sum, rp, op], true); draw(); }
    async function play(tok){ if (!built) { pickCmp(); built = true; } show(rows.concat(sl, [bp, sum, rp, op]), false); w = [0.9, -0.6, 0.4]; bias = 0.2; draw(); await api.sleep(300);
      show(rows, true); await api.beat('A neuron receives numbers. Here are three features of one compound, rescaled so they are comparable.', 3400);
      show(sl, true); await api.beat('Each input has a weight. A positive weight pushes the neuron up. A negative weight pushes it down.', 3400);
      api.cap('Multiply each input by its weight. The contributions are drawn as lines: red adds, grey subtracts.'); await api.sleep(3400);
      show([sum, bp], true); await api.beat('Add the contributions and a bias to get the total.', 3000);
      show([rp, op], true); await api.beat('ReLU is a switch. A positive total passes through. A negative total becomes zero.', 3400);
      api.cap('Now change a weight and watch the output respond.'); var w0 = w[0]; await api.anim(2400, function(t){ w[0] = lerp(w0, -w0, t); draw(); }); await api.sleep(1200); await api.anim(2000, function(t){ w[0] = lerp(-w0, w0, t); draw(); });
      api.key('A network is many of these neurons wired together. Training finds good weights for all of them.'); }
    return { render: render, play: play };
  } });

/* ---------- N2: layers and units ---------- */
L3.add({ short: 'Layers and neurons', title: 'Layers and neurons set how much the network can learn',
  change: 'Drag Hidden layers, Neurons per layer and Dropout in the options, or press Play.',
  watch: 'The diagram follows your settings. The count on the right is every weight and bias the network must learn.',
  build: function(api){
    var dp = api.el('pn', 'left:28px;top:28px;width:640px;height:408px', '<svg viewBox="0 0 620 380" width="620" height="380"></svg>');
    var cp = api.el('pn', 'left:688px;top:28px;width:244px;height:408px', '<h4>Parameters to learn</h4><div class="big pc" style="color:' + C.red + '">0</div><div class="note" style="font-size:12px;color:#6B7280;margin-bottom:8px">weights and biases</div><div class="pl" style="font-size:12px"></div>');
    var seed = 3;
    function draw(p){ p = p || NP(); var f = activeFeats(); if (!f.length) return; var sizes = netSizes(f.length, p.layers, p.units); drawNetSVG(qs('svg', dp), { W: 620, H: 380, sizes: sizes, feats: f, drop: p.drop, seed: seed, flow: true, maxShow: 8 });
      var rows = '', tot = 0; for (var q = 0; q < sizes.length - 1; q++) { var wts = sizes[q] * sizes[q + 1], bs = sizes[q + 1]; tot += wts + bs; rows += '<div style="display:flex;justify-content:space-between;border-bottom:1px solid #D1D8E0;padding:4px 0"><span>' + (q === 0 ? 'inputs' : 'hidden ' + q) + ' to ' + (q === sizes.length - 2 ? 'output' : 'hidden ' + (q + 1)) + '</span><b>' + (wts + bs).toLocaleString() + '</b></div><div style="font-size:11px;color:#6B7280;margin-bottom:4px">' + sizes[q] + ' by ' + sizes[q + 1] + ' weights plus ' + bs + ' biases</div>'; }
      qs('.pl', cp).innerHTML = rows; qs('.pc', cp).textContent = tot.toLocaleString(); }
    function render(){ draw(); }
    async function play(tok){ var base = NP(); $('nlayers').value = 1; $('nunits').value = 8; $('ndrop').value = 0; ['nlayers', 'nunits', 'ndrop'].forEach(function(id){ $('o-' + id).textContent = $(id).value; }); seed = 5; draw(); await api.beat('One hidden layer of 8 neurons: the smallest useful network.', 3000);
      for (var l = 2; l <= 4; l++) { $('nlayers').value = l; $('o-nlayers').textContent = l; draw(); api.cap('Hidden layers: ' + l + '. Each added layer can combine the features found by the layer before it.'); await api.sleep(2400); }
      [32, 128].forEach(function(u){ }); $('nunits').value = 32; $('o-nunits').textContent = 32; draw(); api.cap('32 neurons per layer: more neurons let each layer learn more patterns at once, and the parameter count jumps.'); await api.sleep(3200);
      $('nunits').value = 128; $('o-nunits').textContent = 128; draw(); api.cap('128 neurons per layer. This is now a large network for a few thousand compounds.'); await api.sleep(3200);
      $('ndrop').value = 0.3; $('o-ndrop').textContent = 0.3; draw(); api.cap('Dropout 0.3: dashed neurons are switched off for one training step. Different neurons are dropped each step.'); await api.sleep(3400);
      ['nlayers', 'nunits', 'ndrop'].forEach(function(id){ var v = id === 'nlayers' ? base.layers : (id === 'nunits' ? base.units : base.drop); $(id).value = v; $('o-' + id).textContent = v; }); draw(); L3.refresh(); api.key('More layers and neurons mean more capacity. Capacity needs enough data to constrain it, or the network overfits.'); }
    return { render: render, play: play };
  } });

/* ---------- N3: learning rate ---------- */
L3.add({ short: 'Learning rate', title: 'The learning rate sets the size of each step downhill',
  change: 'Drag Learning rate in the options, then press Play.',
  watch: 'The colours show how wrong a one-neuron model is for each pair of weights. Training walks downhill. Three step sizes are compared.',
  build: function(api){
    var mp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Error for every pair of weights, one neuron on one feature</h4><canvas width="540" height="350"></canvas>');
    var sp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px', '<h4>Three step sizes</h4><div class="lr"></div>');
    var tr = null, DATAX = null, DATAY = null, paths = [];
    var gx = [-2.2, 2.2], gy = [-2.2, 2.2];
    function prep(){ var f = activeFeats()[0] || D.BASE[0], idx = D.TRAIN.filter(function(_, k){ return k % 3 === 0; }), xv = idx.map(function(i){ return D.COL[f][i]; }), yv = idx.map(function(i){ return D.Y[i]; }), mx = mean(xv), my = mean(yv), sx = Math.sqrt(mean(xv.map(function(v){ return (v - mx) * (v - mx); }))) || 1, sy = Math.sqrt(mean(yv.map(function(v){ return (v - my) * (v - my); }))) || 1; DATAX = xv.map(function(v){ return (v - mx) / sx; }); DATAY = yv.map(function(v){ return (v - my) / sy; }); tr = { f: f }; }
    function loss(w, b){ var s = 0, n = DATAX.length; for (var i = 0; i < n; i++) { var e = DATAY[i] - (w * DATAX[i] + b); s += e * e; } return s / n; }
    function grad(w, b){ var gw = 0, gb = 0, n = DATAX.length; for (var i = 0; i < n; i++) { var e = DATAY[i] - (w * DATAX[i] + b); gw += -2 * e * DATAX[i]; gb += -2 * e; } return [gw / n, gb / n]; }
    function lrs(){ var mine = +$('nlr').value * 50; return [{ n: 'Small', lr: 0.02, c: C.bg }, { n: 'Yours', lr: mine, c: C.red }, { n: 'Too large', lr: 1.04, c: C.char }]; }
    function runPaths(){ var out = lrs().map(function(s){ var w = -1.9, b = 1.6, pts = [[w, b, loss(w, b)]]; for (var k = 0; k < 45; k++) { var g = grad(w, b); w -= s.lr * g[0]; b -= s.lr * g[1]; pts.push([w, b, loss(w, b)]); } return { s: s, pts: pts }; }); paths = out; }
    var heat = null;
    function drawHeat(){ var g = cv(qs('canvas', mp)), x = g.x, W = g.w, H = g.h, cell = 6; if (!heat || heat.key !== tr.f) { var cols = Math.ceil(W / cell), rws = Math.ceil(H / cell), vals = [], mxv = 0; for (var r = 0; r < rws; r++) for (var c = 0; c < cols; c++) { var w = gx[0] + (c + 0.5) / cols * (gx[1] - gx[0]), b = gy[1] - (r + 0.5) / rws * (gy[1] - gy[0]), l = loss(w, b); vals.push(l); mxv = Math.max(mxv, Math.min(l, 6)); } heat = { key: tr.f, vals: vals, cols: cols, rws: rws }; }
      for (var r2 = 0; r2 < heat.rws; r2++) for (var c2 = 0; c2 < heat.cols; c2++) { var v = Math.min(heat.vals[r2 * heat.cols + c2], 6) / 6, t = Math.sqrt(v); x.fillStyle = 'rgb(' + Math.round(lerp(255, 225, t)) + ',' + Math.round(lerp(255, 36, t) * (1 - 0.15 * t)) + ',' + Math.round(lerp(255, 42, t)) + ')'; x.fillRect(c2 * cell, r2 * cell, cell, cell); }
      x.strokeStyle = 'rgba(52,55,63,.35)'; x.lineWidth = 1; x.beginPath(); x.moveTo(W / 2, 0); x.lineTo(W / 2, H); x.moveTo(0, H / 2); x.lineTo(W, H / 2); x.stroke(); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText('weight on ' + tr.f, W / 2, H - 6); x.save(); x.translate(12, H / 2); x.rotate(-Math.PI / 2); x.fillText('bias', 0, 0); x.restore(); x.textAlign = 'left'; x.fillText('white = low error, red = high error', 8, 14); return g; }
    function toPx(g, w, b){ return [(w - gx[0]) / (gx[1] - gx[0]) * g.w, (gy[1] - b) / (gy[1] - gy[0]) * g.h]; }
    function draw(upto){ prep0(); var g = drawHeat(), x = g.x; var opt = toPx(g, 0.0, 0.0); x.fillStyle = '#fff'; x.strokeStyle = C.char; x.lineWidth = 2; x.beginPath(); x.arc(opt[0], opt[1], 7, 0, 6.3); x.fill(); x.stroke(); x.fillStyle = C.char; x.fillText('best weights', opt[0] + 10, opt[1] - 8);
      paths.forEach(function(p){ var n = Math.min(upto, p.pts.length - 1); x.strokeStyle = p.s.c; x.lineWidth = 3; x.lineJoin = 'round'; x.beginPath(); for (var k = 0; k <= n; k++) { var q = toPx(g, clamp(p.pts[k][0], -4, 4), clamp(p.pts[k][1], -4, 4)); k ? x.lineTo(q[0], q[1]) : x.moveTo(q[0], q[1]); } x.stroke(); var e = toPx(g, clamp(p.pts[n][0], -4, 4), clamp(p.pts[n][1], -4, 4)); x.fillStyle = p.s.c; x.strokeStyle = '#fff'; x.lineWidth = 2; x.beginPath(); x.arc(e[0], e[1], 6, 0, 6.3); x.fill(); x.stroke(); });
      qs('.lr', sp).innerHTML = paths.map(function(p){ var n = Math.min(upto, p.pts.length - 1), L = p.pts[n][2]; return '<div style="margin:12px 0;border-left:4px solid ' + p.s.c + ';padding-left:10px"><b>' + p.s.n + '</b> <span class="note" style="font-size:12px;color:#6B7280">step size ' + p.s.lr.toFixed(2) + '</span><div style="font-size:12.5px">after ' + n + ' steps, error <b>' + (L > 99 ? 'exploding' : L.toFixed(2)) + '</b></div></div>'; }).join('') + '<p style="font-size:12px;color:#6B7280;margin-top:10px">A simplified picture: one neuron with two weights and plain gradient descent. Real networks have thousands of weights and use Adam, but the step-size trade-off is the same.</p>'; }
    function prep0(){ if (!tr || tr.f !== (activeFeats()[0] || D.BASE[0])) prep(); runPaths(); }
    function render(){ prep0(); draw(45); }
    async function play(tok){ prep0(); draw(0); await api.beat('The network starts with arbitrary weights. Every pair of weights has an error, shown as a colour. White is low error.', 3400);
      api.cap('Training steps downhill. The step size is the learning rate.'); await api.anim(ANIM ? 5200 : 1, function(t){ draw(Math.floor(t * 45)); }); draw(45);
      api.cap('A small step is safe but slow. Your step lands near the bottom. A step that is too large overshoots and bounces out.'); await api.sleep(4200); api.key('Too small wastes time. Too large never settles. Try the learning rate in the options and replay.'); }
    return { render: render, play: play };
  } });

/* ---------- N4: live training ---------- */
L3.add({ short: 'Training', title: 'Training repeats: predict, measure the error, nudge every weight',
  change: 'Press Play, or Train network in the options. Change layers, neurons, learning rate or epochs first.',
  watch: 'Left: the weights change as the network learns. Top right: error on the studied compounds (dark) and on a validation set (red). Bottom right: predictions on unseen compounds.',
  auto: false,
  build: function(api){
    var np = api.el('pn', 'left:28px;top:28px;width:450px;height:408px', '<svg viewBox="0 0 430 380" width="430" height="380"></svg>');
    var lp = api.el('pn', 'left:498px;top:28px;width:434px;height:206px', '<h4>Error per epoch</h4><canvas width="412" height="160"></canvas>');
    var sp = api.el('pn', 'left:498px;top:246px;width:434px;height:190px', '<h4>Unseen compounds</h4><canvas width="412" height="146"></canvas>');
    function draw(model, feats, training){ drawNetSVG(qs('svg', np), { W: 430, H: 380, sizes: netSizes(feats.length, NP().layers, NP().units), feats: feats, model: model, training: training, maxShow: 8, padL: 78, padR: 40 }); }
    function showFinal(){ if (!NN.m) { drawNetSVG(qs('svg', np), { W: 430, H: 380, sizes: netSizes(Math.max(1, activeFeats().length), NP().layers, NP().units), feats: activeFeats(), maxShow: 8, padL: 78, padR: 40, flow: false }); cv(qs('canvas', lp)); cv(qs('canvas', sp)); return; }
      draw(NN.m, NN.feats, false); lossChart(qs('canvas', lp), NN.hist, NN.best); scatterPV(qs('canvas', sp), NN.pred, D.TEST, 'unseen R2 ' + NN.r2.toFixed(2), { max: 200 }); }
    function render(){ showFinal(); }
    async function play(tok){
      if (NN.busy) return; NN.busy = true; var btn = $('ntrain'); btn.disabled = true; var st = $('nst'), p = NP(), f = activeFeats(); if (!f.length) { st.textContent = 'Tick at least one feature.'; NN.busy = false; btn.disabled = false; return; }
      try { nnPrep(); st.textContent = 'Training'; api.cap('The network starts with random weights, so its predictions are poor.'); var tmp = new Float64Array(D.N);
        var res = await nnTrain({ keep: true, onEpoch: async function(ep, s){ api.checkTok(tok); st.textContent = 'Epoch ' + (ep + 1) + ' of ' + p.epochs; progress((ep + 1) / p.epochs);
          if (ep % 2 === 0 || ep === p.epochs - 1) { lossChart(qs('canvas', lp), { tr: s.tr, va: s.va }, 0); var o = tf.tidy(function(){ return s.model.predict(s.prep.allX).dataSync(); }); for (var i = 0; i < D.N; i++) tmp[i] = o[i] + s.prep.ymu; scatterPV(qs('canvas', sp), tmp, D.TEST, 'epoch ' + (ep + 1) + ', R2 ' + r2(D.TEST, tmp).toFixed(2), { max: 200 }); draw(s.model, s.prep.feats, true); }
          if (ep === 0) api.cap('Epoch 1: one pass through the studied compounds. The error drops quickly at first.'); if (ep === Math.floor(p.epochs * 0.2)) api.cap('The error curve flattens as the easy patterns are learned.'); if (ep === Math.floor(p.epochs * 0.6)) api.cap('If the red validation curve turns upward while the dark one keeps falling, the network is memorising.'); await new Promise(function(r){ setTimeout(r, ANIM ? 35 : 0); }); } });
        showFinal(); st.textContent = 'Done. Unseen R2 ' + res.r2.toFixed(2); api.cap('Training stopped. Best epoch ' + res.best + (p.early ? ', and those weights were restored.' : '.')); progress(1); await api.sleep(3200); api.key('Training lowers the error on studied compounds. Only the unseen score tells you whether it generalises.'); }
      catch (e) { if (!(e instanceof Cancel)) { st.textContent = 'Error: ' + e.message; console.error(e); } }
      NN.busy = false; btn.disabled = false; }
    return { render: render, play: play };
  } });

/* ---------- N5: dropout and early stopping ---------- */
L3.add({ short: 'Dropout and early stopping', title: 'Dropout and the best epoch protect against overfitting',
  change: 'Press Run comparison. Change Dropout, Epochs or Neurons per layer in the options first.',
  watch: 'Validation error for a network with no dropout and one with your dropout. The dashed line marks the best epoch of each.',
  auto: false,
  build: function(api){
    var np = api.el('pn', 'left:28px;top:28px;width:360px;height:408px', '<h4>Neurons switched off in each training step</h4><svg viewBox="0 0 340 330" width="340" height="330"></svg><div class="dn note" style="font-size:12px;color:#6B7280"></div>');
    var cp = api.el('pn', 'left:408px;top:28px;width:524px;height:290px', '<h4>Validation error per epoch</h4><canvas width="502" height="240"></canvas>');
    var rp = api.el('pn', 'left:408px;top:330px;width:524px;height:106px', '<div class="cr" style="font-size:13px;color:#4a4f5a">Press Run comparison.</div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-run>Run comparison</button>'); var flick = null, cmp = null;
    function maskDraw(seed){ var p = NP(), f = activeFeats(); if (!f.length) return; drawNetSVG(qs('svg', np), { W: 340, H: 330, sizes: netSizes(f.length, p.layers, p.units), feats: f, drop: p.drop, seed: seed, flow: true, maxShow: 8, padL: 70, padR: 30, noLabels: true }); qs('.dn', np).textContent = 'Dropout ' + p.drop + ': about ' + Math.round(p.drop * 100) + '% of hidden neurons are switched off at each step. Dashed circles are off.'; }
    function drawCmp(){ if (!cmp) { cv(qs('canvas', cp)); return; } var a = cmp.a, b = cmp.b, n = Math.max(a.hist.va.length, b.hist.va.length), g = lineChart(qs('canvas', cp), [{ pts: a.hist.va.map(function(v, i){ return [i + 1, v]; }), color: C.char, nodot: true, label: 'no dropout' }, { pts: b.hist.va.map(function(v, i){ return [i + 1, v]; }), color: C.red, nodot: true, label: 'dropout ' + cmp.p.drop }], { xl: 'epoch', yl: 'validation error', x0: 1, x1: n, y0: 0, xdec: 0, ydec: 2 }), x = g.x;
      [[a, C.char], [b, C.red]].forEach(function(r){ var X = g.sx(r[0].best); x.strokeStyle = r[1]; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(X, g.box.t); x.lineTo(X, g.box.b); x.stroke(); x.setLineDash([]); }); }
    function summary(){ if (!cmp) return; var a = cmp.a, b = cmp.b; qs('.cr', rp).innerHTML = '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px"><div><div class="lbl">No dropout</div><div style="font-size:13px">best epoch <b>' + a.best + '</b>, unseen R2 <b>' + a.r2.toFixed(2) + '</b></div></div><div><div class="lbl" style="color:' + C.red + '">Dropout ' + cmp.p.drop + '</div><div style="font-size:13px">best epoch <b>' + b.best + '</b>, unseen R2 <b>' + b.r2.toFixed(2) + '</b></div></div></div><p style="font-size:12px;color:#6B7280;margin-top:6px">Both networks restore their best epoch. Without that, the last epoch would score lower once validation error has turned upward.</p>'; }
    async function run(){ var btn = qs('[data-run]', ctl); btn.disabled = true; var p = NP(), st = qs('.cr', rp); if (!activeFeats().length) { st.textContent = 'Tick at least one feature.'; btn.disabled = false; return; }
      p.epochs = Math.min(p.epochs, 60); st.textContent = 'Training the network without dropout'; progress(0.2); await tick(); var a = await nnTrain({ params: Object.assign({}, p, { drop: 0 }) }); st.textContent = 'Training the network with dropout ' + p.drop; progress(0.6); await tick(); var b = await nnTrain({ params: p }); cmp = { a: a, b: b, p: p }; progress(1); drawCmp(); summary(); btn.disabled = false; }
    qs('[data-run]', ctl).addEventListener('click', function(){ api.tok++; api.clearTexts(); run().then(function(){ api.cap('The dashed lines mark each network\'s best epoch. Dropout usually keeps the validation error lower for longer.'); }); });
    function render(){ maskDraw(5); drawCmp(); summary(); }
    async function play(tok){ if (flick) clearInterval(flick); var s = 5; maskDraw(s); api.cap('At each training step a random set of neurons is switched off. The network cannot lean on any single neuron.'); for (var k = 0; k < 7; k++) { s++; maskDraw(s); await api.sleep(900); } api.cap('Press Run comparison to train twice, once without dropout and once with your setting.'); await api.sleep(2200); api.key('Dropout trades some training accuracy for a network that generalises. Keeping the best epoch adds a second safeguard.'); }
    return { render: render, play: play };
  } });
