"use strict";
/* Lesson 8: Tree algorithms. How a split is found, when a tree stops, bagging, gradient boosting, tuning and library designs. */
var L7 = new Lesson('treealgo', 'Tree algorithms');
L7.optsHTML = function(){ return optGroup('Endpoint', seg([['HLM', 'HLM clearance'], ['logD', 'logD']], EP, 'data-ep'), 'HLM is Biogen data. logD is AstraZeneca data.') + '<div class="og"><div class="ol">Features</div><div class="hint">These steps use all eight descriptors: logP, TPSA, MW, HBD, HBA, RotB, Fsp3 and AromRings.</div></div>'; };
L7.onData = function(){ };

function taFeats(){ return D.BASE.slice(); }
function taLeaves(n){ return n.f === undefined ? 1 : taLeaves(n.l) + taLeaves(n.r); }
function taSizes(n, out){ if (n.f === undefined) out.push(n.n); else { taSizes(n.l, out); taSizes(n.r, out); } return out; }
function taDepth(n){ return n.f === undefined ? 0 : 1 + Math.max(taDepth(n.l), taDepth(n.r)); }
function taStore(){ D.cache.ta = D.cache.ta || { hist: [], grid: null, evals: {} }; return D.cache.ta; }
function taPearson(a, b, idx){
  var n = idx.length, ma = 0, mb = 0, k; for (k = 0; k < n; k++) { ma += a[idx[k]]; mb += b[idx[k]]; } ma /= n; mb /= n;
  var sab = 0, saa = 0, sbb = 0; for (k = 0; k < n; k++) { var x = a[idx[k]] - ma, y = b[idx[k]] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : null;
}
function taMae(idx, pred){ var s = 0; idx.forEach(function(i){ s += Math.abs(D.Y[i] - pred[i]); }); return s / idx.length; }
function taRow(label, value, color){ return '<div class="rrow">' + label + '<b' + (color ? ' style="color:' + color + '"' : '') + '>' + value + '</b></div>'; }
function taDebounce(fn, ms){ var t = null; return function(){ clearTimeout(t); t = setTimeout(fn, ms); }; }

/* ---------- Step 1: split search ---------- */
L7.add({ short: 'Split search', title: 'How a tree finds its best question',
  change: 'Choose a feature and a number of bins. Compare the exact search with the binned search.',
  watch: 'A split sends compounds below a threshold to the left and the rest to the right, and each side predicts its own average. The gain of a split is how much it reduces the sum of squared errors (SSE), the total of the squared gaps between each compound and its group average: gain = SSE before the split minus SSE after. The exact search tries a threshold between every pair of neighbouring values. The binned search first sorts the values into bins and tries only the bin edges. Fast libraries such as LightGBM work this way.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4 class="pt">Gain at each threshold, at the root of the tree</h4><canvas width="600" height="318" style="margin-top:2px"></canvas><div class="note" style="font-size:12px;margin-top:4px">Grey line: exact search. Red dots: binned search. Dashed lines: the best threshold of each.</div>');
    var side = api.el('pn', 'left:668px;top:28px;width:264px;height:408px;overflow:auto', '<h4>Best split found</h4><div class="sd"></div>');
    var ctl = api.addControl('Feature <select class="sf"></select> Bins <select class="sb">' + [4, 8, 16, 32, 64, 128, 256].map(function(v){ return '<option>' + v + '</option>'; }).join('') + '</select>');
    var sf = qs('.sf', ctl), sb = qs('.sb', ctl); sb.value = '32';
    function fillSel(){ var c = sf.value; sf.innerHTML = D.BASE.map(function(x){ return '<option>' + x + '</option>'; }).join(''); if (D.BASE.indexOf(c) >= 0) sf.value = c; }
    function scan(f, nb){
      var idx = D.TRAIN, n = idx.length, x = new Float64Array(n), y = new Float64Array(n), tot = 0, tot2 = 0, k, MIN = 5;
      for (k = 0; k < n; k++) { x[k] = D.COL[f][idx[k]]; y[k] = D.Y[idx[k]]; tot += y[k]; tot2 += y[k] * y[k]; }
      var sse0 = tot2 - tot * tot / n;
      function gain(cl, sl){ var cr = n - cl, sr = tot - sl; return sl * sl / cl + sr * sr / cr - tot * tot / n; }
      var ord = []; for (k = 0; k < n; k++) ord.push(k); ord.sort(function(a, b){ return x[a] - x[b]; });
      var ex = [], eb = null, cl = 0, sl = 0, uniq = 1;
      for (k = 0; k < n - 1; k++) { var a = ord[k], b = ord[k + 1]; cl++; sl += y[a]; if (x[b] > x[a]) { uniq++; if (cl >= MIN && n - cl >= MIN) { var g = gain(cl, sl), th = (x[a] + x[b]) / 2; ex.push([th, g]); if (!eb || g > eb.g) eb = { th: th, g: g, cl: cl, sl: sl }; } } }
      var B = fineBins(f, nb), e = B.edges[f], codes = B.codes[f], nbins = e.length + 1, cnt = new Float64Array(nbins), sm = new Float64Array(nbins);
      for (k = 0; k < n; k++) { var c = codes[idx[k]]; cnt[c]++; sm[c] += y[k]; }
      var hb = [], bb = null; cl = 0; sl = 0;
      for (var q = 0; q < nbins - 1; q++) { cl += cnt[q]; sl += sm[q]; if (cl >= MIN && n - cl >= MIN) { var g2 = gain(cl, sl); hb.push([e[q], g2]); if (!bb || g2 > bb.g) bb = { th: e[q], g: g2, cl: cl, sl: sl }; } }
      return { n: n, sse0: sse0, tot: tot, ex: ex, eb: eb, hb: hb, bb: bb, uniq: uniq, nbins: nbins };
    }
    function draw(){
      var f = sf.value || D.BASE[0], nb = +sb.value, r = scan(f, nb), g = cv(qs('canvas', plot)), x = g.x, box = { l: 52, r: g.w - 14, t: 14, b: g.h - 40 };
      if (!r.ex.length || !r.eb) { qs('.sd', side).innerHTML = '<p>No valid split.</p>'; return; }
      var x0 = r.ex[0][0], x1 = r.ex[r.ex.length - 1][0], gm = r.eb.g * 1.08; if (x1 <= x0) x1 = x0 + 1;
      axes(g, box, [x0, x1], [0, gm], f, 'gain (drop in SSE)', 4, { ydec: 0 });
      var sx = function(v){ return box.l + (v - x0) / (x1 - x0) * (box.r - box.l); }, sy = function(v){ return box.b - v / gm * (box.b - box.t); };
      x.strokeStyle = C.bg; x.lineWidth = 1.6; x.beginPath(); r.ex.forEach(function(p, i){ i ? x.lineTo(sx(p[0]), sy(p[1])) : x.moveTo(sx(p[0]), sy(p[1])); }); x.stroke();
      x.fillStyle = C.red; r.hb.forEach(function(p){ x.beginPath(); x.arc(sx(p[0]), sy(p[1]), 3.4, 0, 6.3); x.fill(); });
      x.setLineDash([5, 4]); x.lineWidth = 1.4; x.strokeStyle = C.char; x.beginPath(); x.moveTo(sx(r.eb.th), box.t); x.lineTo(sx(r.eb.th), box.b); x.stroke(); x.strokeStyle = C.red; x.beginPath(); x.moveTo(sx(r.bb.th), box.t); x.lineTo(sx(r.bb.th), box.b); x.stroke(); x.setLineDash([]);
      var n = r.n, cl = r.eb.cl, ml = r.eb.sl / cl, mr = (r.tot - r.eb.sl) / (n - cl);
      qs('.sd', side).innerHTML = '<div style="font-size:12.5px;font-weight:700;color:' + C.char + ';margin-bottom:4px">Exact search</div>' + taRow('Threshold', r.eb.th.toFixed(2)) + taRow('Gain', r.eb.g.toFixed(1)) + taRow('Thresholds tried', (r.uniq - 1).toLocaleString()) +
        '<div style="font-size:12.5px;font-weight:700;color:' + C.red + ';margin:12px 0 4px">Binned search, ' + nb + ' bins</div>' + taRow('Threshold', r.bb.th.toFixed(2), C.red) + taRow('Gain', r.bb.g.toFixed(1), C.red) + taRow('Thresholds tried', (r.nbins - 1).toLocaleString(), C.red) + taRow('Share of the exact gain', Math.round(r.bb.g / r.eb.g * 100) + '%', C.red) +
        '<div style="font-size:12.5px;font-weight:700;color:' + C.char + ';margin:12px 0 4px">The exact split</div>' + taRow('SSE before', r.sse0.toFixed(1)) + taRow('SSE after', (r.sse0 - r.eb.g).toFixed(1)) + taRow('Left group', cl + ' compounds, mean ' + ml.toFixed(2)) + taRow('Right group', (n - cl) + ' compounds, mean ' + mr.toFixed(2));
    }
    sf.addEventListener('change', draw); sb.addEventListener('change', draw);
    return { render: function(){ fillSel(); draw(); } };
  } });

/* ---------- Step 2: growing a tree ---------- */
L7.add({ short: 'Growing a tree', title: 'Depth and leaf size decide how far a tree grows',
  change: 'Drag Maximum depth and Minimum leaf size.',
  watch: 'A tree grows by splitting a group, then splitting each part again. Two settings stop it. Maximum depth limits how many questions in a row a compound can be asked. Minimum leaf size is the fewest compounds a final group, called a leaf, may hold: a split is rejected if either side would fall below it. With no limits a tree keeps splitting until each leaf holds one compound, and it memorises the training set. The left chart varies leaf size at the depth you chose. The right chart varies depth at the leaf size you chose.',
  build: function(api){
    var p1 = api.el('pn', 'left:28px;top:28px;width:446px;height:268px', '<h4>R2 by minimum leaf size</h4><canvas width="420" height="214"></canvas>');
    var p2 = api.el('pn', 'left:486px;top:28px;width:446px;height:268px', '<h4>R2 by maximum depth</h4><canvas width="420" height="214"></canvas>');
    var p3 = api.el('pn', 'left:28px;top:308px;width:904px;height:128px', '<h4>The tree you chose</h4><div class="st" style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px"></div>');
    var ctl = api.addControl('Maximum depth <input class="dp" type="range" min="1" max="14" step="1" value="6" style="width:150px"><b class="dv">6</b> Minimum leaf size <input class="ml" type="range" min="1" max="60" step="1" value="5" style="width:150px"><b class="mv">5</b>');
    var dp = qs('.dp', ctl), ml = qs('.ml', ctl), dv = qs('.dv', ctl), mv = qs('.mv', ctl), LEAFS = [1, 2, 5, 10, 20, 50, 100];
    function ev(d, m){ var s = taStore(), k = d + '|' + m; if (!s.evals[k]) { var t = fitTree(taFeats(), D.TRAIN, d, m, 1), pr = treeAll(t, D.N); s.evals[k] = { tree: t, tr: r2(D.TRAIN, pr), te: r2(D.TEST, pr) }; } return s.evals[k]; }
    function draw(){
      var d = +dp.value, m = +ml.value; dv.textContent = d; mv.textContent = m;
      var bars = LEAFS.map(function(v){ return ev(d, v); }), g1 = cv(qs('canvas', p1)), x = g1.x, box = { l: 44, r: g1.w - 10, t: 10, b: g1.h - 36 }, lo = Math.min(0, Math.min.apply(null, bars.map(function(b){ return b.te; }))), hi = 1;
      axes(g1, box, [0, 1], [lo, hi], '', 'R2', 4, { ydec: 1, xdec: 0 }); x.clearRect(box.l - 24, box.b + 3, box.r - box.l + 30, 22);
      var bw = (box.r - box.l) / LEAFS.length, sy = function(v){ return box.b - (v - lo) / (hi - lo) * (box.b - box.t); }, base = sy(0);
      bars.forEach(function(b, i){ var x0 = box.l + i * bw + 6, w = (bw - 12) / 2; x.fillStyle = C.char; x.fillRect(x0, Math.min(base, sy(b.tr)), w, Math.abs(sy(b.tr) - base)); x.fillStyle = C.red; x.fillRect(x0 + w, Math.min(base, sy(b.te)), w, Math.abs(sy(b.te) - base));
        x.fillStyle = C.char; x.textAlign = 'center'; x.font = (LEAFS[i] === m ? '700 ' : '') + '12px Aptos, Segoe UI, sans-serif'; x.fillText(LEAFS[i], box.l + i * bw + bw / 2, box.b + 16); });
      x.font = '600 11px Aptos, Segoe UI, sans-serif'; x.fillStyle = C.char; x.textAlign = 'left'; x.fillText('dark: training R2', box.l + 6, box.t + 12); x.fillStyle = C.red; x.fillText('red: test R2', box.l + 120, box.t + 12);
      x.fillStyle = C.char; x.textAlign = 'center'; x.fillText('minimum leaf size', (box.l + box.r) / 2, g1.h - 6);
      var tr = [], te = []; for (var q = 1; q <= 12; q++) { var e2 = ev(q, m); tr.push([q, e2.tr]); te.push([q, e2.te]); }
      var g2 = lineChart(qs('canvas', p2), [{ pts: tr, color: C.char, label: 'training' }, { pts: te, color: C.red, label: 'test' }], { xl: 'maximum depth', yl: 'R2', y0: Math.min(0, Math.min.apply(null, te.map(function(p){ return p[1]; }))), y1: 1, x0: 1, x1: 12, ydec: 1, xdec: 0 });
      if (g2.sx && d <= 12) { var xx = g2.x; xx.setLineDash([5, 4]); xx.strokeStyle = C.char; xx.lineWidth = 1.2; xx.beginPath(); xx.moveTo(g2.sx(d), g2.box.t); xx.lineTo(g2.sx(d), g2.box.b); xx.stroke(); xx.setLineDash([]); }
      var cur = ev(d, m), sizes = taSizes(cur.tree, []).sort(function(a, b){ return a - b; });
      qs('.st', p3).innerHTML = [['Leaves', taLeaves(cur.tree)], ['Depth reached', taDepth(cur.tree)], ['Smallest leaf', sizes[0]], ['Median leaf', sizes[Math.floor(sizes.length / 2)]], ['Training R2', cur.tr.toFixed(2)], ['Test R2', cur.te.toFixed(2)]].map(function(c, i){ return '<div><div class="note" style="font-size:12px;color:' + C.char + '">' + c[0] + '</div><div class="big" style="font-size:26px;color:' + (i === 5 ? C.red : C.char) + '">' + c[1] + '</div></div>'; }).join('');
    }
    var deb = taDebounce(draw, 100); dp.addEventListener('input', function(){ dv.textContent = dp.value; deb(); }); ml.addEventListener('input', function(){ mv.textContent = ml.value; deb(); });
    return { render: draw };
  } });

/* ---------- Step 3: bagging ---------- */
L7.add({ short: 'Bagging', title: 'Why averaging many trees works, and what random features add',
  change: 'Set the number of trees and the features tried per split, then press Fit forest. Compare runs.',
  watch: 'Bagging trains each tree on a bootstrap sample: compounds drawn at random with replacement, so some repeat and about a third are left out. A forest also lets each split see only a random subset of the features. The forest predicts the average of its trees. If each tree has variance σ² and any two trees correlate at ρ, the average of B trees has variance ρσ² + (1 - ρ)σ²/B. More trees shrink the second term. Fewer features per split tends to lower ρ. Out-of-bag R2 scores each training compound using only the trees that never saw it, which gives a test-like score without a test set.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Test R2 as trees are added</h4><canvas width="536" height="344"></canvas>');
    var side = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<h4>Latest run</h4><div class="sd"></div>');
    var ctl = api.addControl('Trees <input class="nt" type="range" min="2" max="120" step="1" value="40" style="width:140px"><b class="nv">40</b> Features per split <input class="mt" type="range" min="1" max="8" step="1" value="3" style="width:100px"><b class="mv">3</b> <button type="button" class="btn sm" data-fit>Fit forest</button>');
    var nt = qs('.nt', ctl), mt = qs('.mt', ctl), nv = qs('.nv', ctl), mv = qs('.mv', ctl), busy = false;
    function fitForest_(n, m){
      var feats = taFeats(), rand = rng(21 + m * 7 + n), tr = Int32Array.from(D.TRAIN), N = D.N, per = [], oS = new Float64Array(N), oC = new Int32Array(N), acc = new Float64Array(N), pred = new Float64Array(N), curve = [], t0 = performance.now();
      for (var q = 0; q < n; q++) {
        var boot = new Int32Array(tr.length), mark = new Uint8Array(N); for (var z = 0; z < tr.length; z++) { var i = tr[Math.floor(rand() * tr.length)]; boot[z] = i; mark[i] = 1; }
        var t = buildTree(D.B, feats, D.Y, boot, 10, 3, m, rand), p = treeAll(t, N); per.push(p);
        for (var a = 0; a < tr.length; a++) { var ii = tr[a]; if (!mark[ii]) { oS[ii] += p[ii]; oC[ii]++; } }
        for (var j = 0; j < N; j++) acc[j] += p[j]; for (var j2 = 0; j2 < N; j2++) pred[j2] = acc[j2] / (q + 1); curve.push([q + 1, r2(D.TEST, pred)]);
      }
      var oi = [], op = new Float64Array(N); tr.forEach(function(k){ if (oC[k] > 0) { oi.push(k); op[k] = oS[k] / oC[k]; } });
      var K = Math.min(n, 20), single = 0; for (var s = 0; s < K; s++) single += r2(D.TEST, per[s]); single /= K;
      var cs = [], u, v; for (u = 0; u < K; u++) for (v = u + 1; v < K; v++) { var c = taPearson(per[u], per[v], D.TEST); if (c !== null) cs.push(c); }
      return { n: n, m: m, te: curve[curve.length - 1][1], curve: curve, single: single, rho: cs.length ? mean(cs) : null, oob: oi.length > 20 ? r2(oi, op) : null, secs: (performance.now() - t0) / 1000 };
    }
    function draw(){
      var H = taStore().hist, series = [];
      H.slice(-3).forEach(function(h, i, arr){ var latest = i === arr.length - 1; series.push({ pts: h.curve, color: latest ? C.red : (i === 0 && arr.length === 3 ? C.bg : C.char), w: latest ? 3 : 1.8, label: h.m + ' of 8', nodot: !latest }); });
      if (!series.length) { var g0 = cv(qs('canvas', plot)); g0.x.fillStyle = C.char; g0.x.fillText('Press Fit forest.', 60, 40); qs('.sd', side).innerHTML = '<p class="note">No run yet.</p>'; return; }
      lineChart(qs('canvas', plot), series, { xl: 'number of trees', yl: 'test R2', ydec: 2, xdec: 0, x0: 1 });
      var L = H[H.length - 1];
      qs('.sd', side).innerHTML = taRow('Trees, features per split', L.n + ', ' + L.m + ' of 8') + taRow('Forest test R2', L.te.toFixed(2), C.red) + taRow('One tree, test R2 (average)', L.single.toFixed(2)) + taRow('Correlation between trees, ρ', L.rho === null ? 'n/a' : L.rho.toFixed(2)) + taRow('Out-of-bag R2', L.oob === null ? 'n/a' : L.oob.toFixed(2)) + taRow('Fit time', fmtT(L.secs)) +
        '<div style="font-size:12.5px;font-weight:700;color:' + C.char + ';margin:12px 0 4px">Earlier runs</div>' + (H.length > 1 ? H.slice(-6, -1).reverse().map(function(h){ return taRow(h.n + ' trees, ' + h.m + ' of 8', 'test ' + h.te.toFixed(2) + ', ρ ' + (h.rho === null ? 'n/a' : h.rho.toFixed(2))); }).join('') : '<p class="note" style="font-size:12px">Run again with other settings to compare.</p>');
    }
    async function run(){
      if (busy) return; busy = true; api.busy('Fitting ' + nt.value + ' trees'); await tick(); await tick();
      try { taStore().hist.push(fitForest_(+nt.value, +mt.value)); } catch (e) { console.error(e); }
      busy = false; api.idle(); draw();
    }
    nt.addEventListener('input', function(){ nv.textContent = nt.value; }); mt.addEventListener('input', function(){ mv.textContent = mt.value; });
    qs('[data-fit]', ctl).addEventListener('click', run);
    return { render: async function(){ if (!taStore().hist.length) await run(); else draw(); } };
  } });

/* ---------- Step 4: boosting math ---------- */
function taBoostLoss(loss, delta, rounds, lr, depth, sample){
  var feats = taFeats(), tr = Int32Array.from(D.TRAIN), N = D.N, ym = mean(D.TRAIN.map(function(i){ return D.Y[i]; })), F = new Float64Array(N).fill(ym), res = new Float32Array(N), rand = rng(3), steps = [];
  for (var m = 0; m < rounds; m++) {
    for (var i = 0; i < N; i++) { var e = D.Y[i] - F[i]; res[i] = loss === 'l2' ? e : (loss === 'l1' ? (e > 0 ? 1 : (e < 0 ? -1 : 0)) : clamp(e, -delta, delta)); }
    var bt = buildTree(D.B, feats, res, tr, depth, 10, feats.length, rand);
    if (sample) steps.push(sample.map(function(k){ return { F: F[k], g: res[k], h: predictTree(bt, D.B, k) }; }));
    for (var j = 0; j < N; j++) F[j] += lr * predictTree(bt, D.B, j);
  }
  return { F: F, steps: steps };
}
var TA_LOSS = { l2: 'Squared error', l1: 'Absolute error', hu: 'Huber' };
L7.add({ short: 'Boosting math', title: 'What each boosting round fits, and how the loss changes it',
  change: 'Choose a loss and a round. The panel on the right compares all three losses over 100 rounds.',
  watch: 'Boosting starts from one number, here the training mean, and adds trees one at a time. A loss measures how bad a prediction is. At each round the next tree is fitted to the gradient: the direction that lowers the loss fastest. For squared error that is just the residual, measured minus predicted. Absolute error uses only the sign of the residual, so a few extreme compounds pull less. Huber loss acts like squared error for small residuals and like absolute error for large ones. The table follows five training compounds through the first rounds. In this simplified version each leaf holds the mean of the gradient. Real libraries add a line search or a second-order step.',
  build: function(api){
    var lp = api.el('pn', 'left:28px;top:28px;width:560px;height:176px', '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px"><div><h4>Loss against error</h4><canvas class="c1" width="258" height="106"></canvas></div><div><h4>Gradient against error</h4><canvas class="c2" width="258" height="106"></canvas></div></div><div style="font-size:12px;margin-top:2px;color:' + C.char + '"><b>Squared error</b> dark, <b style="color:#6F8196">absolute error</b> blue grey, <b style="color:#E1242A">Huber</b> red. The chosen loss is drawn thick.</div>');
    var tp = api.el('pn', 'left:28px;top:216px;width:560px;height:220px;overflow:hidden', '<h4 class="th"></h4><div class="tb"></div>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<h4>Three losses, 100 rounds</h4><div class="rs"><p class="note" style="font-size:12.5px">Press Compare losses.</p></div>');
    var ctl = api.addControl('Loss <select class="ls">' + Object.keys(TA_LOSS).map(function(k){ return '<option value="' + k + '">' + TA_LOSS[k] + '</option>'; }).join('') + '</select> Round <input class="rd" type="range" min="1" max="8" step="1" value="1" style="width:110px"><b class="rv">1</b> Learning rate <input class="lr" type="range" min="0.05" max="1" step="0.05" value="0.3" style="width:110px"><b class="lv">0.30</b> <button type="button" class="btn sm" data-cmp>Compare losses</button>');
    var ls = qs('.ls', ctl), rd = qs('.rd', ctl), lr = qs('.lr', ctl), rv = qs('.rv', ctl), lv = qs('.lv', ctl), SAMPLE = null, busy = false;
    function delta(){ var ys = D.TRAIN.map(function(i){ return D.Y[i]; }), m = mean(ys); return 0.5 * Math.sqrt(mean(ys.map(function(v){ return (v - m) * (v - m); }))); }
    function lossFn(l, e, dl){ return l === 'l2' ? 0.5 * e * e : (l === 'l1' ? Math.abs(e) : (Math.abs(e) <= dl ? 0.5 * e * e : dl * (Math.abs(e) - dl / 2))); }
    function gradFn(l, e, dl){ return l === 'l2' ? e : (l === 'l1' ? (e > 0 ? 1 : (e < 0 ? -1 : 0)) : clamp(e, -dl, dl)); }
    function curves(){
      var dl = delta(), sel = ls.value, A = [], B = [], k, e; var order = ['l2', 'l1', 'hu'].sort(function(a, b){ return (a === sel) - (b === sel); });
      var col = { l2: C.char, l1: '#6F8196', hu: C.red };
      function ser(fn, l){ var pts = []; for (k = 0; k <= 60; k++) { e = -2 + 4 * k / 60; pts.push([e, fn(l, e, dl)]); } return { pts: pts, color: col[l], w: l === sel ? 4 : 1.4, nodot: true }; }
      lineChart(qs('.c1', lp), order.map(function(l){ return ser(lossFn, l); }), { xl: 'error (measured minus predicted)', yl: 'loss', x0: -2, x1: 2, y0: 0, y1: 2, ydec: 1, xdec: 0 });
      lineChart(qs('.c2', lp), order.map(function(l){ return ser(gradFn, l); }), { xl: 'error (measured minus predicted)', yl: 'gradient', x0: -2, x1: 2, y0: -1.5, y1: 1.5, ydec: 1, xdec: 0 });
    }
    function table(){
      var l = ls.value, dl = delta(), rate = +lr.value, round = +rd.value;
      if (!SAMPLE) { SAMPLE = []; for (var q = 0; q < 5; q++) SAMPLE.push(D.TRAIN[Math.floor((q + 0.5) * D.TRAIN.length / 5)]); }
      var key = [l, rate].join('|'), s = taStore(); s.bl = s.bl || {}; if (!s.bl[key]) s.bl[key] = taBoostLoss(l, dl, 8, rate, 3, SAMPLE);
      var st = s.bl[key].steps[round - 1];
      qs('.th', tp).textContent = TA_LOSS[l] + ', round ' + round + ', learning rate ' + rate.toFixed(2) + (l === 'hu' ? ', δ ' + dl.toFixed(2) : '');
      qs('.tb', tp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12.5px;color:' + C.char + '"><tr style="text-align:right;font-weight:700;line-height:1.15"><td style="text-align:left">Compound</td><td>Measured</td><td>Prediction before</td><td>Gradient</td><td>Tree output</td><td>Prediction after</td></tr>' + SAMPLE.map(function(k, i){ var o = st[i]; return '<tr style="text-align:right;border-top:1px solid ' + C.silver + ';line-height:1.05"><td style="text-align:left">#' + (i + 1) + '</td><td>' + D.Y[k].toFixed(2) + '</td><td>' + o.F.toFixed(2) + '</td><td>' + o.g.toFixed(2) + '</td><td>' + o.h.toFixed(2) + '</td><td><b>' + (o.F + rate * o.h).toFixed(2) + '</b></td></tr>'; }).join('') + '</table><p class="note" style="font-size:11.5px;margin:4px 0 0">Prediction after = prediction before + learning rate times tree output.</p>';
    }
    function draw(){ rv.textContent = rd.value; lv.textContent = (+lr.value).toFixed(2); curves(); table(); }
    async function compare(){
      if (busy) return; busy = true; api.busy('Fitting three models'); await tick(); await tick(); var dl = delta(), rows = [];
      try { ['l2', 'l1', 'hu'].forEach(function(l){ var f = taBoostLoss(l, dl, 100, 0.1, 3).F; rows.push({ l: l, tr: r2(D.TRAIN, f), te: r2(D.TEST, f), rm: rmse(D.TEST, f), ma: taMae(D.TEST, f) }); }); } catch (e) { console.error(e); }
      busy = false; api.idle(); taStore().cmp = true;
      var best = function(k, lowest){ var v = rows.map(function(r){ return r[k]; }); return lowest ? Math.min.apply(null, v) : Math.max.apply(null, v); };
      qs('.rs', rp).innerHTML = rows.map(function(r){ return '<div style="padding:10px 0;border-top:1px solid ' + C.silver + '"><b style="font-size:14px;color:' + C.char + '">' + TA_LOSS[r.l] + '</b>' + taRow('Test R2', r.te.toFixed(3), r.te === best('te') ? C.red : null) + taRow('Test RMSE', r.rm.toFixed(3), r.rm === best('rm', true) ? C.red : null) + taRow('Test MAE', r.ma.toFixed(3), r.ma === best('ma', true) ? C.red : null) + '</div>'; }).join('') + '<p class="note" style="font-size:12px">Red marks the best of the three. RMSE is the typical miss with large misses counted extra. MAE counts every miss equally. 100 rounds, learning rate 0.10, depth 3.</p>';
    }
    ls.addEventListener('change', draw); rd.addEventListener('input', draw); lr.addEventListener('input', taDebounce(draw, 120)); qs('[data-cmp]', ctl).addEventListener('click', compare);
    return { render: function(){ draw(); if (!taStore().cmp) compare(); } };
  } });

/* ---------- Step 5: tuning ---------- */
L7.add({ short: 'Tuning', title: 'Choosing settings without peeking at the test set',
  change: 'Press Run grid. Compare the cell chosen on validation with the cell that is best on test.',
  watch: 'A hyperparameter is a setting you choose before training, such as the learning rate or the depth. To pick one fairly, split the training compounds again. The model learns from 80% and is scored on the other 20%, the validation set. Choose the cell with the best validation score, then report its score on the test set once. If you choose by test score instead, the test set has helped pick the model, and the score you report is too good. This grid fits 100 rounds of boosting for each learning rate and depth.',
  build: function(api){
    var hp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Validation R2 for each pair of settings</h4><div class="hm"></div>');
    var sp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<h4>What the grid says</h4><div class="sd"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-grid>Run grid</button>');
    var LRS = [0.02, 0.05, 0.1, 0.3], DEPTHS = [2, 3, 4, 6], busy = false;
    function split(){ var r = rng(5), tr = [], va = []; D.TRAIN.forEach(function(i){ (r() < 0.2 ? va : tr).push(i); }); return { tr: tr, va: va }; }
    function shade(v){ var t = clamp(v / 0.5, 0, 1); return 'rgba(225,36,42,' + (0.08 + 0.6 * t).toFixed(2) + ')'; }
    function draw(){
      var G = taStore().grid, h = '<div style="display:grid;grid-template-columns:90px repeat(4,1fr);gap:6px;font-size:13px;color:' + C.char + '"><div></div>' + LRS.map(function(v){ return '<div style="text-align:center;font-weight:700">rate ' + v + '</div>'; }).join('');
      var bv = null, bt = null; if (G && G.done) { G.cells.forEach(function(c){ if (!bv || c.va > bv.va) bv = c; if (!bt || c.te > bt.te) bt = c; }); }
      DEPTHS.forEach(function(d){ h += '<div style="font-weight:700;align-self:center">depth ' + d + '</div>'; LRS.forEach(function(l){ var c = G && G.cells.filter(function(x){ return x.lr === l && x.d === d; })[0]; h += '<div style="height:70px;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:' + (c ? shade(c.va) : C.light) + ';border:' + (bv && c === bv ? '3px solid ' + C.red : (bt && c === bt ? '3px dashed ' + C.char : '1px solid ' + C.silver)) + '">' + (c ? '<b style="font-size:17px">' + c.va.toFixed(3) + '</b><span style="font-size:11px">' + (G.done ? 'test ' + c.te.toFixed(3) : '') + '</span>' : '') + '</div>'; }); });
      qs('.hm', hp).innerHTML = h + '</div><p class="note" style="font-size:12px;margin-top:10px">Cell colour and the large number are validation R2. After the run, the small number is test R2. Solid red border: best on validation. Dashed dark border: best on test.</p>';
      var sd = qs('.sd', sp);
      if (!G) sd.innerHTML = '<p class="note">Press Run grid. Sixteen models are fitted.</p>'; else if (!G.done) sd.innerHTML = '<p class="note">Fitted ' + G.cells.length + ' of 16.</p>';
      else { var same = bv === bt; sd.innerHTML = taRow('Chosen on validation', 'rate ' + bv.lr + ', depth ' + bv.d, C.red) + taRow('Its test R2', bv.te.toFixed(3), C.red) + taRow('Best test R2 in the grid', bt.te.toFixed(3)) + taRow('Overstatement if chosen on test', Math.max(0, bt.te - bv.te).toFixed(3)) + '<p style="font-size:13px;line-height:1.5;margin-top:10px;color:' + C.char + '">' + (same ? 'Both rules picked the same cell here, so nothing was overstated. That is luck, and it will not always happen.' : 'The two rules picked different cells. Reporting the best test score would overstate what the chosen model will do on new compounds.') + '</p><p class="note" style="font-size:12px">Validation compounds: ' + G.nv + '. Training compounds: ' + G.nt + '. Test compounds: ' + D.TEST.length + '.</p>'; }
    }
    async function run(){
      if (busy) return; busy = true; var sp_ = split(), G = taStore().grid = { cells: [], done: false, nv: sp_.va.length, nt: sp_.tr.length }; draw();
      for (var a = 0; a < DEPTHS.length; a++) for (var b = 0; b < LRS.length; b++) {
        api.busy('Fitting ' + (G.cells.length + 1) + ' of 16'); await tick(); await tick(); var m = fitBoost(taFeats(), sp_.tr, 100, LRS[b], DEPTHS[a], 0);
        G.cells.push({ lr: LRS[b], d: DEPTHS[a], va: r2(sp_.va, m.F), te: r2(D.TEST, m.F) }); draw();
      }
      G.done = true; busy = false; api.idle(); draw();
    }
    qs('[data-grid]', ctl).addEventListener('click', run);
    return { render: function(){ if (!taStore().grid) return run(); draw(); } };
  } });

/* ---------- Step 6: library designs ---------- */
var TA_LIBS = [
  { name: 'This site', rows: [['Growth', 'Level by level up to a depth limit.'], ['Split search', 'Histogram with 32 bins.'], ['Fits each round', 'The gradient only (first order). Leaf value is the mean of the gradient.'], ['Regularisation', 'Learning rate, depth and minimum leaf size.'], ['Meant for', 'Showing the algorithm. It is not tuned for speed or for large data.']] },
  { name: 'XGBoost', rows: [['Growth', 'Level by level by default. A leaf-wise policy is available.'], ['Split search', 'Exact, approximate or histogram, chosen by tree_method.'], ['Fits each round', 'The gradient and the curvature (second order), a Newton step. Leaf weight = -G / (H + λ).'], ['Regularisation', 'L1 and L2 penalties on leaf weights, and a minimum gain to split (γ).'], ['Known for', 'Learning a default direction for missing values, and wide use in competitions.']] },
  { name: 'LightGBM', rows: [['Growth', 'Leaf-wise: always split the leaf with the largest gain. Size is set by the number of leaves.'], ['Split search', 'Histogram.'], ['Fits each round', 'Gradient and curvature, as in XGBoost.'], ['Regularisation', 'Leaf count, minimum data per leaf, L1 and L2 penalties.'], ['Known for', 'Speed on large data, using gradient-based one-side sampling (keep large-gradient rows) and exclusive feature bundling.']] },
  { name: 'CatBoost', rows: [['Growth', 'Symmetric trees: every node at a level uses the same split.'], ['Split search', 'Histogram.'], ['Fits each round', 'Gradient with ordered boosting, which fits each compound using a model that has not seen it.'], ['Regularisation', 'Depth, learning rate and L2 penalty. Symmetric trees also limit complexity.'], ['Known for', 'Built-in handling of categorical features, and less leakage from the target.']] },
  { name: 'scikit-learn', rows: [['Growth', 'GradientBoostingRegressor grows level by level. HistGradientBoostingRegressor grows leaf-wise.'], ['Split search', 'Exact for the first, histogram for the second.'], ['Fits each round', 'The gradient, with a line search in the first class.'], ['Regularisation', 'Learning rate, depth or leaf count, minimum samples, L2 in the histogram class.'], ['Known for', 'A common API. The histogram class follows the LightGBM design.']] }
];
L7.add({ short: 'Library designs', title: 'How the common boosting libraries differ',
  change: 'Click a library to compare how it grows trees, searches splits and regularises.',
  watch: 'Gradient boosting libraries share one idea but differ in how they build each tree. Level-wise growth splits every node at one depth before going deeper. Leaf-wise growth always splits the single best leaf, which reaches a low error with fewer splits but can overfit small data. A second-order method uses the curvature of the loss as well as its slope, so each step is better scaled. Defaults change between versions, so check the documentation of the one you use.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div class="tl"></div>'), cur = 0;
    function draw(){
      var host = qs('.tl', pn), L = TA_LIBS[cur];
      if (!qs('.tlr', pn)) { host.innerHTML = '<div style="display:grid;grid-template-columns:200px 1fr;gap:22px"><div>' + TA_LIBS.map(function(x, i){ return '<button type="button" data-lib="' + i + '" style="display:block;width:100%;text-align:left;font:inherit;font-size:14px;padding:10px 12px;margin-bottom:6px;border-radius:6px;cursor:pointer;border:1px solid ' + C.silver + ';background:#fff;color:' + C.char + '">' + esc(x.name) + '</button>'; }).join('') + '</div><div class="tlr"></div></div>'; qsa('[data-lib]', pn).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.lib; draw(); }); }); }
      qsa('[data-lib]', pn).forEach(function(b){ var on = +b.dataset.lib === cur; b.style.borderColor = on ? C.red : C.silver; b.style.background = on ? C.light : '#fff'; b.style.fontWeight = on ? 700 : 500; });
      qs('.tlr', pn).innerHTML = '<h4 style="font-size:20px;color:' + C.red + ';margin-bottom:6px">' + esc(L.name) + '</h4>' + L.rows.map(function(r){ return '<div style="display:grid;grid-template-columns:150px 1fr;padding:12px 0;border-top:1px solid ' + C.silver + '"><div class="lbl" style="text-transform:none;letter-spacing:0;font-size:13px;font-weight:700;color:' + C.char + '">' + r[0] + '</div><div style="font-size:15px;line-height:1.45;color:' + C.char + '">' + esc(r[1]) + '</div></div>'; }).join('');
    }
    return { render: draw };
  } });

L7.glossary([
  ['Sum of squared errors (SSE)', 'The total of the squared gaps between each compound and its group average.'],
  ['Gain', 'How much a split reduces the SSE: SSE before minus SSE after.'],
  ['Threshold', 'The feature value that sends a compound left or right at a split.'],
  ['Bin', 'A range of feature values treated as one value when searching for splits.'],
  ['Leaf', 'A final group at the bottom of a tree. Its prediction is the group average.'],
  ['Minimum leaf size', 'The fewest compounds a leaf may hold. A split that breaks it is rejected.'],
  ['Bootstrap sample', 'Compounds drawn at random with replacement, the same size as the training set.'],
  ['Out-of-bag', 'Compounds a given tree did not see in its bootstrap sample.'],
  ['Gradient', 'The direction in which a prediction should move to lower the loss fastest.'],
  ['Loss', 'A number that measures how bad a prediction is. Training lowers it.'],
  ['Huber loss', 'Squared error for small residuals and absolute error for large ones.'],
  ['Hyperparameter', 'A setting chosen before training, such as the learning rate or depth.'],
  ['Validation set', 'Training compounds held back to compare settings. It is not the test set.'],
  ['Level-wise growth', 'Split every node at one depth before going deeper.'],
  ['Leaf-wise growth', 'Always split the leaf with the largest gain.']
]);
setUses(L7, [['endpoint'], ['endpoint'], ['endpoint'], ['endpoint'], ['endpoint'], []]);
