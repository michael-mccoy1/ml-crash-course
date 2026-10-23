"use strict";
/* Lesson 2: Tree models */
var L2 = new Lesson('trees', 'Tree models');
L2.optsHTML = function(){
  return sharedOptions() +
    optRange('tdepth', 'Single tree depth', 1, 14, 1, 4, 'Questions in a row for the single tree.') +
    optRange('tntrees', 'Forest: number of trees', 10, 200, 10, 60, 'Trees that vote in the forest.') +
    optRange('tfdepth', 'Forest tree depth', 3, 16, 1, 10, 'Depth of each tree in the forest.') +
    optRange('trounds', 'Boosting rounds', 20, 400, 10, 150, 'Small trees added one after another.') +
    optRange('tlr', 'Boosting learning rate', 0.01, 0.3, 0.01, 0.08, 'Share of each tree\'s correction that is applied.') +
    '<div class="og"><button type="button" class="btn ghost wide" id="ttrain">Train all three for Compare</button><div class="status" id="tst"></div><div class="rbox" id="tres"></div></div>' + filterGroup();
};
function TP(){ return { depth: +$('tdepth').value, nt: +$('tntrees').value, fd: +$('tfdepth').value, rounds: +$('trounds').value, lr: +$('tlr').value }; }
function tKey(name, extra){ return name + JSON.stringify([EP, activeFeats(), activeTrain().length, filtKey(), extra]); }
function tCache(name, extra, fn){ D.cache.t = D.cache.t || {}; var k = tKey(name, extra); if (!D.cache.t[k]) D.cache.t[k] = fn(); return D.cache.t[k]; }
function tPeek(name, extra){ return D.cache.t ? D.cache.t[tKey(name, extra)] : null; }
function tTree(){ var p = TP(); return tCache('tree', p.depth, function(){ var f = activeFeats(), idx = activeTrain(), t0 = performance.now(), tree = fitTree(f, idx, p.depth, 5, 1), secs = (performance.now() - t0) / 1000, pred = treeAll(tree, D.N);
  var sweep = tCache('sweep', 0, function(){ var out = []; for (var d = 1; d <= 14; d++) { var pr = treeAll(fitTree(f, idx, d, 5, 1), D.N); out.push([d, r2(D.TRAIN, pr), r2(D.TEST, pr)]); } return out; });
  return { tree: tree, pred: pred, secs: secs, show: fitTree(f, idx, Math.min(3, p.depth), 20, 1), sweep: sweep }; }); }
function tForest(){ var p = TP(); return tCache('forest', [p.nt, p.fd], function(){ var fr = fitForest(activeFeats(), activeTrain(), p.nt, p.fd, 7); fr.curve = forestCurve(fr, Math.max(1, Math.round(p.nt / 40))); return fr; }); }
function tForestDepth(){ return tCache('fdepth', 0, function(){ return [3, 6, 10, 14].map(function(d){ var fr = fitForest(activeFeats(), activeTrain(), 30, d, 7); return { d: d, te: r2(D.TEST, fr.pred), tr: r2(D.TRAIN, fr.pred) }; }); }); }
function tBoost(){ var p = TP(); return tCache('boost', [p.rounds, p.lr], function(){ return fitBoost(activeFeats(), activeTrain(), p.rounds, p.lr, 3, Math.max(1, Math.round(p.rounds / 40))); }); }
function tBoostLr(){ var p = TP(); return tCache('boostlr', p.rounds, function(){ return [0.02, 0.08, 0.3].map(function(lr){ return { lr: lr, curve: fitBoost(activeFeats(), activeTrain(), p.rounds, lr, 3, Math.max(1, Math.round(p.rounds / 40))).curve }; }); }); }
function tAblation(){ return tCache('abl', 0, function(){ var f = activeFeats(), idx = activeTrain(), full = fitBoost(f, idx, 60, 0.1, 3), base = r2(D.TEST, full.F); return { base: base, rows: f.map(function(x){ if (f.length < 2) return { f: x, d: 0 }; var m = fitBoost(f.filter(function(y){ return y !== x; }), idx, 60, 0.1, 3); return { f: x, d: r2(D.TEST, m.F) - base }; }) }; }); }
var TNAMES = { tree: 'Decision tree', forest: 'Random forest', boost: 'Gradient boosting' };
function recordMode(mode){
  var p = TP(), f = activeFeats(); if (!f.length) return null;
  var a = mode === 'tree' ? tTree() : (mode === 'forest' ? tForest() : tBoost()), ms = record(TNAMES[mode], f, mode === 'tree' ? a.pred : (mode === 'forest' ? a.pred : a.F), a.secs);
  ms.params = mode === 'tree' ? 'depth ' + p.depth : (mode === 'forest' ? p.nt + ' trees, depth ' + p.fd : p.rounds + ' rounds, rate ' + p.lr); return ms;
}
function recordTrees(){ return ['tree', 'forest', 'boost'].map(recordMode); }
function tResultsBox(){ var el = $('tres'); if (!el) return; var ms = ['Decision tree', 'Random forest', 'Gradient boosting'].map(function(n){ return D.scores[n]; }).filter(Boolean); el.innerHTML = ms.map(function(m, i){ return '<div class="rrow">' + ['Tree', 'Forest', 'Boosting'][i] + '<b>R2 ' + m.r2.toFixed(2) + ', ' + fmtT(m.secs) + '</b></div>'; }).join(''); }
var tTimer = null;
L2.onOpts = function(root){
  var fire = function(){ clearTimeout(tTimer); tTimer = setTimeout(function(){ L2.refresh(); }, 220); };
  ['tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr'].forEach(function(id){ $(id).addEventListener('input', function(){ if (L2.live) L2.live(id); fire(); }); });
  $('ttrain').addEventListener('click', async function(){ var b = this; b.disabled = true; var st = $('tst'); if (!activeFeats().length) { st.textContent = 'Tick at least one feature.'; b.disabled = false; return; } st.textContent = 'Training'; progress(0.2); await tick(); recordTrees(); progress(0.7); await tick(); tResultsBox(); st.textContent = 'Done. Scores are in Compare.'; progress(1); b.disabled = false; L2.refresh(); });
};
L2.onData = function(){ tResultsBox(); };
featListeners.push(function(){ if (L2.built) L2.refresh(); });

/* ---------- tree drawing ---------- */
function thrText(B, f, b){ return +B.edges[f][b].toFixed(2); }
function layoutTree(root, maxD, W, H){ var nodes = [], leaves = 0; (function cnt(n, d){ if (n.f === undefined || d === maxD) { leaves++; return; } cnt(n.l, d + 1); cnt(n.r, d + 1); })(root, 0);
  var dx = (W - 20) / leaves, dy = (H - 56) / Math.max(1, maxD), li = 0; (function place(n, d, parent){ var o = { n: n, d: d, parent: parent }; nodes.push(o); if (n.f === undefined || d === maxD) { o.x = 10 + dx * (li++ + 0.5); o.leaf = true; } else { o.l = place(n.l, d + 1, o); o.r = place(n.r, d + 1, o); o.x = (o.l.x + o.r.x) / 2; } o.y = 28 + d * dy; return o; })(root, 0, null); return nodes; }
function treeSVG(root, B, maxD, W, H){
  var nodes = layoutTree(root, maxD, W, H), nleaf = nodes.filter(function(o){ return o.leaf; }).length, bw = Math.min(110, (W - 20) / nleaf - 8), svg = '', total = root.n, lo = D.ylo, hi = D.yhi;
  nodes.forEach(function(o){ if (o.parent) svg += '<line class="tedge" style="animation-delay:' + (ANIM ? o.d * 320 : 0) + 'ms" x1="' + o.parent.x.toFixed(1) + '" y1="' + (o.parent.y + 19) + '" x2="' + o.x.toFixed(1) + '" y2="' + (o.y - 19) + '" stroke="' + C.bg + '" stroke-width="' + (1.5 + 8 * Math.sqrt(o.n.n / total)).toFixed(1) + '" stroke-linecap="round"/>'; });
  nodes.forEach(function(o, k){ var leaf = o.leaf, fill = leaf ? colorMix((o.n.v - lo) / (hi - lo)) : '#fff', dark = leaf && (o.n.v - lo) / (hi - lo) > 0.62, tc = dark ? '#fff' : C.char, l1, l2;
    if (o.n.f === undefined) { l1 = 'mean ' + o.n.v.toFixed(2); l2 = 'n = ' + o.n.n; } else if (leaf) { l1 = 'mean ' + o.n.v.toFixed(2); l2 = 'n = ' + o.n.n + ', splits more'; } else { l1 = o.n.f + ' <= ' + thrText(B, o.n.f, o.n.b); l2 = 'n = ' + o.n.n; }
    svg += '<g class="tnode" data-k="' + k + '" style="animation-delay:' + (ANIM ? o.d * 320 + 100 : 0) + 'ms"><rect x="' + (o.x - bw / 2).toFixed(1) + '" y="' + (o.y - 19) + '" width="' + bw.toFixed(1) + '" height="38" rx="7" fill="' + fill + '" stroke="' + (leaf ? C.bg : C.char) + '" stroke-width="1.5"/><text x="' + o.x.toFixed(1) + '" y="' + (o.y - 3) + '" text-anchor="middle" font-size="' + (bw < 80 ? 10.5 : 12) + '" font-weight="700" fill="' + tc + '">' + esc(l1) + '</text><text x="' + o.x.toFixed(1) + '" y="' + (o.y + 11) + '" text-anchor="middle" font-size="' + (bw < 80 ? 9.5 : 10.5) + '" fill="' + tc + '" opacity=".85">' + esc(l2) + '</text></g>'; });
  return { svg: svg + '<circle class="ball" r="8" fill="' + C.red + '" stroke="#fff" stroke-width="3" cx="-50" cy="-50"/>', nodes: nodes };
}
function featVals(i){ var v = {}; D.FEATS.forEach(function(f){ v[f] = D.COL[f][i]; }); return v; }
async function pickStruct(){ await loadStructs(); var a = STRUCTS[EP]; return a[Math.floor(Math.random() * a.length)]; }

/* ---------- workbench: pick a model, drag its options, press Train ---------- */
var TB = { mode: 'tree', last: {} };
function glyphTree(x0, y0, w, h, depth, seed, lit){
  var r = rng(seed * 977), s = '', col = lit ? C.red : C.bg, dy = h / (depth + 0.4);
  (function rec(x, y, span, d){
    if (d >= depth) { s += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (depth > 4 ? 1.3 : 2) + '" fill="' + col + '"/>'; return; }
    var lx = x - span / 2 * (0.6 + 0.8 * r()), rx = x + span / 2 * (0.6 + 0.8 * r()), ny = y + dy * (0.8 + 0.4 * r());
    s += '<path d="M' + x.toFixed(1) + ' ' + y.toFixed(1) + 'L' + lx.toFixed(1) + ' ' + ny.toFixed(1) + 'M' + x.toFixed(1) + ' ' + y.toFixed(1) + 'L' + rx.toFixed(1) + ' ' + ny.toFixed(1) + '" stroke="' + col + '" stroke-width="1.1" fill="none"/>';
    rec(lx, ny, span / 2, d + 1); rec(rx, ny, span / 2, d + 1);
  })(x0 + w / 2, y0 + 3, w * 0.9, 0);
  return s;
}
L2.add({ short: 'Workbench', bench: true, uses: ['endpoint', 'features', 'filters', 'tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr'],
  title: 'Set up a tree model, press Train, and watch it fit',
  watch: 'The picture on the left follows your options as you drag them. Train fits the model on the training compounds, then replays the fit slowly: the model builds on the left while the score curve and the predictions on the right update.',
  change: 'Choose a model type, drag its options, then press Train. Slow shows every step, Fast jumps to the end.',
  build: function(api){
    var bb = benchBar(api, { leftHTML: '<div class="seg mode" data-mode>' + [['tree', 'Decision tree'], ['forest', 'Random forest'], ['boost', 'Gradient boosting']].map(function(it){ return '<button type="button" data-v="' + it[0] + '"' + (it[0] === TB.mode ? ' class="on"' : '') + '>' + it[1] + '</button>'; }).join('') + '</div>' });
    var mp = api.el('pn', 'left:28px;top:80px;width:456px;height:372px', '<h4 class="mh"></h4><svg class="static" viewBox="0 0 432 318" width="432" height="318"></svg>');
    var cp = api.el('pn', 'left:504px;top:80px;width:428px;height:178px', '<h4 class="ch"></h4><canvas width="404" height="140"></canvas>');
    var sp = api.el('pn', 'left:504px;top:270px;width:428px;height:182px', '<h4>Predictions on unseen compounds</h4><canvas width="404" height="144"></canvas>');
    var np = api.el('pn', 'left:28px;top:462px;width:904px;height:56px;display:flex;align-items:center;font-size:14px;line-height:1.4', '<div class="mn"></div>');
    var svg = qs('svg', mp), running = false;
    function dimOpts(){ var map = { tree: ['tdepth'], forest: ['tntrees', 'tfdepth'], boost: ['trounds', 'tlr'] }; ['tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr'].forEach(function(id){ var og = $(id).closest('.og'); og.classList.toggle('dim', map[TB.mode].indexOf(id) < 0); }); }
    function msg(t){ svg.innerHTML = '<text x="20" y="40" font-size="13" fill="' + C.red + '">' + t + '</text>'; }
    function pictureTree(dShow){
      var p = TP(), f = activeFeats(), d = dShow || p.depth, dd = Math.min(d, 2);
      var t = tCache('livetree', dd, function(){ return fitTree(f, activeTrain(), dd, 20, 1); }), r = treeSVG(t, D.B, dd, 432, 192), s = r.svg.replace(/<circle class="ball"[^>]*\/>/, ''), leaves = Math.pow(2, d), n = Math.min(leaves, 512), cols = 64, sz = 432 / cols, strip = '';
      for (var i = 0; i < n; i++) strip += '<rect x="' + ((i % cols) * sz + 1).toFixed(1) + '" y="' + (246 + Math.floor(i / cols) * sz).toFixed(1) + '" width="' + (sz - 2).toFixed(1) + '" height="' + (sz - 2).toFixed(1) + '" fill="' + C.red + '" opacity=".85"/>';
      svg.innerHTML = s + '<text x="0" y="232" font-size="12.5" font-weight="700" fill="' + C.char + '">Depth ' + d + ' allows up to ' + leaves.toLocaleString() + ' groups' + (d > 2 ? ' (first 2 levels drawn)' : '') + '</text>' + strip;
    }
    function pictureForest(lit){
      var p = TP(), shown = Math.min(p.nt, 40), cols = Math.ceil(Math.sqrt(shown * 432 / 290)), rows = Math.ceil(shown / cols), gw = 432 / cols, gh = Math.min(290 / rows, gw * 1.15), dv = Math.min(p.fd, 5), s = '';
      for (var k = 0; k < shown; k++) s += glyphTree((k % cols) * gw + 2, Math.floor(k / cols) * gh + 2, gw - 4, gh - 6, dv, k + 1, k < lit);
      s += '<text x="0" y="312" font-size="12.5" font-weight="700" fill="' + C.char + '">' + p.nt + ' trees, each up to depth ' + p.fd + (p.nt > shown ? ' (' + shown + ' drawn)' : '') + '</text>'; svg.innerHTML = s;
    }
    function pictureBoost(lit){
      var p = TP(), W = 432, x0 = 36, x1 = W - 10, yT = 22, yB = 220, s = '', pts = [], K = p.rounds;
      for (var k = 0; k <= K; k += Math.max(1, Math.round(K / 120))) pts.push([x0 + k / K * (x1 - x0), yT + (1 - Math.pow(1 - p.lr, k)) * (yB - yT)]);
      pts.push([x1, yT + (1 - Math.pow(1 - p.lr, K)) * (yB - yT)]);
      s += '<line x1="' + x0 + '" y1="' + yB + '" x2="' + x1 + '" y2="' + yB + '" stroke="' + C.silver + '"/><line x1="' + x0 + '" y1="' + yT + '" x2="' + x0 + '" y2="' + yB + '" stroke="' + C.silver + '"/><text x="' + (x0 - 6) + '" y="' + (yT + 4) + '" font-size="11" text-anchor="end" fill="' + C.mid + '">100%</text><text x="' + (x0 - 6) + '" y="' + (yB + 4) + '" font-size="11" text-anchor="end" fill="' + C.mid + '">0%</text>';
      s += '<path d="M' + pts.map(function(q){ return q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join('L') + '" fill="none" stroke="' + C.red + '" stroke-width="2.6"/>';
      s += '<text x="' + x0 + '" y="14" font-size="12" font-weight="700" fill="' + C.char + '">Error that could remain (share of the start)</text>';
      var left = Math.pow(1 - p.lr, K) * 100; s += '<text x="' + x1 + '" y="' + (yB - 8) + '" font-size="12" text-anchor="end" fill="' + C.dred + '">' + (left < 0.5 ? 'under 1%' : left.toFixed(0) + '%') + ' left after ' + K + ' rounds</text>';
      var nb = Math.min(K, 80), bw = (x1 - x0) / nb; for (var i = 0; i < nb; i++) s += '<rect x="' + (x0 + i * bw + 0.8).toFixed(1) + '" y="262" width="' + (bw - 1.6).toFixed(1) + '" height="34" rx="1.5" fill="' + (i / nb * K < lit ? C.red : C.bg) + '"/>';
      s += '<text x="' + x0 + '" y="316" font-size="12.5" font-weight="700" fill="' + C.char + '">' + K + ' small trees, one per round' + (K > nb ? ' (' + nb + ' drawn)' : '') + '</text>'; svg.innerHTML = s;
    }
    function picture(lit, dShow){ if (!activeFeats().length) { msg('Switch on at least one feature.'); return; } if (TB.mode === 'tree') pictureTree(dShow); else if (TB.mode === 'forest') pictureForest(lit === undefined ? 0 : lit); else pictureBoost(lit === undefined ? 0 : lit); }
    function peek(){ var p = TP(); if (!activeFeats().length) return null; if (TB.mode === 'tree') { var a = tPeek('tree', p.depth); return a ? { pred: a.pred, secs: a.secs, sweep: a.sweep } : null; } if (TB.mode === 'forest') { var b = tPeek('forest', [p.nt, p.fd]); return b ? { pred: b.pred, secs: b.secs, curve: b.curve, per: b.per } : null; } var c = tPeek('boost', [p.rounds, p.lr]); return c ? { pred: c.F, secs: c.secs, curve: c.curve, snaps: c.snaps } : null; }
    function curveOpts(res){ var p = TP(), vals = [], x1; if (TB.mode === 'tree') { res && res.sweep.forEach(function(r){ vals.push(r[1], r[2]); }); x1 = 14; } else { res && res.curve.forEach(function(c){ vals.push(c.te, c.tr); }); x1 = TB.mode === 'forest' ? p.nt : p.rounds; } return { x0: 1, x1: Math.max(2, x1), y0: Math.min(0, Math.min.apply(null, vals.length ? vals : [0])), y1: 1, xl: TB.mode === 'tree' ? 'tree depth' : (TB.mode === 'forest' ? 'trees in the forest' : 'boosting rounds'), yl: 'R2', xdec: 0, ydec: 2 }; }
    function drawCurve(res, upto){
      var o = curveOpts(res), canvas = qs('canvas', cp); qs('.ch', cp).textContent = 'R2 as the model grows (dark: training, red: unseen)';
      if (!res) { lineChart(canvas, [{ pts: [[o.x0, 0], [o.x1, 1]], color: 'rgba(0,0,0,0)', nodot: true }], o); return; }
      var tr, te; if (TB.mode === 'tree') { var sw = res.sweep.filter(function(r){ return r[0] <= upto; }); tr = sw.map(function(r){ return [r[0], r[1]]; }); te = sw.map(function(r){ return [r[0], r[2]]; }); } else { var cu = res.curve.filter(function(c){ return c.k <= upto; }); tr = cu.map(function(c){ return [c.k, c.tr]; }); te = cu.map(function(c){ return [c.k, c.te]; }); }
      lineChart(canvas, [{ pts: tr, color: C.char, nodot: true }, { pts: te, color: C.red }], o);
    }
    function drawScatter(pred, label){ var canvas = qs('canvas', sp); if (!pred) { scatterPV(canvas, new Float32Array(D.N), [], null); return; } scatterPV(canvas, pred, D.TEST, label, { max: 320 }); }
    function note(){
      var p = TP(), t = TB.mode === 'tree' ? 'Depth ' + p.depth + ': the tree asks up to ' + p.depth + ' questions in a row, so it can sort compounds into up to ' + Math.pow(2, p.depth).toLocaleString() + ' groups. Each group predicts the mean of the compounds in it.'
        : (TB.mode === 'forest' ? p.nt + ' trees, each up to depth ' + p.fd + '. Each tree is fit on a random resample of the compounds and looks at a random subset of features at every question. The forest predicts the average of its trees.'
        : p.rounds + ' small trees of depth 3, added one after another. Each tree fits the errors the earlier ones left, and only ' + Math.round(p.lr * 100) + '% of its correction is applied.'); qs('.mn', np).textContent = t;
    }
    function render(){
      dimOpts(); qs('.mh', mp).textContent = TB.mode === 'tree' ? 'The tree you are building' : (TB.mode === 'forest' ? 'The forest you are building' : 'The boosted ensemble you are building'); var res = peek(), p = TP();
      picture(res ? 1e9 : 0); drawCurve(res, 1e9); drawScatter(res ? res.pred : null, res ? 'unseen R2 ' + fix(r2(D.TEST, res.pred), 2) : null); note();
      if (res) { api.stale(''); bb.say('Unseen R2 ' + fix(r2(D.TEST, res.pred), 2) + '. The fit took ' + fmtT(res.secs) + '.'); } else { api.stale(TB.last[TB.mode] ? 'Options changed. Press Train.' : ''); bb.say(TB.last[TB.mode] ? '' : 'Ready. Press Train.'); }
    }
    function fit(mode){ if (mode === 'tree') { var a = tTree(); return { pred: a.pred, secs: a.secs, sweep: a.sweep }; } if (mode === 'forest') { var b = tForest(); return { pred: b.pred, secs: b.secs, curve: b.curve, per: b.per }; } var c = tBoost(); return { pred: c.F, secs: c.secs, curve: c.curve, snaps: c.snaps }; }
    async function replay(mode, res, tok){
      var p = TP(), total = 9000 * speedMul(), f = activeFeats();
      if (mode === 'tree') { var step = total / p.depth; for (var d = 1; d <= p.depth; d++) { api.checkTok(tok); var pr = tCache('treeD', d, function(){ return treeAll(fitTree(f, activeTrain(), d, 5, 1), D.N); }); pictureTree(d); drawCurve(res, d); drawScatter(pr, 'depth ' + d + ', R2 ' + fix(r2(D.TEST, pr), 2)); bb.say('Depth ' + d + ' of ' + p.depth + '. Unseen R2 ' + fix(r2(D.TEST, pr), 2)); await api.sleep(step); } }
      else if (mode === 'forest') { var n = p.nt, cps = Math.min(n, 40), acc = new Float64Array(D.N), pred = new Float64Array(D.N), done = 0, shown = Math.min(n, 40); for (var c = 1; c <= cps; c++) { api.checkTok(tok); var k = Math.round(c * n / cps); while (done < k) { var pt = res.per[done]; for (var i = 0; i < D.N; i++) acc[i] += pt[i]; done++; } for (var j = 0; j < D.N; j++) pred[j] = acc[j] / k; pictureForest(Math.ceil(k / n * shown)); drawCurve(res, k); drawScatter(pred, k + ' trees, R2 ' + fix(r2(D.TEST, pred), 2)); bb.say('Tree ' + k + ' of ' + n + '. Unseen R2 ' + fix(r2(D.TEST, pred), 2)); await api.sleep(total / cps); } }
      else { var sn = res.snaps; for (var q = 0; q < sn.length; q++) { api.checkTok(tok); pictureBoost(sn[q].r); drawCurve(res, sn[q].r); drawScatter(sn[q].p, 'round ' + sn[q].r + ', R2 ' + fix(r2(D.TEST, sn[q].p), 2)); bb.say('Round ' + sn[q].r + ' of ' + p.rounds + '. Unseen R2 ' + fix(r2(D.TEST, sn[q].p), 2)); await api.sleep(total / sn.length); } }
    }
    async function train(){
      if (running) return; if (!activeFeats().length) { bb.say('Switch on at least one feature.'); return; }
      running = true; bb.btn.disabled = true; api.tok++; var tok = api.tok, mode = TB.mode; api.stale('');
      try { api.busy('Fitting the model'); bb.say('Fitting'); await delay(50); var res = fit(mode); api.idle(); await replay(mode, res, tok); TB.last[mode] = true; recordMode(mode); tResultsBox(); }
      catch (e) { if (!(e instanceof Cancel)) console.error(e); }
      api.idle(); running = false; bb.btn.disabled = false; render();
    }
    bb.btn.addEventListener('click', train);
    qsa('[data-mode] button', bb.bar).forEach(function(b){ b.addEventListener('click', function(){ if (b.dataset.v === TB.mode) return; TB.mode = b.dataset.v; qsa('[data-mode] button', bb.bar).forEach(function(x){ x.classList.toggle('on', x === b); }); api.tok++; render(); }); });
    L2.live = function(id){
      var m = { tdepth: 'tree', tntrees: 'forest', tfdepth: 'forest', trounds: 'boost', tlr: 'boost' }[id];
      if (m && m !== TB.mode) { TB.mode = m; qsa('[data-mode] button', bb.bar).forEach(function(x){ x.classList.toggle('on', x.dataset.v === m); }); }
      if (!running) render(); else { api.tok++; }
    };
    return { render: render };
  } });

/* ---------- T1: how a tree chooses a question ---------- */
L2.add({ short: 'Choosing a question', title: 'How a tree picks its first question',
  change: 'Choose a feature under the plot, or drag Single tree depth to refine the red staircase. Walk through plays the scan.',
  watch: 'Every position along the feature is scored by how well it separates low values from high values. The best position becomes the question at the top of the tree.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:580px;height:408px', '<h4 class="t">Training compounds</h4><canvas width="560" height="352"></canvas>');
    var gp = api.el('pn', 'left:628px;top:28px;width:304px;height:196px', '<h4>How good is each cut</h4><canvas width="284" height="146"></canvas>');
    var res = api.el('pn', 'left:628px;top:236px;width:304px;height:200px', '<div class="rs" style="font-size:13px;color:#4a4f5a">Choose a feature, then press Walk through.</div>');
    var ctl = api.addControl('Feature <select></select>'); var sel = qs('select', ctl), S = null, pts = [];
    function fillSel(){ var f = activeFeats(), cur = sel.value; sel.innerHTML = f.map(function(x){ return '<option>' + x + '</option>'; }).join(''); if (f.indexOf(cur) >= 0) sel.value = cur; }
    function compute(){ var feat = sel.value, idx = activeTrain().slice().sort(function(a, b){ return D.COL[feat][a] - D.COL[feat][b]; }), n = idx.length, ys = idx.map(function(i){ return D.Y[i]; }), pre = new Float64Array(n + 1); for (var i = 0; i < n; i++) pre[i + 1] = pre[i] + ys[i];
      var S_ = pre[n], base = S_ * S_ / n, cands = []; for (var q = 0; q < 60; q++) { var k = Math.round(n * (0.04 + 0.92 * q / 59)); if (k < 5 || n - k < 5) continue; var sl = pre[k], sr = S_ - sl, thr = D.COL[feat][idx[k]]; cands.push({ k: k, t: thr, gain: sl * sl / k + sr * sr / (n - k) - base, mL: sl / k, mR: sr / (n - k), nL: k, nR: n - k }); }
      var best = cands.reduce(function(a, b){ return b.gain > a.gain ? b : a; }, cands[0]); var sst = ys.reduce(function(a, v){ return a + (v - S_ / n) * (v - S_ / n); }, 0), r = rng(5); pts = idx.filter(function(){ return r() < Math.min(1, 500 / n); });
      var xs = D.COL[feat], lo = xs[idx[Math.floor(n * 0.01)]], hi = xs[idx[Math.floor(n * 0.99)]]; return { feat: feat, cands: cands, best: best, sst: sst, lo: lo, hi: hi, n: n }; }
    function stair(feat, lo, hi){ var d = TP().depth, key = feat + '|' + d + '|' + activeTrain().length; D.cache.t1 = D.cache.t1 || {}; if (D.cache.t1[key]) return D.cache.t1[key]; var B = fineBins(feat, 150), tree = buildTree(B, [feat], D.Y, Int32Array.from(activeTrain()), d, 5, 1, rng(1)), curve = [], vals = {}; for (var k = 0; k < 160; k++) { vals[feat] = lerp(lo, hi, k / 159); curve.push(predictVals(tree, B, vals)); } var leaves = 0; (function cnt(n){ if (n.f === undefined) leaves++; else { cnt(n.l); cnt(n.r); } })(tree); return (D.cache.t1[key] = { curve: curve, leaves: leaves, depth: d }); }
    function draw(upto, final){
      var g = cv(qs('canvas', plot)), x = g.x, box = { l: 48, r: g.w - 12, t: 10, b: g.h - 40 }, f = S.feat, lo = S.lo, hi = S.hi; axes(g, box, [lo, hi], [D.ylo, D.yhi], f, unitWord(), 5);
      var sx = function(v){ return box.l + (clamp(v, lo, hi) - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, D.ylo, D.yhi) - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      var c = upto >= 0 ? S.cands[Math.min(upto, S.cands.length - 1)] : null; if (final) c = S.manual || S.best;
      pts.forEach(function(i){ var left = c && D.COL[f][i] <= c.t; x.fillStyle = c ? (left ? 'rgba(161,177,195,.85)' : 'rgba(225,36,42,.6)') : 'rgba(161,177,195,.7)'; x.beginPath(); x.arc(sx(D.COL[f][i]), sy(D.Y[i]), 3, 0, 6.3); x.fill(); });
      if (c) { var tx = sx(c.t); x.strokeStyle = final ? C.red : C.char; x.lineWidth = final ? 3 : 1.6; x.beginPath(); x.moveTo(tx, box.t); x.lineTo(tx, box.b); x.stroke(); x.strokeStyle = C.char; x.lineWidth = 3; x.beginPath(); x.moveTo(box.l, sy(c.mL)); x.lineTo(tx, sy(c.mL)); x.stroke(); x.beginPath(); x.moveTo(tx, sy(c.mR)); x.lineTo(box.r, sy(c.mR)); x.stroke(); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('left average ' + c.mL.toFixed(2), box.l + 6, sy(c.mL) - 6); x.textAlign = 'right'; x.fillText('right average ' + c.mR.toFixed(2), box.r - 6, sy(c.mR) - 6); }
      if (final) { var st = stair(f, lo, hi); x.strokeStyle = C.red; x.lineWidth = 3.2; x.lineJoin = 'round'; x.beginPath(); st.curve.forEach(function(v, k){ var X = box.l + k / 159 * (box.r - box.l); k ? x.lineTo(X, sy(v)) : x.moveTo(X, sy(v)); }); x.stroke(); x.fillStyle = C.red; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('red: tree of depth ' + st.depth + ' on this feature, ' + st.leaves + ' groups', box.l + 6, box.t + 12); }
      var g2 = cv(qs('canvas', gp)), x2 = g2.x, b2 = { l: 38, r: g2.w - 8, t: 8, b: g2.h - 26 }, mx = Math.max.apply(null, S.cands.map(function(k){ return k.gain; })), mn = Math.min.apply(null, S.cands.map(function(k){ return k.gain; }));
      x2.strokeStyle = C.silver; x2.beginPath(); x2.moveTo(b2.l, b2.b); x2.lineTo(b2.r, b2.b); x2.stroke(); x2.fillStyle = C.mid; x2.font = '11px Aptos, Segoe UI, sans-serif'; x2.textAlign = 'center'; x2.fillText('cut position along ' + f, (b2.l + b2.r) / 2, g2.h - 6); x2.save(); x2.translate(11, (b2.t + b2.b) / 2); x2.rotate(-Math.PI / 2); x2.fillText('score', 0, 0); x2.restore();
      var gx = function(k){ return b2.l + (S.cands[k].t - lo) / (hi - lo) * (b2.r - b2.l); }, gy = function(v){ return b2.b - (v - mn) / (mx - mn || 1) * (b2.b - b2.t); }; var upc = final ? S.cands.length - 1 : upto; x2.strokeStyle = C.red; x2.lineWidth = 2.4; x2.beginPath(); for (var k = 0; k <= Math.min(upc, S.cands.length - 1); k++) { var X = clamp(gx(k), b2.l, b2.r), Y = gy(S.cands[k].gain); k ? x2.lineTo(X, Y) : x2.moveTo(X, Y); } x2.stroke(); if (final) { x2.fillStyle = C.red; x2.beginPath(); x2.arc(clamp(gx(S.cands.indexOf(S.best)), b2.l, b2.r), gy(S.best.gain), 5, 0, 6.3); x2.fill(); } }
    function show(){ var b = S.manual || S.best; qs('.rs', res).innerHTML = '<div class="lbl">' + (S.manual ? 'Your cut' : 'First question') + '</div><div class="big" style="font-size:22px;color:' + C.red + '">' + esc(S.feat) + ' &lt;= ' + (+b.t.toFixed(2)) + '?</div><div class="nums" style="font-size:12.5px;margin-top:8px"><div><span>Yes group</span><b>' + b.nL + ' compounds, mean ' + b.mL.toFixed(2) + '</b></div><div><span>No group</span><b>' + b.nR + ' compounds, mean ' + b.mR.toFixed(2) + '</b></div><div><span>Spread explained</span><b>' + Math.round(b.gain / S.sst * 100) + '%</b></div><div><span>Depth in options</span><b>' + TP().depth + '</b></div>' + (S.manual ? '<div><span>Best cut explains</span><b style="color:' + C.red + '">' + Math.round(S.best.gain / S.sst * 100) + '%</b></div>' : '') + '</div><p style="font-size:12px;color:#6B7280;margin-top:8px">Drag across the plot to try your own cut. The first cut makes two groups. Repeating the search inside each group, down to the depth in the options, gives the red staircase.</p>'; }
    sel.addEventListener('change', function(){ api.tok++; api.clearTexts(); S = compute(); draw(0, true); show(); });
    (function(){ var cvEl = qs('canvas', plot), down = false; function pick(e){ if (!S) return; var r = cvEl.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * 560, v = S.lo + (px - 48) / (560 - 12 - 48) * (S.hi - S.lo), best = S.cands[0]; S.cands.forEach(function(k){ if (Math.abs(k.t - v) < Math.abs(best.t - v)) best = k; }); S.manual = best; api.tok++; api.clearTexts(); draw(0, true); show(); }
      cvEl.style.cursor = 'ew-resize'; cvEl.addEventListener('pointerdown', function(e){ down = true; cvEl.setPointerCapture(e.pointerId); pick(e); }); cvEl.addEventListener('pointermove', function(e){ if (down) pick(e); }); cvEl.addEventListener('pointerup', function(){ down = false; }); })();
    function render(){ fillSel(); if (!sel.value) return; var man = S && S.manual && S.feat === sel.value; S = compute(); draw(0, true); show(); }
    async function play(tok){ fillSel(); if (!sel.value) return; S = compute(); qs('.rs', res).innerHTML = '<div class="note">Sweeping</div>'; draw(-1, false); await api.beat('Every dot is a training compound. The question is a cut along one feature: is it at most this value?', 3200);
      api.cap('A cut makes two groups. Each group is predicted by its own average, the dark lines.'); var n = S.cands.length; await api.anim(ANIM ? 4800 : 1, function(t){ draw(Math.floor(t * (n - 1)), false); });
      draw(n - 1, true); show(); api.cap('The cut with the best score wins. This becomes the first question of the tree.'); await api.sleep(3400); }
    return { render: render, play: play };
  } });

/* ---------- T3: send a compound down the tree ---------- */
L2.add({ short: 'One compound', title: 'Follow one compound down the tree',
  change: 'Press Next compound for a different molecule. Walk through sends it down the tree.',
  watch: 'At each box the compound answers yes or no using its own feature value and ends in a leaf. The leaf\'s average is the prediction.',
  build: function(api){
    var cur = null;
    var sp = api.el('pn', 'left:28px;top:28px;width:236px;height:196px', '<h4>Compound</h4><div class="struct" style="height:156px"></div>');
    var fp = api.el('pn', 'left:28px;top:234px;width:236px;height:202px', '<div class="nums fv" style="font-size:11.5px"></div>');
    var tp = api.el('pn', 'left:284px;top:28px;width:648px;height:330px', '<svg viewBox="0 0 628 300" width="628" height="300"></svg>');
    var rp = api.el('pn', 'left:284px;top:370px;width:648px;height:66px', '<div class="rr"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var nodes = null, tree = null;
    async function setup(next){ await loadStructs(); if (!cur || next || cur._ep !== EP) { cur = await pickStruct(); cur._ep = EP; } var t = tTree(), p = TP(), md = Math.min(3, p.depth); tree = t.show; var out = treeSVG(t.show, D.B, md, 628, 300); nodes = out.nodes; qs('svg', tp).innerHTML = out.svg;
      qs('.struct', sp).innerHTML = cur.svg; qs('.fv', fp).innerHTML = D.BASE.map(function(f){ return '<div><span>' + f + '</span><b>' + fmtFeat(f, D.COL[f][cur.i]) + '</b></div>'; }).join('') + (D.COL.logD_pred && FSEL.logD_pred ? '<div><span>logD_pred</span><b>' + D.COL.logD_pred[cur.i].toFixed(2) + '</b></div>' : ''); qs('.rr', rp).innerHTML = '<span class="note">Press Play to send this compound down the tree.</span>'; qsa('.tnode', tp).forEach(function(g){ g.style.animation = 'none'; g.style.opacity = 1; }); qsa('.tedge', tp).forEach(function(g){ g.style.animation = 'none'; g.style.opacity = 1; }); }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ api.tok++; await setup(true); var t = api.tok; api.stepDef.safe(function(){ return play(t); }); });
    async function render(){ await setup(false); }
    async function play(tok){ if (!cur) await setup(false); else await setup(false); var gs = qsa('.tnode', tp), ball = qs('.ball', tp), vals = featVals(cur.i), cu = nodes[0], msgs = [], feats = activeFeats(); api.cap('This compound enters at the top of the tree.');
      ball.setAttribute('cx', cu.x); ball.setAttribute('cy', cu.y - 30); gs[0].classList.add('on'); await api.sleep(1200);
      while (cu.l) { var goLeft = codeOf(D.B, cu.n.f, vals[cu.n.f]) <= cu.n.b, nxt = goLeft ? cu.l : cu.r, thr = thrText(D.B, cu.n.f, cu.n.b); api.cap('Is ' + cu.n.f + ' at most ' + thr + '? The compound has ' + fixFeat(cu.n.f, vals[cu.n.f]) + ', so the answer is ' + (goLeft ? 'yes' : 'no') + '.');
        var fx = +ball.getAttribute('cx'), fy = +ball.getAttribute('cy'), tx = nxt.x, ty = nxt.y; await api.anim(900, function(t){ ball.setAttribute('cx', lerp(fx, tx, t)); ball.setAttribute('cy', lerp(fy, ty, t)); }); gs[nodes.indexOf(nxt)].classList.add('on'); cu = nxt; await api.sleep(1500); }
      var pred = predictVals(tree, D.B, vals), meas = D.Y[cur.i], span = D.yhi - D.ylo; qs('.rr', rp).innerHTML = '<div style="display:flex;gap:30px;align-items:center"><div><div class="lbl">Predicted</div><div class="big" style="font-size:24px;color:' + C.red + '">' + pred.toFixed(2) + '</div></div><div><div class="lbl">Measured</div><div class="big" style="font-size:24px">' + meas.toFixed(2) + '</div></div><div style="flex:1"><div class="pbar"><i style="background:' + C.red + ';width:' + clamp((pred - D.ylo) / span * 100, 0, 100) + '%"></i></div><div class="pbar"><i style="background:' + C.char + ';width:' + clamp((meas - D.ylo) / span * 100, 0, 100) + '%"></i></div></div></div>';
      api.cap('The compound lands in a leaf and gets that leaf\'s average as its prediction.'); await api.sleep(2800); }
    return { render: render, play: play };
  } });
function fixFeat(f, v){ return fmtFeat(f, v); }

/* ---------- T4: forest ---------- */
L2.add({ short: 'Forest', title: 'A forest averages many different trees',
  change: 'Drag Forest: number of trees and Forest tree depth. Press Next compound for a different molecule.',
  watch: 'Each dot is one tree\'s prediction for the same compound. The red line is their average. The curve shows unseen R2 as trees are added.',
  build: function(api){
    var cur = null;
    var sp = api.el('pn', 'left:28px;top:28px;width:200px;height:230px', '<h4>Compound</h4><div class="struct" style="height:190px"></div>');
    var fp = api.el('pn', 'left:28px;top:270px;width:200px;height:166px', '<div class="vm" style="font-size:12.5px;color:#4a4f5a"></div>');
    var vp = api.el('pn', 'left:248px;top:28px;width:380px;height:408px', '<h4>Each tree votes</h4><canvas width="358" height="350"></canvas>');
    var cp = api.el('pn', 'left:648px;top:28px;width:284px;height:196px', '<h4>R2 as trees are added</h4><canvas width="262" height="146"></canvas>');
    var dp = api.el('pn', 'left:648px;top:236px;width:284px;height:200px', '<h4>R2 by forest depth, 30 trees</h4><canvas width="262" height="150"></canvas>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    function drawCurve(upto){ var fr = tForest(), pts = fr.curve.filter(function(c){ return c.k <= upto; }); lineChart(qs('canvas', cp), [{ pts: fr.curve.map(function(c){ return [c.k, c.te]; }).filter(function(p_){ return p_[0] <= upto; }), color: C.red }, { pts: fr.curve.map(function(c){ return [c.k, c.tr]; }).filter(function(p_){ return p_[0] <= upto; }), color: C.char }], { xl: 'number of trees', yl: 'R2', y0: 0, y1: 1, x0: 1, x1: fr.per.length, xdec: 0 }); }
    function drawDepth(){ var rows = tForestDepth(), p = TP(), g = cv(qs('canvas', dp)), x = g.x, box = { l: 36, r: g.w - 8, t: 8, b: g.h - 28 }, bw = (box.r - box.l) / rows.length; x.strokeStyle = C.silver; x.beginPath(); x.moveTo(box.l, box.b); x.lineTo(box.r, box.b); x.stroke();
      rows.forEach(function(r, i){ var near = Math.abs(r.d - p.fd) === Math.min.apply(null, rows.map(function(q){ return Math.abs(q.d - p.fd); })), bx = box.l + i * bw + bw * 0.1, w2 = bw * 0.38, h1 = Math.max(0, r.tr) * (box.b - box.t), h2 = Math.max(0, r.te) * (box.b - box.t); x.fillStyle = C.char; x.fillRect(bx, box.b - h1, w2, h1); x.fillStyle = C.red; x.fillRect(bx + w2 + 2, box.b - h2, w2, h2); x.fillStyle = C.char; x.font = (near ? '700 ' : '') + '11px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText('depth ' + r.d, bx + w2, box.b + 14); x.fillText(r.te.toFixed(2), bx + w2 * 1.5, box.b - h2 - 4); }); x.fillStyle = C.mid; x.textAlign = 'left'; x.font = '11px Aptos, Segoe UI, sans-serif'; x.fillText('dark: studied   red: unseen', box.l, box.t + 8); }
    async function setup(next){ await loadStructs(); if (!cur || next || cur._ep !== EP) { cur = await pickStruct(); cur._ep = EP; } qs('.struct', sp).innerHTML = cur.svg; qs('.vm', fp).innerHTML = 'Measured: <b>' + D.Y[cur.i].toFixed(2) + '</b><br>Each of the ' + TP().nt + ' trees sees a random resample of the training compounds and a random subset of features, so no two trees agree.'; }
    function vote(upto, tt){ var fr = tForest(), vals = featVals(cur.i), preds = fr.trees.map(function(t){ return predictVals(t, D.B, vals); }), lo = D.ylo, hi = D.yhi, g = cv(qs('canvas', vp)), x = g.x, box = { l: 14, r: g.w - 10, t: 10, b: g.h - 40 }, bins = 36, bw = (box.r - box.l) / bins, cnt = new Array(bins).fill(0), dots = preds.map(function(p_){ var b = clamp(Math.floor((p_ - lo) / (hi - lo) * bins), 0, bins - 1); return { b: b, h: cnt[b]++ }; }), rad = Math.min(bw / 2 - 1, 6), sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, my = D.Y[cur.i];
      x.strokeStyle = C.silver; x.beginPath(); x.moveTo(box.l, box.b); x.lineTo(box.r, box.b); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'center'; for (var v = Math.ceil(lo); v <= hi; v++) x.fillText(v, sx(v), box.b + 14); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText('prediction of one tree', (box.l + box.r) / 2, box.b + 32);
      for (var d = 0; d < upto; d++) { var dd = dots[d], cyp = box.b - rad - dd.h * (rad * 2 + 1), prog = d === upto - 1 ? tt : 1; x.fillStyle = 'rgba(161,177,195,.95)'; x.beginPath(); x.arc(box.l + (dd.b + 0.5) * bw, lerp(box.t, cyp, prog), rad, 0, 6.3); x.fill(); }
      x.strokeStyle = C.char; x.setLineDash([5, 4]); x.lineWidth = 2; x.beginPath(); x.moveTo(sx(my), box.t); x.lineTo(sx(my), box.b); x.stroke(); x.setLineDash([]); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('measured ' + my.toFixed(2), clamp(sx(my) + 5, 0, box.r - 90), box.t + 10);
      if (upto) { var m = mean(preds.slice(0, upto)); x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); x.moveTo(sx(m), box.t + 18); x.lineTo(sx(m), box.b); x.stroke(); x.fillStyle = C.red; x.fillText('forest ' + m.toFixed(2), clamp(sx(m) + 6, 0, box.r - 90), box.t + 28); } return preds; }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ api.tok++; await setup(true); var t = api.tok; api.stepDef.safe(function(){ return play(t); }); });
    async function render(){ await setup(false); var n = tForest().per.length; vote(n, 1); drawCurve(n); drawDepth(); }
    return { render: render };
  } });

/* ---------- T5: boosting ---------- */
L2.add({ short: 'Boosting', title: 'Boosting corrects the errors a little at a time',
  change: 'Drag Boosting rounds and Boosting learning rate. Drag Round below the stage to see the predictions at any point.',
  watch: 'Dots move toward the diagonal as rounds are added. Dashed lines show other learning rates with the same number of rounds.',
  build: function(api){
    var sp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px', '<h4>Predicted against measured, unseen compounds</h4><canvas width="398" height="352"></canvas>');
    var cp = api.el('pn', 'left:468px;top:28px;width:464px;height:300px', '<h4>Unseen R2 against rounds</h4><canvas width="442" height="250"></canvas>');
    var tp = api.el('pn', 'left:468px;top:340px;width:464px;height:96px', '<div class="bt" style="font-size:12.5px;color:#4a4f5a"></div>');
    function curves(upto){ var p = TP(), b = tBoost(), lrs = tBoostLr(), series = lrs.map(function(l, i){ return { pts: l.curve.filter(function(c){ return c.k <= upto; }).map(function(c){ return [c.k, c.te]; }), color: [C.bg, C.mid, C.tint][i], w: 1.8, dash: [4, 3], nodot: true, label: 'rate ' + l.lr }; });
      series.push({ pts: b.curve.filter(function(c){ return c.k <= upto; }).map(function(c){ return [c.k, c.te]; }), color: C.red, label: 'yours: ' + p.lr }); lineChart(qs('canvas', cp), series, { xl: 'boosting round', yl: 'R2', x0: 0, x1: p.rounds, y0: 0, xdec: 0 }); }
    function frame(k){ var b = tBoost(); var s = b.snaps[Math.min(k, b.snaps.length - 1)]; scatterPV(qs('canvas', sp), s.p, D.TEST, 'round ' + s.r + ', R2 ' + r2(D.TEST, s.p).toFixed(2)); }
    var rc = api.addControl('Round <input type="range" min="0" max="1" step="1" value="1" style="width:260px"><b class="rv"></b>'), rs = qs('input', rc), atEnd = true;
    rs.addEventListener('input', function(){ var b = tBoost(); atEnd = +rs.value >= b.snaps.length - 1; frame(+rs.value); qs('.rv', rc).textContent = b.snaps[Math.min(+rs.value, b.snaps.length - 1)].r; });
    function render(){ var b = tBoost(), p = TP(); rs.max = b.snaps.length - 1; if (atEnd) rs.value = b.snaps.length - 1; frame(+rs.value); qs('.rv', rc).textContent = b.snaps[Math.min(+rs.value, b.snaps.length - 1)].r; curves(p.rounds); qs('.bt', tp).innerHTML = 'Each round fits a small tree to the errors left by all earlier rounds, then adds ' + Math.round(p.lr * 100) + '% of it. A small rate learns slowly and carefully. A large rate learns fast and can overshoot.'; }
    return { render: render };
  } });

/* ---------- T6: which features matter ---------- */
L2.add({ short: 'Which features matter', title: 'Which features the model relies on',
  change: 'Untick a feature in the options.',
  watch: 'Left: how much each feature helped the trees split. Right: how much unseen R2 falls when the model is retrained without that feature.',
  build: function(api){
    var lp = api.el('pn', 'left:28px;top:28px;width:440px;height:408px', '<h4>Split gain inside the boosted trees</h4><div class="ib"></div>');
    var rp = api.el('pn', 'left:488px;top:28px;width:444px;height:408px', '<h4>Change in unseen R2 when a feature is removed</h4><div class="ab"></div>');
    function rows(prog){ var b = tBoost(), ab = tAblation(), tot = Object.keys(b.gain).reduce(function(a, k){ return a + b.gain[k]; }, 0), f = activeFeats(), imp = f.map(function(x){ return [x, (b.gain[x] || 0) / (tot || 1)]; }).sort(function(a, c){ return c[1] - a[1]; }), mx = Math.max.apply(null, imp.map(function(r){ return r[1]; })) || 1;
      qs('.ib', lp).innerHTML = imp.map(function(r){ return '<div style="display:grid;grid-template-columns:78px 1fr 44px;gap:8px;align-items:center;margin:9px 0"><span style="font-weight:600">' + r[0] + '</span><div class="pbar" style="margin:0;height:16px"><i style="background:' + C.red + ';width:' + (r[1] / mx * 100 * prog) + '%;transition:none"></i></div><b style="text-align:right">' + Math.round(r[1] * 100 * prog) + '%</b></div>'; }).join('');
      var ar = ab.rows.slice().sort(function(a, c){ return a.d - c.d; }), mxa = Math.max(0.01, Math.max.apply(null, ar.map(function(r){ return Math.abs(r.d); })));
      qs('.ab', rp).innerHTML = '<div style="font-size:12px;color:#6B7280;margin-bottom:4px">Model with all features: R2 ' + ab.base.toFixed(2) + '. Bars to the left mean accuracy drops without the feature.</div>' + ar.map(function(r){ var w = Math.abs(r.d) / mxa * 46 * prog, neg = r.d < 0; return '<div style="display:grid;grid-template-columns:78px 1fr 56px;gap:8px;align-items:center;margin:9px 0"><span style="font-weight:600">' + r.f + '</span><div style="position:relative;height:16px;background:#EDF2F3;border-radius:3px"><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#6B7280"></div><div style="position:absolute;top:0;bottom:0;border-radius:3px;background:' + (neg ? C.red : C.bg) + ';' + (neg ? 'right:50%;' : 'left:50%;') + 'width:' + w + '%"></div></div><b style="text-align:right">' + (r.d >= 0 ? '+' : '') + (r.d * prog).toFixed(3) + '</b></div>'; }).join(''); }
    function render(){ if (!activeFeats().length) return; rows(1); }
    return { render: render };
  } });

L2.add(filterStepDef('trees'));

/* ---------- results ---------- */
L2.add({ short: 'Results', title: 'Score and training time of the three tree models',
  change: 'Train each model type in the workbench, or press Train all three for Compare in the options.',
  watch: 'Unseen R2 and fit time for every tree model you have trained on this endpoint.',
  build: function(api){
    var cards = [0, 1, 2].map(function(k){ return api.el('abs', 'left:' + (28 + k * 307) + 'px;top:28px;width:290px;height:150px', ''); });
    var tp = api.el('pn', 'left:28px;top:196px;width:904px;height:240px', '<h4>Fit time in your browser</h4><div class="tm"></div>');
    function render(){ var names = ['Decision tree', 'Random forest', 'Gradient boosting'], ms = names.map(function(n){ return D.scores[n] || null; });
      cards.forEach(function(c, k){ c.innerHTML = ms[k] ? scoreCard(names[k] + ' (' + ms[k].params + ')', ms[k], 0) : '<div class="score"><b>' + names[k] + '</b><div class="s" style="margin-top:8px">Not trained yet. Train it in the workbench.</div></div>'; c.firstChild.style.height = '150px'; });
      var mx = Math.max.apply(null, ms.map(function(m){ return m ? m.secs : 0; })) || 1; qs('.tm', tp).innerHTML = ms.map(function(m, k){ return '<div style="display:grid;grid-template-columns:130px 1fr 80px;gap:12px;align-items:center;margin:12px 0;font-size:13px"><b>' + names[k] + '</b><div class="pbar" style="margin:0;height:16px"><i style="background:' + (k === 2 ? C.red : C.char) + ';width:' + (m ? Math.max(1.5, m.secs / mx * 100) : 0) + '%;transition:none"></i></div><span style="text-align:right;font-variant-numeric:tabular-nums">' + (m ? fmtT(m.secs) : 'n/a') + '</span></div>'; }).join('') + '<p class="note" style="font-size:12px;margin-top:10px">A single tree is nearly instant. Forests and boosting fit many trees, so they take longer. Neural networks and the GPT come next.</p>'; }
    return { render: render };
  } });

L2.glossary([
  ['Decision tree', 'Yes/no questions on features. A compound lands in a leaf and gets that leaf\'s average.'],
  ['Split', 'One question in the tree, such as logP <= 2.9.'], ['Depth', 'Questions in a row.'],
  ['Leaf', 'A box at the bottom of a tree. Its prediction is the training average of the compounds in it.'],
  ['Random forest', 'Many deep trees, each on a random resample of the compounds and a random subset of features. Predictions are averaged.'],
  ['Ensemble', 'Several models combined into one prediction.'],
  ['Gradient boosting', 'Small trees added in sequence, each fitting the error left by the earlier ones.'],
  ['Learning rate', 'Share of each new tree\'s correction that is applied.'],
  ['Studied R2', 'R2 on the compounds the model trained on. It rises with depth even when the model is overfitting.'],
  ['Unseen R2', 'R2 on held-out compounds. This is the score that matters.'],
  ['Feature importance', 'How much a feature helped the model. Depends on how it is measured.'],
  ['Ablation', 'Retraining without one feature to see how much the score falls.']
]);

setUses(L2, [null, ['endpoint', 'features', 'filters', 'tdepth'], ['endpoint', 'features', 'filters', 'tdepth'], ['endpoint', 'features', 'filters', 'tntrees', 'tfdepth'], ['endpoint', 'features', 'filters', 'trounds', 'tlr'], ['endpoint', 'features', 'filters'], null, ['endpoint', 'features', 'filters', 'tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr']]);
