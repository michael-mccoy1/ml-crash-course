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
function TP(){ var g = function(id, d){ var e = $(id); return e ? +e.value : d; }; return { depth: g('tdepth', 4), nt: g('tntrees', 60), fd: g('tfdepth', 10), rounds: g('trounds', 150), lr: g('tlr', 0.08) }; }
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
function recordMode(mode, auto){
  var p = TP(), f = activeFeats(); if (!f.length) return null;
  var a = mode === 'tree' ? tTree() : (mode === 'forest' ? tForest() : tBoost()), ms = record(TNAMES[mode], f, mode === 'tree' ? a.pred : (mode === 'forest' ? a.pred : a.F), a.secs);
  ms.auto = !!auto; ms.params = mode === 'tree' ? 'depth ' + p.depth : (mode === 'forest' ? p.nt + ' trees, depth ' + p.fd : p.rounds + ' rounds, rate ' + p.lr); return ms;
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
    var wide = bw >= 84, fv = o.n.v.toFixed(2); if (o.n.f === undefined || leaf) { l1 = (wide ? 'mean ' : '') + fv; l2 = 'n = ' + o.n.n; } else if (wide) { l1 = o.n.f + ' <= ' + thrText(B, o.n.f, o.n.b); l2 = 'n = ' + o.n.n; } else { l1 = o.n.f; l2 = '<= ' + thrText(B, o.n.f, o.n.b); }
    svg += '<g class="tnode" data-k="' + k + '" style="animation-delay:' + (ANIM ? o.d * 320 + 100 : 0) + 'ms"><rect x="' + (o.x - bw / 2).toFixed(1) + '" y="' + (o.y - 19) + '" width="' + bw.toFixed(1) + '" height="38" rx="7" fill="' + fill + '" stroke="' + (leaf ? C.bg : C.char) + '" stroke-width="1.5"/><text x="' + o.x.toFixed(1) + '" y="' + (o.y - 3) + '" text-anchor="middle" font-size="' + (bw < 60 ? 9.5 : (bw < 84 ? 10.5 : 12)) + '" font-weight="700" fill="' + tc + '">' + esc(l1) + '</text><text x="' + o.x.toFixed(1) + '" y="' + (o.y + 11) + '" text-anchor="middle" font-size="' + (bw < 60 ? 9 : (bw < 84 ? 9.5 : 10.5)) + '" fill="' + tc + '" opacity=".85">' + esc(l2) + '</text></g>'; });
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
  title: 'Build a model, press Train, and watch it fit',
  watch: 'A decision tree is a flowchart of yes or no questions about a compound. A random forest trains many trees, each on a different random draw of the compounds, and averages them. Gradient boosting builds small trees one after another, each fixing the mistakes the earlier ones left. In the top-right chart, the dark curve is the score on training compounds and the red curve is the score on test compounds.',
  change: 'Pick a model type, set its options in the options panel, then press Train.',
  build: function(api){
    var bb = benchBar(api, { leftHTML: '<div class="seg mode" data-mode>' + [['tree', 'Decision tree'], ['forest', 'Random forest'], ['boost', 'Gradient boosting']].map(function(it){ return '<button type="button" data-v="' + it[0] + '"' + (it[0] === TB.mode ? ' class="on"' : '') + '>' + it[1] + '</button>'; }).join('') + '</div>' });
    var mp = api.el('pn', 'left:28px;top:80px;width:576px;height:372px', '<h4 class="mh"></h4><svg class="static" viewBox="0 0 552 318" width="552" height="318"></svg>');
    var cp = api.el('pn', 'left:624px;top:80px;width:308px;height:178px', '<h4 class="ch"></h4><canvas width="284" height="140"></canvas>');
    var sp = api.el('pn', 'left:624px;top:270px;width:308px;height:182px', '<h4>Predictions on test compounds</h4><canvas width="284" height="144"></canvas>');
    var np = api.el('pn', 'left:28px;top:462px;width:904px;height:56px;display:flex;align-items:center;font-size:14px;line-height:1.4', '<div class="mn"></div>');
    var svg = qs('svg', mp), running = false;
    function dimOpts(){ var map = { tree: ['tdepth'], forest: ['tntrees', 'tfdepth'], boost: ['trounds', 'tlr'] }; ['tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr'].forEach(function(id){ var og = $(id).closest('.og'); og.classList.toggle('dim', map[TB.mode].indexOf(id) < 0); }); }
    function msg(t){ svg.innerHTML = '<text x="20" y="40" font-size="13" fill="' + C.red + '">' + t + '</text>'; }
    function pictureTree(dShow){
      var p = TP(), f = activeFeats(), d = dShow || p.depth, dd = Math.min(d, 2);
      var t = tCache('livetree', dd, function(){ return fitTree(f, activeTrain(), dd, 20, 1); }), r = treeSVG(t, D.B, dd, 552, 200), s = r.svg.replace(/<circle class="ball"[^>]*\/>/, ''), leaves = Math.pow(2, d), n = Math.min(leaves, 512), cols = 64, sz = 552 / cols, strip = '';
      for (var i = 0; i < n; i++) strip += '<rect x="' + ((i % cols) * sz + 1).toFixed(1) + '" y="' + (258 + Math.floor(i / cols) * sz).toFixed(1) + '" width="' + (sz - 2).toFixed(1) + '" height="' + (sz - 2).toFixed(1) + '" fill="' + C.red + '" opacity=".85"/>';
      svg.innerHTML = s + '<text x="0" y="244" font-size="13" font-weight="700" fill="' + C.char + '">Depth ' + d + ' allows up to ' + leaves.toLocaleString() + ' groups' + (d > 2 ? ' (first 2 levels drawn)' : '') + '</text>' + strip;
    }
    function miniTree(x0, y0, w, h, depth, col){
      var out = '', dy = h / depth; (function rec(x, y, span, d){ if (d === depth) { out += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="2.6" fill="' + col + '"/>'; return; }
        var ny = y + dy, lx = x - span / 2, rx = x + span / 2; out += '<path d="M' + x.toFixed(1) + ' ' + y.toFixed(1) + 'L' + lx.toFixed(1) + ' ' + ny.toFixed(1) + 'M' + x.toFixed(1) + ' ' + y.toFixed(1) + 'L' + rx.toFixed(1) + ' ' + ny.toFixed(1) + '" stroke="' + col + '" stroke-width="1.4" fill="none"/>'; if (d > 0 || depth < 2) out += ''; rec(lx, ny, span / 2, d + 1); rec(rx, ny, span / 2, d + 1); })(x0 + w / 2, y0 + 4, w * 0.46, 0); return out;
    }
    function pictureForest(lit){
      var p = TP(), shown = Math.min(p.nt, 12), cols = 4, rows = Math.ceil(shown / cols), cw = 552 / cols, ch = 264 / 3, dv = clamp(p.fd - 5, 3, 5), s = '';
      for (var k = 0; k < shown; k++) { var cx = (k % cols) * cw, cy = Math.floor(k / cols) * ch, on = k < lit; s += '<rect x="' + (cx + 5) + '" y="' + (cy + 4) + '" width="' + (cw - 10) + '" height="' + (ch - 8) + '" rx="6" fill="' + (on ? '#EDF2F4' : '#fff') + '" stroke="' + (on ? C.red : C.silver) + '"/>' + miniTree(cx + 14, cy + 24, cw - 28, ch - 40, dv, on ? C.red : C.bg) + '<text x="' + (cx + 14) + '" y="' + (cy + 20) + '" font-size="11" font-weight="700" fill="' + C.mid + '">tree ' + (k + 1) + '</text>'; }
      s += '<text x="0" y="296" font-size="13" font-weight="700" fill="' + C.char + '">' + p.nt + ' trees, each up to depth ' + p.fd + '</text><text x="0" y="313" font-size="11.5" fill="' + C.mid + '">' + (p.nt > shown ? shown + ' drawn. ' : '') + 'Every tree is fit on a different random sample, so no two are alike.</text>'; svg.innerHTML = s;
    }
    function pictureBoost(lit){
      var p = TP(), K = p.rounds, n = Math.min(K, 6), s = '', cw = 58, gap = 18, y0 = 22, ch = 76, xs = 4;
      s += '<text x="0" y="12" font-size="12.5" font-weight="700" fill="' + C.char + '">Prediction = start value + a share of each small tree</text>';
      var cards = []; for (var i = 0; i < n; i++) cards.push({ label: 'tree ' + (i + 1), idx: i / n * K }); cards.push({ label: 'tree ' + K, idx: K - 0.5, last: true });
      cards.forEach(function(c, i){ var x = xs + i * (cw + gap) + (c.last ? 10 : 0), on = c.idx < lit, col = on ? C.red : C.bg;
        s += '<rect x="' + x + '" y="' + y0 + '" width="' + cw + '" height="' + ch + '" rx="6" fill="' + (on ? '#EDF2F4' : '#fff') + '" stroke="' + (on ? C.red : C.silver) + '"/>' + miniTree(x + 8, y0 + 8, cw - 16, ch - 32, 2, col) + '<text x="' + (x + cw / 2) + '" y="' + (y0 + ch - 8) + '" font-size="10" text-anchor="middle" fill="' + C.mid + '">' + c.label + '</text>';
        if (i < cards.length - 1) s += '<text x="' + (x + cw + (c.last ? 0 : gap / 2)) + '" y="' + (y0 + ch / 2 + 5) + '" font-size="' + (i === n - 1 ? 16 : 16) + '" text-anchor="middle" fill="' + C.char + '">' + (i === n - 1 ? '...' : '+') + '</text>'; });
      var by = 118; s += '<text x="0" y="' + by + '" font-size="12" fill="' + C.char + '">Each tree adds <tspan font-weight="700" fill="' + C.red + '">' + Math.round(p.lr * 100) + '%</tspan> of its correction (the learning rate)</text><rect x="0" y="' + (by + 8) + '" width="552" height="9" rx="4" fill="' + C.silver + '"/><rect x="0" y="' + (by + 8) + '" width="' + (552 * p.lr).toFixed(1) + '" height="9" rx="4" fill="' + C.red + '"/>';
      var x0 = 40, x1 = 540, yT = 176, yB = 292, pts = [], step = Math.max(1, Math.round(K / 120));
      for (var k = 0; k <= K; k += step) pts.push([x0 + k / K * (x1 - x0), yT + (1 - Math.pow(1 - p.lr, k)) * (yB - yT)]); pts.push([x1, yT + (1 - Math.pow(1 - p.lr, K)) * (yB - yT)]);
      s += '<text x="0" y="168" font-size="12" font-weight="700" fill="' + C.char + '">Error that could remain after each round</text><line x1="' + x0 + '" y1="' + yB + '" x2="' + x1 + '" y2="' + yB + '" stroke="' + C.silver + '"/><line x1="' + x0 + '" y1="' + yT + '" x2="' + x0 + '" y2="' + yB + '" stroke="' + C.silver + '"/><text x="' + (x0 - 5) + '" y="' + (yT + 4) + '" font-size="10.5" text-anchor="end" fill="' + C.mid + '">100%</text><text x="' + (x0 - 5) + '" y="' + (yB + 4) + '" font-size="10.5" text-anchor="end" fill="' + C.mid + '">0%</text><text x="' + x0 + '" y="' + (yB + 16) + '" font-size="10.5" fill="' + C.mid + '">round 0</text><text x="' + x1 + '" y="' + (yB + 16) + '" font-size="10.5" text-anchor="end" fill="' + C.mid + '">round ' + K + '</text>';
      s += '<path d="M' + pts.map(function(q){ return q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join('L') + '" fill="none" stroke="' + C.red + '" stroke-width="2.6"/>';
      var left = Math.pow(1 - p.lr, K) * 100; s += '<text x="' + x1 + '" y="' + (yB - 8) + '" font-size="11.5" text-anchor="end" fill="' + C.dred + '">' + (left < 0.5 ? 'under 1%' : left.toFixed(0) + '%') + ' left at the end if every tree fit perfectly</text>'; svg.innerHTML = s;
    }
    function picture(lit, dShow){ if (!activeFeats().length) { msg('Switch on at least one feature.'); return; } if (TB.mode === 'tree') pictureTree(dShow); else if (TB.mode === 'forest') pictureForest(lit === undefined ? 0 : lit); else pictureBoost(lit === undefined ? 0 : lit); }
    function peek(){ var p = TP(); if (!activeFeats().length) return null; if (TB.mode === 'tree') { var a = tPeek('tree', p.depth); return a ? { pred: a.pred, secs: a.secs, sweep: a.sweep } : null; } if (TB.mode === 'forest') { var b = tPeek('forest', [p.nt, p.fd]); return b ? { pred: b.pred, secs: b.secs, curve: b.curve, per: b.per } : null; } var c = tPeek('boost', [p.rounds, p.lr]); return c ? { pred: c.F, secs: c.secs, curve: c.curve, snaps: c.snaps } : null; }
    function curveOpts(res){ var p = TP(), vals = [], x1; if (TB.mode === 'tree') { res && res.sweep.forEach(function(r){ vals.push(r[1], r[2]); }); x1 = 14; } else { res && res.curve.forEach(function(c){ vals.push(c.te, c.tr); }); x1 = TB.mode === 'forest' ? p.nt : p.rounds; } return { x0: 1, x1: Math.max(2, x1), y0: Math.min(0, Math.min.apply(null, vals.length ? vals : [0])), y1: 1, xl: TB.mode === 'tree' ? 'tree depth' : (TB.mode === 'forest' ? 'trees in the forest' : 'boosting rounds'), yl: 'R2', xdec: 0, ydec: 2 }; }
    function drawCurve(res, upto){
      var o = curveOpts(res), canvas = qs('canvas', cp); qs('.ch', cp).textContent = 'R2: dark training, red unseen';
      if (!res) { lineChart(canvas, [{ pts: [[o.x0, 0], [o.x1, 1]], color: 'rgba(0,0,0,0)', nodot: true }], o); return; }
      var tr, te; if (TB.mode === 'tree') { var sw = res.sweep.filter(function(r){ return r[0] <= upto; }); tr = sw.map(function(r){ return [r[0], r[1]]; }); te = sw.map(function(r){ return [r[0], r[2]]; }); } else { var cu = res.curve.filter(function(c){ return c.k <= upto; }); tr = cu.map(function(c){ return [c.k, c.tr]; }); te = cu.map(function(c){ return [c.k, c.te]; }); }
      lineChart(canvas, [{ pts: tr, color: C.char, nodot: true }, { pts: te, color: C.red }], o);
    }
    function drawScatter(pred, label){ var canvas = qs('canvas', sp); if (!pred) { scatterPV(canvas, new Float32Array(D.N), [], null); return; } scatterPV(canvas, pred, D.TEST, label, { max: 320 }); }
    function note(){
      var p = TP(), t = TB.mode === 'tree' ? 'Depth ' + p.depth + ' allows up to ' + Math.pow(2, p.depth).toLocaleString() + ' groups. Each group predicts the average of its training compounds.'
        : (TB.mode === 'forest' ? p.nt + ' trees, each allowed ' + p.fd + ' questions. Each learns from a different random draw of the compounds. The answer is their average.'
        : p.rounds + ' small trees, one after another. Each fixes what the last got wrong, at ' + Math.round(p.lr * 100) + '% strength (the learning rate).'); qs('.mn', np).textContent = t;
    }
    function render(){
      dimOpts(); qs('.mh', mp).textContent = TB.mode === 'tree' ? 'The tree you are building' : (TB.mode === 'forest' ? 'The forest you are building' : 'The boosted ensemble you are building'); var res = peek(), p = TP();
      picture(res ? 1e9 : 0); drawCurve(res, 1e9); drawScatter(res ? res.pred : null, res ? 'test R2 ' + fix(r2(D.TEST, res.pred), 2) : null); note();
      if (res) { api.stale(''); bb.say('Test R2 ' + fix(r2(D.TEST, res.pred), 2) + '. The fit took ' + fmtT(res.secs) + '.'); } else { api.stale(TB.last[TB.mode] ? 'Options changed. Press Train.' : ''); bb.say(TB.last[TB.mode] ? '' : 'Ready. Press Train.'); }
    }
    function fit(mode){ if (mode === 'tree') { var a = tTree(); return { pred: a.pred, secs: a.secs, sweep: a.sweep }; } if (mode === 'forest') { var b = tForest(); return { pred: b.pred, secs: b.secs, curve: b.curve, per: b.per }; } var c = tBoost(); return { pred: c.F, secs: c.secs, curve: c.curve, snaps: c.snaps }; }
    async function replay(mode, res, tok){
      var p = TP(), total = 9000 * speedMul(), f = activeFeats();
      if (mode === 'tree') { var step = total / p.depth; for (var d = 1; d <= p.depth; d++) { api.checkTok(tok); var pr = tCache('treeD', d, function(){ return treeAll(fitTree(f, activeTrain(), d, 5, 1), D.N); }); pictureTree(d); drawCurve(res, d); drawScatter(pr, 'depth ' + d + ', R2 ' + fix(r2(D.TEST, pr), 2)); bb.say('Depth ' + d + ' of ' + p.depth + '. Test R2 ' + fix(r2(D.TEST, pr), 2)); await api.sleep(step); } }
      else if (mode === 'forest') { var n = p.nt, cps = Math.min(n, 40), acc = new Float64Array(D.N), pred = new Float64Array(D.N), done = 0, shown = Math.min(n, 40); for (var c = 1; c <= cps; c++) { api.checkTok(tok); var k = Math.round(c * n / cps); while (done < k) { var pt = res.per[done]; for (var i = 0; i < D.N; i++) acc[i] += pt[i]; done++; } for (var j = 0; j < D.N; j++) pred[j] = acc[j] / k; pictureForest(Math.ceil(k / n * shown)); drawCurve(res, k); drawScatter(pred, k + ' trees, R2 ' + fix(r2(D.TEST, pred), 2)); bb.say('Tree ' + k + ' of ' + n + '. Test R2 ' + fix(r2(D.TEST, pred), 2)); await api.sleep(total / cps); } }
      else { var sn = res.snaps; for (var q = 0; q < sn.length; q++) { api.checkTok(tok); pictureBoost(sn[q].r); drawCurve(res, sn[q].r); drawScatter(sn[q].p, 'round ' + sn[q].r + ', R2 ' + fix(r2(D.TEST, sn[q].p), 2)); bb.say('Round ' + sn[q].r + ' of ' + p.rounds + '. Test R2 ' + fix(r2(D.TEST, sn[q].p), 2)); await api.sleep(total / sn.length); } }
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
L2.add({ short: 'Choosing a question', title: 'A tree asks one question to split compounds into two groups',
  change: 'Drag across the plot to move the cut, then press Find the best cut.',
  watch: 'A tree starts with all compounds in one group and asks one question to split them in two, such as "is logP below 2.9?". Each group predicts its own average, the short horizontal line. A good question leaves the dots close to their line. The miss score below measures how far the dots sit from their group\'s line, so lower is better.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:580px;height:408px', '<h4>Each dot is one training compound</h4><canvas width="556" height="346"></canvas>');
    var tp = api.el('pn', 'left:628px;top:28px;width:304px;height:272px', '<div class="qq" style="font-size:13px;line-height:1.45"></div>');
    var gp = api.el('pn', 'left:628px;top:310px;width:304px;height:126px', '<h4>Miss score at every possible cut</h4><canvas width="284" height="86"></canvas>');
    var ctl = api.addControl('Feature <select></select> <button type="button" class="btn sm" data-best>Find the best cut</button>'); var sel = qs('select', ctl), S = null, pts = [], running = false;
    function fillSel(){ var f = activeFeats(), cur = sel.value; sel.innerHTML = f.map(function(x){ return '<option>' + x + '</option>'; }).join(''); if (f.indexOf(cur) >= 0) sel.value = cur; }
    function compute(){
      var feat = sel.value, idx = activeTrain().slice().sort(function(a, b){ return D.COL[feat][a] - D.COL[feat][b]; }), n = idx.length, ys = idx.map(function(i){ return D.Y[i]; }), pre = new Float64Array(n + 1); for (var i = 0; i < n; i++) pre[i + 1] = pre[i] + ys[i];
      var T = pre[n], sst = ys.reduce(function(a, v){ return a + (v - T / n) * (v - T / n); }, 0), cands = [];
      for (var q = 0; q < 60; q++) { var k = Math.round(n * (0.04 + 0.92 * q / 59)); if (k < 5 || n - k < 5) continue; var sl = pre[k], sr = T - sl, gain = sl * sl / k + sr * sr / (n - k) - T * T / n; cands.push({ t: D.COL[feat][idx[k]], mL: sl / k, mR: sr / (n - k), nL: k, nR: n - k, miss: (sst - gain) / n }); }
      var best = cands.reduce(function(a, b){ return b.miss < a.miss ? b : a; }, cands[0]), r = rng(5), xs = D.COL[feat]; pts = idx.filter(function(){ return r() < Math.min(1, 500 / n); });
      var keep = S && S.feat === feat && S.cur ? cands.reduce(function(a, b){ return Math.abs(b.t - S.cur.t) < Math.abs(a.t - S.cur.t) ? b : a; }, cands[0]) : cands[Math.floor(cands.length * 0.2)];
      return { feat: feat, cands: cands, best: best, cur: keep, base: sst / n, lo: xs[idx[Math.floor(n * 0.01)]], hi: xs[idx[Math.floor(n * 0.99)]], n: n };
    }
    function draw(){
      if (!S) return; var c = S.cur, g = cv(qs('canvas', plot)), x = g.x, box = { l: 48, r: g.w - 12, t: 10, b: g.h - 40 }, f = S.feat, lo = S.lo, hi = S.hi; axes(g, box, [lo, hi], [D.ylo, D.yhi], f, unitWord(), 5);
      var sx = function(v){ return box.l + (clamp(v, lo, hi) - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, D.ylo, D.yhi) - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      pts.forEach(function(i){ var left = D.COL[f][i] <= c.t; x.fillStyle = left ? 'rgba(161,177,195,.9)' : 'rgba(225,36,42,.6)'; x.beginPath(); x.arc(sx(D.COL[f][i]), sy(D.Y[i]), 3, 0, 6.3); x.fill(); });
      var tx = sx(c.t); x.strokeStyle = C.char; x.lineWidth = 2; x.setLineDash([6, 4]); x.beginPath(); x.moveTo(tx, box.t); x.lineTo(tx, box.b); x.stroke(); x.setLineDash([]);
      x.strokeStyle = C.char; x.lineWidth = 3.4; x.beginPath(); x.moveTo(box.l, sy(c.mL)); x.lineTo(tx, sy(c.mL)); x.stroke(); x.beginPath(); x.moveTo(tx, sy(c.mR)); x.lineTo(box.r, sy(c.mR)); x.stroke(); x.lineWidth = 1;
      x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillStyle = C.char; x.textAlign = 'left'; x.fillText('left group average ' + c.mL.toFixed(2), box.l + 6, sy(c.mL) - 7); x.textAlign = 'right'; x.fillText('right group average ' + c.mR.toFixed(2), box.r - 6, sy(c.mR) - 7);
      x.textAlign = 'center'; x.fillStyle = C.mid; x.fillText('cut: drag across the plot to move it', tx, box.t + 12);
      var g2 = cv(qs('canvas', gp)), x2 = g2.x, b2 = { l: 34, r: g2.w - 8, t: 8, b: g2.h - 24 }, mx = Math.max.apply(null, S.cands.map(function(k){ return k.miss; })), mn = Math.min.apply(null, S.cands.map(function(k){ return k.miss; })), gx = function(t){ return b2.l + (clamp(t, lo, hi) - lo) / (hi - lo) * (b2.r - b2.l); }, gy = function(v){ return b2.b - (v - mn) / (mx - mn || 1) * (b2.b - b2.t); };
      x2.strokeStyle = C.silver; x2.beginPath(); x2.moveTo(b2.l, b2.b); x2.lineTo(b2.r, b2.b); x2.stroke(); x2.fillStyle = C.mid; x2.font = '11px Aptos, Segoe UI, sans-serif'; x2.textAlign = 'center'; x2.fillText('where the cut is, along ' + f, (b2.l + b2.r) / 2, g2.h - 6); x2.textAlign = 'right'; x2.fillText('high', b2.l - 4, b2.t + 8); x2.fillText('low', b2.l - 4, b2.b);
      x2.strokeStyle = C.bg; x2.lineWidth = 2.4; x2.beginPath(); S.cands.forEach(function(k, i){ i ? x2.lineTo(gx(k.t), gy(k.miss)) : x2.moveTo(gx(k.t), gy(k.miss)); }); x2.stroke();
      x2.fillStyle = C.red; x2.beginPath(); x2.arc(gx(S.best.t), gy(S.best.miss), 5, 0, 6.3); x2.fill(); x2.fillStyle = C.char; x2.beginPath(); x2.arc(gx(c.t), gy(c.miss), 5.5, 0, 6.3); x2.fill(); x2.strokeStyle = '#fff'; x2.lineWidth = 1.5; x2.stroke();
      var cut = Math.round((S.base - c.miss) / S.base * 100), isBest = c === S.best;
      qs('.qq', tp).innerHTML = '<div class="lbl">The question this cut asks</div><div class="big" style="font-size:19px;color:' + C.red + ';margin:3px 0 7px">Is ' + esc(f) + ' at most ' + (+c.t.toFixed(2)) + '?</div><div class="nums" style="font-size:12.5px"><div><span>Yes, left group</span><b>' + c.nL + ' compounds, average ' + c.mL.toFixed(2) + '</b></div><div><span>No, right group</span><b>' + c.nR + ' compounds, average ' + c.mR.toFixed(2) + '</b></div></div><div class="lbl" style="margin-top:10px">Miss score (lower is better)</div><div class="big" style="font-size:26px">' + c.miss.toFixed(3) + '</div><div class="note" style="font-size:12px;line-height:1.4">How far the dots sit from their group\'s line, on average. Big misses count extra. With no cut, one line for everyone, it would be ' + S.base.toFixed(3) + '. This cut removes ' + cut + '% of that. ' + (isBest ? '<b style="color:' + C.red + '">This is the best cut the tree can find.</b>' : 'The red dot below is the best cut.') + '</div>';
    }
    function pick(px){ if (!S) return; var cvEl = qs('canvas', plot), r = cvEl.getBoundingClientRect(), v = S.lo + ((px - r.left) / r.width * 556 - 48) / (556 - 12 - 48) * (S.hi - S.lo); S.cur = S.cands.reduce(function(a, b){ return Math.abs(b.t - v) < Math.abs(a.t - v) ? b : a; }, S.cands[0]); draw(); }
    (function(){ var cvEl = qs('canvas', plot), down = false; cvEl.style.cursor = 'ew-resize'; cvEl.addEventListener('pointerdown', function(e){ down = true; api.tok++; cvEl.setPointerCapture(e.pointerId); pick(e.clientX); }); cvEl.addEventListener('pointermove', function(e){ if (down) pick(e.clientX); }); cvEl.addEventListener('pointerup', function(){ down = false; }); })();
    sel.addEventListener('change', function(){ api.tok++; S = compute(); draw(); });
    qs('[data-best]', ctl).addEventListener('click', async function(){ if (!S || running) return; running = true; api.tok++; var tok = api.tok; try { for (var k = 0; k < S.cands.length; k++) { api.checkTok(tok); S.cur = S.cands[k]; draw(); await api.sleep(45); } S.cur = S.best; draw(); } catch (e) { if (!(e instanceof Cancel)) console.error(e); } running = false; });
    function render(){ fillSel(); if (!sel.value) return; S = compute(); draw(); }
    return { render: render };
  } });

/* ---------- T3: send a compound down the tree ---------- */
L2.add({ short: 'One compound', title: 'Follow one compound down the tree',
  change: 'Press Next compound.',
  watch: 'Once a tree is built, a compound is sent down it. At each box it answers one yes or no question and takes the matching branch. It ends in a box at the bottom called a leaf. The prediction is the average lab value of the training compounds that ended up in that leaf.',
  build: function(api){
    var cur = null;
    var sp = api.el('pn', 'left:28px;top:28px;width:236px;height:196px', '<h4>Compound</h4><div class="struct" style="height:156px"></div>');
    var fp = api.el('pn', 'left:28px;top:234px;width:236px;height:202px', '<div class="nums fv" style="font-size:11.5px"></div>');
    var tp = api.el('pn', 'left:284px;top:28px;width:648px;height:330px', '<svg viewBox="0 0 628 300" width="628" height="300"></svg>');
    var rp = api.el('pn', 'left:284px;top:370px;width:648px;height:66px', '<div class="rr"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var nodes = null, tree = null;
    async function setup(next){ await loadStructs(); if (!cur || next || cur._ep !== EP) { cur = await pickStruct(); cur._ep = EP; } var t = tTree(), p = TP(), md = Math.min(3, p.depth); tree = t.show; var out = treeSVG(t.show, D.B, md, 628, 300); nodes = out.nodes; qs('svg', tp).innerHTML = out.svg;
      qs('.struct', sp).innerHTML = cur.svg; qs('.fv', fp).innerHTML = D.BASE.map(function(f){ return '<div><span>' + f + '</span><b>' + fmtFeat(f, D.COL[f][cur.i]) + '</b></div>'; }).join('') + (D.COL.logD_pred && FSEL.logD_pred ? '<div><span>logD_pred</span><b>' + D.COL.logD_pred[cur.i].toFixed(2) + '</b></div>' : ''); qs('.rr', rp).innerHTML = '<span class="note">Press Walk through to send this compound down the tree.</span>'; qsa('.tnode', tp).forEach(function(g){ g.style.animation = 'none'; g.style.opacity = 1; }); qsa('.tedge', tp).forEach(function(g){ g.style.animation = 'none'; g.style.opacity = 1; }); }
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
L2.add({ short: 'Forest', title: 'A forest averages many trees',
  change: 'Drag Forest: number of trees in the options panel. Press Next compound for another molecule.',
  watch: 'A random forest is many trees, each trained on a different random draw of the compounds, so each tree is a little different. To predict, every tree gives its own answer and the forest returns the average. Each grey dot is one tree\'s answer for the same compound. The red line is their average. The charts show how the score changes with more trees and with deeper trees.',
  build: function(api){
    var cur = null;
    var sp = api.el('pn', 'left:28px;top:28px;width:200px;height:230px', '<h4>Compound</h4><div class="struct" style="height:190px"></div>');
    var fp = api.el('pn', 'left:28px;top:270px;width:200px;height:166px', '<div class="vm" style="font-size:12.5px;color:#34373F"></div>');
    var vp = api.el('pn', 'left:248px;top:28px;width:380px;height:408px', '<h4>Each tree votes</h4><canvas width="358" height="350"></canvas>');
    var cp = api.el('pn', 'left:648px;top:28px;width:284px;height:196px', '<h4>R2 as trees are added</h4><canvas width="262" height="146"></canvas>');
    var dp = api.el('pn', 'left:648px;top:236px;width:284px;height:200px', '<h4>R2 by forest depth, 30 trees</h4><canvas width="262" height="150"></canvas>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    function drawCurve(upto){ var fr = tForest(), pts = fr.curve.filter(function(c){ return c.k <= upto; }); lineChart(qs('canvas', cp), [{ pts: fr.curve.map(function(c){ return [c.k, c.te]; }).filter(function(p_){ return p_[0] <= upto; }), color: C.red }, { pts: fr.curve.map(function(c){ return [c.k, c.tr]; }).filter(function(p_){ return p_[0] <= upto; }), color: C.char }], { xl: 'number of trees', yl: 'R2', y0: 0, y1: 1, x0: 1, x1: fr.per.length, xdec: 0 }); }
    function drawDepth(){ var rows = tForestDepth(), p = TP(), g = cv(qs('canvas', dp)), x = g.x, box = { l: 36, r: g.w - 8, t: 8, b: g.h - 28 }, bw = (box.r - box.l) / rows.length; x.strokeStyle = C.silver; x.beginPath(); x.moveTo(box.l, box.b); x.lineTo(box.r, box.b); x.stroke();
      rows.forEach(function(r, i){ var near = Math.abs(r.d - p.fd) === Math.min.apply(null, rows.map(function(q){ return Math.abs(q.d - p.fd); })), bx = box.l + i * bw + bw * 0.1, w2 = bw * 0.38, h1 = Math.max(0, r.tr) * (box.b - box.t), h2 = Math.max(0, r.te) * (box.b - box.t); x.fillStyle = C.char; x.fillRect(bx, box.b - h1, w2, h1); x.fillStyle = C.red; x.fillRect(bx + w2 + 2, box.b - h2, w2, h2); x.fillStyle = C.char; x.font = (near ? '700 ' : '') + '11px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText('depth ' + r.d, bx + w2, box.b + 14); x.fillText(r.te.toFixed(2), bx + w2 * 1.5, box.b - h2 - 4); }); x.fillStyle = C.mid; x.textAlign = 'left'; x.font = '11px Aptos, Segoe UI, sans-serif'; x.fillText('dark: studied   red: unseen', box.l, box.t + 8); }
    async function setup(next){ await loadStructs(); if (!cur || next || cur._ep !== EP) { cur = await pickStruct(); cur._ep = EP; } qs('.struct', sp).innerHTML = cur.svg; qs('.vm', fp).innerHTML = 'Measured: <b>' + D.Y[cur.i].toFixed(2) + '</b><br>Each of the ' + TP().nt + ' trees learned from a different random draw of the training compounds, so the trees do not all agree.'; }
    function vote(upto, tt){ var fr = tForest(), vals = featVals(cur.i), preds = fr.trees.map(function(t){ return predictVals(t, D.B, vals); }), lo = D.ylo, hi = D.yhi, g = cv(qs('canvas', vp)), x = g.x, box = { l: 14, r: g.w - 10, t: 10, b: g.h - 40 }, bins = 36, bw = (box.r - box.l) / bins, cnt = new Array(bins).fill(0), dots = preds.map(function(p_){ var b = clamp(Math.floor((p_ - lo) / (hi - lo) * bins), 0, bins - 1); return { b: b, h: cnt[b]++ }; }), rad = Math.min(bw / 2 - 1, 6), sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, my = D.Y[cur.i];
      x.strokeStyle = C.silver; x.beginPath(); x.moveTo(box.l, box.b); x.lineTo(box.r, box.b); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'center'; for (var v = Math.ceil(lo); v <= hi; v++) x.fillText(v, sx(v), box.b + 14); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText('prediction of one tree', (box.l + box.r) / 2, box.b + 32);
      for (var d = 0; d < upto; d++) { var dd = dots[d], cyp = box.b - rad - dd.h * (rad * 2 + 1), prog = d === upto - 1 ? tt : 1; x.fillStyle = 'rgba(161,177,195,.95)'; x.beginPath(); x.arc(box.l + (dd.b + 0.5) * bw, lerp(box.t, cyp, prog), rad, 0, 6.3); x.fill(); }
      x.strokeStyle = C.char; x.setLineDash([5, 4]); x.lineWidth = 2; x.beginPath(); x.moveTo(sx(my), box.t); x.lineTo(sx(my), box.b); x.stroke(); x.setLineDash([]); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('measured ' + my.toFixed(2), clamp(sx(my) + 5, 0, box.r - 90), box.t + 10);
      if (upto) { var m = mean(preds.slice(0, upto)); x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); x.moveTo(sx(m), box.t + 18); x.lineTo(sx(m), box.b); x.stroke(); x.fillStyle = C.red; x.fillText('forest ' + m.toFixed(2), clamp(sx(m) + 6, 0, box.r - 90), box.t + 28); } return preds; }
    async function render(){ await setup(false); var n = tForest().per.length; vote(n, 1); drawCurve(n); drawDepth(); }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ api.tok++; await setup(true); await render(); });
    return { render: render };
  } });

/* ---------- T5: boosting ---------- */
L2.add({ short: 'Boosting', title: 'Boosting fixes mistakes a little at a time',
  change: 'Drag Round below the plot. Change Boosting learning rate in the options panel.',
  watch: 'Boosting builds small trees one after another. The first tree makes a rough prediction. Each next tree is trained on the mistakes left so far, and its correction is added with a small weight. The learning rate is that weight: a small one adds corrections slowly and carefully. The dots move toward the diagonal as trees are added.',
  build: function(api){
    var sp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px', '<h4>Predicted against measured, test compounds</h4><canvas width="398" height="352"></canvas>');
    var cp = api.el('pn', 'left:468px;top:28px;width:464px;height:300px', '<h4>Test R2 against rounds</h4><canvas width="442" height="250"></canvas>');
    var tp = api.el('pn', 'left:468px;top:340px;width:464px;height:96px', '<div class="bt" style="font-size:12.5px;color:#34373F"></div>');
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
  change: 'Untick a feature in the options panel.',
  watch: 'Feature importance asks which features the model relies on. Left: split gain, how much each feature reduced the miss score across all the questions that used it. Right: ablation, the drop in test R2 when the model is trained again without that feature. The two can disagree, because a correlated feature can take over the work of a missing one.',
  build: function(api){
    var lp = api.el('pn', 'left:28px;top:28px;width:440px;height:408px', '<h4>Split gain inside the boosted trees</h4><div class="ib"></div>');
    var rp = api.el('pn', 'left:488px;top:28px;width:444px;height:408px', '<h4>Change in test R2 when a feature is removed</h4><div class="ab"></div>');
    function rows(prog){ var b = tBoost(), ab = tAblation(), tot = Object.keys(b.gain).reduce(function(a, k){ return a + b.gain[k]; }, 0), f = activeFeats(), imp = f.map(function(x){ return [x, (b.gain[x] || 0) / (tot || 1)]; }).sort(function(a, c){ return c[1] - a[1]; }), mx = Math.max.apply(null, imp.map(function(r){ return r[1]; })) || 1;
      qs('.ib', lp).innerHTML = imp.map(function(r){ return '<div style="display:grid;grid-template-columns:78px 1fr 44px;gap:8px;align-items:center;margin:9px 0"><span style="font-weight:600">' + r[0] + '</span><div class="pbar" style="margin:0;height:16px"><i style="background:' + C.red + ';width:' + (r[1] / mx * 100 * prog) + '%;transition:none"></i></div><b style="text-align:right">' + Math.round(r[1] * 100 * prog) + '%</b></div>'; }).join('');
      var ar = ab.rows.slice().sort(function(a, c){ return a.d - c.d; }), mxa = Math.max(0.01, Math.max.apply(null, ar.map(function(r){ return Math.abs(r.d); })));
      qs('.ab', rp).innerHTML = '<div style="font-size:12px;color:#34373F;margin-bottom:4px">Model with all features: R2 ' + ab.base.toFixed(2) + '. Bars to the left mean accuracy drops without the feature.</div>' + ar.map(function(r){ var w = Math.abs(r.d) / mxa * 46 * prog, neg = r.d < 0; return '<div style="display:grid;grid-template-columns:78px 1fr 56px;gap:8px;align-items:center;margin:9px 0"><span style="font-weight:600">' + r.f + '</span><div style="position:relative;height:16px;background:#EDF2F4;border-radius:3px"><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#34373F"></div><div style="position:absolute;top:0;bottom:0;border-radius:3px;background:' + (neg ? C.red : C.bg) + ';' + (neg ? 'right:50%;' : 'left:50%;') + 'width:' + w + '%"></div></div><b style="text-align:right">' + (r.d >= 0 ? '+' : '') + (r.d * prog).toFixed(3) + '</b></div>'; }).join(''); }
    function render(){ if (!activeFeats().length) return; rows(1); }
    return { render: render };
  } });

L2.add(filterStepDef('trees'));

/* ---------- results ---------- */
L2.add({ short: 'Results', title: 'Score and fit time of each tree model',
  change: 'Train a model type in the workbench to replace its card.',
  watch: 'This step compares the three tree models on test R2 and on fit time, the seconds taken to train in your browser. Each card starts with the current options.',
  build: function(api){
    var cards = [0, 1, 2].map(function(k){ return api.el('abs', 'left:' + (28 + k * 307) + 'px;top:28px;width:290px;height:150px', ''); });
    var tp = api.el('pn', 'left:28px;top:196px;width:904px;height:240px', '<h4>Fit time in your browser</h4><div class="tm"></div>');
    function auto(mode){
      var f = activeFeats(); if (!f.length) return null; var p = TP(), a = mode === 'tree' ? tTree() : (mode === 'forest' ? tForest() : tBoost()), pred = mode === 'boost' ? a.F : a.pred;
      return { r2: r2(D.TEST, pred), sp: spearman(D.TEST, pred), tr: r2(D.TRAIN, pred), secs: a.secs, auto: true, params: mode === 'tree' ? 'depth ' + p.depth : (mode === 'forest' ? p.nt + ' trees, depth ' + p.fd : p.rounds + ' rounds, rate ' + p.lr) };
    }
    async function render(){
      var names = ['Decision tree', 'Random forest', 'Gradient boosting'], modes = ['tree', 'forest', 'boost'];
      api.busy('Fitting the three models with the current options'); await tick(); await tick();
      var ms = names.map(function(n, k){ return D.scores[n] || auto(modes[k]); }); api.idle();
      cards.forEach(function(c, k){ var m = ms[k]; c.innerHTML = m ? scoreCard(names[k] + ' (' + m.params + (m.auto ? ', current options' : ', trained') + ')', m, 0) : '<div class="score"><b>' + names[k] + '</b><div class="s" style="margin-top:8px">Switch on a feature.</div></div>'; c.firstChild.style.height = '150px'; });
      var mx = Math.max.apply(null, ms.map(function(m){ return m ? m.secs : 0; })) || 1; qs('.tm', tp).innerHTML = ms.map(function(m, k){ return '<div style="display:grid;grid-template-columns:130px 1fr 80px;gap:12px;align-items:center;margin:12px 0;font-size:13px"><b>' + names[k] + '</b><div class="pbar" style="margin:0;height:16px"><i style="background:' + (k === 2 ? C.red : C.char) + ';width:' + (m ? Math.max(1.5, m.secs / mx * 100) : 0) + '%;transition:none"></i></div><span style="text-align:right;font-variant-numeric:tabular-nums">' + (m ? fmtT(m.secs) : 'n/a') + '</span></div>'; }).join('') + '<p class="note" style="font-size:12px;margin-top:10px">Cards marked current options use the options in the panel. Train a model in the workbench and its card is replaced.</p>'; }
    return { render: render };
  } });

/* ---------- all three tree types side by side ---------- */
L2.add({ short: 'Is it useful', uses: ['endpoint', 'features', 'filters', 'tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr'],
  title: 'All three side by side',
  watch: 'Each plot fits one feature on its own, using the options panel settings, and draws the model\'s prediction as a line through the compounds. The tree predicts in blocks. The forest smooths the blocks. Boosting follows the dots most closely. Outside the range of the data all three go flat, while the dashed straight line keeps going.',
  change: 'Click any plot to compare the three predictions at that point. Pick another feature below the plots.',
  build: function(api){
    var KINDS = [['tree', 'Decision tree', 'Quick and readable. Predicts in blocks and only repeats values it has seen.'], ['forest', 'Random forest', 'Steadier than one tree. Smoother, but still flat outside the data.'], ['boost', 'Gradient boosting', 'Usually the most accurate on tables like this. Fits closely, so watch for overfitting.']];
    var P = KINDS.map(function(k, i){ return api.el('pn', 'left:' + (28 + i * 310) + 'px;top:28px;width:284px;height:296px', '<h4>' + k[1] + '</h4><svg viewBox="0 0 260 46" width="260" height="46"></svg><canvas width="260" height="192"></canvas>'); });
    function icoTree(x0, y0, w, h, depth, col){ var out = '', dy = h / depth; (function rec(x, y, span, d){ if (d === depth) { out += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="2.2" fill="' + col + '"/>'; return; } var ny = y + dy, lx = x - span / 2, rx = x + span / 2; out += '<path d="M' + x.toFixed(1) + ' ' + y.toFixed(1) + 'L' + lx.toFixed(1) + ' ' + ny.toFixed(1) + 'M' + x.toFixed(1) + ' ' + y.toFixed(1) + 'L' + rx.toFixed(1) + ' ' + ny.toFixed(1) + '" stroke="' + col + '" stroke-width="1.3" fill="none"/>'; rec(lx, ny, span / 2, d + 1); rec(rx, ny, span / 2, d + 1); })(x0 + w / 2, y0 + 3, w * 0.46, 0); return out; }
    (function(){ var s0 = icoTree(100, 2, 60, 38, 3, C.red) + '<text x="170" y="26" font-size="11" fill="' + C.mid + '">one tree</text>', s1 = '', s2 = '', i;
      for (i = 0; i < 6; i++) s1 += icoTree(6 + i * 42, 4, 36, 34, 3, C.bg); 
      for (i = 0; i < 4; i++) { s2 += icoTree(8 + i * 62, 4, 40, 34, 2, C.red); if (i < 3) s2 += '<text x="' + (54 + i * 62) + '" y="26" font-size="14" fill="' + C.char + '">+</text>'; } s2 += '<text x="252" y="26" font-size="12" fill="' + C.mid + '" text-anchor="end">...</text>';
      [s0, s1, s2].forEach(function(s, k){ qs('svg', P[k]).innerHTML = s; }); })();
    var Cd = KINDS.map(function(k, i){ return api.el('pn', 'left:' + (28 + i * 310) + 'px;top:336px;width:284px;height:104px;overflow:hidden', '<div class="vc" style="font-size:12px;line-height:1.4"></div>'); });
    var ctl = api.addControl('Feature <select></select>'), sel = qs('select', ctl), M = null, probe = null;
    function fill(){ var f = activeFeats(), cur = sel.value; sel.innerHTML = f.map(function(x){ return '<option>' + x + '</option>'; }).join(''); if (f.indexOf(cur) >= 0) sel.value = cur; }
    function fit(){
      var feat = sel.value, idx = activeTrain(), p = TP(), xs = idx.map(function(i){ return D.COL[feat][i]; }), ys = idx.map(function(i){ return D.Y[i]; }), mn = Infinity, mx = -Infinity; xs.forEach(function(v){ if (v < mn) mn = v; if (v > mx) mx = v; });
      var mxm = mean(xs), mym = mean(ys), cov = 0, vr = 0; xs.forEach(function(v, i){ cov += (v - mxm) * (ys[i] - mym); vr += (v - mxm) * (v - mxm); }); var slope = cov / (vr || 1), span = mx - mn, r = rng(5), pts = []; idx.forEach(function(i){ if (r() < Math.min(1, 300 / idx.length)) pts.push(i); });
      var tree = fitTree([feat], idx, p.depth, 5, 1), fr = fitForest([feat], idx, p.nt, p.fd, 7), bo = fitBoost([feat], idx, p.rounds, p.lr, 3), all = [tTree(), tForest(), tBoost()], preds = [all[0].pred, all[1].pred, all[2].F];
      return { feat: feat, tree: tree, fr: fr, bo: bo, slope: slope, icpt: mym - slope * mxm, mn: mn, mx: mx, lo: mn - 0.4 * span, hi: mx + 0.4 * span, pts: pts, r2: preds.map(function(q){ return r2(D.TEST, q); }), secs: [all[0].secs, all[1].secs, all[2].secs] };
    }
    function val(k, x){ var v = {}; v[M.feat] = x; if (k === 0) return predictVals(M.tree, D.B, v); if (k === 1) { var s = 0; M.fr.trees.forEach(function(t){ s += predictVals(t, D.B, v); }); return s / M.fr.trees.length; } return boostPredictVals(M.bo, D.B, v); }
    function line(x){ return M.slope * x + M.icpt; }
    function draw(){
      if (!M) return; var px = probe === null ? M.hi - 0.1 * (M.hi - M.mx) : probe, cols = [C.char, C.mid, C.red];
      KINDS.forEach(function(kd, k){
        var g = cv(qs('canvas', P[k])), x = g.x, box = { l: 36, r: g.w - 8, t: 6, b: g.h - 40 }, lo = M.lo, hi = M.hi, ylo = D.ylo, yhi = D.yhi; axes(g, box, [lo, hi], [ylo, yhi], M.feat, '', 3);
        var sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - ylo) / (yhi - ylo) * (box.b - box.t); };
        x.fillStyle = 'rgba(52,55,63,.07)'; x.fillRect(box.l, box.t, sx(M.mn) - box.l, box.b - box.t); x.fillRect(sx(M.mx), box.t, box.r - sx(M.mx), box.b - box.t);
        x.fillStyle = 'rgba(161,177,195,.7)'; M.pts.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[M.feat][i]), sy(clamp(D.Y[i], ylo, yhi)), 2.2, 0, 6.3); x.fill(); });
        x.save(); x.beginPath(); x.rect(box.l, box.t, box.r - box.l, box.b - box.t); x.clip();
        x.strokeStyle = C.char; x.globalAlpha = 0.55; x.lineWidth = 1.6; x.setLineDash([6, 4]); x.beginPath(); x.moveTo(sx(lo), sy(line(lo))); x.lineTo(sx(hi), sy(line(hi))); x.stroke(); x.setLineDash([]); x.globalAlpha = 1;
        x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); for (var q = 0; q <= 160; q++) { var xv = lo + (hi - lo) * q / 160, yv = val(k, xv); q ? x.lineTo(sx(xv), sy(yv)) : x.moveTo(sx(xv), sy(yv)); } x.stroke(); x.restore(); x.lineWidth = 1;
        x.strokeStyle = C.char; x.beginPath(); x.moveTo(sx(px), box.t); x.lineTo(sx(px), box.b); x.stroke(); x.fillStyle = C.red; x.strokeStyle = '#fff'; x.lineWidth = 2; x.beginPath(); x.arc(sx(px), sy(clamp(val(k, px), ylo, yhi)), 5.5, 0, 6.3); x.fill(); x.stroke(); x.lineWidth = 1;
        qs('.vc', Cd[k]).innerHTML = '<div style="display:flex;justify-content:space-between;align-items:baseline"><span><b style="font-size:18px;color:' + C.red + '">' + M.r2[k].toFixed(2) + '</b> <span class="note" style="font-size:11px">test R2, all features</span></span><span class="note" style="font-size:11px">fit ' + fmtT(M.secs[k]) + '</span></div><div style="margin:2px 0">At ' + esc(M.feat) + ' ' + fmtFeat(M.feat, px) + ': predicts <b>' + val(k, px).toFixed(2) + '</b></div><div class="note" style="font-size:11.5px">' + kd[2] + '</div>';
      });
    }
    P.forEach(function(pnl, k){ var cvEl = qs('canvas', pnl); cvEl.style.cursor = 'crosshair'; cvEl.addEventListener('pointerdown', function(e){ if (!M) return; var r = cvEl.getBoundingClientRect(), x0 = (e.clientX - r.left) / r.width * 260; probe = clamp(M.lo + (x0 - 36) / (260 - 8 - 36) * (M.hi - M.lo), M.lo, M.hi); draw(); }); });
    sel.addEventListener('change', function(){ probe = null; M = fit(); draw(); });
    function render(){ fill(); if (!sel.value) return; M = fit(); draw(); }
    return { render: render };
  } });

L2.add(seedStepDef('trees'));

L2.glossary([
  ['Feature', 'A number that describes a compound, such as logP or molecular weight.'],
  ['Decision tree', 'A flowchart of yes or no questions about a compound. The answers send it to a group, and the average of that group is the prediction.'],
  ['Split', 'One question in the tree, such as "is logP at most 2.9?". It divides a group in two.'], ['Depth', 'How many questions in a row a tree may ask.'],
  ['Leaf', 'A box at the bottom of a tree. Its prediction is the average of the training compounds that ended up in it.'],
  ['Miss score', 'How far compounds sit from their group\'s average, on average, with big misses counting extra. Lower is better.'],
  ['Random forest', 'Many trees, each trained on a different random draw of the compounds. The forest answers with the average of its trees.'],
  ['Ensemble', 'Several models combined into one prediction.'],
  ['Gradient boosting', 'Trees built one after another, each trained on the mistakes the earlier ones left.'],
  ['Learning rate', 'How strongly each new tree\'s correction is added. Small means slow and careful.'],
  ['Training R2', 'The score on the compounds the model learned from. It can look excellent even when the model is memorising.'],
  ['Test R2', 'The score on compounds held back from training. This is the score that matters.'],
  ['Feature importance', 'How much a feature helped the model. The answer depends on how it is measured.'],
  ['Ablation', 'Training again without one feature to see how much the score falls.']
]);

L2.move('Workbench', 1);
setUses(L2, [['endpoint', 'features', 'filters', 'tdepth'], null, ['endpoint', 'features', 'filters', 'tdepth'], ['endpoint', 'features', 'filters', 'tntrees', 'tfdepth'], ['endpoint', 'features', 'filters', 'trounds', 'tlr'], ['endpoint', 'features', 'filters'], null, ['endpoint', 'features', 'filters', 'tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr'], null, null]);
