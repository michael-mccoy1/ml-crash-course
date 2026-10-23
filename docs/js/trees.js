"use strict";
/* Lesson 3: Tree models */
var L2 = new Lesson('trees', 'Tree models');
L2.optsHTML = function(){
  return sharedOptions() +
    optRange('tdepth', 'Single tree depth', 1, 14, 1, 4, 'Questions in a row for the single tree.') +
    optRange('tntrees', 'Forest: number of trees', 10, 200, 10, 60, 'Trees that vote in the forest.') +
    optRange('tfdepth', 'Forest tree depth', 3, 16, 1, 10, 'Depth of each tree in the forest.') +
    optRange('trounds', 'Boosting rounds', 20, 400, 10, 150, 'Small trees added one after another.') +
    optRange('tlr', 'Boosting learning rate', 0.01, 0.3, 0.01, 0.08, 'Share of each tree\'s correction that is applied.') +
    '<div class="og"><button type="button" class="btn wide" id="ttrain">Train all three</button><div class="status" id="tst"></div><div class="rbox" id="tres"></div></div>';
};
function TP(){ return { depth: +$('tdepth').value, nt: +$('tntrees').value, fd: +$('tfdepth').value, rounds: +$('trounds').value, lr: +$('tlr').value }; }
function tCache(name, extra, fn){ D.cache.t = D.cache.t || {}; var k = name + JSON.stringify([EP, activeFeats(), activeTrain().length, filterOn() ? FILT : 0, extra]); if (!D.cache.t[k]) D.cache.t[k] = fn(); return D.cache.t[k]; }
function tTree(){ var p = TP(); return tCache('tree', p.depth, function(){ var f = activeFeats(), idx = activeTrain(), t0 = performance.now(), tree = fitTree(f, idx, p.depth, 5, 1), secs = (performance.now() - t0) / 1000, pred = treeAll(tree, D.N);
  var sweep = tCache('sweep', 0, function(){ var out = []; for (var d = 1; d <= 14; d++) { var pr = treeAll(fitTree(f, idx, d, 5, 1), D.N); out.push([d, r2(D.TRAIN, pr), r2(D.TEST, pr)]); } return out; });
  return { tree: tree, pred: pred, secs: secs, show: fitTree(f, idx, Math.min(3, p.depth), 20, 1), sweep: sweep }; }); }
function tForest(){ var p = TP(); return tCache('forest', [p.nt, p.fd], function(){ var fr = fitForest(activeFeats(), activeTrain(), p.nt, p.fd, 7); fr.curve = forestCurve(fr, Math.max(1, Math.round(p.nt / 40))); return fr; }); }
function tForestDepth(){ return tCache('fdepth', 0, function(){ return [3, 6, 10, 14].map(function(d){ var fr = fitForest(activeFeats(), activeTrain(), 30, d, 7); return { d: d, te: r2(D.TEST, fr.pred), tr: r2(D.TRAIN, fr.pred) }; }); }); }
function tBoost(){ var p = TP(); return tCache('boost', [p.rounds, p.lr], function(){ return fitBoost(activeFeats(), activeTrain(), p.rounds, p.lr, 3, Math.max(1, Math.round(p.rounds / 40))); }); }
function tBoostLr(){ var p = TP(); return tCache('boostlr', p.rounds, function(){ return [0.02, 0.08, 0.3].map(function(lr){ return { lr: lr, curve: fitBoost(activeFeats(), activeTrain(), p.rounds, lr, 3, Math.max(1, Math.round(p.rounds / 40))).curve }; }); }); }
function tAblation(){ return tCache('abl', 0, function(){ var f = activeFeats(), idx = activeTrain(), full = fitBoost(f, idx, 60, 0.1, 3), base = r2(D.TEST, full.F); return { base: base, rows: f.map(function(x){ if (f.length < 2) return { f: x, d: 0 }; var m = fitBoost(f.filter(function(y){ return y !== x; }), idx, 60, 0.1, 3); return { f: x, d: r2(D.TEST, m.F) - base }; }) }; }); }
function recordTrees(){
  var p = TP(), a = tTree(), b = tForest(), c = tBoost(), f = activeFeats(); if (!f.length) return;
  var names = ['Decision tree', 'Random forest', 'Gradient boosting']; var ms = [record(names[0], f, a.pred, a.secs), record(names[1], f, b.pred, b.secs), record(names[2], f, c.F, c.secs)];
  ms[0].params = 'depth ' + p.depth; ms[1].params = p.nt + ' trees, depth ' + p.fd; ms[2].params = p.rounds + ' rounds, rate ' + p.lr; return ms;
}
function tResultsBox(){ var el = $('tres'); if (!el) return; var ms = ['Decision tree', 'Random forest', 'Gradient boosting'].map(function(n){ return D.scores[n]; }).filter(Boolean); el.innerHTML = ms.map(function(m, i){ return '<div class="rrow">' + ['Tree', 'Forest', 'Boosting'][i] + '<b>R2 ' + m.r2.toFixed(2) + ', ' + fmtT(m.secs) + '</b></div>'; }).join(''); }
var tTimer = null;
L2.onOpts = function(root){
  var fire = function(){ clearTimeout(tTimer); tTimer = setTimeout(function(){ L2.refresh(); }, 220); };
  ['tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr'].forEach(function(id){ $(id).addEventListener('input', fire); });
  $('ttrain').addEventListener('click', async function(){ var b = this; b.disabled = true; var st = $('tst'); if (!activeFeats().length) { st.textContent = 'Tick at least one feature.'; b.disabled = false; return; } st.textContent = 'Training'; progress(0.2); await tick(); recordTrees(); progress(0.7); await tick(); tBoostLr(); tResultsBox(); st.textContent = 'Done. Scores are in the Compare lesson.'; progress(1); b.disabled = false; L2.refresh(); });
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

/* ---------- T1: how a tree chooses a question ---------- */
L2.add({ short: 'Choosing a question', title: 'A tree picks the question that separates the answers best',
  change: 'Press Play. Choose a different feature below. Drag Single tree depth in the options to refine the red staircase.',
  watch: 'The line sweeps across the feature. Each position is scored by how well it splits low from high values. The best position becomes the first question.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:580px;height:408px', '<h4 class="t">Training compounds</h4><canvas width="560" height="352"></canvas>');
    var gp = api.el('pn', 'left:628px;top:28px;width:304px;height:196px', '<h4>How good is each cut</h4><canvas width="284" height="146"></canvas>');
    var res = api.el('pn', 'left:628px;top:236px;width:304px;height:200px', '<div class="rs" style="font-size:13px;color:#4a4f5a">Press Play.</div>');
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
      draw(n - 1, true); show(); api.cap('The cut with the best score wins. This becomes the first question of the tree.'); await api.sleep(3400); api.key('A tree is built by repeating this search on every group until it reaches the depth limit.'); }
    return { render: render, play: play };
  } });

/* ---------- T2: depth ---------- */
L2.add({ short: 'Depth', title: 'Depth is how many questions in a row a tree may ask',
  change: 'Drag Single tree depth in the options, or press Play to step through depths.',
  watch: 'The tree grows one level at a time. The chart shows the studied and unseen R2 at each depth.',
  build: function(api){
    var tp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Decision tree, top three levels</h4><svg viewBox="0 0 580 330" width="580" height="330"></svg><div class="note tn" style="font-size:12px;color:#6B7280;margin-top:2px"></div>');
    var cp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<h4>R2 by depth</h4><canvas width="262" height="250"></canvas><div class="cn" style="font-size:12.5px;margin-top:6px"></div><div class="legend" style="font-size:11px;margin-top:4px"><span style="color:' + C.char + '">dark: studied</span> <span style="color:' + C.red + '">red: unseen</span></div>');
    var ctl = api.addControl('Depth <input type="range" min="1" max="14" step="1" value="4" style="width:200px"><b class="dv">4</b>'); var dr = qs('input', ctl);
    function draw(){ var t = tTree(), p = TP(); var out = treeSVG(t.show, D.B, Math.min(3, p.depth), 580, 330); qs('svg', tp).innerHTML = out.svg; qs('.tn', tp).textContent = p.depth > 3 ? 'The top 3 of ' + p.depth + ' levels are drawn. The bottom boxes keep splitting.' : 'All ' + p.depth + ' level' + (p.depth > 1 ? 's' : '') + ' are drawn.';
      var sw = t.sweep, g = lineChart(qs('canvas', cp), [{ pts: sw.map(function(r){ return [r[0], r[1]]; }), color: C.char, nodot: true }, { pts: sw.map(function(r){ return [r[0], r[2]]; }), color: C.red, nodot: true }], { xl: 'tree depth', yl: 'R2', y1: 1, y0: Math.min(0, Math.min.apply(null, sw.map(function(r){ return r[2]; }))), x0: 1, x1: 14, xdec: 0 });
      var x = g.x, X = g.sx(p.depth); x.strokeStyle = C.mid; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(X, g.box.t); x.lineTo(X, g.box.b); x.stroke(); x.setLineDash([]); var row = sw[p.depth - 1]; qs('.cn', cp).innerHTML = 'Depth ' + p.depth + ': studied R2 <b>' + row[1].toFixed(2) + '</b>, unseen R2 <b style="color:' + C.red + '">' + row[2].toFixed(2) + '</b>'; qs('.dv', ctl).textContent = p.depth; dr.value = p.depth; }
    dr.addEventListener('input', function(){ api.tok++; api.clearTexts(); $('tdepth').value = dr.value; $('o-tdepth').textContent = dr.value; draw(); });
    function render(){ draw(); }
    async function play(tok){ var best = tTree().sweep.reduce(function(a, b){ return b[2] > a[2] ? b : a; }, [1, 0, -9]);
      for (var d = 1; d <= 10; d++) { $('tdepth').value = d; $('o-tdepth').textContent = d; draw(); if (d === 1) api.cap('Depth 1: one question, two groups.'); if (d === 2) api.cap('Depth 2: each group is split again.'); if (d === 3) api.cap('Depth 3: eight groups. Boxes are coloured by their average.'); if (d === best[0]) api.cap('Around depth ' + d + ' the unseen R2 peaks at ' + best[2].toFixed(2) + '.'); if (d === 8) api.cap('Deeper trees keep fitting the studied compounds better, but unseen R2 stops following.'); await api.sleep(d < 4 ? 1900 : 1300); }
      api.key('Choose the depth where the unseen score peaks, never the depth where the studied score is highest.'); }
    return { render: render, play: play };
  } });

/* ---------- T3: send a compound down the tree ---------- */
L2.add({ short: 'One compound', title: 'Follow one compound down the tree',
  change: 'Press Play, then Next compound.',
  watch: 'At every box the compound answers yes or no using its own feature value, until it lands in a leaf and receives that leaf\'s average.',
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
      api.cap('The compound lands in a leaf and gets that leaf\'s average as its prediction.'); await api.sleep(2800); api.key('A tree prediction is just the average of the training compounds that share the same answers to the questions.'); }
    return { render: render, play: play };
  } });
function fixFeat(f, v){ return fmtFeat(f, v); }

/* ---------- T4: forest ---------- */
L2.add({ short: 'Forest', title: 'A forest averages many different trees',
  change: 'Drag Forest: number of trees and Forest tree depth in the options.',
  watch: 'Each dot is one tree\'s prediction for the same compound. The red line is their average. The chart shows how unseen R2 improves as trees are added.',
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
    async function play(tok){ await setup(false); var fr = tForest(), n = fr.per.length; drawDepth(); vote(0, 1); drawCurve(0); await api.beat('Each tree gives its own prediction for this compound. They disagree.', 2800);
      var step = Math.max(1, Math.ceil(n / 50)); for (var k = step; k < n + step; k += step) { var kk = Math.min(k, n); await api.anim(ANIM ? 90 : 1, function(t){ vote(kk, t); }); drawCurve(kk); }
      var m = mean(fr.trees.map(function(t){ return predictVals(t, D.B, featVals(cur.i)); })); api.cap('Their average is ' + m.toFixed(2) + '. Averaging cancels out errors that individual trees make, and the unseen R2 climbs.'); await api.sleep(3600); api.cap('Deeper trees fit the studied compounds better, but the unseen R2 gains little. More trees help more than more depth.'); await api.sleep(3600); api.key('A forest is many noisy trees whose average is more accurate than any one of them.'); }
    return { render: render, play: play };
  } });

/* ---------- T5: boosting ---------- */
L2.add({ short: 'Boosting', title: 'Boosting adds small trees that fix the remaining errors',
  change: 'Drag Boosting rounds and Boosting learning rate in the options.',
  watch: 'Dots start at one value and move toward the diagonal round by round. The chart compares three learning rates with the one you chose.',
  build: function(api){
    var sp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px', '<h4>Predicted against measured, unseen compounds</h4><canvas width="398" height="352"></canvas>');
    var cp = api.el('pn', 'left:468px;top:28px;width:464px;height:300px', '<h4>Unseen R2 against rounds</h4><canvas width="442" height="250"></canvas>');
    var tp = api.el('pn', 'left:468px;top:340px;width:464px;height:96px', '<div class="bt" style="font-size:12.5px;color:#4a4f5a"></div>');
    function curves(upto){ var p = TP(), b = tBoost(), lrs = tBoostLr(), series = lrs.map(function(l, i){ return { pts: l.curve.filter(function(c){ return c.k <= upto; }).map(function(c){ return [c.k, c.te]; }), color: [C.bg, C.mid, C.tint][i], w: 1.8, dash: [4, 3], nodot: true, label: 'rate ' + l.lr }; });
      series.push({ pts: b.curve.filter(function(c){ return c.k <= upto; }).map(function(c){ return [c.k, c.te]; }), color: C.red, label: 'yours: ' + p.lr }); lineChart(qs('canvas', cp), series, { xl: 'boosting round', yl: 'R2', x0: 0, x1: p.rounds, y0: 0, xdec: 0 }); }
    function frame(k){ var b = tBoost(); var s = b.snaps[Math.min(k, b.snaps.length - 1)]; scatterPV(qs('canvas', sp), s.p, D.TEST, 'round ' + s.r + ', R2 ' + r2(D.TEST, s.p).toFixed(2)); }
    function render(){ var b = tBoost(), p = TP(); frame(b.snaps.length - 1); curves(p.rounds); qs('.bt', tp).innerHTML = 'Each round fits a small tree to the errors left by all earlier rounds, then adds a fraction of it (the learning rate). A small rate learns slowly and carefully. A large rate learns fast and can overshoot.'; }
    async function play(tok){ var b = tBoost(), p = TP(); frame(0); curves(0); await api.beat('Round 0: every compound is predicted as the training average, so all dots sit on one horizontal line.', 3000); api.cap('Each round adds one small tree that corrects the errors that remain.');
      for (var k = 0; k < b.snaps.length; k++) { frame(k); curves(b.snaps[k].r); await api.sleep(k < 12 ? 150 : 90); } api.cap('Learning rate ' + p.lr + ' after ' + p.rounds + ' rounds. The dashed lines show what other rates would do with the same number of rounds.'); await api.sleep(3600); api.key('Boosting is slow, steady error correction. The learning rate sets the step size.'); }
    return { render: render, play: play };
  } });

/* ---------- T6: which features matter ---------- */
L2.add({ short: 'Which features matter', title: 'Two ways to ask which features the model uses',
  change: 'Untick features in the options, or press Play.',
  watch: 'Left: how much each feature helped the trees split. Right: how much the unseen R2 drops when the model is retrained without that feature.',
  build: function(api){
    var lp = api.el('pn', 'left:28px;top:28px;width:440px;height:408px', '<h4>Split gain inside the boosted trees</h4><div class="ib"></div>');
    var rp = api.el('pn', 'left:488px;top:28px;width:444px;height:408px', '<h4>Change in unseen R2 when a feature is removed</h4><div class="ab"></div>');
    function rows(prog){ var b = tBoost(), ab = tAblation(), tot = Object.keys(b.gain).reduce(function(a, k){ return a + b.gain[k]; }, 0), f = activeFeats(), imp = f.map(function(x){ return [x, (b.gain[x] || 0) / (tot || 1)]; }).sort(function(a, c){ return c[1] - a[1]; }), mx = Math.max.apply(null, imp.map(function(r){ return r[1]; })) || 1;
      qs('.ib', lp).innerHTML = imp.map(function(r){ return '<div style="display:grid;grid-template-columns:78px 1fr 44px;gap:8px;align-items:center;margin:9px 0"><span style="font-weight:600">' + r[0] + '</span><div class="pbar" style="margin:0;height:16px"><i style="background:' + C.red + ';width:' + (r[1] / mx * 100 * prog) + '%;transition:none"></i></div><b style="text-align:right">' + Math.round(r[1] * 100 * prog) + '%</b></div>'; }).join('');
      var ar = ab.rows.slice().sort(function(a, c){ return a.d - c.d; }), mxa = Math.max(0.01, Math.max.apply(null, ar.map(function(r){ return Math.abs(r.d); })));
      qs('.ab', rp).innerHTML = '<div style="font-size:12px;color:#6B7280;margin-bottom:4px">Model with all features: R2 ' + ab.base.toFixed(2) + '. Bars to the left mean accuracy drops without the feature.</div>' + ar.map(function(r){ var w = Math.abs(r.d) / mxa * 46 * prog, neg = r.d < 0; return '<div style="display:grid;grid-template-columns:78px 1fr 56px;gap:8px;align-items:center;margin:9px 0"><span style="font-weight:600">' + r.f + '</span><div style="position:relative;height:16px;background:#EDF2F3;border-radius:3px"><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#6B7280"></div><div style="position:absolute;top:0;bottom:0;border-radius:3px;background:' + (neg ? C.red : C.bg) + ';' + (neg ? 'right:50%;' : 'left:50%;') + 'width:' + w + '%"></div></div><b style="text-align:right">' + (r.d >= 0 ? '+' : '') + (r.d * prog).toFixed(3) + '</b></div>'; }).join(''); }
    function render(){ if (!activeFeats().length) return; rows(1); }
    async function play(tok){ if (!activeFeats().length) return; rows(0); await api.beat('First view: which features did the trees choose most often, and how much did each split help?', 3000); await api.anim(2600, function(t){ rows(t); });
      api.cap('Second view: retrain the model without one feature at a time and see how much the unseen R2 falls.'); await api.sleep(2400); api.cap('Features that carry information no other feature has hurt the most when removed. Correlated features can hide each other.'); await api.sleep(3600);
      api.key('Importance depends on the question. Check both views before saying which feature drives clearance.'); }
    return { render: render, play: play };
  } });

/* ---------- T7: results ---------- */
L2.add({ short: 'Results', title: 'Score and training time of the three tree models',
  change: 'Press Train all three in the options after changing anything.',
  watch: 'Accuracy on unseen compounds against the time it took to fit each model.',
  build: function(api){
    var cards = [0, 1, 2].map(function(k){ return api.el('abs fd', 'left:' + (28 + k * 307) + 'px;top:28px;width:290px;height:150px', ''); });
    var tp = api.el('pn', 'left:28px;top:196px;width:904px;height:240px', '<h4>Training time in your browser</h4><div class="tm"></div>');
    function draw(on){ var ms = recordTrees(); if (!ms) return; tResultsBox(); cards.forEach(function(c, k){ c.innerHTML = scoreCard(['Decision tree', 'Random forest', 'Gradient boosting'][k] + ' (' + ms[k].params + ')', ms[k], 0); c.firstChild.style.height = '150px'; if (on) setTimeout(function(){ c.classList.add('on'); }, ANIM ? k * 250 : 0); else c.classList.add('on'); });
      var mx = Math.max.apply(null, ms.map(function(m){ return m.secs; })) || 1; qs('.tm', tp).innerHTML = ms.map(function(m, k){ return '<div style="display:grid;grid-template-columns:130px 1fr 80px;gap:12px;align-items:center;margin:12px 0;font-size:13px"><b>' + ['Decision tree', 'Random forest', 'Gradient boosting'][k] + '</b><div class="pbar" style="margin:0;height:16px"><i style="background:' + (k === 2 ? C.red : C.char) + ';width:' + Math.max(1.5, m.secs / mx * 100) + '%"></i></div><span style="text-align:right;font-variant-numeric:tabular-nums">' + fmtT(m.secs) + '</span></div>'; }).join('') + '<p class="note" style="font-size:12px;margin-top:10px">A single tree is nearly instant. Forests and boosting cost more because they fit many trees. Compare with neural networks and the GPT in the later lessons.</p>'; }
    function render(){ cards.forEach(function(c){ c.classList.remove('on'); }); draw(false); }
    async function play(tok){ cards.forEach(function(c){ c.classList.remove('on'); }); await api.sleep(300); draw(true); await api.sleep(1600); api.cap('Compare R2 on unseen compounds first, then ask whether the extra training time bought accuracy.'); await api.sleep(3200); api.key('Ensembles usually beat a single tree. Check the unseen score before paying for more trees.'); }
    return { render: render, play: play };
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

setUses(L2, [['tdepth', 'features', 'endpoint'], ['tdepth', 'features', 'endpoint'], ['tdepth', 'features', 'endpoint'], ['tntrees', 'tfdepth', 'features', 'endpoint'], ['trounds', 'tlr', 'features', 'endpoint'], ['features', 'endpoint'], ['tdepth', 'tntrees', 'tfdepth', 'trounds', 'tlr', 'features', 'endpoint']]);
