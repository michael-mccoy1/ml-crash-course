"use strict";
/* Beginner steps for lessons 02 to 06. Written for chemists with no machine learning experience. */

/* ---------- 02 Tree models: twenty questions ---------- */
LESSONS.trees.add({ level: 'beginner', short: 'Twenty questions', title: 'Decision tree structure',
  change: 'Drag Tree depth, or select Walk through.',
  watch: 'A decision tree predicts by routing a compound through a sequence of threshold questions on its features, such as logP above 3. Each answer sends the compound to the left or right branch. Tree depth is the maximum number of questions in a row. Compounds that reach the same leaf form a group, and the prediction for each of them is the mean measured value of the training compounds in that group, shown as the red line. Each dot is one training compound, and its height is the measured value. As depth increases, the groups become more homogeneous.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:904px;height:360px', '<canvas width="880" height="336"></canvas>');
    var tp = api.el('pn', 'left:28px;top:396px;width:904px;height:44px;padding-top:10px', '<div class="st" style="font-size:13px;color:' + C.char + '"></div>');
    var ctl = api.addControl('Tree depth <input type="range" min="0" max="3" step="1" value="2" style="width:200px"> <b class="qv">2</b>');
    var rg = qs('input', ctl), qv = qs('.qv', ctl), FE = ['logP', 'TPSA', 'MW'];
    function sample(){ var key = EP + 'tw'; if (D.cache[key]) return D.cache[key]; var r = rng(5), sh = D.TRAIN.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; } return (D.cache[key] = sh.slice(0, 260)); }
    function leaves(node, path, out){ if (node.f === undefined) { out.push({ node: node, path: path }); return out; } var thr = D.B.edges[node.f][node.b], w = fmtFeat(node.f, thr); leaves(node.l, path.concat(node.f + ' at most ' + w), out); leaves(node.r, path.concat(node.f + ' above ' + w), out); return out; }
    function draw(){
      var d = +rg.value, feats = FE.filter(function(f){ return D.COL[f]; }), tree = fitTree(feats, D.TRAIN, d, 25, 1), lv = leaves(tree, [], []), g = cv(qs('canvas', cp)), x = g.x, n = lv.length, w = (g.w - 60) / n, top = 74, bot = g.h - 26, ys = function(v){ return bot - (v - D.ylo) / (D.yhi - D.ylo) * (bot - top); };
      qv.textContent = d; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText('measured', 44, top - 8);
      var map = new Map(); lv.forEach(function(l, k){ map.set(l.node, k); l.pts = []; });
      sample().forEach(function(i){ var nd = tree; while (nd.f !== undefined) nd = D.B.codes[nd.f][i] <= nd.b ? nd.l : nd.r; lv[map.get(nd)].pts.push(i); });
      lv.forEach(function(l, k){
        var x0 = 52 + k * w; x.fillStyle = k % 2 ? 'rgba(237,242,244,.9)' : 'rgba(255,255,255,0)'; x.fillRect(x0, top - 6, w, bot - top + 12);
        var rj = rng(k + 9); x.fillStyle = 'rgba(52,55,63,.55)'; l.pts.forEach(function(i){ x.beginPath(); x.arc(x0 + 8 + rj() * (w - 16), ys(D.Y[i]), 2.6, 0, 6.3); x.fill(); });
        x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); x.moveTo(x0 + 4, ys(l.node.v)); x.lineTo(x0 + w - 4, ys(l.node.v)); x.stroke();
        x.fillStyle = C.red; x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText(l.node.v.toFixed(2), x0 + w / 2, ys(l.node.v) - 6);
        x.fillStyle = C.char; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; if (!l.path.length) x.fillText('all compounds', x0 + w / 2, 18); l.path.forEach(function(p, q){ x.fillText(p, x0 + w / 2, 18 + q * 13); });
        x.fillStyle = C.mid; x.fillText(l.node.n + ' compounds', x0 + w / 2, g.h - 8);
      });
      x.strokeStyle = C.silver; x.lineWidth = 1; x.beginPath(); x.moveTo(52, top - 6); x.lineTo(52, bot + 6); x.stroke();
      var te = r2(D.TEST, treeAll(tree, D.N)); qs('.st', tp).innerHTML = 'Depth <b>' + d + '</b>, <b>' + n + '</b> leaf' + (n === 1 ? '' : 'ves') + '. Test R2: <b style="color:' + C.red + '">' + te.toFixed(2) + '</b> (1 is a perfect fit, 0 equals predicting the training mean).';
    }
    rg.addEventListener('input', draw);
    async function play(tok){ rg.value = 0; draw(); api.cap('Depth 0: every compound receives the training mean.'); await api.sleep(3000); for (var k = 1; k <= 3; k++) { api.checkTok(tok); rg.value = k; draw(); api.cap(k === 1 ? 'Depth 1: one question splits the compounds into two groups.' : (k === 2 ? 'Depth 2: each group is split again.' : 'Depth 3: the groups are more homogeneous and the group means follow the data more closely.')); await api.sleep(3000); } api.cap(null); }
    return { render: draw, play: play };
  } });

/* ---------- 03 Neural networks: linear neuron ---------- */
LESSONS.nn.add({ level: 'beginner', short: 'A dial and a sum', title: 'Linear neuron',
  change: 'Adjust the two weights and the bias, or select Fit.',
  watch: 'The basic unit of a neural network is the neuron. A neuron computes a weighted sum of its inputs and adds a bias: prediction = w1 x1 + w2 x2 + b. The weights and the bias are the parameters that training adjusts. Here one neuron takes two standardised features, logP and TPSA, and predicts the measured value. Each point is a training compound, plotted as predicted against measured value, and points on the dashed line are predicted exactly. Training finds the parameters that minimise the mean squared error between prediction and measurement. A full network has thousands of such parameters, adjusted by an optimiser.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:500px;height:408px', '<h4>Predicted against measured, 260 training compounds</h4><canvas width="470" height="350"></canvas>');
    var sp = api.el('pn', 'left:548px;top:28px;width:384px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Weight, logP <input type="range" data-d="0" min="-1" max="1" step="0.01" value="0" style="width:120px"> Weight, TPSA <input type="range" data-d="1" min="-1" max="1" step="0.01" value="0" style="width:120px"> Bias <input type="range" data-d="2" min="-1" max="1" step="0.01" value="0" style="width:120px"> <button type="button" class="btn sm" data-tune>Fit</button> <button type="button" class="btn sm ghost" data-zero>Reset</button>');
    var rs = qsa('[data-d]', ctl), S = null;
    function prep(){ var key = EP + 'nb'; if (D.cache[key]) return D.cache[key]; var r = rng(8), sh = D.TRAIN.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; } var idx = sh.slice(0, 260), a1 = idx.map(function(i){ return D.COL.logP[i]; }), a2 = idx.map(function(i){ return D.COL.TPSA[i]; }), ys = idx.map(function(i){ return D.Y[i]; }), m1 = mean(a1), m2 = mean(a2), my = mean(ys), s1 = Math.sqrt(mean(a1.map(function(v){ return (v - m1) * (v - m1); }))) || 1, s2 = Math.sqrt(mean(a2.map(function(v){ return (v - m2) * (v - m2); }))) || 1, sy = Math.sqrt(mean(ys.map(function(v){ return (v - my) * (v - my); }))) || 1;
      var X = idx.map(function(i, k){ return [(a1[k] - m1) / s1, (a2[k] - m2) / s2, (ys[k] - my) / sy]; });
      /* least-squares fit of the two weights and the bias */
      var A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bv = [0, 0, 0]; X.forEach(function(p){ var v = [p[0], p[1], 1]; for (var u = 0; u < 3; u++) { bv[u] += v[u] * p[2]; for (var w = 0; w < 3; w++) A[u][w] += v[u] * v[w]; } });
      var M = A.map(function(row, u){ return row.concat([bv[u]]); }); for (var c = 0; c < 3; c++) { var pv = c; for (var rr = c + 1; rr < 3; rr++) if (Math.abs(M[rr][c]) > Math.abs(M[pv][c])) pv = rr; var tmp = M[c]; M[c] = M[pv]; M[pv] = tmp; for (var r2_ = 0; r2_ < 3; r2_++) if (r2_ !== c) { var f = M[r2_][c] / M[c][c]; for (var k2 = c; k2 < 4; k2++) M[r2_][k2] -= f * M[c][k2]; } }
      var best = [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]].map(function(v){ return clamp(v, -1, 1); });
      return (D.cache[key] = { idx: idx, X: X, my: my, sy: sy, best: best });
    }
    function draw(){
      S = prep(); var d = rs.map(function(r){ return +r.value; }), p = new Float64Array(D.N), se = 0;
      S.idx.forEach(function(i, k){ var z = d[0] * S.X[k][0] + d[1] * S.X[k][1] + d[2], v = S.my + S.sy * z; p[i] = v; se += (v - D.Y[i]) * (v - D.Y[i]); });
      var rm = Math.sqrt(se / S.idx.length), bd = S.best, sb = 0; S.idx.forEach(function(i, k){ var v = S.my + S.sy * (bd[0] * S.X[k][0] + bd[1] * S.X[k][1] + bd[2]); sb += (v - D.Y[i]) * (v - D.Y[i]); }); var rb = Math.sqrt(sb / S.idx.length);
      scatterPV(qs('canvas', cp), p, S.idx, '', { max: 260 });
      qs('.nb', sp).innerHTML = '<div class="ol">Current model</div><p style="font-size:14px">prediction = <b>' + d[0].toFixed(2) + '</b> logP + <b>' + d[1].toFixed(2) + '</b> TPSA + <b>' + d[2].toFixed(2) + '</b></p><p style="font-size:12px;color:' + C.mid + '">The features are standardised, so 0 is the mean and 1 is one standard deviation. The sum is rescaled to the units of the target.</p><div class="ol" style="margin-top:12px">RMSE</div><div class="big" style="color:' + C.red + '">' + rm.toFixed(2) + '</div><p style="font-size:12px;margin-top:2px">in ' + esc(unitWord()) + ' units. Lower is better.</p><p style="margin-top:10px">The least-squares fit of these three parameters has an RMSE of <b>' + rb.toFixed(2) + '</b>. ' + (rm < rb + 0.01 ? 'The current parameters match it.' : 'Select Fit to apply the least-squares values.') + '</p>';
    }
    rs.forEach(function(r){ r.addEventListener('input', draw); });
    qs('[data-zero]', ctl).addEventListener('click', function(){ rs.forEach(function(r){ r.value = 0; }); draw(); });
    qs('[data-tune]', ctl).addEventListener('click', function(){ S = prep(); var from = rs.map(function(r){ return +r.value; }); animateFree(1600, function(t){ rs.forEach(function(r, k){ r.value = from[k] + (S.best[k] - from[k]) * t; }); draw(); }); });
    async function play(tok){ rs.forEach(function(r){ r.value = 0; }); draw(); api.cap('With every weight and the bias at zero, the neuron predicts the same value for every compound.'); await api.sleep(3000); api.cap('The optimiser adjusts the parameters to bring the points toward the dashed line.'); S = prep(); await api.anim(2600, function(t){ rs.forEach(function(r, k){ r.value = S.best[k] * t; }); draw(); }); api.cap('Training is this search for parameters that minimise the error.'); await api.sleep(3000); api.cap(null); }
    return { render: draw, play: play };
  } });

/* ---------- 04 GPT: autocomplete for molecules ---------- */
var BG = { ready: false, c1: {}, c2: {}, n: 0, toks: 0 };
function bgBuild(){
  if (BG.ready) return; var all = (STRUCTS.HLM || []).concat(STRUCTS.logD || []); BG.n = all.length;
  all.forEach(function(e){ var t = ['^', '^'].concat(e.smi.match(TOK) || []).concat(['$']); for (var i = 2; i < t.length; i++) { var k1 = t[i - 1], k2 = t[i - 2] + ' ' + t[i - 1]; (BG.c1[k1] = BG.c1[k1] || {})[t[i]] = ((BG.c1[k1] || {})[t[i]] || 0) + 1; (BG.c2[k2] = BG.c2[k2] || {})[t[i]] = ((BG.c2[k2] || {})[t[i]] || 0) + 1; BG.toks++; } });
  BG.ready = true;
}
function bgNext(ctx){ var a = ctx[ctx.length - 2] || '^', b = ctx[ctx.length - 1] || '^', t2 = BG.c2[a + ' ' + b], tot2 = t2 ? Object.keys(t2).reduce(function(s, k){ return s + t2[k]; }, 0) : 0, use = tot2 >= 4 ? t2 : BG.c1[b], tot = use ? Object.keys(use).reduce(function(s, k){ return s + use[k]; }, 0) : 0; if (!use) return { list: [], total: 0, used: 0 };
  return { list: Object.keys(use).map(function(k){ return [k, use[k] / tot, use[k]]; }).sort(function(p, q){ return q[1] - p[1]; }), total: tot, used: use === t2 ? 2 : 1 }; }
LESSONS.gpt.add({ level: 'beginner', short: 'Autocomplete for molecules', title: 'Next-token prediction',
  change: 'Select a token to append it, or select Sample. Select Reset to start again.',
  watch: 'A GPT is a language model: it estimates the probability of the next token given the tokens before it. Here a molecule is a SMILES string split into tokens (atoms, bonds, brackets and ring digits). The model is an n-gram model. It counts which token follows the previous two tokens in 180 structures from the course data sets and converts the counts to probabilities. Select a token to append it, or sample the next token in proportion to the probabilities. Choosing the end token completes the string. A GPT replaces the counts with a neural network that conditions on all earlier tokens, not only the last two.',
  build: function(api){
    var ctx = [], done = false;
    var tp = api.el('pn', 'left:28px;top:28px;width:904px;height:96px', '<h4>SMILES so far</h4><div class="sm" style="font:600 18px Cascadia Mono,Consolas,monospace;color:' + C.char + ';min-height:30px;word-break:break-all"></div>');
    var bp = api.el('pn', 'left:28px;top:136px;width:520px;height:300px;overflow:auto', '<h4>Next-token probabilities</h4><div class="bars"></div>');
    var mp = api.el('pn', 'left:568px;top:136px;width:364px;height:300px;overflow:auto', '<div class="mo" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-pick>Sample</button> <button type="button" class="btn sm ghost" data-reset>Reset</button> <button type="button" class="btn sm ghost" data-check>Check with RDKit</button>');
    function show(){
      var smi = ctx.join(''); qs('.sm', tp).textContent = smi || '(empty)';
      var nx = bgNext(['^', '^'].concat(ctx)), top = nx.list.slice(0, 7), mx = top.length ? top[0][1] : 1;
      qs('.bars', bp).innerHTML = done ? '<p class="note">The end token was chosen, so the string is complete.</p>' : (top.length ? top.map(function(t){ var end = t[0] === '$'; return '<div class="bgb" data-t="' + esc(t[0]) + '" style="display:grid;grid-template-columns:84px 1fr 46px;gap:8px;align-items:center;margin:6px 0;cursor:pointer"><span style="font:600 14px Cascadia Mono,Consolas,monospace;border:1px solid ' + C.silver + ';border-radius:4px;padding:3px 6px;background:#fff;text-align:center">' + (end ? 'end' : esc(t[0])) + '</span><div style="height:14px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:14px;border-radius:3px;background:' + C.red + ';width:' + (t[1] / mx * 100) + '%"></i></div><b style="text-align:right;font-size:12px">' + Math.round(t[1] * 100) + '%</b></div>'; }).join('') : '<p class="note">No structure in the set continues from this context. Select Reset.</p>');
      var last = ctx.length ? ctx[ctx.length - 1] : '(start)';
      qs('.mo', mp).innerHTML = '<div class="ol">Source of the probabilities</div><p>The model conditions on the last ' + (nx.used === 2 ? 'two tokens' : 'token') + (ctx.length ? ' (<b>' + esc(ctx.slice(-2).join(' ')) + '</b>)' : '') + ' and counts what followed them in the ' + BG.n + ' example structures: ' + nx.total + ' occurrence' + (nx.total === 1 ? '' : 's') + '.</p><p>A GPT conditions on all earlier tokens and is trained on millions of molecules, so its probabilities are much better calibrated.</p><div id="bgm"></div>';
      var tops = qsa('.bgb', bp); tops.forEach(function(e){ e.addEventListener('click', function(){ add(e.dataset.t); }); });
    }
    function add(t){ if (done) return; if (t === '$') { done = true; } else ctx.push(t); if (ctx.length > 70) done = true; show(); }
    qs('[data-pick]', ctl).addEventListener('click', function(){ if (done) return; var nx = bgNext(['^', '^'].concat(ctx)); if (!nx.list.length) return; var r = Math.random() * nx.total, acc = 0, ch = nx.list[0][0]; for (var i = 0; i < nx.list.length; i++) { acc += nx.list[i][2]; if (r <= acc) { ch = nx.list[i][0]; break; } } add(ch); });
    qs('[data-reset]', ctl).addEventListener('click', function(){ ctx = []; done = false; show(); });
    qs('[data-check]', ctl).addEventListener('click', async function(){ var smi = ctx.join(''); if (!smi) return; await inReady(api); var ok = rdkValid(smi), box = qs('#bgm', mp); box.innerHTML = ok ? '<p><b style="color:#1E6B40">RDKit parses this string as a valid molecule.</b></p><div class="struct" style="height:170px">' + rdkSvg(smi, 300, 170) + '</div>' : '<p><b style="color:' + C.red + '">RDKit cannot parse this string.</b> The model reproduces token patterns and does not check chemical validity. A longer context reduces how often this happens.</p>'; });
    async function render(){ await loadStructs(); bgBuild(); show(); }
    async function play(tok){
      await render(); ctx = []; done = false; show(); api.cap('The first-token probabilities reflect how the example structures begin.'); await api.sleep(2800);
      for (var k = 0; k < 14 && !done; k++) { api.checkTok(tok); qs('[data-pick]', ctl).click(); await api.sleep(550); } api.cap('Each appended token changes the probabilities of the next. Repeating the step generates a string.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: effect of pre-training ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'A head start', title: 'Effect of pre-training',
  change: 'Select the number of labelled compounds and compare the two models.',
  watch: 'Pre-training trains a model on a large related task before it is fitted to the endpoint of interest. Here Chemprop, a graph neural network, is fitted to HLM clearance in two ways: from random weights, and from weights pre-trained on logD, an endpoint with many more measurements. Each bar is the mean test R2 of two runs on scaffold-split HLM test compounds, for the stated number of labelled HLM compounds. R2 of 1 is a perfect fit and 0 equals predicting the training mean.',
  build: function(api){
    var sel = 500, NS = [50, 100, 200, 500, 1000];
    var bp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Test R2 from saved Chemprop runs on HLM clearance</h4><div class="gb"></div>');
    var tp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="gt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Labelled compounds ' + inSeg(NS.map(function(n){ return n.toLocaleString(); }), 3, 'data-n'));
    function avg(mode, n){ var r = EXTRAS.chemprop_ft.runs.filter(function(x){ return x.mode === mode && x.n === n; }); return mean(r.map(function(x){ return x.r2; })); }
    function bar(label, sub, v, col){ var w = clamp(v, 0, 0.6) / 0.6 * 100; return '<div style="margin:14px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="font-size:30px;margin:0;color:' + col + '">' + v.toFixed(2) + '</span></div><div style="font-size:12px;margin-bottom:4px">' + sub + '</div><div style="height:18px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:18px;width:' + w + '%;background:' + col + ';transition:width .4s"></i></div></div>'; }
    function draw(){
      if (!EXTRAS) return; var a = avg('scratch', sel), b = avg('full', sel);
      qs('.gb', bp).innerHTML = '<div class="ol">With ' + sel.toLocaleString() + ' labelled compounds</div>' + bar('From scratch', 'Random initial weights', a, C.char) + bar('Pre-trained on logD', 'Weights initialised from logD training, then fitted to HLM', b, C.red) + '<div style="font-size:11px;color:' + C.mid + ';margin-top:6px">The bar scale runs from 0 to 0.6.</div>';
      qs('.gt', tp).innerHTML = '<div class="ol">Effect of pre-training</div><div class="big" style="color:' + C.red + '">' + (b - a >= 0 ? '+' : '') + (b - a).toFixed(2) + '</div><div style="font-size:12px">change in test R2</div><p style="margin-top:12px">' + (sel <= 200 ? 'With 200 or fewer labelled compounds the pre-trained model scores clearly higher.' : (sel < 1000 ? 'The pre-trained model stays ahead as the labelled set grows.' : 'With 1,000 labelled compounds both models improve, and the pre-trained model still scores higher.')) + '</p><p>logD is cheap to measure, so thousands of values are available. Pre-training on them gives the model general structural information before any HLM value is used.</p>';
    }
    qsa('[data-n] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = NS[+b.dataset.v]; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ await loadExtras(); draw(); }
    async function play(tok){ await render(); for (var k = 0; k < NS.length; k++) { api.checkTok(tok); sel = NS[k]; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); draw(); api.cap(k === 0 ? 'With 50 labelled compounds neither model scores well, but the pre-trained model starts higher.' : (k === NS.length - 1 ? 'With 1,000 compounds the pre-trained model scores about twice as high as the model trained from scratch.' : 'Both models improve as labelled compounds are added.')); await api.sleep(2600); } api.cap(null); }
    return { render: render, play: play };
  } });

/* ---------- 06 Compare: fast or accurate ---------- */
LESSONS.compare.add({ level: 'beginner', short: 'Fast or accurate', title: 'Accuracy and training time',
  change: 'Select a model. Select Show dominated model to mark the model that another model beats on both axes.',
  watch: 'Models trade accuracy against training time. The three models below were run on the HLM data. Height is test R2, and horizontal position is training time on a logarithmic axis. A model that is both less accurate and slower than another is dominated, and there is no reason to select it.',
  build: function(api){
    var cur = 0, show = false, M = null, MD = { 'LightGBM': 'Gradient-boosted decision trees: trees added in sequence, each fitted to the residuals of the previous ones.', 'Random forest': 'Decision trees fitted to bootstrap samples of the compounds, with their predictions averaged.', 'Chemprop': 'A graph neural network that takes the molecular graph as input.' };
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Accuracy against training time</h4><canvas width="530" height="350"></canvas>');
    var tp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="ct" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-show>Show dominated model</button>');
    function models(){ return EXTRAS.refs.HLM.filter(function(r){ return r[2] !== null; }).map(function(r){ return { n: r[0], v: r[1], t: r[2] }; }); }
    function draw(){
      if (!EXTRAS) return; M = models(); var g = cv(qs('canvas', cp)), x = g.x, box = { l: 52, r: g.w - 20, t: 20, b: g.h - 44 }, lo = Math.log10(1), hi = Math.log10(100), sx = function(t){ return box.l + (Math.log10(t) - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - v / 0.6 * (box.b - box.t); };
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right'; [0, 0.2, 0.4, 0.6].forEach(function(v){ x.beginPath(); x.moveTo(box.l, sy(v)); x.lineTo(box.r, sy(v)); x.stroke(); x.fillText(v.toFixed(1), box.l - 6, sy(v) + 4); });
      x.textAlign = 'center'; [[1, '1 s'], [10, '10 s'], [100, '100 s']].forEach(function(p){ x.beginPath(); x.moveTo(sx(p[0]), box.t); x.lineTo(sx(p[0]), box.b); x.stroke(); x.fillText(p[1], sx(p[0]), box.b + 16); });
      x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('training time (log scale)', (box.l + box.r) / 2, box.b + 36); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('accuracy (R2)', 0, 0); x.restore();
      var dom = M.map(function(a, i){ return M.some(function(b, j){ return j !== i && b.v >= a.v && b.t <= a.t && (b.v > a.v || b.t < a.t); }); });
      if (show) M.forEach(function(a, i){ if (!dom[i]) return; M.forEach(function(b, j){ if (j !== i && b.v >= a.v && b.t <= a.t) { x.strokeStyle = C.red; x.setLineDash([5, 4]); x.lineWidth = 2; x.beginPath(); x.moveTo(sx(a.t), sy(a.v)); x.lineTo(sx(b.t), sy(b.v)); x.stroke(); x.setLineDash([]); } }); });
      M.forEach(function(m, i){ x.fillStyle = show && dom[i] ? C.bg : (i === cur ? C.red : C.char); x.beginPath(); x.arc(sx(m.t), sy(m.v), i === cur ? 9 : 7, 0, 6.3); x.fill(); x.fillStyle = C.char; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; var lab = m.n.split(',')[0]; x.fillText(lab, sx(m.t), sy(m.v) - 14); });
      g.hit = M.map(function(m){ return [sx(m.t), sy(m.v)]; }); qs('canvas', cp)._hit = g.hit;
      var m = M[cur], d = dom[cur];
      qs('.ct', tp).innerHTML = '<div class="ol">' + esc(m.n) + '</div><div style="display:flex;gap:22px"><div><div class="big" style="color:' + C.red + ';font-size:30px">' + m.v.toFixed(2) + '</div><div style="font-size:12px">accuracy (R2)</div></div><div><div class="big" style="font-size:30px">' + (m.t >= 10 ? Math.round(m.t) : m.t.toFixed(1)) + ' s</div><div style="font-size:12px">training time</div></div></div><p style="margin-top:10px">' + (MD[m.n.split(',')[0]] || '') + '</p><p>' + (d ? 'Another model is both more accurate and faster, so this model is dominated.' : (m.t > 20 ? 'Slow, but not dominated.' : 'Not dominated. The choice among non-dominated models depends on whether accuracy or training time is the constraint.')) + '</p><p style="font-size:12px;color:' + C.mid + '">Saved reference runs made with more features than the course lessons use. They illustrate the trade-off and are not a general ranking.</p>';
    }
    qs('canvas', cp).addEventListener('click', function(e){ var c = qs('canvas', cp), r = c.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * 530, py = (e.clientY - r.top) / r.height * 350, h = c._hit || []; var best = -1, bd = 1e9; h.forEach(function(p, i){ var dd = Math.hypot(p[0] - px, p[1] - py); if (dd < bd) { bd = dd; best = i; } }); if (best >= 0 && bd < 40) { cur = best; draw(); } });
    qs('[data-show]', ctl).addEventListener('click', function(){ show = !show; this.textContent = show ? 'Hide dominated model' : 'Show dominated model'; draw(); });
    async function render(){ await loadExtras(); draw(); }
    async function play(tok){ await render(); show = false; cur = 0; draw(); api.cap('Each point is a model. Higher is more accurate and further right is slower.'); await api.sleep(3200); show = true; qs('[data-show]', ctl).textContent = 'Hide dominated model'; draw(); api.cap('The grey model is slower and less accurate than the first model, so it is dominated.'); await api.sleep(3400); api.cap(null); }
    return { render: render, play: play };
  } });
