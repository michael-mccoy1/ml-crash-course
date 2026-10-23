"use strict";
/* Lesson 4 continued: steps 6 to 9 and glossary */
function nnAct(i){
  var P = NN.prep, A = nnActivations(i), L = NN.m.sizes.length - 1, layers = []; for (var l = 1; l <= L; l++) layers.push(Array.from(A[l]));
  return { i: i, x: Array.from(A[0]), layers: layers, raw: P.feats.map(function(f){ return D.COL[f][i]; }), max: layers.map(function(a){ return Math.max.apply(null, a.slice(0, 9).concat([1e-6])); }), pred: layers[layers.length - 1][0] + P.ymu, meas: D.Y[i] };
}

/* ---------- N6: one compound through the trained network ---------- */
L3.add({ short: 'One compound', title: 'Follow one compound through the trained network',
  change: 'Press Play, then Next compound. The first time, a network is trained with your current options.',
  watch: 'The compound\'s features light up the inputs. Each hidden layer turns them into new numbers. The last neuron gives the prediction.',
  build: function(api){
    var cur = null, act = null;
    var sp = api.el('pn', 'left:28px;top:28px;width:230px;height:226px', '<h4>Compound</h4><div class="struct" style="height:186px"></div>');
    var fp = api.el('pn', 'left:28px;top:266px;width:230px;height:170px', '<div class="smi" style="font-size:10.5px;max-height:44px;overflow:hidden"></div><div class="note" style="font-size:11.5px;color:#6B7280;margin-top:6px">Red inputs are above average for this feature, dark inputs are below.</div>');
    var dp = api.el('pn', 'left:278px;top:28px;width:470px;height:408px', '<svg viewBox="0 0 450 380" width="450" height="380"></svg>');
    var op = api.el('pn fd', 'left:768px;top:28px;width:164px;height:408px', '<h4>Prediction</h4><div class="big pv" style="color:' + C.red + '">0</div><div class="lbl" style="margin-top:6px">measured</div><div class="big mv" style="font-size:26px">0</div><div class="pbar"><i style="background:' + C.red + ';width:0"></i></div><div class="pbar"><i style="background:' + C.char + ';width:0"></i></div><div class="note er" style="font-size:12px;color:#6B7280"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    function draw(stage){ drawNetSVG(qs('svg', dp), { W: 450, H: 380, sizes: netSizes(NN.prep.feats.length, NN.params.layers, NN.params.units), feats: NN.prep.feats, model: NN.m, act: act, stage: stage, maxShow: 9, padL: 92, padR: 44, sel: NN.sel }); }
    async function setup(next){ await loadStructs(); if (!(await nnReady(api))) return false;
      if (!cur || next || cur._ep !== EP) { cur = await pickStruct(); cur._ep = EP; } act = nnAct(cur.i); NN.act = act; qs('.struct', sp).innerHTML = cur.svg; qs('.smi', fp).textContent = cur.smi; op.classList.remove('on'); draw(-1); return true; }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ api.tok++; if (!(await setup(true))) return; var t = api.tok; api.stepDef.safe(function(){ return play(t); }); });
    async function render(){ if (!(await setup(false))) return; draw(99); showOut(); }
    function showOut(){ var span = D.yhi - D.ylo; qs('.pv', op).textContent = act.pred.toFixed(2); qs('.mv', op).textContent = act.meas.toFixed(2); var bars = qsa('.pbar i', op); bars[0].style.width = clamp((act.pred - D.ylo) / span * 100, 0, 100) + '%'; bars[1].style.width = clamp((act.meas - D.ylo) / span * 100, 0, 100) + '%'; qs('.er', op).textContent = 'Error ' + (act.pred - act.meas >= 0 ? '+' : '') + (act.pred - act.meas).toFixed(2); op.classList.add('on'); }
    async function play(tok){ if (!(await setup(false))) return; var L = NN.params.layers; draw(-1); op.classList.remove('on'); await api.beat('Start with the compound and its features. This network has ' + NN.prep.feats.length + ' inputs.', 2800);
      draw(0); await api.beat('The features enter the input neurons. Red neurons are above average for the training set, dark ones below.', 3200);
      for (var l = 1; l <= L; l++) { draw(l); await api.beat('Hidden layer ' + l + ': every neuron adds up its weighted inputs and applies ReLU. Darker red means more active.', 3200); }
      draw(L + 1); showOut(); await api.beat('The last neuron combines the final layer into one number, the prediction. Compare it with the lab value.', 3400); api.key('A network turns features into a prediction through layers of weighted sums. Nothing in it is hand-written.'); }
    return { render: render, play: play };
  } });

/* ---------- N7: inside one neuron ---------- */
L3.add({ short: 'Inside a neuron', title: 'Open one neuron and read its arithmetic',
  change: 'Click any circle in hidden layer 1, or press Play.',
  watch: 'The table shows the real learned weight for each input, the contribution it makes for this compound, and the running total.',
  build: function(api){
    var dp = api.el('pn', 'left:28px;top:28px;width:360px;height:408px', '<h4>Click a neuron in hidden layer 1</h4><svg viewBox="0 0 340 360" width="340" height="360"></svg>');
    var tp = api.el('pn', 'left:408px;top:28px;width:524px;height:408px', '<div class="nt"></div>');
    function draw(){ drawNetSVG(qs('svg', dp), { W: 340, H: 360, sizes: netSizes(NN.prep.feats.length, NN.params.layers, NN.params.units), feats: NN.prep.feats, model: NN.m, act: NN.act, stage: 99, maxShow: 9, padL: 78, padR: 24, sel: NN.sel, noLabels: true, onPick: function(k){ api.tok++; NN.sel = k; table(1); draw(); } }); }
    function table(prog){
      var ker = NN.m.W[0], bias = NN.m.b[0], units = NN.m.sizes[1], j = NN.sel, f = NN.prep.feats, a = NN.act, rows = '', z = bias[j], zs = [];
      var n = Math.max(1, Math.round(prog * (f.length + 2)));
      f.forEach(function(ft, k){ var wt = ker[k * units + j], c = a.x[k] * wt; z += c; zs.push(z); if (k < n) rows += '<tr><td>' + esc(ft) + '</td><td>' + fmtFeat(ft, a.raw[k]) + '</td><td>' + a.x[k].toFixed(2) + '</td><td style="color:' + (wt >= 0 ? C.red : C.char) + '">' + (wt >= 0 ? '+' : '') + wt.toFixed(2) + '</td><td><b>' + (c >= 0 ? '+' : '') + c.toFixed(2) + '</b></td><td style="width:90px"><div style="position:relative;height:10px;background:#EDF2F3;border-radius:3px"><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#6B7280"></div><div style="position:absolute;top:0;bottom:0;border-radius:3px;background:' + (c >= 0 ? C.red : C.bg) + ';' + (c >= 0 ? 'left:50%;' : 'right:50%;') + 'width:' + Math.min(50, Math.abs(c) * 14) + '%"></div></div></td><td style="text-align:right">' + zs[k].toFixed(2) + '</td></tr>'; });
      var showB = n >= f.length + 1, showR = n >= f.length + 2;
      qs('.nt', tp).innerHTML = '<h4>Neuron ' + (j + 1) + ' of hidden layer 1</h4><div class="tblwrap"><table style="font-size:12px"><thead><tr><th>Feature</th><th>Value</th><th>Rescaled</th><th>Weight</th><th>Contribution</th><th></th><th style="text-align:right">Running total</th></tr></thead><tbody>' + rows + (showB ? '<tr><td>bias</td><td></td><td></td><td></td><td><b>' + (bias[j] >= 0 ? '+' : '') + bias[j].toFixed(2) + '</b></td><td></td><td style="text-align:right">' + z.toFixed(2) + '</td></tr>' : '') + '</tbody></table></div>' + (showR ? '<p style="font-size:13px;margin-top:10px">Total <b>' + z.toFixed(2) + '</b>. ReLU turns it into <b style="color:' + C.red + '">' + Math.max(0, z).toFixed(2) + '</b>' + (z <= 0 ? ', so this neuron is silent for this compound.' : ', so this neuron passes a signal to the next layer.') + '</p>' : '');
    }
    async function ready(){ if (!(await nnReady(api))) return false; if (!NN.act) NN.act = nnAct(D.TEST[Math.floor(Math.random() * D.TEST.length)]); return true; }
    async function render(){ if (!(await ready())) return; draw(); table(1); }
    async function play(tok){ if (!(await ready())) return; draw(); table(0); await api.beat('Every neuron in the first hidden layer reads all the inputs. Pick one.', 2600); api.cap('Each row is one input: its value, the learned weight, and what it adds to the total.');
      for (var k = 1; k <= NN.prep.feats.length + 2; k++) { table(k / (NN.prep.feats.length + 2)); await api.sleep(750); } api.cap('The bias is added, then ReLU switches the neuron on or off. Click another neuron to compare.'); await api.sleep(3200); api.key('Each neuron weighs the same inputs differently, so each one detects a different pattern.'); }
    return { render: render, play: play };
  } });

/* ---------- N8: what the network learned ---------- */
L3.add({ short: 'What it learned', title: 'See what the network learned about one feature',
  change: 'Choose a feature in the control below, then press Play.',
  watch: 'Each grey line is one unseen compound: the prediction as that feature changes while the compound\'s other features stay fixed. The red line is the average.',
  build: function(api){
    var pp = api.el('pn', 'left:28px;top:28px;width:640px;height:408px', '<h4>Prediction as one feature changes</h4><canvas width="618" height="350"></canvas>');
    var sp = api.el('pn', 'left:688px;top:28px;width:244px;height:408px', '<h4>How to read it</h4><div class="rd" style="font-size:12.5px;color:#4a4f5a"></div>');
    var ctl = api.addControl('Feature <select></select>'); var sel = qs('select', ctl), cache = null;
    function fill(){ var f = NN.prep ? NN.prep.feats : activeFeats(), cur = sel.value; sel.innerHTML = f.map(function(x){ return '<option>' + x + '</option>'; }).join(''); if (f.indexOf(cur) >= 0) sel.value = cur; }
    function compute(){ var f = sel.value, P = NN.prep, ci = P.feats.indexOf(f), r = rng(21), ids = D.TEST.filter(function(){ return r() < 15 / D.TEST.length; }).slice(0, 15), vals = D.TRAIN.map(function(i){ return D.COL[f][i]; }).sort(function(a, b){ return a - b; }), lo = vals[Math.floor(vals.length * 0.03)], hi = vals[Math.floor(vals.length * 0.97)], xs = Array.from({ length: 36 }, function(_, k){ return lo + (hi - lo) * k / 35; });
      var lines = ids.map(function(i){ var base = P.rowOf(i), rows = xs.map(function(v){ var rw = base.slice(); rw[ci] = clamp((v - P.mu[f]) / P.sd[f], -5, 5); return rw; }); return nnPredictRows(rows); }); var avg = xs.map(function(_, k){ return mean(lines.map(function(l){ return l[k]; })); }); return { f: f, xs: xs, lines: lines, avg: avg, lo: lo, hi: hi }; }
    function draw(prog){ if (!cache) return; var g = cv(qs('canvas', pp)), x = g.x, box = { l: 52, r: g.w - 12, t: 12, b: g.h - 42 }; axes(g, box, [cache.lo, cache.hi], [D.ylo, D.yhi], cache.f, unitWord(), 5); var sx = function(v){ return box.l + (v - cache.lo) / (cache.hi - cache.lo) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, D.ylo, D.yhi) - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); }, n = Math.max(1, Math.round(prog * cache.xs.length));
      cache.lines.forEach(function(l){ x.strokeStyle = 'rgba(52,55,63,.28)'; x.lineWidth = 1.5; x.beginPath(); for (var k = 0; k < n; k++) { k ? x.lineTo(sx(cache.xs[k]), sy(l[k])) : x.moveTo(sx(cache.xs[k]), sy(l[k])); } x.stroke(); }); x.strokeStyle = C.red; x.lineWidth = 4; x.beginPath(); for (var k2 = 0; k2 < n; k2++) { k2 ? x.lineTo(sx(cache.xs[k2]), sy(cache.avg[k2])) : x.moveTo(sx(cache.xs[k2]), sy(cache.avg[k2])); } x.stroke();
      var d = cache.avg[cache.avg.length - 1] - cache.avg[0]; qs('.rd', sp).innerHTML = '<p><b style="color:' + C.red + '">Red line</b>: average prediction over 15 unseen compounds.</p><p><b>Grey lines</b>: each compound on its own. They differ because the other features differ.</p><p>Across the range of ' + esc(cache.f) + ', the average prediction moves by <b>' + (d >= 0 ? '+' : '') + d.toFixed(2) + '</b>.</p><p class="note" style="color:#6B7280">A curve that is not a straight line shows the network learned a non-linear effect.</p>'; }
    sel.addEventListener('change', function(){ api.tok++; cache = compute(); draw(1); });
    async function ready(){ if (!(await nnReady(api))) return false; fill(); if (sel.value) cache = compute(); return true; }
    async function render(){ if (!(await ready())) return; draw(1); }
    async function play(tok){ if (!(await ready())) return; if (!cache) return; draw(0); await api.beat('Take 15 unseen compounds. For each one, change only ' + cache.f + ' and ask the network for a new prediction.', 3200); await api.anim(ANIM ? 4200 : 1, function(t){ draw(t); }); api.cap('The red line is the average effect of ' + cache.f + ' according to the network.'); await api.sleep(3400); api.key('You can look inside a neural network by asking it what happens when one input changes.'); }
    return { render: render, play: play };
  } });

/* ---------- N9: Chemprop and message passing ---------- */
L3.add({ short: 'Chemprop', title: 'Chemprop passes messages along the bonds of the molecular graph',
  change: 'Press Play. Click an atom to start from it, or drag the rounds control.',
  watch: 'Atoms inside the red zone have received information from the chosen atom. After more rounds the zone grows by one bond.',
  build: function(api){
    var G = [], cur = { g: 0, c: 0, k: 0 }, playing = false, EL = { N: C.red, O: C.dred, S: '#9a7400', F: '#2c7a57', Cl: '#2c7a57', Br: '#2c7a57' };
    var gp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<svg viewBox="0 0 540 372" width="540" height="372"></svg>');
    var sp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px', '<div class="cx"></div>');
    var ctl = api.addControl('Molecule <select></select> Rounds <input type="range" min="0" max="5" step="1" value="0" style="width:120px"><b class="kv">0</b>'); var msel = qs('select', ctl), rr = qs('input', ctl);
    function layout(g){ var xs = g.atoms.map(function(a){ return a[1]; }), ys = g.atoms.map(function(a){ return a[2]; }), x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys), sc = Math.min(480 / ((x1 - x0) || 1), 320 / ((y1 - y0) || 1)), ox = 270 - (x1 - x0) * sc / 2, oy = 186 - (y1 - y0) * sc / 2; return g.atoms.map(function(a){ return [ox + (a[1] - x0) * sc, oy + (a[2] - y0) * sc]; }); }
    function dist(g, c){ var d = g.atoms.map(function(){ return 99; }), adj = g.atoms.map(function(){ return []; }); g.bonds.forEach(function(b){ adj[b[0]].push(b[1]); adj[b[1]].push(b[0]); }); d[c] = 0; var q = [c]; while (q.length) { var u = q.shift(); adj[u].forEach(function(v){ if (d[v] === 99) { d[v] = d[u] + 1; q.push(v); } }); } return d; }
    function draw(){
      var g = G[cur.g], pos = layout(g), d = dist(g, cur.c), k = cur.k, s = '';
      g.bonds.forEach(function(b){ var on = d[b[0]] <= k && d[b[1]] <= k, x1 = pos[b[0]][0], y1 = pos[b[0]][1], x2 = pos[b[1]][0], y2 = pos[b[1]][1], dx = y2 - y1, dy = x1 - x2, L = Math.sqrt(dx * dx + dy * dy) || 1, o = b[2] === 2 ? 2.6 : 0, col = on ? C.red : C.bg, wd = on ? 3.2 : 1.8;
        if (b[2] === 2) s += '<line x1="' + (x1 + dx / L * o) + '" y1="' + (y1 + dy / L * o) + '" x2="' + (x2 + dx / L * o) + '" y2="' + (y2 + dy / L * o) + '" stroke="' + col + '" stroke-width="' + wd + '"/><line x1="' + (x1 - dx / L * o) + '" y1="' + (y1 - dy / L * o) + '" x2="' + (x2 - dx / L * o) + '" y2="' + (y2 - dy / L * o) + '" stroke="' + col + '" stroke-width="' + wd + '"/>'; else s += '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" stroke="' + col + '" stroke-width="' + wd + '"/>'; });
      var reach = 0; g.atoms.forEach(function(a, i){ var on = d[i] <= k; if (on) reach++; var fo = on ? Math.max(0.3, 1 - 0.16 * d[i]) : 1, isC = i === cur.c; s += '<circle data-a="' + i + '" cx="' + pos[i][0] + '" cy="' + pos[i][1] + '" r="' + (isC ? 14 : 11) + '" fill="' + (isC ? C.dred : (on ? C.red : '#fff')) + '" fill-opacity="' + (isC ? 1 : (on ? fo : 1)) + '" stroke="' + (on ? C.dred : C.char) + '" stroke-width="' + (isC ? 3 : 1.6) + '" style="cursor:pointer"/>'; if (a[0] !== 'C') s += '<text x="' + pos[i][0] + '" y="' + (pos[i][1] + 4) + '" text-anchor="middle" font-size="11" font-weight="700" fill="' + (on || isC ? '#fff' : (EL[a[0]] || C.char)) + '" pointer-events="none">' + a[0] + '</text>'; });
      qs('svg', gp).innerHTML = s + '<g class="dots"></g>'; qsa('circle[data-a]', gp).forEach(function(c){ c.addEventListener('click', function(){ if (playing) return; api.tok++; cur.c = +c.dataset.a; draw(); }); });
      var cp = EXTRAS && EXTRAS.meta && EXTRAS.meta['Chemprop (D-MPNN)'];
      qs('.cx', sp).innerHTML = '<h4>Round ' + k + '</h4><p style="font-size:13px">Atom ' + (cur.c + 1) + ' (' + g.atoms[cur.c][0] + ') holds information from <b>' + reach + ' of ' + g.atoms.length + '</b> atoms.</p><ol style="font-size:12px;color:#4a4f5a;padding-left:16px;line-height:1.5;margin:8px 0"><li>Each atom starts with a vector describing itself: element, charge, ring membership.</li><li>Each round, every atom sends a message to its bonded neighbours and updates itself with what it receives.</li><li>After k rounds an atom knows its k-bond neighbourhood.</li><li>All atom vectors are pooled into one vector for the molecule.</li><li>A small network turns that vector into the prediction.</li></ol>' + (cp ? '<div class="score"><b>Chemprop on HLM, notebook 06</b><div class="v">' + cp.R2.toFixed(2) + '</div><div class="s">unseen R2, same scaffold split<br>' + Math.round(cp.train_secs) + ' s on a laptop CPU, in Python</div></div>' : '<p class="note">Run notebook 06 to add the Chemprop score.</p>'); qs('.kv', ctl).textContent = k; rr.value = k; }
    async function pulse(step){ var g = G[cur.g], pos = layout(g), d = dist(g, cur.c), edges = g.bonds.filter(function(b){ return (d[b[0]] === step - 1 && d[b[1]] === step) || (d[b[1]] === step - 1 && d[b[0]] === step); }); if (!ANIM || !edges.length) return;
      var layer = qs('.dots', gp), dots = edges.map(function(b){ var a = d[b[0]] === step - 1 ? b[0] : b[1], z = d[b[0]] === step - 1 ? b[1] : b[0], c = document.createElementNS('http://www.w3.org/2000/svg', 'circle'); c.setAttribute('r', 5); c.setAttribute('fill', C.dred); layer.appendChild(c); return { c: c, a: pos[a], z: pos[z] }; });
      await api.anim(650, function(t){ dots.forEach(function(o){ o.c.setAttribute('cx', lerp(o.a[0], o.z[0], t)); o.c.setAttribute('cy', lerp(o.a[1], o.z[1], t)); }); }); }
    async function setK(k, withPulse){ if (withPulse && k > 0) { cur.k = k - 1; draw(); await pulse(k); } cur.k = k; draw(); }
    rr.addEventListener('input', function(){ if (!playing) { api.tok++; cur.k = +rr.value; draw(); } });
    msel.addEventListener('change', function(){ api.tok++; cur = { g: +msel.value, c: 0, k: +rr.value }; draw(); });
    async function ready(){ var ex = await loadExtras(); G = ex.graphs; if (!msel.options.length) G.forEach(function(g, i){ var o = document.createElement('option'); o.value = i; o.textContent = g.smi.length > 26 ? g.smi.slice(0, 26) + '...' : g.smi; msel.appendChild(o); }); }
    async function render(){ await ready(); draw(); }
    async function play(tok){ await ready(); playing = true; try { cur.k = 0; draw(); await api.beat('A molecule is a graph: atoms are nodes and bonds are edges. Chemprop works on this graph directly, with no descriptor table.', 3600);
      for (var k = 1; k <= 4; k++) { await setK(k, true); api.cap(k === 1 ? 'Round 1: the chosen atom has heard from its bonded neighbours.' : 'Round ' + k + ': information has travelled ' + k + ' bonds from the chosen atom.'); await api.sleep(2000); }
      api.cap('After a few rounds every atom carries a summary of the chemistry around it. These are pooled and passed to a small network.'); await api.sleep(3400); api.key('Chemprop learns its own features from the graph. It needs Python and more compute, so it is run in notebook 06.'); } finally { playing = false; } }
    return { render: render, play: play };
  } });

L3.glossary([
  ['Neural network', 'A model built from many small calculators (neurons) in layers. It learns by adjusting the strength of the connections.'],
  ['Neuron', 'Multiplies each input by a weight, adds a bias, and passes the total through an activation.'],
  ['Weight', 'Strength of one connection. Training changes the weights.'], ['Bias', 'A learned offset added in each neuron.'],
  ['Layer', 'A column of neurons: input, hidden or output.'], ['ReLU', 'Negative values become 0 and positive values pass unchanged.'],
  ['Parameters', 'All weights and biases together.'], ['Epoch', 'One pass through the training data.'],
  ['Loss', 'Average squared error. Training tries to lower it.'], ['Gradient descent', 'Repeatedly stepping the weights in the direction that lowers the loss.'],
  ['Learning rate', 'Size of each weight update. Too small is slow, too large is unstable.'],
  ['Dropout', 'Randomly switching off neurons during training to curb overfitting.'],
  ['Validation set', 'Training compounds held aside to watch for overfitting. Different from the unseen set.'],
  ['Best epoch', 'The epoch with the lowest validation loss. Keeping it is early stopping.'],
  ['Standardising', 'Rescaling each feature to a similar range so large numbers do not dominate.'],
  ['Message passing', 'Atoms exchange vectors with bonded neighbours over several rounds.'],
  ['Chemprop', 'A message-passing neural network for molecules that reads the graph instead of descriptors.']
]);

[5, 6, 7].forEach(function(i){ L3.steps[i].needsNet = true; });
setUses(L3, [['features', 'endpoint'], ['nlayers', 'nunits', 'ndrop', 'features'], ['nlr', 'nepochs', 'features', 'endpoint'], ['nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs', 'nearly', 'features', 'endpoint'], ['nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs', 'features', 'endpoint'], ['nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs', 'nearly', 'features', 'endpoint'], ['nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs', 'nearly', 'features', 'endpoint'], ['nlayers', 'nunits', 'ndrop', 'nlr', 'nepochs', 'nearly', 'features', 'endpoint'], []]);
