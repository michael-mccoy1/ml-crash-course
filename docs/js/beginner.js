"use strict";
/* One pure-beginner step for lessons 02 to 06. Each uses plain words, a picture, and no background knowledge. */

/* ---------- 02 Tree models: twenty questions ---------- */
LESSONS.trees.add({ level: 'beginner', short: 'Twenty questions', title: 'A tree sorts compounds by asking questions',
  change: 'Drag the slider to let the tree ask more questions, or press Walk through.',
  watch: 'Have you played twenty questions? Each yes or no answer narrows the choices. A decision tree does the same with compounds. Each dot is a compound, and its height is the value the lab measured. The tree asks a question about a number, such as "is logP above 3?" and sends the compound left or right. After a few questions the compounds are sorted into groups. The red line in each group is the group average, and the average is the prediction for any new compound that lands in the group. Notice how the groups become tighter as the tree asks more questions.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:904px;height:360px', '<canvas width="880" height="336"></canvas>');
    var tp = api.el('pn', 'left:28px;top:396px;width:904px;height:44px;padding-top:10px', '<div class="st" style="font-size:13px;color:' + C.char + '"></div>');
    var ctl = api.addControl('Questions the tree may ask <input type="range" min="0" max="3" step="1" value="2" style="width:200px"> <b class="qv">2</b>');
    var rg = qs('input', ctl), qv = qs('.qv', ctl), FE = ['logP', 'TPSA', 'MW'];
    function sample(){ var key = EP + 'tw'; if (D.cache[key]) return D.cache[key]; var r = rng(5), sh = D.TRAIN.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; } return (D.cache[key] = sh.slice(0, 260)); }
    function leaves(node, path, out){ if (node.f === undefined) { out.push({ node: node, path: path }); return out; } var thr = D.B.edges[node.f][node.b], w = fmtFeat(node.f, thr); leaves(node.l, path.concat(node.f + ' at most ' + w), out); leaves(node.r, path.concat(node.f + ' above ' + w), out); return out; }
    function draw(){
      var d = +rg.value, feats = FE.filter(function(f){ return D.COL[f]; }), tree = fitTree(feats, D.TRAIN, d, 25, 1), lv = leaves(tree, [], []), g = cv(qs('canvas', cp)), x = g.x, n = lv.length, w = (g.w - 60) / n, top = 74, bot = g.h - 26, ys = function(v){ return bot - (v - D.ylo) / (D.yhi - D.ylo) * (bot - top); };
      qv.textContent = d; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText(unitWord().split(' ')[0] === 'HLM' ? 'lab value' : 'lab value', 44, top - 8);
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
      var te = r2(D.TEST, treeAll(tree, D.N)); qs('.st', tp).innerHTML = 'The tree has asked <b>' + d + '</b> question' + (d === 1 ? '' : 's') + ' in a row and made <b>' + n + '</b> group' + (n === 1 ? '' : 's') + '. Score on test compounds, which it has never studied: <b style="color:' + C.red + '">' + te.toFixed(2) + '</b> (1 is perfect, 0 is no better than the overall average).';
    }
    rg.addEventListener('input', draw);
    async function play(tok){ rg.value = 0; draw(); api.cap('Without any question, every compound gets the same answer: the overall average.'); await api.sleep(3000); for (var k = 1; k <= 3; k++) { api.checkTok(tok); rg.value = k; draw(); api.cap(k === 1 ? 'One question splits the compounds into two groups.' : (k === 2 ? 'A second question splits each group again.' : 'Three questions in a row. The groups are tighter around their averages.')); await api.sleep(3000); } api.cap(null); }
    return { render: draw, play: play };
  } });

/* ---------- 03 Neural networks: a neuron is a sum with dials ---------- */
LESSONS.nn.add({ level: 'beginner', short: 'A dial and a sum', title: 'A neuron is a sum with three dials',
  change: 'Turn the three dials, or press Tune the dials for me.',
  watch: 'A neural network is built from small units called neurons. A neuron takes in numbers, multiplies each by a dial, adds the results together, and adds a base amount. The dials are called weights and the base amount is called the bias. Here a single neuron reads two features of a compound, logP (how oily it is) and TPSA (how polar it is), and predicts the lab value. Each dot on the chart is a compound. The closer the dots lie to the dashed line, the better the predictions. Learning means turning the dials until the dots line up. A big network has thousands of dials, and the computer turns them automatically.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:500px;height:408px', '<h4>Predicted against measured, 260 training compounds</h4><canvas width="470" height="350"></canvas>');
    var sp = api.el('pn', 'left:548px;top:28px;width:384px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Dial 1 (logP) <input type="range" data-d="0" min="-1" max="1" step="0.01" value="0" style="width:120px"> Dial 2 (TPSA) <input type="range" data-d="1" min="-1" max="1" step="0.01" value="0" style="width:120px"> Base amount <input type="range" data-d="2" min="-1" max="1" step="0.01" value="0" style="width:120px"> <button type="button" class="btn sm" data-tune>Tune the dials for me</button> <button type="button" class="btn sm ghost" data-zero>Reset</button>');
    var rs = qsa('[data-d]', ctl), S = null;
    function prep(){ var key = EP + 'nb'; if (D.cache[key]) return D.cache[key]; var r = rng(8), sh = D.TRAIN.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; } var idx = sh.slice(0, 260), a1 = idx.map(function(i){ return D.COL.logP[i]; }), a2 = idx.map(function(i){ return D.COL.TPSA[i]; }), ys = idx.map(function(i){ return D.Y[i]; }), m1 = mean(a1), m2 = mean(a2), my = mean(ys), s1 = Math.sqrt(mean(a1.map(function(v){ return (v - m1) * (v - m1); }))) || 1, s2 = Math.sqrt(mean(a2.map(function(v){ return (v - m2) * (v - m2); }))) || 1, sy = Math.sqrt(mean(ys.map(function(v){ return (v - my) * (v - my); }))) || 1;
      var X = idx.map(function(i, k){ return [(a1[k] - m1) / s1, (a2[k] - m2) / s2, (ys[k] - my) / sy]; });
      /* the dials that fit best: least squares on the two inputs and the base amount */
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
      qs('.nb', sp).innerHTML = '<div class="ol">The neuron right now</div><p style="font-size:14px">prediction = <b>' + d[0].toFixed(2) + '</b> times logP + <b>' + d[1].toFixed(2) + '</b> times TPSA + <b>' + d[2].toFixed(2) + '</b></p><p style="font-size:12px;color:' + C.mid + '">The features are first rescaled so that 0 means average and 1 means one typical step above average. The sum is then scaled back to the units of the lab value.</p><div class="ol" style="margin-top:12px">Typical miss</div><div class="big" style="color:' + C.red + '">' + rm.toFixed(2) + '</div><p style="font-size:12px;margin-top:2px">in ' + esc(unitWord()) + ' units. Smaller is better.</p><p style="margin-top:10px">The best these three dials can do is a typical miss of <b>' + rb.toFixed(2) + '</b>. ' + (rm < rb + 0.01 ? 'You have found it.' : 'Press Tune the dials for me to see the computer find it.') + '</p>';
    }
    rs.forEach(function(r){ r.addEventListener('input', draw); });
    qs('[data-zero]', ctl).addEventListener('click', function(){ rs.forEach(function(r){ r.value = 0; }); draw(); });
    qs('[data-tune]', ctl).addEventListener('click', function(){ S = prep(); var from = rs.map(function(r){ return +r.value; }); animateFree(1600, function(t){ rs.forEach(function(r, k){ r.value = from[k] + (S.best[k] - from[k]) * t; }); draw(); }); });
    async function play(tok){ rs.forEach(function(r){ r.value = 0; }); draw(); api.cap('With all the dials at zero, the neuron predicts the same value for every compound.'); await api.sleep(3000); api.cap('Now the computer turns the dials, trying to make the dots line up with the dashed line.'); S = prep(); await api.anim(2600, function(t){ rs.forEach(function(r, k){ r.value = S.best[k] * t; }); draw(); }); api.cap('That is learning: finding dial settings that make the predictions match the lab values.'); await api.sleep(3000); api.cap(null); }
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
LESSONS.gpt.add({ level: 'beginner', short: 'Autocomplete for molecules', title: 'Writing a molecule one piece at a time',
  change: 'Click a piece to add it, or press Pick for me to let the computer choose. Press Start again to try another.',
  watch: 'A GPT is a model that predicts what comes next. Your phone does something similar when it suggests the next word. Here a molecule is written as a line of text, and the text is cut into small pieces called tokens: an atom, a bond or a bracket. The model has counted which token comes next in a set of 180 real molecules, and the bars show the chances. Click a piece to add it, or let the computer pick in proportion to the chances. When the end piece is chosen the molecule is finished. This is a simple counting version. A real GPT does the same job with a much larger memory, and it looks at every earlier token, not just the last two.',
  build: function(api){
    var ctx = [], done = false;
    var tp = api.el('pn', 'left:28px;top:28px;width:904px;height:96px', '<h4>The molecule so far, as text</h4><div class="sm" style="font:600 18px Cascadia Mono,Consolas,monospace;color:' + C.char + ';min-height:30px;word-break:break-all"></div>');
    var bp = api.el('pn', 'left:28px;top:136px;width:520px;height:300px;overflow:auto', '<h4>What usually comes next</h4><div class="bars"></div>');
    var mp = api.el('pn', 'left:568px;top:136px;width:364px;height:300px;overflow:auto', '<div class="mo" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-pick>Pick for me</button> <button type="button" class="btn sm ghost" data-reset>Start again</button> <button type="button" class="btn sm ghost" data-check>Check the molecule</button>');
    function show(){
      var smi = ctx.join(''); qs('.sm', tp).textContent = smi || '(empty: choose the first piece)';
      var nx = bgNext(['^', '^'].concat(ctx)), top = nx.list.slice(0, 7), mx = top.length ? top[0][1] : 1;
      qs('.bars', bp).innerHTML = done ? '<p class="note">The end piece was chosen, so the molecule is finished.</p>' : (top.length ? top.map(function(t){ var end = t[0] === '$'; return '<div class="bgb" data-t="' + esc(t[0]) + '" style="display:grid;grid-template-columns:84px 1fr 46px;gap:8px;align-items:center;margin:6px 0;cursor:pointer"><span style="font:600 14px Cascadia Mono,Consolas,monospace;border:1px solid ' + C.silver + ';border-radius:4px;padding:3px 6px;background:#fff;text-align:center">' + (end ? 'end' : esc(t[0])) + '</span><div style="height:14px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:14px;border-radius:3px;background:' + C.red + ';width:' + (t[1] / mx * 100) + '%"></i></div><b style="text-align:right;font-size:12px">' + Math.round(t[1] * 100) + '%</b></div>'; }).join('') : '<p class="note">No example in the set continues like this, so the model has nothing to suggest. Press Start again.</p>');
      var last = ctx.length ? ctx[ctx.length - 1] : '(start)';
      qs('.mo', mp).innerHTML = '<div class="ol">How the chances were found</div><p>The model looked at the last ' + (nx.used === 2 ? 'two pieces' : 'piece') + ' you wrote' + (ctx.length ? ', <b>' + esc(ctx.slice(-2).join(' ')) + '</b>' : '') + ', and counted what followed them in the ' + BG.n + ' example molecules. That was ' + nx.total + ' time' + (nx.total === 1 ? '' : 's') + '.</p><p>A real GPT keeps far more of the text in view, and it has learned from millions of molecules, so its suggestions are much better.</p><div id="bgm"></div>';
      var tops = qsa('.bgb', bp); tops.forEach(function(e){ e.addEventListener('click', function(){ add(e.dataset.t); }); });
    }
    function add(t){ if (done) return; if (t === '$') { done = true; } else ctx.push(t); if (ctx.length > 70) done = true; show(); }
    qs('[data-pick]', ctl).addEventListener('click', function(){ if (done) return; var nx = bgNext(['^', '^'].concat(ctx)); if (!nx.list.length) return; var r = Math.random() * nx.total, acc = 0, ch = nx.list[0][0]; for (var i = 0; i < nx.list.length; i++) { acc += nx.list[i][2]; if (r <= acc) { ch = nx.list[i][0]; break; } } add(ch); });
    qs('[data-reset]', ctl).addEventListener('click', function(){ ctx = []; done = false; show(); });
    qs('[data-check]', ctl).addEventListener('click', async function(){ var smi = ctx.join(''); if (!smi) return; await inReady(api); var ok = rdkValid(smi), box = qs('#bgm', mp); box.innerHTML = ok ? '<p><b style="color:#1E6B40">The chemistry program RDKit can read this as a molecule.</b></p><div class="struct" style="height:170px">' + rdkSvg(smi, 300, 170) + '</div>' : '<p><b style="color:' + C.red + '">The chemistry program RDKit cannot read this as a molecule.</b> A GPT writes text that follows common patterns. Nothing checks the chemistry, so some strings are not molecules. A longer memory makes this happen less often.</p>'; });
    async function render(){ await loadStructs(); bgBuild(); show(); }
    async function play(tok){
      await render(); ctx = []; done = false; show(); api.cap('At the start the model suggests the pieces that most often begin a molecule.'); await api.sleep(2800);
      for (var k = 0; k < 14 && !done; k++) { api.checkTok(tok); qs('[data-pick]', ctl).click(); await api.sleep(550); } api.cap('Each piece changes the chances for the next one. Repeating this writes a whole molecule.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: a head start ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'A head start', title: 'A model that has studied first learns faster',
  change: 'Choose how many labelled compounds are available, and compare the two models.',
  watch: 'Imagine two students asked to learn about drug clearance from a small pile of notes. One has never studied chemistry. The other has already read a large library about related molecules. The second student gets a head start. Models work the same way. Training a model first on a large related set is called pre-training. Here the same model learns HLM clearance in two ways: starting from nothing, and starting from a model that was first trained on thousands of values of logD, a close cousin of logP that is cheap to measure. Each result is the average of two runs on test compounds, using R2. R2 of 1 is perfect and 0 is no better than the overall average. Labelled compounds are the ones that have a lab value, and they are costly to measure.',
  build: function(api){
    var sel = 500, NS = [50, 100, 200, 500, 1000];
    var bp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Test score (R2) from saved runs of a neural network on HLM clearance</h4><div class="gb"></div>');
    var tp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="gt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Labelled compounds ' + inSeg(NS.map(function(n){ return n.toLocaleString(); }), 3, 'data-n'));
    function avg(mode, n){ var r = EXTRAS.chemprop_ft.runs.filter(function(x){ return x.mode === mode && x.n === n; }); return mean(r.map(function(x){ return x.r2; })); }
    function bar(label, sub, v, col){ var w = clamp(v, 0, 0.6) / 0.6 * 100; return '<div style="margin:14px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="font-size:30px;margin:0;color:' + col + '">' + v.toFixed(2) + '</span></div><div style="font-size:12px;margin-bottom:4px">' + sub + '</div><div style="height:18px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:18px;width:' + w + '%;background:' + col + ';transition:width .4s"></i></div></div>'; }
    function draw(){
      if (!EXTRAS) return; var a = avg('scratch', sel), b = avg('full', sel);
      qs('.gb', bp).innerHTML = '<div class="ol">With ' + sel.toLocaleString() + ' labelled compounds</div>' + bar('Starts from nothing', 'Learns only from these compounds', a, C.char) + bar('Starts from a pre-trained model', 'Studied logD data first, then these compounds', b, C.red) + '<div style="font-size:11px;color:' + C.mid + ';margin-top:6px">The bar scale runs from 0 to 0.6.</div>';
      qs('.gt', tp).innerHTML = '<div class="ol">The head start</div><div class="big" style="color:' + C.red + '">' + (b - a >= 0 ? '+' : '') + (b - a).toFixed(2) + '</div><div style="font-size:12px">score gained by studying first</div><p style="margin-top:12px">' + (sel <= 200 ? 'With very few labelled compounds the model that studied first is clearly ahead.' : (sel < 1000 ? 'The head start is still there as more compounds arrive.' : 'With many labelled compounds both models do well, but the one that studied first is still ahead.')) + '</p><p>The related set was logD data. It is cheap to measure, so there is a lot of it. It teaches the model general facts about molecules before it sees any clearance values.</p>';
    }
    qsa('[data-n] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = NS[+b.dataset.v]; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ await loadExtras(); draw(); }
    async function play(tok){ await render(); for (var k = 0; k < NS.length; k++) { api.checkTok(tok); sel = NS[k]; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); draw(); api.cap(k === 0 ? 'With 50 labelled compounds, neither model does well, but the pre-trained one starts ahead.' : (k === NS.length - 1 ? 'With 1,000 compounds the pre-trained model scores about twice as high.' : 'More compounds help both models.')); await api.sleep(2600); } api.cap(null); }
    return { render: render, play: play };
  } });

/* ---------- 06 Compare: fast or accurate ---------- */
LESSONS.compare.add({ level: 'beginner', short: 'Fast or accurate', title: 'A model can be slow, or inaccurate, or both',
  change: 'Click a model to read about it. Press Show the beaten model to see which one never makes sense to choose.',
  watch: 'Choosing a model is a trade. Some models are more accurate, and some take less time to train. Here are three models that were run on the HLM clearance data. The height shows accuracy on test compounds (R2, where 1 is perfect). The distance to the right shows how long training took, on a scale that grows by factors of ten. A model that is both slower and less accurate than another is called a dominated model, and there is never a reason to choose it.',
  build: function(api){
    var cur = 0, show = false, M = null, MD = { 'LightGBM': 'A model made of many small decision trees, added one after another. Each new tree fixes the mistakes of the ones before.', 'Random forest': 'A model that averages many decision trees, each trained on a different random draw of the compounds.', 'Chemprop': 'A neural network that reads the molecule as a drawing made of atoms and bonds.' };
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Accuracy against training time</h4><canvas width="530" height="350"></canvas>');
    var tp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="ct" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-show>Show the beaten model</button>');
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
      qs('.ct', tp).innerHTML = '<div class="ol">' + esc(m.n) + '</div><div style="display:flex;gap:22px"><div><div class="big" style="color:' + C.red + ';font-size:30px">' + m.v.toFixed(2) + '</div><div style="font-size:12px">accuracy (R2)</div></div><div><div class="big" style="font-size:30px">' + (m.t >= 10 ? Math.round(m.t) : m.t.toFixed(1)) + ' s</div><div style="font-size:12px">training time</div></div></div><p style="margin-top:10px">' + (MD[m.n.split(',')[0]] || '') + '</p><p>' + (d ? 'Another model is both more accurate and faster. This one is dominated, so there is no reason to choose it.' : (m.t > 20 ? 'Slow, but not beaten on both counts.' : 'Not beaten on both counts. Choosing between this and the others depends on whether you need speed or accuracy.')) + '</p><p style="font-size:12px;color:' + C.mid + '">These are saved runs, made earlier with more features than the lessons use. They show the idea of a trade, not a ranking for every problem.</p>';
    }
    qs('canvas', cp).addEventListener('click', function(e){ var c = qs('canvas', cp), r = c.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * 530, py = (e.clientY - r.top) / r.height * 350, h = c._hit || []; var best = -1, bd = 1e9; h.forEach(function(p, i){ var dd = Math.hypot(p[0] - px, p[1] - py); if (dd < bd) { bd = dd; best = i; } }); if (best >= 0 && bd < 40) { cur = best; draw(); } });
    qs('[data-show]', ctl).addEventListener('click', function(){ show = !show; this.textContent = show ? 'Hide' : 'Show the beaten model'; draw(); });
    async function render(){ await loadExtras(); draw(); }
    async function play(tok){ await render(); show = false; cur = 0; draw(); api.cap('Each dot is a model. Higher is more accurate. Further right is slower.'); await api.sleep(3200); show = true; qs('[data-show]', ctl).textContent = 'Hide'; draw(); api.cap('The grey model is beaten on both counts by the first model, so nobody would choose it.'); await api.sleep(3400); api.cap(null); }
    return { render: render, play: play };
  } });
