"use strict";
/* Extra beginner steps for lessons 02 (trees) and 08 (tree algorithms).
   Written for readers with no machine learning experience and minimal chemistry.
   Each step assumes only the earlier beginner steps. */

/* ---------- shared helpers ---------- */
function bt2Sample(key, n, seed){
  var k = EP + key; if (D.cache[k]) return D.cache[k]; var r = rng(seed), sh = D.TRAIN.slice();
  for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
  return (D.cache[k] = sh.slice(0, n));
}
/* sorted prefix sums of the lab value for one feature, over all training compounds: any cut can then be scored at once */
function bt2Pre(f){
  var k = EP + 'pre' + f; if (D.cache[k]) return D.cache[k]; var idx = D.TRAIN.slice().sort(function(a, b){ return D.COL[f][a] - D.COL[f][b]; }), n = idx.length, xs = new Float64Array(n), cs = new Float64Array(n + 1), cs2 = new Float64Array(n + 1);
  idx.forEach(function(i, q){ xs[q] = D.COL[f][i]; var y = D.Y[i]; cs[q + 1] = cs[q] + y; cs2[q + 1] = cs2[q] + y * y; });
  return (D.cache[k] = { xs: xs, cs: cs, cs2: cs2, n: n });
}
/* typical miss when the first m compounds form one group and the rest form another */
function bt2Miss(P, m){ var n = P.n, T = P.cs[n], T2 = P.cs2[n], sl = P.cs2[m] - P.cs[m] * P.cs[m] / m, sr = (T2 - P.cs2[m]) - (T - P.cs[m]) * (T - P.cs[m]) / (n - m); return Math.sqrt(Math.max(0, sl + sr) / n); }
function bt2Base(P){ var n = P.n, T = P.cs[n]; return Math.sqrt(Math.max(0, P.cs2[n] / n - (T / n) * (T / n))); }
function bt2Best(P){ var best = 20, bv = Infinity; for (var m = 20; m <= P.n - 20; m++) { var v = bt2Miss(P, m); if (v < bv) { bv = v; best = m; } } return { m: best, v: bv }; }
function bt2Num(v, d){ return (Math.abs(v) < Math.pow(10, -d) / 2 ? 0 : v).toFixed(d); }

/* ====================================================================
   Lesson 02, step: how a tree chooses its question
   ==================================================================== */
LESSONS.trees.add({ level: 'beginner', short: 'Choosing the best question', title: 'A tree picks the question that sorts compounds best',
  change: 'Drag the cut along the feature, or press Show the best cut. Switch the feature to try a different question.',
  watch: 'In the last step the tree asked questions such as "is logP above 3?". How does it choose which question to ask? It tries them all and keeps the best. A question is a cut: compounds on one side go into one group and the rest into another. Each group is predicted by its own average, shown as a line. The typical miss is the size of a typical mistake: how far a compound sits from its group average, with big misses counted extra. A good cut makes the groups tight, so the typical miss is small. The curve on the right shows the typical miss for every possible cut. The lowest point is the best cut.',
  build: function(api){
    var feat = 'logP', m = 0, FE = ['logP', 'TPSA', 'MW'];
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4 class="t"></h4><canvas width="530" height="350"></canvas>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:hidden', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div><canvas class="c2" width="294" height="180" style="margin-top:4px"></canvas>');
    var ctl = api.addControl('Question about ' + inSeg(FE, 0, 'data-f') + ' Cut position <input type="range" class="cut" style="width:240px"> <button type="button" class="btn sm" data-best>Show the best cut</button>');
    var rg = qs('.cut', ctl);
    function setup(){ var P = bt2Pre(feat), b = bt2Best(P); rg.min = 20; rg.max = P.n - 20; rg.step = Math.max(1, Math.round(P.n / 300)); return { P: P, b: b }; }
    function draw(){
      var S = setup(), P = S.P, n = P.n; m = clamp(m || Math.round(n * 0.15), 20, n - 20); rg.value = m;
      var cutV = P.xs[m], xlo = P.xs[Math.floor(n * 0.01)], xhi = P.xs[Math.floor(n * 0.99)], miss = bt2Miss(P, m), base = bt2Base(P), ml = P.cs[m] / m, mr = (P.cs[n] - P.cs[m]) / (n - m);
      qs('.t', cp).textContent = 'Each dot is a compound. The cut sends it left or right.';
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, sx = function(v){ return box.l + (clamp(v, xlo, xhi) - xlo) / (xhi - xlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      axes(g, box, [xlo, xhi], [D.ylo, D.yhi], feat, 'lab value', 4);
      var cx = sx(cutV); x.fillStyle = 'rgba(161,177,196,.18)'; x.fillRect(box.l, box.t, cx - box.l, box.b - box.t); x.fillStyle = 'rgba(225,36,42,.07)'; x.fillRect(cx, box.t, box.r - cx, box.b - box.t);
      x.fillStyle = 'rgba(52,55,63,.5)'; bt2Sample('bt2dots', 380, 4).forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[feat][i]), sy(D.Y[i]), 2.6, 0, 6.3); x.fill(); });
      x.strokeStyle = C.char; x.lineWidth = 2; x.beginPath(); x.moveTo(box.l + 4, sy(ml)); x.lineTo(cx - 4, sy(ml)); x.stroke(); x.strokeStyle = C.red; x.beginPath(); x.moveTo(cx + 4, sy(mr)); x.lineTo(box.r - 4, sy(mr)); x.stroke();
      x.strokeStyle = C.char; x.setLineDash([5, 4]); x.lineWidth = 1.5; x.beginPath(); x.moveTo(cx, box.t); x.lineTo(cx, box.b); x.stroke(); x.setLineDash([]);
      x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillStyle = C.char; x.fillText('group average ' + ml.toFixed(2), (box.l + cx) / 2, box.t + 14); x.fillStyle = C.red; x.fillText('group average ' + mr.toFixed(2), (cx + box.r) / 2, box.t + 14);
      var rows = FE.filter(function(f){ return D.COL[f]; }).map(function(f){ var Q = bt2Pre(f), bb = bt2Best(Q); return { f: f, m: bb.m, v: bb.v, c: Q.xs[bb.m] }; }), win = rows.reduce(function(a, b){ return b.v < a.v ? b : a; });
      qs('.nb', rp).innerHTML = '<div class="ol">Question: ' + feat + ' at most ' + fmtFeat(feat, cutV) + '?</div><div style="display:flex;gap:18px;align-items:flex-end"><div><div class="big" style="color:' + C.red + ';font-size:30px;margin:0">' + miss.toFixed(2) + '</div><div style="font-size:12px">typical miss with this cut</div></div><div><div class="big" style="font-size:22px;margin:0;color:' + C.mid + '">' + base.toFixed(2) + '</div><div style="font-size:12px">with no question</div></div></div>' +
        '<div style="margin-top:8px;font-size:12.5px">Best cut for ' + feat + ': at most <b>' + fmtFeat(feat, S.b.m < n ? P.xs[S.b.m] : 0) + '</b>, typical miss <b>' + S.b.v.toFixed(2) + '</b>. ' + (Math.abs(miss - S.b.v) < 0.003 ? '<b style="color:#1E6B40">You are at the best cut.</b>' : 'Your cut is not the best yet.') + '</div>' +
        '<div style="margin-top:6px;font-size:12.5px">The tree tries every feature. Best typical miss: ' + rows.map(function(r){ return (r.f === win.f ? '<b>' : '') + r.f + ' ' + r.v.toFixed(2) + (r.f === win.f ? '</b>' : ''); }).join(', ') + '. It asks about <b>' + win.f + '</b> first.</div>';
      var pts = []; for (var q = 20; q <= n - 20; q += Math.max(1, Math.round(n / 120))) if (P.xs[q] >= xlo && P.xs[q] <= xhi) pts.push([P.xs[q], bt2Miss(P, q)]);
      var lo = Math.floor((S.b.v - 0.01) * 50) / 50, hi = Math.ceil((base + 0.01) * 50) / 50;
      lineChart(qs('.c2', rp), [{ pts: pts, color: C.char, w: 2.4, nodot: true, label: 'every possible cut' }, { pts: [[xlo, base], [xhi, base]], color: C.bg, w: 1.6, dash: [5, 4], nodot: true, label: 'no question' }, { pts: [[cutV, miss]], color: C.red, w: 2, label: 'your cut' }], { xl: feat + ' at the cut', yl: 'typical miss', x0: xlo, x1: xhi, y0: lo, y1: hi });
    }
    rg.addEventListener('input', function(){ m = +rg.value; draw(); });
    qsa('[data-f] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ feat = FE[+b.dataset.v]; qsa('[data-f] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); m = Math.round(bt2Pre(feat).n * 0.15); draw(); }); });
    qs('[data-best]', ctl).addEventListener('click', function(){ var from = m, to = bt2Best(bt2Pre(feat)).m; animateFree(900, function(t){ m = Math.round(from + (to - from) * t); draw(); }); });
    async function render(){ draw(); }
    async function play(tok){
      var P = bt2Pre(feat), n = P.n, b = bt2Best(P).m; m = Math.round(n * 0.12); draw(); api.cap('A cut near the left edge puts only a few compounds in one group. The other group is almost everyone, so the typical miss barely improves.'); await api.sleep(3600);
      await api.anim(3600, function(t){ m = Math.round(n * 0.12 + (n * 0.88 - n * 0.12) * t); draw(); }); api.cap('Sliding the cut changes the groups. The typical miss moves up and down.'); await api.sleep(2200);
      var from = m; await api.anim(1400, function(t){ m = Math.round(from + (b - from) * t); draw(); }); api.cap('The lowest point of the curve is the best cut. The tree picks it, then repeats the search inside each group.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 02, step: a forest is a crowd that votes
   ==================================================================== */
LESSONS.trees.add({ level: 'beginner', short: 'A crowd of trees', title: 'Many trees voting beat one tree',
  change: 'Drag the slider to add trees to the crowd. Press Next compound to try another one.',
  watch: 'One tree can be unlucky. Its answer depends on exactly which compounds it studied. A forest fixes that by growing many trees, and each tree studies a different random draw of the training compounds. The trees then disagree a little. Each dot is one tree\'s guess for one compound. The red line is the average of the guesses, and that average is the forest\'s answer. The black line is the value the lab measured. Averaging cancels the unlucky guesses, so the crowd usually lands closer than a single tree. The numbers on the right check this on 300 test compounds, which no tree studied.',
  build: function(api){
    var K = 6, cur = 0, NT = 15, fr = null;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Each dot is one tree\'s guess for this compound</h4><canvas width="570" height="350"></canvas>');
    var rp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Trees in the crowd <input type="range" min="1" max="15" step="1" value="6" style="width:200px"> <b class="kv">6</b> <button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var rg = qs('input', ctl), kv = qs('.kv', ctl);
    function prep(){
      var key = EP + 'crowd'; if (D.cache[key]) return D.cache[key]; var feats = ['logP', 'TPSA', 'MW'].filter(function(f){ return D.COL[f]; }), f = fitForest(feats, D.TRAIN, NT, 6, 7), r = rng(12), sh = D.TEST.slice();
      for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
      var ev = sh.slice(0, 300), one = [], crowd = [], acc = new Float64Array(D.N);
      f.per.forEach(function(p, k){ one.push(rmse(ev, p)); for (var i = 0; i < D.N; i++) acc[i] += p[i]; var avg = new Float64Array(D.N); for (var q = 0; q < D.N; q++) avg[q] = acc[q] / (k + 1); crowd.push(rmse(ev, avg)); });
      return (D.cache[key] = { fr: f, order: sh, one: mean(one), crowd: crowd });
    }
    function draw(){
      var S = prep(), i = S.order[cur % S.order.length], k = K; kv.textContent = k;
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 30, r: g.w - 24, t: 30, b: g.h - 54 }, sx = function(v){ return box.l + (clamp(v, D.ylo, D.yhi) - D.ylo) / (D.yhi - D.ylo) * (box.r - box.l); };
      var guesses = S.fr.per.slice(0, k).map(function(p){ return p[i]; }), avg = mean(guesses), truth = D.Y[i];
      x.strokeStyle = C.silver; x.lineWidth = 1; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; var tk = niceTicks(D.ylo, D.yhi, 6); tk.v.forEach(function(v){ x.beginPath(); x.moveTo(sx(v), box.t); x.lineTo(sx(v), box.b); x.stroke(); x.fillText(tickText(v, tk.step), sx(v), box.b + 16); });
      x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('guessed lab value (' + unitWord() + ')', (box.l + box.r) / 2, box.b + 36);
      var rj = rng(i + 5); guesses.forEach(function(v, q){ var y = box.t + 22 + (q % 8) * ((box.b - box.t - 44) / 8) + rj() * 6; x.fillStyle = 'rgba(52,55,63,.7)'; x.beginPath(); x.arc(sx(v), y, 6, 0, 6.3); x.fill(); x.strokeStyle = 'rgba(52,55,63,.18)'; x.beginPath(); x.moveTo(sx(v), y); x.lineTo(sx(avg), y); x.stroke(); });
      x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); x.moveTo(sx(avg), box.t); x.lineTo(sx(avg), box.b); x.stroke(); x.fillStyle = C.red; x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = sx(avg) > box.r - 120 ? 'right' : 'left'; x.fillText('crowd average ' + avg.toFixed(2), sx(avg) + (x.textAlign === 'left' ? 6 : -6), box.t - 6 + 4);
      x.strokeStyle = C.char; x.lineWidth = 2.5; x.setLineDash([6, 4]); x.beginPath(); x.moveTo(sx(truth), box.t + 10); x.lineTo(sx(truth), box.b); x.stroke(); x.setLineDash([]); x.fillStyle = C.char; x.textAlign = Math.abs(sx(truth) - sx(avg)) < 150 ? (sx(truth) < sx(avg) ? 'right' : 'left') : 'left'; x.fillText('lab ' + truth.toFixed(2), sx(truth) + (x.textAlign === 'left' ? 6 : -6), box.b - 6);
      var spread = Math.max.apply(null, guesses) - Math.min.apply(null, guesses);
      qs('.nb', rp).innerHTML = '<div class="ol">This compound</div><p style="margin:2px 0 8px">The ' + k + ' tree' + (k > 1 ? 's' : '') + ' guess between <b>' + Math.min.apply(null, guesses).toFixed(2) + '</b> and <b>' + Math.max.apply(null, guesses).toFixed(2) + '</b>. The crowd average is <b style="color:' + C.red + '">' + avg.toFixed(2) + '</b>. The lab measured <b>' + truth.toFixed(2) + '</b>.</p>' +
        '<div class="ol" style="margin-top:12px">On 300 test compounds</div><div style="display:flex;gap:16px;margin-top:4px"><div><div class="big" style="color:' + C.mid + ';font-size:28px;margin:0">' + S.one.toFixed(2) + '</div><div style="font-size:12px">typical miss of one tree</div></div><div><div class="big" style="color:' + C.red + ';font-size:28px;margin:0">' + S.crowd[k - 1].toFixed(2) + '</div><div style="font-size:12px">typical miss of a crowd of ' + k + '</div></div></div>' +
        '<p style="margin-top:10px">' + (k === 1 ? 'With one tree there is nothing to average.' : (S.crowd[k - 1] < S.one - 0.005 ? 'The crowd misses by less than a typical single tree.' : 'The crowd is about as good as a single tree here.')) + ' Typical miss is the size of a typical mistake. Smaller is better.</p>' +
        '<p style="font-size:12px;color:' + C.mid + '">Each tree was grown on a different random draw of the training compounds, and each looked at only some of the features when choosing a question.</p>';
    }
    rg.addEventListener('input', function(){ K = +rg.value; draw(); });
    qs('[data-next]', ctl).addEventListener('click', function(){ cur++; draw(); });
    async function render(){ draw(); }
    async function play(tok){
      K = 1; rg.value = 1; draw(); api.cap('One tree makes one guess. It may be far from the lab value.'); await api.sleep(3000);
      await api.anim(4200, function(t){ K = Math.max(1, Math.round(1 + 14 * t)); rg.value = K; draw(); }); api.cap('Each new tree adds a guess. The average settles, and it usually lands closer to the lab value.'); await api.sleep(3400); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 02, step: boosting as a team that fixes earlier mistakes
   ==================================================================== */
LESSONS.trees.add({ level: 'beginner', short: 'Fixing mistakes round by round', title: 'Another way to use many trees: fix the leftover mistakes',
  change: 'Drag the slider to add rounds of correction, or press Walk through.',
  watch: 'A forest grows its trees at the same time and averages them. There is another way. Grow one small tree first. It makes mistakes. Then grow a second tree whose only job is to fix those leftover mistakes, and add its fix to the answer, a little at a time. Then a third tree fixes what is still wrong, and so on. Each step is called a round. This is called boosting. The red line is the model, and the thin red bars show how far each compound is from the line. Watch the bars shrink as the rounds go by. Here the trees look only at logP, so the model is a line you can see.',
  build: function(api){
    var R = 6, MAXR = 30;
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Model after the chosen number of rounds</h4><canvas width="530" height="350"></canvas>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:hidden', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div><canvas class="c2" width="294" height="200" style="margin-top:4px"></canvas>');
    var ctl = api.addControl('Rounds of correction <input type="range" min="0" max="30" step="1" value="6" style="width:240px"> <b class="rv">6</b>');
    var rg = qs('input', ctl), rv = qs('.rv', ctl);
    function prep(){
      var key = EP + 'boostbeg'; if (D.cache[key]) return D.cache[key]; var m = fitBoost(['logP'], D.TRAIN, MAXR, 0.3, 2, 1), miss = [], base = new Float64Array(D.N).fill(m.base); miss.push(rmse(D.TEST, base)); m.snaps.forEach(function(s){ miss.push(rmse(D.TEST, s.p)); });
      var dots = bt2Sample('bt2boost', 360, 6).slice().sort(function(a, b){ return D.COL.logP[a] - D.COL.logP[b]; });
      return (D.cache[key] = { m: m, miss: miss, dots: dots, base: base });
    }
    function draw(){
      var S = prep(); rv.textContent = R; var p = R === 0 ? S.base : S.m.snaps[R - 1].p, xs = S.dots.map(function(i){ return D.COL.logP[i]; }), xlo = sortedCol('logP')[Math.floor(D.N * 0.01)], xhi = sortedCol('logP')[Math.floor(D.N * 0.99)];
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, sx = function(v){ return box.l + (clamp(v, xlo, xhi) - xlo) / (xhi - xlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      axes(g, box, [xlo, xhi], [D.ylo, D.yhi], 'logP', 'lab value', 4);
      S.dots.forEach(function(i, q){ if (q % 4 !== 0) return; x.strokeStyle = 'rgba(225,36,42,.28)'; x.lineWidth = 1.4; x.beginPath(); x.moveTo(sx(D.COL.logP[i]), sy(D.Y[i])); x.lineTo(sx(D.COL.logP[i]), sy(p[i])); x.stroke(); });
      x.fillStyle = 'rgba(52,55,63,.5)'; S.dots.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL.logP[i]), sy(D.Y[i]), 2.5, 0, 6.3); x.fill(); });
      x.strokeStyle = C.red; x.lineWidth = 3.2; x.lineJoin = 'round'; x.beginPath(); S.dots.forEach(function(i, q){ var px = sx(D.COL.logP[i]), py = sy(p[i]); q ? x.lineTo(px, py) : x.moveTo(px, py); }); x.stroke();
      var mi = S.miss[R], m0 = S.miss[0], lo = Math.floor((Math.min.apply(null, S.miss) - 0.01) * 50) / 50, hi = Math.ceil((m0 + 0.01) * 50) / 50, pts = S.miss.slice(0, R + 1).map(function(v, q){ return [q, v]; });
      qs('.nb', rp).innerHTML = '<div class="ol">' + (R === 0 ? 'Round 0: just the overall average' : 'After ' + R + ' round' + (R > 1 ? 's' : '')) + '</div><div style="display:flex;gap:16px;align-items:flex-end"><div><div class="big" style="color:' + C.red + ';font-size:30px;margin:0">' + mi.toFixed(2) + '</div><div style="font-size:12px">typical miss on test compounds</div></div></div><p style="margin-top:8px">' + (R === 0 ? 'Every compound gets the same answer, so the thin bars are long.' : (R < 6 ? 'Each round fixes some of the leftover mistakes, and the bars shorten.' : 'Later rounds fix less and less, so the gains get small.')) + '</p>';
      lineChart(qs('.c2', rp), [{ pts: pts, color: C.red, w: 2.6, label: 'typical miss, test compounds' }], { xl: 'rounds of correction', yl: 'typical miss', x0: 0, x1: MAXR, y0: lo, y1: hi });
    }
    rg.addEventListener('input', function(){ R = +rg.value; draw(); });
    async function render(){ draw(); }
    async function play(tok){
      R = 0; rg.value = 0; draw(); api.cap('Round 0: the model is just the overall average. The thin bars show every mistake.'); await api.sleep(3200);
      await api.anim(5200, function(t){ R = Math.round(MAXR * t); rg.value = R; draw(); }); api.cap('Each round adds a small tree that fixes some of what is left. The bars shrink and the typical miss falls.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 08, step: how a cut is scored
   ==================================================================== */
LESSONS.treealgo.add({ level: 'beginner', short: 'Scoring a cut', title: 'How a cut is scored: add up the squares',
  change: 'Drag the cut between compounds, or press Show the best cut. Compare the total of the squares before and after.',
  watch: 'A tree needs a way to say that one cut is better than another. The method is simple arithmetic. Take a group of compounds and find its average. For each compound, the miss is the gap between its lab value and the average. Draw a square whose side is that miss, and its area is the miss times itself. Big misses make much bigger squares, so they count extra. Add up the areas. This total is the sum of squared errors, or SSE for short. A good cut splits the compounds into two groups that each sit close to their own average, so the squares shrink. The gain of a cut is how much smaller the total becomes: the total before minus the total after.',
  build: function(api){
    var k = 0, N = 10;
    var cp = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<canvas width="880" height="384"></canvas>');
    var ctl = api.addControl('Cut after compound number <input type="range" min="0" max="9" step="1" value="0" style="width:200px"> <b class="kv">none</b> <button type="button" class="btn sm" data-best>Show the best cut</button>');
    var rg = qs('input', ctl), kv = qs('.kv', ctl);
    function pick(){
      var key = EP + 'sqcut'; if (D.cache[key]) return D.cache[key]; var r = rng(21), sh = D.TRAIN.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
      var ids = sh.slice(0, N).sort(function(a, b){ return D.COL.logP[a] - D.COL.logP[b]; }); return (D.cache[key] = ids);
    }
    function sse(vals){ if (!vals.length) return 0; var m = mean(vals); return vals.reduce(function(s, v){ return s + (v - m) * (v - m); }, 0); }
    function cutSSE(ids, c){ var v = ids.map(function(i){ return D.Y[i]; }); return c === 0 ? sse(v) : sse(v.slice(0, c)) + sse(v.slice(c)); }
    function bestCut(ids){ var b = 1, bv = Infinity; for (var c = 1; c < ids.length; c++) { var v = cutSSE(ids, c); if (v < bv) { bv = v; b = c; } } return b; }
    function draw(){
      var ids = pick(), v = ids.map(function(i){ return D.Y[i]; }), g = cv(qs('canvas', cp)), x = g.x, W = g.w, left = 60, slot = (W - left - 20) / N, top = 66, pb = 236, sb = g.h - 12;
      var lo = Math.min.apply(null, v) - 0.15, hi = Math.max.apply(null, v) + 0.15, sy = function(val){ return pb - (val - lo) / (hi - lo) * (pb - top); };
      var groups = k === 0 ? [[0, N]] : [[0, k], [k, N]], total0 = cutSSE(ids, 0), tot = cutSSE(ids, k), maxMiss = Math.max.apply(null, v.map(function(y){ return Math.abs(y - mean(v)); })) || 1, scale = 52 / Math.max(maxMiss, 0.4);
      kv.textContent = k === 0 ? 'none' : k;
      x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText('lab value', 54, top - 20); x.textAlign = 'left'; x.fillText('Compounds in order of logP, lowest on the left. The line is the average of each group.', left, 14);
      x.strokeStyle = C.silver; x.lineWidth = 1; var tk = niceTicks(lo, hi, 4); x.textAlign = 'right'; tk.v.forEach(function(t){ x.beginPath(); x.moveTo(left - 4, sy(t)); x.lineTo(W - 20, sy(t)); x.stroke(); x.fillText(tickText(t, tk.step), left - 8, sy(t) + 4); });
      groups.forEach(function(gr, gi){ var m = mean(v.slice(gr[0], gr[1])), x0 = left + gr[0] * slot, x1 = left + gr[1] * slot; x.fillStyle = gi ? 'rgba(225,36,42,.07)' : 'rgba(161,177,196,.2)'; x.fillRect(x0 + 2, top - 6, x1 - x0 - 4, pb - top + 12); x.strokeStyle = gi ? C.red : C.char; x.lineWidth = 2.6; x.beginPath(); x.moveTo(x0 + 6, sy(m)); x.lineTo(x1 - 6, sy(m)); x.stroke(); x.fillStyle = gi ? C.red : C.char; x.font = '700 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('average ' + m.toFixed(2), (x0 + x1) / 2, top - 14);
        for (var q = gr[0]; q < gr[1]; q++) { var cxp = left + (q + 0.5) * slot, miss = v[q] - m; x.strokeStyle = gi ? 'rgba(225,36,42,.6)' : 'rgba(52,55,63,.5)'; x.lineWidth = 1.5; x.beginPath(); x.moveTo(cxp, sy(v[q])); x.lineTo(cxp, sy(m)); x.stroke(); x.fillStyle = C.char; x.beginPath(); x.arc(cxp, sy(v[q]), 5, 0, 6.3); x.fill();
          var side = Math.abs(miss) * scale; x.fillStyle = gi ? 'rgba(225,36,42,.28)' : 'rgba(111,129,150,.38)'; x.strokeStyle = gi ? C.red : '#6F8196'; x.lineWidth = 1.2; x.fillRect(cxp - side / 2, sb - side - 14, side, side); x.strokeRect(cxp - side / 2, sb - side - 14, side, side); x.fillStyle = C.char; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText((miss * miss).toFixed(2), cxp, sb); } });
      x.fillStyle = C.mid; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText('Below each compound: a square with side equal to its miss. The number is its area.', left, pb + 24);
      var gain = total0 - tot; x.textAlign = 'right'; x.font = '700 14px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.char; x.fillText('Total of squares, no cut: ' + total0.toFixed(2), W - 20, pb + 24);
      x.fillStyle = C.red; x.fillText(k === 0 ? 'Choose a cut to see the gain.' : 'With this cut: ' + tot.toFixed(2) + '.  Gain: ' + gain.toFixed(2), W - 20, pb + 44);
      if (k > 0 && k === bestCut(ids)) { x.fillStyle = '#1E6B40'; x.fillText('This is the best cut for these ten compounds.', W - 20, pb + 64); }
    }
    rg.addEventListener('input', function(){ k = +rg.value; draw(); });
    qs('[data-best]', ctl).addEventListener('click', function(){ k = bestCut(pick()); rg.value = k; draw(); });
    async function render(){ draw(); }
    async function play(tok){
      k = 0; rg.value = 0; draw(); api.cap('With no cut, all ten compounds share one average. The squares show how far each one is from it.'); await api.sleep(3600);
      k = 2; rg.value = 2; draw(); api.cap('A poor cut leaves the squares large. The gain is small.'); await api.sleep(3200);
      k = bestCut(pick()); rg.value = k; draw(); api.cap('The best cut makes two tight groups. The squares shrink, and the gain is as large as it can be. The tree keeps the cut with the biggest gain.'); await api.sleep(4200); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 08, step: bins
   ==================================================================== */
LESSONS.treealgo.add({ level: 'beginner', short: 'Fewer places to cut', title: 'Grouping values into bins makes the search fast',
  change: 'Choose how many bins to use. Compare the best cut that the bins allow with the best cut of all.',
  watch: 'To find the best cut for a feature, a tree could test a cut between every pair of neighbouring compounds. With thousands of compounds that is thousands of cuts for every feature, at every question. A faster way is to sort the values into bins first. A bin is a range of values treated as one value, so a cut is only allowed between bins. Fewer bins means far fewer cuts to test. The cost is that the best exact cut might fall inside a bin and be missed. Here 120 compounds are used so you can see the cuts. The grey lines show the cuts that are allowed. The red line is the best allowed cut, and the dashed black line is the best cut of all.',
  build: function(api){
    var NB = 8, BINS = [3, 5, 8, 12, 20, 40, 120], NS = 120;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Compounds by logP. Grey lines mark the cuts that are allowed</h4><canvas width="570" height="350"></canvas>');
    var rp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Number of bins ' + inSeg(BINS.map(function(b){ return b === NS ? 'all' : b; }), 2, 'data-b'));
    function data(){
      var key = EP + 'binsd'; if (D.cache[key]) return D.cache[key]; var ids = bt2Sample('bt2bin', NS, 31).slice().sort(function(a, b){ return D.COL.logP[a] - D.COL.logP[b]; }), y = ids.map(function(i){ return D.Y[i]; }), xs = ids.map(function(i){ return D.COL.logP[i]; });
      var cs = [0], cs2 = [0]; y.forEach(function(v, q){ cs.push(cs[q] + v); cs2.push(cs2[q] + v * v); });
      function gainAt(m){ var n = NS, T = cs[n], T2 = cs2[n], sl = cs2[m] - cs[m] * cs[m] / m, sr = (T2 - cs2[m]) - (T - cs[m]) * (T - cs[m]) / (n - m); return (T2 - T * T / n) - (sl + sr); }
      var bestAll = 1, bg = -1; for (var m = 1; m < NS; m++) { var gg = gainAt(m); if (gg > bg) { bg = gg; bestAll = m; } }
      return (D.cache[key] = { ids: ids, y: y, xs: xs, gainAt: gainAt, bestAll: bestAll, bg: bg });
    }
    function draw(){
      var S = data(), nb = NB, edges = []; for (var q = 1; q < nb; q++) { var m = Math.round(q * NS / nb); if (m > 0 && m < NS && edges.indexOf(m) < 0) edges.push(m); } if (nb >= NS) { edges = []; for (var q2 = 1; q2 < NS; q2++) edges.push(q2); }
      var bestB = edges[0], bgv = -1; edges.forEach(function(m){ var gg = S.gainAt(m); if (gg > bgv) { bgv = gg; bestB = m; } });
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, xlo = Math.min.apply(null, S.xs), xhi = Math.max.apply(null, S.xs), sx = function(v){ return box.l + (v - xlo) / (xhi - xlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      axes(g, box, [xlo, xhi], [D.ylo, D.yhi], 'logP', 'lab value', 4);
      var mid = function(m){ return (S.xs[m - 1] + S.xs[m]) / 2; };
      x.strokeStyle = 'rgba(111,129,150,.55)'; x.lineWidth = 1; edges.forEach(function(m){ var px = sx(mid(m)); x.beginPath(); x.moveTo(px, box.t); x.lineTo(px, box.b); x.stroke(); });
      x.fillStyle = 'rgba(52,55,63,.7)'; S.xs.forEach(function(v, q){ x.beginPath(); x.arc(sx(v), sy(S.y[q]), 3, 0, 6.3); x.fill(); });
      x.strokeStyle = C.char; x.setLineDash([6, 4]); x.lineWidth = 2; var pa = sx(mid(S.bestAll)); x.beginPath(); x.moveTo(pa, box.t); x.lineTo(pa, box.b); x.stroke(); x.setLineDash([]);
      x.strokeStyle = C.red; x.lineWidth = 3; var pb2 = sx(mid(bestB)); x.beginPath(); x.moveTo(pb2, box.t); x.lineTo(pb2, box.b); x.stroke();
      var keep = bgv / S.bg * 100, nFull = D.TRAIN.length;
      qs('.nb', rp).innerHTML = '<div class="ol">' + (nb >= NS ? 'No bins: every value' : nb + ' bins') + '</div><div style="display:flex;gap:16px"><div><div class="big" style="color:' + C.red + ';font-size:30px;margin:0">' + edges.length + '</div><div style="font-size:12px">cuts to test here</div></div><div><div class="big" style="font-size:30px;margin:0;color:' + C.mid + '">' + Math.round(keep) + '%</div><div style="font-size:12px">of the best possible gain</div></div></div>' +
        '<p style="margin-top:10px">' + (keep > 99.5 ? 'The best allowed cut is the best cut of all.' : (keep > 90 ? 'The best allowed cut is close to the best cut of all.' : 'Too few bins: the best cut fell inside a bin and was missed.')) + '</p>' +
        '<p>With all <b>' + nFull.toLocaleString() + '</b> training compounds, an exact search tests up to <b>' + (nFull - 1).toLocaleString() + '</b> cuts for each feature. With 64 bins it tests at most <b>63</b>. Tree libraries use bins like these, because the cost of missing a little gain is small.</p><p style="font-size:12px;color:' + C.mid + '">Gain here is the same measure as in the last step: how much a cut shrinks the total of the squared misses.</p>';
    }
    qsa('[data-b] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ NB = BINS[+b.dataset.v]; qsa('[data-b] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); draw(); }); });
    async function render(){ draw(); }
    async function play(tok){
      for (var q = BINS.length - 1; q >= 0; q--) { api.checkTok(tok); NB = BINS[q]; qsa('[data-b] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === q); }); draw(); api.cap(q === BINS.length - 1 ? 'Testing a cut between every pair of compounds: many cuts, slow search.' : (q === 0 ? 'Only three bins: very few cuts, but the best cut can be missed.' : 'Fewer bins means fewer cuts to test. The best allowed cut is often nearly as good.')); await api.sleep(2200); }
      NB = 8; qsa('[data-b] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === 2); }); draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 08, step: random draws with replacement
   ==================================================================== */
LESSONS.treealgo.add({ level: 'beginner', short: 'Random draws', title: 'A random draw leaves some compounds out and repeats others',
  change: 'Press Draw again for a new random draw, or press Repeat 200 times.',
  watch: 'A forest trains each tree on a different random draw of the compounds. Here is how a draw works. Put the compounds in a bag. Pull one out at random, write it down, and put it back. Do that as many times as there are compounds. Because each one goes back, a compound can be drawn twice, or three times, or never. The compounds drawn zero times are called out-of-bag for that tree. The tree never studied them, so they make a fair test for it. The tiles below show 24 compounds. A bigger tile means the compound was drawn more times.',
  build: function(api){
    var N = 24, counts = new Array(N).fill(0), draws = [], avg = null, rnd = Math.random;
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Draw 24 times from a bag of 24 compounds, putting each one back</h4><canvas width="530" height="350"></canvas>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-draw>Draw again</button> <button type="button" class="btn sm ghost" data-rep>Repeat 200 times</button>');
    function newDraw(){ counts = new Array(N).fill(0); draws = []; for (var q = 0; q < N; q++) { var c = Math.floor(rnd() * N); counts[c]++; draws.push(c); } }
    function draw(upTo){
      var shown = upTo === undefined ? draws.length : upTo, cs = new Array(N).fill(0); for (var q = 0; q < shown; q++) cs[draws[q]]++;
      var g = cv(qs('canvas', cp)), x = g.x, cols = 8, rows = 3, cw = g.w / cols, ch = (g.h - 20) / rows;
      for (var i = 0; i < N; i++) { var cx = (i % cols + 0.5) * cw, cy = Math.floor(i / cols) * ch + ch / 2 + 6, n = cs[i], r = n === 0 ? 17 : 17 + n * 5;
        x.beginPath(); x.arc(cx, cy, r, 0, 6.3); if (n === 0) { x.fillStyle = '#fff'; x.fill(); x.setLineDash([4, 3]); x.strokeStyle = C.bg; x.lineWidth = 1.8; x.stroke(); x.setLineDash([]); } else { x.fillStyle = n === 1 ? 'rgba(52,55,63,.82)' : (n === 2 ? 'rgba(225,36,42,.8)' : C.dred); x.fill(); }
        x.fillStyle = n === 0 ? C.mid : '#fff'; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('#' + (i + 1), cx, cy + 4); if (n > 1) { x.fillStyle = C.char; x.font = '700 11px Aptos, "Segoe UI", sans-serif'; x.fillText(n + ' times', cx, cy + r + 13); } }
      var z = cs.filter(function(c){ return c === 0; }).length, one = cs.filter(function(c){ return c === 1; }).length, two = cs.filter(function(c){ return c === 2; }).length, more = cs.filter(function(c){ return c >= 3; }).length;
      qs('.nb', rp).innerHTML = '<div class="ol">This draw (' + shown + ' of ' + N + ' pulls)</div>' + [['Never drawn (out-of-bag)', z, C.bg], ['Drawn once', one, C.char], ['Drawn twice', two, C.red], ['Drawn three or more times', more, C.dred]].map(function(r){ return '<div style="display:grid;grid-template-columns:150px 1fr 26px;gap:8px;align-items:center;margin:6px 0;font-size:12.5px"><span>' + r[0] + '</span><div style="height:12px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:12px;width:' + (r[1] / N * 100) + '%;border-radius:3px;background:' + r[2] + '"></i></div><b style="text-align:right">' + r[1] + '</b></div>'; }).join('') +
        '<p style="margin-top:10px">' + (shown < N ? 'The draw is in progress.' : 'Out-of-bag in this draw: <b>' + z + '</b> of ' + N + ' compounds, about <b>' + Math.round(100 * z / N) + '%</b>.') + '</p>' +
        '<div class="ol" style="margin-top:10px">Many draws</div><p style="margin:2px 0">' + (avg === null ? 'Press Repeat 200 times to see the average share left out.' : 'Over 200 draws, <b style="color:' + C.red + '">' + avg.toFixed(0) + '%</b> of the compounds were left out each time. For large sets the share settles near 37%, which is about 1 in 3.') + '</p>' +
        '<p style="font-size:12px;color:' + C.mid + '">Each tree in a forest gets its own draw. The compounds it left out can be used to check how well it learned.</p>';
    }
    qs('[data-draw]', ctl).addEventListener('click', function(){ newDraw(); draw(); });
    qs('[data-rep]', ctl).addEventListener('click', function(){ var tot = 0; for (var r = 0; r < 200; r++) { var c = new Array(N).fill(0); for (var q = 0; q < N; q++) c[Math.floor(rnd() * N)]++; tot += c.filter(function(v){ return v === 0; }).length / N; } avg = tot / 200 * 100; draw(); });
    async function render(){ if (!draws.length) newDraw(); draw(); }
    async function play(tok){
      newDraw(); draw(0); api.cap('A bag holds 24 compounds. Pull one out, note it, and put it back.'); await api.sleep(2600);
      for (var q = 1; q <= N; q++) { api.checkTok(tok); draw(q); await api.sleep(ANIM ? 170 : 0); }
      api.cap('Some compounds were drawn more than once. The dashed tiles were never drawn. They are out-of-bag for this tree.'); await api.sleep(3800); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- guide entries ---------- */
GUIDE.steps['trees/Choosing the best question'] = { notice: 'The typical miss is lowest at the best cut. The tree picks that cut and repeats the search inside each group.', q: 'Why does the tree try every feature and every cut?', a: 'It cannot know in advance which question sorts the compounds best. It scores them all and keeps the one with the smallest typical miss.' };
GUIDE.steps['trees/A crowd of trees'] = { notice: 'The crowd average usually lands nearer the lab value than most single guesses.', q: 'Why do the trees in a forest disagree?', a: 'Each one studied a different random draw of the training compounds, so each learned slightly different questions.' };
GUIDE.steps['trees/Fixing mistakes round by round'] = { notice: 'The thin bars shorten quickly at first and slowly later.', q: 'What is each new tree trained to do?', a: 'To fix the mistakes the earlier trees left behind.' };
GUIDE.steps['treealgo/Scoring a cut'] = { notice: 'The best cut gives the smallest total of squares.', q: 'Why use squares instead of the plain gaps?', a: 'A square makes a big miss count much more than a small one, so a cut that leaves a few large misses scores badly.' };
GUIDE.steps['treealgo/Fewer places to cut'] = { notice: 'Even with few bins, the best allowed cut keeps most of the gain.', q: 'What is the price of using bins?', a: 'The best exact cut may fall inside a bin and be missed, so the chosen cut can be slightly worse.' };
GUIDE.steps['treealgo/Random draws'] = { notice: 'About one compound in three is never drawn.', q: 'Why are the out-of-bag compounds useful?', a: 'The tree never studied them, so they make a fair test of that tree.' };
