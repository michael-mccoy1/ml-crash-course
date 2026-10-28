"use strict";
/* Extra beginner steps for lessons 06 and 07, and two more for lesson 01.
   Written for readers with no machine learning experience and minimal chemistry.
   Every invented example is labelled as invented on the page. */

/* ---------- shared helpers ---------- */
function m2Gauss(r){ var u = Math.max(r(), 1e-9), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * v); }
function m2Sd(a){ var m = mean(a); return Math.sqrt(mean(a.map(function(v){ return (v - m) * (v - m); }))) || 1; }
function m2Pick(arr, n, seed){ var r = rng(seed), sh = arr.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; } return sh.slice(0, n); }
function m2Score(ys, ps){ var my = mean(ys), a = 0, b = 0; for (var i = 0; i < ys.length; i++) { a += (ys[i] - ps[i]) * (ys[i] - ps[i]); b += (ys[i] - my) * (ys[i] - my); } return b ? 1 - a / b : 0; }
/* a plain scatter plot with axes, for pairs of numbers that are not tied to the dataset */
function m2Scatter(el, xs, ys, xr, yr, xl, yl, o){
  o = o || {}; var g = cv(el), box = { l: 52, r: g.w - 14, t: 14, b: g.h - 40 }, x = g.x; axes(g, box, xr, yr, xl, yl, 4);
  var sx = function(v){ return box.l + (v - xr[0]) / (xr[1] - xr[0]) * (box.r - box.l); }, sy = function(v){ return box.b - (v - yr[0]) / (yr[1] - yr[0]) * (box.b - box.t); };
  if (o.diag) { x.strokeStyle = C.char; x.setLineDash([5, 5]); x.beginPath(); x.moveTo(sx(xr[0]), sy(yr[0])); x.lineTo(sx(xr[1]), sy(yr[1])); x.stroke(); x.setLineDash([]); }
  if (o.under) o.under(x, sx, sy, box);
  x.fillStyle = o.fill || 'rgba(225,36,42,.6)'; for (var i = 0; i < xs.length; i++) { x.beginPath(); x.arc(sx(clamp(xs[i], xr[0], xr[1])), sy(clamp(ys[i], yr[0], yr[1])), o.r || 3.4, 0, 6.3); x.fill(); }
  if (o.over) o.over(x, sx, sy, box); g.sx = sx; g.sy = sy; g.box = box; return g;
}
function m2Big(label, num, sub, col){ return '<div style="margin:0 0 12px"><div class="ol" style="margin-bottom:0">' + label + '</div><div style="display:flex;align-items:baseline;gap:10px"><span class="big" style="color:' + (col || C.red) + ';font-size:30px;margin:0">' + num + '</span></div><div style="font-size:12px;color:' + C.char + '">' + sub + '</div></div>'; }

/* ---------- 06 Compare: what the scores mean ---------- */
LESSONS.compare.add({ level: 'beginner', short: 'What the scores mean', title: 'A score is a grade for the predictions',
  change: 'Drag the two sliders and watch the three numbers change.',
  watch: 'Three numbers are used to grade a model. Each one answers a different question. The score, called R2, is a grade from 0 to 1. A model that predicts the overall average for every compound earns 0. A model that is exactly right earns 1. The typical miss, called RMSE, is how far off the predictions usually are, in the same units as the lab value. The ranking number, called Spearman, asks only whether the model put the compounds in the right order: 1 means the order is perfect. The four small pictures use invented dots to show what a few scores look like. The large picture uses real lab values with invented predictions, so that you can control how good the predictions are.',
  build: function(api){
    var tp = api.el('pn', 'left:28px;top:20px;width:904px;height:176px', '<h4>What different scores look like: invented dots, one per compound. Across is the lab value, up is the prediction, and the dashed line is a perfect prediction.</h4><canvas width="880" height="140"></canvas>');
    var cp = api.el('pn', 'left:28px;top:206px;width:520px;height:314px', '<h4>Real lab values, invented predictions</h4><canvas width="490" height="262"></canvas>');
    var np = api.el('pn', 'left:568px;top:206px;width:364px;height:314px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('How noisy the predictions are <input type="range" data-s min="0.05" max="1.2" step="0.01" value="0.45" style="width:150px"> Squeeze toward the average <input type="range" data-k min="0.2" max="1" step="0.01" value="1" style="width:150px">');
    var rs = qs('[data-s]', ctl), rk = qs('[data-k]', ctl), mini = null, S = null;
    function drawMini(){
      var g = cv(qs('canvas', tp)), x = g.x, rr = rng(21), n = 60, M = [], E = [], k, i;
      if (!mini) { for (i = 0; i < n; i++) { M.push(clamp(m2Gauss(rr), -2.2, 2.2)); E.push(clamp(m2Gauss(rr), -2.2, 2.2)); } mini = { M: M, E: E }; }
      M = mini.M; E = mini.E; var targets = [0, 0.3, 0.6, 0.9], notes = ['no better than the average', 'a rough guide, with wide scatter', 'close, with some scatter', 'very close to the lab value'];
      for (k = 0; k < 4; k++) {
        var ox = 6 + k * 219, w = 200, top = 4, bot = 92, s = 0, lo_ = 0, hi_ = 1.6; if (k) { for (var it = 0; it < 24; it++) { s = (lo_ + hi_) / 2; if (m2Score(M, M.map(function(m, j){ return m + s * E[j]; })) > targets[k]) lo_ = s; else hi_ = s; } } var P = M.map(function(m, j){ return k === 0 ? 0.04 * E[j] : m + s * E[j]; }), sc = m2Score(M, P), sx = function(v){ return ox + 8 + (clamp(v, -2.8, 2.8) + 2.8) / 5.6 * (w - 16); }, sy = function(v){ return bot - (clamp(v, -2.8, 2.8) + 2.8) / 5.6 * (bot - top); };
        x.strokeStyle = C.silver; x.lineWidth = 1; x.strokeRect(ox, top, w, bot - top); x.strokeStyle = C.char; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(sx(-2.8), sy(-2.8)); x.lineTo(sx(2.8), sy(2.8)); x.stroke(); x.setLineDash([]);
        x.fillStyle = 'rgba(225,36,42,.65)'; M.forEach(function(m, j){ x.beginPath(); x.arc(sx(m), sy(P[j]), 2.8, 0, 6.3); x.fill(); });
        x.fillStyle = C.char; x.textAlign = 'center'; x.font = '700 13px Aptos, "Segoe UI", sans-serif'; x.fillText('score ' + (k === 0 ? '0' : sc.toFixed(1)), ox + w / 2, bot + 16); x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.fillText(notes[k], ox + w / 2, bot + 31);
      }
    }
    function prep(){ if (S) return S; var idx = m2Pick(D.TEST, 150, 3), r = rng(31), E = idx.map(function(){ return m2Gauss(r); }), ys = idx.map(function(i){ return D.Y[i]; }); S = { idx: idx, E: E, mu: mean(ys), sd: m2Sd(ys) }; return S; }
    function draw(){
      drawMini(); var P = prep(), s = +rs.value, k = +rk.value, pred = new Float64Array(D.N);
      P.idx.forEach(function(i, j){ var raw = D.Y[i] + s * P.sd * P.E[j]; pred[i] = P.mu + k * (raw - P.mu); });
      scatterPV(qs('canvas', cp), pred, P.idx, '', { max: 150 });
      var sc = r2(P.idx, pred), rm = rmse(P.idx, pred), sp = spearman(P.idx, pred);
      qs('.nb', np).innerHTML = m2Big('Score (R2)', sc.toFixed(2), 'The grade. 1 is perfect, and 0 is no better than giving every compound the average.') + m2Big('Typical miss (RMSE)', rm.toFixed(2), 'How far off a prediction usually is, in ' + esc(unitWord()) + ' units. Smaller is better.', C.char) + m2Big('Ranking (Spearman)', sp.toFixed(2), 'Did the model put the compounds in the right order? 1 is a perfect order.', '#6F8196') +
        (k < 0.9 ? '<p style="font-size:12.5px;color:' + C.dred + '">Squeezing the predictions toward the average makes the typical miss and the score worse, but the order of the compounds does not change, so the ranking number barely moves.</p>' : '<p style="font-size:12.5px">Try the second slider. A model can rank compounds well and still be far off on the actual values.</p>');
    }
    rs.addEventListener('input', draw); rk.addEventListener('input', draw);
    async function play(tok){
      rk.value = 1; rs.value = 1.1; draw(); api.cap('Very noisy predictions. The dots form a cloud, and the score is close to zero.'); await api.sleep(3000);
      await api.anim(2600, function(t){ rs.value = 1.1 - 1.05 * t; draw(); }); api.cap('As the predictions improve, the dots tighten around the dashed line and the score rises toward 1.'); await api.sleep(3000);
      await api.anim(2200, function(t){ rk.value = 1 - 0.7 * t; draw(); }); api.cap('Squeezing every prediction toward the average hurts the score, but the order of the compounds is unchanged.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });

/* ---------- 06 Compare: why no score is perfect ---------- */
LESSONS.compare.add({ level: 'beginner', short: 'Why no score is perfect', title: 'Even the lab cannot repeat itself exactly',
  change: 'Drag the slider to change how noisy the lab is. Press Measure again to repeat every measurement.',
  watch: 'Measure the same compound twice in the lab and you do not get exactly the same number. The small difference is called noise. Here each dot is one compound. Across is the first measurement and up is a second measurement of the same compound. If the lab were perfect, every dot would sit on the dashed line. The lab values come from the real data, but the repeat measurements are invented for this exercise, so that you can turn the noise up and down. A model cannot predict random noise. So even a model that knew the true value of every compound would score below 1 against a noisy measurement. That limit is called the ceiling, and a model that gets close to it has done as well as anyone could.',
  build: function(api){
    var seed = 1, S = null;
    var cp = api.el('pn', 'left:28px;top:20px;width:520px;height:500px', '<h4>The same compounds measured twice (second measurement invented)</h4><canvas width="490" height="440"></canvas>');
    var np = api.el('pn', 'left:568px;top:20px;width:364px;height:500px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('How noisy the lab is <input type="range" min="0.05" max="0.8" step="0.01" value="0.3" style="width:200px"> <b class="nv"></b> <button type="button" class="btn sm ghost" data-again>Measure again</button>');
    var rg = qs('input', ctl), nv = qs('.nv', ctl);
    function prep(){ var idx = m2Pick(D.TRAIN, 130, 4); S = { idx: idx, T: idx.map(function(i){ return D.Y[i]; }) }; var r = rng(100 + seed); S.E1 = idx.map(function(){ return m2Gauss(r); }); S.E2 = idx.map(function(){ return m2Gauss(r); }); }
    function draw(){
      if (!S) prep(); var sd = +rg.value, M1 = S.T.map(function(t, j){ return t + sd * S.E1[j]; }), M2 = S.T.map(function(t, j){ return t + sd * S.E2[j]; });
      nv.textContent = sd < 0.2 ? 'quiet' : (sd < 0.5 ? 'typical' : 'very noisy');
      m2Scatter(qs('canvas', cp), M1, M2, [D.ylo, D.yhi], [D.ylo, D.yhi], 'first measurement', 'second measurement', { diag: true });
      var rep = m2Score(M1, M2), ceil = m2Score(M1, S.T), diff = Math.sqrt(mean(M1.map(function(v, j){ return (v - M2[j]) * (v - M2[j]); })));
      qs('.nb', np).innerHTML = m2Big('Agreement between the two measurements', rep.toFixed(2), 'Scored like a model: the second measurement is the prediction and the first is the lab value.', C.char) + m2Big('Ceiling for any model', ceil.toFixed(2), 'A model that knew the true value of every compound would score only this against one noisy measurement.') + m2Big('Typical gap between repeats', diff.toFixed(2), 'In ' + esc(unitWord()) + ' units.', '#6F8196') +
        '<p style="font-size:12.5px;margin-top:6px">So a score of ' + (ceil > 0.2 ? (ceil - 0.1).toFixed(2) : ceil.toFixed(2)) + ' in a lab this noisy is a very good result, even though it is far below 1. Real assays also vary from one day to the next. How much depends on the assay, and it is worth asking the lab.</p>';
    }
    rg.addEventListener('input', draw); qs('[data-again]', ctl).addEventListener('click', function(){ seed++; prep(); draw(); });
    async function play(tok){
      rg.value = 0.05; draw(); api.cap('A very quiet lab: the second measurement almost always matches the first.'); await api.sleep(3000);
      await api.anim(3200, function(t){ rg.value = 0.05 + 0.6 * t; draw(); }); api.cap('As the noise grows, repeat measurements drift apart, and the ceiling for any model falls with them.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });

/* ---------- 07 Role of AI: what a model has not seen ---------- */
LESSONS.ai.add({ level: 'beginner', short: 'What a model has not seen', title: 'A model is only reliable near what it studied',
  change: 'Drag the slider to change how many compounds the model studies, starting from the least oily.',
  watch: 'A model learns from examples, so it knows only the kinds of compounds it studied. To see this, we let a model study only the less oily compounds, up to a limit set by the slider, and then ask it about test compounds. Each test compound is one vertical line. The dark dot is the lab value and the end of the line is the prediction. The shaded band is the range of oiliness that the model studied. Grey lines are inside the band and red lines are outside it. Compare the typical miss inside and outside the band.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:20px;width:610px;height:500px', '<h4>Test compounds: lab value (dark dot) and prediction (end of the line)</h4><canvas width="580" height="440"></canvas>');
    var np = api.el('pn', 'left:658px;top:20px;width:274px;height:500px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Share of the compounds the model studies, least oily first <input type="range" min="30" max="95" step="5" value="40" style="width:200px"> <b class="pv"></b>');
    var rg = qs('input', ctl), pv = qs('.pv', ctl), FE = ['logP', 'TPSA', 'MW'];
    function draw(){
      var share = +rg.value / 100, tr = D.TRAIN, lp = D.COL.logP, sorted = tr.map(function(i){ return lp[i]; }).sort(function(a, b){ return a - b; }), cut = sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))], lo0 = sorted[0];
      pv.textContent = Math.round(share * 100) + '%'; var key = EP + 'unseen' + rg.value, T = D.cache[key];
      if (!T) { var sub = tr.filter(function(i){ return lp[i] <= cut; }), tree = fitTree(FE.filter(function(f){ return D.COL[f]; }), sub, 5, 15, 1); T = D.cache[key] = { pred: treeAll(tree, D.N), n: sub.length }; }
      var idx = m2Pick(D.TEST, 260, 6), xs = idx.map(function(i){ return lp[i]; }), xlo = Math.min(-1, Math.floor(Math.min.apply(null, xs))), xhi = Math.min(9, Math.ceil(Math.max.apply(null, xs)));
      var inS = 0, inN = 0, outS = 0, outN = 0, outB = 0; idx.forEach(function(i){ var d = D.Y[i] - T.pred[i], e = d * d; if (lp[i] <= cut) { inS += e; inN++; } else { outS += e; outN++; outB += d; } });
      m2Scatter(qs('canvas', cp), [], [], [xlo, xhi], [D.ylo, D.yhi], 'logP (how oily the compound is)', 'lab value', { under: function(x, sx, sy, box){
        x.fillStyle = 'rgba(161,177,196,.22)'; x.fillRect(sx(Math.max(xlo, lo0)), box.t, sx(Math.min(xhi, cut)) - sx(Math.max(xlo, lo0)), box.b - box.t); x.fillStyle = C.char; x.font = '700 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('studied by the model', (sx(Math.max(xlo, lo0)) + sx(Math.min(xhi, cut))) / 2, box.t + 14); if (cut < xhi) { x.fillStyle = C.red; x.fillText('not studied', (sx(cut) + sx(xhi)) / 2, box.t + 14); }
        idx.forEach(function(i){ var inside = lp[i] <= cut; x.strokeStyle = inside ? 'rgba(111,129,150,.55)' : 'rgba(225,36,42,.6)'; x.lineWidth = 1.4; x.beginPath(); x.moveTo(sx(lp[i]), sy(clamp(D.Y[i], D.ylo, D.yhi))); x.lineTo(sx(lp[i]), sy(clamp(T.pred[i], D.ylo, D.yhi))); x.stroke(); if (!inside) { x.fillStyle = C.red; x.fillRect(sx(lp[i]) - 2, sy(clamp(T.pred[i], D.ylo, D.yhi)) - 2, 4, 4); } });
        x.fillStyle = 'rgba(52,55,63,.8)'; idx.forEach(function(i){ x.beginPath(); x.arc(sx(lp[i]), sy(clamp(D.Y[i], D.ylo, D.yhi)), 2.6, 0, 6.3); x.fill(); }); } });
      var ri = inN ? Math.sqrt(inS / inN) : NaN, ro = outN ? Math.sqrt(outS / outN) : NaN;
      qs('.nb', np).innerHTML = '<div class="ol">The model studied</div><div style="margin-bottom:10px"><b>' + T.n.toLocaleString() + '</b> compounds, all with logP up to <b>' + cut.toFixed(1) + '</b>.</div>' + m2Big('Typical miss inside the band', isNaN(ri) ? 'n/a' : ri.toFixed(2), inN + ' test compounds. Grey lines.', '#6F8196') + m2Big('Typical miss outside the band', isNaN(ro) ? 'n/a' : ro.toFixed(2), outN + ' test compounds. Red lines.') +
        (outN && Math.abs(outB / outN) > 0.1 ? m2Big('Average lean outside the band', (outB / outN > 0 ? 'too low by ' : 'too high by ') + Math.abs(outB / outN).toFixed(2), 'The errors outside the band lean one way, instead of cancelling out.', C.char) : '') + '<p style="font-size:12.5px">' + (isNaN(ro) ? 'Almost every test compound is inside the band.' : 'Outside the band the model has nothing to compare with, so it repeats what it learned from the oiliest compounds it did see. Slide to the left to make the region the model did not study larger and compare the two typical misses. The red squares mark where the model put its prediction.') + '</p>';
    }
    rg.addEventListener('input', draw);
    async function play(tok){
      rg.value = 90; draw(); api.cap('The model studied almost the whole range of oiliness, so very few compounds are outside the band.'); await api.sleep(3000);
      await api.anim(3600, function(t){ rg.value = Math.round((90 - 60 * t) / 5) * 5; draw(); }); api.cap('As the model studies fewer of the oily compounds, more test compounds fall outside the band, and the model misses them by more.'); await api.sleep(3400); api.cap(null);
    }
    return { render: draw, play: play };
  } });

/* ---------- 07 Role of AI: checking an answer ---------- */
LESSONS.ai.add({ level: 'beginner', short: 'Checking an answer', title: 'Should you trust this answer?',
  change: 'Read each claim and choose Trust or Do not trust. An explanation appears after your choice.',
  watch: 'A model gives an answer every time you ask, even when it should not. Before you act on an answer, ask three questions. Can the thing be checked at all? Is the compound like the ones the model studied? Does the answer have a measured score behind it? Below are three claims. The compounds and names are invented for this exercise. Choose which ones you would trust.',
  build: function(api){
    var ch = [null, null, null], tree = null;
    var cp = api.el('pn', 'left:28px;top:20px;width:904px;height:500px;overflow:auto', '<div class="cl"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm ghost" data-reset>Try again</button>');
    function stats(){
      var key = EP + 'chk'; if (D.cache[key]) return D.cache[key]; var fs = ['logP', 'TPSA', 'MW'].filter(function(f){ return D.COL[f]; }), t = fitTree(fs, D.TRAIN, 4, 20, 1), mw = D.TRAIN.map(function(i){ return D.COL.MW[i]; }), mwMax = Math.max.apply(null, mw), med = {}; fs.forEach(function(f){ var a = D.TRAIN.map(function(i){ return D.COL[f][i]; }).sort(function(p, q){ return p - q; }); med[f] = a[Math.floor(a.length / 2)]; });
      var tp = treeAll(t, D.N), big = Object.assign({}, med); big.MW = Math.round(mwMax * 1.3);
      return (D.cache[key] = { tree: t, mwMax: mwMax, med: med, big: big, pMed: predictVals(t, D.B, med), pBig: predictVals(t, D.B, big), miss: rmse(D.TEST, tp) });
    }
    function draw(){
      var s = stats(), claims = [
        { text: 'A chatbot says: "The compound <b>Zorvanil-7</b> has a clearance of 0.40 and is safe to make."', note: 'The name and the claim are invented for this exercise.', trust: false,
          why: 'Do not trust it. A name is not a structure. Without a drawing of the molecule nobody can compute its features or check the claim, and the chatbot cannot measure anything. Writing a confident number is easy, and it is not evidence.' },
        { text: 'A model studied compounds up to a molecular weight of <b>' + Math.round(s.mwMax) + '</b>. It is asked about a compound with molecular weight <b>' + s.big.MW + '</b> and predicts <b>' + s.pBig.toFixed(2) + '</b>.', note: 'The compound is invented for this exercise.', trust: false,
          why: 'Do not trust it. This compound is heavier than any compound the model studied, so the model is guessing outside its experience. A tree gives the same answer it gives for the heaviest compounds it did see, whatever the real value is.' },
        { text: 'The same model is asked about a compound with typical values: logP <b>' + s.med.logP.toFixed(1) + '</b>, TPSA <b>' + Math.round(s.med.TPSA) + '</b>, molecular weight <b>' + Math.round(s.med.MW) + '</b>. It predicts <b>' + s.pMed.toFixed(2) + '</b>.', note: 'The compound is invented for this exercise.', trust: true,
          why: 'A fair guess to use, with care. The compound sits in the middle of what the model studied, and the model has a test score behind it, with a typical miss of about ' + s.miss.toFixed(2) + ' ' + esc(unitWord()) + ' units on test compounds. It is still a prediction, so check with the lab before a big decision.' }];
      qs('.cl', cp).innerHTML = claims.map(function(c, i){ var a = ch[i]; return '<div style="border:1px solid ' + (a === null ? C.silver : (a === c.trust ? '#2E7D4F' : C.red)) + ';border-radius:8px;padding:12px 14px;margin:0 0 12px;background:' + (a === null ? '#fff' : (a === c.trust ? '#EAF5EE' : '#FCECEC')) + '"><div class="ol">Claim ' + (i + 1) + '</div><div style="font-size:14px;line-height:1.5;color:' + C.char + '">' + c.text + '</div><div style="font-size:11.5px;color:' + C.mid + ';margin:2px 0 8px">' + c.note + '</div><div style="display:flex;gap:8px"><button type="button" class="btn sm' + (a === true ? '' : ' ghost') + '" data-c="' + i + '" data-v="1"' + (a !== null ? ' disabled' : '') + '>Trust</button><button type="button" class="btn sm' + (a === false ? '' : ' ghost') + '" data-c="' + i + '" data-v="0"' + (a !== null ? ' disabled' : '') + '>Do not trust</button></div>' + (a === null ? '' : '<div style="font-size:13px;line-height:1.5;margin-top:8px;color:' + C.char + '"><b>' + (a === c.trust ? 'Good call. ' : 'Not quite. ') + '</b>' + c.why + '</div>') + '</div>'; }).join('') + (ch.every(function(a){ return a !== null; }) ? '<div style="font-size:14px;font-weight:700;color:' + C.char + '">You got ' + claims.filter(function(c, i){ return ch[i] === c.trust; }).length + ' of 3. The three questions: can it be checked, is it like what the model studied, and is there a score behind it?</div>' : '');
      qsa('[data-c]', cp).forEach(function(b){ b.addEventListener('click', function(){ ch[+b.dataset.c] = b.dataset.v === '1'; draw(); }); });
    }
    qs('[data-reset]', ctl).addEventListener('click', function(){ ch = [null, null, null]; draw(); });
    async function play(tok){ ch = [null, null, null]; draw(); api.cap('Three claims. Decide which you would trust.'); await api.sleep(2600); for (var i = 0; i < 3; i++) { api.checkTok(tok); ch[i] = i === 2; draw(); api.cap(i === 0 ? 'A name with no structure cannot be checked.' : (i === 1 ? 'This compound is outside what the model studied.' : 'This one is typical of what the model studied, and it has a score behind it.')); await api.sleep(3200); } api.cap(null); }
    return { render: draw, play: play };
  } });

/* ---------- 01 Data and ML basics: more practice ---------- */
LESSONS.start.add({ level: 'beginner', short: 'More practice: learning from examples', title: 'Learning from examples, with houses',
  change: 'Drag the two sliders to fit a line to the dots, or press Find the best line.',
  watch: 'Before we go back to compounds, try the same idea with something familiar. Each dot is a house: across is its size and up is its price. The houses are invented for this exercise. A line through the dots is a simple model: give it a size and it returns a price. The line has two settings, how steep it is and where it starts. Each short vertical bar is a miss, the gap between the line and the real price. The typical miss is the average size of those gaps. Learning means finding the settings that make the typical miss as small as possible. A computer can do the search for you.',
  build: function(api){
    var H = null, cp, np, ctl, rs, rb;
    cp = api.el('pn', 'left:28px;top:20px;width:560px;height:500px', '<h4>Invented houses: size against price</h4><canvas width="530" height="440"></canvas>');
    np = api.el('pn', 'left:608px;top:20px;width:324px;height:500px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    ctl = api.addControl('How steep the line is <input type="range" data-d="0" min="0" max="4" step="0.05" value="0.6" style="width:130px"> Where it starts <input type="range" data-d="1" min="-100" max="300" step="5" value="260" style="width:130px"> <button type="button" class="btn sm" data-best>Find the best line</button> <button type="button" class="btn sm ghost" data-new>New houses</button>');
    rs = qsa('[data-d]', ctl); var seed = 2;
    function make(){ var r = rng(40 + seed), n = 28, xs = [], ys = []; for (var i = 0; i < n; i++) { var s = 45 + r() * 145; xs.push(s); ys.push(Math.max(40, 1.9 * s + 55 + 38 * m2Gauss(r))); } var mx = mean(xs), my = mean(ys), sxy = 0, sxx = 0; xs.forEach(function(v, i){ sxy += (v - mx) * (ys[i] - my); sxx += (v - mx) * (v - mx); }); var b = sxy / sxx; H = { xs: xs, ys: ys, b: b, a: my - b * mx, sd: m2Sd(ys) }; }
    function draw(){
      if (!H) make(); var b = +rs[0].value, a = +rs[1].value, miss = Math.sqrt(mean(H.xs.map(function(v, i){ var e = a + b * v - H.ys[i]; return e * e; }))), best = Math.sqrt(mean(H.xs.map(function(v, i){ var e = H.a + H.b * v - H.ys[i]; return e * e; })));
      m2Scatter(qs('canvas', cp), H.xs, H.ys, [40, 200], [0, 520], 'size of the house (square metres)', 'price (thousands)', { fill: 'rgba(52,55,63,.7)', r: 4, under: function(x, sx, sy){
        H.xs.forEach(function(v, i){ var ly = clamp(a + b * v, 0, 520); x.strokeStyle = 'rgba(225,36,42,.45)'; x.lineWidth = 2; x.beginPath(); x.moveTo(sx(v), sy(H.ys[i])); x.lineTo(sx(v), sy(ly)); x.stroke(); });
        x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); x.moveTo(sx(40), sy(clamp(a + b * 40, -50, 580))); x.lineTo(sx(200), sy(clamp(a + b * 200, -50, 580))); x.stroke(); } });
      qs('.nb', np).innerHTML = m2Big('Typical miss', miss.toFixed(0), 'in thousands. The average size of the red bars. Smaller is better.') + '<div style="font-size:13px">If you ignored the size and gave every house the average price, the typical miss would be <b>' + H.sd.toFixed(0) + '</b>.</div><div style="font-size:13px;margin-top:6px">The best line can do is a typical miss of <b>' + best.toFixed(0) + '</b>.</div><p style="margin-top:10px">' + (miss <= best + 2 ? 'You found it. This is what learning means: settings that make the typical miss as small as possible.' : 'Keep adjusting, or press Find the best line to watch the computer search.') + '</p><p style="font-size:12px;color:' + C.mid + '">Learning a model for compounds works the same way. There are more numbers to describe each compound, and many more settings to adjust.</p>';
    }
    rs.forEach(function(r){ r.addEventListener('input', draw); });
    qs('[data-new]', ctl).addEventListener('click', function(){ seed++; make(); draw(); });
    qs('[data-best]', ctl).addEventListener('click', function(){ if (!H) make(); var f0 = +rs[0].value, f1 = +rs[1].value; animateFree(1500, function(t){ rs[0].value = f0 + (clamp(H.b, 0, 4) - f0) * t; rs[1].value = f1 + (clamp(H.a, -100, 300) - f1) * t; draw(); }); });
    async function play(tok){ if (!H) make(); rs[0].value = 0.6; rs[1].value = 260; draw(); api.cap('A poor line: it misses most houses by a lot.'); await api.sleep(3000); api.cap('Now the computer searches for the settings with the smallest typical miss.'); await api.anim(2800, function(t){ rs[0].value = 0.6 + (clamp(H.b, 0, 4) - 0.6) * t; rs[1].value = 260 + (clamp(H.a, -100, 300) - 260) * t; draw(); }); api.cap('The line sits in the middle of the dots. That is a model that has learned from examples.'); await api.sleep(3200); api.cap(null); }
    return { render: draw, play: play };
  } });

LESSONS.start.add({ level: 'beginner', short: 'More practice: describing with numbers', title: 'A model sees only a list of numbers',
  change: 'Choose a house, a person or a compound. Change the numbers and watch what the model is given.',
  watch: 'A model cannot look at a house, a person or a molecule. It is given a list of numbers that describe it. Each number is called a feature. A house might be described by its size, its number of bedrooms and its age. A person might be described by height, age and resting heart rate. A compound is described by numbers such as logP, TPSA and molecular weight. Two things with similar lists of numbers are treated alike. The house prediction uses an invented formula, only to show the idea. The compound prediction uses a real model trained on the data in this course.',
  build: function(api){
    var mode = 2, V = null, ST = null;
    function fmtV(v, step){ return step < 1 ? (+v).toFixed(1) : Math.round(v).toLocaleString(); }
    var lp = api.el('pn', 'left:28px;top:20px;width:540px;height:500px;overflow:auto', '<div class="sl"></div>');
    var rp = api.el('pn', 'left:588px;top:20px;width:344px;height:500px;overflow:auto', '<div class="rt" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Describe a ' + inSeg(['house', 'person', 'compound'], 2, 'data-m'));
    var SPEC = [
      [['Size (square metres)', 40, 220, 5, 110], ['Bedrooms', 1, 6, 1, 3], ['Age (years)', 0, 90, 1, 25], ['Distance to the centre (km)', 1, 30, 1, 8]],
      [['Height (cm)', 140, 205, 1, 172], ['Age (years)', 18, 90, 1, 40], ['Resting heart rate (beats a minute)', 45, 100, 1, 64], ['Steps a day', 1000, 20000, 500, 8000]]];
    function compoundSpec(){
      if (ST) return ST; var arr = STRUCTS[EP] || [], e = arr[3] || arr[0], fs = ['logP', 'TPSA', 'MW'].filter(function(f){ return D.COL[f]; }), rg = {}; fs.forEach(function(f){ var a = D.TRAIN.map(function(i){ return D.COL[f][i]; }).sort(function(p, q){ return p - q; }); rg[f] = [a[Math.floor(a.length * 0.02)], a[Math.floor(a.length * 0.98)]]; });
      var key = EP + 'feat3'; D.cache[key] = D.cache[key] || fitTree(fs, D.TRAIN, 4, 20, 1);
      return (ST = { e: e, fs: fs, rg: rg, tree: D.cache[key], ep: EP });
    }
    function build(){
      var sl = qs('.sl', lp), rt = qs('.rt', rp), rows, pred;
      if (mode < 2) { rows = SPEC[mode]; V = rows.map(function(r){ return r[4]; }); }
      else { var s = compoundSpec(); rows = s.fs.map(function(f){ var r = s.rg[f], step = f === 'logP' ? 0.1 : 1; return [f === 'logP' ? 'logP (how oily)' : (f === 'TPSA' ? 'TPSA (how much of its surface attracts water)' : 'MW (molecular weight)'), Math.floor(r[0] * 10) / 10, Math.ceil(r[1]), step, Math.round(D.COL[f][s.e.i] / step) * step]; }); rows.forEach(function(r){ r[4] = +r[4].toFixed(1); }); V = rows.map(function(r){ return r[4]; }); }
      var pic = mode === 2 ? '<div class="struct" style="height:130px;width:200px;margin:4px 0 10px">' + compoundSpec().e.svg + '</div>' : '<div style="font-size:54px;line-height:1;margin:4px 0 10px">' + (mode === 0 ? '&#127968;' : '&#128100;') + '</div>';
      sl.innerHTML = '<div class="ol">' + ['A house (invented)', 'A person (invented)', 'A compound from the data'][mode] + '</div>' + pic + '<p style="font-size:13px;color:' + C.char + '">Move a slider to change one number. Each number is a feature.</p>' + rows.map(function(r, i){ return '<div style="margin:8px 0"><label style="display:flex;justify-content:space-between;font-size:13px;font-weight:600">' + esc(r[0]) + '<output>' + fmtV(V[i], r[3]) + '</output></label><input type="range" data-f="' + i + '" min="' + r[1] + '" max="' + r[2] + '" step="' + r[3] + '" value="' + r[4] + '" style="width:100%"></div>'; }).join('');
      qsa('[data-f]', sl).forEach(function(r){ r.addEventListener('input', function(){ V[+r.dataset.f] = +r.value; qs('output', r.parentNode).textContent = fmtV(+r.value, +r.step); show(rows); }); });
      show(rows);
    }
    function show(rows){
      var rt = qs('.rt', rp), list = '<div class="ol">What the model is given</div><div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px">' + rows.map(function(r, i){ return '<span style="border:1px solid ' + C.silver + ';border-radius:6px;padding:4px 9px;background:#fff;font-size:12.5px"><span style="color:' + C.mid + '">' + esc(r[0].split(' (')[0]) + '</span> <b>' + fmtV(V[i], rows[i][3]) + '</b></span>'; }).join('') + '</div>', out;
      if (mode === 0) { var p = 2.0 * V[0] + 15 * V[1] - 1.2 * V[2] - 3 * V[3] + 40; out = '<div class="ol">What it guesses</div><div class="big" style="color:' + C.red + '">' + Math.round(p) + '</div><div style="font-size:12px">price in thousands, from an invented formula: bigger and newer houses closer to the centre cost more.</div>'; }
      else if (mode === 1) out = '<div class="ol">What it could guess</div><p>With enough examples of people, a model could learn to guess something about them from these four numbers, for example how fit they are. Nothing in the model sees a face or a name. It sees only the four numbers.</p>';
      else { var s = compoundSpec(), vals = {}; s.fs.forEach(function(f, i){ vals[f] = V[i]; }); var pr = predictVals(s.tree, D.B, vals); out = '<div class="ol">What it guesses</div><div class="big" style="color:' + C.red + '">' + pr.toFixed(2) + '</div><div style="font-size:12px">' + esc(unitWord()) + ', from a real model trained on ' + D.TRAIN.length.toLocaleString() + ' compounds of the data. The lab measured this compound at <b>' + D.Y[s.e.i].toFixed(2) + '</b>.</div>'; }
      rt.innerHTML = list + out + '<p style="margin-top:14px;font-size:12.5px;color:' + C.mid + '">Notice what the model does not see: the picture, the name, or anything else that is not on the list.</p>';
    }
    qsa('[data-m] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ mode = +b.dataset.v; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); build(); }); });
    async function render(){ await loadStructs(); if (ST && ST.ep !== EP) ST = null; build(); }
    async function play(tok){ await render(); for (var k = 0; k < 3; k++) { api.checkTok(tok); mode = k; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === mode); }); build(); api.cap(['A house described by four numbers.', 'A person described by four numbers.', 'A compound described by three numbers. The model gives a real prediction.'][k]); await api.sleep(3200); } api.cap(null); }
    return { render: render, play: play };
  } });

/* ---------- guide entries ---------- */
GUIDE.steps['compare/What the scores mean'] = { notice: 'Squeezing the predictions toward the average hurts the score but leaves the ranking alone.', q: 'Can a model rank compounds well and still have a poor score?', a: 'Yes. The order of the compounds can be perfect while every prediction is shifted or squeezed, so the typical miss is large.' };
GUIDE.steps['compare/Why no score is perfect'] = { notice: 'The ceiling falls as the lab gets noisier.', q: 'Why can a perfect model not score 1?', a: 'The lab value it is scored against contains random noise, and no model can predict random noise.' };
GUIDE.steps['ai/What a model has not seen'] = { notice: 'The red lines, outside the band, are usually longer.', q: 'Why does the model do worse outside the band?', a: 'It has no examples there, so it repeats what it learned from the nearest compounds it did study.' };
GUIDE.steps['ai/Checking an answer'] = { notice: 'Ask three questions before trusting an answer.', q: 'What are the three questions?', a: 'Can it be checked at all, is the compound like the ones the model studied, and is there a measured score behind it?' };
GUIDE.steps['start/More practice: learning from examples'] = { notice: 'The typical miss falls as the line fits the dots better.', q: 'What does learning mean here?', a: 'Finding the settings of the line that make the typical miss as small as possible.' };
GUIDE.steps['start/More practice: describing with numbers'] = { notice: 'The model sees only the list of numbers.', q: 'What is a feature?', a: 'One number that describes a thing, such as the size of a house or the logP of a compound.' };
