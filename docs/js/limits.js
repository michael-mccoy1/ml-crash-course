"use strict";
/* Lesson 7: Role of AI. What these models do, where they stop, and what to check before relying on one. */
var L6 = new Lesson('ai', 'Role of AI');
L6.optsHTML = function(){ return optGroup('Endpoint', seg([['HLM', 'HLM clearance'], ['logD', 'logD']], EP, 'data-ep'), 'HLM is Biogen data. logD is AstraZeneca data.'); };
L6.onData = function(){ };

/* ---------- Step 1: what these models do ---------- */
var AI_MODELS = [
  { id: 'trees', name: 'Tree models', lesson: 'trees', num: '02', input: 'Feature numbers such as logP and TPSA.', output: 'A predicted lab value.', learns: 'Compounds that have a lab value.', limit: 'Outside the range of the training data the prediction goes flat.' },
  { id: 'nn', name: 'Neural network', lesson: 'nn', num: '03', input: 'Feature numbers such as logP and TPSA.', output: 'A predicted lab value.', learns: 'Compounds that have a lab value.', limit: 'The score changes with the random seed, and it needs more data than a tree to catch up.' },
  { id: 'gpt', name: 'GPT', lesson: 'gpt', num: '04', input: 'The tokens of a SMILES string so far.', output: 'The next token.', learns: 'Molecule text only. No lab values.', limit: 'It can write invalid molecules, and it predicts no property.' },
  { id: 'pre', name: 'Pre-trained model', lesson: 'pre', num: '05', input: 'A molecule drawn as atoms and bonds.', output: 'A list of numbers that describes the molecule.', learns: 'Millions of molecules with no lab values.', limit: 'It does not always beat ordinary features. Compare before you adopt it.' },
  { id: 'chemprop', name: 'Chemprop', lesson: 'nn', num: '03 and 05', input: 'A molecule drawn as atoms and bonds.', output: 'A predicted lab value.', learns: 'Compounds that have a lab value, and optionally a related dataset first.', limit: 'It is slower to train than a tree model, and fine-tuning only helped with enough compounds.' }
];
L6.add({ short: 'What these models do', title: 'Every model here finds a pattern in examples',
  change: 'Click each model to see what goes in, what comes out, what it learned from and where it stops.',
  watch: 'All the models in this course follow one recipe. Collect examples that pair an input with a known answer. Let the model adjust its numbers until its predictions match the answers. Then give it a new input. A trained model is a pattern finder. It does not know chemistry or biology, only what the patterns in its examples say.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div class="aim"></div>'), cur = 0;
    function draw(){
      var m = AI_MODELS[cur], rows = [['Input', m.input], ['Output', m.output], ['Learned from', m.learns], ['A limit you saw', m.limit]];
      qs('.aim', pn).innerHTML = seg(AI_MODELS.map(function(x, i){ return [String(i), x.name]; }), String(cur), 'data-ai') +
        '<div style="display:grid;grid-template-columns:150px 1fr;gap:0;margin-top:18px;font-size:16px;line-height:1.45">' + rows.map(function(r, i){ return '<div class="lbl" style="padding:14px 0;border-top:1px solid ' + C.silver + ';text-transform:none;letter-spacing:0;font-size:13px;font-weight:700;color:' + (i === 3 ? C.red : C.char) + '">' + r[0] + '</div><div style="padding:14px 0;border-top:1px solid ' + C.silver + ';color:' + C.char + '">' + esc(r[1]) + '</div>'; }).join('') + '</div>' +
        '<div style="margin-top:14px;display:flex;justify-content:space-between;align-items:center;gap:12px"><span style="font-size:13px;color:' + C.char + '">Covered in lesson ' + m.num + '.</span><button type="button" class="btn sm ghost" data-go>Open lesson</button></div>';
      qsa('[data-ai] button', pn).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.v; draw(); }); });
      qs('[data-go]', pn).addEventListener('click', function(){ showLesson(m.lesson); });
    }
    return { render: draw };
  } });

/* ---------- Step 2: outside the training range ---------- */
L6.add({ short: 'Outside the training range', title: 'A model is only reliable inside the range it trained on',
  change: 'Choose a feature and drag the training range below the plot.',
  watch: 'A model has only seen its training compounds. Here a tree is trained on compounds whose feature value is below a cut-off, shown as a dashed line. It is then scored on test compounds on each side of the line. A test R2 of 0 means the model is no better than guessing the average, and below 0 means it is worse. The range is the span of feature values the model saw.',
  build: function(api){
    var G_ = 200, st = null;
    var plot = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4 class="pt"></h4><canvas width="600" height="334" style="margin-top:2px"></canvas><div class="note" style="font-size:12px;margin-top:4px">Blue dots: trained on. Dark rings: test compounds inside the range. Red rings: test compounds outside. Shaded: outside the range.</div>');
    var side = api.el('pn', 'left:668px;top:28px;width:264px;height:408px', '<h4>Test R2 on each side of the line</h4><div class="sd"></div>');
    var ctl = api.addControl('Feature <select></select> Train on the lowest <input type="range" min="20" max="90" step="5" value="60" style="width:170px"><b class="dv">60</b>% of values');
    var sel = qs('select', ctl), sl = qs('input', ctl), dv = qs('.dv', ctl);
    function feats(){ var f = activeFeats(); return f.length ? f : D.BASE.slice(); }
    function fillSel(){ var f = feats(), c = sel.value; sel.innerHTML = f.map(function(x){ return '<option>' + x + '</option>'; }).join(''); if (f.indexOf(c) >= 0) sel.value = c; }
    function fit(){
      var f = sel.value || feats()[0], pct = +sl.value, vals = D.TRAIN.map(function(i){ return D.COL[f][i]; }).sort(function(a, b){ return a - b; }), cut = vals[Math.max(0, Math.floor(vals.length * pct / 100) - 1)];
      var B = fineBins(f, 200), idx = D.TRAIN.filter(function(i){ return D.COL[f][i] <= cut; }), tree = buildTree(B, [f], D.Y, Int32Array.from(idx), 4, 5, 1, rng(1));
      var s = sortedCol(f), x0 = s[Math.floor(s.length * 0.01)], x1 = s[Math.floor(s.length * 0.99)], cu = new Float32Array(G_), v = {};
      for (var k = 0; k < G_; k++) { v[f] = lerp(x0, x1, k / (G_ - 1)); cu[k] = predictVals(tree, B, v); }
      var pred = new Float32Array(D.N), inn = [], out = []; D.TEST.forEach(function(i){ pred[i] = predictTree(tree, B, i); (D.COL[f][i] <= cut ? inn : out).push(i); });
      var rnd = rng(9), shT = [], shI = [], shO = []; idx.forEach(function(i){ if (rnd() < 0.2) shT.push(i); }); inn.forEach(function(i){ if (rnd() < 0.5) shI.push(i); }); out.forEach(function(i){ if (rnd() < 0.5) shO.push(i); });
      st = { f: f, pct: pct, cut: cut, cu: cu, x0: x0, x1: x1, rIn: inn.length >= 15 ? r2(inn, pred) : null, rOut: out.length >= 15 ? r2(out, pred) : null, nIn: inn.length, nOut: out.length, shT: shT, shI: shI, shO: shO };
    }
    function draw(){
      if (!st) return; var g = cv(qs('canvas', plot)), x = g.x, box = { l: 50, r: g.w - 12, t: 12, b: g.h - 42 }, yl = D.ylo, yh = D.yhi;
      var sx = function(v){ return box.l + (clamp(v, st.x0, st.x1) - st.x0) / (st.x1 - st.x0) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, yl, yh) - yl) / (yh - yl) * (box.b - box.t); }, f = st.f;
      axes(g, box, [st.x0, st.x1], [yl, yh], f, unitWord(), 5);
      var cx = sx(st.cut); x.fillStyle = 'rgba(237,242,244,.85)'; x.fillRect(cx, box.t, box.r - cx, box.b - box.t);
      x.fillStyle = 'rgba(161,177,195,.7)'; st.shT.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[f][i]), sy(D.Y[i]), 3, 0, 6.3); x.fill(); });
      x.lineWidth = 1.4; x.strokeStyle = C.char; st.shI.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[f][i]), sy(D.Y[i]), 3.3, 0, 6.3); x.stroke(); }); x.strokeStyle = C.dred; st.shO.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[f][i]), sy(D.Y[i]), 3.3, 0, 6.3); x.stroke(); });
      x.setLineDash([6, 5]); x.strokeStyle = C.char; x.lineWidth = 1.6; x.beginPath(); x.moveTo(cx, box.t); x.lineTo(cx, box.b); x.stroke(); x.setLineDash([]);
      x.strokeStyle = C.red; x.lineWidth = 3.4; x.lineJoin = 'round'; x.beginPath(); for (var k = 0; k < G_; k++) { var px = box.l + k / (G_ - 1) * (box.r - box.l); k ? x.lineTo(px, sy(st.cu[k])) : x.moveTo(px, sy(st.cu[k])); } x.stroke();
            qs('.pt', plot).textContent = 'Measured value against ' + f + ', with a decision tree fit';
      function bar(label, n, v, col){ var w = v === null ? 0 : clamp(v, 0, 0.5) / 0.5 * 100; return '<div style="margin-bottom:14px"><div style="font-size:12.5px;color:' + C.char + '">' + label + ' <span style="color:' + C.char + '">(' + n + ' compounds)</span></div><div class="pbar" style="height:12px;margin:3px 0"><i style="background:' + col + ';width:' + w + '%;transition:none"></i></div><b style="font-size:20px;color:' + col + '">' + (v === null ? 'n/a' : v.toFixed(2)) + '</b></div>'; }
      var msg = st.rIn === null || st.rOut === null ? 'Too few test compounds on one side to score it.' : (st.rOut < -0.02 ? 'Outside the range the model is worse than guessing the average.' : (st.rOut <= 0.02 ? 'Outside the range the model is no better than the average.' : (st.rOut >= st.rIn ? 'Here the score holds up outside the range.' : 'Outside the range the score drops from ' + st.rIn.toFixed(2) + ' to ' + st.rOut.toFixed(2) + '.')));
      qs('.sd', side).innerHTML = bar('Inside the trained range', st.nIn, st.rIn, C.char) + bar('Outside the trained range', st.nOut, st.rOut, C.red) + '<p style="font-size:13px;line-height:1.45;margin:0;color:' + C.char + '">' + msg + '</p><p class="note" style="font-size:12px;margin-top:8px">Bars show test R2 from 0 to 0.5. A negative score shows as an empty bar.</p>';
      dv.textContent = st.pct;
    }
    function refresh(){ fillSel(); fit(); draw(); }
    sel.addEventListener('change', refresh); sl.addEventListener('input', function(){ dv.textContent = sl.value; fit(); draw(); });
    return { render: refresh };
  } });

/* ---------- Step 3: lab noise ---------- */
L6.add({ short: 'Lab noise', title: 'Noise in the lab data caps the best possible score',
  change: 'Drag Assay noise and watch the test R2 ceiling.',
  watch: 'A lab measurement is never exact. Measure the same compound again and the number changes a little. That spread is assay noise. A model is scored against the measured value, so noise costs it score even when the model is right. Here each test compound is measured a second time, with random noise of the size you choose. The R2 between the two sets of numbers is the most any model could be expected to reach.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4>First measurement against a repeat measurement, test compounds</h4><canvas width="600" height="352" style="margin-top:2px"></canvas>');
    var side = api.el('pn', 'left:668px;top:28px;width:264px;height:408px', '<h4>R2 between the two measurements</h4><div class="sd"></div>');
    var ctl = api.addControl('Assay noise (SD) <input type="range" min="0" max="1" step="0.05" value="0.3" style="width:240px"><b class="dv">0.30</b>');
    var sl = qs('input', ctl), dv = qs('.dv', ctl), Z = null, SDY = 1;
    function prep(){ var r = rng(11); Z = {}; D.TEST.forEach(function(i){ Z[i] = gauss(r); }); var m = mean(D.TEST.map(function(i){ return D.Y[i]; })); SDY = Math.sqrt(mean(D.TEST.map(function(i){ return (D.Y[i] - m) * (D.Y[i] - m); }))); sl.max = (Math.ceil(SDY * 1.2 * 20) / 20).toFixed(2); if (+sl.value > +sl.max) sl.value = sl.max; Z.ep = EP; }
    function best(){ var b = null; Object.keys(D.scores || {}).forEach(function(k){ if (b === null || D.scores[k].r2 > b) b = D.scores[k].r2; }); return b; }
    function draw(){
      if (!Z || Z.ep !== EP) prep(); var s = +sl.value, pred = new Float32Array(D.N); D.TEST.forEach(function(i){ pred[i] = D.Y[i] + Z[i] * s; });
      var ceil = r2(D.TEST, pred), g = cv(qs('canvas', plot)), x = g.x, box = { l: 50, r: g.w - 12, t: 12, b: g.h - 42 }, yl = D.ylo, yh = D.yhi;
      axes(g, box, [yl, yh], [yl, yh], 'first measurement', 'repeat measurement', 5);
      var sx = function(v){ return box.l + (clamp(v, yl, yh) - yl) / (yh - yl) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, yl, yh) - yl) / (yh - yl) * (box.b - box.t); };
      x.setLineDash([5, 5]); x.strokeStyle = C.char; x.lineWidth = 1.4; x.beginPath(); x.moveTo(sx(yl), sy(yl)); x.lineTo(sx(yh), sy(yh)); x.stroke(); x.setLineDash([]);
      x.fillStyle = 'rgba(225,36,42,.45)'; D.TEST.forEach(function(i){ x.beginPath(); x.arc(sx(D.Y[i]), sy(pred[i]), 3, 0, 6.3); x.fill(); });
      var b = best(); qs('.sd', side).innerHTML = '<div class="big" style="color:' + C.red + '">' + ceil.toFixed(2) + '</div><div class="note" style="font-size:12px;color:' + C.char + ';margin-bottom:12px">ceiling for test R2 at this noise</div>' +
        (b === null ? '<p style="font-size:13px;line-height:1.45;color:' + C.char + '">Train a model in lessons 02 or 03 to compare its test R2 with this ceiling.</p>' : '<div class="rrow">Your best model so far<b>' + b.toFixed(2) + '</b></div><p style="font-size:13px;line-height:1.45;margin-top:10px;color:' + C.char + '">' + (b >= ceil ? 'Your model already scores above this ceiling, so the real noise in this assay is smaller than ' + s.toFixed(2) + '.' : 'Your model is below this ceiling. If the real noise is this large, there is room to improve.') + '</p>') +
        '<p class="note" style="font-size:12px;margin-top:10px">Noise is the standard deviation (SD) of the random error added to each value, in the units of the target.' + (D.cens ? ' Compounds at the assay floor share one first value, so they form the vertical column.' : '') + '</p>';
      dv.textContent = s.toFixed(2);
    }
    sl.addEventListener('input', draw);
    return { render: function(){ Z = null; draw(); } };
  } });

/* ---------- Step 4: sort what AI can and cannot do ---------- */
var AI_CLAIMS = [
  [true, 'Rank new compounds by predicted clearance when they resemble the training compounds.', 'This is what the tree and network models do. The test R2 says how well.'],
  [false, 'Predict well for chemistry unlike anything in its training data.', 'Outside the range it trained on, a model has nothing to go on. The test R2 fell to zero or below in the outside-the-range step.'],
  [true, 'Pick out patterns across thousands of compounds that nobody would read one by one.', 'A model reads every training compound and keeps the pattern as weights.'],
  [false, 'Score better than the noise in the lab measurements allows.', 'The measured value is the target. Noise in it caps the test R2 for any model.'],
  [true, 'Suggest new molecules for a chemist to review.', 'A GPT trained on SMILES writes candidates. Some are invalid and all of them need checking.'],
  [false, 'Say reliably when one of its own predictions is wrong.', 'A model returns a number with no warning. Runs with different seeds can disagree, and only a measurement settles it.'],
  [true, 'Use what it learned from one dataset to help with a related one.', 'Chemprop trained on logD first and then fine-tuned on 1,000 HLM compounds scored 0.41, against 0.24 from scratch.'],
  [false, 'Explain why a compound is cleared quickly.', 'A model finds patterns that predict. It does not find causes. Feature importance shows what the model leaned on, not what the liver does.'],
  [true, 'Show which of your features it relied on most.', 'Split gain and ablation in the tree lesson both measure this, though the two can disagree.'],
  [false, 'Guarantee that every molecule a GPT writes is valid.', 'It learns text patterns. RDKit rejects some of what it writes, more at a high temperature.']
];
AI_CLAIMS = [0, 1, 3, 2, 5, 4, 6, 8, 7, 9].map(function(k){ return AI_CLAIMS[k]; });
L6.add({ short: 'Can it or cannot it', title: 'Sort what AI can and cannot do',
  change: 'Read each statement and choose AI can or AI cannot. The reason appears after you choose.',
  watch: 'Each statement below describes something people ask of a model like the ones in this course. Decide whether a trained model can do it. After you answer, the reason points back to what the lessons showed.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div class="aic"></div>'), i = 0, score = 0, answered = null;
    function draw(){
      var el = qs('.aic', pn);
      if (i >= AI_CLAIMS.length) { el.innerHTML = '<div class="big" style="color:' + C.red + '">' + score + ' of ' + AI_CLAIMS.length + '</div><p style="font-size:16px;line-height:1.5;color:' + C.char + ';margin:8px 0 18px">' + (score === AI_CLAIMS.length ? 'You sorted every statement the way the lessons did.' : 'That is how many you sorted the way the lessons did. For the others, reread the reason.') + '</p><button type="button" class="btn" data-again>Start again</button>'; qs('[data-again]', el).addEventListener('click', function(){ i = 0; score = 0; answered = null; draw(); }); return; }
      var c = AI_CLAIMS[i];
      el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;color:' + C.char + '"><span>Statement ' + (i + 1) + ' of ' + AI_CLAIMS.length + '</span><span>Matches so far: ' + score + '</span></div>' +
        '<div style="font-size:22px;line-height:1.35;font-weight:700;font-family:var(--display);color:' + C.char + ';margin:22px 0 22px;min-height:90px">' + esc(c[1]) + '</div>' +
        '<div style="display:flex;gap:12px"><button type="button" class="btn" data-a="1"' + (answered !== null ? ' disabled' : '') + '>AI can</button><button type="button" class="btn ghost" data-a="0"' + (answered !== null ? ' disabled' : '') + '>AI cannot</button></div>' +
        (answered === null ? '' : '<div style="margin-top:22px;padding:14px 16px;border-radius:6px;background:' + C.light + ';border-left:4px solid ' + (answered === c[0] ? C.char : C.red) + '"><b style="color:' + (answered === c[0] ? C.char : C.red) + '">' + (answered === c[0] ? 'Matches the lessons. ' : 'Not quite. ') + (c[0] ? 'AI can.' : 'AI cannot.') + '</b><div style="margin-top:4px;font-size:15px;line-height:1.5;color:' + C.char + '">' + esc(c[2]) + '</div></div><div style="margin-top:14px"><button type="button" class="btn sm" data-next>' + (i === AI_CLAIMS.length - 1 ? 'See result' : 'Next statement') + '</button></div>');
      qsa('[data-a]', el).forEach(function(b){ b.addEventListener('click', function(){ answered = b.dataset.a === '1'; if (answered === c[0]) score++; draw(); }); });
      var nx = qs('[data-next]', el); if (nx) nx.addEventListener('click', function(){ i++; answered = null; draw(); });
    }
    return { render: draw };
  } });

/* ---------- Step 5: using it in a project ---------- */
var AI_CHECKS = [
  ['Is the compound like the training compounds?', 'A model is reliable inside the range it trained on. Compare the new compound with what the model saw.', 'start', 'Lesson 01, the test set'],
  ['Was the test fair?', 'Hold test compounds back by scaffold, not at random, and score only on those.', 'start', 'Lesson 01, the test set'],
  ['Is the gain bigger than the noise?', 'Compare a difference in score with the spread between seeds and with the noise in the assay.', 'trees', 'Lesson 02, seeds and ensembles'],
  ['Does a simpler model do as well?', 'Put a tree model or a descriptor model next to the new one, on the same split and with its training time.', 'compare', 'Lesson 06, the leaderboard'],
  ['Would a chemist agree, and can you measure it?', 'A prediction is a reason to run an experiment, not a replacement for one. Measure the compounds the decision depends on.', null, '']
];
L6.add({ short: 'Using it in a project', title: 'Five checks before you rely on a prediction',
  change: 'Read the five checks. Each one has a button that opens the lesson where it was shown.',
  watch: 'A model gives a number. It does not know whether the number is right, so a person has to check. Models propose and rank. People test, decide and answer for the result. These five checks use what the earlier lessons showed.',
  build: function(api){
    var els = AI_CHECKS.map(function(c, i){ var top = 28 + i * 80;
      return api.el('pn fd', 'left:28px;top:' + top + 'px;width:904px;height:72px;display:flex;align-items:center;gap:16px', '<div class="big" style="color:' + C.red + ';font-size:28px;width:34px">' + (i + 1) + '</div><div style="flex:1"><div style="font-size:16px;font-weight:700;color:' + C.char + '">' + esc(c[0]) + '</div><div style="font-size:13.5px;line-height:1.4;color:' + C.char + '">' + esc(c[1]) + '</div></div>' + (c[2] ? '<button type="button" class="btn sm ghost" data-go="' + c[2] + '" style="white-space:nowrap">' + esc(c[3]) + '</button>' : '')); });
    els.forEach(function(e){ var b = qs('[data-go]', e); if (b) b.addEventListener('click', function(){ showLesson(b.dataset.go); }); });
    function render(){ els.forEach(function(e){ e.classList.add('on'); }); }
    return { render: render };
  } });

L6.glossary([
  ['Pattern finder', 'A model fitted to examples. It keeps the patterns that link inputs to answers.'],
  ['Training range', 'The span of feature values among the training compounds.'],
  ['Extrapolation', 'Predicting for inputs outside the training range.'],
  ['Test set', 'Compounds held back from training and used only to score the model.'],
  ['Test R2', 'R2 on the test set. 1 is perfect and 0 is no better than the average.'],
  ['Assay noise', 'Random error in a lab measurement. It limits the test R2 any model can reach.'],
  ['Repeat measurement', 'A second measurement of the same compound.']
]);
setUses(L6, [[], ['endpoint'], ['endpoint'], [], []]);
