"use strict";
/* Lesson 7: Role of AI. What AI is, what it is asked to do, where it stops, and what to check before relying on it. */
var L6 = new Lesson('ai', 'Role of AI');
L6.optsHTML = function(){ return optGroup('Endpoint', seg([['HLM', 'HLM clearance'], ['logD', 'logD']], EP, 'data-ep'), 'HLM is Biogen data. logD is AstraZeneca data.'); };
L6.onData = function(){ };

/* ---------- Step 1: what AI means ---------- */
var AI_LAYERS = [
  { name: 'Artificial intelligence', what: 'Software that does something we would call intelligent if a person did it, such as recognising patterns, planning, deciding or using language.', course: 'All six lessons.', outside: 'Rule-based programs that follow steps a person wrote, route planners, and chat assistants.' },
  { name: 'Machine learning', what: 'AI whose rules are learned from examples instead of written by hand.', course: 'Tree models and neural networks that predict clearance and logD (lessons 02 and 03).', outside: 'Spam filters, product recommendations and credit scoring.' },
  { name: 'Deep learning', what: 'Machine learning with neural networks that have many layers.', course: 'The neural network, Chemprop, the GPT and the pre-trained models (lessons 03 to 05).', outside: 'Image recognition, speech recognition and protein structure prediction.' },
  { name: 'Generative AI', what: 'Deep learning that writes new text, images or molecules instead of returning one score.', course: 'The small GPT that writes SMILES (lesson 04).', outside: 'Chat assistants such as ChatGPT and Claude. These are large language models (LLMs): GPT-style models trained on very large amounts of text.' }
];
L6.add({ short: 'What AI means', title: 'Machine learning predicts, generative AI produces',
  change: 'Click a layer to see what it means, where it appeared in this course and where else you meet it.',
  watch: 'Artificial intelligence (AI) is the widest term. Machine learning is the part of AI that learns from examples, and most of it predicts: give it a compound and it returns a number. Deep learning is machine learning with many-layered neural networks. Generative AI is deep learning that produces new content, such as text, code or molecules. Each layer sits inside the one before it, but the two jobs differ. The rest of this tab keeps prediction and generation apart.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div style="display:grid;grid-template-columns:340px 1fr;gap:28px"><div class="ail"></div><div class="aid"></div></div>'), cur = 0;
    function box(i){ var L = AI_LAYERS[i], on = i === cur, inner = i < AI_LAYERS.length - 1 ? box(i + 1) : ''; return '<div data-l="' + i + '" style="cursor:pointer;border:2px solid ' + (on ? C.red : C.silver) + ';border-radius:10px;padding:8px 10px 10px;background:' + (on ? C.light : '#fff') + ';margin-top:' + (i ? 8 : 0) + 'px"><div style="font-size:14px;font-weight:700;color:' + C.char + '">' + L.name + '</div>' + inner + '</div>'; }
    function draw(){
      var L = AI_LAYERS[cur], rows = [['What it is', L.what], ['In this course', L.course], ['Elsewhere', L.outside]], host = qs('.ail', pn);
      if (!host.firstChild) { host.innerHTML = box(0); qsa('[data-l]', pn).forEach(function(e){ e.addEventListener('click', function(ev){ ev.stopPropagation(); cur = +e.dataset.l; draw(); }); }); }
      qsa('[data-l]', pn).forEach(function(e){ var on = +e.dataset.l === cur; e.style.borderColor = on ? C.red : C.silver; e.style.background = on ? C.light : '#fff'; });
      qs('.aid', pn).innerHTML = '<h4 style="font-size:20px;margin-bottom:6px;color:' + C.red + '">' + L.name + '</h4>' + rows.map(function(r){ return '<div style="padding:12px 0;border-top:1px solid ' + C.silver + '"><div class="lbl" style="text-transform:none;letter-spacing:0;font-size:13px;font-weight:700;color:' + C.char + '">' + r[0] + '</div><div style="font-size:16px;line-height:1.45;color:' + C.char + ';margin-top:2px">' + esc(r[1]) + '</div></div>'; }).join('');
    }
    return { render: draw };
  } });

/* ---------- Step 2: predict or generate ---------- */
var AI_TASKS = [
  { name: 'Estimate clearance for a new compound', ml: [2, 'It learns from measured values and is scored on test compounds it never saw. The best reference runs in this course reach R2 0.4 to 0.5.'], gen: [0, 'A chat model has no measured clearance for your compounds. It can give a number that sounds right, with no test score behind it.'] },
  { name: 'Score 5,000 library compounds', ml: [2, 'Once trained, a model scores thousands of compounds in seconds, with the same method for each.'], gen: [0, 'It is not built to give consistent numbers at scale, and each answer would need checking.'] },
  { name: 'Propose new molecules', ml: [0, 'A prediction model scores molecules you give it. It does not write new ones.'], gen: [1, 'A model like the GPT in lesson 04 writes candidates. Some are invalid, and none comes with proof that it is active or can be made.'] },
  { name: 'Summarise a set of papers', ml: [0, 'A clearance model reads numbers, not text.'], gen: [1, 'A language model drafts a summary quickly. It can state something the paper does not say, so open the source.'] },
  { name: 'Write an analysis script', ml: [0, 'A prediction model does not write code.'], gen: [1, 'It drafts code that saves time. Read it, then run it on a case with a known answer.'] },
  { name: 'Explain why a compound is cleared quickly', ml: [0, 'It finds patterns that predict. Feature importance shows what it leaned on, not what the liver does.'], gen: [1, 'It can write a plausible explanation from what it read. Plausible is not tested, so treat it as a hypothesis for an experiment.'] }
];
var AI_VERDICT = [['Does not fit', C.red, '#fff', C.red], ['Fits with checks', C.bg, C.char, C.bg], ['Fits well', C.char, '#fff', C.char]];
L6.add({ short: 'Predict or generate', title: 'Which tool fits which job',
  change: 'Click a job to see how well machine learning and generative AI each fit it.',
  watch: 'Machine learning is used to predict: it returns a number for a compound, and that number can be scored against measurements. Generative AI is used to produce: it writes text, code or molecules. Its output has no measurement behind it, so a person must check it. Choose a job and compare the two. A tool that fits well still needs the checks in the last step.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div class="ait"></div>'), cur = 0;
    function card(title, v, text){ var vd = AI_VERDICT[v[0]]; return '<div style="border:1px solid ' + C.silver + ';border-radius:8px;padding:14px 16px"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><b style="font-size:16px;color:' + C.char + '">' + title + '</b><span style="font-size:12px;font-weight:700;padding:3px 10px;border-radius:999px;background:' + vd[1] + ';color:' + vd[2] + '">' + vd[0] + '</span></div><div style="font-size:15px;line-height:1.5;color:' + C.char + ';margin-top:10px">' + esc(text) + '</div></div>'; }
    function draw(){
      var t = AI_TASKS[cur], host = qs('.ait', pn);
      if (!qs('.aitr', pn)) { host.innerHTML = '<div style="display:grid;grid-template-columns:260px 1fr;gap:22px"><div>' + AI_TASKS.map(function(x, i){ return '<button type="button" data-t="' + i + '" style="display:block;width:100%;text-align:left;font:inherit;font-size:14px;padding:10px 12px;margin-bottom:6px;border-radius:6px;cursor:pointer;border:1px solid ' + C.silver + ';background:#fff;color:' + C.char + '">' + esc(x.name) + '</button>'; }).join('') + '</div><div class="aitr" style="display:grid;gap:14px;align-content:start"></div></div>'; qsa('[data-t]', pn).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.t; draw(); }); }); }
      qsa('[data-t]', pn).forEach(function(b){ var on = +b.dataset.t === cur; b.style.borderColor = on ? C.red : C.silver; b.style.background = on ? C.light : '#fff'; b.style.fontWeight = on ? 700 : 500; });
      qs('.aitr', pn).innerHTML = card('Machine learning', t.ml, t.ml[1]) + card('Generative AI', t.gen, t.gen[1]);
    }
    return { render: draw };
  } });

/* ---------- Step 3: outside the training range ---------- */
L6.add({ short: 'Outside the training range', title: 'A prediction model is reliable only near what it learned from',
  change: 'Choose a feature and drag the training range below the plot.',
  watch: 'A machine learning model has only its examples to go on. This demonstration uses a tree model on our compounds. It is trained on compounds whose feature value is below a cut-off, shown as a dashed line, and then scored on test compounds on each side of the line. A test R2 of 0 means no better than guessing the average, and below 0 means worse. The range is the span of feature values the model saw.',
  build: function(api){
    var G_ = 200, st = null;
    var plot = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4 class="pt"></h4><canvas width="600" height="334" style="margin-top:2px"></canvas><div class="note" style="font-size:12px;margin-top:4px">Blue: trained on. Dark rings: test inside. Red rings: test outside.</div>');
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

/* ---------- Step 4: lab noise ---------- */
L6.add({ short: 'Lab noise', title: 'Noisy data caps how accurate a prediction can look',
  change: 'Drag Assay noise and watch the test R2 ceiling.',
  watch: 'A lab measurement is never exact. Measure the same compound again and the number changes a little. That spread is assay noise. A prediction model is judged against the measured value, so noise costs it score even when it is right. Here each test compound is measured a second time, with random noise of the size you choose. The R2 between the two sets of numbers is the most any prediction could be expected to reach.',
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

/* ---------- Step 5: how generative AI goes wrong ---------- */
var AI_FAILS = [
  { name: 'Invalid output', look: 'The GPT in lesson 04 writes text that looks like a molecule, and RDKit rejects some of it, more at a high temperature.', catch_: 'Run a validator such as RDKit on every candidate.' },
  { name: 'Fluent but false', look: 'A language model writes sentences that sound right whether or not they are. A wrong fact reads the same as a true one.', catch_: 'Open the cited source and read the passage. Ask for no claim you cannot check.' },
  { name: 'Copying, not creating', look: 'A generator can reproduce molecules from its training set. The test in lesson 04 counts how many of its molecules are new.', catch_: 'Compare each candidate with the training molecules.' },
  { name: 'No score', look: 'A generator writes a molecule. It does not say how active it is or how fast the liver clears it.', catch_: 'Pass candidates to a prediction model, then measure the best ones.' },
  { name: 'Frozen knowledge', look: 'A language model knows only its training text, up to a cut-off date, unless it is given documents or search tools.', catch_: 'Give it the documents, or search for recent work yourself.' }
];
L6.add({ short: 'How generative AI goes wrong', title: 'Where generative AI fails',
  change: 'Click a failure to see what it looks like and how to catch it.',
  watch: 'A prediction model can be scored against measurements. A generative model produces new content, so there is no measured answer to score it against. That is its use and its weakness. It can draft, propose and write quickly. It cannot tell you that what it wrote is right. Five common failures are listed here.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div class="aif"></div>'), cur = 0;
    function draw(){
      var f = AI_FAILS[cur], host = qs('.aif', pn);
      if (!qs('.aifr', pn)) { host.innerHTML = '<div style="display:grid;grid-template-columns:260px 1fr;gap:22px"><div>' + AI_FAILS.map(function(x, i){ return '<button type="button" data-f="' + i + '" style="display:block;width:100%;text-align:left;font:inherit;font-size:14px;padding:10px 12px;margin-bottom:6px;border-radius:6px;cursor:pointer;border:1px solid ' + C.silver + ';background:#fff;color:' + C.char + '">' + (i + 1) + '. ' + esc(x.name) + '</button>'; }).join('') + '</div><div class="aifr"></div></div>'; qsa('[data-f]', pn).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.f; draw(); }); }); }
      qsa('[data-f]', pn).forEach(function(b){ var on = +b.dataset.f === cur; b.style.borderColor = on ? C.red : C.silver; b.style.background = on ? C.light : '#fff'; b.style.fontWeight = on ? 700 : 500; });
      qs('.aifr', pn).innerHTML = '<h4 style="font-size:20px;color:' + C.red + ';margin-bottom:6px">' + esc(f.name) + '</h4><div style="padding:12px 0;border-top:1px solid ' + C.silver + '"><div class="lbl" style="text-transform:none;letter-spacing:0;font-size:13px;font-weight:700;color:' + C.char + '">What it looks like</div><div style="font-size:16px;line-height:1.45;color:' + C.char + ';margin-top:2px">' + esc(f.look) + '</div></div><div style="padding:12px 0;border-top:1px solid ' + C.silver + '"><div class="lbl" style="text-transform:none;letter-spacing:0;font-size:13px;font-weight:700;color:' + C.char + '">How to catch it</div><div style="font-size:16px;line-height:1.45;color:' + C.char + ';margin-top:2px">' + esc(f.catch_) + '</div></div>';
    }
    return { render: draw };
  } });

/* ---------- Step 6: sort what AI can and cannot do ---------- */
var AI_CLAIMS = [
  [true, 'Rank new compounds by predicted clearance when they resemble the compounds it learned from.', 'This is what the tree and network models do. The test R2 says how well.', 'Machine learning'],
  [true, 'Draft a summary of a set of papers in seconds.', 'A language model reads the text you give it and writes a summary. A person still checks it against the papers.', 'Generative AI'],
  [true, 'Suggest new molecules for a chemist to review.', 'A GPT trained on SMILES writes candidates. Some are invalid and all of them need checking.', 'Generative AI'],
  [true, 'Write code for an analysis that you then review and run.', 'Language models write code well enough to save time. The code still needs reading and a test on a case with a known answer.', 'Generative AI'],
  [true, 'Use what it learned from one dataset to help with a related one.', 'Chemprop trained on logD first and then fine-tuned on 1,000 HLM compounds scored 0.41, against 0.24 from scratch.', 'Machine learning'],
  [false, 'Predict well for chemistry unlike anything it learned from.', 'Outside the range it trained on, AI has nothing to go on. The test R2 fell to zero or below in the outside-the-range step.', 'Both'],
  [false, 'Tell you reliably when its own answer is wrong.', 'A prediction comes with no warning, and a language model can state a wrong fact in the same confident tone as a right one. Only a check against a measurement or a source settles it.', 'Both'],
  [false, 'Score better than the noise in the lab measurements allows.', 'The measured value is the answer key. Noise in it caps the score for any AI.', 'Both'],
  [false, 'Explain why a compound is cleared quickly.', 'AI finds patterns that predict. It does not find causes. Feature importance shows what a model leaned on, not what the liver does.', 'Both'],
  [false, 'Take responsibility for a decision.', 'AI proposes and ranks. A person decides and answers for the result.', 'Both']
];
AI_CLAIMS = [0, 5, 1, 6, 7, 2, 8, 3, 9, 4].map(function(k){ return AI_CLAIMS[k]; });
L6.add({ short: 'Can it or cannot it', title: 'Sort what AI can and cannot do',
  change: 'Read each statement and choose AI can or AI cannot. The reason appears after you choose.',
  watch: 'Each statement describes something people ask of AI. Decide whether AI can do it. After you answer, the reason points back to the lessons, and a tag says whether the answer is about machine learning, generative AI or both.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div class="aic"></div>'), i = 0, score = 0, answered = null;
    function draw(){
      var el = qs('.aic', pn);
      if (i >= AI_CLAIMS.length) { el.innerHTML = '<div class="big" style="color:' + C.red + '">' + score + ' of ' + AI_CLAIMS.length + '</div><p style="font-size:16px;line-height:1.5;color:' + C.char + ';margin:8px 0 18px">' + (score === AI_CLAIMS.length ? 'You sorted every statement the way the lessons did.' : 'That is how many you sorted the way the lessons did. For the others, reread the reason.') + '</p><button type="button" class="btn" data-again>Start again</button>'; qs('[data-again]', el).addEventListener('click', function(){ i = 0; score = 0; answered = null; draw(); }); return; }
      var c = AI_CLAIMS[i];
      el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;color:' + C.char + '"><span>Statement ' + (i + 1) + ' of ' + AI_CLAIMS.length + '</span><span>Matches so far: ' + score + '</span></div>' +
        '<div style="font-size:22px;line-height:1.35;font-weight:700;font-family:var(--display);color:' + C.char + ';margin:22px 0 22px;min-height:90px">' + esc(c[1]) + '</div>' +
        '<div style="display:flex;gap:12px"><button type="button" class="btn" data-a="1"' + (answered !== null ? ' disabled' : '') + '>AI can</button><button type="button" class="btn ghost" data-a="0"' + (answered !== null ? ' disabled' : '') + '>AI cannot</button></div>' +
        (answered === null ? '' : '<div style="margin-top:22px;padding:14px 16px;border-radius:6px;background:' + C.light + ';border-left:4px solid ' + (answered === c[0] ? C.char : C.red) + '"><b style="color:' + (answered === c[0] ? C.char : C.red) + '">' + (answered === c[0] ? 'Matches the lessons. ' : 'Not quite. ') + (c[0] ? 'AI can.' : 'AI cannot.') + '</b> <span style="font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;background:' + C.bg + ';color:' + C.char + ';margin-left:6px">' + c[3] + '</span><div style="margin-top:4px;font-size:15px;line-height:1.5;color:' + C.char + '">' + esc(c[2]) + '</div></div><div style="margin-top:14px"><button type="button" class="btn sm" data-next>' + (i === AI_CLAIMS.length - 1 ? 'See result' : 'Next statement') + '</button></div>');
      qsa('[data-a]', el).forEach(function(b){ b.addEventListener('click', function(){ answered = b.dataset.a === '1'; if (answered === c[0]) score++; draw(); }); });
      var nx = qs('[data-next]', el); if (nx) nx.addEventListener('click', function(){ i++; answered = null; draw(); });
    }
    return { render: draw };
  } });

/* ---------- Step 7: using AI in a project ---------- */
var AI_CHECKS = [
  ['Is the question like what it learned from?', 'A model is reliable near its examples. Compare the new compound, or the new topic, with what it was built on.', 'start', 'Lesson 01, the test set'],
  ['Was it tested fairly?', 'Score it on test compounds held back by scaffold, not at random.', 'start', 'Lesson 01, the test set'],
  ['Is the gain bigger than the noise?', 'Compare a difference in score with the spread between seeds and with the noise in the assay.', 'trees', 'Lesson 02, seeds and ensembles'],
  ['Does a simpler method do as well?', 'Put a tree model or a descriptor model next to the AI, on the same split and with its training time.', 'compare', 'Lesson 06, the leaderboard'],
  ['Can a person verify it?', 'Generative output has no score of its own. Open the source, run the code on a known case, or measure the compound. A person decides and answers for the result.', null, '']
];
L6.add({ short: 'Using it in a project', title: 'Checks before relying on AI',
  change: 'Read the five checks. Each one has a button that opens the lesson where it was shown.',
  watch: 'AI gives an answer. It does not know whether the answer is right, so a person has to check. AI proposes and ranks. People test, decide and answer for the result. These five checks use what the earlier lessons showed.',
  build: function(api){
    var els = AI_CHECKS.map(function(c, i){ var top = 28 + i * 80;
      return api.el('pn fd', 'left:28px;top:' + top + 'px;width:904px;height:72px;display:flex;align-items:center;gap:16px', '<div class="big" style="color:' + C.red + ';font-size:28px;width:34px">' + (i + 1) + '</div><div style="flex:1"><div style="font-size:16px;font-weight:700;color:' + C.char + '">' + esc(c[0]) + '</div><div style="font-size:13.5px;line-height:1.4;color:' + C.char + '">' + esc(c[1]) + '</div></div>' + (c[2] ? '<button type="button" class="btn sm ghost" data-go="' + c[2] + '" style="white-space:nowrap">' + esc(c[3]) + '</button>' : '')); });
    els.forEach(function(e){ var b = qs('[data-go]', e); if (b) b.addEventListener('click', function(){ showLesson(b.dataset.go); }); });
    function render(){ els.forEach(function(e){ e.classList.add('on'); }); }
    return { render: render };
  } });

L6.glossary([
  ['Artificial intelligence', 'Software that does something we would call intelligent if a person did it.'],
  ['Machine learning', 'AI whose rules are learned from examples instead of written by hand.'],
  ['Deep learning', 'Machine learning with neural networks that have many layers.'],
  ['Generative AI', 'Deep learning that writes new text, images or molecules.'],
  ['Hallucination', 'Text from a language model that sounds right and is false.'],
  ['Large language model', 'A GPT-style model trained on very large amounts of text. ChatGPT and Claude are examples.'],
  ['Training range', 'The span of feature values among the training compounds.'],
  ['Test set', 'Compounds held back from training and used only to score the model.'],
  ['Test R2', 'R2 on the test set. 1 is perfect and 0 is no better than the average.'],
  ['Assay noise', 'Random error in a lab measurement. It limits the score any AI can reach.'],
  ['Repeat measurement', 'A second measurement of the same compound.']
]);
setUses(L6, [[], [], ['endpoint'], ['endpoint'], [], [], []]);
