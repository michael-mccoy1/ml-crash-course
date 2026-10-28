"use strict";
/* Lesson 01, Beginner section: how molecules and assay results become model input, and how performance is reported.
   Written for chemists who have not worked with machine learning. */
var IN_MOLS = [
  { n: 'Ethanol', s: 'CCO' },
  { n: 'Hexane', s: 'CCCCCC' },
  { n: 'Glucose', s: 'OCC1OC(O)C(O)C(O)C1O' },
  { n: 'Caffeine', s: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  { n: 'Paracetamol', s: 'CC(=O)Nc1ccc(O)cc1' },
  { n: 'Aspirin', s: 'CC(=O)Oc1ccccc1C(=O)O' },
  { n: 'Ibuprofen', s: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O' }
];
var IN_CACHE = {};
async function inReady(api){ if (RDK) return; api.busy('Loading RDKit'); await loadRDKit(); api.idle(); }
function inInfo(m){ if (!m._i) { var d = rdkDescriptors(m.s); m._i = { logp: d.CrippenClogP, tpsa: d.tpsa, mw: d.amw, heavy: d.NumHeavyAtoms, rings: d.NumRings, hbd: d.NumHBD, hba: d.NumHBA, rotb: d.NumRotatableBonds, fsp3: d.FractionCSP3 }; } return m._i; }
function inSvg(m, w, hh){ var k = m.s + w + 'x' + hh; if (!IN_CACHE[k]) IN_CACHE[k] = rdkSvg(m.s, w, hh); return IN_CACHE[k]; }
function inSeg(items, cur, attr){ return '<span class="seg" ' + attr + '>' + items.map(function(it, i){ return '<button type="button" data-v="' + i + '"' + (i === cur ? ' class="on"' : '') + '>' + esc(it) + '</button>'; }).join('') + '</span>'; }
function inFp(m){ if (!m._fp) { var mol = RDK.get_mol(m.s); m._fp = mol.get_morgan_fp(JSON.stringify({ radius: 2, nBits: 1024 })); mol.delete(); } return m._fp; }

/* ---------- Molecular representations ---------- */
L0.add({ short: 'Molecular representations', title: 'Molecular representations',
  change: 'Select molecules A and B. Compare the three representations and the fingerprint similarity.',
  watch: 'A model takes a fixed-length vector of numbers as input, so each molecule must be converted to one. Three representations are common. A SMILES string is text, used by sequence models such as the GPT in lesson 04. Descriptors are computed properties, one vector entry per descriptor, and are the features used in most of this course. A circular fingerprint hashes each atom environment within a given radius to a position in a fixed-length bit vector. The similarity of two fingerprints is the Tanimoto coefficient: the number of bits set in both, divided by the number set in either.',
  build: function(api){
    var a = 5, b = 4;
    var pa = api.el('pn', 'left:28px;top:20px;width:290px;height:240px', '<h4 class="ha"></h4><div class="struct" style="height:196px"></div>');
    var pb = api.el('pn', 'left:28px;top:272px;width:290px;height:240px', '<h4 class="hb"></h4><div class="struct" style="height:196px"></div>');
    var sp = api.el('pn', 'left:338px;top:20px;width:594px;height:96px', '<h4>SMILES</h4><div class="sm" style="font:13px Cascadia Mono,Consolas,monospace;color:' + C.char + ';line-height:1.7;word-break:break-all"></div>');
    var dp = api.el('pn', 'left:338px;top:128px;width:594px;height:196px', '<h4>Descriptors</h4><div class="ds"></div>');
    var fp = api.el('pn', 'left:338px;top:336px;width:594px;height:176px', '<h4>Morgan fingerprint, radius 2, 1024 bits</h4><div class="fpw" style="display:flex;gap:26px;align-items:flex-start"></div>');
    var ctl = api.addControl('A ' + inSeg(IN_MOLS.map(function(m){ return m.n; }), a, 'data-a') + ' B ' + inSeg(IN_MOLS.map(function(m){ return m.n; }), b, 'data-b'));
    function grid(bits, other, mode){
      var cell = 4, n = 32, c = '<svg width="' + (n * cell) + '" height="' + (n * cell) + '" style="background:' + C.light + ';display:block">'; for (var i = 0; i < bits.length; i++) { var on = bits[i] === '1', o = other ? other[i] === '1' : false, col = null; if (mode === 'both') { if (on && o) col = C.red; else if (on || o) col = C.bg; } else if (on) col = C.char; if (col) c += '<rect x="' + (i % n) * cell + '" y="' + Math.floor(i / n) * cell + '" width="' + (cell - 0.6) + '" height="' + (cell - 0.6) + '" fill="' + col + '"/>'; } return c + '</svg>';
    }
    function draw(){
      if (!RDK) return; var A = IN_MOLS[a], B = IN_MOLS[b], fa = inFp(A), fb = inFp(B), ia = inInfo(A), ib = inInfo(B), cnt = function(f){ return (f.match(/1/g) || []).length; }, both = 0, either = 0;
      for (var i = 0; i < fa.length; i++) { if (fa[i] === '1' && fb[i] === '1') both++; if (fa[i] === '1' || fb[i] === '1') either++; }
      qs('.ha', pa).textContent = 'A: ' + A.n; qs('.hb', pb).textContent = 'B: ' + B.n; qs('.struct', pa).innerHTML = inSvg(A, 250, 190); qs('.struct', pb).innerHTML = inSvg(B, 250, 190);
      qs('.sm', sp).innerHTML = '<b>A</b> ' + esc(A.s) + ' (' + (A.s.match(TOK) || []).length + ' tokens)<br><b>B</b> ' + esc(B.s) + ' (' + (B.s.match(TOK) || []).length + ' tokens)';
      var rows = [['Molecular weight', 'mw', 0], ['Crippen logP', 'logp', 2], ['TPSA', 'tpsa', 1], ['H-bond donors', 'hbd', 0], ['H-bond acceptors', 'hba', 0], ['Rotatable bonds', 'rotb', 0], ['Rings', 'rings', 0], ['Fsp3', 'fsp3', 2]];
      qs('.ds', dp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12.5px;color:' + C.char + '"><tr style="font-weight:700"><td></td>' + rows.map(function(r){ return '<td style="text-align:right;padding:2px 6px">' + r[0] + '</td>'; }).join('') + '</tr><tr style="border-top:1px solid ' + C.silver + '"><td style="font-weight:700">A</td>' + rows.map(function(r){ return '<td style="text-align:right;padding:5px 6px">' + (+ia[r[1]]).toFixed(r[2]) + '</td>'; }).join('') + '</tr><tr style="border-top:1px solid ' + C.silver + '"><td style="font-weight:700">B</td>' + rows.map(function(r){ return '<td style="text-align:right;padding:5px 6px">' + (+ib[r[1]]).toFixed(r[2]) + '</td>'; }).join('') + '</tr></table><p style="font-size:12.5px;margin:10px 0 0;color:' + C.char + '">Each molecule becomes a vector of 8 numbers. A model trained on descriptors receives this vector and nothing else.</p>';
      qs('.fpw', fp).innerHTML = '<div><div style="font-size:11px;margin-bottom:3px">A: ' + cnt(fa) + ' bits set</div>' + grid(fa, null) + '</div><div><div style="font-size:11px;margin-bottom:3px">B: ' + cnt(fb) + ' bits set</div>' + grid(fb, null) + '</div><div><div style="font-size:11px;margin-bottom:3px">Overlap (red: both)</div>' + grid(fa, fb, 'both') + '</div><div style="font-size:12.5px;color:' + C.char + ';line-height:1.5;padding-top:14px"><div class="big" style="color:' + C.red + ';font-size:30px">' + (either ? (both / either).toFixed(2) : '0.00') + '</div>Tanimoto similarity<br><span style="font-size:11.5px;color:' + C.mid + '">' + both + ' shared bits of ' + either + ' set in either</span></div>';
    }
    qsa('[data-a] button', ctl).forEach(function(x){ x.addEventListener('click', function(){ a = +x.dataset.v; qsa('[data-a] button', ctl).forEach(function(y){ y.classList.toggle('on', y === x); }); draw(); }); });
    qsa('[data-b] button', ctl).forEach(function(x){ x.addEventListener('click', function(){ b = +x.dataset.v; qsa('[data-b] button', ctl).forEach(function(y){ y.classList.toggle('on', y === x); }); draw(); }); });
    function pick(seg, k){ qsa(seg + ' button', ctl).forEach(function(y){ y.classList.toggle('on', +y.dataset.v === k); }); }
    async function render(){ await inReady(api); draw(); }
    async function play(tok){
      await render(); a = 5; b = 6; pick('[data-a]', a); pick('[data-b]', b); draw(); api.cap('Aspirin and ibuprofen share the carboxylic acid and the aromatic ring, so they share fingerprint bits.'); await api.sleep(3600);
      a = 5; b = 1; pick('[data-b]', b); draw(); api.cap('Aspirin and hexane share almost no bits. The Tanimoto similarity is near zero.'); await api.sleep(3400); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- Dataset structure ---------- */
L0.add({ short: 'The data is a table', title: 'Dataset structure',
  change: 'Select Predict to estimate the target of the new compound. Select Resample to draw another set.',
  watch: 'A supervised model is trained on a table. Each row is one compound. The feature columns are numeric inputs computed from the structure: here logP, TPSA and molecular weight (MW). The target column is the measured value to be predicted. Training uses rows whose target is known. At prediction time the model receives the features of a new compound and returns an estimate of its target. The last row shows this: the target is withheld from the model, then shown for comparison.',
  build: function(api){
    var seed = 3, FEATS = ['logP', 'TPSA', 'MW'], newPred = false;
    var tp = api.el('pn', 'left:28px;top:28px;width:904px;height:408px;overflow:auto', '<div class="tb"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-pred>Predict</button> <button type="button" class="btn sm ghost" data-next>Resample</button>');
    var rows = null, nw = null, tree = null;
    function pick(){
      var arr = STRUCTS[EP] || [], test = arr.filter(function(x){ return D.test[x.i]; }), r = rng(seed * 7 + 1);
      function sh(a){ var o = a.slice(); for (var i = o.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = o[i]; o[i] = o[j]; o[j] = t; } return o; }
      rows = sh(D.TRAIN.slice(0, 600)).slice(0, 5).map(function(i){ return structOf(i) || { i: i, svg: '' }; }); nw = (test.length ? sh(test) : sh(arr))[0]; newPred = false;
      var key = EP + 'intro'; D.cache[key] = D.cache[key] || fitTree(D.BASE.slice(), D.TRAIN, 5, 20, 1); tree = D.cache[key];
    }
    function cell(f, i){ return '<td style="text-align:right">' + fmtFeat(f, D.COL[f][i]) + '</td>'; }
    function draw(){
      if (!rows) return; var fs = FEATS.filter(function(f){ return D.COL[f]; }), pv = predictTree(tree, D.B, nw.i), mv = D.Y[nw.i];
      function row(x, isNew){ return '<tr style="border-top:1px solid ' + C.silver + (isNew ? ';background:#FBE9E9' : '') + '"><td style="width:120px">' + (x.svg ? '<div class="struct" style="height:54px;width:110px">' + x.svg + '</div>' : '<span style="font-size:12px;color:' + C.mid + '">compound ' + (x.i + 1) + '</span>') + '</td>' + fs.map(function(f){ return cell(f, x.i); }).join('') + '<td style="text-align:right;border-left:2px solid ' + C.red + ';font-weight:700;width:210px">' + (isNew ? (newPred ? '<span style="color:' + C.red + '">' + pv.toFixed(2) + '</span> <span style="font-weight:400;font-size:11px">predicted, measured ' + mv.toFixed(2) + '</span>' : '<span style="color:' + C.red + ';font-size:16px">withheld</span>') : D.Y[x.i].toFixed(2)) + '</td></tr>'; }
      qs('.tb', tp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:14px;color:' + C.char + '"><thead><tr><th></th><th colspan="' + fs.length + '" style="text-align:center;background:' + C.light + ';padding:6px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:' + C.mid + '">Features: model input</th><th style="text-align:center;background:#FBE9E9;color:' + C.dred + ';padding:6px;font-size:11px;letter-spacing:.1em;text-transform:uppercase">Target: measured value</th></tr><tr style="font-weight:700"><th style="text-align:left;padding:6px">Compound</th>' + fs.map(function(f){ return '<th style="text-align:right;padding:6px">' + f + '</th>'; }).join('') + '<th style="text-align:right;padding:6px;border-left:2px solid ' + C.red + '">' + esc(unitWord()) + '</th></tr></thead><tbody>' + rows.map(function(x){ return row(x, false); }).join('') + '<tr><td colspan="' + (fs.length + 2) + '" style="padding:8px 0 2px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:' + C.red + ';font-weight:700">New compound, target withheld</td></tr>' + row(nw, true) + '</tbody></table>' +
        '<p style="font-size:12.5px;margin:10px 0 0;color:' + C.char + '">The model is fitted to the rows above. ' + (newPred ? 'Its prediction for the new compound is shown beside the measured value. The difference is the prediction error.' : 'The new compound has the same kinds of features. Its target is not available to the model.') + '</p>';
    }
    qs('[data-pred]', ctl).addEventListener('click', function(){ newPred = true; draw(); });
    qs('[data-next]', ctl).addEventListener('click', function(){ seed++; pick(); draw(); });
    async function render(){ await loadStructs(); pick(); draw(); }
    async function play(tok){
      await render(); api.cap('Each row is a compound with a measured target. The feature columns are the model input.'); await api.sleep(3400);
      api.cap('A new compound has features but no target available to the model.'); await api.sleep(2800); newPred = true; draw(); api.cap('The model returns a prediction. The measured value gives the prediction error.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- Training and test performance ---------- */
L0.add({ short: 'Practice and real exam', title: 'Training and test performance',
  change: 'Drag Tree depth and compare training and test R2.',
  watch: 'A model that fits its training compounds well can still predict new compounds poorly. This is overfitting: the model has fitted noise specific to the training set. Performance is therefore reported on a test set, compounds that were excluded from training. R2 is the fraction of the variance in the measured values explained by the predictions: 1 is a perfect fit and 0 equals predicting the training mean for every compound. The model here is a decision tree, a sequence of threshold questions on the features, such as logP above 3. Tree depth is the maximum number of questions in a row, and a greater depth gives a more flexible model.',
  build: function(api){
    var dp = 4;
    var gp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px;overflow:auto', '<div class="gb"></div>');
    var cp = api.el('pn', 'left:468px;top:28px;width:464px;height:408px', '<h4>R2 against tree depth</h4><canvas width="436" height="340"></canvas>');
    var ctl = api.addControl('Tree depth <input type="range" min="1" max="14" step="1" value="4" style="width:240px"> <b class="dv">4</b>');
    var rg = qs('input', ctl), dv = qs('.dv', ctl);
    function curves(){ var key = EP + 'exam'; if (D.cache[key]) return D.cache[key]; var o = []; for (var d = 1; d <= 14; d++) { var t = fitTree(D.BASE.slice(), D.TRAIN, d, 5, 1), p = treeAll(t, D.N); o.push({ d: d, tr: r2(D.TRAIN, p), te: r2(D.TEST, p) }); } return (D.cache[key] = o); }
    function bar(label, sub, v, col){ var w = clamp(v, 0, 1) * 100; return '<div style="margin:12px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="color:' + col + ';font-size:28px;margin:0">' + v.toFixed(2) + '</span></div><div style="font-size:12px;margin-bottom:4px">' + sub + '</div><div style="height:16px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:16px;width:' + w + '%;background:' + col + ';transition:width .4s"></i></div>' + (v < 0 ? '<div style="font-size:11px;color:' + C.red + '">Negative R2: worse than predicting the training mean.</div>' : '') + '</div>'; }
    function draw(){
      dv.textContent = rg.value; dp = +rg.value; var cs = curves(), c = cs[dp - 1], gap = c.tr - c.te;
      var msg = dp <= 2 ? 'Underfitting: both R2 values are low, so the tree is too shallow to capture the trend.' : (gap < 0.25 ? 'The training and test R2 values are close.' : 'Overfitting: training R2 keeps rising while test R2 stops rising and then falls.');
      qs('.gb', gp).innerHTML = '<div class="ol">Depth ' + dp + '</div>' + bar('Training R2', 'Compounds used to fit the model', c.tr, C.char) + bar('Test R2', 'Compounds excluded from training', c.te, C.red) + '<p style="font-size:13px;margin-top:10px;line-height:1.5;color:' + C.char + '">' + msg + '</p><p style="font-size:12px;color:' + C.mid + '">Test R2 is the estimate of performance on new compounds.</p>';
      lineChart(qs('canvas', cp), [{ pts: cs.map(function(q){ return [q.d, q.tr]; }), color: C.char, w: 2.6, nodot: true, label: 'training set' }, { pts: cs.map(function(q){ return [q.d, q.te]; }), color: C.red, w: 2.8, nodot: true, label: 'test set' }, { pts: [[dp, c.tr], [dp, c.te]], color: C.bg, w: 2, nodot: true, dash: [4, 4] }], { xl: 'tree depth', yl: 'R2', x0: 1, x1: 14, y0: Math.min(0, Math.floor(Math.min.apply(null, cs.map(function(q){ return q.te; })) * 10) / 10), y1: 1 });
    }
    rg.addEventListener('input', draw);
    async function render(){ draw(); }
    async function play(tok){
      rg.value = 1; draw(); api.cap('Depth 1: the tree underfits. Both R2 values are low.'); await api.sleep(2600);
      await api.anim(4200, function(t){ rg.value = Math.round(1 + 13 * t); draw(); }); api.cap('Training R2 rises with depth. Test R2 stops rising and then falls.'); await api.sleep(3600);
      rg.value = 4; draw(); api.cap('A moderate depth gives the highest test R2.'); await api.sleep(2800); api.cap(null);
    }
    return { render: render, play: play };
  } });
