"use strict";
/* Lesson 01: plain-language opening steps for readers with no machine learning or chemistry background.
   They are added after start.js has built the rest of the lesson, then moved to the right places. */
var IN_MOLS = [
  { n: 'Ethanol', s: 'CCO', t: 'The alcohol in drinks. A very small molecule.' },
  { n: 'Hexane', s: 'CCCCCC', t: 'A component of fuel. It contains only carbon and hydrogen.' },
  { n: 'Glucose', s: 'OCC1OC(O)C(O)C(O)C1O', t: 'A sugar. It has many oxygen atoms.' },
  { n: 'Caffeine', s: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', t: 'The stimulant in coffee and tea.' },
  { n: 'Paracetamol', s: 'CC(=O)Nc1ccc(O)cc1', t: 'A common medicine for pain and fever.' },
  { n: 'Aspirin', s: 'CC(=O)Oc1ccccc1C(=O)O', t: 'A medicine for pain and fever.' },
  { n: 'Ibuprofen', s: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', t: 'A medicine for pain and swelling.' }
];
var IN_CACHE = {};
async function inReady(api){ if (RDK) return; api.busy('Loading the chemistry toolkit'); await loadRDKit(); api.idle(); }
function inInfo(m){ if (!m._i) { var d = rdkDescriptors(m.s); m._i = { logp: d.CrippenClogP, tpsa: d.tpsa, mw: d.amw, heavy: d.NumHeavyAtoms, rings: d.NumRings, hbd: d.NumHBD }; } return m._i; }
function inSvg(m, w, hh){ var k = m.s + w + 'x' + hh; if (!IN_CACHE[k]) IN_CACHE[k] = rdkSvg(m.s, w, hh); return IN_CACHE[k]; }
function inSeg(items, cur, attr){ return '<span class="seg" ' + attr + '>' + items.map(function(it, i){ return '<button type="button" data-v="' + i + '"' + (i === cur ? ' class="on"' : '') + '>' + esc(it) + '</button>'; }).join('') + '</span>'; }

/* ---------- Why this matters ---------- */
L0.add({ short: 'Why this matters', title: 'Why predict how the liver treats a compound',
  change: 'Drag the slider to change how fast the liver clears the compound, or press Walk through.',
  watch: 'A medicine works only while enough of it is in the blood. The liver slowly removes compounds from the blood, and this removal is called clearance. A compound that is cleared fast needs more doses a day. One that is cleared slowly lasts longer. Chemists want to know the clearance of a compound before they spend weeks making and testing it. In the lab, clearance is measured by mixing a compound with human liver samples and seeing how fast it disappears. The measurement is called HLM clearance. This course shows how a computer model can predict clearance from a drawing of the molecule. The curve below is an idealised picture, not measured data.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Amount of compound in the blood after one dose</h4><canvas width="530" height="330"></canvas>');
    var fp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="fx" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('How fast the liver clears the compound <input type="range" min="1" max="10" step="0.1" value="4" style="width:260px"> <b class="lv"></b>');
    var rg = qs('input', ctl), lv = qs('.lv', ctl);
    function draw(){
      var lvl = +rg.value, k = 0.045 * lvl, hrs = Math.log(4) / k, pts = [], x;
      lv.textContent = lvl < 3 ? 'slow' : (lvl < 7 ? 'medium' : 'fast');
      for (x = 0; x <= 24.01; x += 0.5) pts.push([x, 100 * Math.exp(-k * x)]);
      lineChart(qs('canvas', cp), [{ pts: pts, color: C.red, w: 3, nodot: true, label: 'compound in the blood' }, { pts: [[0, 25], [24, 25]], color: C.char, w: 1.6, dash: [6, 5], nodot: true, label: 'level needed to work' }], { xl: 'hours after the dose', yl: 'amount (percent of the first peak)', x0: 0, x1: 24, y0: 0, y1: 100 });
      var doses = Math.max(1, Math.ceil(24 / hrs - 0.001));
      qs('.fx', fp).innerHTML = '<div class="ol">With this setting</div><div class="big" style="color:' + C.red + ';font-size:34px;margin:2px 0 0">' + (hrs > 24 ? 'over 24' : hrs.toFixed(1)) + ' h</div><div style="margin-bottom:8px">before the compound drops below the level needed to work.</div><div style="margin-bottom:14px">About <b>' + doses + '</b> dose' + (doses > 1 ? 's' : '') + ' a day to stay above that level.</div>' +
        '<div class="ol">Why a model helps</div><ol style="margin:4px 0 0 18px;padding:0"><li style="margin-bottom:6px">Each new compound idea needs lab work to measure its clearance.</li><li style="margin-bottom:6px">Chemists think of far more ideas than they can make and test.</li><li>A model learns from compounds that were already measured, and predicts clearance for ideas not yet made.</li></ol>';
    }
    rg.addEventListener('input', draw);
    async function play(tok){
      rg.value = 2; draw(); await api.beat('A person takes a dose. The compound enters the blood, and then the liver starts to remove it.', 3000);
      await api.anim(2600, function(t){ rg.value = 2 + 7 * t; draw(); }); await api.beat('When the liver clears a compound fast, it drops below the useful level sooner.', 3000);
      await api.anim(2000, function(t){ rg.value = 9 - 5 * t; draw(); }); await api.beat('So clearance decides how often a patient must take the medicine. A model that predicts it saves lab work.', 3200);
    }
    return { render: draw, play: play };
  } });

/* ---------- What a molecule is ---------- */
L0.add({ short: 'What a molecule is', title: 'How to read a drawing of a molecule',
  change: 'Pick a molecule. Look for the rings, and for the letters that mark atoms other than carbon.',
  watch: 'A molecule is a group of atoms joined by bonds. Chemists draw it as a picture. Each line is a bond. Each corner or line end is a carbon atom unless a letter says otherwise: O is oxygen and N is nitrogen. A closed loop is a ring. The size, the rings and the mix of atoms decide how a compound behaves in the body. A model cannot read the picture, so later steps turn it into numbers.',
  build: function(api){
    var cur = 3, loaded = false;
    var big = api.el('pn', 'left:28px;top:28px;width:520px;height:408px', '<h4 class="nm"></h4><div class="struct" style="height:350px"></div>');
    var side = api.el('pn', 'left:568px;top:28px;width:364px;height:408px;overflow:auto', '<div class="sd" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Molecule ' + inSeg(IN_MOLS.map(function(m){ return m.n; }), cur, 'data-m'));
    function draw(){
      if (!RDK) return; var m = IN_MOLS[cur], d = inInfo(m);
      qs('.nm', big).textContent = m.n; qs('.struct', big).innerHTML = inSvg(m, 480, 340);
      qs('.sd', side).innerHTML = '<div class="ol">' + esc(m.n) + '</div><p>' + esc(m.t) + '</p><div class="nums" style="font-size:13px"><div><span>Atoms drawn (not counting hydrogen)</span><b>' + d.heavy + '</b></div><div><span>Rings</span><b>' + d.rings + '</b></div><div><span>Molecular weight</span><b>' + d.mw.toFixed(0) + '</b></div></div>' +
        '<div class="ol" style="margin-top:14px">How to read the picture</div><ul style="margin:4px 0 0 18px;padding:0"><li style="margin-bottom:5px">A line is a bond between two atoms.</li><li style="margin-bottom:5px">A corner or a line end with no letter is a carbon atom.</li><li style="margin-bottom:5px">A letter marks another atom: <b style="color:#891515">O</b> oxygen, <b style="color:#E1242A">N</b> nitrogen.</li><li style="margin-bottom:5px">Hydrogen atoms are left out. They fill the gaps.</li><li>A closed loop is a ring.</li></ul>' +
        '<p style="margin-top:10px">Molecular weight is the combined mass of the atoms, in units where one hydrogen atom is about 1.</p>';
    }
    qsa('[data-m] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.v; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ await inReady(api); loaded = true; draw(); }
    async function play(tok){
      await render(); for (var k = 0; k < IN_MOLS.length; k++) { api.checkTok(tok); cur = k; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === cur); }); draw(); api.cap(IN_MOLS[k].n + '. ' + IN_MOLS[k].t); await api.sleep(2300); }
      cur = 3; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === cur); }); draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- Oily or watery ---------- */
L0.add({ short: 'Oily or watery', title: 'Some compounds prefer oil and some prefer water',
  change: 'Pick a molecule and watch where its particles settle.',
  watch: 'Shake a compound with oil and water, and it spreads between the two layers. A compound that prefers oil is called lipophilic, which means oil-loving. A compound that prefers water is hydrophilic. A number called logP says which. A logP of 0 means the compound spreads equally between the layers. Each step up of 1 means ten times more in the oil. Each step down of 1 means ten times more in the water. A free chemistry program called RDKit estimates logP from the atoms in the molecule. Oily compounds tend to be broken down faster by the liver. This is a tendency, not a rule.',
  build: function(api){
    var cur = 5, N = 56, P = [], alive = true, last = 0, W = 220, TH = 340;
    var tp = api.el('pn', 'left:28px;top:28px;width:300px;height:408px', '<h4>Oil and water</h4><canvas width="270" height="350"></canvas>');
    var bp = api.el('pn', 'left:348px;top:28px;width:584px;height:408px;overflow:auto', '<div class="bd" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Molecule ' + inSeg(IN_MOLS.map(function(m){ return m.n; }), cur, 'data-m'));
    var rnd = rng(11), zs = []; for (var i = 0; i < N; i++) { zs.push(rnd()); P.push({ x: 40 + rnd() * 190, y: 40 + rnd() * 270, vx: 0, vy: 0 }); }
    function frac(m){ var lp = inInfo(m).logp; return 1 / (1 + Math.pow(10, -lp)); }
    function zone(i, p){ return zs[i] < p ? 0 : 1; }
    function info(){
      if (!RDK) return; var m = IN_MOLS[cur], d = inInfo(m), p = frac(m), ms = IN_MOLS.map(function(x){ return inInfo(x).logp; }), lo = Math.min.apply(null, ms), hi = Math.max.apply(null, ms), span = Math.max(Math.abs(lo), Math.abs(hi), 1);
      var ratio = Math.pow(10, Math.abs(d.logp));
      qs('.bd', bp).innerHTML = '<div class="ol">' + esc(m.n) + '</div><div style="display:flex;align-items:flex-end;gap:18px"><div><div class="big" style="color:' + C.red + '">' + d.logp.toFixed(1) + '</div><div style="font-size:12px">logP</div></div><div style="font-size:14px;padding-bottom:6px">' + (Math.abs(d.logp) < 0.3 ? 'About equal amounts in oil and water.' : 'About <b>' + (ratio >= 10 ? Math.round(ratio) : ratio.toFixed(1)) + ' times</b> more in ' + (d.logp > 0 ? 'oil' : 'water') + ' than in the other layer.') + '</div></div>' +
        '<div class="ol" style="margin-top:16px">All seven molecules, from watery to oily</div>' + IN_MOLS.map(function(x, k){ var v = Math.abs(ms[k]) < 0.05 ? 0 : ms[k], w = Math.abs(v) / span * 46; return '<div style="display:grid;grid-template-columns:96px 1fr 44px;gap:8px;align-items:center;margin:5px 0;font-size:12.5px' + (k === cur ? ';font-weight:700' : '') + '"><span>' + esc(x.n) + '</span><div style="position:relative;height:14px;background:' + C.light + ';border-radius:3px"><i style="position:absolute;top:0;height:14px;border-radius:3px;background:' + (k === cur ? C.red : C.bg) + ';' + (v >= 0 ? 'left:50%;width:' + w + '%' : 'right:50%;width:' + w + '%') + '"></i><i style="position:absolute;left:50%;top:-2px;height:18px;width:1px;background:' + C.char + '"></i></div><b style="text-align:right">' + v.toFixed(1) + '</b></div>'; }).join('') +
        '<div style="display:flex;justify-content:space-between;font-size:11px;color:' + C.mid + ';margin:2px 100px 0 104px"><span>prefers water</span><span>prefers oil</span></div>' +
        '<p style="margin-top:12px">Compare hexane, which has only carbon and hydrogen, with glucose, which has many oxygen atoms that attract water.</p>';
    }
    function frame(now){
      if (!alive) return; requestAnimationFrame(frame); var r = qs('canvas', tp).getBoundingClientRect(); if (r.bottom < 0 || r.top > innerHeight || r.width === 0) { last = now; return; } var dt = Math.min(0.05, (now - last) / 1000); last = now;
      var g = cv(qs('canvas', tp)), x = g.x, p = RDK ? frac(IN_MOLS[cur]) : 0.5, midY = 175, i, pt;
      x.fillStyle = '#F3E7C9'; x.fillRect(30, 30, 210, midY - 30); x.fillStyle = '#D6E2EE'; x.fillRect(30, midY, 210, 310 - midY + 10);
      x.strokeStyle = C.char; x.lineWidth = 2; x.beginPath(); x.moveTo(30, 24); x.lineTo(30, 330); x.lineTo(240, 330); x.lineTo(240, 24); x.stroke();
      x.fillStyle = C.char; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText('oil', 36, 46); x.fillText('water', 36, 344);
      for (i = 0; i < N; i++) { pt = P[i]; var z = zone(i, p), y0 = z === 0 ? 56 : midY + 14, y1 = z === 0 ? midY - 10 : 316; pt.vx += (Math.random() - 0.5) * 260 * dt; pt.vy += (Math.random() - 0.5) * 260 * dt; if (pt.y < y0) pt.vy += 700 * dt; if (pt.y > y1) pt.vy -= 700 * dt; if (pt.x < 42) pt.vx += 500 * dt; if (pt.x > 228) pt.vx -= 500 * dt; pt.vx *= 0.9; pt.vy *= 0.9; pt.x += pt.vx * dt; pt.y += pt.vy * dt; if (pt.y > midY - 10 && pt.y < midY + 14 && z !== (pt.y < midY ? 0 : 1)) pt.vy += (z === 0 ? -1 : 1) * 500 * dt; x.fillStyle = C.red; x.beginPath(); x.arc(pt.x, pt.y, 4.4, 0, 6.3); x.fill(); }
      var inOil = 0; for (i = 0; i < N; i++) if (zone(i, p) === 0) inOil++; x.textAlign = 'right'; x.fillStyle = C.char; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.fillText(Math.round(100 * inOil / N) + '% in oil', 234, 46); x.fillText(Math.round(100 - 100 * inOil / N) + '% in water', 234, 344);
    }
    requestAnimationFrame(function(t){ last = t; frame(t); });
    qsa('[data-m] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.v; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); info(); }); });
    async function render(){ await inReady(api); info(); }
    async function play(tok){
      await render(); var order = [2, 1, 6, 0, 3]; api.cap('Each red dot is a particle of the compound. It settles in the layer it prefers.');
      for (var k = 0; k < order.length; k++) { api.checkTok(tok); cur = order[k]; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === cur); }); info(); api.cap(IN_MOLS[cur].n + ' has logP ' + inInfo(IN_MOLS[cur]).logp.toFixed(1) + '.'); await api.sleep(2600); }
      api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- The data is a table ---------- */
L0.add({ short: 'The data is a table', title: 'The data is a table of examples',
  change: 'Press Predict the new compound. Press Different compounds for another set.',
  watch: 'Machine learning starts with a table. Each row is one compound. The columns on the left are features: numbers that describe the compound, such as logP (how oily it is), TPSA (how much of its surface attracts water) and MW (molecular weight, the combined mass of its atoms). The last column is the target: the answer the lab measured. A model studies rows where the answer is known. Then it is given a new row with the answer hidden, and it fills in a guess.',
  build: function(api){
    var seed = 3, FEATS = ['logP', 'TPSA', 'MW'], newPred = false;
    var tp = api.el('pn', 'left:28px;top:28px;width:904px;height:408px;overflow:auto', '<div class="tb"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-pred>Predict the new compound</button> <button type="button" class="btn sm ghost" data-next>Different compounds</button>');
    var rows = null, nw = null, tree = null;
    function pick(){
      var arr = STRUCTS[EP] || [], test = arr.filter(function(x){ return D.test[x.i]; }), r = rng(seed * 7 + 1);
      function sh(a){ var o = a.slice(); for (var i = o.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = o[i]; o[i] = o[j]; o[j] = t; } return o; }
      rows = sh(D.TRAIN.slice(0, 600)).slice(0, 5).map(function(i){ return structOf(i) || { i: i, svg: '' }; }); nw = (test.length ? sh(test) : sh(arr))[0]; newPred = false;
      var key = EP + 'intro'; D.cache[key] = D.cache[key] || fitTree(D.BASE.slice(), D.TRAIN, 5, 20, 1); tree = D.cache[key];
    }
    function cell(f, i){ return '<td style="text-align:right">' + fmtFeat(f, D.COL[f][i]) + '</td>'; }
    function draw(reveal){
      if (!rows) return; var fs = FEATS.filter(function(f){ return D.COL[f]; }), pv = predictTree(tree, D.B, nw.i), mv = D.Y[nw.i];
      function row(x, isNew){ return '<tr style="border-top:1px solid ' + C.silver + (isNew ? ';background:#FBE9E9' : '') + '"><td style="width:120px">' + (x.svg ? '<div class="struct" style="height:54px;width:110px">' + x.svg + '</div>' : '<span style="font-size:12px;color:' + C.mid + '">compound ' + (x.i + 1) + '</span>') + '</td>' + fs.map(function(f){ return cell(f, x.i); }).join('') + '<td style="text-align:right;border-left:2px solid ' + C.red + ';font-weight:700;width:190px">' + (isNew ? (newPred ? '<span style="color:' + C.red + '">' + pv.toFixed(2) + '</span> <span style="font-weight:400;font-size:11px">guess, lab: ' + mv.toFixed(2) + '</span>' : '<span style="color:' + C.red + ';font-size:20px">?</span>') : D.Y[x.i].toFixed(2)) + '</td></tr>'; }
      qs('.tb', tp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:14px;color:' + C.char + '"><thead><tr><th></th><th colspan="' + fs.length + '" style="text-align:center;background:' + C.light + ';padding:6px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:' + C.mid + '">Features: what the model is given</th><th style="text-align:center;background:#FBE9E9;color:' + C.dred + ';padding:6px;font-size:11px;letter-spacing:.1em;text-transform:uppercase">Target: what we predict</th></tr><tr style="font-weight:700"><th style="text-align:left;padding:6px">Compound</th>' + fs.map(function(f){ return '<th style="text-align:right;padding:6px">' + f + '</th>'; }).join('') + '<th style="text-align:right;padding:6px;border-left:2px solid ' + C.red + '">' + esc(unitWord()) + '</th></tr></thead><tbody>' + rows.map(function(x){ return row(x, false); }).join('') + '<tr><td colspan="' + (fs.length + 2) + '" style="padding:8px 0 2px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:' + C.red + ';font-weight:700">A new compound, answer hidden</td></tr>' + row(nw, true) + '</tbody></table>' +
        '<p style="font-size:12.5px;margin:10px 0 0;color:' + C.char + '">The rows above the new compound are the examples. The model looks for a pattern linking the feature columns to the target column. ' + (newPred ? 'Its guess for the new compound is shown beside the value the lab measured. The gap is the error.' : 'The new compound has the same kind of features, but the model has never seen its answer.') + '</p>';
    }
    qs('[data-pred]', ctl).addEventListener('click', function(){ newPred = true; draw(); });
    qs('[data-next]', ctl).addEventListener('click', function(){ seed++; pick(); draw(); });
    async function render(){ await loadStructs(); pick(); draw(); }
    async function play(tok){
      await render(); api.cap('Every row is a compound that was made and measured. The features are on the left and the measured answer is on the right.'); await api.sleep(3600);
      api.cap('Now a new compound arrives. We know its features, but the answer is hidden.'); await api.sleep(3000); newPred = true; draw(); api.cap('The model reads the features and guesses. The lab value shows how close the guess was.'); await api.sleep(3400); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- Practice and real exam ---------- */
L0.add({ short: 'Practice and real exam', title: 'A model can memorise its practice questions',
  change: 'Drag the slider to make the model study its practice compounds harder, and watch the two scores.',
  watch: 'Think of a student who prepares for an exam with practice questions. A student who memorises the answers scores full marks on the practice questions and badly on new ones. A student who learns the pattern does well on both. A model is the same. The practice questions are the training compounds, which the model studies. The new questions are the test compounds, which are held back and never studied. The score here is R2: 1 is perfect, and 0 is no better than always guessing the average. The model here is a flowchart of yes or no questions about a compound, such as "is logP above 3?". The slider sets how many questions in a row it may ask. More questions means more memorising.',
  build: function(api){
    var dp = 4;
    var gp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px;overflow:auto', '<div class="gb"></div>');
    var cp = api.el('pn', 'left:468px;top:28px;width:464px;height:408px', '<h4>Score as the model studies harder</h4><canvas width="436" height="340"></canvas>');
    var ctl = api.addControl('Questions the model may ask in a row <input type="range" min="1" max="14" step="1" value="4" style="width:240px"> <b class="dv">4</b>');
    var rg = qs('input', ctl), dv = qs('.dv', ctl);
    function curves(){ var key = EP + 'exam'; if (D.cache[key]) return D.cache[key]; var o = []; for (var d = 1; d <= 14; d++) { var t = fitTree(D.BASE.slice(), D.TRAIN, d, 5, 1), p = treeAll(t, D.N); o.push({ d: d, tr: r2(D.TRAIN, p), te: r2(D.TEST, p) }); } return (D.cache[key] = o); }
    function bar(label, sub, v, col){ var w = clamp(v, 0, 1) * 100; return '<div style="margin:12px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="color:' + col + ';font-size:28px;margin:0">' + v.toFixed(2) + '</span></div><div style="font-size:12px;margin-bottom:4px">' + sub + '</div><div style="height:16px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:16px;width:' + w + '%;background:' + col + ';transition:width .4s"></i></div>' + (v < 0 ? '<div style="font-size:11px;color:' + C.red + '">Below zero: worse than guessing the average.</div>' : '') + '</div>'; }
    function draw(){
      dv.textContent = rg.value; dp = +rg.value; var cs = curves(), c = cs[dp - 1], gap = c.tr - c.te;
      var msg = dp <= 2 ? 'The model asks too few questions. It misses the pattern even in the practice compounds.' : (gap < 0.25 ? 'The model has learned the pattern. The two scores are close.' : 'The model has started to memorise. Practice scores climb while the score on new compounds stops rising or falls.');
      qs('.gb', gp).innerHTML = '<div class="ol">Depth ' + dp + '</div>' + bar('Practice questions', 'Training compounds, studied by the model', c.tr, C.char) + bar('New questions', 'Test compounds, held back and never studied', c.te, C.red) + '<p style="font-size:13px;margin-top:10px;line-height:1.5;color:' + C.char + '">' + msg + '</p><p style="font-size:12px;color:' + C.mid + '">What counts is the score on the new questions. A model that is only good at its practice questions is not useful for compounds nobody has made yet.</p>';
      lineChart(qs('canvas', cp), [{ pts: cs.map(function(q){ return [q.d, q.tr]; }), color: C.char, w: 2.6, nodot: true, label: 'practice (training compounds)' }, { pts: cs.map(function(q){ return [q.d, q.te]; }), color: C.red, w: 2.8, nodot: true, label: 'new (test compounds)' }, { pts: [[dp, c.tr], [dp, c.te]], color: C.bg, w: 2, nodot: true, dash: [4, 4] }], { xl: 'questions the model may ask in a row', yl: 'score (R2)', x0: 1, x1: 14, y0: Math.min(0, Math.floor(Math.min.apply(null, cs.map(function(q){ return q.te; })) * 10) / 10), y1: 1 });
    }
    rg.addEventListener('input', draw);
    async function render(){ draw(); }
    async function play(tok){
      rg.value = 1; draw(); api.cap('One question only. The model is too simple to learn the pattern.'); await api.sleep(2600);
      await api.anim(4200, function(t){ rg.value = Math.round(1 + 13 * t); draw(); }); api.cap('As the model asks more questions, its practice score keeps climbing. Its score on new compounds does not.'); await api.sleep(3600);
      rg.value = 4; draw(); api.cap('A middle setting does best on new compounds.'); await api.sleep(2800); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* put the new steps where a newcomer needs them */
L0.move('Why this matters', 0); L0.move('What a molecule is', 1); L0.move('Oily or watery', 2); L0.move('The data is a table', 3); L0.move('Practice and real exam', 7);
