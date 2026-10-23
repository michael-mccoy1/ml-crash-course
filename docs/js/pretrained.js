"use strict";
/* Lesson 5: pre-trained transformers. GROVER is explained, and a miniature of pre-training runs in the browser. */
var L6 = new Lesson('pre', 'Pre-trained models');
L6.optsHTML = function(){
  return sharedOptions() + optRange('plabels', 'Labelled compounds', 25, 2400, 25, 400, 'How many compounds with lab values a model may learn from.') +
    optGroup('Fine-tuning: what to update', seg([['head', 'Head only'], ['all', 'Everything']], FT.mode, 'data-ftmode'), 'Head only keeps the GPT frozen. Everything also lets the GPT\'s own weights move.') +
    optRange('ftlr', 'Fine-tuning learning rate', 0.0002, 0.01, 0.0002, 0.002, 'Size of each weight update.') + optRange('ftsteps', 'Fine-tuning steps', 20, 200, 10, 80, 'Mini-batches of 16 labelled compounds.') +
    filterGroup() + optGroup('About this lesson', '<div class="hint">GROVER is a published model that runs in Python. The steps here explain the idea and run small models in your browser.</div>');
};
var pTimer = null;
L6.onOpts = function(root){
  ['plabels', 'ftlr', 'ftsteps'].forEach(function(id){ $(id).addEventListener('input', function(){ if (L6.live) L6.live(); clearTimeout(pTimer); pTimer = setTimeout(function(){ L6.refresh(); }, 160); }); });
  qsa('.seg[data-ftmode] button', root).forEach(function(b){ b.addEventListener('click', function(){ FT.mode = b.dataset.v; qsa('.seg[data-ftmode] button', root).forEach(function(x){ x.classList.toggle('on', x === b); }); if (L6.live) L6.live(); }); });
};
featListeners.push(function(){ if (L6.built) L6.refresh(); });
function logX(n, lo, hi, a, b){ return a + (Math.log(n) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (b - a); }
function curveChart(canvas, series, marker, o){
  var g = cv(canvas), x = g.x, box = { l: 46, r: g.w - 14, t: 14, b: g.h - 46 }, all = []; series.forEach(function(s){ s.pts.forEach(function(p){ all.push(p[1]); }); });
  var lo = 20, hi = Math.max.apply(null, series.map(function(s){ return s.pts.length ? s.pts[s.pts.length - 1][0] : 25; }).concat([100])), y1 = Math.max(0.3, Math.max.apply(null, all.concat([0])) * 1.15), y0 = Math.min(0, Math.min.apply(null, all.concat([0])));
  x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; for (var k = 0; k <= 4; k++) { var y = box.b - k / 4 * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText((y0 + (y1 - y0) * k / 4).toFixed(2), box.l - 6, y + 4); }
  var sx = function(n){ return logX(n, lo, hi, box.l, box.r); }, sy = function(v){ return box.b - (v - y0) / (y1 - y0) * (box.b - box.t); };
  var sx = function(n){ return logX(n, lo, hi, box.l, box.r); }, sy = function(v){ return box.b - (v - y0) / (y1 - y0) * (box.b - box.t); };
  for (var base = 10; base <= hi; base *= 10) for (var m = 1; m <= 9; m++) { var tn = m * base; if (tn < lo || tn > hi * 1.001) continue; var major = m === 1; x.strokeStyle = major ? 'rgba(52,55,63,.28)' : 'rgba(52,55,63,.09)'; x.lineWidth = major ? 1.3 : 1; x.beginPath(); x.moveTo(sx(tn), box.t); x.lineTo(sx(tn), box.b + (major ? 6 : 3)); x.stroke(); if (major) { x.fillStyle = C.char; x.font = '700 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText(tn.toLocaleString(), sx(tn), box.b + 20); } else if (m === 2 || m === 5) { x.fillStyle = C.mid; x.font = '11px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText(tn.toLocaleString(), sx(tn), box.b + 17); } }
  x.lineWidth = 1; x.font = '12px Aptos, Segoe UI, sans-serif';
  x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText('labelled compounds used for learning (log scale: each bold line is 10 times the one before)', (box.l + box.r) / 2, g.h - 6); x.save(); x.translate(12, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('R2 on unseen compounds', 0, 0); x.restore(); x.font = '12px Aptos, Segoe UI, sans-serif';
  if (marker) { var mx = sx(clamp(marker, lo, hi)); x.strokeStyle = C.mid; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(mx, box.t); x.lineTo(mx, box.b); x.stroke(); x.setLineDash([]); }
  series.forEach(function(s){ if (!s.pts.length) return; x.strokeStyle = s.color; x.lineWidth = s.w || 2.8; x.setLineDash(s.dash || []); x.beginPath(); s.pts.forEach(function(p, i){ i ? x.lineTo(sx(p[0]), sy(p[1])) : x.moveTo(sx(p[0]), sy(p[1])); }); x.stroke(); x.setLineDash([]); x.fillStyle = s.color; s.pts.forEach(function(p){ x.beginPath(); x.arc(sx(p[0]), sy(p[1]), 3.4, 0, 6.3); x.fill(); }); });
  if (marker && o && o.fade) { var fx = sx(clamp(marker, lo, hi)); x.fillStyle = 'rgba(255,255,255,.62)'; x.fillRect(fx, box.t, box.r - fx, box.b - box.t); x.fillStyle = C.mid; x.font = '600 11px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('more labelled compounds than you chose', fx + 8, box.t + 14); }
  if (marker && o && o.dot) series.forEach(function(s){ if (!s.pts.length) return; var m = clamp(marker, s.pts[0][0], s.pts[s.pts.length - 1][0]), v = interp(s.pts, m); x.fillStyle = s.color; x.strokeStyle = '#fff'; x.lineWidth = 2.2; x.beginPath(); x.arc(sx(clamp(marker, lo, hi)), sy(v), 7, 0, 6.3); x.fill(); x.stroke(); x.lineWidth = 1; });
  g.sx = sx; g.sy = sy; g.box = box; return g;
}
function interp(pts, n){ if (!pts.length) return NaN; if (n <= pts[0][0]) return pts[0][1]; for (var i = 1; i < pts.length; i++) if (n <= pts[i][0]) { var t = (Math.log(n) - Math.log(pts[i - 1][0])) / (Math.log(pts[i][0]) - Math.log(pts[i - 1][0])); return lerp(pts[i - 1][1], pts[i][1], t); } return pts[pts.length - 1][1]; }

/* ---------- step 1: why pre-train ---------- */
function pScratch(){
  var feats = activeFeats(); if (!feats.length) return null; D.cache.pc = D.cache.pc || {}; var key = EP + '|' + feats.join(',') + '|' + filtKey(); if (D.cache.pc[key]) return D.cache.pc[key]; var pool = activeTrain();
  var sizes = [25, 50, 100, 200, 400, 800, 1600, pool.length].filter(function(n, i, a){ return n <= pool.length && a.indexOf(n) === i; }), pts = sizes.map(function(n){ var rs = []; for (var rep = 0; rep < 3; rep++) { var r = rng(7 + rep), sh = pool.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; } var m = fitBoost(feats, sh.slice(0, n), 60, 0.1, 3); rs.push(r2(D.TEST, m.F)); } return [n, mean(rs)]; });
  return (D.cache.pc[key] = pts);
}
L6.add({ short: 'Why pre-train', uses: ['endpoint', 'features', 'plabels'],
  title: 'Labelled data is scarce, unlabelled molecules are plentiful',
  watch: 'Labelled compounds, meaning compounds with a lab value, are costly to measure. Molecules without lab values are cheap and plentiful. A model trained from scratch needs many labelled compounds before it scores well, and the curve shows the slow start. Pre-training teaches a model general chemistry from unlabelled molecules first, so it needs fewer labels later.',
  change: 'Drag Labelled compounds below the plot. The dashed line moves along the curve.',
  build: function(api){ mirrorSlider(api, 'plabels', 'Labelled compounds');
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Learning from scratch: score against labelled compounds</h4><canvas width="576" height="350"></canvas>');
    var sp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<div class="ps" style="font-size:13px;line-height:1.45"></div>');
    function bar(label, n, mx, col){ var w = Math.max(1.5, Math.log10(n) / Math.log10(mx) * 100); return '<div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><span>' + label + '</span><b>' + n.toLocaleString() + '</b></div><div class="pbar" style="margin:3px 0 0;height:12px"><i style="background:' + col + ';width:' + w + '%;transition:none"></i></div></div>'; }
    async function render(){
      if (!activeFeats().length) { cv(qs('canvas', cp)); qs('.ps', sp).innerHTML = '<span class="note">Switch on at least one feature.</span>'; return; }
      var key = EP + '|' + activeFeats().join(',') + '|' + filtKey(), cached = !!(D.cache.pc && D.cache.pc[key]); if (!cached) { api.busy('Fitting models on growing subsets'); await waitRelease(); await tick(); } var pts = pScratch(); api.idle(); var n = +$('plabels').value;
      curveChart(qs('canvas', cp), [{ pts: pts, color: C.char }], n, { dot: true, fade: true }); var at = interp(pts, n);
      qs('.ps', sp).innerHTML = '<div class="lbl">At ' + n + ' labelled compounds</div><div class="big" style="font-size:30px;color:' + C.red + '">' + at.toFixed(2) + '</div><div class="note" style="font-size:12px">unseen R2, learning from scratch</div><p style="margin-top:12px">With all ' + activeTrain().length.toLocaleString() + ' labelled training compounds the score is <b>' + pts[pts.length - 1][1].toFixed(2) + '</b>. With only ' + pts[0][0] + ' it is <b>' + pts[0][1].toFixed(2) + '</b>, because the model has seen too few examples of chemistry.</p><div class="lbl" style="margin-top:12px">Scale, on a log axis</div>' + bar('Labelled compounds you chose', n, 1e7, C.char) + bar('Molecules GROVER pre-trained on, no labels', 1e7, 1e7, C.red) + '<p class="note" style="font-size:12px">GROVER: Rong et al., NeurIPS 2020.</p>';
    }
    return { render: render };
  } });

/* ---------- step 2: fill in the blank with atoms ---------- */
var ELNAMES = { C: 'carbon', N: 'nitrogen', O: 'oxygen', S: 'sulfur', F: 'fluorine', Cl: 'chlorine', Br: 'bromine' };
function bondWord(o){ return o === 1 ? 'single' : (o === 2 ? 'double' : (o === 3 ? 'triple' : 'aromatic')); }
L6.add({ short: 'Learning without labels', uses: [],
  title: 'Pre-training starts by hiding part of a molecule and guessing it',
  watch: 'Pre-training needs no lab values. It hides part of a molecule and asks the model to guess the hidden part, and the molecule itself supplies the answer. This is called self-supervised learning. Here one atom is hidden. Use its neighbours and bonds as clues, then pick the element.',
  change: 'Pick an element. Press Hide another atom to play again.',
  build: function(api){
    var gp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<svg viewBox="0 0 580 372" width="580" height="372"></svg>');
    var sp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<div class="mk" style="font-size:13px;line-height:1.45"></div>');
    var ctl = api.addControl('Molecule <select></select> <button type="button" class="btn sm" data-hide>Hide another atom</button> <span class="tally note"></span>'), msel = qs('select', ctl), GR = [], cur = { g: 0, m: -1, pick: null }, tally = { ok: 0, n: 0 };
    function layout(g){ var xs = g.atoms.map(function(a){ return a[1]; }), ys = g.atoms.map(function(a){ return a[2]; }), x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys), sc = Math.min(480 / ((x1 - x0) || 1), 320 / ((y1 - y0) || 1)), ox = 290 - (x1 - x0) * sc / 2, oy = 186 - (y1 - y0) * sc / 2; return g.atoms.map(function(a){ return [ox + (a[1] - x0) * sc, oy + (a[2] - y0) * sc]; }); }
    function nbrs(g, i){ var out = []; g.bonds.forEach(function(b){ if (b[0] === i) out.push([b[1], b[2]]); else if (b[1] === i) out.push([b[0], b[2]]); }); return out; }
    function pickAtom(){ var g = GR[cur.g], c = []; g.atoms.forEach(function(a, i){ if (nbrs(g, i).length) c.push(i); }); cur.m = c[Math.floor(Math.random() * c.length)]; cur.pick = null; }
    function draw(){
      var g = GR[cur.g], pos = layout(g), s = '', nb = nbrs(g, cur.m).map(function(q){ return q[0]; }), done = cur.pick !== null;
      g.bonds.forEach(function(b){ var hot = (b[0] === cur.m || b[1] === cur.m); s += '<line x1="' + pos[b[0]][0] + '" y1="' + pos[b[0]][1] + '" x2="' + pos[b[1]][0] + '" y2="' + pos[b[1]][1] + '" stroke="' + (hot ? C.red : C.bg) + '" stroke-width="' + (hot ? 3.4 : 2) + '"/>'; });
      g.atoms.forEach(function(a, i){ var m = i === cur.m, isN = nb.indexOf(i) >= 0, ok = done && cur.pick === a[0]; s += '<circle cx="' + pos[i][0] + '" cy="' + pos[i][1] + '" r="' + (m ? 17 : 12) + '" fill="' + (m ? (done ? (ok ? '#450A0A' : C.dred) : C.char) : '#fff') + '" stroke="' + (isN ? C.red : C.char) + '" stroke-width="' + (isN ? 3 : 1.6) + '"/><text x="' + pos[i][0] + '" y="' + (pos[i][1] + 5) + '" text-anchor="middle" font-size="' + (m ? 15 : 11.5) + '" font-weight="700" fill="' + (m ? '#fff' : C.char) + '">' + (m ? (done ? a[0] : '?') : (a[0] === 'C' ? '' : a[0])) + '</text>'; });
      qs('svg', gp).innerHTML = s; var ns = nbrs(g, cur.m), tot = ns.reduce(function(t, q){ return t + (q[1] === 1 || q[1] === 2 || q[1] === 3 ? q[1] : 1.5); }, 0), truth = g.atoms[cur.m][0];
      var clue = '<div class="lbl">The clue: what surrounds the hidden atom</div><ul style="margin:6px 0 8px;padding-left:18px">' + ns.map(function(q){ return '<li>' + (ELNAMES[g.atoms[q[0]][0]] || g.atoms[q[0]][0]) + ', ' + bondWord(q[1]) + ' bond</li>'; }).join('') + '</ul><p style="margin:0 0 10px">Its bonds to these neighbours add up to <b>' + tot + '</b>. Hydrogens are not drawn, so there may be more.</p>';
      var opts = Object.keys(ELNAMES).map(function(e){ return '<button type="button" class="btn sm ghost" data-el="' + e + '" style="margin:0 5px 5px 0"' + (done ? ' disabled' : '') + '>' + e + '</button>'; }).join('');
      qs('.mk', sp).innerHTML = clue + (done ? '<div style="padding:8px 10px;border-radius:6px;background:' + (cur.pick === truth ? '#EDF2F4' : '#EDF2F4') + '"><b>' + (cur.pick === truth ? 'Correct.' : 'Not quite.') + '</b> The hidden atom is ' + ELNAMES[truth] + ' (' + truth + ').</div><p style="margin-top:10px;font-size:12.5px">Nobody measured anything. The answer came from the molecule. A pre-trained model plays this game on millions of molecules until it guesses well, and in doing so it learns chemistry.</p>' : '<div class="lbl" style="margin-bottom:6px">Which element is it?</div>' + opts);
      qsa('[data-el]', sp).forEach(function(b){ b.addEventListener('click', function(){ cur.pick = b.dataset.el; tally.n++; if (cur.pick === truth) tally.ok++; draw(); }); }); qs('.tally', ctl).textContent = tally.n ? 'Right ' + tally.ok + ' of ' + tally.n : '';
    }
    msel.addEventListener('change', function(){ cur.g = +msel.value; pickAtom(); draw(); }); qs('[data-hide]', ctl).addEventListener('click', function(){ pickAtom(); draw(); });
    async function render(){ var ex = await loadExtras(); GR = ex.graphs; if (!msel.options.length) { GR.forEach(function(g, i){ var o = document.createElement('option'); o.value = i; o.textContent = g.smi.length > 26 ? g.smi.slice(0, 26) + '...' : g.smi; msel.appendChild(o); }); pickAtom(); } draw(); }
    return { render: render };
  } });

/* ---------- step 3: which chemical groups are present ---------- */
var MOTIFS = [['Amide', 'C(=O)N'], ['Carboxylic acid', 'C(=O)[OX2H1]'], ['Ester', 'C(=O)O[#6]'], ['Benzene ring', 'c1ccccc1'], ['Amine (not an amide)', '[NX3;!$(NC=[O,S]);!$(N-a);!$(NS=O)]'], ['Halogen', '[F,Cl,Br,I]'], ['Sulfonamide', 'S(=O)(=O)N'], ['Hydroxyl', '[OX2H][#6]'], ['Nitrile', 'C#N']];
function hlSvg(smi, atoms, bonds, w, h){ var m = RDK.get_mol(smi); if (!m || !m.is_valid()) { if (m) m.delete(); return ''; } var o = Object.assign({}, RDK_DRAW, { width: w, height: h, atoms: atoms || [], bonds: bonds || [], highlightColour: [0.88, 0.14, 0.16] }), svg = m.get_svg_with_highlights(JSON.stringify(o)); m.delete(); return svg.replace(/<\?xml[^>]*\?>/, '').replace(/width='\d+px' height='\d+px'/, "width='100%' height='100%'"); }
function motifMatch(smi, smarts){ var m = RDK.get_mol(smi), q = RDK.get_qmol(smarts), r = null; try { var j = JSON.parse(m.get_substruct_match(q)); if (j && j.atoms && j.atoms.length) r = j; } catch (e) { r = null; } if (m) m.delete(); if (q) q.delete(); return r; }
L6.add({ short: 'Which groups are present', uses: ['endpoint'],
  title: 'Which chemical groups does the molecule contain',
  watch: 'A functional group is a recognisable piece of a molecule, such as an amide or a carboxylic acid. Software can find groups from the structure alone, so no experiment is needed. Asking a model which groups a molecule contains is a second pre-training task. The model can only answer it by learning what each group looks like.',
  change: 'Click a group marked present to highlight it.',
  build: function(api){
    var sp = api.el('pn', 'left:28px;top:28px;width:440px;height:408px', '<h4>Structure</h4><div class="struct" style="height:350px"></div>');
    var lp = api.el('pn', 'left:488px;top:28px;width:444px;height:408px', '<h4>Groups the pre-training task asks about</h4><div class="ml"></div>');
    var ctl = api.addControl('Molecule <select></select> <button type="button" class="btn sm" data-load>Load RDKit (about 7 MB)</button>'), sel = qs('select', ctl), btn = qs('[data-load]', ctl), cur = null, hit = -1, found = null;
    function show(){ if (!cur) return; var ml = qs('.ml', lp);
      if (!window.RDK) { qs('.struct', sp).innerHTML = cur.svg; ml.innerHTML = '<p class="note">Press Load RDKit to find the groups in this molecule. RDKit runs in your browser.</p>'; btn.style.display = ''; return; }
      btn.style.display = 'none'; found = MOTIFS.map(function(m){ return motifMatch(cur.smi, m[1]); }); var r = found[hit];
      qs('.struct', sp).innerHTML = hlSvg(cur.smi, r ? r.atoms : [], r ? r.bonds : [], 420, 350);
      ml.innerHTML = MOTIFS.map(function(m, i){ var on = !!found[i]; return '<div data-i="' + i + '" style="display:flex;justify-content:space-between;align-items:center;padding:7px 10px;margin-bottom:6px;border-radius:6px;border:1px solid ' + (hit === i ? C.red : C.silver) + ';background:' + (on ? '#fff' : '#EDF2F4') + ';color:' + (on ? C.char : C.mid) + ';cursor:' + (on ? 'pointer' : 'default') + '"><span style="font-weight:600">' + m[0] + '</span><b style="color:' + (on ? C.red : C.mid) + '">' + (on ? 'present' : 'absent') + '</b></div>'; }).join('') + '<p class="note" style="font-size:12px;margin-top:8px">A pre-trained model is asked to predict this list from the molecule, so it has to learn what these groups look like.</p>';
      qsa('[data-i]', ml).forEach(function(d){ d.addEventListener('click', function(){ var i = +d.dataset.i; if (!found[i]) return; hit = hit === i ? -1 : i; show(); }); }); }
    sel.addEventListener('change', function(){ cur = STRUCTS[EP][+sel.value]; hit = -1; show(); });
    btn.addEventListener('click', async function(){ btn.disabled = true; btn.textContent = 'Loading'; try { await loadRDKit(); } catch (e) { btn.textContent = 'RDKit did not load'; return; } btn.disabled = false; show(); });
    async function render(){ await loadStructs(); var list = STRUCTS[EP].slice(0, 30); if (sel.dataset.ep !== EP) { sel.dataset.ep = EP; sel.innerHTML = list.map(function(s, i){ return '<option value="' + i + '">' + esc(s.smi.length > 28 ? s.smi.slice(0, 28) + '...' : s.smi) + '</option>'; }).join(''); cur = list[0]; hit = -1; } show(); }
    return { render: render };
  } });

/* ---------- step 4: architecture and scale ---------- */
L6.add({ short: 'The model and its size', uses: [],
  title: 'GROVER reads atoms and bonds',
  watch: 'GROVER is a published pre-trained model. It reads a molecule as a graph and turns it into one list of numbers that describes it. A small final layer called the head turns that list into a prediction such as clearance. GROVER has about 100 million weights and was pre-trained on about 10 million molecules, far too much to train in a browser. The bars compare it with the small GPT you can train in this course.',
  change: 'Nothing to drag here. Compare the lengths of the bars.',
  build: function(api){
    var dp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<svg viewBox="0 0 580 372" width="580" height="372"></svg>');
    var sp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<div class="pz" style="font-size:13px;line-height:1.45"></div>');
    function box(x, y, w, h, t, sub, hot){ return '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="7" fill="' + (hot ? '#EDF2F4' : '#fff') + '" stroke="' + (hot ? C.red : C.silver) + '" stroke-width="' + (hot ? 2 : 1.2) + '"/><text x="' + (x + 12) + '" y="' + (y + 21) + '" font-size="13" font-weight="700" fill="' + C.char + '">' + t + '</text>' + (sub ? '<text x="' + (x + 12) + '" y="' + (y + 38) + '" font-size="11" fill="' + C.mid + '">' + sub + '</text>' : ''); }
    function arrow(x1, y1, x2, y2){ return '<path d="M' + x1 + ' ' + y1 + 'L' + x2 + ' ' + y2 + '" stroke="' + C.red + '" stroke-width="2.2" fill="none"/><path d="M' + (x2 - 5) + ' ' + (y2 - 7) + 'L' + x2 + ' ' + y2 + 'L' + (x2 + 5) + ' ' + (y2 - 7) + '" stroke="' + C.red + '" stroke-width="2.2" fill="none"/>'; }
    async function render(){
      var ex = await loadExtras(), g = ex.graphs[0], xs = g.atoms.map(function(a){ return a[1]; }), ys = g.atoms.map(function(a){ return a[2]; }), x0 = Math.min.apply(null, xs), y0 = Math.min.apply(null, ys), sc = Math.min(120 / ((Math.max.apply(null, xs) - x0) || 1), 80 / ((Math.max.apply(null, ys) - y0) || 1)), pos = g.atoms.map(function(a){ return [30 + (a[1] - x0) * sc, 96 + (a[2] - y0) * sc]; }), s = '';
      s += '<text x="10" y="22" font-size="12" font-weight="700" fill="' + C.char + '">1  The molecule as a graph</text>'; g.bonds.forEach(function(b){ s += '<line x1="' + pos[b[0]][0] + '" y1="' + pos[b[0]][1] + '" x2="' + pos[b[1]][0] + '" y2="' + pos[b[1]][1] + '" stroke="' + C.bg + '" stroke-width="2"/>'; }); g.atoms.forEach(function(a, i){ s += '<circle cx="' + pos[i][0] + '" cy="' + pos[i][1] + '" r="6" fill="' + (a[0] === 'C' ? '#fff' : C.red) + '" stroke="' + C.char + '" stroke-width="1.4"/>'; });
      s += arrow(160, 130, 200, 130) + '<text x="205" y="22" font-size="12" font-weight="700" fill="' + C.char + '">2  Graph transformer, many layers</text>';
      for (var l = 0; l < 3; l++) s += box(205, 34 + l * 74, 190, 62, 'Layer ' + (l + 1), l === 0 ? 'atoms pass messages along bonds' : (l === 1 ? 'every atom attends to every atom' : 'repeated many more times'), l === 1);
      s += arrow(395, 130, 430, 130) + '<text x="435" y="22" font-size="12" font-weight="700" fill="' + C.char + '">3  One vector</text>'; for (var k = 0; k < 14; k++) s += '<rect x="440" y="' + (36 + k * 9.6) + '" width="120" height="7" rx="2" fill="' + (k % 3 ? C.bg : C.red) + '" opacity="' + (0.35 + 0.65 * ((k * 37 % 11) / 11)).toFixed(2) + '"/>';
      s += arrow(500, 176, 500, 236) + box(405, 240, 170, 54, 'Pre-training task', 'hide atoms, name groups', true) + '<text x="10" y="262" font-size="12" fill="' + C.mid + '">No lab values needed for this part.</text>';
      s += arrow(500, 296, 500, 318) + box(405, 322, 170, 44, 'Fine-tuning head', 'predicts HLM, logD, ...', false);
      qs('svg', dp).innerHTML = s;
      var gp = window.G && G.params ? G.params : 108032, gs = window.G && G.seqs ? G.seqs.length : 1335;
      function bar(label, n, col){ var w = Math.max(1.5, Math.log10(n) / 8 * 100); return '<div style="margin:7px 0"><div style="display:flex;justify-content:space-between"><span>' + label + '</span><b>' + n.toLocaleString() + '</b></div><div class="pbar" style="margin:3px 0 0;height:12px"><i style="background:' + col + ';width:' + w + '%;transition:none"></i></div></div>'; }
      qs('.pz', sp).innerHTML = '<div class="lbl">Parameters (numbers the model learns)</div>' + bar('GROVER, about', 1e8, C.red) + bar('The GPT in this course', gp, C.char) + '<div class="lbl" style="margin-top:12px">Molecules used for pre-training</div>' + bar('GROVER', 1e7, C.red) + bar('The GPT in this course', gs, C.char) + '<p class="note" style="font-size:12px;margin-top:10px">Bars use a log scale. GROVER numbers are from Rong et al., NeurIPS 2020: about 100 million parameters and 10 million unlabelled molecules. Nothing on this page runs GROVER.</p>';
    }
    return { render: render };
  } });

/* ---------- step 5: pre-train then fine-tune, in miniature ---------- */
function ridgeFit(X, y, d, lam){
  var n = y.length, mu = new Float64Array(d), sd = new Float64Array(d), ym = mean(Array.from(y)); for (var j = 0; j < d; j++) { var s = 0; for (var i = 0; i < n; i++) s += X[i * d + j]; mu[j] = s / n; var v = 0; for (var i2 = 0; i2 < n; i2++) { var e = X[i2 * d + j] - mu[j]; v += e * e; } sd[j] = Math.sqrt(v / n) || 1; }
  var A = []; for (var a = 0; a < d; a++) { A.push(new Float64Array(d + 1)); A[a][a] = lam; } var b = new Float64Array(d), z = new Float64Array(d);
  for (var r = 0; r < n; r++) { for (var q = 0; q < d; q++) z[q] = (X[r * d + q] - mu[q]) / sd[q]; var yr = y[r] - ym; for (var p = 0; p < d; p++) { b[p] += z[p] * yr; for (var c = p; c < d; c++) A[p][c] += z[p] * z[c]; } }
  for (var p2 = 0; p2 < d; p2++) for (var c2 = 0; c2 < p2; c2++) A[p2][c2] = A[c2][p2]; for (var t = 0; t < d; t++) A[t][d] = b[t];
  for (var col = 0; col < d; col++) { var piv = col; for (var rr = col + 1; rr < d; rr++) if (Math.abs(A[rr][col]) > Math.abs(A[piv][col])) piv = rr; var tmp = A[col]; A[col] = A[piv]; A[piv] = tmp; var dv = A[col][col] || 1e-9; for (var cc = col; cc <= d; cc++) A[col][cc] /= dv; for (var r3 = 0; r3 < d; r3++) if (r3 !== col) { var f = A[r3][col]; if (f) for (var c3 = col; c3 <= d; c3++) A[r3][c3] -= f * A[col][c3]; } }
  var w = new Float64Array(d); for (var k = 0; k < d; k++) w[k] = A[k][d]; return { w: w, mu: mu, sd: sd, ym: ym, d: d };
}
function ridgePredict(m, X, i){ var s = m.ym, d = m.d; for (var j = 0; j < d; j++) s += m.w[j] * (X[i * d + j] - m.mu[j]) / m.sd[j]; return s; }
async function gptEmbed(d, PP, nl){
  gptSetup(); var rows = [], seqs = [], lens = [];
  for (var i = 0; i < d.N; i++) { var t = d.smi_all[i].match(TOK) || []; if (t.length > TLEN - 2) continue; var ids = [1], ok = true; for (var k = 0; k < t.length; k++) { var id = G.stoi[t[k]]; if (id === undefined) { ok = false; break; } ids.push(id); } if (!ok) continue; var L = ids.length; while (ids.length < TLEN) ids.push(0); rows.push(i); seqs.push(ids); lens.push(L); }
  var E = new Float32Array(rows.length * DM);
  for (var b0 = 0; b0 < seqs.length; b0 += 96) { var nb = Math.min(96, seqs.length - b0), out = tf.tidy(function(){ var idt = tf.tensor2d(seqs.slice(b0, b0 + nb), null, 'int32'), h = gforward(idt, nl, false, PP, true), m = tf.tensor2d(lens.slice(b0, b0 + nb).map(function(L2){ return Array.from({ length: TLEN }, function(_, q){ return q < L2 ? 1 : 0; }); })).expandDims(-1); return h.mul(m).sum(1).div(m.sum(1)).dataSync(); }); E.set(out, b0 * DM); await tick(); }
  return { rows: rows, E: E };
}
var PRE = { key: null, pre: null, rnd: null, curve: null, ckey: null };
async function preEmbeddings(){
  var key = JSON.stringify([EP, G.sig, G.params]); if (PRE.key === key && PRE.pre) return PRE; var nl = G.NL, PP = gMake(nl);
  try { PRE.pre = await gptEmbed(D, null, nl); PRE.rnd = await gptEmbed(D, PP, nl); } finally { Object.keys(PP).forEach(function(k){ PP[k].dispose(); }); } PRE.key = key; PRE.curve = null; return PRE;
}
function preCurve(){
  var feats = activeFeats(), key = JSON.stringify([PRE.key, feats, filtKey()]); if (PRE.ckey === key && PRE.curve) return PRE.curve;
  var rows = PRE.pre.rows, trn = [], tst = [], inSet = new Set(activeTrain()); rows.forEach(function(row, p){ if (D.test[row]) tst.push(p); else if (inSet.has(row)) trn.push(p); }); var tstRows = tst.map(function(p){ return rows[p]; }), sizes = [25, 50, 100, 200, 400, 800, 1600, trn.length].filter(function(n, i, a){ return n <= trn.length && a.indexOf(n) === i; }), res = { sizes: sizes, pre: [], rnd: [], scr: [], nTrain: trn.length, nTest: tst.length };
  sizes.forEach(function(n){ var a = [], b = [], c = []; for (var rep = 0; rep < 3; rep++) { var r = rng(11 + rep), sh = trn.slice(); for (var q = sh.length - 1; q > 0; q--) { var j = Math.floor(r() * (q + 1)), t = sh[q]; sh[q] = sh[j]; sh[j] = t; } var sub = sh.slice(0, n), y = Float64Array.from(sub.map(function(p){ return D.Y[rows[p]]; }));
      [['pre', PRE.pre.E, a], ['rnd', PRE.rnd.E, b]].forEach(function(cfg){ var X = new Float32Array(sub.length * DM); sub.forEach(function(p, k){ X.set(cfg[1].subarray(p * DM, p * DM + DM), k * DM); }); var m = ridgeFit(X, y, DM, 30), pred = new Float64Array(D.N); tst.forEach(function(p){ pred[rows[p]] = ridgePredict(m, cfg[1], p); }); cfg[2].push(r2(tstRows, pred)); });
      if (feats.length) { var bm = fitBoost(feats, sub.map(function(p){ return rows[p]; }), 60, 0.1, 3); c.push(r2(tstRows, bm.F)); } }
    res.pre.push([n, mean(a)]); res.rnd.push([n, mean(b)]); res.scr.push([n, c.length ? mean(c) : 0]); });
  PRE.curve = res; PRE.ckey = key; return res;
}
L6.add({ short: 'Fixed features', uses: ['endpoint', 'features', 'filters', 'plabels'],
  title: 'Use the pre-trained GPT as it is',
  watch: 'A pre-trained model can be used as a fixed feature maker. The GPT turns each molecule into 64 numbers, and a simple line fit learns the lab values from them. Red is the trained GPT. Grey is the same GPT with random weights, which shows what the 64 numbers give without pre-training. Dark is ordinary features such as logP. The gap between red and grey is what pre-training added.',
  change: 'Drag Labelled compounds below the plot. Press Recompute after retraining the GPT.',
  build: function(api){ mirrorSlider(api, 'plabels', 'Labelled compounds');
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Score against labelled compounds, unseen compounds</h4><canvas width="576" height="350"></canvas>');
    var sp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<div class="pm" style="font-size:13px;line-height:1.45"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-re>Recompute</button> <span class="note" style="font-size:12px">Uses the GPT as it is trained now.</span>'), busy = false;
    async function render(force){
      if (busy) return; if (!(await gReady(api))) { cv(qs('canvas', cp)); qs('.pm', sp).innerHTML = ''; return; } busy = true;
      try { if (force) PRE.key = null; api.busy('Turning molecules into numbers with the GPT'); await tick(); await preEmbeddings(); api.busy('Fitting models on growing subsets'); await tick(); var cur = preCurve(); api.idle(); var n = +$('plabels').value, a = interp(cur.pre, n), b = interp(cur.rnd, n), c = interp(cur.scr, n);
        curveChart(qs('canvas', cp), [{ pts: cur.scr, color: C.char, w: 2.4 }, { pts: cur.rnd, color: C.bg, dash: [6, 5] }, { pts: cur.pre, color: C.red }], n, { dot: true, fade: true });
        qs('.pm', sp).innerHTML = '<div class="lbl">At ' + n + ' labelled compounds</div><div class="nums" style="font-size:13px;margin-top:4px"><div><span style="color:' + C.red + '">Trained GPT</span><b>' + a.toFixed(2) + '</b></div><div><span>Random-weight GPT</span><b>' + b.toFixed(2) + '</b></div><div><span>Descriptors only</span><b>' + c.toFixed(2) + '</b></div></div><p style="margin-top:12px">' + (a > b + 0.01 ? 'Pre-training helped: ' + (a - b).toFixed(2) + ' higher than random weights.' : 'No gain yet. Train the GPT for more steps, then press Recompute.') + '</p><p class="note" style="font-size:12px">' + cur.nTrain + ' labelled and ' + cur.nTest + ' unseen compounds. Mean of three draws.</p>'; }
      catch (e) { console.error(e); qs('.pm', sp).innerHTML = '<span style="color:' + C.red + '">' + esc(e.message) + '</span>'; }
      api.idle(); busy = false;
    }
    qs('[data-re]', ctl).addEventListener('click', function(){ render(true); });
    return { render: function(){ return render(false); } };
  } });

/* ---------- fine-tuning: start from the pre-trained GPT and teach it the labels ---------- */
var FT = { mode: 'head', tok: null, scr: {}, last: null };
function gptTok(d){
  if (FT.tok && FT.tok.ep === EP) return FT.tok; gptSetup(); var rows = [], seqs = [], lens = [], pos = {};
  for (var i = 0; i < d.N; i++) { var t = d.smi_all[i].match(TOK) || []; if (t.length > TLEN - 2) continue; var ids = [1], ok = true; for (var k = 0; k < t.length; k++) { var id = G.stoi[t[k]]; if (id === undefined) { ok = false; break; } ids.push(id); } if (!ok) continue; var L = ids.length; while (ids.length < TLEN) ids.push(0); pos[i] = rows.length; rows.push(i); seqs.push(ids); lens.push(L); }
  return (FT.tok = { ep: EP, rows: rows, seqs: seqs, lens: lens, pos: pos });
}
function gClone(P){ var Q = {}; Object.keys(P).forEach(function(k){ Q[k] = tf.variable(P[k].clone()); }); return Q; }
function gFree(P){ Object.keys(P).forEach(function(k){ P[k].dispose(); }); }
function maskT(lens){ return tf.tensor2d(lens.map(function(L){ return Array.from({ length: TLEN }, function(_, q){ return q < L ? 1 : 0; }); })).expandDims(-1); }
function poolRows(T, idx, PP, nl){ var idt = tf.tensor2d(idx.map(function(i){ return T.seqs[i]; }), null, 'int32'), h = gforward(idt, nl, false, PP, true), m = maskT(idx.map(function(i){ return T.lens[i]; })); return h.mul(m).sum(1).div(m.sum(1)); }
async function embedIdx(T, idx, PP, nl){ var E = new Float32Array(idx.length * DM); for (var b = 0; b < idx.length; b += 96) { var part = idx.slice(b, b + 96), out = tf.tidy(function(){ return poolRows(T, part, PP, nl).dataSync(); }); E.set(out, b * DM); await tick(); } return E; }
/* one fine-tuning run. init: 'pre' starts from the trained GPT, 'scratch' starts from random weights. */
async function ftRun(o){
  var T = o.T, nl = G.NL, N = o.tr.length, mode = o.init === 'pre' ? o.mode : 'all', ym = mean(o.tr.map(function(i){ return D.Y[T.rows[i]]; })), ysd = Math.sqrt(mean(o.tr.map(function(i){ var e = D.Y[T.rows[i]] - ym; return e * e; }))) || 1, y = o.tr.map(function(i){ return (D.Y[T.rows[i]] - ym) / ysd; }), rowsTe = o.te.map(function(i){ return T.rows[i]; });
  var P = o.init === 'pre' ? (mode === 'head' ? G.P : gClone(G.P)) : gMake(nl), head = { W: tf.variable(tf.randomNormal([DM, 1], 0, 0.05)), b: tf.variable(tf.zeros([1])) }, vars = (mode === 'head' ? [] : Object.keys(P).map(function(k){ return P[k]; })).concat([head.W, head.b]), opt = tf.train.adam(o.lr), r = rng(33), B = Math.min(16, N), curve = [], t0 = performance.now(), over = 0, mu = null, sd = null, Etr = null, Ete = null, xs = null;
  try {
    if (mode === 'head') { Etr = await embedIdx(T, o.tr, P, nl); Ete = await embedIdx(T, o.te, P, nl); mu = new Float32Array(DM); sd = new Float32Array(DM); for (var j = 0; j < DM; j++) { var s = 0; for (var i = 0; i < N; i++) s += Etr[i * DM + j]; mu[j] = s / N; var v = 0; for (var i2 = 0; i2 < N; i2++) { var e = Etr[i2 * DM + j] - mu[j]; v += e * e; } sd[j] = Math.sqrt(v / N) || 1; } var nrm = function(E, n){ var out = new Float32Array(n * DM); for (var a = 0; a < n; a++) for (var q = 0; q < DM; q++) out[a * DM + q] = (E[a * DM + q] - mu[q]) / sd[q]; return tf.tensor2d(out, [n, DM]); }; xs = { tr: nrm(Etr, N), te: nrm(Ete, o.te.length) }; }
    function evalTe(){ return tf.tidy(function(){ if (mode === 'head') return xs.te.matMul(head.W).add(head.b).reshape([-1]).dataSync(); var out = []; for (var b = 0; b < o.te.length; b += 128) { var part = o.te.slice(b, b + 128), p = poolRows(T, part, P, nl).matMul(head.W).add(head.b).reshape([-1]).arraySync(); out = out.concat(p); } return out; }); }
    function moved(){ if (o.init !== 'pre' || mode === 'head') return 0; return tf.tidy(function(){ var num = 0, den = 0; Object.keys(P).forEach(function(k){ num += P[k].sub(G.P[k]).square().sum().dataSync()[0]; den += G.P[k].square().sum().dataSync()[0]; }); return Math.sqrt(num / (den || 1)); }); }
    var every = Math.max(1, Math.round(o.steps / 12)), run = 0;
    async function record(s, lossVal){ var c0 = performance.now(), p = evalTe(), pred = new Float64Array(D.N); rowsTe.forEach(function(row, k){ pred[row] = p[k] * ysd + ym; }); var pt = { s: s, r2: r2(rowsTe, pred), loss: lossVal, moved: moved() }; curve.push(pt); if (o.onPoint) await o.onPoint(pt, curve); over += performance.now() - c0; return pt; }
    await record(0, null);
    for (var s = 1; s <= o.steps; s++) {
      var idx = []; for (var q = 0; q < B; q++) idx.push(Math.floor(r() * N)); var yb = tf.tensor1d(idx.map(function(k){ return y[k]; })), lv;
      var cost = opt.minimize(function(){ var p = mode === 'head' ? tf.gather(xs.tr, tf.tensor1d(idx, 'int32')).matMul(head.W).add(head.b).reshape([-1]) : poolRows(T, idx.map(function(k){ return o.tr[k]; }), P, nl).matMul(head.W).add(head.b).reshape([-1]); return p.sub(yb).square().mean(); }, true, vars); lv = cost.dataSync()[0]; cost.dispose(); yb.dispose(); run = run * 0.7 + lv * 0.3 * (run ? 1 : 3.33);
      if (s % every === 0 || s === o.steps) await record(s, run || lv); else if (s % 4 === 0) await tick();
    }
    return { curve: curve, r2: curve[curve.length - 1].r2, secs: (performance.now() - t0 - over) / 1000 };
  } finally { opt.dispose(); head.W.dispose(); head.b.dispose(); if (xs) { xs.tr.dispose(); xs.te.dispose(); } if (o.init === 'scratch' || (o.init === 'pre' && mode === 'all')) gFree(P); }
}
L6.add({ short: 'Fine-tune', bench: true, uses: ['endpoint', 'filters', 'plabels', 'ftmode', 'ftlr', 'ftsteps'],
  title: 'Fine-tune the pre-trained GPT on labelled compounds',
  watch: 'Fine-tuning continues training a pre-trained model on your labelled compounds. A new head makes the prediction: one weight for each of the 64 numbers, plus a bias. Head only freezes the GPT, meaning its weights do not change, and trains just the head. Everything lets all the weights change. Grey starts from random weights instead and is the baseline. The learning rate is the size of each weight change.',
  change: 'Choose Head only or Everything in the options panel, set the learning rate and steps, then press Train.',
  build: function(api){
    var bb = benchBar(api);
    var mp = api.el('pn', 'left:28px;top:80px;width:440px;height:372px', '<h4>What is being trained</h4><svg viewBox="0 0 416 330" width="416" height="330"></svg>');
    var cp = api.el('pn', 'left:488px;top:80px;width:444px;height:178px', '<h4>R2 on unseen compounds, by fine-tuning step</h4><canvas width="420" height="140"></canvas>');
    var ep = api.el('pn', 'left:488px;top:270px;width:444px;height:182px', '<h4>Error on the labelled compounds (standardised)</h4><canvas width="420" height="144"></canvas>');
    var mn = api.el('pn', 'left:28px;top:462px;width:904px;height:56px;display:flex;align-items:center;font-size:14px;line-height:1.4', '<div class="mn"></div>');
    var svg = qs('svg', mp), running = false, cur = null, scr = null, pre = null, movedNow = 0;
    function params(){ return { mode: FT.mode, lr: +$('ftlr').value, steps: +$('ftsteps').value, n: +$('plabels').value }; }
    function setup(){ var T = gptTok(D), inSet = new Set(activeTrain()), tr = [], te = []; T.rows.forEach(function(row, p){ if (D.test[row]) te.push(p); else if (inSet.has(row)) tr.push(p); }); var r = rng(5); for (var a = tr.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = tr[a]; tr[a] = tr[j]; tr[j] = t; } return { T: T, tr: tr, te: te }; }
    function box(x, y, w, h, title, sub, state){ var col = state === 'frozen' ? C.mid : (state === 'update' ? C.red : C.char); return '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="7" fill="' + (state === 'update' ? '#EDF2F4' : (state === 'frozen' ? '#EDF2F4' : '#fff')) + '" stroke="' + col + '" stroke-width="' + (state ? 2 : 1.2) + '"/><text x="' + (x + 12) + '" y="' + (y + 22) + '" font-size="13" font-weight="700" fill="' + C.char + '">' + title + '</text><text x="' + (x + 12) + '" y="' + (y + 39) + '" font-size="11" fill="' + C.mid + '">' + sub + '</text>' + (state ? '<text x="' + (x + w - 10) + '" y="' + (y + 22) + '" font-size="11" font-weight="700" text-anchor="end" fill="' + col + '">' + (state === 'frozen' ? 'FROZEN' : 'UPDATING') + '</text>' : ''); }
    function arrow(y1, y2){ return '<path d="M208 ' + y1 + 'L208 ' + y2 + '" stroke="' + C.red + '" stroke-width="2" fill="none"/><path d="M203 ' + (y2 - 6) + 'L208 ' + y2 + 'L213 ' + (y2 - 6) + '" stroke="' + C.red + '" stroke-width="2" fill="none"/>'; }
    function schematic(){ var p = params(), frozen = p.mode === 'head', n = Math.min(p.n, cur ? cur.tr.length : p.n), s = '';
      s += box(8, 4, 400, 44, 'SMILES tokens of a labelled compound', n + ' labelled compounds (after your filters)', '') + arrow(50, 62) + box(8, 64, 400, 92, 'Pre-trained GPT', G.NL + ' layers, ' + (G.params || 0).toLocaleString() + ' weights', frozen ? 'frozen' : 'update');
      var bw = Math.min(380, movedNow * 100 * 4); s += '<rect x="20" y="134" width="376" height="10" rx="4" fill="' + C.silver + '"/><rect x="20" y="134" width="' + Math.max(0, bw).toFixed(1) + '" height="10" rx="4" fill="' + C.red + '"/><text x="20" y="128" font-size="10.5" fill="' + C.mid + '">weights moved away from the pre-trained values: ' + (movedNow * 100).toFixed(1) + '%</text>';
      s += arrow(158, 172) + box(8, 174, 400, 44, 'Average over the tokens', 'one vector of ' + DM + ' numbers for the molecule', '') + arrow(220, 232) + box(8, 234, 400, 54, 'Head: ' + (DM + 1) + ' new numbers', 'predicts ' + unitWord(), 'update') + '<text x="8" y="308" font-size="11.5" fill="' + C.mid + '">The head always starts random and always learns.</text><text x="8" y="324" font-size="11.5" fill="' + C.mid + '">The question is whether the GPT underneath should move too.</text>'; svg.innerHTML = s; }
    function charts(){ var p = params(), x1 = p.steps, all = []; [scr, pre].forEach(function(r){ if (r) r.curve.forEach(function(q){ all.push(q.r2); }); }); var y0 = Math.min(0, Math.min.apply(null, all.concat([0]))), y1 = Math.max(0.3, Math.max.apply(null, all.concat([0])) * 1.1), o = { xl: 'fine-tuning step', yl: 'R2', x0: 0, x1: x1, y0: y0, y1: y1, xdec: 0, ydec: 2 };
      var s1 = [], s2 = []; if (scr) { s1.push({ pts: scr.curve.map(function(q){ return [q.s, q.r2]; }), color: C.bg, nodot: true, w: 2.4 }); s2.push({ pts: scr.curve.filter(function(q){ return q.loss !== null; }).map(function(q){ return [q.s, q.loss]; }), color: C.bg, nodot: true, w: 2.4 }); } if (pre) { s1.push({ pts: pre.curve.map(function(q){ return [q.s, q.r2]; }), color: C.red }); s2.push({ pts: pre.curve.filter(function(q){ return q.loss !== null; }).map(function(q){ return [q.s, q.loss]; }), color: C.red }); }
      if (!s1.length) s1.push({ pts: [[0, y0], [x1, y1]], color: 'rgba(0,0,0,0)', nodot: true }); lineChart(qs('canvas', cp), s1, o); var lo2 = { xl: 'fine-tuning step', yl: 'error', x0: 0, x1: x1, y0: 0, xdec: 0, ydec: 2 }; if (!s2.some(function(s){ return s.pts.length; })) s2 = [{ pts: [[0, 0], [x1, 1]], color: 'rgba(0,0,0,0)', nodot: true }]; lineChart(qs('canvas', ep), s2.filter(function(s){ return s.pts.length; }), lo2); }
    function explain(){ var p = params(); if (!pre || !scr) { qs('.mn', mn).textContent = 'Head only cannot damage the GPT. Everything can fit better but may forget what pre-training taught. Press Train to compare.'; return; }
      var d = pre.r2 - scr.r2, fin = pre.curve[pre.curve.length - 1], t = 'R2 ' + pre.r2.toFixed(2) + ', against ' + scr.r2.toFixed(2) + ' from random weights. ' + (d > 0.02 ? 'Pre-training helped by ' + d.toFixed(2) + '.' : (d < -0.02 ? 'Pre-training did worse here.' : 'About equal.')); if (p.mode === 'all') t += ' The GPT\'s weights moved ' + (fin.moved * 100).toFixed(1) + '%.'; qs('.mn', mn).textContent = t; }
    function render(){
      if (running) return; schematic(); charts(); explain(); if (!G.trained) { api.gate('Train the GPT in its workbench first.'); return; } api.idle();
      var s = setup(); cur = s; schematic(); var p = params(); if (!s.tr.length) { bb.say('No labelled compounds left. Loosen the filters.'); return; } bb.say(FT.last && FT.last.key === keyOf() ? 'Unseen R2 ' + fix(FT.last.pre.r2, 2) + ' after ' + p.steps + ' steps.' : 'Ready. Press Train.'); api.stale(FT.last && FT.last.key !== keyOf() ? 'Options changed. Press Train.' : '');
    }
    function keyOf(){ var p = params(); return JSON.stringify([EP, filtKey(), p.mode, p.lr, p.steps, p.n, G.sig, G.params, G.secs]); }
    function scrKey(){ var p = params(); return JSON.stringify([EP, filtKey(), p.lr, p.steps, p.n, G.NL]); }
    async function train(){
      if (running || G.busy) return; if (!G.trained) { api.gate('Train the GPT in its workbench first.'); return; } var s = setup(); cur = s; if (!s.tr.length) { bb.say('No labelled compounds left. Loosen the filters.'); return; }
      running = true; bb.btn.disabled = true; api.tok++; var tok = api.tok, p = params(), n = Math.min(p.n, s.tr.length), tr = s.tr.slice(0, n); api.stale(''); movedNow = 0; pre = null;
      try {
        var sk = scrKey(); scr = FT.scr[sk] || null; if (!scr) { api.busy('Training a GPT from random weights on the same compounds'); await tick(); await tick(); scr = await ftRun({ T: s.T, tr: tr, te: s.te, init: 'scratch', mode: 'all', lr: p.lr, steps: p.steps }); FT.scr[sk] = scr; api.idle(); } api.checkTok(tok); schematic(); charts();
        var per = clamp(14000 * speedMul() / p.steps, 0, 400);
        pre = { curve: [], r2: 0 }; await ftRun({ T: s.T, tr: tr, te: s.te, init: 'pre', mode: p.mode, lr: p.lr, steps: p.steps, onPoint: async function(pt, curve){ api.checkTok(tok); pre = { curve: curve.slice(), r2: pt.r2 }; movedNow = pt.moved; schematic(); charts(); bb.say('Step ' + pt.s + ' of ' + p.steps + '. Unseen R2 ' + fix(pt.r2, 2)); if (per > 2 && pt.s > 0) await api.sleep(per * (p.steps / 12)); } }).then(function(res){ pre = { curve: res.curve, r2: res.r2 }; });
        FT.last = { key: keyOf(), pre: pre, scr: scr };
      } catch (e) { if (!(e instanceof Cancel)) { bb.say('Error: ' + e.message); console.error(e); } }
      api.idle(); running = false; bb.btn.disabled = false; schematic(); charts(); explain(); if (FT.last && FT.last.key === keyOf()) { bb.say('Unseen R2 ' + fix(FT.last.pre.r2, 2) + ' after ' + p.steps + ' steps.'); }
    }
    bb.btn.addEventListener('click', train);
    L6.live = function(){ if (running) api.tok++; else { pre = FT.last && FT.last.key === keyOf() ? FT.last.pre : null; scr = pre ? FT.last.scr : null; render(); } };
    return { render: function(){ if (!running) { pre = FT.last && FT.last.key === keyOf() ? FT.last.pre : null; scr = pre ? FT.last.scr : null; } render(); } };
  } });

/* ---------- fine-tuning Chemprop: results measured in Python ---------- */
L6.add({ short: 'Fine-tune Chemprop', uses: ['plabels'],
  title: 'Fine-tuning Chemprop',
  watch: 'Chemprop was first trained on 4,200 logD values, then adapted to HLM using a few HLM compounds. From scratch means starting with random weights. Frozen means the part that reads the molecule (the encoder) keeps its logD weights and only the final layers learn. Full means every weight can change. The chart shows the HLM score on unseen scaffolds as the number of labelled HLM compounds grows.',
  change: 'Drag Labelled compounds below the plot. Use the boxes to show or hide a line.',
  build: function(api){ mirrorSlider(api, 'plabels', 'Labelled compounds');
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>HLM score on unseen scaffolds, by number of labelled HLM compounds</h4><canvas width="576" height="350"></canvas>');
    var sp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<div class="cf" style="font-size:13px;line-height:1.45"></div>');
    var ctl = api.addControl('<label><input type="checkbox" data-m="scratch" checked> From scratch</label> <label><input type="checkbox" data-m="frozen" checked> Frozen encoder</label> <label><input type="checkbox" data-m="full" checked> Full fine-tuning</label>'), show = { scratch: true, frozen: true, full: true };
    var COL = { scratch: C.bg, frozen: C.char, full: C.red }, NAME = { scratch: 'From scratch', frozen: 'Frozen encoder', full: 'Full fine-tuning' };
    function agg(runs){ var out = {}; ['scratch', 'frozen', 'full'].forEach(function(m){ var by = {}; runs.filter(function(r){ return r.mode === m; }).forEach(function(r){ (by[r.n] = by[r.n] || []).push(r); }); out[m] = Object.keys(by).map(Number).sort(function(a, b){ return a - b; }).map(function(n){ return { n: n, r2: mean(by[n].map(function(r){ return r.r2; })), secs: mean(by[n].map(function(r){ return r.secs; })), k: by[n].length }; }); }); return out; }
    qsa('input[data-m]', ctl).forEach(function(b){ b.addEventListener('change', function(){ show[b.dataset.m] = b.checked; render(); }); });
    async function render(){
      var ex = await loadExtras(), ft = ex.chemprop_ft; if (!ft || !ft.runs || !ft.runs.length) { cv(qs('canvas', cp)); qs('.cf', sp).innerHTML = '<span class="note">No Chemprop results are available.</span>'; return; }
      var A = agg(ft.runs), n = +$('plabels').value, series = ['scratch', 'frozen', 'full'].filter(function(m){ return show[m]; }).map(function(m){ return { pts: A[m].map(function(p){ return [p.n, p.r2]; }), color: COL[m], dash: m === 'scratch' ? [6, 5] : null }; });
      curveChart(qs('canvas', cp), series.length ? series : [{ pts: [], color: C.bg }], n, { dot: true, fade: true });
      var rows = ['scratch', 'frozen', 'full'].map(function(m){ var v = interp(A[m].map(function(p){ return [p.n, p.r2]; }), clamp(n, A[m][0].n, A[m][A[m].length - 1].n)); return '<div><span style="color:' + (m === 'scratch' ? C.mid : COL[m]) + '">' + NAME[m] + '</span><b>' + v.toFixed(2) + '</b></div>'; }).join(''), last = A.scratch[A.scratch.length - 1];
      qs('.cf', sp).innerHTML = '<div class="lbl">At about ' + clamp(n, 50, 1000) + ' labelled compounds</div><div class="nums" style="margin:4px 0 10px">' + rows + '</div><p style="margin:0 0 8px"><b>How to read it.</b> Higher is better. The first point is 50 labelled compounds, where a model from scratch has almost nothing to learn from.</p><p style="margin:0 0 8px">The logD model that all three non-scratch starts came from scored ' + ft.pretrain.r2.toFixed(2) + ' on its own unseen scaffolds, after ' + Math.round(ft.pretrain.secs) + ' seconds of CPU training.</p><p class="note" style="font-size:12px">Each point is the mean of ' + last.k + ' random draws of the labelled compounds, 30 epochs each. Differences of a few hundredths are within noise.</p>';
    }
    return { render: render };
  } });

/* ---------- step 6: using it in a project ---------- */
L6.add({ short: 'Using it in practice', uses: [],
  title: 'How a pre-trained model is used on your own data',
  watch: 'A model like GROVER is trained once by its authors. You take the published weights and add your own labelled data. The four stages below show how, and the cautions list what can go wrong.',
  change: 'Read the four stages.',
  build: function(api){
    var st = [['1  Get the pre-trained model', 'Download the weights and code that the authors published. This is the expensive part, and it is already done.'], ['2  Describe your compounds', 'Run your SMILES through the model to get one vector per compound. These vectors replace, or sit next to, descriptors such as logP.'], ['3  Learn from your labels', 'Fit a small model on the vectors, or fine-tune the network itself. Fewer labels and a gentler learning rate favour keeping the pre-trained weights nearly fixed.'], ['4  Test it', 'Use the same scaffold split and the same unseen compounds as your baseline.']];
    st.forEach(function(s, i){ api.el('pn', 'left:' + (28 + i * 227) + 'px;top:28px;width:214px;height:200px', '<h4 style="color:' + C.red + '">' + s[0] + '</h4><div style="font-size:13px;line-height:1.5;margin-top:4px">' + s[1] + '</div>'); });
    api.el('pn', 'left:28px;top:244px;width:904px;height:192px', '<h4>Cautions</h4><ul style="margin:6px 0 0;padding-left:18px;font-size:13px;line-height:1.55"><li>A bigger pre-trained model does not guarantee a better score on your endpoint. Compare it with a descriptor model on the same split before adopting it.</li><li>The pre-training molecules decide what the model knows. A model that never saw your chemistry may transfer poorly.</li><li>These models need Python and usually a GPU. They do not run in this page.</li><li>The reference is Rong et al., Self-Supervised Graph Transformer on Large-Scale Molecular Data, NeurIPS 2020 (arXiv 2007.02835).</li></ul>');
    return { render: function(){} };
  } });

L6.glossary([
  ['Pre-training', 'Training a large model first on a task that needs no lab values, such as filling in hidden parts of molecules.'],
  ['Fine-tuning', 'Adjusting a pre-trained model, or adding a small model on top, using your own labelled compounds.'],
  ['Self-supervised', 'The labels come from the data itself, for example a hidden atom whose identity is known, so no experiment is needed.'],
  ['Unlabelled molecule', 'A structure with no measured value attached.'],
  ['Molecular graph', 'A molecule drawn as nodes (atoms) joined by edges (bonds).'],
  ['Graph transformer', 'A transformer that lets atoms exchange information along bonds and attend to every other atom.'],
  ['GROVER', 'A graph transformer pre-trained on about 10 million unlabelled molecules, with about 100 million parameters (Rong et al., NeurIPS 2020).'],
  ['Embedding', 'The list of numbers a model uses to describe a token, an atom or a whole molecule.'],
  ['Ridge regression', 'A simple linear model with a penalty that keeps its weights small.'],
  ['Functional group', 'A small, recognisable piece of structure such as an amide or a carboxylic acid.'],
  ['Parameters', 'All the weights and biases a model learns.']
]);
setUses(L6, [['endpoint', 'features', 'plabels'], [], ['endpoint'], [], ['endpoint', 'features', 'filters', 'plabels'], null, ['plabels'], []]);
