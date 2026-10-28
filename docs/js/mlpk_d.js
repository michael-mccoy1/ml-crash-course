"use strict";
/* Lesson 11, Pro steps. Written for PK-literate scientists who are new to machine learning.
   Steps: LLMs in the workflow, Generative design, ML credibility, Validation design.
   Data: the Biogen HLM set and the AstraZeneca logD set (public), through loadDataset.
   Generated structures: data/mlpk_gen.json, strings recorded from the lesson 04 demo GPT.
   ICH M15 statements were read from the final text (EMA Step 5 document). */

function mdSeg(attr, items, cur){ return '<span class="seg" ' + attr + '>' + items.map(function(it, i){ return '<button type="button" data-v="' + i + '"' + (i === cur ? ' class="on"' : '') + '>' + esc(it) + '</button>'; }).join('') + '</span>'; }
function mdPop32(v){ v = v - ((v >>> 1) & 0x55555555); v = (v & 0x33333333) + ((v >>> 2) & 0x33333333); return (((v + (v >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24; }

/* ---------- 1. LLMs in the workflow ---------- */
var MD_TASKS = [
  { n: 'Extracting parameters from the literature',
    f: 'A value is attributed to the wrong species, dose or matrix. Units are dropped or converted incorrectly. A reported value does not appear in the source.',
    v: 'Trace every value to a table or figure in the source. Check species, matrix, dose and units against the source. Recompute derived quantities, for example CL from dose and AUC, and compare.' },
  { n: 'Drafting model code',
    f: 'The code runs and returns plausible curves while an equation, exponent, unit or index is wrong.',
    v: 'Compare the output with closed-form solutions and limiting cases, test mass balance, and write unit tests. ICH M15 Section 3 states that verification of key user-generated code, equations and calculations should be documented and available for review.' },
  { n: 'Summarising papers and reports',
    f: 'A condition or limitation is omitted, a conclusion is reversed, or a detail is added that is not in the text.',
    v: 'Compare the summary with the source passages. Extract the key numbers with a script and compare them with the numbers in the summary.' },
  { n: 'Drafting analysis documents',
    f: 'A statement is not supported by any analysis output, or a number differs from the output it cites.',
    v: 'Map each statement to a table, figure or program output, and regenerate the numbers from the analysis. A person remains accountable for the content (lesson 07, Checks before use).' }
];
var MD_P = { dose: 100, V: 50, CL: 10, ka: 1, F: 0.8 };
/* The constructed snippet, in two versions. The planted error is (ka + ke) in place of (ka - ke). */
function mdConcBug(t, p){ var ke = p.CL / p.V, amp = p.F * p.dose * p.ka / (p.V * (p.ka + ke)); return amp * (Math.exp(-ke * t) - Math.exp(-p.ka * t)); }
function mdConcOk(t, p){ var ke = p.CL / p.V, amp = p.F * p.dose * p.ka / (p.V * (p.ka - ke)); return amp * (Math.exp(-ke * t) - Math.exp(-p.ka * t)); }
var MD_CHECKS = [
  { n: 'Concentration at time 0 is zero', e: 'The oral model starts with no drug in plasma.', run: function(f, p){ var v = f(0, p); return { got: v, exp: 0, fmt: function(x){ return x.toExponential(2) + ' mg/L'; }, abs: true }; } },
  { n: 'Concentrations are never negative', e: 'The smallest concentration over 0 to 200 h is at least 0.', run: function(f, p){ var m = Infinity; for (var t = 0; t <= 200; t += 0.05) m = Math.min(m, f(t, p)); return { got: m, exp: 0, fmt: function(x){ return x.toExponential(2) + ' mg/L'; }, abs: true, geq: true }; } },
  { n: 'Terminal half-life equals ln 2 times V divided by CL', e: 'ln 2 times V divided by CL.', run: function(f, p){ var ke = p.CL / p.V, k = Math.log(f(40, p) / f(60, p)) / 20; return { got: Math.LN2 / k, exp: Math.LN2 / ke, fmt: function(x){ return x.toFixed(3) + ' h'; } }; } },
  { n: 'Time of peak equals ln(ka / ke) divided by (ka - ke)', e: 'The analytical time of the maximum.', run: function(f, p){ var ke = p.CL / p.V, best = 0, tb = 0; for (var t = 0; t <= 24; t += 0.005) { var c = f(t, p); if (c > best) { best = c; tb = t; } } return { got: tb, exp: Math.log(p.ka / ke) / (p.ka - ke), fmt: function(x){ return x.toFixed(3) + ' h'; }, tol: 0.01 }; } },
  { n: 'AUC from 0 to infinity equals F times dose divided by CL', e: 'F times dose divided by CL.', run: function(f, p){ var a = 0, dt = 0.01, prev = 0; for (var t = dt; t <= 400; t += dt) { var c = f(t, p); a += 0.5 * (prev + c) * dt; prev = c; } return { got: a, exp: p.F * p.dose / p.CL, fmt: function(x){ return x.toFixed(3) + ' mg h/L'; } }; } },
  { n: 'Cmax equals the closed-form value from the PK engine', e: 'PKE.oneComp, an independent implementation tested against the ODE solution.', run: function(f, p){ var best = 0; for (var t = 0; t <= 24; t += 0.005) best = Math.max(best, f(t, p)); var s = PKE.oneComp({ dose: p.dose, V: p.V, CL: p.CL, ka: p.ka, F: p.F, route: 'oral' }).summary; return { got: best, exp: s.cmax, fmt: function(x){ return x.toFixed(4) + ' mg/L'; } }; } }
];
function mdEval(chk, f){ var r = chk.run(f, MD_P), tol = chk.tol || 1e-3, ok; if (chk.abs) ok = chk.geq ? r.got >= -1e-12 : Math.abs(r.got - r.exp) < 1e-9; else ok = Math.abs(r.got - r.exp) <= tol * Math.max(1, Math.abs(r.exp)); r.ok = ok; return r; }

LESSONS.mlpk.add({ level: 'pro', short: 'LLMs in the workflow', title: 'LLMs in the workflow',
  change: 'Select a task. In the exercise, select a check and press Run check, or press Run all checks, then press Correct the code.',
  watch: 'Language models are used around PK and MIDD work for four tasks, listed on the left. A language model returns text that is probable given its input (lesson 07). Probable text is not necessarily correct, and each task has a characteristic failure and a check that does not depend on a language model. ICH M15 does not mention language models. Its Section 3 asks that verification of key user-generated code, equations and calculations be documented. The exercise applies that verification to a code snippet that was constructed for this exercise and contains one planted error. A check detects the error only if it constrains the quantity the error changes.',
  build: function(api){
    var cur = 0, sel = 4, fixed = false, results = {};
    var tp = api.el('pn', 'left:28px;top:20px;width:430px;height:232px;overflow:hidden', '<h4>Tasks</h4><div class="mdt"></div>');
    var dp = api.el('pn', 'left:28px;top:262px;width:430px;height:258px;overflow:auto', '<div class="mdd" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ep = api.el('pn', 'left:472px;top:20px;width:460px;height:500px;overflow:auto', '<h4>Exercise: verify constructed model code</h4><div class="mdx" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Check ' + mdSeg('data-ck', MD_CHECKS.map(function(c, i){ return String(i + 1); }), sel) + ' <button type="button" class="btn sm" data-run>Run check</button> <button type="button" class="btn sm ghost" data-all>Run all checks</button> <button type="button" class="btn sm ghost" data-fix>Correct the code</button>');
    function drawTasks(){
      qs('.mdt', tp).innerHTML = MD_TASKS.map(function(t, k){ var on = k === cur; return '<button type="button" data-k="' + k + '" style="display:block;width:100%;text-align:left;font:inherit;font-size:13.5px;padding:8px 10px;margin-bottom:5px;border-radius:6px;cursor:pointer;border:1.5px solid ' + (on ? C.red : C.silver) + ';background:' + (on ? '#FBE9E9' : '#fff') + ';color:' + C.char + ';font-weight:' + (on ? 700 : 500) + '">' + esc(t.n) + '</button>'; }).join('');
      var t = MD_TASKS[cur];
      qs('.mdd', dp).innerHTML = '<div class="ol">Characteristic failure</div><p style="margin:2px 0 10px">' + esc(t.f) + '</p><div class="ol">Verification without a language model</div><p style="margin:2px 0 10px">' + esc(t.v) + '</p><p style="font-size:12px;color:' + C.mid + '">The failures listed are properties of text generation (lesson 07). They are not estimates of how often a given model fails.</p>';
    }
    function code(){
      var op = fixed ? '-' : '+', hl = fixed ? '' : ';background:#FBE9E9';
      return '<pre style="margin:0 0 8px;padding:8px 10px;background:' + C.light + ';border-radius:5px;font:12px/1.5 Cascadia Mono,Consolas,monospace;white-space:pre-wrap">// Constructed for this exercise' + (fixed ? ' (corrected)' : ' (contains one planted error)') + '\nfunction conc(t, dose, V, CL, ka, F) {\n  var ke = CL / V;                       // 1/h\n  <span style="font-weight:700' + hl + '">var amp = F * dose * ka / (V * (ka ' + op + ' ke));</span>\n  return amp * (Math.exp(-ke * t) - Math.exp(-ka * t));\n}</pre><div style="font-size:12px;color:' + C.mid + ';margin-bottom:8px">Parameters: dose ' + MD_P.dose + ' mg, V ' + MD_P.V + ' L, CL ' + MD_P.CL + ' L/h, ka ' + MD_P.ka + ' 1/h, F ' + MD_P.F + '. Oral one-compartment model.</div>';
    }
    function draw(){
      var f = fixed ? mdConcOk : mdConcBug, rows = MD_CHECKS.map(function(c, i){ var r = results[i], on = i === sel;
        return '<tr style="border-top:1px solid ' + C.silver + (on ? ';background:#FBE9E9' : '') + '"><td style="padding:5px 6px;width:22px;font-weight:700">' + (i + 1) + '</td><td style="padding:5px 6px">' + esc(c.n) + '</td><td style="padding:5px 6px;text-align:right;white-space:nowrap">' + (r ? (r.ok ? '<span style="color:#1E6B40;font-weight:700">passes</span>' : '<span style="color:' + C.red + ';font-weight:700">fails</span>') : '<span style="color:' + C.mid + '">not run</span>') + '</td></tr>'; }).join('');
      var det = '', c = MD_CHECKS[sel], r = results[sel];
      if (r) det = '<div style="margin-top:8px;padding:8px 10px;border:1px solid ' + C.silver + ';border-radius:5px;background:#fff"><b>Check ' + (sel + 1) + '.</b> ' + esc(c.n) + '<br>Expected: ' + esc(r.fmt(r.exp)) + ' (' + esc(c.e) + ')<br>From the code: ' + esc(r.fmt(r.got)) + '<br>' + (r.ok ? 'The values agree.' : 'The values differ by ' + (r.exp ? (100 * Math.abs(r.got / r.exp - 1)).toFixed(1) + '%' : 'a nonzero amount') + '.') + '</div>';
      var all = Object.keys(results).length === MD_CHECKS.length, nf = MD_CHECKS.filter(function(_, i){ return results[i] && !results[i].ok; }).length, np = MD_CHECKS.length - nf;
      var sum = all ? '<p style="margin:8px 0 0">' + (fixed ? 'After the correction all six checks pass.' : np + ' of the 6 checks pass and ' + nf + ' fail. Checks 1 to 4 test the shape of the curve and do not depend on the amplitude, so they cannot detect an error in the amplitude. Checks 5 and 6 constrain the amplitude and detect it.') + '</p>' : '';
      qs('.mdx', ep).innerHTML = code() + '<table style="width:100%;border-collapse:collapse;font-size:12.5px">' + rows + '</table>' + det + sum;
    }
    function runOne(i){ results[i] = mdEval(MD_CHECKS[i], fixed ? mdConcOk : mdConcBug); }
    qs('.mdt', tp).addEventListener('click', function(e){ var b = e.target.closest('button'); if (!b) return; cur = +b.dataset.k; drawTasks(); });
    qsa('[data-ck] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.v; qsa('[data-ck] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    qs('[data-run]', ctl).addEventListener('click', function(){ runOne(sel); draw(); });
    qs('[data-all]', ctl).addEventListener('click', function(){ MD_CHECKS.forEach(function(_, i){ runOne(i); }); draw(); });
    qs('[data-fix]', ctl).addEventListener('click', function(){ fixed = !fixed; this.textContent = fixed ? 'Restore the error' : 'Correct the code'; results = {}; draw(); });
    function pick(k){ sel = k; qsa('[data-ck] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); }
    function render(){ drawTasks(); draw(); }
    async function play(tok){
      fixed = false; results = {}; qs('[data-fix]', ctl).textContent = 'Correct the code'; cur = 0; render(); api.cap('Four tasks. Each has a characteristic failure and a check that does not use a language model.'); await api.sleep(3200);
      cur = 1; render(); api.cap('For drafted model code the check is numerical: compare with a closed-form solution.'); await api.sleep(3000);
      pick(0); runOne(0); draw(); api.cap('Check 1 passes. The planted error changes the amplitude of the curve, not its value at time 0.'); await api.sleep(3200);
      pick(4); runOne(4); draw(); api.cap('Check 5 fails. AUC from the code is lower than F times dose divided by CL.'); await api.sleep(3400);
      MD_CHECKS.forEach(function(_, i){ runOne(i); }); draw(); api.cap('Four of six checks pass with the error present. Only checks that constrain the amplitude detect it.'); await api.sleep(3600);
      fixed = true; results = {}; qs('[data-fix]', ctl).textContent = 'Restore the error'; MD_CHECKS.forEach(function(_, i){ runOne(i); }); draw(); api.cap('After the correction all six checks pass.'); await api.sleep(3000); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/LLMs in the workflow'] = { notice: 'Four of the six checks pass although the code contains an error.', q: 'Why do checks 1 to 4 not detect the planted error?', a: 'The error changes the amplitude of the concentration curve. Checks 1 to 4 test its value at time 0, its sign, its terminal slope and the time of its peak, none of which depend on the amplitude.' };

/* ---------- 2. Generative design ---------- */
var MDG = { p: null, J: null, ds: null, az: null, hm: null, lm: null, fps: null, set: null, cache: {}, live: { '0.6': [], '0.8': [], '1': [] }, T: 0.8, thr: 1.2, sim: 0.35 };
var MD_MAP = { logP: 'CrippenClogP', TPSA: 'tpsa', MW: 'amw', HBD: 'NumHBD', HBA: 'NumHBA', RotB: 'NumRotatableBonds', Fsp3: 'FractionCSP3', AromRings: 'NumAromaticRings' };
function mdFp(smi){ var m = RDK.get_mol(smi); if (!m || !m.is_valid()) { if (m) m.delete(); return null; } var s = m.get_morgan_fp(JSON.stringify({ radius: 2, nBits: 1024 })); m.delete(); var w = new Uint32Array(32); for (var i = 0; i < 1024; i++) if (s.charCodeAt(i) === 49) w[i >> 5] |= (1 << (i & 31)); return w; }
function mdTan(a, b){ var inter = 0, uni = 0; for (var i = 0; i < 32; i++) { inter += mdPop32(a[i] & b[i]); uni += mdPop32(a[i] | b[i]); } return uni ? inter / uni : 0; }
async function mdGenReady(api){
  if (!MDG.p) MDG.p = (async function(){
    var ds = await loadDataset('HLM'), az = await loadDataset('logD'); await tick(); MDG.J = await loadJSON('mlpk_gen'); await loadRDKit(); await tick();
    MDG.ds = ds; MDG.az = az; MDG.hm = fitBoost(ds.BASE.slice(), ds.TRAIN, 150, 0.08, 3, 0, ds); await tick(); MDG.lm = fitBoost(az.BASE.slice(), az.TRAIN, 150, 0.1, 4, 0, az); await tick();
    var fps = []; for (var k = 0; k < ds.TRAIN.length; k++) { var f = mdFp(ds.smi_all[ds.TRAIN[k]]); if (f) fps.push(f); if (k % 400 === 399) await tick(); } MDG.fps = fps;
    MDG.set = {}; ds.smi_all.forEach(function(s){ MDG.set[s] = 1; });
  })();
  api.busy('Fitting the property models and computing training fingerprints'); try { await MDG.p; } finally { api.idle(); }
}
function mdInfo(s){
  if (MDG.cache[s]) return MDG.cache[s]; var o = { s: s, valid: false };
  if (rdkValid(s)) { o.valid = true; o.novel = !MDG.set[s]; var d = rdkDescriptors(s), vals = {}; Object.keys(MD_MAP).forEach(function(k){ vals[k] = d[MD_MAP[k]]; });
    o.clint = boostPredictVals(MDG.hm, MDG.ds.B, vals); o.logd = boostPredictVals(MDG.lm, MDG.az.B, vals); var fp = mdFp(s), best = 0; if (fp) for (var i = 0; i < MDG.fps.length; i++) { var t = mdTan(fp, MDG.fps[i]); if (t > best) best = t; } o.nn = best; }
  return (MDG.cache[s] = o);
}
LESSONS.mlpk.add({ level: 'pro', short: 'Generative design', title: 'Generative design under PK constraints',
  change: 'Select a sampling temperature. Set the clearance limit and the similarity threshold. Select Sample 12 more to draw further strings from the GPT.',
  watch: 'A generative model proposes structures, and property models filter them. The strings below were sampled from the lesson 04 demo GPT and recorded once, because sampling in the page is slow. Each filter reduces the set. Valid means RDKit parses the string. Novel means it is not among the 3,087 HLM SMILES, compared as written. The HLM CLint and logD models are boosted trees trained on the course training compounds. The applicability filter keeps candidates whose nearest HLM training compound has a Morgan fingerprint Tanimoto similarity at or above the threshold, which is a setting here. The predictions have no measurement behind them.',
  build: function(api){
    var busyLive = false;
    var fp = api.el('pn', 'left:28px;top:20px;width:300px;height:500px;overflow:auto', '<h4>Filters</h4><div class="mdf" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var hp = api.el('pn', 'left:340px;top:20px;width:592px;height:250px', '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px"><div><h4>Predicted log10 CLint</h4><canvas class="mdh" width="270" height="196"></canvas></div><div><h4>Predicted logD against CLint</h4><canvas class="mds" width="270" height="196"></canvas></div></div><div style="font-size:11.5px;color:' + C.mid + '">Red: nearest training compound within the similarity threshold. Grey: outside. Dashed line: clearance limit.</div>');
    var cp = api.el('pn', 'left:340px;top:280px;width:592px;height:240px;overflow:auto', '<h4>Candidates that pass every filter</h4><div class="mdc" style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px"></div>');
    var ctl = api.addControl('Temperature ' + mdSeg('data-t', ['0.6', '0.8', '1.0'], 1) + ' CLint at most <input type="range" data-thr min="0.7" max="3" step="0.05" value="1.2" style="width:130px"> <b class="tv">1.20</b> Similarity at least <input type="range" data-sim min="0.1" max="0.7" step="0.05" value="0.35" style="width:110px"> <b class="sv">0.35</b> <button type="button" class="btn sm" data-live>Sample 12 more</button>');
    function pool(){ var key = String(MDG.T === 1 ? 1 : MDG.T); return (MDG.J.temperatures[MDG.T.toFixed(1)] || []).concat(MDG.live[key] || []); }
    function compute(){
      var P = pool(), uniq = {}, gen = P.length, valid = [], novel = [];
      P.forEach(function(s){ var o = mdInfo(s); if (o.valid) { valid.push(o); if (o.novel && !uniq[s]) { uniq[s] = 1; novel.push(o); } } });
      var low = novel.filter(function(o){ return o.clint <= MDG.thr; }), ad = low.filter(function(o){ return o.nn >= MDG.sim; });
      return { gen: gen, valid: valid.length, novel: novel, low: low, ad: ad };
    }
    function bar(label, n, tot, col){ return '<div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><span>' + label + '</span><b>' + n + '</b></div><div style="height:12px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:12px;border-radius:3px;background:' + col + ';width:' + (tot ? Math.max(n ? 2 : 0, n / tot * 100) : 0) + '%"></i></div></div>'; }
    function draw(){
      if (!MDG.fps) return; var r = compute(), tr = MDG.ds.TRAIN.length;
      qs('.mdf', fp).innerHTML = bar('Generated at temperature ' + MDG.T.toFixed(1), r.gen, r.gen, C.bg) + bar('Valid (RDKit parses the string)', r.valid, r.gen, C.bg) + bar('Novel and unique', r.novel.length, r.gen, C.bg) + bar('Predicted CLint at most ' + MDG.thr.toFixed(2), r.low.length, r.gen, '#F78C85') + bar('Nearest HLM compound at similarity ' + MDG.sim.toFixed(2) + ' or higher', r.ad.length, r.gen, C.red) +
        '<p style="margin-top:12px;font-size:12px;color:' + C.mid + '">CLint is log10 mL/min/kg. The HLM model was fitted on ' + tr.toLocaleString() + ' training compounds and the logD model on ' + MDG.az.TRAIN.length.toLocaleString() + ' AstraZeneca training compounds, each with the eight descriptors of lesson 01. No measurement exists for any of these structures. The counts describe the filters, not the quality of the candidates.</p>';
      var g = cv(qs('.mdh', hp)), x = g.x, box = { l: 36, r: g.w - 8, t: 8, b: g.h - 40 }, lo = 0.5, hi = 3.5, nb = 20, bw = (hi - lo) / nb, bins = [], k;
      for (k = 0; k < nb; k++) bins.push([0, 0]); r.novel.forEach(function(o){ var j = clamp(Math.floor((o.clint - lo) / bw), 0, nb - 1); bins[j][o.nn >= MDG.sim ? 0 : 1]++; });
      var mx = Math.max(4, Math.max.apply(null, bins.map(function(b){ return b[0] + b[1]; }))); axes(g, box, [lo, hi], [0, mx], 'predicted log10 CLint', 'candidates', 4, { ydec: 0 });
      var sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - v / mx * (box.b - box.t); };
      bins.forEach(function(b, j){ var x0 = sx(lo + j * bw) + 1, w = (box.r - box.l) / nb - 2; x.fillStyle = C.bg; x.fillRect(x0, sy(b[1]), w, sy(0) - sy(b[1])); x.fillStyle = C.red; x.fillRect(x0, sy(b[0] + b[1]), w, sy(b[1]) - sy(b[0] + b[1])); });
      x.strokeStyle = C.char; x.lineWidth = 1.5; x.setLineDash([5, 4]); x.beginPath(); x.moveTo(sx(MDG.thr), box.t); x.lineTo(sx(MDG.thr), box.b); x.stroke(); x.setLineDash([]);
      var g2 = cv(qs('.mds', hp)), x2 = g2.x, b2 = { l: 36, r: g2.w - 8, t: 8, b: g2.h - 40 }; axes(g2, b2, [0.5, 3.5], [-1, 6], 'predicted log10 CLint', 'predicted logD', 4, { ydec: 0 });
      var px = function(v){ return b2.l + (v - 0.5) / 3 * (b2.r - b2.l); }, py = function(v){ return b2.b - (clamp(v, -1, 6) + 1) / 7 * (b2.b - b2.t); };
      r.novel.forEach(function(o){ x2.fillStyle = o.nn >= MDG.sim ? 'rgba(225,36,42,.75)' : 'rgba(161,177,195,.8)'; x2.beginPath(); x2.arc(px(o.clint), py(o.logd), 3.2, 0, 6.3); x2.fill(); });
      x2.strokeStyle = C.char; x2.setLineDash([5, 4]); x2.beginPath(); x2.moveTo(px(MDG.thr), b2.t); x2.lineTo(px(MDG.thr), b2.b); x2.stroke(); x2.setLineDash([]);
      var top = r.ad.slice().sort(function(a, b){ return b.nn - a.nn; }).slice(0, 6), host = qs('.mdc', cp);
      host.innerHTML = top.length ? top.map(function(o){ return '<div style="border:1px solid ' + C.silver + ';border-radius:6px;padding:4px;background:#fff"><div class="struct" style="height:86px">' + rdkSvg(o.s, 170, 86) + '</div><div style="font-size:11.5px;line-height:1.4;padding:2px 4px">Predicted CLint <b>' + o.clint.toFixed(2) + '</b>, logD <b>' + o.logd.toFixed(2) + '</b><br>Nearest training compound, Tanimoto <b>' + o.nn.toFixed(2) + '</b></div></div>'; }).join('') : '<div style="grid-column:1 / 4;font-size:13px;color:' + C.mid + '">No candidate passes every filter at these settings. Raise the clearance limit or lower the similarity threshold.</div>';
    }
    qsa('[data-t] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ MDG.T = [0.6, 0.8, 1][+b.dataset.v]; qsa('[data-t] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    qs('[data-thr]', ctl).addEventListener('input', function(){ MDG.thr = +this.value; qs('.tv', ctl).textContent = MDG.thr.toFixed(2); draw(); });
    qs('[data-sim]', ctl).addEventListener('input', function(){ MDG.sim = +this.value; qs('.sv', ctl).textContent = MDG.sim.toFixed(2); draw(); });
    qs('[data-live]', ctl).addEventListener('click', async function(){
      if (busyLive) return; busyLive = true; var btn = this; btn.disabled = true;
      try { gptSetup(); if (!(await gReady(api))) return; var key = String(MDG.T === 1 ? 1 : MDG.T), tok = api.tok;
        for (var k = 0; k < 12; k++) { api.busy('Sampling string ' + (k + 1) + ' of 12 from the GPT'); MDG.live[key].push(await gWrite([1], MDG.T, 46)); }
        api.idle(); draw(); } catch (e) { api.idle(); console.error(e); } finally { busyLive = false; btn.disabled = false; }
    });
    async function render(){ await mdGenReady(api); draw(); }
    async function play(tok){
      await render(); MDG.thr = 1.2; MDG.sim = 0.35; qs('[data-thr]', ctl).value = 1.2; qs('[data-sim]', ctl).value = 0.35; qs('.tv', ctl).textContent = '1.20'; qs('.sv', ctl).textContent = '0.35'; draw();
      api.cap('Strings sampled from the demo GPT. Each filter removes part of the set.'); await api.sleep(3200);
      api.cap('Valid and novel structures are then scored with the CLint and logD models.'); await api.sleep(3200);
      MDG.sim = 0.5; qs('[data-sim]', ctl).value = 0.5; qs('.sv', ctl).textContent = '0.50'; draw(); api.cap('A stricter similarity threshold removes candidates that lie far from the training compounds.'); await api.sleep(3400);
      api.cap('The predictions for these structures have no measurement behind them.'); await api.sleep(2600); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/Generative design'] = { notice: 'Each filter reduces the set, and the candidates left have predictions but no measurements.', q: 'What does the similarity threshold test, and what does it not test?', a: 'It tests how close a candidate is to the nearest training compound of the property model, a measure of applicability. It does not test whether the predicted CLint is correct for that compound.' };

/* ---------- 3. ML credibility ---------- */
var MD_COMP = [
  { n: 'Parameter predictor feeding PBPK', q: 'Is the predicted exposure of the compound consistent with the planned first-in-human dose?', cou: 'A boosted-trees model trained on HLM data predicts CLint. The prediction feeds an IVIVE and PBPK model. No clinical data exist for the compound.',
    nt: { code: 'The training and prediction code, the feature computation and the units conversion to hepatic clearance.', data: 'Training set composition, assay protocol and censored values at the assay floor (lesson 01, Target distribution and Censored measurements).', struct: 'The prediction enters a fixed mechanistic structure. The ML model does not change that structure.', limit: 'The applicability domain of the structure-based model for the new compound (lesson 07, Applicability domain).', assume: 'Values held constant for fup, blood to plasma ratio and absorption (Predicted parameters as PBPK inputs).', ml: 'Overfitting: report test R2 and RMSE on a split that matches the intended use (lesson 01, Training and test performance; Validation design).', robust: 'Propagate the prediction error to exposure (Error propagation to exposure) and use ensembles (Ensembles for parameter uncertainty).', perf: 'RMSE in log10 units and the fold error of the predicted exposure, in addition to R2.', ext: 'Compounds from a later period or a different series than the training compounds (Validation design).', sim: 'Monte Carlo propagation of parameter uncertainty (lesson 10, Uncertainty propagation).' } },
  { n: 'Surrogate model of a PBPK simulator', q: 'Can the surrogate replace the simulator for a population exposure simulation?', cou: 'A neural network or boosted model trained on simulator runs returns AUC and Cmax from the simulator inputs, in place of the simulator.',
    nt: { code: 'The simulator that generates the training runs, and the surrogate training and prediction code. Compare surrogate output with the simulator.', data: 'The training data are simulator runs. The sampled input ranges define the region of validity.', struct: 'The surrogate has no mechanistic structure. It approximates the simulator inside the sampled region.', limit: 'Inputs outside the sampled ranges are extrapolation (Surrogate models).', assume: 'Assumptions of the simulator carry over unchanged.', ml: 'Overfitting to the simulation design: evaluate on runs not used in training.', robust: 'Compare the surrogate error with the numerical and parameter uncertainty of the simulator.', perf: 'Error relative to the simulator, in the units of the outputs that answer the question of interest.', ext: 'Fresh simulator runs, or observed data, not used to build the surrogate.', sim: 'Simulation scenarios must lie inside the sampled input region.' } },
  { n: 'Covariate screen in population PK', q: 'Which covariates enter the population model?', cou: 'Tree ensembles rank covariates from observed subject data, and the selected covariates enter the population PK model.',
    nt: { code: 'The screening code and the data pipeline from subject records.', data: 'Sample size, covariate correlation and missing values affect the ranking (Covariate screening).', struct: 'Selected covariates enter the population PK model. The structural model is specified separately.', limit: 'Covariate ranges in the data limit the populations the model describes.', assume: 'Handling of missing values and of correlated covariates.', ml: 'Overfitting and selection of spurious covariates: check on held-out subjects (Validation design, leave-one-subject-out).', robust: 'Stability of the selected covariates across resamples of the subjects.', perf: 'Precision and bias of the final population model, not only the screening score.', ext: 'Independent subjects or an independent study.', sim: 'Simulations with the selected covariates should account for parameter uncertainty.' } },
  { n: 'Generative model proposing structures', q: 'Which structures are proposed for synthesis and testing?', cou: 'A generative model proposes structures that property models then filter. The output is a set of candidates for experiment.',
    nt: { code: 'Sampling code, validity checking and the property models (Generative design).', data: 'The training structures determine what the generator reproduces.', struct: 'No mechanistic structure. Property predictions come from separate models.', limit: 'Predictions for generated structures hold only inside the applicability domain of the property models (Applicability domain).', assume: 'Thresholds on predicted properties and similarity are settings.', ml: 'Overfitting appears as reproduction of training structures: report the fraction that is novel (lesson 04, Design test).', robust: 'Sensitivity of the selected set to the thresholds and to the sampling temperature.', perf: 'Validity, novelty and the error of the property models.', ext: 'Measurement of the selected candidates.', sim: 'Not applicable in the same form.' }, scope: true }
];
var MD_EV = [
  { k: 'code', t: 'Verification of user-generated code, equations and calculations is documented and available for review.', s: 'Section 3, Verification', c: 'LLMs in the workflow (this lesson)' },
  { k: 'data', t: 'The relevance and appropriateness of the data are justified. The rationale for exclusion is given and the potential for bias assessed.', s: 'Section 3, Validation and applicability assessment', c: 'Lesson 01, Target distribution' },
  { k: 'struct', t: 'Model structure and parameters are consistent with available knowledge of drug characteristics, pharmacology and physiology, when relevant.', s: 'Section 3, Validation and applicability assessment', c: 'Lesson 10, PBPK model structure' },
  { k: 'limit', t: 'Limitations of the data and the model are described and discussed.', s: 'Section 3, Validation and applicability assessment', c: 'Lesson 07, Applicability domain' },
  { k: 'assume', t: 'Key assumptions are identified, described and justified, and alternatives are considered.', s: 'Section 3, Validation and applicability assessment', c: 'Predicted parameters as PBPK inputs' },
  { k: 'ml', t: 'Method-specific issues are considered. M15 gives overfitting for an artificial intelligence or machine learning model as an example.', s: 'Section 3, Validation and applicability assessment', c: 'Lesson 01, Training and test performance' },
  { k: 'robust', t: 'Model robustness is assessed: dependency on data, parameters, assumptions and associated uncertainty, for example by sensitivity analysis.', s: 'Section 3, Validation and applicability assessment', c: 'Error propagation to exposure' },
  { k: 'perf', t: 'Performance (precision and bias) is assessed with graphical and numerical metrics. Metrics that relate to the question of interest are prioritised.', s: 'Section 3, Validation and applicability assessment', c: 'Lesson 01, Evaluation metrics' },
  { k: 'ext', t: 'External validation with independent data is encouraged. Depending on the question of interest, context of use and model risk, it can be essential.', s: 'Section 3, Validation and applicability assessment', c: 'Validation design (this lesson)' },
  { k: 'sim', t: 'Simulation methods and scenarios are described so that their plausibility and relevance to applicability can be evaluated, and account for parameter and assumption uncertainties.', s: 'Section 3, Validation and applicability assessment', c: 'Lesson 10, Uncertainty propagation' }
];
var MD_LV = ['Low', 'Medium', 'High'];
function mdRisk(inf, con){
  if (inf === 0 && con === 0) return { r: 'Low', note: 'Both ratings are low, so M15 gives model risk as low (Section 2.1.5).' };
  if (inf === 2 && con === 2) return { r: 'High', note: 'Both ratings are high, so M15 gives model risk as high (Section 2.1.5).' };
  if (inf === con) return { r: 'Not fixed', note: 'M15 gives the outcome only when both ratings are low or both are high. The rating needs a justification (Section 2.1.5).' };
  return { r: 'Not fixed', note: 'The ratings differ. M15 states that model risk may be driven by the more influential of the two elements, and that the consideration should be captured in the justification (Section 2.1.5).' };
}
LESSONS.mlpk.add({ level: 'pro', short: 'ML credibility', title: 'Credibility of ML components',
  change: 'Select a component. Set the model influence and the consequence of a wrong decision. Read the model risk and the evidence list.',
  watch: 'ICH M15 assesses MIDD evidence with a question of interest, a context of use, a rating of model influence and a rating of the consequence of a wrong decision. Model risk combines the last two, and model evaluation should be commensurate with it (Sections 2.1 and 3). M15 lists artificial intelligence and machine learning among MIDD methods (Section 1.2) and names overfitting as a method-specific issue for them (Section 3). It does not list evidence by component or by risk level. The left column of the evidence list paraphrases the M15 Section 3 recommendations. The right column is the course interpretation for the selected component and the course step that covers it.',
  build: function(api){
    var cur = 0, inf = 1, con = 1;
    var lp = api.el('pn', 'left:28px;top:20px;width:340px;height:500px;overflow:auto', '<h4>Component</h4><div class="mdk"></div><div class="mdq" style="font-size:12.5px;line-height:1.45;color:' + C.char + ';margin-top:10px"></div><div class="mdr" style="margin-top:10px"></div>');
    var rp = api.el('pn', 'left:380px;top:20px;width:552px;height:500px;overflow:auto', '<h4>Evidence in M15 Section 3, applied to the component</h4><div class="mde" style="font-size:12.5px;line-height:1.42;color:' + C.char + '"></div>');
    var ctl = api.addControl('Model influence ' + mdSeg('data-inf', MD_LV, inf) + ' Consequence of wrong decision ' + mdSeg('data-con', MD_LV, con));
    function draw(){
      var c = MD_COMP[cur], r = mdRisk(inf, con);
      qs('.mdk', lp).innerHTML = MD_COMP.map(function(t, k){ var on = k === cur; return '<button type="button" data-k="' + k + '" style="display:block;width:100%;text-align:left;font:inherit;font-size:13px;padding:7px 10px;margin-bottom:5px;border-radius:6px;cursor:pointer;border:1.5px solid ' + (on ? C.red : C.silver) + ';background:' + (on ? '#FBE9E9' : '#fff') + ';color:' + C.char + ';font-weight:' + (on ? 700 : 500) + '">' + esc(t.n) + '</button>'; }).join('');
      qs('.mdq', lp).innerHTML = '<div class="ol">Question of interest (constructed example)</div><p style="margin:2px 0 8px">' + esc(c.q) + '</p><div class="ol">Context of use (constructed example)</div><p style="margin:2px 0 0">' + esc(c.cou) + '</p>' + (c.scope ? '<p style="margin:8px 0 0;color:' + C.mid + '">Whether M15 applies depends on whether the output is used as MIDD evidence (Section 1.2). The evidence elements below apply to any decision that relies on the model.</p>' : '');
      qs('.mdr', lp).innerHTML = '<div class="ol">Model risk under M15</div><div style="display:flex;gap:12px;align-items:center"><div class="big" style="font-size:22px;margin:0;white-space:nowrap;color:' + (r.r === 'High' ? C.red : (r.r === 'Not fixed' ? C.char : '#1E6B40')) + '">' + (r.r === 'Not fixed' ? 'Not fixed' : r.r) + '</div><div style="font-size:12px;line-height:1.4">' + esc(r.note) + '</div></div><p style="font-size:12px;margin:8px 0 0;color:' + C.mid + '">Model impact is rated separately (Section 2.1.6) and is not part of model risk.</p>';
      qs('.mde', rp).innerHTML = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700"><td style="padding:3px 6px 6px;width:44%">M15 recommendation (paraphrased)</td><td style="padding:3px 6px 6px">For this component, and the course step</td></tr>' + MD_EV.map(function(e){ return '<tr style="border-top:1px solid ' + C.silver + ';vertical-align:top"><td style="padding:6px"><span>' + esc(e.t) + '</span><div style="font-size:11px;color:' + C.mid + ';margin-top:2px">' + esc(e.s) + '</div></td><td style="padding:6px">' + esc(c.nt[e.k]) + '<div style="font-size:11px;color:' + C.mid + ';margin-top:2px">Course: ' + esc(e.c) + '</div></td></tr>'; }).join('') + '</table><p style="font-size:12px;margin:8px 0 0;color:' + C.mid + '">M15 also recommends a predefined model analysis plan, with changes justified and documented in the model analysis report (Sections 4.1 and 4.2). No course step covers it.</p>';
    }
    qs('.mdk', lp).addEventListener('click', function(e){ var b = e.target.closest('button'); if (!b) return; cur = +b.dataset.k; draw(); });
    qsa('[data-inf] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ inf = +b.dataset.v; qsa('[data-inf] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    qsa('[data-con] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ con = +b.dataset.v; qsa('[data-con] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    function set(seg, v){ qsa(seg + ' button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === v); }); }
    async function play(tok){
      cur = 0; inf = 0; con = 0; set('[data-inf]', 0); set('[data-con]', 0); draw(); api.cap('A parameter predictor with low influence and low consequence: M15 gives model risk as low.'); await api.sleep(3600);
      inf = 2; con = 2; set('[data-inf]', 2); set('[data-con]', 2); draw(); api.cap('Both ratings high: M15 gives model risk as high, and model evaluation is commensurate with it.'); await api.sleep(3600);
      inf = 2; con = 0; set('[data-inf]', 2); set('[data-con]', 0); draw(); api.cap('Differing ratings: M15 does not fix the outcome. The justification states which element drives the risk.'); await api.sleep(3800);
      cur = 1; draw(); api.cap('The evidence recommendations apply to every component. What they mean differs by component.'); await api.sleep(3400); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['mlpk/ML credibility'] = { notice: 'M15 fixes model risk only when the influence and consequence ratings are both low or both high.', q: 'Which M15 section names overfitting for an AI or ML model, and what is it an example of?', a: 'Section 3, Validation and applicability assessment: method-specific issues should be considered, and M15 gives overfitting for an artificial intelligence or machine learning model as an example.' };

/* ---------- 4. Validation design ---------- */
var MD_SPLITS = [
  { n: 'Random split', q: 'Performance on compounds drawn from the same distribution as the training compounds.', w: 'Optimistic when close analogues of the test compounds are in the training set.' },
  { n: 'Scaffold split', q: 'Performance on scaffolds that are absent from the training set.', w: 'Estimates transfer to new chemical series.' },
  { n: 'Time-based split', q: 'Performance on compounds made after the model was built.', w: 'Follows prospective use. Sensitive to drift in series and assays.' },
  { n: 'Split by project or series', q: 'Performance on a project that contributed no training compounds.', w: 'Estimates transfer to a new programme.' },
  { n: 'Leave-one-subject-out', q: 'Performance for a subject whose observations were not used in fitting.', w: 'Required for subject-level prediction when each subject contributes many observations.' }
];
function mdBoost(X, y, tr, te, rounds){
  var n = y.length, feats = ['x1', 'x2', 'x3'], B = { edges: {}, codes: {} };
  feats.forEach(function(f, c){ var v = tr.map(function(i){ return X[i][c]; }).sort(function(a, b){ return a - b; }), e = []; for (var q = 1; q < 32; q++) { var xv = v[Math.floor(q / 32 * v.length)]; if (!e.length || xv > e[e.length - 1]) e.push(xv); } B.edges[f] = e; var codes = new Uint8Array(n); for (var i = 0; i < n; i++) { var val = X[i][c], k = 0; while (k < e.length && val > e[k]) k++; codes[i] = k; } B.codes[f] = codes; });
  var ds = { N: n, Y: Float32Array.from(y), B: B, TRAIN: tr, TEST: te, COL: {} };
  var m = fitBoost(feats, tr, rounds || 80, 0.1, 3, 0, ds); return { r2: r2(te, m.F, ds.Y), F: m.F };
}
/* Simulated compounds in time order. Series k appears over one interval of time and has its own feature centre, its own offset and its own slope on x1. */
function mdSim(drift){
  var r = rng(11), N = 1200, S = 6, X = [], y = [], t = [], ser = [], mu = [], off = [], slope = [], k, i;
  for (k = 0; k < S; k++) { mu.push([(r() - 0.5) * 4, (r() - 0.5) * 4, (r() - 0.5) * 4]); off.push((r() - 0.5) * 3); slope.push((r() - 0.5) * 2); }
  function g(){ return (r() + r() + r() + r() - 2) * 1.2; }
  for (i = 0; i < N; i++) { var tt = i / N; k = Math.min(S - 1, Math.floor(tt * S)); var x = [mu[k][0] + g(), mu[k][1] + g(), mu[k][2] + g()]; X.push(x); t.push(tt); ser.push(k); y.push(1.0 * x[0] + 0.5 * x[1] + drift * (off[k] + slope[k] * x[0]) + 0.35 * g()); }
  return { X: X, y: y, t: t, ser: ser, N: N };
}
LESSONS.mlpk.add({ level: 'pro', short: 'Validation design', title: 'Validation design',
  change: 'Drag Drift strength. Select the split to display. Read the R2 of each split on the simulated data and on the HLM data.',
  watch: 'A validation split defines what the test score estimates. The table lists five strategies and the question each one answers. The HLM panel computes boosted-tree R2 for the random split and the scaffold split of the course data, as in lesson 01. The simulated panel is built for this step: 1,200 compounds in time order from six series, each with its own feature region, offset and slope on one feature. At drift 0 the relation is the same in every series. As drift rises, the relation differs between series. A random split places compounds of every series in the training set, so the estimate stays high. A time-based split tests on the latest 30% of compounds, which come from series that are absent from training or only partly present in it, so the estimate falls.',
  build: function(api){
    var drift = 0.8, show = 0, hlm = null, simC = {};
    var tp = api.el('pn', 'left:28px;top:20px;width:430px;height:256px;overflow:auto', '<h4>What each split estimates</h4><div class="mds"></div>');
    var hp = api.el('pn', 'left:28px;top:286px;width:430px;height:234px;overflow:auto', '<h4>HLM data, boosted trees</h4><div class="mdh" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var sp = api.el('pn', 'left:472px;top:20px;width:460px;height:500px', '<h4>Simulated compounds in time order (simulated, not measured)</h4><div style="font-size:11.5px;color:' + C.mid + ';margin:0 0 4px">Grey: training compounds. Red: test compounds.</div><canvas class="mdp" width="430" height="250"></canvas><div class="mdb" style="margin-top:8px"></div>');
    var ctl = api.addControl('Drift strength <input type="range" min="0" max="1" step="0.05" value="0.8" style="width:170px"> <b class="dv">0.80</b> Split shown ' + mdSeg('data-sp', ['Random', 'Time-based'], show));
    function sim(){ var key = drift.toFixed(2); if (simC[key]) return simC[key]; var d = mdSim(drift), N = d.N, rr = rng(5), idx = []; for (var i = 0; i < N; i++) idx.push(i);
      for (var a = N - 1; a > 0; a--) { var j = Math.floor(rr() * (a + 1)), tmp = idx[a]; idx[a] = idx[j]; idx[j] = tmp; }
      var nTr = Math.round(N * 0.7), rTr = idx.slice(0, nTr).sort(function(a, b){ return a - b; }), rTe = idx.slice(nTr).sort(function(a, b){ return a - b; }), tTr = [], tTe = []; for (i = 0; i < N; i++) (i < nTr ? tTr : tTe).push(i);
      var a1 = mdBoost(d.X, d.y, rTr, rTe), a2 = mdBoost(d.X, d.y, tTr, tTe); return (simC[key] = { d: d, rTr: rTr, rTe: rTe, tTr: tTr, tTe: tTe, rand: a1.r2, time: a2.r2 }); }
    function drawTable(){
      qs('.mds', tp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12.5px;line-height:1.4;color:' + C.char + '">' + MD_SPLITS.map(function(s, k){ var on = (k === 0 && show === 0) || (k === 2 && show === 1); return '<tr style="border-top:1px solid ' + C.silver + ';vertical-align:top' + (on ? ';background:#FBE9E9' : '') + '"><td style="padding:5px 6px;width:30%;font-weight:700">' + esc(s.n) + '</td><td style="padding:5px 6px">' + esc(s.q) + '<div style="color:' + C.mid + ';font-size:12px;margin-top:2px">' + esc(s.w) + '</div></td></tr>'; }).join('') + '</table>';
    }
    function drawHLM(){
      if (!hlm) { qs('.mdh', hp).textContent = 'Computing'; return; }
      qs('.mdh', hp).innerHTML = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px"><div><div class="ol">Random split</div><div class="big" style="color:' + C.char + ';font-size:28px;margin:0">' + hlm.rand.toFixed(2) + '</div><div style="font-size:12px">test R2, ' + hlm.nR + ' test compounds</div><div style="font-size:12px;margin-top:4px">Mean similarity to the closest training compound: ' + hlm.simR.toFixed(2) + '</div></div><div><div class="ol">Scaffold split</div><div class="big" style="color:' + C.red + ';font-size:28px;margin:0">' + hlm.scaf.toFixed(2) + '</div><div style="font-size:12px">test R2, ' + hlm.nS + ' test compounds</div><div style="font-size:12px;margin-top:4px">Mean similarity to the closest training compound: ' + hlm.simS.toFixed(2) + '</div></div></div><p style="margin:10px 0 0;font-size:12.5px">' + (Math.abs(hlm.rand - hlm.scaf) < 0.05 ? 'On this data set the two estimates are close, as are the mean similarities to the closest training compound.' : 'The two estimates differ by ' + Math.abs(hlm.rand - hlm.scaf).toFixed(2) + ' R2.') + ' The split that matches the intended use is the one to report.</p>';
    }
    function drawSim(){
      var S = sim(), d = S.d, g = cv(qs('.mdp', sp)), x = g.x, box = { l: 40, r: g.w - 10, t: 10, b: g.h - 42 }, ys = d.y, lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), tr = show === 0 ? S.rTr : S.tTr, te = show === 0 ? S.rTe : S.tTe;
      axes(g, box, [0, 1], [Math.floor(lo), Math.ceil(hi)], 'time of synthesis (relative)', 'measured value (simulated)', 4, { xdec: 1, ydec: 0 });
      var sx = function(v){ return box.l + v * (box.r - box.l); }, sy = function(v){ return box.b - (v - Math.floor(lo)) / (Math.ceil(hi) - Math.floor(lo)) * (box.b - box.t); };
      x.fillStyle = 'rgba(161,177,195,.65)'; tr.forEach(function(i){ x.beginPath(); x.arc(sx(d.t[i]), sy(d.y[i]), 2.2, 0, 6.3); x.fill(); });
      x.fillStyle = 'rgba(225,36,42,.8)'; te.forEach(function(i){ x.beginPath(); x.arc(sx(d.t[i]), sy(d.y[i]), 2.4, 0, 6.3); x.fill(); });
      
      var bar = function(label, v, col){ return '<div style="display:grid;grid-template-columns:120px 1fr 52px;gap:8px;align-items:center;margin:5px 0;font-size:12.5px"><span>' + label + '</span><div style="height:14px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:14px;border-radius:3px;background:' + col + ';width:' + (clamp(v, 0, 1) * 100) + '%"></i></div><b style="text-align:right">' + v.toFixed(2) + '</b></div>'; };
      qs('.mdb', sp).innerHTML = bar('Random split', S.rand, show === 0 ? C.red : C.bg) + bar('Time-based split', S.time, show === 1 ? C.red : C.bg) + '<p style="font-size:12.5px;margin:8px 0 0;color:' + C.char + '">Test R2 of a boosted-trees model on 70% training and 30% test compounds. ' + (S.rand - S.time > 0.1 ? 'The random split overstates performance on the latest compounds by ' + (S.rand - S.time).toFixed(2) + ' R2.' : 'At this drift the two estimates are close.') + '</p>';
    }
    function draw(){ drawTable(); drawHLM(); drawSim(); }
    async function ready(){
      if (hlm) return; var ds = await loadDataset('HLM'); await tick(); var rtr = [], rte = []; ds.rtest.forEach(function(t, i){ (t ? rte : rtr).push(i); });
      var fr = fitBoost(ds.BASE.slice(), rtr, 60, 0.1, 3, 0, ds), fs = fitBoost(ds.BASE.slice(), ds.TRAIN, 60, 0.1, 3, 0, ds);
      hlm = { rand: r2(rte, fr.F, ds.Y), scaf: r2(ds.TEST, fs.F, ds.Y), nR: rte.length, nS: ds.TEST.length, simR: ds.sim.random.mean, simS: ds.sim.scaffold.mean };
    }
    var rg = qs('input[type=range]', ctl);
    rg.addEventListener('input', function(){ drift = +rg.value; qs('.dv', ctl).textContent = drift.toFixed(2); draw(); });
    qsa('[data-sp] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ show = +b.dataset.v; qsa('[data-sp] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ draw(); api.busy('Fitting the split comparison on the HLM data'); try { await ready(); } finally { api.idle(); } draw(); }
    async function play(tok){
      await render(); drift = 0; rg.value = 0; qs('.dv', ctl).textContent = '0.00'; draw(); api.cap('At drift 0 the relation is identical in every series, and the two splits agree.'); await api.sleep(3400);
      await api.anim(2600, function(t){ drift = Math.round(t * 20) / 20; rg.value = drift; qs('.dv', ctl).textContent = drift.toFixed(2); draw(); }); api.cap('As drift rises, the random split stays high and the time-based split falls.'); await api.sleep(3400);
      show = 1; qsa('[data-sp] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 1); }); draw(); api.cap('In the time-based split the test compounds are the latest 30%, from series that are absent from training or only partly present in it.'); await api.sleep(3400); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/Validation design'] = { notice: 'The random split stays high as drift rises. The time-based split falls.', q: 'Which split estimates performance for compounds made after the model was built?', a: 'The time-based split. Random splitting places compounds of every period in the training set, so it estimates performance on compounds like those already seen.' };
