"use strict";
/* Lesson 5: GPT. A tiny decoder-only transformer trained on SMILES in the browser. */
var L4 = new Lesson('gpt', 'GPT');
L4.optsHTML = function(){
  return optRange('glayers', 'Transformer layers', 1, 3, 1, 2, 'Blocks stacked in the model. More blocks, more capacity.') +
    optRange('gsteps', 'Training steps', 100, 1500, 50, 250, 'Each step uses a batch of 32 molecules.') +
    optRange('gtemp', 'Temperature', 0.3, 1.6, 0.1, 0.8, 'Low favours likely tokens. High gives more variety.') +
    '<div class="og"><div class="ol">Last training</div><div class="rbox" id="gres"><div class="hint">Press Train in the workbench.</div></div><div class="hint">The GPT always reads the Biogen SMILES, whichever endpoint is selected.</div></div>' + filterGroup({ labels: false });
};
var TOK = /\[[^\]]+\]|Br|Cl|%\d\d|./g, TLEN = 48, DM = 64, HEADS = 4;
var G = { P: null, NL: 2, vocab: null, stoi: null, V: 0, seqs: null, trained: false, losses: [], secs: null, steps: 0, params: 0, busy: false, emb0: null, readIds: null, sig: null };
function GP(){ var g = function(id, d){ var e = $(id); return e ? +e.value : d; }; return { nl: g('glayers', 2), steps: g('gsteps', 250), temp: g('gtemp', 0.8) }; }
function gptCorpusSig(){ return JSON.stringify([FILT.mw, FILT.lp]); }
function gptSetup(){
  if (G.vocab) return; var d = DATA.HLM, sm = d.smi_all || [], n = sm.length, toks = new Array(n), set = {}, pool = [];
  for (var i = 0; i < n; i++) { var t = sm[i].match(TOK) || []; toks[i] = t; if (!d.test[i] && sm[i].length <= 55 && t.length <= TLEN - 2 && pool.length < 1500) { pool.push(i); t.forEach(function(x){ set[x] = 1; }); } }
  G.tokLen = toks.map(function(t){ return t.length; }); G.vocab = ['<pad>', '<bos>', '<eos>'].concat(Object.keys(set).sort()); G.stoi = {}; G.vocab.forEach(function(t, k){ G.stoi[t] = k; }); G.V = G.vocab.length;
  G.allIds = new Array(n).fill(null); pool.forEach(function(i){ var ids = [1].concat(toks[i].map(function(x){ return G.stoi[x]; }), [2]); while (ids.length < TLEN + 1) ids.push(0); G.allIds[i] = ids; }); gCorpus();
}
function gCorpus(){
  var d = DATA.HLM, sig = gptCorpusSig(); if (G.csig === sig && G.seqs) return; G.seqs = []; G.smiles = []; for (var i = 0; i < d.N; i++) if (G.allIds[i] && passRange(d, i)) { G.seqs.push(G.allIds[i]); G.smiles.push(d.smi_all[i]); }
  if (G.seqs.length < 40) { G.seqs = []; G.smiles = []; for (var j = 0; j < d.N; j++) if (G.allIds[j]) { G.seqs.push(G.allIds[j]); G.smiles.push(d.smi_all[j]); } } G.csig = sig; G.readIds = null;
}
function tokName(i){ return i === 0 ? '[pad]' : (i === 1 ? '[start]' : (i === 2 ? '[end]' : G.vocab[i])); }
function gMake(nl){
  var V = G.V;
  var nrm = function(s){ return tf.variable(tf.randomNormal(s, 0, 0.02)); }, one = function(n){ return tf.variable(tf.ones([n])); }, zero = function(n){ return tf.variable(tf.zeros([n])); };
  var P = { tok: nrm([V, DM]), pos: nrm([TLEN, DM]), lnfg: one(DM), lnfb: zero(DM), Wout: nrm([DM, V]) };
  for (var l = 0; l < nl; l++) { P['g1_' + l] = one(DM); P['b1_' + l] = zero(DM); P['Wqkv' + l] = nrm([DM, 3 * DM]); P['bqkv' + l] = zero(3 * DM); P['Wo' + l] = nrm([DM, DM]); P['bo' + l] = zero(DM); P['g2_' + l] = one(DM); P['b2_' + l] = zero(DM); P['W1' + l] = nrm([DM, 4 * DM]); P['c1' + l] = zero(4 * DM); P['W2' + l] = nrm([4 * DM, DM]); P['c2' + l] = zero(DM); }
  return P;
}
function gInit(nl){
  gptSetup(); gCorpus(); if (G.P) Object.keys(G.P).forEach(function(k){ G.P[k].dispose(); }); G.NL = nl; var P = gMake(nl);
  G.P = P; G.trained = false; G.losses = []; G.emb0 = P.tok.arraySync(); G.params = Object.keys(P).reduce(function(a, k){ return a + P[k].size; }, 0);
}
function glin(x, W, b){ var s = x.shape, y = x.reshape([-1, s[s.length - 1]]).matMul(W); if (b) y = y.add(b); return y.reshape(s.slice(0, -1).concat([W.shape[1]])); }
function gln(x, g, b){ var mm = tf.moments(x, -1, true); return x.sub(mm.mean).div(mm.variance.add(1e-5).sqrt()).mul(g).add(b); }
function gforward(ids, nl, wantAtt, PP, wantHid){
  var P = PP || G.P, B = ids.shape[0], T = ids.shape[1], dh = DM / HEADS, atts = [], x = tf.gather(P.tok, ids.flatten()).reshape([B, T, DM]).add(P.pos.slice([0, 0], [T, DM])), mask = tf.linalg.bandPart(tf.ones([T, T]), -1, 0).sub(1).mul(1e9);
  for (var l = 0; l < nl; l++) { var h = gln(x, P['g1_' + l], P['b1_' + l]), qkv = glin(h, P['Wqkv' + l], P['bqkv' + l]), part = function(k){ return qkv.slice([0, 0, k * DM], [B, T, DM]).reshape([B, T, HEADS, dh]).transpose([0, 2, 1, 3]); };
    var q = part(0), k = part(1), v = part(2), a = tf.softmax(q.matMul(k, false, true).div(Math.sqrt(dh)).add(mask)); if (wantAtt) atts.push(a);
    x = x.add(glin(a.matMul(v).transpose([0, 2, 1, 3]).reshape([B, T, DM]), P['Wo' + l], P['bo' + l])); var h2 = gln(x, P['g2_' + l], P['b2_' + l]); x = x.add(glin(tf.relu(glin(h2, P['W1' + l], P['c1' + l])), P['W2' + l], P['c2' + l])); }
  if (wantHid) return gln(x, P.lnfg, P.lnfb);
  var logits = glin(gln(x, P.lnfg, P.lnfb), P.Wout, null); return wantAtt ? { logits: logits, atts: atts } : logits;
}
function gLoss(batch, nl){ var V = G.V, inp = tf.tensor2d(batch.map(function(s){ return s.slice(0, TLEN); }), null, 'int32'), tgt = tf.tensor2d(batch.map(function(s){ return s.slice(1, TLEN + 1); }), null, 'int32'), lp = tf.logSoftmax(gforward(inp, nl)), nll = tf.neg(lp.mul(tf.oneHot(tgt.flatten(), V).reshape([batch.length, TLEN, V])).sum(-1)), m = tf.notEqual(tgt, 0).cast('float32'); return nll.mul(m).sum().div(m.sum()); }
function gDist(ids, temp){ var ban = tf.tensor1d(G.vocab.map(function(_, i){ return i === 0 || i === 1 ? -1e9 : 0; })); var out = tf.tidy(function(){ var lg = gforward(tf.tensor2d([ids], null, 'int32'), G.NL).slice([0, ids.length - 1, 0], [1, 1, G.V]).reshape([G.V]).div(temp).add(ban); return tf.softmax(lg).dataSync(); }); ban.dispose(); return out; }
function gSample(p){ var r = Math.random(), c = 0; for (var i = 0; i < p.length; i++) { c += p[i]; if (r <= c) return i; } return p.length - 1; }
async function gWrite(prefix, temp, maxTok){ var ids = prefix.slice(); for (var t = ids.length; t < maxTok; t++) { var nx = gSample(gDist(ids, temp)); if (nx === 2) break; ids.push(nx); if (t % 10 === 9) await tick(); } return ids.slice(1).map(function(i){ return G.vocab[i]; }).join(''); }
function syntaxOk(s){ if (!s) return false; var depth = 0, rings = {}, t = s.match(TOK) || []; for (var i = 0; i < t.length; i++) { var x = t[i]; if (x === '(') depth++; else if (x === ')') { depth--; if (depth < 0) return false; } else if (x[0] === '[') { if (x.slice(-1) !== ']') return false; } else if (/^\d$/.test(x) || /^%\d\d$/.test(x)) rings[x] = (rings[x] || 0) + 1; } if (depth !== 0) return false; for (var k in rings) if (rings[k] % 2) return false; return !/[=#\-(\/\\]/.test(t[t.length - 1]) && !/^[=#)]/.test(t[0]); }
function tokCat(t){ if (/^[A-Za-z]/.test(t) && t[0] !== '[') return 'atom'; if (t[0] === '[') return 'bracket'; if (/^\d$|^%/.test(t)) return 'ring'; if (t === '(' || t === ')') return 'branch'; if (t === '<pad>' || t === '<bos>' || t === '<eos>') return 'special'; return 'bond'; }
var CATC = { atom: C.red, ring: C.char, branch: C.bg, bond: '#F78C85', bracket: C.dred, special: '#D1D8E0' };

/* training */
async function gTrain(o){
  while (G.busy) await new Promise(function(r){ setTimeout(r, 250); }); if (G.trained && !gStale() && !(o && o.force)) return G;
  gptSetup(); gCorpus(); var p = GP(); gInit(p.nl); var opt = tf.train.adam(0.003), vars = Object.keys(G.P).map(function(k){ return G.P[k]; }), r = rng(5), gT = 0, steps = p.steps; G.busy = true; G.losses = [];
  try { for (var s = 0; s < steps; s++) { var batch = []; for (var b = 0; b < 32; b++) batch.push(G.seqs[Math.floor(r() * G.seqs.length)]); var tm = performance.now(), c = opt.minimize(function(){ return gLoss(batch, G.NL); }, true, vars); G.losses.push(c.dataSync()[0]); c.dispose(); gT += performance.now() - tm; if (o && o.onStep) await o.onStep(s, steps); if (s % 6 === 5) await tick(); }
    G.trained = true; G.secs = gT / 1000; G.steps = steps; G.sig = JSON.stringify([p.nl, steps, gptCorpusSig()]); G.demo = false; } finally { opt.dispose(); G.busy = false; }
  gResultsBox(); return G;
}
function gResultsBox(){ var el = $('gres'); if (!el) return; if (!G.trained) { el.innerHTML = '<div class="hint">Press Train in the workbench.</div>'; return; } el.innerHTML = (G.demo ? '<div class="hint">A demo model is loaded so every step works. Press Train to train your own.</div>' : '') + '<div class="rrow">Final loss<b>' + G.losses[G.losses.length - 1].toFixed(2) + '</b></div><div class="rrow">Corpus<b>' + G.seqs.length + ' SMILES</b></div><div class="rrow">Parameters<b>' + G.params.toLocaleString() + '</b></div><div class="rrow">Training time<b>' + fmtT(G.secs) + '</b></div>' + (gStale() ? '<div class="hint" style="color:#E1242A">Options changed since this training.</div>' : ''); }
function gStale(){ return !G.trained || JSON.stringify([GP().nl, GP().steps, gptCorpusSig()]) !== G.sig; }
function gStaleText(){ return G.demo ? 'Showing the demo model (' + G.steps + ' training steps). Press Train in the workbench to train your own.' : 'Options changed since the last Train.'; }
async function gptLoadDemo(){
  if (G.trained || G.busy || G.demoTried) return G.trained; G.demoTried = true; if (typeof tf === 'undefined') return false;
  try {
    var r = await fetch('data/transformer_demo.json?v=f6'); if (!r.ok) return false; var j = await r.json(); gptSetup(); gCorpus(); if (JSON.stringify(j.vocab) !== JSON.stringify(G.vocab)) return false; if (G.trained || G.busy) return G.trained;
    if (G.P) Object.keys(G.P).forEach(function(k){ G.P[k].dispose(); }); var P = {}; Object.keys(j.params).forEach(function(k){ var q = j.params[k]; P[k] = tf.variable(tf.tensor(q.data, q.shape)); });
    var P0 = gMake(j.nl); G.emb0 = P0.tok.arraySync(); Object.keys(P0).forEach(function(k){ P0[k].dispose(); });
    G.P = P; G.NL = j.nl; G.losses = j.losses; G.steps = j.steps; G.secs = j.secs; G.params = Object.keys(P).reduce(function(a, k){ return a + P[k].size; }, 0); G.trained = true; G.demo = true; G.sig = JSON.stringify([j.nl, j.steps, gptCorpusSig()]); G.readIds = null;
    (async function(){ var lens = [], ok = 0, txt = ''; for (var k = 0; k < 20; k++) { txt = await gWrite([1], 0.8, 46); lens.push((txt.match(TOK) || []).length); if (syntaxOk(txt)) ok++; } G.last = txt; G.gen = { sig: gptCorpusSig(), lens: lens, ok: ok }; if (LESSONS.gpt && LESSONS.gpt.built) LESSONS.gpt.refresh(); })();
    gResultsBox(); if (LESSONS.gpt && LESSONS.gpt.built) LESSONS.gpt.refresh(); if (LESSONS.pre && LESSONS.pre.built) LESSONS.pre.refresh(); return true;
  } catch (e) { console.error(e); return false; }
}
var NOTF = 'TensorFlow.js did not load in this browser, so the GPT cannot run here.';
async function gReady(api){
  if (typeof tf === 'undefined') { if (api) api.gate(NOTF); return false; }
  if (!G.trained) await gptLoadDemo();
  if (G.trained) { if (api) { api.idle(); api.stale(gStale() ? gStaleText() : ''); } return true; }
  if (api) api.gate('Train the GPT in the workbench first.'); return false;
}
var gTimer = null;
L4.onOpts = function(){ gResultsBox(); var fire = function(){ clearTimeout(gTimer); gTimer = setTimeout(function(){ gResultsBox(); L4.refresh(); }, 200); }; ['glayers', 'gsteps', 'gtemp'].forEach(function(id){ $(id).addEventListener('input', function(){ if (L4.live) L4.live(); fire(); }); }); };

/* ---------- workbench: set up the GPT, press Train ---------- */
function gMeanAbs(name){ return tf.tidy(function(){ return G.P[name].abs().mean().dataSync()[0]; }); }
L4.add({ short: 'Workbench', bench: true, uses: ['filters', 'glayers', 'gsteps', 'gtemp'],
  title: 'Train the GPT and watch it learn',
  watch: 'A GPT is trained on one task: predict the next token of a SMILES string. Training starts from random weights and adjusts them to lower the next-token loss, the negative log probability the model assigns to the correct token. Left: the model and the size of its weights. Top right: the loss. The coloured boxes show the probability assigned to each correct token. The dark box shows a molecule the model generates.',
  change: 'Set the options in the options panel, then press Train.',
  build: function(api){
    var bb = benchBar(api);
    var np = api.el('pn', 'left:28px;top:80px;width:456px;height:372px', '<h4 class="sh">The model</h4><svg viewBox="0 0 432 318" width="432" height="318"></svg>');
    var lp = api.el('pn', 'left:504px;top:80px;width:428px;height:148px', '<h4>Loss per training step</h4><canvas width="404" height="112"></canvas>');
    var rp = api.el('pn', 'left:504px;top:240px;width:428px;height:212px', '<h4>Probability of the correct next token, one training molecule</h4><div class="rv" style="min-height:62px"></div><div class="rs note" style="font-size:12px;color:#34373F;margin:4px 0 8px"></div><h4>Generated sample</h4><div class="at smi" style="background:#34373F;color:#fff;border-radius:6px;padding:8px 10px;min-height:42px;font-size:12px;word-break:break-all"></div><div class="ac" style="font-size:12px;color:#34373F;margin-top:5px"></div>');
    var mn = api.el('pn', 'left:28px;top:462px;width:904px;height:56px;display:flex;align-items:center;font-size:14px;line-height:1.4', '<div class="mn"></div>');
    var svg = qs('svg', np), running = false, writing = false;
    function fresh(){ return G.trained && !gStale(); }
    function stack(){
      gptSetup(); var N = GP().nl, V = G.V, items = [{ t: 'Next-token probabilities', s: V + ' probabilities', k: 'out', p: 'Wout' }], tot = V * DM + TLEN * DM + 2 * DM + DM * V, per = DM * 3 * DM + 3 * DM + DM * DM + DM + DM * 4 * DM + 4 * DM + 4 * DM * DM + DM + 4 * DM; tot += N * per;
      for (var b = N - 1; b >= 0; b--) { items.push({ t: 'Block ' + (b + 1) + ': feed-forward', s: DM + ' to ' + 4 * DM + ' to ' + DM, k: 'ff' + b, p: ['W1' + b, 'W2' + b] }); items.push({ t: 'Block ' + (b + 1) + ': attention', s: HEADS + ' heads', k: 'at' + b, p: ['Wqkv' + b, 'Wo' + b] }); }
      items.push({ t: 'Embeddings', s: V + ' tokens, ' + DM + ' numbers each', k: 'emb', p: 'tok' }); items.push({ t: 'Token IDs and positions', s: 'the text so far, as numbers', k: 'in', p: null });
      var live = G.P && G.NL === N, st = items.map(function(it){ if (!live || !it.p) return null; var ps = [].concat(it.p), v = 0; ps.forEach(function(n){ v += gMeanAbs(n); }); return v / ps.length; }), mx = Math.max.apply(null, st.map(function(v){ return v || 0; }).concat([1e-6])), bh = 26, gap = 7, y = 4, s = '';
      items.forEach(function(it, i){
        s += '<rect x="6" y="' + y + '" width="290" height="' + bh + '" rx="5" fill="#fff" stroke="' + C.silver + '"/><text x="16" y="' + (y + 12) + '" font-size="12" font-weight="700" fill="' + C.char + '">' + it.t + '</text><text x="16" y="' + (y + 23) + '" font-size="9.5" fill="' + C.mid + '">' + it.s + '</text>';
        if (st[i] !== null) { var w = Math.max(2, st[i] / mx * 100); s += '<rect x="312" y="' + (y + 8) + '" width="110" height="10" rx="3" fill="' + C.silver + '"/><rect x="312" y="' + (y + 8) + '" width="' + w + '" height="10" rx="3" fill="' + C.red + '"/>'; }
        if (i < items.length - 1) s += '<path d="M151 ' + (y + bh + gap - 1) + 'L151 ' + (y + bh + 1) + '" stroke="' + C.red + '" stroke-width="1.6"/>'; y += bh + gap; });
      s += '<text x="422" y="' + (y + 4) + '" font-size="11" text-anchor="end" fill="' + C.mid + '">bar: average size of the weights</text><text x="6" y="' + (y + 4) + '" font-size="12" font-weight="700" fill="' + C.dred + '">' + tot.toLocaleString() + ' parameters</text>'; svg.innerHTML = s;
    }
    function lossDraw(n){ var p = GP(), pts = G.losses.slice(0, n === undefined ? G.losses.length : n).map(function(v, i){ return [i + 1, v]; }), o = { xl: 'training step', yl: 'loss', x0: 1, x1: Math.max(2, p.steps), xdec: 0, ydec: 1 }; if (!pts.length) { var g0 = lineChart(qs('canvas', lp), [{ pts: [[1, 0], [Math.max(2, p.steps), 3]], color: 'rgba(0,0,0,0)', nodot: true }], Object.assign({ y0: 0 }, o)); g0.x.fillStyle = C.mid; g0.x.textAlign = 'center'; g0.x.fillText('The loss curve appears after Train', (g0.box.l + g0.box.r) / 2, (g0.box.t + g0.box.b) / 2); } else lineChart(qs('canvas', lp), [{ pts: pts, color: C.red, nodot: true }], o); }
    function readViz(){
      if (!G.P) { qs('.rv', rp).innerHTML = ''; qs('.rs', rp).textContent = ''; return; } if (!G.readIds) { var sm = G.smiles.filter(function(x){ return x.match(TOK).length <= 22; })[2] || G.smiles[0], t = sm.match(TOK); G.readIds = [1].concat(t.map(function(x){ return G.stoi[x]; }), [2]); }
      var ids = G.readIds, pr; tf.tidy(function(){ var pm = tf.softmax(gforward(tf.tensor2d([ids.slice(0, -1)], null, 'int32'), G.NL)).arraySync()[0]; pr = ids.slice(1).map(function(id, k){ return pm[k][id]; }); });
      qs('.rv', rp).innerHTML = ids.slice(1).map(function(id, k){ var q = pr[k]; return '<span class="tok" style="animation:none;text-align:center;padding:4px 5px;margin:0 3px 3px 0;font-size:10.5px;background:rgba(225,36,42,' + (0.06 + 0.94 * q).toFixed(2) + ');color:' + (q > 0.5 ? '#fff' : C.char) + '"><span>' + esc(tokName(id)) + '</span><small style="display:block;font-size:8.5px;margin-top:2px;opacity:.85">' + Math.round(q * 100) + '%</small></span>'; }).join('');
      qs('.rs', rp).textContent = 'Mean probability ' + Math.round(mean(pr) * 100) + '%. A uniform distribution would give ' + (100 / G.V).toFixed(1) + '%.';
    }
    function attempt(txt){ qs('.at', rp).textContent = txt || ''; qs('.ac', rp).textContent = txt ? (syntaxOk(txt) ? 'Brackets and rings balance.' : 'Brackets or rings do not balance.') : ''; }
    function note(){ var p = GP(); qs('.mn', mn).textContent = p.nl + ' transformer layer' + (p.nl > 1 ? 's' : '') + ', ' + HEADS + ' attention heads. ' + p.steps + ' training steps of 32 molecules each.'; }
    function render(){
      stack(); note(); var p = GP(); if (typeof tf === 'undefined') { api.gate(NOTF); return; }
      if (!G.trained && !G.busy && !G.demoTried) { gptLoadDemo().then(function(){ L4.refresh(); }); }
      if (G.trained) { lossDraw(); readViz(); attempt(G.last || ''); api.stale(gStale() ? gStaleText() : ''); bb.say(G.demo ? 'Demo model: ' + G.steps + ' steps, final loss ' + G.losses[G.losses.length - 1].toFixed(2) + '. Press Train to train your own.' : 'Final loss ' + G.losses[G.losses.length - 1].toFixed(2) + '. Training took ' + fmtT(G.secs) + '.'); }
      else { lossDraw(0); if (!G.busy) { if (!G.P || G.NL !== p.nl) gInit(p.nl); readViz(); stack(); if (!writing) { writing = true; gWrite([1], 1.0, 30).then(function(t){ writing = false; if (!G.trained && !running) attempt(t); }); } } api.stale(''); bb.say('Ready. Press Train.'); }
    }
    async function train(){
      if (running || G.busy) return; running = true; bb.btn.disabled = true; api.tok++; var tok = api.tok, p = GP(); api.stale('');
      try {
        gptSetup(); gCorpus(); G.readIds = null; api.busy('Preparing the model'); await delay(40); gInit(p.nl); api.idle(); stack(); lossDraw(0); readViz(); attempt(await gWrite([1], 1.0, 34)); G.last = '';
        var per = clamp(25000 * speedMul() / p.steps, 3, 150), every = Math.max(1, Math.round(p.steps / 8));
        await gTrain({ force: true, onStep: async function(s, n){
          api.checkTok(tok);
          if (s % 5 === 4 || s === n - 1) { lossDraw(s + 1); bb.say('Step ' + (s + 1) + ' of ' + n + '. Loss ' + G.losses[s].toFixed(2)); progress((s + 1) / n); }
          if (s % 10 === 9 || s === n - 1) { stack(); readViz(); }
          if ((s + 1) % every === 0 && s < n - 1) { attempt(await gWrite([1], p.temp, 40)); }
          if (per > 3) await api.sleep(per);
        } });
        api.checkTok(tok); bb.say('Sampling molecules'); var lens = [], ok = 0, txt = ''; for (var k = 0; k < 20; k++) { txt = await gWrite([1], p.temp, 46); lens.push((txt.match(TOK) || []).length); if (syntaxOk(txt)) ok++; attempt(txt); api.checkTok(tok); }
        G.last = txt; G.gen = { sig: gptCorpusSig(), lens: lens, ok: ok };
      } catch (e) { if (!(e instanceof Cancel)) { bb.say('Error: ' + e.message); console.error(e); } }
      api.idle(); running = false; bb.btn.disabled = false; progress(1); gResultsBox(); L4.refresh();
    }
    bb.btn.addEventListener('click', train);
    L4.live = function(){ if (running) api.tok++; else render(); };
    return { render: render };
  } });

/* ---------- G1: text becomes tokens ---------- */
L4.add({ short: 'Tokens', title: 'A GPT reads a molecule as tokens',
  change: 'Press Next compound.',
  watch: 'A SMILES string encodes a molecule as text, for example CC(=O)N. The text is split into tokens, usually one atom or symbol each, and each token is assigned an integer ID. The set of all tokens is the vocabulary. The model receives only the ID numbers.',
  build: function(api){
    var cur = null;
    var sp = api.el('pn fd', 'left:28px;top:28px;width:250px;height:226px', '<h4>Compound drawn by RDKit</h4><div class="struct" style="height:186px"></div>');
    var tp = api.el('pn fd', 'left:298px;top:28px;width:634px;height:96px', '<h4>SMILES text</h4><div class="smi" style="min-height:20px"></div>');
    var kp = api.el('pn fd', 'left:298px;top:136px;width:634px;height:150px', '<h4>Tokens and their ID numbers</h4><div class="tk"></div>');
    var vp = api.el('pn fd', 'left:28px;top:298px;width:904px;height:138px', '<h4 class="vh">Vocabulary</h4><div class="vc" style="line-height:1.1"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    async function setup(next){ await loadStructs(); gptSetup(); if (!cur || next) { var a = STRUCTS.HLM.filter(function(e){ return e.smi.match(TOK).length <= 30; }); cur = a[Math.floor(Math.random() * a.length)]; } qs('.struct', sp).innerHTML = cur.svg; qs('.smi', tp).textContent = ''; var toks = cur.smi.match(TOK);
      qs('.tk', kp).innerHTML = '<div style="display:flex;flex-wrap:wrap;gap:4px">' + ['[start]'].concat(toks, ['[end]']).map(function(t, k){ var id = k === 0 ? 1 : (k === toks.length + 1 ? 2 : G.stoi[t]); return '<div class="tkc fl" style="text-align:center"><span class="tok ' + (k === 0 || k === toks.length + 1 ? 's' : '') + '" style="margin:0">' + esc(t) + '</span><div style="font:600 11px Cascadia Mono,Consolas,monospace;color:#34373F;margin-top:3px">' + (id === undefined ? '?' : id) + '</div></div>'; }).join('') + '</div>';
      var used = {}; toks.forEach(function(t){ used[t] = 1; }); qs('.vh', vp).textContent = 'Vocabulary: ' + G.V + ' tokens, taken from ' + G.seqs.length + ' training SMILES'; qs('.vc', vp).innerHTML = G.vocab.map(function(t, i){ return '<span style="display:inline-block;margin:2px 6px 2px 0;font:600 11.5px Cascadia Mono,Consolas,monospace;padding:3px 6px;border-radius:4px;border:1px solid ' + (used[t] ? C.red : C.silver) + ';background:' + (used[t] ? '#fff' : '#EDF2F4') + ';color:' + (used[t] ? C.red : C.mid) + '">' + esc(tokName(i)) + ' ' + i + '</span>'; }).join('');
      [sp, tp, kp, vp].forEach(function(e){ e.classList.remove('on'); }); qsa('.tkc', kp).forEach(function(e){ e.classList.remove('on'); }); }
    async function render(){ await setup(false); [sp, tp, kp, vp].forEach(function(e){ e.classList.add('on'); }); qs('.smi', tp).textContent = cur.smi; qsa('.tkc', kp).forEach(function(e){ e.classList.add('on'); }); }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ api.tok++; await setup(true); await render(); });
    return { render: render };
  } });

function sepRatio(xy){ var cats = G.vocab.map(function(t){ return tokCat(t); }), win = 0, nw = 0, bw = 0, nb = 0; for (var i = 3; i < xy.length; i++) for (var j = i + 1; j < xy.length; j++) { var d = Math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]); if (cats[i] === cats[j]) { win += d; nw++; } else { bw += d; nb++; } } return (win / Math.max(1, nw)) / ((bw / Math.max(1, nb)) || 1); }
/* ---------- G2: embeddings ---------- */
L4.add({ short: 'Embeddings', title: 'Each token becomes a list of numbers the model can adjust',
  change: 'Switch between Untrained and Trained below the plot.',
  watch: 'An embedding is the vector of numbers that represents one token. Here each token has 64 numbers. At the start they are random. Training adjusts them so that tokens used in similar contexts receive similar vectors. The map projects the 64 numbers onto two dimensions with principal component analysis (PCA), so that tokens with similar embeddings appear close together.',
  build: function(api){
    var ep = api.el('pn', 'left:28px;top:28px;width:420px;height:408px', '<h4>First 14 of the 64 numbers for each token</h4><div class="em"></div>');
    var mp = api.el('pn', 'left:468px;top:28px;width:464px;height:408px', '<h4 class="mt">Map of all tokens (PCA of their 64 numbers)</h4><canvas width="440" height="326"></canvas><div class="legend ml" style="font-size:11px;margin-top:4px"></div>');
    var ctl = api.addControl('<select><option value="0">Untrained</option><option value="1">Trained</option></select>'); var sel = qs('select', ctl);
    function matrix(trained){ gptSetup(); if (!G.P) gInit(+$('glayers').value); return trained && G.trained ? G.P.tok.arraySync() : G.emb0; }
    function pca(M){ var n = M.length, d = M[0].length, mu = new Array(d).fill(0); M.forEach(function(r){ r.forEach(function(v, j){ mu[j] += v / n; }); }); var X = M.map(function(r){ return r.map(function(v, j){ return v - mu[j]; }); }); var cov = []; for (var a = 0; a < d; a++) { cov.push(new Array(d).fill(0)); } X.forEach(function(r){ for (var a = 0; a < d; a++) for (var b = a; b < d; b++) cov[a][b] += r[a] * r[b] / n; }); for (var a2 = 0; a2 < d; a2++) for (var b2 = 0; b2 < a2; b2++) cov[a2][b2] = cov[b2][a2];
      var comps = []; for (var c = 0; c < 2; c++) { var v = Array.from({ length: d }, function(_, i){ return Math.sin(i * 1.7 + c); }); for (var it = 0; it < 60; it++) { var w = cov.map(function(row){ return row.reduce(function(s, x, j){ return s + x * v[j]; }, 0); }); comps.forEach(function(u){ var dp = w.reduce(function(s, x, j){ return s + x * u[j]; }, 0); w = w.map(function(x, j){ return x - dp * u[j]; }); }); var nrm = Math.sqrt(w.reduce(function(s, x){ return s + x * x; }, 0)) || 1; v = w.map(function(x){ return x / nrm; }); } comps.push(v); }
      return X.map(function(r){ return [r.reduce(function(s, x, j){ return s + x * comps[0][j]; }, 0), r.reduce(function(s, x, j){ return s + x * comps[1][j]; }, 0)]; }); }
    var curXY = null;
    function strips(trained){ var M = matrix(trained), toks = ['C', 'c', 'N', 'O', '(', ')', '1', '2', '=', 'Cl'].filter(function(t){ return G.stoi[t] !== undefined; }), mx = 1e-6; toks.forEach(function(t){ M[G.stoi[t]].slice(0, 14).forEach(function(v){ mx = Math.max(mx, Math.abs(v)); }); });
      qs('.em', ep).innerHTML = toks.map(function(t){ return '<div style="display:flex;align-items:center;gap:10px;margin:5px 0"><span class="tok" style="width:46px;text-align:center;margin:0">' + esc(t) + '</span><div style="display:flex;gap:2px">' + M[G.stoi[t]].slice(0, 14).map(function(v){ var a = Math.min(1, Math.abs(v) / mx); return '<span style="display:block;width:19px;height:26px;border-radius:3px;background:' + (v >= 0 ? C.red : C.char) + ';opacity:' + (0.1 + 0.9 * a).toFixed(2) + '"></span>'; }).join('') + '</div></div>'; }).join('') + '<p style="font-size:12px;color:#34373F;margin-top:10px">Red: positive number. Dark: negative. Stronger colour: larger magnitude.</p>'; }
    function mapDraw(xy, t0){ var g = cv(qs('canvas', mp)), x = g.x, W = g.w, H = g.h, mx = Math.max.apply(null, xy.map(function(p){ return Math.abs(p[0]); })) || 1, my = Math.max.apply(null, xy.map(function(p){ return Math.abs(p[1]); })) || 1; x.strokeStyle = C.silver; x.beginPath(); x.moveTo(W / 2, 8); x.lineTo(W / 2, H - 8); x.moveTo(8, H / 2); x.lineTo(W - 8, H / 2); x.stroke();
      var pts = []; xy.forEach(function(p, i){ var t = G.vocab[i]; if (i < 3) return; var cx = W / 2 + p[0] / mx * (W / 2 - 30), cy = H / 2 - p[1] / my * (H / 2 - 22), col = CATC[tokCat(t)]; x.fillStyle = col; x.beginPath(); x.arc(cx, cy, 5.5, 0, 6.3); x.fill(); pts.push({ t: t, cx: cx, cy: cy }); });
      var placed = []; x.font = '600 11px Cascadia Mono, Consolas, monospace'; pts.forEach(function(o){ var w = x.measureText(o.t).width, c = [[o.cx - w / 2, o.cy - 9], [o.cx - w / 2, o.cy + 19], [o.cx + 9, o.cy + 4], [o.cx - 9 - w, o.cy + 4]], pick = null; c.forEach(function(q){ if (pick) return; var r = { l: q[0] - 2, r: q[0] + w + 2, t: q[1] - 11, b: q[1] + 3 }; if (r.l < 2 || r.r > W - 2 || r.t < 2 || r.b > H - 2) return; if (placed.some(function(z){ return r.l < z.r && r.r > z.l && r.t < z.b && r.b > z.t; })) return; if (pts.some(function(z){ return z !== o && z.cx > r.l - 4 && z.cx < r.r + 4 && z.cy > r.t - 4 && z.cy < r.b + 4; })) return; pick = { q: q, r: r }; }); if (pick) { placed.push(pick.r); x.fillStyle = C.char; x.textAlign = 'left'; x.fillText(o.t, pick.q[0], pick.q[1]); } }); }
    function legend(){ qs('.ml', mp).innerHTML = Object.keys(CATC).filter(function(k){ return k !== 'special'; }).map(function(k){ return '<span style="margin-right:10px"><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:' + CATC[k] + ';margin-right:4px"></i>' + k + '</span>'; }).join(''); }
    function show(trained){ strips(trained); curXY = pca(matrix(trained)); mapDraw(curXY); legend(); qs('.mt', mp).textContent = trained && G.trained ? 'Map of all tokens after training' : 'Map of all tokens before training'; }
    sel.addEventListener('change', function(){ api.tok++; show(sel.value === '1'); });
    function render(){ if (typeof tf === 'undefined') { api.gate(NOTF); return; } gptSetup(); if (!G.P) gInit(+$('glayers').value); show(sel.value === '1' && G.trained); }
    return { render: render };
  } });

/* ---------- G3: the transformer stack ---------- */
L4.add({ short: 'The stack', title: 'The model is a stack of blocks',
  change: 'Drag Transformer layers in the options panel. Press Walk through.',
  watch: 'A GPT is a stack of identical transformer blocks, read here from the bottom. Tokens enter as embeddings. In each block every token first attends to the earlier tokens (attention) and is then passed through a small feed-forward network. The output layer returns a probability for each token in the vocabulary. Parameters are all the weights and biases the model learns. Each additional block adds the same number.',
  build: function(api){
    var dp = api.el('pn', 'left:28px;top:28px;width:904px;height:410px', '<svg viewBox="0 0 880 384" width="880" height="384"></svg>');
    function per(){ return DM * 3 * DM + 3 * DM + DM * DM + DM + DM * 4 * DM + 4 * DM + 4 * DM * DM + DM + 4 * DM; }
    function draw(active){
      gptSetup(); var N = GP().nl, V = G.V, s = '', H = 384, items = [{ k: 'in', t: 'Token IDs and positions', u: 'the text so far, as numbers' }, { k: 'emb', t: 'Embeddings', u: V + ' tokens, ' + DM + ' numbers each, plus position' }];
      for (var b = 0; b < N; b++) { items.push({ k: 'at' + b, t: 'Block ' + (b + 1) + ': attention', u: HEADS + ' heads, each token attends to earlier tokens' }); items.push({ k: 'ff' + b, t: 'Block ' + (b + 1) + ': feed-forward', u: DM + ' to ' + 4 * DM + ' to ' + DM + ' numbers, per token' }); }
      items.push({ k: 'out', t: 'Next-token probabilities', u: V + ' probabilities' });
      var n = items.length, gap = 14, bh = Math.min(48, Math.floor((H - 8) / n) - gap), total = V * DM + TLEN * DM + N * per() + 2 * DM + DM * V, x = 40, w = 380;
      items.forEach(function(it, i){
        var y = H - 4 - (i + 1) * bh - i * gap, hot = active === it.k;
        s += '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + bh + '" rx="6" fill="' + (hot ? '#EDF2F4' : '#fff') + '" stroke="' + (hot ? C.red : C.silver) + '" stroke-width="' + (hot ? 2.5 : 1) + '"/><text x="' + (x + 12) + '" y="' + (y + 17) + '" font-size="13" font-weight="700" fill="' + C.char + '">' + it.t + '</text>' + (bh >= 34 ? '<text x="' + (x + 12) + '" y="' + (y + 32) + '" font-size="11" fill="' + C.mid + '">' + it.u + '</text>' : '');
        if (i < n - 1) s += '<path d="M' + (x + w / 2) + ' ' + (y - 1) + 'L' + (x + w / 2) + ' ' + (y - gap + 4) + '" stroke="' + C.red + '" stroke-width="2"/><path d="M' + (x + w / 2 - 5) + ' ' + (y - 1 - 5) + 'L' + (x + w / 2) + ' ' + (y - 1) + 'L' + (x + w / 2 + 5) + ' ' + (y - 1 - 5) + '" stroke="' + C.red + '" stroke-width="2" fill="none"/>';
      });
      var px = 470; s += '<text x="' + px + '" y="30" font-size="12" font-weight="700" fill="' + C.char + '">This design</text>';
      [['Transformer layers', N], ['Attention heads per layer', HEADS], ['Numbers per token', DM], ['Vocabulary', V], ['Parameters in total', total.toLocaleString()]].forEach(function(f, i){ s += '<text x="' + px + '" y="' + (62 + i * 28) + '" font-size="13" fill="' + C.mid + '">' + f[0] + '</text><text x="' + (px + 300) + '" y="' + (62 + i * 28) + '" font-size="14" font-weight="700" fill="' + (i === 4 ? C.red : C.char) + '" text-anchor="end">' + f[1] + '</text><line x1="' + px + '" y1="' + (69 + i * 28) + '" x2="' + (px + 300) + '" y2="' + (69 + i * 28) + '" stroke="' + C.silver + '"/>'; });
      var ty = 236; s += '<text x="' + px + '" y="' + ty + '" font-size="12" font-weight="700" fill="' + C.char + '">What each part does</text>';
      [['Attention', 'Each token weights the earlier tokens by relevance and combines their values.'], ['Feed-forward', 'A small network is then applied to each token separately.'], ['Repeat', 'Later blocks operate on the outputs of earlier blocks.']].forEach(function(r, i){ s += '<text x="' + px + '" y="' + (ty + 28 + i * 44) + '" font-size="12.5" font-weight="700" fill="' + C.red + '">' + r[0] + '</text><text x="' + px + '" y="' + (ty + 45 + i * 44) + '" font-size="12" fill="' + C.char + '">' + r[1] + '</text>'; });
      qs('svg', dp).innerHTML = s;
    }
    function render(){ draw(''); }
    async function play(tok){ var N = GP().nl; draw('in'); await api.beat('The text so far enters as token IDs, each with a position.', 2600); draw('emb'); await api.beat('Embeddings map each ID to a vector of 64 numbers.', 2600);
      for (var b = 0; b < N; b++) { draw('at' + b); await api.beat('Block ' + (b + 1) + ' attention: each token weights the earlier tokens by relevance.', 3000); draw('ff' + b); await api.beat('Block ' + (b + 1) + ' feed-forward: a small network is applied to each token separately.', 2600); }
      draw('out'); await api.beat('The output layer converts the last position into a probability for every token in the vocabulary.', 3000); draw(''); }
    return { render: render, play: play };
  } });

L4.add({ short: 'Attention', title: 'Attention shows which earlier tokens a token looks at',
  change: 'Click any token. Change the layer or head below the plot.',
  watch: 'Attention weights each earlier token when the model computes a representation of the selected token. The bars show these weights, which sum to 100%. Later tokens are greyed out because they are masked: the model may use only earlier tokens (the causal mask). A head is one of several attention patterns computed in parallel in a layer.',
  build: function(api){
    var hp = api.el('pn', 'left:28px;top:28px;width:904px;height:330px;overflow:hidden', '<div class="ac" style="display:flex;flex-wrap:wrap;gap:6px 5px"></div>');
    var tp = api.el('pn', 'left:28px;top:370px;width:904px;height:70px;display:flex;align-items:center;font-size:14px;line-height:1.45', '<div class="an"></div>');
    var ctl = api.addControl('Molecule <select class="ms"></select> Layer <select class="ls"></select> Head <select class="hs"></select>'); var ms = qs('.ms', ctl), ls = qs('.ls', ctl), hs = qs('.hs', ctl), data = null, sel = -1;
    function fillSel(){ gptSetup(); if (!ms.options.length) { G.smiles.filter(function(s){ return s.match(TOK).length <= 22; }).slice(0, 8).forEach(function(s){ var o = document.createElement('option'); o.value = s; o.textContent = (s.length > 24 ? s.slice(0, 24) + '...' : s); ms.appendChild(o); }); } var nl = G.NL; ls.innerHTML = Array.from({ length: nl }, function(_, i){ return '<option value="' + i + '">' + (i + 1) + '</option>'; }).join(''); hs.innerHTML = '<option value="avg">average</option>' + Array.from({ length: HEADS }, function(_, i){ return '<option value="' + i + '">' + (i + 1) + '</option>'; }).join(''); ls.value = nl - 1; }
    function compute(){ var t = ms.value.match(TOK), ids = [1].concat(t.map(function(x){ return G.stoi[x] === undefined ? 0 : G.stoi[x]; })), labels = ['start'].concat(t), A; tf.tidy(function(){ var o = gforward(tf.tensor2d([ids], null, 'int32'), G.NL, true), a = o.atts[+ls.value]; A = (hs.value === 'avg' ? a.mean(1) : a.slice([0, +hs.value, 0, 0], [1, 1, ids.length, ids.length]).reshape([1, ids.length, ids.length])).arraySync()[0]; }); data = { A: A, labels: labels, n: ids.length }; if (sel < 0 || sel >= data.n) sel = data.n - 1; }
    function draw(){
      if (!data) return; var n = data.n, row = data.A[sel], BH = 90, cells = '', mx = 1e-6; for (var q = 0; q <= sel; q++) if (q !== sel || sel === 0) mx = Math.max(mx, row[q]);
      for (var j = 0; j < n; j++) { var fut = j > sel, w = fut ? 0 : row[j], h = Math.round(w / mx * BH), top = j === sel;
        cells += '<div data-j="' + j + '" style="width:56px;text-align:center;cursor:pointer;opacity:' + (fut ? 0.35 : 1) + '"><div style="height:' + (BH + 16) + 'px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center"><span style="font-size:11px;color:' + C.char + ';height:14px">' + (!fut && w >= 0.03 ? Math.round(w * 100) + '%' : '') + '</span><div style="width:30px;height:' + h + 'px;background:' + C.red + ';border-radius:3px 3px 0 0;opacity:' + (0.4 + 0.6 * w / mx).toFixed(2) + '"></div></div><div class="tok" style="animation:none;margin:3px 0 0;padding:6px 3px;width:50px;text-align:center;' + (top ? 'background:' + C.red + ';color:#fff;border-color:' + C.red : (j === 0 ? 'background:' + C.char + ';color:#fff;border-color:' + C.char : '')) + '">' + esc(data.labels[j]) + '</div></div>'; }
      qs('.ac', hp).innerHTML = cells; qsa('[data-j]', hp).forEach(function(d){ d.addEventListener('click', function(){ sel = +d.dataset.j; draw(); }); });
      var top3 = []; for (var k = 0; k < sel; k++) top3.push([row[k], k]); top3.sort(function(a, b){ return b[0] - a[0]; });
      qs('.an', tp).innerHTML = sel === 0 ? 'The first token has no earlier token to attend to. Select a later token.' : 'The selected token <b>' + esc(data.labels[sel]) + '</b> attends mostly to ' + top3.slice(0, 3).map(function(t){ return '<b>' + esc(data.labels[t[1]]) + '</b> (' + Math.round(t[0] * 100) + '%)'; }).join(', ') + '. The bars add up to 100%. Layer ' + (+ls.value + 1) + ', head ' + (hs.value === 'avg' ? 'average of ' + HEADS : (+hs.value + 1)) + '.';
    }
    [ms, ls, hs].forEach(function(e){ e.addEventListener('change', function(){ api.tok++; if (e === ms) sel = -1; compute(); draw(); }); });
    async function ready(){ gptSetup(); if (!(await gReady(api))) return false; if (!ms.options.length || +ls.options.length !== G.NL) fillSel(); compute(); return true; }
    async function render(){ if (!(await ready())) return; draw(); }
    return { render: render };
  } });

/* ---------- G6: next token and temperature ---------- */
L4.add({ short: 'Next token', title: 'The model outputs odds, and temperature reshapes them',
  change: 'Drag Temperature below the plot, or press Auto-sample.',
  watch: 'At each step the model assigns a probability to every possible next token. Temperature divides the raw scores (logits) before the softmax converts them to probabilities. A low temperature concentrates probability on the most likely token, which gives repetitive output. A high temperature flattens the distribution, which gives more varied output and more invalid SMILES. Auto-sample samples one token at a time from the distribution.',
  build: function(api){
    var ip = api.el('pn', 'left:28px;top:28px;width:440px;height:170px', '<h4>SMILES so far</h4><input type="text" class="pf" spellcheck="false" style="font-family:Cascadia Mono,Consolas,monospace" value="CC(=O)Nc1ccc"><div class="pr" style="margin-top:10px"></div>');
    var wp = api.el('pn', 'left:28px;top:210px;width:440px;height:226px', '<h4>Sampled output</h4><div class="wt smi" style="min-height:60px;background:#34373F;color:#fff;border-radius:6px;padding:10px;font-size:12.5px"></div><div class="note" style="font-size:12px;color:#34373F;margin-top:8px">Press Auto-sample to add tokens one at a time, each sampled from the probabilities on the right.</div>');
    var bp = api.el('pn', 'left:488px;top:28px;width:444px;height:408px', '<h4 class="bh">Probability of each next token</h4><div class="pb probs"></div><div class="en" style="font-size:12.5px;margin-top:12px"></div>');
    var ctl = api.addControl('Temperature <input type="range" min="0.3" max="1.6" step="0.1" value="0.8" style="width:160px"><b class="tv">0.8</b> <button type="button" class="btn sm" data-auto>Auto-sample</button>'); var tr = qs('input', ctl), writing = false;
    function ids(){ var t = qs('.pf', ip).value.trim().match(TOK) || [], out = [1]; for (var i = 0; i < t.length; i++) { if (G.stoi[t[i]] === undefined) return { err: t[i] }; out.push(G.stoi[t[i]]); } return { ids: out }; }
    function showPrefix(arr, newIdx){ qs('.pr', ip).innerHTML = arr.map(function(i, k){ return '<span class="tok ' + (k === 0 ? 's' : (k === newIdx ? 'new' : '')) + '" style="animation:none">' + esc(tokName(i)) + '</span>'; }).join(''); }
    function bars(p, picked){ var top = Array.from(p).map(function(v, i){ return [v, i]; }).sort(function(a, b){ return b[0] - a[0]; }).slice(0, 10), ent = -Array.from(p).reduce(function(s, v){ return s + (v > 1e-9 ? v * Math.log(v) : 0); }, 0);
      qs('.pb', bp).innerHTML = top.map(function(e){ return '<div class="' + (e[1] === picked ? 'pick' : '') + '"><span>' + esc(tokName(e[1])) + '</span><span class="tr"><i style="width:' + (e[0] * 100).toFixed(1) + '%"></i></span><span>' + Math.round(e[0] * 100) + '%</span></div>'; }).join('');
      qs('.en', bp).innerHTML = 'Spread of the distribution (entropy) <b>' + ent.toFixed(2) + '</b>. Zero means one certain token. Higher means more uncertainty.<br>Temperature ' + (+tr.value).toFixed(1) + ': ' + (+tr.value < 0.6 ? 'the most likely token dominates.' : (+tr.value > 1.1 ? 'low-probability tokens are sampled regularly.' : 'moderate diversity.')); }
    function update(){ var r = ids(); if (r.err) { qs('.pr', ip).innerHTML = '<span class="note" style="color:' + C.red + '">Token ' + esc(r.err) + ' is not in the vocabulary.</span>'; return null; } showPrefix(r.ids); if (!G.trained) { qs('.pb', bp).innerHTML = '<span class="note">Train the GPT in the workbench first.</span>'; return r; } var p = gDist(r.ids, +tr.value); bars(p); return r; }
    tr.addEventListener('input', function(){ api.tok++; qs('.tv', ctl).textContent = tr.value; $('gtemp').value = tr.value; $('o-gtemp').textContent = tr.value; update(); }); qs('.pf', ip).addEventListener('input', function(){ api.tok++; update(); });
    qs('[data-auto]', ctl).addEventListener('click', async function(){ if (writing || !G.trained) return; writing = true; var btn = this; btn.disabled = true; var r = ids(); if (r.err) { writing = false; btn.disabled = false; return; } var arr = r.ids.slice(), tok = ++api.tok; try { for (var t = arr.length; t < TLEN; t++) { api.checkTok(tok); var p = gDist(arr, +tr.value), nx = gSample(p); bars(p, nx); await api.sleep(700); if (nx === 2) { qs('.wt', wp).textContent += '  [end]'; break; } arr.push(nx); showPrefix(arr, arr.length - 1); qs('.wt', wp).textContent = arr.slice(1).map(function(i){ return G.vocab[i]; }).join(''); await api.sleep(250); } } catch (e) { if (!(e instanceof Cancel)) console.error(e); } writing = false; btn.disabled = false; });
    async function ready(){ gptSetup(); if (!(await gReady(api))) return false; tr.value = $('gtemp').value; qs('.tv', ctl).textContent = tr.value; return true; }
    async function render(){ if (!(await ready())) return; update(); }
    return { render: render };
  } });

/* ---------- G7: generate and check with RDKit ---------- */
L4.add({ short: 'Generate and check', title: 'Generate molecules and let RDKit check them',
  change: 'Press Generate below the plot.',
  watch: 'Generate samples molecules from the GPT one token at a time at the chosen temperature. Each card is one SMILES string. RDKit parses it and draws the structure if it is valid, and valid molecules have a red outline. The GPT is trained on text patterns only, so some output is not a valid molecule.',
  auto: false,
  build: function(api){
    var gp = api.el('pn', 'left:28px;top:28px;width:640px;height:408px', '<h4>Twelve generated molecules</h4><div class="mol3 gridm" style="margin-top:4px"></div>');
    var sp = api.el('pn', 'left:688px;top:28px;width:244px;height:408px', '<h4>Quality at this temperature</h4><div class="sq" style="font-size:13px;color:#34373F">Press Generate.</div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-go>Generate</button>'); var busy = false;
    async function go(){
      if (busy) return; busy = true; var btn = qs('[data-go]', ctl); btn.disabled = true; var sq = qs('.sq', sp), temp = GP().temp;
      try { if (!(await gReady(api))) { busy = false; btn.disabled = false; return; } sq.textContent = RDK ? 'Generating' : 'Loading RDKit (about 7 MB)'; progress(0.2); await loadRDKit(); progress(0.4); var out = []; for (var k = 0; k < 12; k++) { out.push(await gWrite([1], temp, 46)); qs('.gridm', gp).innerHTML = out.map(function(s, i){ return '<div class="molc" style="min-height:92px;font-size:10px;word-break:break-all;color:#34373F">' + esc(s) + '</div>'; }).join(''); }
        var valid = out.map(function(s){ return rdkValid(s); }), seen = {}; G.smiles.forEach(function(s){ seen[s] = 1; });
        qs('.gridm', gp).innerHTML = out.map(function(s, i){ var svg = valid[i] ? rdkSvg(s, 200, 130) : null; return '<div class="molc ' + (valid[i] ? 'ok' : 'bad') + '" style="animation-delay:' + (ANIM ? i * 80 : 0) + 'ms"><div class="struct" style="height:84px">' + (svg || '<div style="height:84px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#34373F;padding:4px;word-break:break-all">' + esc(s.slice(0, 40)) + '</div>') + '</div><small>' + (valid[i] ? 'valid' : 'RDKit rejects it') + '</small></div>'; }).join('');
        var nv = valid.filter(Boolean).length, uniq = {}; out.forEach(function(s, i){ if (valid[i]) uniq[s] = 1; }); var nu = Object.keys(uniq).length, nn = Object.keys(uniq).filter(function(s){ return !seen[s]; }).length;
        sq.innerHTML = '<div class="nums" style="font-size:13px"><div><span>Temperature</span><b>' + temp.toFixed(1) + '</b></div><div><span>Valid by RDKit</span><b style="color:' + C.red + '">' + nv + ' of 12</b></div><div><span>Passing the simple bracket check</span><b>' + out.filter(syntaxOk).length + ' of 12</b></div><div><span>Unique among valid</span><b>' + nu + '</b></div><div><span>New (not in training)</span><b>' + nn + '</b></div></div><p style="font-size:12px;margin-top:12px;color:#34373F">A small model trained for few steps produces text that resembles SMILES but is rarely valid. More training steps raise the valid fraction. Valence, aromaticity and stereochemistry are checked by RDKit and are not guaranteed by the GPT.</p>'; progress(1); }
      catch (e) { sq.innerHTML = '<span style="color:' + C.red + '">' + esc(e.message) + '</span>'; console.error(e); }
      busy = false; btn.disabled = false; }
    qs('[data-go]', ctl).addEventListener('click', function(){ api.tok++; api.clearTexts(); go().then(function(){ api.cap('A string is valid only if RDKit can parse it. The GPT is trained on text patterns only.'); }); });
    return { render: function(){}, play: undefined };
  } });

L4.glossary([
  ['GPT', 'A model trained to predict the next token, then used to generate text one token at a time.'], ['Token', 'A unit of text. Here: an atom, bond symbol, bracket or ring-closure digit.'],
  ['Vocabulary', 'The set of tokens, each with an integer ID.'], ['Embedding', 'The vector of numbers that represents one token. Training adjusts it.'],
  ['Attention', 'Each token weighs earlier tokens by how relevant they are.'], ['Head', 'One of several attention patterns computed in parallel inside a layer.'],
  ['Transformer', 'A stack of attention and feed-forward blocks.'], ['Causal mask', 'Stops a token looking at later tokens, so the model only uses the past.'],
  ['Pre-training', 'Learning general patterns before any specific task. No clearance values are used here.'], ['Next-token loss', 'The negative log probability the model assigns to the correct next token.'],
  ['Temperature', 'Divides the logits before the softmax. Low values favour the most probable token. High values give more varied samples.'], ['Entropy', 'A measure of how spread out the probabilities are.'],
  ['SMILES', 'Text for a molecule, such as CC(=O)N.'], ['PCA', 'Principal component analysis. Projects vectors onto the directions of largest variance, here to a 2D map.'],
  ['Valid molecule', 'A SMILES string that RDKit can read as a chemically sensible structure.']
]);

L4.add(filterStepDef('gpt'));

/* ---------- test it: can a GPT design molecules you can trust? ---------- */
var GV = { key: null, res: null, corpus: null };
L4.add({ short: 'Is it useful', uses: ['gtemp', 'glayers', 'gsteps'],
  title: 'Test it: can a GPT design molecules?',
  watch: 'This test samples about 90 molecules, 30 at each of three temperatures. Valid means RDKit accepts the string as a molecule. New means the molecule is not in the training set, so the GPT did not reproduce a training structure. A useful design model needs both.',
  change: 'Press Run test below the plot.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px', '<h4>Fraction of samples, by temperature</h4><canvas width="396" height="300"></canvas><div class="note" style="font-size:12px;margin-top:4px">Grey: accepted by RDKit. Red: accepted and new.</div>');
    var gp = api.el('pn', 'left:468px;top:28px;width:464px;height:250px;overflow:hidden', '<h4>Valid molecules sampled</h4><div class="gm" style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px"></div>');
    var sp = api.el('pn', 'left:468px;top:290px;width:464px;height:146px;overflow:auto', '<div class="vs" style="font-size:12px;line-height:1.35"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-run>Run test</button>'), busy = false, TEMPS = [0.4, 0.8, 1.2], PER = 30;
    function sig(){ return JSON.stringify([G.sig, G.params, G.demo ? 'demo' : G.secs]); }
    function show(r){
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 40, r: g.w - 8, t: 16, b: g.h - 44 }, gw = (box.r - box.l) / TEMPS.length;
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; x.font = '11px Aptos, Segoe UI, sans-serif'; for (var k = 0; k <= 4; k++) { var y = box.b - k / 4 * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText((k * 25) + '%', box.l - 5, y + 4); }
      r.rows.forEach(function(w, i){ var bx = box.l + i * gw + gw * 0.14, bw = gw * 0.34;[[w.valid / w.n, C.bg], [w.novel / w.n, C.red]].forEach(function(b, j){ var hh = b[0] * (box.b - box.t); x.fillStyle = b[1]; x.fillRect(bx + j * (bw + 4), box.b - hh, bw, hh); x.fillStyle = C.char; x.font = '700 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText(Math.round(b[0] * 100) + '%', bx + j * (bw + 4) + bw / 2, box.b - hh - 5); }); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText('temperature ' + w.t, box.l + i * gw + gw / 2, box.b + 18); });
      qs('.gm', gp).innerHTML = r.samples.map(function(s){ return '<div class="molc ok" style="padding:2px"><div class="struct" style="height:70px">' + s + '</div></div>'; }).join('') || '<span class="note">No valid molecule yet. Train the GPT for more steps.</span>';
      var mid = r.rows[1];
      qs('.vs', sp).innerHTML = '<p style="margin:0 0 5px">At temperature ' + mid.t + ', RDKit accepted <b>' + mid.valid + ' of ' + mid.n + '</b> and <b>' + mid.novel + '</b> were new. Every one still needs checks.</p><div class="lbl" style="color:#450A0A">Use a GPT when</div> you want many new structures and can filter them. <div class="lbl" style="color:' + C.red + ';margin-top:3px">Be careful when</div> you need every output valid, or a predicted value.';
    }
    async function run(){
      if (busy) return; if (!(await gReady(api))) return; if (GV.res && GV.key === sig()) { show(GV.res); return; } busy = true; qs('[data-run]', ctl).disabled = true;
      try {
        api.busy('Loading RDKit'); await loadRDKit(); gptSetup(); if (!GV.corpus) { GV.corpus = {}; G.smiles.forEach(function(s){ var m = RDK.get_mol(s); if (m && m.is_valid()) GV.corpus[m.get_smiles()] = 1; if (m) m.delete(); }); }
        var rows = [], samples = [];
        for (var t = 0; t < TEMPS.length; t++) { var seen = {}, valid = 0, novel = 0; for (var k = 0; k < PER; k++) { api.busy('Sampling molecules at temperature ' + TEMPS[t] + ': ' + (k + 1) + ' of ' + PER); var txt = await gWrite([1], TEMPS[t], 46), m = RDK.get_mol(txt); if (m && m.is_valid()) { var can = m.get_smiles(); valid++; if (!GV.corpus[can] && !seen[can]) novel++; seen[can] = 1; if (t === 1 && samples.length < 6) { var svg = rdkSvg(txt, 200, 90); if (svg) samples.push(svg); } } if (m) m.delete(); } rows.push({ t: TEMPS[t], n: PER, valid: valid, novel: novel }); }
        GV.res = { rows: rows, samples: samples }; GV.key = sig(); api.idle(); show(GV.res);
      } catch (e) { console.error(e); api.idle(); qs('.vs', sp).innerHTML = '<span style="color:' + C.red + '">' + esc(e.message) + '</span>'; }
      busy = false; qs('[data-run]', ctl).disabled = false;
    }
    qs('[data-run]', ctl).addEventListener('click', run);
    function render(){ if (GV.res && GV.key === sig()) show(GV.res); else { cv(qs('canvas', cp)); qs('.gm', gp).innerHTML = ''; qs('.vs', sp).innerHTML = '<span class="note">Press Run test to sample molecules from the GPT and check them with RDKit.</span>'; } }
    return { render: render };
  } });

L4.move('Workbench', 3);
setUses(L4, [[], ['glayers', 'gsteps'], ['glayers'], null, ['glayers', 'gsteps'], ['gtemp', 'glayers', 'gsteps'], ['gtemp', 'glayers', 'gsteps'], null, null]);
