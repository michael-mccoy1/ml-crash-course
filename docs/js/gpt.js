"use strict";
/* Lesson 5: GPT. A tiny decoder-only transformer trained on SMILES in the browser. */
var L4 = new Lesson('gpt', 'GPT');
L4.optsHTML = function(){
  return optRange('glayers', 'Transformer layers', 1, 3, 1, 2, 'Attention and feed-forward blocks stacked.') +
    optRange('gsteps', 'Training steps', 100, 1500, 50, 250, 'Mini-batches of 32 SMILES.') +
    optRange('gtemp', 'Temperature', 0.3, 1.6, 0.1, 0.8, 'Low is safe, high is varied.') +
    '<div class="og"><button type="button" class="btn wide" id="gtrain">Train GPT</button><div class="status" id="gst"></div><div class="rbox" id="gres"></div><div class="hint">The GPT always reads the Biogen SMILES, whichever endpoint is selected.</div></div>';
};
var TOK = /\[[^\]]+\]|Br|Cl|%\d\d|./g, TLEN = 48, DM = 64, HEADS = 4;
var GLIVE = [], GA = { promise: null };
var G = { P: null, NL: 2, vocab: null, stoi: null, V: 0, seqs: null, trained: false, losses: [], secs: null, steps: 0, params: 0, busy: false, emb0: null, readIds: null, sig: null };
function GP(){ return { nl: +$('glayers').value, steps: +$('gsteps').value, temp: +$('gtemp').value }; }
function gptSetup(){
  if (G.vocab) return; var sm = (DATA.HLM && DATA.HLM.smiles) || []; var toks = sm.map(function(s){ return s.match(TOK); }).filter(function(t){ return t.length <= TLEN - 2; });
  var set = {}; toks.forEach(function(t){ t.forEach(function(x){ set[x] = 1; }); }); G.vocab = ['<pad>', '<bos>', '<eos>'].concat(Object.keys(set).sort()); G.stoi = {}; G.vocab.forEach(function(t, i){ G.stoi[t] = i; }); G.V = G.vocab.length;
  G.seqs = toks.map(function(t){ var ids = [1].concat(t.map(function(x){ return G.stoi[x]; }), [2]); while (ids.length < TLEN + 1) ids.push(0); return ids; }); G.smiles = sm.filter(function(s){ return s.match(TOK).length <= TLEN - 2; });
}
function tokName(i){ return i === 0 ? '[pad]' : (i === 1 ? '[start]' : (i === 2 ? '[end]' : G.vocab[i])); }
function gInit(nl){
  gptSetup(); if (G.P) Object.keys(G.P).forEach(function(k){ G.P[k].dispose(); }); G.NL = nl; var V = G.V;
  var nrm = function(s){ return tf.variable(tf.randomNormal(s, 0, 0.02)); }, one = function(n){ return tf.variable(tf.ones([n])); }, zero = function(n){ return tf.variable(tf.zeros([n])); };
  var P = { tok: nrm([V, DM]), pos: nrm([TLEN, DM]), lnfg: one(DM), lnfb: zero(DM), Wout: nrm([DM, V]) };
  for (var l = 0; l < nl; l++) { P['g1_' + l] = one(DM); P['b1_' + l] = zero(DM); P['Wqkv' + l] = nrm([DM, 3 * DM]); P['bqkv' + l] = zero(3 * DM); P['Wo' + l] = nrm([DM, DM]); P['bo' + l] = zero(DM); P['g2_' + l] = one(DM); P['b2_' + l] = zero(DM); P['W1' + l] = nrm([DM, 4 * DM]); P['c1' + l] = zero(4 * DM); P['W2' + l] = nrm([4 * DM, DM]); P['c2' + l] = zero(DM); }
  G.P = P; G.trained = false; G.losses = []; G.emb0 = P.tok.arraySync(); G.params = Object.keys(P).reduce(function(a, k){ return a + P[k].size; }, 0);
}
function glin(x, W, b){ var s = x.shape, y = x.reshape([-1, s[s.length - 1]]).matMul(W); if (b) y = y.add(b); return y.reshape(s.slice(0, -1).concat([W.shape[1]])); }
function gln(x, g, b){ var mm = tf.moments(x, -1, true); return x.sub(mm.mean).div(mm.variance.add(1e-5).sqrt()).mul(g).add(b); }
function gforward(ids, nl, wantAtt){
  var P = G.P, B = ids.shape[0], T = ids.shape[1], dh = DM / HEADS, atts = [], x = tf.gather(P.tok, ids.flatten()).reshape([B, T, DM]).add(P.pos.slice([0, 0], [T, DM])), mask = tf.linalg.bandPart(tf.ones([T, T]), -1, 0).sub(1).mul(1e9);
  for (var l = 0; l < nl; l++) { var h = gln(x, P['g1_' + l], P['b1_' + l]), qkv = glin(h, P['Wqkv' + l], P['bqkv' + l]), part = function(k){ return qkv.slice([0, 0, k * DM], [B, T, DM]).reshape([B, T, HEADS, dh]).transpose([0, 2, 1, 3]); };
    var q = part(0), k = part(1), v = part(2), a = tf.softmax(q.matMul(k, false, true).div(Math.sqrt(dh)).add(mask)); if (wantAtt) atts.push(a);
    x = x.add(glin(a.matMul(v).transpose([0, 2, 1, 3]).reshape([B, T, DM]), P['Wo' + l], P['bo' + l])); var h2 = gln(x, P['g2_' + l], P['b2_' + l]); x = x.add(glin(tf.relu(glin(h2, P['W1' + l], P['c1' + l])), P['W2' + l], P['c2' + l])); }
  var logits = glin(gln(x, P.lnfg, P.lnfb), P.Wout, null); return wantAtt ? { logits: logits, atts: atts } : logits;
}
function gLoss(batch, nl){ var V = G.V, inp = tf.tensor2d(batch.map(function(s){ return s.slice(0, TLEN); }), null, 'int32'), tgt = tf.tensor2d(batch.map(function(s){ return s.slice(1, TLEN + 1); }), null, 'int32'), lp = tf.logSoftmax(gforward(inp, nl)), nll = tf.neg(lp.mul(tf.oneHot(tgt.flatten(), V).reshape([batch.length, TLEN, V])).sum(-1)), m = tf.notEqual(tgt, 0).cast('float32'); return nll.mul(m).sum().div(m.sum()); }
function gDist(ids, temp){ var ban = tf.tensor1d(G.vocab.map(function(_, i){ return i === 0 || i === 1 ? -1e9 : 0; })); var out = tf.tidy(function(){ var lg = gforward(tf.tensor2d([ids], null, 'int32'), G.NL).slice([0, ids.length - 1, 0], [1, 1, G.V]).reshape([G.V]).div(temp).add(ban); return tf.softmax(lg).dataSync(); }); ban.dispose(); return out; }
function gSample(p){ var r = Math.random(), c = 0; for (var i = 0; i < p.length; i++) { c += p[i]; if (r <= c) return i; } return p.length - 1; }
async function gWrite(prefix, temp, maxTok){ var ids = prefix.slice(); for (var t = ids.length; t < maxTok; t++) { var nx = gSample(gDist(ids, temp)); if (nx === 2) break; ids.push(nx); if (t % 10 === 9) await tick(); } return ids.slice(1).map(function(i){ return G.vocab[i]; }).join(''); }
function syntaxOk(s){ if (!s) return false; var depth = 0, rings = {}, t = s.match(TOK) || []; for (var i = 0; i < t.length; i++) { var x = t[i]; if (x === '(') depth++; else if (x === ')') { depth--; if (depth < 0) return false; } else if (x[0] === '[') { if (x.slice(-1) !== ']') return false; } else if (/^\d$/.test(x) || /^%\d\d$/.test(x)) rings[x] = (rings[x] || 0) + 1; } if (depth !== 0) return false; for (var k in rings) if (rings[k] % 2) return false; return !/[=#\-(\/\\]/.test(t[t.length - 1]) && !/^[=#)]/.test(t[0]); }
function tokCat(t){ if (/^[A-Za-z]/.test(t) && t[0] !== '[') return 'atom'; if (t[0] === '[') return 'bracket'; if (/^\d$|^%/.test(t)) return 'ring'; if (t === '(' || t === ')') return 'branch'; if (t === '<pad>' || t === '<bos>' || t === '<eos>') return 'special'; return 'bond'; }
var CATC = { atom: C.red, ring: C.char, branch: C.bg, bond: C.dred, bracket: '#9a7400', special: C.mid };

/* training */
async function gTrain(o){
  while (G.busy) await new Promise(function(r){ setTimeout(r, 250); }); if (G.trained && !gStale() && !(o && o.force)) return G;
  gptSetup(); var p = GP(); gInit(p.nl); var opt = tf.train.adam(0.003), vars = Object.keys(G.P).map(function(k){ return G.P[k]; }), r = rng(5), gT = 0, steps = p.steps; G.busy = true; G.losses = [];
  try { for (var s = 0; s < steps; s++) { var batch = []; for (var b = 0; b < 32; b++) batch.push(G.seqs[Math.floor(r() * G.seqs.length)]); var tm = performance.now(), c = opt.minimize(function(){ return gLoss(batch, G.NL); }, true, vars); G.losses.push(c.dataSync()[0]); c.dispose(); gT += performance.now() - tm; GLIVE.forEach(function(fn){ try { fn(s, steps); } catch (e) { console.error(e); } }); if (o && o.onStep) await o.onStep(s, steps); if (s % 6 === 5) await tick(); }
    G.trained = true; G.secs = gT / 1000; G.steps = steps; G.sig = JSON.stringify([p.nl, steps]); } finally { opt.dispose(); G.busy = false; }
  gResultsBox(); return G;
}
function gResultsBox(){ var el = $('gres'); if (!el) return; if (!G.trained) { el.innerHTML = ''; return; } el.innerHTML = '<div class="rrow">Final loss<b>' + G.losses[G.losses.length - 1].toFixed(2) + '</b></div><div class="rrow">Parameters<b>' + G.params.toLocaleString() + '</b></div><div class="rrow">Training time<b>' + fmtT(G.secs) + '</b></div>' + (gStale() ? '<div class="hint" style="color:#E1242A">Settings changed since this model was trained.</div>' : ''); }
function gStale(){ return !G.trained || JSON.stringify([GP().nl, GP().steps]) !== G.sig; }
function gRun(){
  if (GA.promise) return GA.promise;
  GA.promise = (async function(){
    var st = $('gst');
    try { await waitRelease(); await tick(); await gTrain({ onStep: async function(s, n){ if (s % 5 === 0) { var t = 'Training the GPT: step ' + (s + 1) + ' of ' + n; if (st) st.textContent = t; progress((s + 1) / n); L4.steps.forEach(function(x){ if (x.needsModel && x.visible) x.api.busy(t); }); } } }); if (st) st.textContent = 'Trained. Final loss ' + G.losses[G.losses.length - 1].toFixed(2); progress(1); }
    finally { GA.promise = null; }
  })();
  return GA.promise;
}
async function gReady(api){ for (var k = 0; k < 3; k++) { if (!gStale()) { if (api) api.idle(); return true; } if (api) api.busy('Training the GPT'); await gRun(); } if (api) api.idle(); return !gStale(); }
async function ensureGPT(){ await gReady(null); }
var gTimer = null;
L4.onOpts = function(){ var fire = function(){ clearTimeout(gTimer); gTimer = setTimeout(function(){ gResultsBox(); L4.refresh(); }, 200); }; ['glayers', 'gsteps', 'gtemp'].forEach(function(id){ $(id).addEventListener('input', fire); });
  $('gtrain').addEventListener('click', function(){ var s = L4.steps[3]; s.sec.scrollIntoView({ behavior: ANIM ? 'smooth' : 'auto', block: 'start' }); setTimeout(function(){ s.play(); }, ANIM ? 500 : 0); }); };

/* ---------- G1: text becomes tokens ---------- */
L4.add({ short: 'Tokens', title: 'A GPT reads text as a sequence of tokens',
  change: 'Press Play, then Next compound.',
  watch: 'The structure becomes a SMILES string. The string is cut into tokens, and each token has an ID number. The model only ever sees the IDs.',
  build: function(api){
    var cur = null;
    var sp = api.el('pn fd', 'left:28px;top:28px;width:250px;height:226px', '<h4>Compound drawn by RDKit</h4><div class="struct" style="height:186px"></div>');
    var tp = api.el('pn fd', 'left:298px;top:28px;width:634px;height:96px', '<h4>SMILES text</h4><div class="smi" style="min-height:20px"></div>');
    var kp = api.el('pn fd', 'left:298px;top:136px;width:634px;height:150px', '<h4>Tokens and their ID numbers</h4><div class="tk"></div>');
    var vp = api.el('pn fd', 'left:28px;top:298px;width:904px;height:138px', '<h4 class="vh">Vocabulary</h4><div class="vc" style="line-height:1.1"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    async function setup(next){ await loadStructs(); gptSetup(); if (!cur || next) { var a = STRUCTS.HLM.filter(function(e){ return e.smi.match(TOK).length <= 30; }); cur = a[Math.floor(Math.random() * a.length)]; } qs('.struct', sp).innerHTML = cur.svg; qs('.smi', tp).textContent = ''; var toks = cur.smi.match(TOK);
      qs('.tk', kp).innerHTML = '<div style="display:flex;flex-wrap:wrap;gap:4px">' + ['[start]'].concat(toks, ['[end]']).map(function(t, k){ var id = k === 0 ? 1 : (k === toks.length + 1 ? 2 : G.stoi[t]); return '<div class="tkc fl" style="text-align:center"><span class="tok ' + (k === 0 || k === toks.length + 1 ? 's' : '') + '" style="margin:0">' + esc(t) + '</span><div style="font:600 11px Cascadia Mono,Consolas,monospace;color:#6B7280;margin-top:3px">' + (id === undefined ? '?' : id) + '</div></div>'; }).join('') + '</div>';
      var used = {}; toks.forEach(function(t){ used[t] = 1; }); qs('.vh', vp).textContent = 'Vocabulary: ' + G.V + ' tokens, learned from ' + G.seqs.length + ' training SMILES'; qs('.vc', vp).innerHTML = G.vocab.map(function(t, i){ return '<span style="display:inline-block;margin:2px 6px 2px 0;font:600 11.5px Cascadia Mono,Consolas,monospace;padding:3px 6px;border-radius:4px;border:1px solid ' + (used[t] ? C.red : C.silver) + ';background:' + (used[t] ? '#fff' : '#F6F8F9') + ';color:' + (used[t] ? C.red : C.mid) + '">' + esc(tokName(i)) + ' ' + i + '</span>'; }).join('');
      [sp, tp, kp, vp].forEach(function(e){ e.classList.remove('on'); }); qsa('.tkc', kp).forEach(function(e){ e.classList.remove('on'); }); }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ api.tok++; await setup(true); var t = api.tok; api.stepDef.safe(function(){ return play(t); }); });
    async function render(){ await setup(false); [sp, tp, kp, vp].forEach(function(e){ e.classList.add('on'); }); qs('.smi', tp).textContent = cur.smi; qsa('.tkc', kp).forEach(function(e){ e.classList.add('on'); }); }
    async function play(tok){ await setup(false); sp.classList.add('on'); await api.beat('A chemist sees a structure. A language model needs text.', 2400); tp.classList.add('on'); api.cap('RDKit writes the structure as a SMILES string. Letters are atoms, symbols are bonds and branches, digits close rings.'); await typeInto(qs('.smi', tp), cur.smi, api, 60); await api.sleep(1400);
      kp.classList.add('on'); api.cap('A tokenizer cuts the string into pieces. Two-letter atoms such as Cl and Br stay together.'); var cs = qsa('.tkc', kp); for (var k = 0; k < cs.length; k++) { api.checkTok(tok); cs[k].classList.add('on'); await api.sleep(110); } await api.sleep(1600);
      vp.classList.add('on'); await api.beat('Every token has an ID from a fixed vocabulary. The tokens in this compound are outlined in red.', 3400); api.key('The model never sees the drawing. It reads a list of ID numbers and learns which usually follows which.'); }
    return { render: render, play: play };
  } });

function sepRatio(xy){ var cats = G.vocab.map(function(t){ return tokCat(t); }), win = 0, nw = 0, bw = 0, nb = 0; for (var i = 3; i < xy.length; i++) for (var j = i + 1; j < xy.length; j++) { var d = Math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]); if (cats[i] === cats[j]) { win += d; nw++; } else { bw += d; nb++; } } return (win / Math.max(1, nw)) / ((bw / Math.max(1, nb)) || 1); }
/* ---------- G2: embeddings ---------- */
L4.add({ short: 'Embeddings', title: 'Each token becomes a list of numbers the model can adjust',
  change: 'Press Play. Train the GPT in step 4 and replay to see the map change.',
  watch: 'At the start every token has random numbers. After training, similar tokens sit close together on the map.',
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
      qs('.em', ep).innerHTML = toks.map(function(t){ return '<div style="display:flex;align-items:center;gap:10px;margin:5px 0"><span class="tok" style="width:46px;text-align:center;margin:0">' + esc(t) + '</span><div style="display:flex;gap:2px">' + M[G.stoi[t]].slice(0, 14).map(function(v){ var a = Math.min(1, Math.abs(v) / mx); return '<span style="display:block;width:19px;height:26px;border-radius:3px;background:' + (v >= 0 ? C.red : C.char) + ';opacity:' + (0.1 + 0.9 * a).toFixed(2) + '"></span>'; }).join('') + '</div></div>'; }).join('') + '<p style="font-size:12px;color:#6B7280;margin-top:10px">Red: positive number. Dark: negative. Stronger colour: larger magnitude.</p>'; }
    function mapDraw(xy, t0){ var g = cv(qs('canvas', mp)), x = g.x, W = g.w, H = g.h, mx = Math.max.apply(null, xy.map(function(p){ return Math.abs(p[0]); })) || 1, my = Math.max.apply(null, xy.map(function(p){ return Math.abs(p[1]); })) || 1; x.strokeStyle = C.silver; x.beginPath(); x.moveTo(W / 2, 8); x.lineTo(W / 2, H - 8); x.moveTo(8, H / 2); x.lineTo(W - 8, H / 2); x.stroke();
      xy.forEach(function(p, i){ var t = G.vocab[i]; if (i < 3) return; var cx = W / 2 + p[0] / mx * (W / 2 - 30), cy = H / 2 - p[1] / my * (H / 2 - 22), col = CATC[tokCat(t)]; x.fillStyle = col; x.beginPath(); x.arc(cx, cy, 5.5, 0, 6.3); x.fill(); x.fillStyle = C.char; x.font = '600 11px Cascadia Mono, Consolas, monospace'; x.textAlign = 'center'; x.fillText(t, cx, cy - 9); }); }
    function legend(){ qs('.ml', mp).innerHTML = Object.keys(CATC).filter(function(k){ return k !== 'special'; }).map(function(k){ return '<span style="margin-right:10px"><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:' + CATC[k] + ';margin-right:4px"></i>' + k + '</span>'; }).join(''); }
    function show(trained){ strips(trained); curXY = pca(matrix(trained)); mapDraw(curXY); legend(); qs('.mt', mp).textContent = trained && G.trained ? 'Map of all tokens after training' : 'Map of all tokens before training'; }
    sel.addEventListener('change', function(){ api.tok++; show(sel.value === '1'); });
    function render(){ gptSetup(); if (!G.P) gInit(+$('glayers').value); show(sel.value === '1' && G.trained); }
    async function play(tok){ gptSetup(); if (!G.P) gInit(+$('glayers').value); sel.value = '0'; show(false); await api.beat('Each token ID is looked up in a table of 64 numbers. At the start the numbers are random.', 3400);
      if (!G.trained) { api.cap('Train the GPT in step 4 and come back: the map will show what it learned.'); await api.sleep(3200); api.key('Embeddings are the model\'s own vocabulary of numbers. Training moves similar tokens close together.'); return; }
      var a = pca(G.emb0), b = pca(G.P.tok.arraySync()); api.cap('After training the same table has changed. Watch the tokens move on the map.'); sel.value = '1'; strips(true); await api.anim(2800, function(t){ mapDraw(a.map(function(p, i){ return [lerp(p[0], b[i][0], t), lerp(p[1], b[i][1], t)]; })); }); legend(); qs('.mt', mp).textContent = 'Map of all tokens after training'; await api.sleep(1600);
      var s0 = sepRatio(a), s1 = sepRatio(b); if (s1 < s0) api.cap('Tokens of the same kind (atoms, ring digits, bonds) are now ' + Math.round((1 - s1) * 100) + '% closer to each other than to other tokens, up from ' + Math.round((1 - s0) * 100) + '% before training.'); else api.cap('After only ' + G.steps + ' steps the groups are still weak. Train for more steps and replay to see tokens of the same kind drift together.'); await api.sleep(4200);
      api.key('Nobody labels atoms or rings. Any grouping comes only from predicting the next token, and it grows with training.'); }
    return { render: render, play: play };
  } });

/* ---------- G3: the transformer stack ---------- */
L4.add({ short: 'The stack', title: 'A transformer repeats attention and a small network',
  change: 'Drag Transformer layers in the options, or press Play.',
  watch: 'Each block lets every token look back at earlier tokens, then processes each token on its own. The parameter count follows your settings.',
  build: function(api){
    var dp = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<svg viewBox="0 0 880 380" width="880" height="380"></svg>');
    function per(){ return DM * 3 * DM + 3 * DM + DM * DM + DM + DM * 4 * DM + 4 * DM + 4 * DM * DM + DM + 4 * DM; }
    function draw(active){
      gptSetup(); var N = GP().nl, V = G.V, s = '<defs><marker id="gar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="' + C.red + '"/></marker></defs>', bh = 52, gap = 12, top = 20, blocks = N, total = V * DM + TLEN * DM + N * per() + 2 * DM + DM * V;
      function box(x, y, w, h, title, sub, hot){ s += '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="6" fill="' + (hot ? '#FDECEC' : '#fff') + '" stroke="' + (hot ? C.red : C.silver) + '" stroke-width="' + (hot ? 2.5 : 1) + '"/><text x="' + (x + 12) + '" y="' + (y + 22) + '" font-size="13" font-weight="700" fill="' + C.char + '">' + title + '</text>' + (sub ? '<text x="' + (x + 12) + '" y="' + (y + 40) + '" font-size="11" fill="' + C.mid + '">' + sub + '</text>' : ''); }
      function arrow(x1, y1, x2, y2){ s += '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" stroke="' + C.red + '" stroke-width="2" marker-end="url(#gar)"/>'; }
      var cx = 40, w = 360; box(cx, 316, w, 48, 'Token IDs and positions', 'the text so far, as numbers', active === 'in'); box(cx, 252, w, 48, 'Embeddings', V + ' tokens, ' + DM + ' numbers each, plus position', active === 'emb'); arrow(cx + w / 2, 314, cx + w / 2, 302); arrow(cx + w / 2, 250, cx + w / 2, 238);
      var y = 238; for (var b = 0; b < blocks; b++) { var by = 238 - (b + 1) * (2 * 40 + 18) + 40; var yb = 238 - (b + 1) * 112 + 18; box(cx, yb + 52, w, 40, 'Block ' + (b + 1) + ': feed-forward', DM + ' to ' + 4 * DM + ' to ' + DM + ' numbers, per token', active === 'ff' + b); box(cx, yb, w, 44, 'Block ' + (b + 1) + ': attention', HEADS + ' heads, each token looks back', active === 'at' + b); arrow(cx + w / 2, yb + 50, cx + w / 2, yb + 46); arrow(cx + w / 2, yb + 98, cx + w / 2, yb + 94); }
      var topY = 238 - blocks * 112 + 18; if (topY < 36) topY = 36; box(cx, 4, w, 28, 'Next-token odds: ' + V + ' probabilities', '', active === 'out'); arrow(cx + w / 2, 36, cx + w / 2, 33);
      var px = 450; s += '<text x="' + px + '" y="40" font-size="12" font-weight="700" fill="' + C.char + '">This design</text>'; var facts = [['Transformer layers', N], ['Attention heads per layer', HEADS], ['Numbers per token', DM], ['Vocabulary', V], ['Parameters in total', total.toLocaleString()]]; facts.forEach(function(f, i){ s += '<text x="' + px + '" y="' + (70 + i * 28) + '" font-size="13" fill="' + C.mid + '">' + f[0] + '</text><text x="' + (px + 280) + '" y="' + (70 + i * 28) + '" font-size="14" font-weight="700" fill="' + (i === 4 ? C.red : C.char) + '" text-anchor="end">' + f[1] + '</text><line x1="' + px + '" y1="' + (76 + i * 28) + '" x2="' + (px + 280) + '" y2="' + (76 + i * 28) + '" stroke="' + C.silver + '"/>'; });
      var tx = 450, ty = 230; s += '<text x="' + tx + '" y="' + ty + '" font-size="12" font-weight="700" fill="' + C.char + '">What each part does</text>'; [['Attention', 'Each token weighs the earlier tokens by relevance and mixes in what matters.'], ['Feed-forward', 'A small network processes each token separately.'], ['Repeat', 'Stacking blocks lets later blocks build on what earlier ones found.']].forEach(function(r, i){ s += '<text x="' + tx + '" y="' + (ty + 26 + i * 40) + '" font-size="12.5" font-weight="700" fill="' + C.red + '">' + r[0] + '</text><text x="' + tx + '" y="' + (ty + 42 + i * 40) + '" font-size="12" fill="' + C.char + '">' + r[1] + '</text>'; });
      qs('svg', dp).innerHTML = s; }
    function render(){ draw(''); }
    async function play(tok){ var N = GP().nl; draw('in'); await api.beat('The text so far enters as token IDs, each with a position.', 2600); draw('emb'); await api.beat('Embeddings turn IDs into numbers. This is where the model reads the text.', 2600);
      for (var b = 0; b < N; b++) { draw('at' + b); await api.beat('Block ' + (b + 1) + ' attention: every token looks back at earlier tokens and decides which matter.', 3000); draw('ff' + b); await api.beat('Block ' + (b + 1) + ' feed-forward: a small network processes each token.', 2600); }
      draw('out'); await api.beat('The top turns the last token into a probability for every token in the vocabulary.', 3000); draw(''); api.key('The same two pieces repeat. More layers give more capacity, and more parameters to train.'); }
    return { render: render, play: play };
  } });

/* ---------- G4: training ---------- */
L4.add({ short: 'Training', title: 'Training teaches the model to guess the next token',
  change: 'Change Transformer layers or Training steps and the GPT retrains here, live, when you let go of the slider. Or press Play for a narrated run.',
  watch: 'Top: how sure the model is about each token of one real molecule, from pale (surprised) to dark red (sure). Bottom: loss and the molecule it writes at that moment.',
  auto: false,
  build: function(api){
    var rp = api.el('pn', 'left:28px;top:28px;width:904px;height:164px', '<h4>One real molecule, token by token: confidence in the correct next token</h4><div class="rv" style="margin-top:6px"></div><div class="rs note" style="font-size:12px;color:#6B7280;margin-top:6px"></div>');
    var lp = api.el('pn', 'left:28px;top:204px;width:520px;height:232px', '<h4>Loss</h4><canvas width="498" height="184"></canvas>');
    var ap = api.el('pn', 'left:568px;top:204px;width:364px;height:232px', '<h4>The molecule it writes now</h4><div class="at smi" style="background:#34373F;color:#fff;border-radius:6px;padding:10px;min-height:62px;font-size:12.5px;word-break:break-all">Not trained yet</div><div class="ac" style="font-size:12px;color:#6B7280;margin-top:8px"></div>');
    function readViz(){ if (!G.P) return; if (!G.readIds) { var s = G.smiles.filter(function(x){ return x.match(TOK).length <= 24; })[2] || G.smiles[0], t = s.match(TOK); G.readIds = [1].concat(t.map(function(x){ return G.stoi[x]; }), [2]); }
      var ids = G.readIds, pr; tf.tidy(function(){ var pm = tf.softmax(gforward(tf.tensor2d([ids.slice(0, -1)], null, 'int32'), G.NL)).arraySync()[0]; pr = ids.slice(1).map(function(id, k){ return pm[k][id]; }); });
      qs('.rv', rp).innerHTML = ids.slice(1).map(function(id, k){ var q = pr[k]; return '<span class="tok" style="animation:none;text-align:center;background:rgba(225,36,42,' + (0.06 + 0.94 * q).toFixed(2) + ');color:' + (q > 0.5 ? '#fff' : C.char) + '"><span>' + esc(tokName(id)) + '</span><small style="font-size:9.5px;margin-top:4px;opacity:.85">' + Math.round(q * 100) + '%</small></span>'; }).join(''); qs('.rs', rp).textContent = 'Average confidence ' + Math.round(mean(pr) * 100) + '%. Guessing at random would give about ' + (100 / G.V).toFixed(1) + '%.'; }
    function lossDraw(n){ var pts = G.losses.slice(0, n === undefined ? G.losses.length : n).map(function(v, i){ return [i + 1, v]; }); if (!pts.length) { cv(qs('canvas', lp)); return; } lineChart(qs('canvas', lp), [{ pts: pts, color: C.red }], { xl: 'training step', yl: 'loss', x0: 1, x1: Math.max(2, pts.length), xdec: 0, ydec: 2 }); }
    function attempt(txt){ qs('.at', ap).textContent = txt; qs('.ac', ap).textContent = syntaxOk(txt) ? 'Brackets and rings balance.' : 'Brackets or rings do not balance yet.'; }
    GLIVE.push(function(s, n){ var st = L4.steps[3]; if (!st || !st.visible) return; if (s === 0) { G.readIds = null; readViz(); attempt('(just started)'); } if (s % 10 === 9) lossDraw(s + 1); if (s % 20 === 19 || s === n - 1) readViz(); if (s % 100 === 99 || s === n - 1) gWrite([1], 0.8, 40).then(attempt); });
    function showFinal(){ if (!G.P || !G.trained) { qs('.rv', rp).innerHTML = '<span class="note">The GPT is not trained yet. It trains here when you press Play or change the options.</span>'; cv(qs('canvas', lp)); return; } readViz(); lossDraw(); }
    async function render(){ showFinal(); if (gStale() && !GA.promise && !G.busy) { var st = $('gst'); if (st) st.textContent = 'Training the GPT'; gReady(null); } }
    async function play(tok){
      if (G.busy || GA.promise) return; var btn = $('gtrain'); btn.disabled = true; var st = $('gst'), p = GP(); gptSetup(); G.readIds = null;
      try { gInit(p.nl); readViz(); lossDraw(0); attempt(await gWrite([1], 1.0, 34)); api.cap('Before training the model knows nothing. Its confidence is close to random guessing.'); await api.sleep(2600);
        await gTrain({ force: true, onStep: async function(s, n){ api.checkTok(tok); if (s % 10 === 9) { st.textContent = 'Step ' + (s + 1) + ' of ' + n; progress((s + 1) / n); }
          if (s === 20) api.cap('Early on it learns which tokens are common: carbon, rings and branches.'); if (s === Math.floor(n * 0.4)) api.cap('Now it knows that a ring digit must be closed later and a bracket must be matched.'); if (s === Math.floor(n * 0.8)) api.cap('The boxes darken as it becomes sure of more tokens.'); } });
        readViz(); lossDraw(); st.textContent = 'Done'; progress(1); attempt(await gWrite([1], GP().temp, 40)); api.cap('Training finished. Final loss ' + G.losses[G.losses.length - 1].toFixed(2) + '.'); await api.sleep(3000); api.key('Training never labels anything. The model learns SMILES grammar by guessing the next token millions of times.'); gResultsBox(); L4.refresh(); }
      catch (e) { if (!(e instanceof Cancel)) { st.textContent = 'Error: ' + e.message; console.error(e); } }
      btn.disabled = false; }
    return { render: render, play: play };
  } });

/* ---------- G5: attention ---------- */
L4.add({ short: 'Attention', title: 'Attention shows which earlier tokens each token looks at',
  change: 'Choose a molecule, layer and head with the controls below. The first time, the GPT is trained with your options.',
  watch: 'Left: every token (row) and the earlier tokens it attends to (columns). Right: the last token and the tokens it looks at most. The upper triangle is empty because a GPT cannot look ahead.',
  build: function(api){
    var hp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px', '<h4 class="ht">Attention of every token</h4><div class="hm" style="overflow:auto"></div>');
    var ap = api.el('pn', 'left:468px;top:28px;width:464px;height:408px', '<h4>Where the last token looks</h4><svg viewBox="0 0 442 300" width="442" height="300"></svg><div class="an" style="font-size:12.5px;margin-top:6px"></div>');
    var ctl = api.addControl('Molecule <select class="ms"></select> Layer <select class="ls"></select> Head <select class="hs"></select>'); var ms = qs('.ms', ctl), ls = qs('.ls', ctl), hs = qs('.hs', ctl), data = null;
    function fillSel(){ gptSetup(); if (!ms.options.length) { G.smiles.filter(function(s){ return s.match(TOK).length <= 22; }).slice(0, 8).forEach(function(s, i){ var o = document.createElement('option'); o.value = s; o.textContent = (s.length > 24 ? s.slice(0, 24) + '...' : s); ms.appendChild(o); }); } var nl = G.NL; ls.innerHTML = Array.from({ length: nl }, function(_, i){ return '<option value="' + i + '">' + (i + 1) + '</option>'; }).join(''); hs.innerHTML = '<option value="avg">average</option>' + Array.from({ length: HEADS }, function(_, i){ return '<option value="' + i + '">' + (i + 1) + '</option>'; }).join(''); ls.value = nl - 1; }
    function compute(){ var t = ms.value.match(TOK), ids = [1].concat(t.map(function(x){ return G.stoi[x] === undefined ? 0 : G.stoi[x]; })), labels = ['[s]'].concat(t), A; tf.tidy(function(){ var o = gforward(tf.tensor2d([ids], null, 'int32'), G.NL, true), a = o.atts[+ls.value]; A = (hs.value === 'avg' ? a.mean(1) : a.slice([0, +hs.value, 0, 0], [1, 1, ids.length, ids.length]).reshape([1, ids.length, ids.length])).arraySync()[0]; }); data = { A: A, labels: labels, n: ids.length }; }
    function draw(){ if (!data) return; var n = data.n, A = data.A, cs = clamp(Math.floor(340 / n), 8, 16), off = 38, svg = ''; for (var i = 0; i < n; i++) { svg += '<text x="' + (off - 4) + '" y="' + (off + i * cs + cs - 3) + '" text-anchor="end" font-size="' + Math.min(10, cs) + '" fill="' + C.char + '">' + esc(data.labels[i]) + '</text><text transform="translate(' + (off + i * cs + cs - 3) + ',' + (off - 4) + ') rotate(-90)" font-size="' + Math.min(10, cs) + '" fill="' + C.char + '">' + esc(data.labels[i]) + '</text>'; for (var j = 0; j <= i; j++) svg += '<rect x="' + (off + j * cs) + '" y="' + (off + i * cs) + '" width="' + cs + '" height="' + cs + '" fill="' + C.red + '" fill-opacity="' + Math.min(1, A[i][j] * 1.1).toFixed(3) + '" stroke="#fff" stroke-width=".6"/>'; }
      var side = off + n * cs + 4; qs('.hm', hp).innerHTML = '<svg width="' + side + '" height="' + side + '" viewBox="0 0 ' + side + ' ' + side + '">' + svg + '</svg>'; qs('.ht', hp).textContent = 'Attention of every token, layer ' + (+ls.value + 1) + ', head ' + (hs.value === 'avg' ? 'average' : (+hs.value + 1));
      var last = n - 1, W = 442, cw = Math.min(30, (W - 20) / n), x0 = (W - cw * n) / 2, s2 = ''; for (var q = 0; q < n; q++) { var x = x0 + (q + 0.5) * cw; s2 += '<rect x="' + (x - cw / 2 + 1) + '" y="262" width="' + (cw - 2) + '" height="26" rx="4" fill="' + (q === last ? C.red : '#fff') + '" stroke="' + (q === last ? C.red : C.silver) + '"/><text x="' + x + '" y="279" text-anchor="middle" font-size="' + (data.labels[q].length > 2 ? 9 : 12) + '" font-family="Consolas,monospace" fill="' + (q === last ? '#fff' : C.char) + '">' + esc(data.labels[q]) + '</text>'; if (q < last) { var w = A[last][q], d = last - q, ym = 240 - Math.min(210, 20 + d * 12); s2 += '<path d="M' + (x0 + (last + 0.5) * cw) + ' 260 Q' + ((x + x0 + (last + 0.5) * cw) / 2) + ' ' + ym + ' ' + x + ' 260" fill="none" stroke="' + C.red + '" stroke-opacity="' + (0.12 + 0.88 * Math.min(1, w * 1.6)).toFixed(2) + '" stroke-width="' + (0.8 + 7 * w).toFixed(1) + '"/>'; } }
      qs('svg', ap).innerHTML = s2; var top = []; for (var j2 = 0; j2 < last; j2++) top.push([A[last][j2], j2]); top.sort(function(a, b){ return b[0] - a[0]; }); qs('.an', ap).innerHTML = last ? 'The last token <b>' + esc(data.labels[last]) + '</b> looks mostly at ' + top.slice(0, 3).map(function(t){ return '<b>' + esc(data.labels[t[1]]) + '</b> (' + Math.round(t[0] * 100) + '%)'; }).join(', ') + '.' : ''; }
    [ms, ls, hs].forEach(function(e){ e.addEventListener('change', function(){ api.tok++; compute(); draw(); }); });
    async function ready(){ gptSetup(); if (!(await gReady(api))) return false; if (!ms.options.length || +ls.options.length !== G.NL) fillSel(); compute(); return true; }
    async function render(){ if (!(await ready())) return; draw(); }
    async function play(tok){ if (!(await ready())) return; var cur = ls.value; api.cap('Each row is a token being processed. Dark red cells mark the earlier tokens it pays attention to.'); draw(); await api.sleep(3600);
      for (var l = 0; l < G.NL; l++) { ls.value = l; hs.value = 'avg'; compute(); draw(); api.cap('Layer ' + (l + 1) + ', average of the ' + HEADS + ' heads.'); await api.sleep(2600); }
      hs.value = '0'; ls.value = G.NL - 1; compute(); draw(); api.cap('One head on its own. Different heads learn different habits, such as looking at the previous token or at an opening bracket.'); await api.sleep(3600); hs.value = 'avg'; compute(); draw(); api.key('Attention is how the model finds relevant context, such as the digit that opened a ring it must now close.'); }
    return { render: render, play: play };
  } });

/* ---------- G6: next token and temperature ---------- */
L4.add({ short: 'Next token', title: 'The model outputs odds, and temperature reshapes them',
  change: 'Edit the SMILES start, drag Temperature in the options, or press Play to sweep it.',
  watch: 'Bars show the probability of each possible next token. A low temperature concentrates on the favourite. A high temperature spreads the odds.',
  build: function(api){
    var ip = api.el('pn', 'left:28px;top:28px;width:440px;height:170px', '<h4>SMILES so far</h4><input type="text" class="pf" spellcheck="false" style="font-family:Cascadia Mono,Consolas,monospace" value="CC(=O)Nc1ccc"><div class="pr" style="margin-top:10px"></div>');
    var wp = api.el('pn', 'left:28px;top:210px;width:440px;height:226px', '<h4>Written by sampling</h4><div class="wt smi" style="min-height:60px;background:#34373F;color:#fff;border-radius:6px;padding:10px;font-size:12.5px"></div><div class="note" style="font-size:12px;color:#6B7280;margin-top:8px">Press Auto-write to add tokens one at a time, each drawn from the odds on the right.</div>');
    var bp = api.el('pn', 'left:488px;top:28px;width:444px;height:408px', '<h4 class="bh">Probability of each next token</h4><div class="pb probs"></div><div class="en" style="font-size:12.5px;margin-top:12px"></div>');
    var ctl = api.addControl('Temperature <input type="range" min="0.3" max="1.6" step="0.1" value="0.8" style="width:160px"><b class="tv">0.8</b> <button type="button" class="btn sm" data-auto>Auto-write</button>'); var tr = qs('input', ctl), writing = false;
    function ids(){ var t = qs('.pf', ip).value.trim().match(TOK) || [], out = [1]; for (var i = 0; i < t.length; i++) { if (G.stoi[t[i]] === undefined) return { err: t[i] }; out.push(G.stoi[t[i]]); } return { ids: out }; }
    function showPrefix(arr, newIdx){ qs('.pr', ip).innerHTML = arr.map(function(i, k){ return '<span class="tok ' + (k === 0 ? 's' : (k === newIdx ? 'new' : '')) + '" style="animation:none">' + esc(tokName(i)) + '</span>'; }).join(''); }
    function bars(p, picked){ var top = Array.from(p).map(function(v, i){ return [v, i]; }).sort(function(a, b){ return b[0] - a[0]; }).slice(0, 10), ent = -Array.from(p).reduce(function(s, v){ return s + (v > 1e-9 ? v * Math.log(v) : 0); }, 0);
      qs('.pb', bp).innerHTML = top.map(function(e){ return '<div class="' + (e[1] === picked ? 'pick' : '') + '"><span>' + esc(tokName(e[1])) + '</span><span class="tr"><i style="width:' + (e[0] * 100).toFixed(1) + '%"></i></span><span>' + Math.round(e[0] * 100) + '%</span></div>'; }).join('');
      qs('.en', bp).innerHTML = 'Spread of the odds (entropy) <b>' + ent.toFixed(2) + '</b>. Zero means one certain token. Higher means more uncertainty.<br>Temperature ' + (+tr.value).toFixed(1) + ': ' + (+tr.value < 0.6 ? 'the favourite dominates.' : (+tr.value > 1.1 ? 'unlikely tokens now have real chances.' : 'a balance between safe and varied.')); }
    function update(){ var r = ids(); if (r.err) { qs('.pr', ip).innerHTML = '<span class="note" style="color:' + C.red + '">Token ' + esc(r.err) + ' is not in the vocabulary.</span>'; return null; } showPrefix(r.ids); if (!G.trained) { qs('.pb', bp).innerHTML = '<span class="note">Train the GPT first (step 4 or the options).</span>'; return r; } var p = gDist(r.ids, +tr.value); bars(p); return r; }
    tr.addEventListener('input', function(){ api.tok++; qs('.tv', ctl).textContent = tr.value; $('gtemp').value = tr.value; $('o-gtemp').textContent = tr.value; update(); }); qs('.pf', ip).addEventListener('input', function(){ api.tok++; update(); });
    qs('[data-auto]', ctl).addEventListener('click', async function(){ if (writing || !G.trained) return; writing = true; var btn = this; btn.disabled = true; var r = ids(); if (r.err) { writing = false; btn.disabled = false; return; } var arr = r.ids.slice(), tok = ++api.tok; try { for (var t = arr.length; t < TLEN; t++) { api.checkTok(tok); var p = gDist(arr, +tr.value), nx = gSample(p); bars(p, nx); await api.sleep(700); if (nx === 2) { qs('.wt', wp).textContent += '  [end]'; break; } arr.push(nx); showPrefix(arr, arr.length - 1); qs('.wt', wp).textContent = arr.slice(1).map(function(i){ return G.vocab[i]; }).join(''); await api.sleep(250); } } catch (e) { if (!(e instanceof Cancel)) console.error(e); } writing = false; btn.disabled = false; });
    async function ready(){ gptSetup(); if (!(await gReady(api))) return false; tr.value = $('gtemp').value; qs('.tv', ctl).textContent = tr.value; return true; }
    async function render(){ if (!(await ready())) return; update(); }
    async function play(tok){ if (!(await ready())) return; update(); api.cap('Given the text so far, the model outputs a probability for every token in its vocabulary.'); await api.sleep(3200); api.cap('Now sweep the temperature from low to high.'); for (var t = 0.3; t <= 1.61; t += 0.1) { tr.value = t.toFixed(1); qs('.tv', ctl).textContent = tr.value; update(); await api.sleep(450); }
      api.cap('At low temperature the model nearly always picks the favourite. At high temperature it explores, and makes more mistakes.'); await api.sleep(3600); tr.value = $('gtemp').value; qs('.tv', ctl).textContent = tr.value; update(); api.key('Temperature does not change what the model knows. It changes how adventurous the sampling is.'); }
    return { render: render, play: play };
  } });

/* ---------- G7: generate and check with RDKit ---------- */
L4.add({ short: 'Generate and check', title: 'Generate molecules and let RDKit check them',
  change: 'Press Generate. The first time, RDKit downloads (about 7 MB). Change Temperature in the options and generate again.',
  watch: 'Each card is a molecule the GPT wrote. RDKit says whether it is chemically valid and draws it. Valid molecules have a red outline.',
  auto: false,
  build: function(api){
    var gp = api.el('pn', 'left:28px;top:28px;width:640px;height:408px', '<h4>Twelve generated molecules</h4><div class="mol3 gridm" style="margin-top:4px"></div>');
    var sp = api.el('pn', 'left:688px;top:28px;width:244px;height:408px', '<h4>Quality at this temperature</h4><div class="sq" style="font-size:13px;color:#4a4f5a">Press Generate.</div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-go>Generate</button>'); var busy = false;
    async function go(){
      if (busy) return; busy = true; var btn = qs('[data-go]', ctl); btn.disabled = true; var sq = qs('.sq', sp), temp = GP().temp;
      try { sq.textContent = 'Preparing the GPT'; await gReady(api); sq.textContent = RDK ? 'Generating' : 'Loading RDKit (about 7 MB)'; progress(0.2); await loadRDKit(); progress(0.4); var out = []; for (var k = 0; k < 12; k++) { out.push(await gWrite([1], temp, 46)); qs('.gridm', gp).innerHTML = out.map(function(s, i){ return '<div class="molc" style="min-height:92px;font-size:10px;word-break:break-all;color:#6B7280">' + esc(s) + '</div>'; }).join(''); }
        var valid = out.map(function(s){ return rdkValid(s); }), seen = {}; G.smiles.forEach(function(s){ seen[s] = 1; });
        qs('.gridm', gp).innerHTML = out.map(function(s, i){ var svg = valid[i] ? rdkSvg(s, 200, 130) : null; return '<div class="molc ' + (valid[i] ? 'ok' : 'bad') + '" style="animation-delay:' + (ANIM ? i * 80 : 0) + 'ms"><div class="struct" style="height:84px">' + (svg || '<div style="height:84px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#6B7280;padding:4px;word-break:break-all">' + esc(s.slice(0, 40)) + '</div>') + '</div><small>' + (valid[i] ? 'valid' : 'RDKit rejects it') + '</small></div>'; }).join('');
        var nv = valid.filter(Boolean).length, uniq = {}; out.forEach(function(s, i){ if (valid[i]) uniq[s] = 1; }); var nu = Object.keys(uniq).length, nn = Object.keys(uniq).filter(function(s){ return !seen[s]; }).length;
        sq.innerHTML = '<div class="nums" style="font-size:13px"><div><span>Temperature</span><b>' + temp.toFixed(1) + '</b></div><div><span>Valid by RDKit</span><b style="color:' + C.red + '">' + nv + ' of 12</b></div><div><span>Passing the simple bracket check</span><b>' + out.filter(syntaxOk).length + ' of 12</b></div><div><span>Unique among valid</span><b>' + nu + '</b></div><div><span>New (not in training)</span><b>' + nn + '</b></div></div><p style="font-size:12px;margin-top:12px;color:#4a4f5a">A tiny model trained for under a minute writes text that looks like SMILES but is rarely valid. More training steps raise the share. Valences, aromatic rings and stereochemistry are checked by RDKit, never learned for certain by the GPT.</p>'; progress(1); }
      catch (e) { sq.innerHTML = '<span style="color:' + C.red + '">' + esc(e.message) + '</span>'; console.error(e); }
      busy = false; btn.disabled = false; }
    qs('[data-go]', ctl).addEventListener('click', function(){ api.tok++; api.clearTexts(); go().then(function(){ api.cap('A molecule counts as valid only if RDKit can parse it. The GPT only learned patterns of text.'); }); });
    return { render: function(){}, play: undefined };
  } });

L4.glossary([
  ['GPT', 'A model trained to predict the next token, then used to write text one token at a time.'], ['Token', 'A piece of text. Here: an atom, bond, bracket or ring digit.'],
  ['Vocabulary', 'The tokens the model knows, each with an ID number.'], ['Embedding', 'The list of numbers that represents one token. Training adjusts them.'],
  ['Attention', 'Each token weighs earlier tokens by how relevant they are.'], ['Head', 'One of several attention patterns computed in parallel inside a layer.'],
  ['Transformer', 'A stack of attention and feed-forward blocks.'], ['Causal mask', 'Stops a token looking at later tokens, so the model only uses the past.'],
  ['Pre-training', 'Learning general patterns before any specific task. No clearance values are used here.'], ['Next-token loss', 'How surprised the model is by the true next token.'],
  ['Temperature', 'Low picks the likeliest token, high picks more varied ones.'], ['Entropy', 'A measure of how spread out the probabilities are.'],
  ['SMILES', 'Text for a molecule, such as CC(=O)N.'], ['PCA', 'A way to squash many numbers per item onto a 2D map while keeping the main differences.'],
  ['Valid molecule', 'A SMILES string that RDKit can read as a chemically sensible structure.']
]);

[4, 5, 6].forEach(function(i){ L4.steps[i].needsModel = true; });
setUses(L4, [[], ['glayers', 'gsteps'], ['glayers'], ['glayers', 'gsteps'], ['glayers', 'gsteps'], ['gtemp', 'glayers', 'gsteps'], ['gtemp', 'glayers', 'gsteps']]);
