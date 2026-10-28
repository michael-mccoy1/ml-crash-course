"use strict";
/* Beginner steps for lessons 04 and 05. They assume chemistry knowledge and no machine learning background. */

/* ---------- shared helpers for these steps ---------- */
function b2Toks(smi){ return smi.match(TOK) || []; }
function b2Col(t){
  if (/^\d$|^%\d\d$/.test(t)) return ['#2E7D4F', 'ring closure'];
  if (t === '(' || t === ')') return [C.bg, 'branch'];
  if (/^[=#\-\/\\]$/.test(t)) return [C.tint, 'bond symbol'];
  if (t[0] === '[') return ['#6F8196', 'special atom'];
  if (t === 'N' || t === 'n') return [C.red, 'nitrogen'];
  if (t === 'O' || t === 'o') return [C.dred, 'oxygen'];
  if (t === 'C' || t === 'c') return [C.char, 'carbon'];
  if (/^(Cl|Br|F|I|S|s|P|B)$/.test(t)) return ['#6F8196', 'other atom'];
  return ['#8A939E', 'other'];
}
function b2IsAtom(t){ var n = b2Col(t)[1]; return n === 'carbon' || n === 'nitrogen' || n === 'oxygen' || n === 'other atom' || n === 'special atom'; }
function b2Chip(t, extra){ var c = b2Col(t)[0]; return '<span style="display:inline-block;margin:2px;padding:4px 8px;border-radius:5px;border:2px solid ' + c + ';font:600 15px Cascadia Mono,Consolas,monospace;color:' + c + ';background:#fff;' + (extra || '') + '">' + esc(t) + '</span>'; }
var B2 = { ready: false, mols: [], toks: [], set: {}, T: {}, LR: {}, PL: {} };
function b2Build(){
  if (B2.ready) return; var all = (STRUCTS.HLM || []).concat(STRUCTS.logD || []), k, i;
  B2.mols = all.map(function(e){ return e.smi; }); B2.toks = B2.mols.map(b2Toks); B2.mols.forEach(function(s){ B2.set[s] = 1; });
  for (k = 1; k <= 8; k++) B2.T[k] = {};
  B2.toks.forEach(function(tk){
    var t = ['^', '^', '^', '^', '^', '^', '^', '^'].concat(tk).concat(['$']);
    for (i = 8; i < t.length; i++) { for (k = 1; k <= 8; k++) { var key = t.slice(i - k, i).join('\u0001'), m = B2.T[k][key] || (B2.T[k][key] = {}); m[t[i]] = (m[t[i]] || 0) + 1; } }
    for (i = 1; i < tk.length - 1; i++) { var kk = tk[i - 1] + '\u0001' + tk[i + 1], mm = B2.LR[kk] || (B2.LR[kk] = {}); mm[tk[i]] = (mm[tk[i]] || 0) + 1; var pl = B2.PL[tk[i - 1]] || (B2.PL[tk[i - 1]] = {}); pl[tk[i]] = (pl[tk[i]] || 0) + 1; }
  });
  B2.ready = true;
}
function b2Pick(m, rnd){ var keys = Object.keys(m), tot = 0, i; for (i = 0; i < keys.length; i++) tot += m[keys[i]]; var r = rnd() * tot, acc = 0; for (i = 0; i < keys.length; i++) { acc += m[keys[i]]; if (r <= acc) return keys[i]; } return keys[keys.length - 1]; }
function b2Gen(k, rnd){ var out = []; for (var n = 0; n < 90; n++) { var h = ['^', '^', '^', '^', '^', '^', '^', '^'].concat(out), nx = null; for (var kk = k; kk >= 1 && !nx; kk--) nx = B2.T[kk][h.slice(-kk).join('\u0001')]; if (!nx) break; var t = b2Pick(nx, rnd); if (t === '$') break; out.push(t); } return out.join(''); }

/* do all branches and rings close? */
function b2Closed(smi){ var tk = b2Toks(smi), depth = 0, open = {}, i; for (i = 0; i < tk.length; i++) { var t = tk[i]; if (t === '(') depth++; else if (t === ')') { depth--; if (depth < 0) return false; } else if (/^\d$|^%\d\d$/.test(t)) open[t] = !open[t]; } if (depth !== 0) return false; for (var k in open) if (open[k]) return false; return tk.length > 0; }

/* ---------- 04 GPT: cutting into pieces ---------- */
LESSONS.gpt.add({ level: 'beginner', short: 'Cutting into pieces', title: 'The text of a molecule is cut into small pieces',
  change: 'Select a molecule or enter a SMILES string. The tokens update.',
  watch: 'A language model operates on discrete tokens, not on raw text. SMILES is tokenised with a regular expression: bracket atoms such as [nH] and the two-letter atoms Cl and Br are single tokens, and every other character is its own token. Each token is mapped to an integer ID, and the model receives only the sequence of IDs. Tokens are coloured by type below. Tokenisation is purely textual and does not check chemistry, so any string can be tokenised.',
  build: function(api){
    var cur = 3, smi = IN_MOLS[3].s;
    var tp = api.el('pn', 'left:28px;top:20px;width:904px;height:176px;overflow:auto', '<h4>SMILES string split into tokens</h4><div class="chips" style="display:block;margin-top:6px"></div>');
    var lp = api.el('pn', 'left:28px;top:208px;width:440px;height:312px;overflow:auto', '<div class="lf" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var dp = api.el('pn', 'left:488px;top:208px;width:444px;height:312px', '<h4>Structure drawn by RDKit</h4><div class="struct" style="height:262px"></div>');
    var ctl = api.addControl('Molecule ' + inSeg(IN_MOLS.map(function(m){ return m.n; }), cur, 'data-m') + ' or enter a SMILES <input type="text" class="ti" style="width:230px;margin-left:6px" spellcheck="false">');
    var ti = qs('.ti', ctl); ti.value = smi;
    function draw(){
      var tk = b2Toks(smi), kinds = {}; tk.forEach(function(t){ kinds[b2Col(t)[1]] = 1; });
      qs('.chips', tp).innerHTML = tk.length ? tk.map(function(t){ return b2Chip(t); }).join('') : '<span class="note">Enter a SMILES string, for example CCO.</span>';
      var legend = [['carbon', C.char], ['nitrogen', C.red], ['oxygen', C.dred], ['ring closure', '#2E7D4F'], ['branch', C.bg], ['bond symbol', C.tint], ['other atom', '#6F8196']].map(function(x){ return '<span style="display:inline-block;margin:2px 8px 2px 0;white-space:nowrap"><i style="display:inline-block;width:12px;height:12px;border-radius:3px;border:2px solid ' + x[1] + ';vertical-align:-2px"></i> ' + x[0] + '</span>'; }).join('');
      var distinct = {}; tk.forEach(function(t){ distinct[t] = 1; });
      qs('.lf', lp).innerHTML = '<div class="ol">Token counts</div><div style="display:flex;gap:26px"><div><div class="big" style="color:' + C.red + ';font-size:34px">' + tk.length + '</div><div style="font-size:12px">tokens in this SMILES</div></div><div><div class="big" style="font-size:34px">' + Object.keys(distinct).length + '</div><div style="font-size:12px">distinct tokens</div></div></div><p style="margin-top:10px">A token is the smallest unit the model reads: an atom, a bond symbol, a bracket, a parenthesis or a ring-closure digit.</p><div class="ol" style="margin-top:8px">Token types</div><div style="font-size:12px">' + legend + '</div><p style="margin-top:8px;font-size:12px;color:' + C.mid + '">Implicit hydrogens are not tokens.</p>';
      var box = qs('.struct', dp); if (!RDK) { box.innerHTML = ''; return; }
      if (smi && rdkValid(smi)) box.innerHTML = rdkSvg(smi, 400, 250); else box.innerHTML = '<p class="note" style="padding:20px">' + (smi ? 'RDKit cannot parse this string, so no structure is drawn. It is still tokenised, because tokenisation does not check chemistry.' : 'Nothing to draw yet.') + '</p>';
    }
    qsa('[data-m] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.v; smi = IN_MOLS[cur].s; ti.value = smi; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    ti.addEventListener('input', function(){ smi = ti.value.trim(); qsa('[data-m] button', ctl).forEach(function(x){ x.classList.remove('on'); }); draw(); });
    async function render(){ await inReady(api); draw(); }
    async function play(tok){
      await render(); var order = [0, 4, 3, 6];
      for (var k = 0; k < order.length; k++) { api.checkTok(tok); cur = order[k]; smi = IN_MOLS[cur].s; ti.value = smi; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === cur); }); draw(); api.cap(k === 0 ? 'Ethanol: three tokens (C, C, O).' : (k === 1 ? 'Paracetamol: the ring-closure digit 1 appears twice, at the opening and closing atoms.' : (k === 2 ? 'Caffeine: two rings and several substituents give a longer token sequence.' : 'Every SMILES becomes a sequence of tokens, which is the model input.'))); await api.sleep(3200); }
      api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 04 GPT: a number for every piece ---------- */
var B2E = { done: false, toks: [], prof: {}, id: {}, classes: ['an atom', 'a ring-closure digit', 'a branch opening', 'a branch closing', 'a bond symbol', 'a bracket atom', 'the end of the string'] };
function b2Class(t){ if (t === '$') return 6; if (/^\d$|^%\d\d$/.test(t)) return 1; if (t === '(') return 2; if (t === ')') return 3; if (/^[=#\-\/\\]$/.test(t)) return 4; if (t[0] === '[') return 5; return 0; }
function b2EmbBuild(){
  if (B2E.done) return; var cnt = {}, fol = {};
  B2.toks.forEach(function(tk){ var t = tk.concat(['$']); t.forEach(function(x, i){ cnt[x] = (cnt[x] || 0) + 1; if (i + 1 < t.length) { var f = fol[x] || (fol[x] = [0, 0, 0, 0, 0, 0, 0]); f[b2Class(t[i + 1])]++; } }); });
  var list = Object.keys(cnt).filter(function(x){ return x !== '$' && fol[x]; }).sort(function(a, b){ return cnt[b] - cnt[a]; }).slice(0, 14);
  B2E.toks = list; list.forEach(function(x, i){ var f = fol[x], s = f.reduce(function(a, b){ return a + b; }, 0) || 1; B2E.prof[x] = f.map(function(v){ return v / s; }); B2E.id[x] = i + 1; B2E.n = cnt[x]; B2E.cnt = B2E.cnt || {}; B2E.cnt[x] = cnt[x]; });
  B2E.done = true;
}
function b2Cos(a, b){ var d = 0, na = 0, nb = 0; for (var i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / (Math.sqrt(na * nb) || 1); }
LESSONS.gpt.add({ level: 'beginner', short: 'A number for every piece', title: 'A model needs numbers, so every piece gets some',
  change: 'Select a token in the list, or press Next token. Compare its vector with the others.',
  watch: 'A model operates on numbers, so each token is converted in two stages. First, each token receives an integer ID, here ranked by frequency, so the most common token is 1. An ID carries no information about the token. Second, each ID is mapped to a vector of numbers, an embedding, which the model adjusts during training so that tokens used in similar contexts have similar vectors. The vectors in this step are built by hand for illustration: each token is described by the distribution of token types that follows it in the 180 structures in structs.json. A GPT learns its embeddings during training.',
  build: function(api){
    var cur = 0;
    var lp = api.el('pn', 'left:28px;top:20px;width:470px;height:500px;overflow:auto', '<h4>The 14 most frequent tokens in 180 structures</h4><div class="lst"></div>');
    var rp = api.el('pn', 'left:518px;top:20px;width:414px;height:500px;overflow:auto', '<div class="det" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm ghost" data-next>Next token</button>');
    function spark(p, col){ return '<span style="display:inline-flex;gap:2px;align-items:flex-end;height:24px">' + p.map(function(v){ return '<i style="display:block;width:7px;height:' + Math.max(2, v * 24) + 'px;background:' + col + ';border-radius:1px"></i>'; }).join('') + '</span>'; }
    function draw(){
      var T = B2E.toks, t = T[cur], p = B2E.prof[t];
      qs('.lst', lp).innerHTML = '<div style="display:grid;grid-template-columns:44px 74px 1fr 90px;gap:8px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:' + C.mid + ';font-weight:700;margin:6px 0 2px"><span>ID</span><span>Token</span><span>Embedding</span><span style="text-align:right">Count</span></div>' + T.map(function(x, i){ return '<div class="rw" data-i="' + i + '" style="display:grid;grid-template-columns:44px 74px 1fr 90px;gap:8px;align-items:center;padding:3px 4px;border-radius:5px;cursor:pointer;border:1.5px solid ' + (i === cur ? C.red : 'transparent') + ';background:' + (i === cur ? '#FBE9E9' : '#fff') + '"><b>' + B2E.id[x] + '</b><span>' + b2Chip(x, 'margin:0;padding:1px 8px;font-size:13px') + '</span>' + spark(B2E.prof[x], i === cur ? C.red : C.char) + '<span style="text-align:right;font-size:12px">' + B2E.cnt[x].toLocaleString() + '</span></div>'; }).join('');
      var sims = T.map(function(x, i){ return i === cur ? null : { x: x, s: b2Cos(p, B2E.prof[x]) }; }).filter(Boolean).sort(function(a, b){ return b.s - a.s; });
      qs('.det', rp).innerHTML = '<div class="ol">Token</div><div style="display:flex;align-items:center;gap:14px">' + b2Chip(t, 'font-size:24px;padding:6px 14px') + '<div><div style="font-size:12px">ID</div><div class="big" style="font-size:30px;color:' + C.red + ';margin:0">' + B2E.id[t] + '</div></div></div>' +
        '<div class="ol" style="margin-top:14px">Distribution of the next token type</div>' + B2E.classes.map(function(c, k){ return '<div style="display:grid;grid-template-columns:120px 1fr 40px;gap:8px;align-items:center;margin:4px 0;font-size:12.5px"><span>' + c + '</span><div style="height:12px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:12px;width:' + (p[k] * 100) + '%;background:' + C.red + ';border-radius:3px"></i></div><b style="text-align:right">' + Math.round(p[k] * 100) + '%</b></div>'; }).join('') +
        '<div class="ol" style="margin-top:14px">Most similar tokens (cosine similarity)</div>' + sims.slice(0, 3).map(function(s){ return '<div style="display:flex;align-items:center;gap:10px;margin:4px 0">' + b2Chip(s.x, 'margin:0;padding:1px 8px;font-size:13px') + '<span style="font-size:12.5px">' + Math.max(0, s.s).toFixed(2) + '</span></div>'; }).join('') +
        '<p style="margin-top:12px;font-size:12.5px">Tokens used in similar contexts have similar vectors. A model can therefore group tokens by use without being given any chemistry.</p>';
      qsa('.rw', lp).forEach(function(e){ e.addEventListener('click', function(){ cur = +e.dataset.i; draw(); }); });
    }
    qs('[data-next]', ctl).addEventListener('click', function(){ cur = (cur + 1) % B2E.toks.length; draw(); });
    async function render(){ await loadStructs(); b2Build(); b2EmbBuild(); draw(); }
    async function play(tok){
      await render(); cur = 0; draw(); api.cap('Each token has an integer ID, which carries no information about the token.'); await api.sleep(3200);
      for (var k = 0; k < 4; k++) { api.checkTok(tok); cur = [0, 1, 2, 3][k]; draw(); api.cap(k === 0 ? 'It also has an embedding vector, here the distribution of what follows it.' : 'Tokens used in similar contexts have similar vectors.'); await api.sleep(2800); }
      api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 04 GPT: why memory matters ---------- */
var B3 = { res: {} };
function b3Run(k){
  if (B3.res[k]) return B3.res[k]; var rnd = rng(1000 + k), N = 100, ok = 0, okNew = 0, cl = 0, samples = [], first = null;
  for (var i = 0; i < N; i++) { var s = b2Gen(k, rnd), closed = b2Closed(s), valid = s.length > 0 && rdkValid(s), copy = !!B2.set[s]; if (closed) cl++; if (valid) ok++; if (valid && !copy) { okNew++; if (!first) first = s; } if (samples.length < 6) samples.push({ s: s, ok: valid, copy: copy, closed: closed }); }
  return (B3.res[k] = { closed: cl / N, valid: ok / N, validNew: okNew / N, samples: samples, first: first });
}
LESSONS.gpt.add({ level: 'beginner', short: 'Why memory matters', title: 'A longer memory writes better molecules',
  change: 'Select a context length and compare the fractions.',
  watch: 'A language model predicts the next token from the preceding tokens. The number of preceding tokens it uses is the context length. SMILES contains long-range constraints: a ring-closure digit must be repeated to close the ring, and every opening parenthesis needs a closing one. A model with a short context cannot track which rings and branches are still open. The model here is an n-gram model: it estimates the next-token distribution from counts of the previous n tokens, and it is trained only on the 180 structures in structs.json. For each context length it generates 100 strings. The first bar is the fraction of strings in which every ring and branch is closed. RDKit then tries to parse each string, and a string it accepts is valid. A long context can also reproduce a training structure, so the last bar counts valid strings that are not in the training set. The fractions are low because 180 structures is a very small training set. GPTs are trained on far larger sets.',
  build: function(api){
    var KS = [1, 3, 6, 8], cur = 0;
    var lp = api.el('pn', 'left:28px;top:20px;width:540px;height:500px;overflow:auto', '<div class="bars"></div><div class="smp"></div>');
    var rp = api.el('pn', 'left:588px;top:20px;width:344px;height:500px;overflow:auto', '<h4 class="rh">A valid sample not in the training set</h4><div class="struct" style="height:190px"></div><div class="rt" style="font-size:13px;line-height:1.5;color:' + C.char + ';margin-top:8px"></div>');
    var ctl = api.addControl('Context length (tokens) ' + inSeg(KS.map(function(k){ return String(k); }), cur, 'data-k'));
    function bar(label, v, col){ return '<div style="display:grid;grid-template-columns:150px 1fr 44px;gap:8px;align-items:center;margin:5px 0;font-size:12.5px"><span>' + label + '</span><div style="height:14px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:14px;width:' + (v * 100) + '%;background:' + col + ';border-radius:3px;transition:width .4s"></i></div><b style="text-align:right">' + Math.round(v * 100) + '%</b></div>'; }
    function draw(){
      var rows = KS.map(function(k, i){ var r = b3Run(k); return '<div style="border:1.5px solid ' + (i === cur ? C.red : C.silver) + ';border-radius:8px;padding:8px 12px;margin-bottom:8px;background:' + (i === cur ? '#FBE9E9' : '#fff') + '"><div style="font-weight:700;margin-bottom:2px">Context length ' + k + '</div>' + bar('rings and branches closed', r.closed, '#6F8196') + bar('valid by RDKit', r.valid, '#2E7D4F') + bar('valid, not in training set', r.validNew, C.red) + '</div>'; }).join('');
      qs('.bars', lp).innerHTML = '<h4>100 generated strings per context length</h4><div style="margin-top:8px">' + rows + '</div>';
      var r = b3Run(KS[cur]);
      qs('.smp', lp).innerHTML = '<div class="ol" style="margin-top:4px">Six samples, context length ' + KS[cur] + '</div>' + r.samples.map(function(s){ return '<div style="display:flex;gap:8px;align-items:flex-start;margin:4px 0;font-size:12.5px"><b style="color:' + (s.ok ? '#1E6B40' : C.red) + ';min-width:96px">' + (s.ok ? (s.copy ? 'valid, in training set' : 'valid, not in training set') : (s.closed ? 'closed, invalid' : 'not closed')) + '</b><span style="font-family:Cascadia Mono,Consolas,monospace;word-break:break-all">' + esc(s.s || '(empty)') + '</span></div>'; }).join('');
      var box = qs('.struct', rp); box.innerHTML = r.first ? rdkSvg(r.first, 300, 180) : ''; qs('.rt', rp).innerHTML = r.first ? '<span style="font-family:Cascadia Mono,Consolas,monospace;word-break:break-all">' + esc(r.first) + '</span><p style="margin-top:8px">' + (KS[cur] === 1 ? 'With a context of one token the model cannot track open rings and branches, so few strings are valid.' : (KS[cur] >= 8 ? 'With a longer context more strings close their rings and branches and are valid. Some reproduce training structures.' : 'A longer context lets the model track open rings and branches.')) + '</p>' : '<p>At this context length none of the 100 strings was a valid structure outside the training set. Most did not close their rings and branches. Select a longer context.</p>';
    }
    qsa('[data-k] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.v; qsa('[data-k] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ await loadStructs(); await inReady(api); b2Build(); api.busy('Generating and checking 400 strings'); await tick(); await tick(); KS.forEach(b3Run); api.idle(); draw(); }
    async function play(tok){
      await render(); for (var k = 0; k < KS.length; k++) { api.checkTok(tok); cur = k; qsa('[data-k] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === cur); }); draw(); api.cap(k === 0 ? 'Context length 1: the model cannot track open rings.' : (k === KS.length - 1 ? 'A longer context closes more rings and gives more valid strings. A usable generator needs far more training data than 180 structures.' : 'Longer context, more strings with every ring and branch closed.')); await api.sleep(3000); } api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: fill in the blank ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'Fill in the blank', title: 'A molecule can test itself: hide a piece and guess it',
  change: 'Select the token you predict was masked. Press Next molecule for another example.',
  watch: 'Self-supervised learning builds its labels from the data. For a SMILES string, one atom token is masked and the task is to predict it from the surrounding tokens. The masked token is the label, so no measurement is required and any collection of structures can be used for training. Below, one atom is masked. Select the token you predict from the tokens on either side. The reference predictor is an n-gram model: it counts which atom occurred between the same neighbours in 180 structures.',
  build: function(api){
    var seed = 1, score = { you: 0, model: 0, n: 0 }, ex = null, answered = false;
    var tp = api.el('pn', 'left:28px;top:20px;width:904px;height:130px', '<h4>SMILES with one atom masked</h4><div class="chips" style="display:block;margin-top:6px"></div>');
    var cp = api.el('pn', 'left:28px;top:162px;width:440px;height:358px;overflow:auto', '<div class="ch"></div>');
    var rp = api.el('pn', 'left:488px;top:162px;width:444px;height:358px;overflow:auto', '<div class="rs" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm ghost" data-next>Next molecule</button> <span class="sco" style="font-size:12.5px;margin-left:10px"></span>');
    function dist(l, r){ var m = B2.LR[l + '\u0001' + r], src = m; var tot = 0, k; if (m) for (k in m) tot += m[k]; if (tot < 2) { src = B2.PL[l] || {}; tot = 0; for (k in src) tot += src[k]; }
      var list = Object.keys(src).filter(b2IsAtom).map(function(t){ return [t, src[t]]; }).sort(function(a, b){ return b[1] - a[1]; }), s = list.reduce(function(a, b){ return a + b[1]; }, 0) || 1; return list.map(function(x){ return [x[0], x[1] / s]; }); }
    function pickEx(){
      var r = rng(seed * 31 + 7), tries = 0; while (tries++ < 400) { var mi = Math.floor(r() * B2.toks.length), tk = B2.toks[mi]; if (tk.length < 12 || tk.length > 36) continue; var cand = []; for (var i = 2; i < tk.length - 2; i++) if (b2IsAtom(tk[i]) && !/[\[\]]/.test(tk[i - 1] + tk[i + 1])) cand.push(i); if (!cand.length) continue; var at = cand[Math.floor(r() * cand.length)]; var d = dist(tk[at - 1], tk[at + 1]); if (d.length < 2) continue;
        var opts = d.slice(0, 4).map(function(x){ return x[0]; }); if (opts.indexOf(tk[at]) < 0) opts.push(tk[at]); ['C', 'N', 'O', 'c', 'n'].forEach(function(x){ if (opts.length < 5 && opts.indexOf(x) < 0) opts.push(x); }); for (var a = opts.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = opts[a]; opts[a] = opts[j]; opts[j] = t; }
        return { mi: mi, tk: tk, at: at, truth: tk[at], dist: d, opts: opts, smi: B2.mols[mi] }; }
      return null;
    }
    function show(choice){
      var e = ex, shown = choice !== undefined;
      qs('.chips', tp).innerHTML = e.tk.map(function(t, i){ if (i !== e.at) return b2Chip(t); return shown ? b2Chip(e.truth, 'background:' + (choice === e.truth ? '#EAF5EE' : '#FCECEC') + ';box-shadow:0 0 0 3px ' + (choice === e.truth ? '#2E7D4F' : C.red)) : '<span style="display:inline-block;margin:2px;padding:4px 14px;border-radius:5px;border:2px dashed ' + C.red + ';font:700 15px Cascadia Mono,Consolas,monospace;color:' + C.red + ';background:#FBE9E9">?</span>'; }).join('');
      qs('.ch', cp).innerHTML = '<div class="ol">Which token was masked?</div><p style="font-size:12.5px;margin:2px 0 8px">Preceding token: ' + b2Chip(e.tk[e.at - 1], 'margin:0 4px;padding:1px 7px;font-size:13px') + ' Following token: ' + b2Chip(e.tk[e.at + 1], 'margin:0 4px;padding:1px 7px;font-size:13px') + '</p><div style="display:flex;flex-wrap:wrap;gap:8px">' + e.opts.map(function(o){ var c = b2Col(o)[0], st = shown ? (o === e.truth ? 'background:#EAF5EE;border-color:#2E7D4F' : (o === choice ? 'background:#FCECEC;border-color:' + C.red : '')) : ''; return '<button type="button" class="opt" data-o="' + esc(o) + '" style="font:700 18px Cascadia Mono,Consolas,monospace;color:' + c + ';border:2px solid ' + c + ';border-radius:6px;background:#fff;padding:8px 18px;cursor:' + (shown ? 'default' : 'pointer') + ';' + st + '"' + (shown ? ' disabled' : '') + '>' + esc(o) + '</button>'; }).join('') + '</div>';
      var top = e.dist.slice(0, 5);
      qs('.rs', rp).innerHTML = shown ? '<div class="ol">Reference predictor (n-gram counts)</div>' + top.map(function(x){ return '<div style="display:grid;grid-template-columns:56px 1fr 40px;gap:8px;align-items:center;margin:5px 0"><span>' + b2Chip(x[0], 'margin:0;padding:1px 8px;font-size:13px') + '</span><div style="height:13px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:13px;width:' + (x[1] * 100) + '%;background:' + (x[0] === e.truth ? '#2E7D4F' : C.red) + ';border-radius:3px"></i></div><b style="font-size:12px;text-align:right">' + Math.round(x[1] * 100) + '%</b></div>'; }).join('') + '<p style="margin-top:10px"><b>' + (choice === e.truth ? 'Correct.' : 'The masked token was ' + esc(e.truth) + '.') + '</b> The reference predictor ' + (top[0][0] === e.truth ? 'was also correct.' : 'chose ' + esc(top[0][0]) + '.') + '</p><p>No measurement was required: the label came from the structure. A model trained on this task over millions of structures learns which atoms and bonds co-occur.</p>' : '<div class="ol">No measured value required</div><p>The label is the masked token itself. Use the neighbouring tokens, then select.</p><p style="color:' + C.mid + '">After you select, the probabilities of the reference predictor are shown, counted from 180 structures.</p>';
      qs('.sco', ctl).textContent = score.n ? 'You: ' + score.you + ' of ' + score.n + '. Reference predictor: ' + score.model + ' of ' + score.n + '.' : '';
      qsa('.opt', cp).forEach(function(b){ b.addEventListener('click', function(){ if (answered) return; answered = true; var ch = b.dataset.o; score.n++; if (ch === ex.truth) score.you++; if (ex.dist[0][0] === ex.truth) score.model++; show(ch); }); });
    }
    function next(){ seed++; ex = pickEx(); answered = false; if (ex) show(); }
    qs('[data-next]', ctl).addEventListener('click', next);
    async function render(){ await loadStructs(); b2Build(); if (!ex) ex = pickEx(); show(answered ? ex.truth : undefined); }
    async function play(tok){
      await render(); score = { you: 0, model: 0, n: 0 }; next(); api.cap('One atom token is masked. The structure contains its own label.'); await api.sleep(3200);
      var b = qsa('.opt', cp).filter(function(x){ return x.dataset.o === ex.truth; })[0]; if (b) b.click(); api.cap('No measured value was needed. Any structure can be converted into a masked-token example.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: labelled or unlabelled ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'Labelled or unlabelled', title: 'Molecules without lab values are far easier to find',
  change: 'Compare the red and grey dots. Press Hide labelled compounds to remove the red dots.',
  watch: 'A labelled compound has a measured value for the endpoint, here a clearance or a logD. Each label requires an assay, so labelled data sets are small. An unlabelled molecule has only a structure, and structures are available in very large collections. Each dot below represents 2,500 molecules. Grey dots: the roughly 10 million unlabelled molecules used to pre-train GROVER (Rong et al., 2020). Red dots: the labelled compounds in the two public sets used in this course, rounded up to whole dots. Pre-training uses the grey set first, and the labelled compounds are then used for the specific endpoint.',
  build: function(api){
    var counts = { hlm: 3087, logd: 4200, big: 10000000 }, shown = 1, UNIT = 2500;
    var cp = api.el('pn', 'left:28px;top:20px;width:610px;height:500px', '<h4>One dot is 2,500 molecules</h4><canvas width="580" height="440"></canvas>');
    var tp = api.el('pn', 'left:658px;top:20px;width:274px;height:500px;overflow:auto', '<div class="ct" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm ghost" data-hide>Hide labelled compounds</button>');
    var pos = null;
    function layout(){ var r = rng(4), cols = 80, rows = 50, all = []; for (var i = 0; i < cols * rows; i++) all.push(i); for (var a = all.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = all[a]; all[a] = all[j]; all[j] = t; } return { cols: cols, rows: rows, hlm: all[0], logd: [all[1], all[2]], total: counts.big / UNIT }; }
    function draw(){
      if (!pos) pos = layout(); var g = cv(qs('canvas', cp)), x = g.x, cw = (g.w - 20) / pos.cols, rw = (g.h - 20) / pos.rows, i;
      var lab = {}; lab[pos.hlm] = 'h'; pos.logd.forEach(function(k){ lab[k] = 'l'; });
      for (i = 0; i < pos.cols * pos.rows; i++) { var c = i % pos.cols, r = Math.floor(i / pos.cols), cx = 10 + c * cw + cw / 2, cy = 10 + r * rw + rw / 2; if (lab[i] && shown) { x.fillStyle = lab[i] === 'h' ? C.red : C.dred; x.beginPath(); x.arc(cx, cy, 4.6, 0, 6.3); x.fill(); x.strokeStyle = '#fff'; x.lineWidth = 1.5; x.stroke(); } else { x.fillStyle = 'rgba(111,129,150,.55)'; x.beginPath(); x.arc(cx, cy, 2.2, 0, 6.3); x.fill(); } }
      qs('.ct', tp).innerHTML = '<div class="ol">Counts</div><div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><b style="color:' + C.red + '">HLM clearance</b><span>' + counts.hlm.toLocaleString() + '</span></div><div style="font-size:12px">compounds with a measured clearance</div></div><div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><b style="color:' + C.dred + '">logD</b><span>' + counts.logd.toLocaleString() + '</span></div><div style="font-size:12px">compounds with a measured logD</div></div><div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><b style="color:#6F8196">Structures only</b><span>about 10,000,000</span></div><div style="font-size:12px">molecules without measured values, used to pre-train GROVER</div></div><p style="margin-top:12px">There are about <b>' + Math.round(counts.big / counts.hlm).toLocaleString() + '</b> unlabelled molecules for every labelled HLM compound.</p><p>Pre-training uses the grey set first. The few labelled compounds are then used for the specific endpoint.</p>';
    }
    qs('[data-hide]', ctl).addEventListener('click', function(){ shown = !shown; this.textContent = shown ? 'Hide labelled compounds' : 'Show labelled compounds'; draw(); });
    async function render(){ var h = await loadDataset('HLM'), l = await loadDataset('logD'); counts.hlm = h.N; counts.logd = l.N; draw(); }
    async function play(tok){
      await render(); shown = 0; qs('[data-hide]', ctl).textContent = 'Show labelled compounds'; draw(); api.cap('Each grey dot is 2,500 structures without measured values.'); await api.sleep(3400);
      shown = 1; qs('[data-hide]', ctl).textContent = 'Hide labelled compounds'; draw(); api.cap('The red dots are all compounds with measured values in the two public sets used here.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: three ways to use it ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'Three ways to use it', title: 'Three ways to put a pre-trained model to work',
  change: 'Select 50 or 1,000 labelled compounds and compare the three strategies.',
  watch: 'A pre-trained model has learned general structure, but it does not predict clearance. There are three ways to use it. Feature extraction: the model converts each molecule to a vector and makes no prediction itself. Frozen encoder: a small trainable head is added on top of the vector and the pre-trained weights stay unchanged. Full fine-tuning: the pre-trained weights and the head are updated together on the labelled compounds. The bars show test R2 from saved Chemprop runs on HLM clearance, each the mean of two runs. R2 of 1 is perfect and 0 equals predicting the mean. The encoder was pre-trained on logD data.',
  build: function(api){
    var NS = [50, 1000], sel = 0;
    var bp = api.el('pn', 'left:28px;top:20px;width:560px;height:500px;overflow:auto', '<h4>Test score on HLM clearance</h4><div class="bb"></div>');
    var tp = api.el('pn', 'left:608px;top:20px;width:324px;height:500px;overflow:auto', '<div class="tt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Labelled compounds ' + inSeg(NS.map(function(n){ return n.toLocaleString(); }), sel, 'data-n'));
    function avg(mode, n){ var r = EXTRAS.chemprop_ft.runs.filter(function(x){ return x.mode === mode && x.n === n; }); return mean(r.map(function(x){ return x.r2; })); }
    function row(label, sub, v, col){ var lo = -0.6, hi = 0.6, zero = (0 - lo) / (hi - lo) * 100, w = Math.abs(v) / (hi - lo) * 100; return '<div style="margin:14px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="font-size:26px;margin:0;color:' + col + '">' + v.toFixed(2) + '</span></div><div style="font-size:12px;margin-bottom:4px">' + sub + '</div><div style="position:relative;height:16px;background:' + C.light + ';border-radius:4px"><i style="position:absolute;top:0;height:16px;background:' + col + ';border-radius:3px;' + (v >= 0 ? 'left:' + zero + '%;width:' + w + '%' : 'right:' + (100 - zero) + '%;width:' + w + '%') + '"></i><i style="position:absolute;top:-2px;left:' + zero + '%;height:20px;width:1px;background:' + C.char + '"></i></div></div>'; }
    function draw(){
      if (!EXTRAS) return; var n = NS[sel], a = avg('scratch', n), f = avg('frozen', n), u = avg('full', n);
      qs('.bb', bp).innerHTML = '<div class="ol">With ' + n.toLocaleString() + ' labelled compounds</div><div style="border:1.5px dashed ' + C.silver + ';border-radius:8px;padding:8px 12px;margin:8px 0;font-size:12.5px"><b>1. Feature extraction.</b> The model returns a vector for each molecule. It does not predict clearance, so there is no score.</div>' + row('2. Frozen encoder plus head', 'Pre-trained weights unchanged', f, '#6F8196') + row('3. Full fine-tuning', 'All weights updated together', u, C.red) + row('Baseline: training from scratch', 'No pre-training', a, C.char) + '<div style="font-size:11px;color:' + C.mid + '">The vertical line marks R2 of 0. A bar to its left is worse than predicting the mean.</div>';
      qs('.tt', tp).innerHTML = '<div class="ol">Observations</div>' + (n === 50 ? '<p>With 50 labelled compounds the frozen encoder with a new head scores ' + f.toFixed(2) + ', which is worse than predicting the mean. Fifty compounds are too few to fit even the head reliably.</p><p>Full fine-tuning (' + u.toFixed(2) + ') scores above training from scratch (' + a.toFixed(2) + '), but all three are weak.</p>' : '<p>With 1,000 labelled compounds full fine-tuning scores highest (' + u.toFixed(2) + ' against ' + a.toFixed(2) + ' from scratch).</p><p>The frozen encoder with a new head (' + f.toFixed(2) + ') scores above scratch and is cheaper to train, because only the head is updated.</p>') + '<p style="color:' + C.mid + ';font-size:12px">Saved runs of Chemprop on HLM clearance. The pre-trained encoder was trained on logD first.</p>';
    }
    qsa('[data-n] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.v; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ await loadExtras(); draw(); }
    async function play(tok){
      await render(); for (var k = 0; k < NS.length; k++) { api.checkTok(tok); sel = k; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); draw(); api.cap(k === 0 ? 'At 50 labelled compounds all three are weak, and the frozen encoder scores below zero.' : 'At 1,000 labelled compounds full fine-tuning scores highest, and the frozen encoder exceeds scratch.'); await api.sleep(3600); } api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- guide entries for the new steps ---------- */
GUIDE.steps['gpt/Cutting into pieces'] = { notice: 'The same token types appear in every SMILES.', q: 'What is a token?', a: 'The smallest unit the model reads: an atom, a bond symbol, a bracket, a parenthesis or a ring-closure digit.' };
GUIDE.steps['gpt/A number for every piece'] = { notice: 'Tokens used in similar contexts have similar vectors.', q: 'Why map each token to a vector instead of using its ID directly?', a: 'An ID is an arbitrary label with no structure. A vector can encode how the token is used, so similar tokens receive similar vectors.' };
GUIDE.steps['gpt/Why memory matters'] = { notice: 'The fraction of strings with every ring closed rises with context length.', q: 'Why does a short context produce many invalid strings?', a: 'The model cannot track which rings and branches are open, so it does not close them.' };
GUIDE.steps['pre/Fill in the blank'] = { notice: 'The label is always part of the structure itself.', q: 'Why does this task need no measured values?', a: 'The masked token is the label, and the structure already contains it.' };
GUIDE.steps['pre/Labelled or unlabelled'] = { notice: 'The red dots are rounded up to whole dots. They remain very few.', q: 'Why are unlabelled molecules far easier to obtain than labelled compounds?', a: 'A structure needs no assay. A label needs a measurement for every compound.' };
GUIDE.steps['pre/Three ways to use it'] = { notice: 'At 50 compounds the frozen encoder scores below zero.', q: 'Which strategy scores highest with 1,000 labelled compounds?', a: 'Full fine-tuning, which scores above both the frozen encoder and training from scratch.' };
