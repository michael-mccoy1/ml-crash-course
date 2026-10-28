"use strict";
/* Extra beginner steps. Written for readers with no machine learning experience and minimal chemistry. */

/* ---------- shared helpers for these steps ---------- */
function b2Toks(smi){ return smi.match(TOK) || []; }
function b2Col(t){
  if (/^\d$|^%\d\d$/.test(t)) return ['#2E7D4F', 'ring number'];
  if (t === '(' || t === ')') return [C.bg, 'branch'];
  if (/^[=#\-\/\\]$/.test(t)) return [C.tint, 'bond sign'];
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
  change: 'Pick a molecule, or type your own line of text, and watch it being cut into pieces.',
  watch: 'A model that works with text cannot read a whole word or sentence at once. It first cuts the text into small pieces, and each piece is called a token. A molecule is written as a line of text, so it is cut the same way. In this text, a letter such as C, N or O is an atom: carbon, nitrogen or oxygen. A lower-case letter means the atom sits in a flat ring. A number marks where a ring opens and closes. Brackets mark a side branch. An equals sign marks a double bond. Each colour below is one kind of token. The cutting is the same for every molecule, so a model sees every molecule as a row of tokens.',
  build: function(api){
    var cur = 3, smi = IN_MOLS[3].s;
    var tp = api.el('pn', 'left:28px;top:20px;width:904px;height:176px;overflow:auto', '<h4>The molecule as text, cut into tokens</h4><div class="chips" style="display:block;margin-top:6px"></div>');
    var lp = api.el('pn', 'left:28px;top:208px;width:440px;height:312px;overflow:auto', '<div class="lf" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var dp = api.el('pn', 'left:488px;top:208px;width:444px;height:312px', '<h4>The same molecule as a drawing</h4><div class="struct" style="height:262px"></div>');
    var ctl = api.addControl('Molecule ' + inSeg(IN_MOLS.map(function(m){ return m.n; }), cur, 'data-m') + ' or type one <input type="text" class="ti" style="width:230px;margin-left:6px" spellcheck="false">');
    var ti = qs('.ti', ctl); ti.value = smi;
    function draw(){
      var tk = b2Toks(smi), kinds = {}; tk.forEach(function(t){ kinds[b2Col(t)[1]] = 1; });
      qs('.chips', tp).innerHTML = tk.length ? tk.map(function(t){ return b2Chip(t); }).join('') : '<span class="note">Type some text, such as CCO.</span>';
      var legend = [['carbon', C.char], ['nitrogen', C.red], ['oxygen', C.dred], ['ring number', '#2E7D4F'], ['branch', C.bg], ['bond sign', C.tint], ['other atom', '#6F8196']].map(function(x){ return '<span style="display:inline-block;margin:2px 8px 2px 0;white-space:nowrap"><i style="display:inline-block;width:12px;height:12px;border-radius:3px;border:2px solid ' + x[1] + ';vertical-align:-2px"></i> ' + x[0] + '</span>'; }).join('');
      var distinct = {}; tk.forEach(function(t){ distinct[t] = 1; });
      qs('.lf', lp).innerHTML = '<div class="ol">Counting the pieces</div><div style="display:flex;gap:26px"><div><div class="big" style="color:' + C.red + ';font-size:34px">' + tk.length + '</div><div style="font-size:12px">tokens in this molecule</div></div><div><div class="big" style="font-size:34px">' + Object.keys(distinct).length + '</div><div style="font-size:12px">different tokens</div></div></div><p style="margin-top:10px">A token is one small piece of the text: an atom, a bond sign, a bracket or a ring number.</p><div class="ol" style="margin-top:8px">Colour key</div><div style="font-size:12px">' + legend + '</div><p style="margin-top:8px;font-size:12px;color:' + C.mid + '">Letters that are not carbon, nitrogen or oxygen are other atoms, such as Cl for chlorine and F for fluorine. Hydrogen atoms are not written. They fill the gaps.</p>';
      var box = qs('.struct', dp); if (!RDK) { box.innerHTML = ''; return; }
      if (smi && rdkValid(smi)) box.innerHTML = rdkSvg(smi, 400, 250); else box.innerHTML = '<p class="note" style="padding:20px">' + (smi ? 'The chemistry program RDKit cannot read this text as a molecule, so there is no drawing. It is still cut into tokens, because cutting only looks at the text.' : 'Nothing to draw yet.') + '</p>';
    }
    qsa('[data-m] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.v; smi = IN_MOLS[cur].s; ti.value = smi; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    ti.addEventListener('input', function(){ smi = ti.value.trim(); qsa('[data-m] button', ctl).forEach(function(x){ x.classList.remove('on'); }); draw(); });
    async function render(){ await inReady(api); draw(); }
    async function play(tok){
      await render(); var order = [0, 4, 3, 6];
      for (var k = 0; k < order.length; k++) { api.checkTok(tok); cur = order[k]; smi = IN_MOLS[cur].s; ti.value = smi; qsa('[data-m] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === cur); }); draw(); api.cap(k === 0 ? 'Ethanol is only three tokens: carbon, carbon, oxygen.' : (k === 1 ? 'Paracetamol has a ring. The ring opens with the number 1 and closes with the number 1.' : (k === 2 ? 'Caffeine has two rings and many side branches, so it has many more tokens.' : 'Every molecule becomes a row of tokens. The model reads that row.'))); await api.sleep(3200); }
      api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 04 GPT: a number for every piece ---------- */
var B2E = { done: false, toks: [], prof: {}, id: {}, classes: ['an atom', 'a ring number', 'a branch opening', 'a branch closing', 'a bond sign', 'a special atom', 'the end'] };
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
  change: 'Click a piece in the list, or press Next piece, and compare its list of numbers with the others.',
  watch: 'A model can only work with numbers, so each token is turned into numbers in two stages. First, every token gets an ID number, like a seat number. The most common token gets 1, the next gets 2, and so on. An ID number says nothing about the token itself. So the model also gives each token a short list of numbers that describes how it behaves. Pieces that behave alike get alike lists. Here we describe each token by what usually comes right after it: an atom, a ring number, the end, and so on. The bars show those shares. A real GPT does not count by hand. It learns its own lists during training. These lists are called embeddings.',
  build: function(api){
    var cur = 0;
    var lp = api.el('pn', 'left:28px;top:20px;width:470px;height:500px;overflow:auto', '<h4>The 14 most common pieces in 180 real molecules</h4><div class="lst"></div>');
    var rp = api.el('pn', 'left:518px;top:20px;width:414px;height:500px;overflow:auto', '<div class="det" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm ghost" data-next>Next piece</button>');
    function spark(p, col){ return '<span style="display:inline-flex;gap:2px;align-items:flex-end;height:24px">' + p.map(function(v){ return '<i style="display:block;width:7px;height:' + Math.max(2, v * 24) + 'px;background:' + col + ';border-radius:1px"></i>'; }).join('') + '</span>'; }
    function draw(){
      var T = B2E.toks, t = T[cur], p = B2E.prof[t];
      qs('.lst', lp).innerHTML = '<div style="display:grid;grid-template-columns:44px 74px 1fr 90px;gap:8px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:' + C.mid + ';font-weight:700;margin:6px 0 2px"><span>ID</span><span>Piece</span><span>Its list of numbers</span><span style="text-align:right">Times seen</span></div>' + T.map(function(x, i){ return '<div class="rw" data-i="' + i + '" style="display:grid;grid-template-columns:44px 74px 1fr 90px;gap:8px;align-items:center;padding:3px 4px;border-radius:5px;cursor:pointer;border:1.5px solid ' + (i === cur ? C.red : 'transparent') + ';background:' + (i === cur ? '#FBE9E9' : '#fff') + '"><b>' + B2E.id[x] + '</b><span>' + b2Chip(x, 'margin:0;padding:1px 8px;font-size:13px') + '</span>' + spark(B2E.prof[x], i === cur ? C.red : C.char) + '<span style="text-align:right;font-size:12px">' + B2E.cnt[x].toLocaleString() + '</span></div>'; }).join('');
      var sims = T.map(function(x, i){ return i === cur ? null : { x: x, s: b2Cos(p, B2E.prof[x]) }; }).filter(Boolean).sort(function(a, b){ return b.s - a.s; });
      qs('.det', rp).innerHTML = '<div class="ol">The piece</div><div style="display:flex;align-items:center;gap:14px">' + b2Chip(t, 'font-size:24px;padding:6px 14px') + '<div><div style="font-size:12px">ID number</div><div class="big" style="font-size:30px;color:' + C.red + ';margin:0">' + B2E.id[t] + '</div></div></div>' +
        '<div class="ol" style="margin-top:14px">What usually comes right after it</div>' + B2E.classes.map(function(c, k){ return '<div style="display:grid;grid-template-columns:120px 1fr 40px;gap:8px;align-items:center;margin:4px 0;font-size:12.5px"><span>' + c + '</span><div style="height:12px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:12px;width:' + (p[k] * 100) + '%;background:' + C.red + ';border-radius:3px"></i></div><b style="text-align:right">' + Math.round(p[k] * 100) + '%</b></div>'; }).join('') +
        '<div class="ol" style="margin-top:14px">Pieces that behave most like it</div>' + sims.slice(0, 3).map(function(s){ return '<div style="display:flex;align-items:center;gap:10px;margin:4px 0">' + b2Chip(s.x, 'margin:0;padding:1px 8px;font-size:13px') + '<span style="font-size:12.5px">' + Math.round(Math.max(0, s.s) * 100) + '% alike</span></div>'; }).join('') +
        '<p style="margin-top:12px;font-size:12.5px">Pieces that behave alike get alike lists of numbers. This is how a model can tell which pieces are similar without being told any chemistry.</p>';
      qsa('.rw', lp).forEach(function(e){ e.addEventListener('click', function(){ cur = +e.dataset.i; draw(); }); });
    }
    qs('[data-next]', ctl).addEventListener('click', function(){ cur = (cur + 1) % B2E.toks.length; draw(); });
    async function render(){ await loadStructs(); b2Build(); b2EmbBuild(); draw(); }
    async function play(tok){
      await render(); cur = 0; draw(); api.cap('Each piece has an ID number, a seat number that says nothing about the piece.'); await api.sleep(3200);
      for (var k = 0; k < 4; k++) { api.checkTok(tok); cur = [0, 1, 2, 3][k]; draw(); api.cap(k === 0 ? 'And it has a short list of numbers that describes how it behaves.' : 'Compare the bars. Pieces that behave alike have alike bars.'); await api.sleep(2800); }
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
  change: 'Choose how many earlier pieces the model remembers, and compare how many of its molecules are real.',
  watch: 'When the model picks the next piece, it can look back at the pieces before it. This is its memory. Memory matters in chemistry. A ring opens with a number, and the same number must appear again to close it. A bracket that opens must close. A model that remembers only the last piece soon forgets that a ring is still open. Here the same counting model, trained on 180 real molecules, writes 100 new strings with a memory of 1, 3, 6 and 8 pieces. The first bar counts strings in which every ring and every branch is closed. The chemistry program RDKit then checks each string. A string that RDKit can read as a molecule is called valid. A very long memory can also copy a molecule it has already seen, so the last bar counts only valid molecules that are not copies. The numbers are low because 180 molecules is a tiny training set. A real GPT learns from millions.',
  build: function(api){
    var KS = [1, 3, 6, 8], cur = 0;
    var lp = api.el('pn', 'left:28px;top:20px;width:540px;height:500px;overflow:auto', '<div class="bars"></div><div class="smp"></div>');
    var rp = api.el('pn', 'left:588px;top:20px;width:344px;height:500px;overflow:auto', '<h4 class="rh">A valid new molecule it wrote</h4><div class="struct" style="height:190px"></div><div class="rt" style="font-size:13px;line-height:1.5;color:' + C.char + ';margin-top:8px"></div>');
    var ctl = api.addControl('Memory ' + inSeg(KS.map(function(k){ return k + (k === 1 ? ' piece' : ' pieces'); }), cur, 'data-k'));
    function bar(label, v, col){ return '<div style="display:grid;grid-template-columns:150px 1fr 44px;gap:8px;align-items:center;margin:5px 0;font-size:12.5px"><span>' + label + '</span><div style="height:14px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:14px;width:' + (v * 100) + '%;background:' + col + ';border-radius:3px;transition:width .4s"></i></div><b style="text-align:right">' + Math.round(v * 100) + '%</b></div>'; }
    function draw(){
      var rows = KS.map(function(k, i){ var r = b3Run(k); return '<div style="border:1.5px solid ' + (i === cur ? C.red : C.silver) + ';border-radius:8px;padding:8px 12px;margin-bottom:8px;background:' + (i === cur ? '#FBE9E9' : '#fff') + '"><div style="font-weight:700;margin-bottom:2px">Memory of ' + k + (k === 1 ? ' piece' : ' pieces') + '</div>' + bar('rings and branches closed', r.closed, '#6F8196') + bar('valid molecules', r.valid, '#2E7D4F') + bar('valid and not a copy', r.validNew, C.red) + '</div>'; }).join('');
      qs('.bars', lp).innerHTML = '<h4>100 strings written at each memory size</h4><div style="margin-top:8px">' + rows + '</div>';
      var r = b3Run(KS[cur]);
      qs('.smp', lp).innerHTML = '<div class="ol" style="margin-top:4px">Six written with a memory of ' + KS[cur] + (KS[cur] === 1 ? ' piece' : ' pieces') + '</div>' + r.samples.map(function(s){ return '<div style="display:flex;gap:8px;align-items:flex-start;margin:4px 0;font-size:12.5px"><b style="color:' + (s.ok ? '#1E6B40' : C.red) + ';min-width:96px">' + (s.ok ? (s.copy ? 'valid, a copy' : 'valid, new') : (s.closed ? 'closed, not valid' : 'not closed')) + '</b><span style="font-family:Cascadia Mono,Consolas,monospace;word-break:break-all">' + esc(s.s || '(empty)') + '</span></div>'; }).join('');
      var box = qs('.struct', rp); box.innerHTML = r.first ? rdkSvg(r.first, 300, 180) : ''; qs('.rt', rp).innerHTML = r.first ? '<span style="font-family:Cascadia Mono,Consolas,monospace;word-break:break-all">' + esc(r.first) + '</span><p style="margin-top:8px">' + (KS[cur] === 1 ? 'With a memory of one piece the model forgets what it opened, so few strings are valid.' : (KS[cur] >= 8 ? 'With a long memory more strings close their rings and branches and are valid. Some are copies of molecules the model has already seen.' : 'A longer memory helps the model remember to close its rings and branches.')) + '</p>' : '<p>With a memory this short, none of the 100 strings was a valid new molecule. Most never closed their rings and branches. Try a longer memory.</p>';
    }
    qsa('[data-k] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ cur = +b.dataset.v; qsa('[data-k] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ await loadStructs(); await inReady(api); b2Build(); api.busy('Writing and checking 400 strings'); await tick(); await tick(); KS.forEach(b3Run); api.idle(); draw(); }
    async function play(tok){
      await render(); for (var k = 0; k < KS.length; k++) { api.checkTok(tok); cur = k; qsa('[data-k] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === cur); }); draw(); api.cap(k === 0 ? 'With a memory of one piece, the model forgets that a ring is open.' : (k === KS.length - 1 ? 'A long memory closes more rings and writes more valid molecules. A real GPT needs far more molecules to do this well.' : 'More memory, more strings with every ring closed.')); await api.sleep(3000); } api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: fill in the blank ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'Fill in the blank', title: 'A molecule can test itself: hide a piece and guess it',
  change: 'Pick the piece you think was hidden. Press Next molecule for another one.',
  watch: 'To learn from a molecule, a model needs a question with a known answer. A molecule can supply its own. Hide one piece of its text, and ask what belongs in the gap. The answer is the piece that was hidden, so no experiment and no lab value is needed. Any molecule from any collection becomes a free exercise. This is called self-supervised learning, because the data supervises itself. Below, one atom is hidden. Look at the pieces on either side, and then choose. The computer makes its own guess from counts taken over 180 real molecules.',
  build: function(api){
    var seed = 1, score = { you: 0, model: 0, n: 0 }, ex = null, answered = false;
    var tp = api.el('pn', 'left:28px;top:20px;width:904px;height:130px', '<h4>A molecule with one piece hidden</h4><div class="chips" style="display:block;margin-top:6px"></div>');
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
      qs('.ch', cp).innerHTML = '<div class="ol">What belongs in the gap?</div><p style="font-size:12.5px;margin:2px 0 8px">The piece before it is ' + b2Chip(e.tk[e.at - 1], 'margin:0 4px;padding:1px 7px;font-size:13px') + ' and the piece after it is ' + b2Chip(e.tk[e.at + 1], 'margin:0 4px;padding:1px 7px;font-size:13px') + '.</p><div style="display:flex;flex-wrap:wrap;gap:8px">' + e.opts.map(function(o){ var c = b2Col(o)[0], st = shown ? (o === e.truth ? 'background:#EAF5EE;border-color:#2E7D4F' : (o === choice ? 'background:#FCECEC;border-color:' + C.red : '')) : ''; return '<button type="button" class="opt" data-o="' + esc(o) + '" style="font:700 18px Cascadia Mono,Consolas,monospace;color:' + c + ';border:2px solid ' + c + ';border-radius:6px;background:#fff;padding:8px 18px;cursor:' + (shown ? 'default' : 'pointer') + ';' + st + '"' + (shown ? ' disabled' : '') + '>' + esc(o) + '</button>'; }).join('') + '</div>';
      var top = e.dist.slice(0, 5);
      qs('.rs', rp).innerHTML = shown ? '<div class="ol">The computer\'s guess</div>' + top.map(function(x){ return '<div style="display:grid;grid-template-columns:56px 1fr 40px;gap:8px;align-items:center;margin:5px 0"><span>' + b2Chip(x[0], 'margin:0;padding:1px 8px;font-size:13px') + '</span><div style="height:13px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:13px;width:' + (x[1] * 100) + '%;background:' + (x[0] === e.truth ? '#2E7D4F' : C.red) + ';border-radius:3px"></i></div><b style="font-size:12px;text-align:right">' + Math.round(x[1] * 100) + '%</b></div>'; }).join('') + '<p style="margin-top:10px"><b>' + (choice === e.truth ? 'You got it.' : 'The hidden piece was ' + esc(e.truth) + '.') + '</b> The computer ' + (top[0][0] === e.truth ? 'got it too.' : 'guessed ' + esc(top[0][0]) + '.') + '</p><p>Nobody had to measure anything. The molecule held the answer. A model that plays this game on millions of molecules learns which atoms and bonds go together.</p>' : '<div class="ol">No lab value needed</div><p>The answer is already in the molecule. Look at the pieces on either side of the gap, then choose.</p><p style="color:' + C.mid + '">After you choose, the computer shows its own chances, counted from 180 real molecules.</p>';
      qs('.sco', ctl).textContent = score.n ? 'You: ' + score.you + ' of ' + score.n + '. Computer: ' + score.model + ' of ' + score.n + '.' : '';
      qsa('.opt', cp).forEach(function(b){ b.addEventListener('click', function(){ if (answered) return; answered = true; var ch = b.dataset.o; score.n++; if (ch === ex.truth) score.you++; if (ex.dist[0][0] === ex.truth) score.model++; show(ch); }); });
    }
    function next(){ seed++; ex = pickEx(); answered = false; if (ex) show(); }
    qs('[data-next]', ctl).addEventListener('click', next);
    async function render(){ await loadStructs(); b2Build(); if (!ex) ex = pickEx(); show(answered ? ex.truth : undefined); }
    async function play(tok){
      await render(); score = { you: 0, model: 0, n: 0 }; next(); api.cap('One piece of this molecule is hidden. The molecule itself knows the answer.'); await api.sleep(3200);
      var b = qsa('.opt', cp).filter(function(x){ return x.dataset.o === ex.truth; })[0]; if (b) b.click(); api.cap('No lab value was needed. Any molecule can be turned into a question like this.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: labelled or unlabelled ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'Labelled or unlabelled', title: 'Molecules without lab values are far easier to find',
  change: 'Look for the red dots. Press Walk through to see them appear.',
  watch: 'A labelled compound is one with a measured lab value, such as a clearance. Each label costs a lab test, so labelled sets are small. An unlabelled molecule has only its structure. A structure is just a drawing, and huge collections of structures already exist. The picture shows one dot for every 2,500 molecules. The grey dots stand for the roughly 10 million unlabelled molecules used to pre-train the model GROVER (Rong et al., 2020). The red dots are the labelled compounds in the two public sets used in this course. The red dots are rounded up to whole dots, so they look larger than they are. A pre-trained model gets to learn from all the grey dots first.',
  build: function(api){
    var counts = { hlm: 3087, logd: 4200, big: 10000000 }, shown = 1, UNIT = 2500;
    var cp = api.el('pn', 'left:28px;top:20px;width:610px;height:500px', '<h4>One dot is 2,500 molecules</h4><canvas width="580" height="440"></canvas>');
    var tp = api.el('pn', 'left:658px;top:20px;width:274px;height:500px;overflow:auto', '<div class="ct" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm ghost" data-hide>Hide the labelled dots</button>');
    var pos = null;
    function layout(){ var r = rng(4), cols = 80, rows = 50, all = []; for (var i = 0; i < cols * rows; i++) all.push(i); for (var a = all.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = all[a]; all[a] = all[j]; all[j] = t; } return { cols: cols, rows: rows, hlm: all[0], logd: [all[1], all[2]], total: counts.big / UNIT }; }
    function draw(){
      if (!pos) pos = layout(); var g = cv(qs('canvas', cp)), x = g.x, cw = (g.w - 20) / pos.cols, rw = (g.h - 20) / pos.rows, i;
      var lab = {}; lab[pos.hlm] = 'h'; pos.logd.forEach(function(k){ lab[k] = 'l'; });
      for (i = 0; i < pos.cols * pos.rows; i++) { var c = i % pos.cols, r = Math.floor(i / pos.cols), cx = 10 + c * cw + cw / 2, cy = 10 + r * rw + rw / 2; if (lab[i] && shown) { x.fillStyle = lab[i] === 'h' ? C.red : C.dred; x.beginPath(); x.arc(cx, cy, 4.6, 0, 6.3); x.fill(); x.strokeStyle = '#fff'; x.lineWidth = 1.5; x.stroke(); } else { x.fillStyle = 'rgba(111,129,150,.55)'; x.beginPath(); x.arc(cx, cy, 2.2, 0, 6.3); x.fill(); } }
      qs('.ct', tp).innerHTML = '<div class="ol">The counts</div><div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><b style="color:' + C.red + '">HLM clearance</b><span>' + counts.hlm.toLocaleString() + '</span></div><div style="font-size:12px">compounds with a measured clearance</div></div><div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><b style="color:' + C.dred + '">logD</b><span>' + counts.logd.toLocaleString() + '</span></div><div style="font-size:12px">compounds with a measured logD, a cousin of logP that is cheaper to measure</div></div><div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><b style="color:#6F8196">Structures only</b><span>about 10,000,000</span></div><div style="font-size:12px">molecules with no lab value, used to pre-train GROVER</div></div><p style="margin-top:12px">There are about <b>' + Math.round(counts.big / counts.hlm).toLocaleString() + '</b> unlabelled molecules for every labelled HLM compound.</p><p>Pre-training lets a model learn from the grey dots first, then use the few red dots for the specific question.</p>';
    }
    qs('[data-hide]', ctl).addEventListener('click', function(){ shown = !shown; this.textContent = shown ? 'Hide the labelled dots' : 'Show the labelled dots'; draw(); });
    async function render(){ var h = await loadDataset('HLM'), l = await loadDataset('logD'); counts.hlm = h.N; counts.logd = l.N; draw(); }
    async function play(tok){
      await render(); shown = 0; qs('[data-hide]', ctl).textContent = 'Show the labelled dots'; draw(); api.cap('Every grey dot is 2,500 molecules whose structure is known. None has a lab value.'); await api.sleep(3400);
      shown = 1; qs('[data-hide]', ctl).textContent = 'Hide the labelled dots'; draw(); api.cap('These few red dots are all the compounds with a measured value in the two public sets used here.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 05 Pre-trained models: three ways to use it ---------- */
LESSONS.pre.add({ level: 'beginner', short: 'Three ways to use it', title: 'Three ways to put a pre-trained model to work',
  change: 'Choose 50 or 1,000 labelled compounds and compare the three ways.',
  watch: 'A pre-trained model has learned general facts about molecules, but it does not yet predict clearance. There are three ways to use it. One: use it as it is, to turn each molecule into a list of numbers. On its own that gives no clearance. Two: add a small model on top that reads those numbers and predicts clearance, and leave the pre-trained part alone. Three: adjust everything, so the pre-trained part and the small model learn together from your labelled compounds. The bars show the real test scores (R2, where 1 is perfect and 0 is no better than the overall average) from saved runs on HLM clearance. Each bar is the average of two runs.',
  build: function(api){
    var NS = [50, 1000], sel = 0;
    var bp = api.el('pn', 'left:28px;top:20px;width:560px;height:500px;overflow:auto', '<h4>Test score on HLM clearance</h4><div class="bb"></div>');
    var tp = api.el('pn', 'left:608px;top:20px;width:324px;height:500px;overflow:auto', '<div class="tt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Labelled compounds ' + inSeg(NS.map(function(n){ return n.toLocaleString(); }), sel, 'data-n'));
    function avg(mode, n){ var r = EXTRAS.chemprop_ft.runs.filter(function(x){ return x.mode === mode && x.n === n; }); return mean(r.map(function(x){ return x.r2; })); }
    function row(label, sub, v, col){ var lo = -0.6, hi = 0.6, zero = (0 - lo) / (hi - lo) * 100, w = Math.abs(v) / (hi - lo) * 100; return '<div style="margin:14px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="font-size:26px;margin:0;color:' + col + '">' + v.toFixed(2) + '</span></div><div style="font-size:12px;margin-bottom:4px">' + sub + '</div><div style="position:relative;height:16px;background:' + C.light + ';border-radius:4px"><i style="position:absolute;top:0;height:16px;background:' + col + ';border-radius:3px;' + (v >= 0 ? 'left:' + zero + '%;width:' + w + '%' : 'right:' + (100 - zero) + '%;width:' + w + '%') + '"></i><i style="position:absolute;top:-2px;left:' + zero + '%;height:20px;width:1px;background:' + C.char + '"></i></div></div>'; }
    function draw(){
      if (!EXTRAS) return; var n = NS[sel], a = avg('scratch', n), f = avg('frozen', n), u = avg('full', n);
      qs('.bb', bp).innerHTML = '<div class="ol">With ' + n.toLocaleString() + ' labelled compounds</div><div style="border:1.5px dashed ' + C.silver + ';border-radius:8px;padding:8px 12px;margin:8px 0;font-size:12.5px"><b>1. Use it as it is.</b> The model turns each molecule into numbers. It gives no clearance on its own, so there is no score.</div>' + row('2. A small model on top', 'The pre-trained part is left alone', f, '#6F8196') + row('3. Adjust everything', 'The pre-trained part and the small model learn together', u, C.red) + row('For comparison: start from nothing', 'No pre-training at all', a, C.char) + '<div style="font-size:11px;color:' + C.mid + '">The line down the middle is a score of 0. A bar to the left of it is worse than guessing the overall average.</div>';
      qs('.tt', tp).innerHTML = '<div class="ol">What to notice</div>' + (n === 50 ? '<p>With only 50 labelled compounds, a small model on top does worse than guessing the overall average. Fifty compounds are too few to teach even a small model.</p><p>Adjusting everything still beats starting from nothing, but all three are weak. More labelled compounds are needed.</p>' : '<p>With 1,000 labelled compounds, adjusting everything is clearly best. It scores about twice as high as starting from nothing.</p><p>A small model on top is now useful, and it is cheaper, because only the small part is trained.</p>') + '<p style="color:' + C.mid + ';font-size:12px">These are saved runs of a neural network on HLM clearance. The pre-trained part learned from logD data first.</p>';
    }
    qsa('[data-n] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.v; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function render(){ await loadExtras(); draw(); }
    async function play(tok){
      await render(); for (var k = 0; k < NS.length; k++) { api.checkTok(tok); sel = k; qsa('[data-n] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); draw(); api.cap(k === 0 ? 'With 50 compounds, all three ways are weak, and a small model on top is worse than guessing the average.' : 'With 1,000 compounds, adjusting everything is best, and a small model on top has become useful.'); await api.sleep(3600); } api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- guide entries for the new steps ---------- */
GUIDE.steps['gpt/Cutting into pieces'] = { notice: 'The same few kinds of piece appear in every molecule.', q: 'What is a token?', a: 'One small piece of the text: an atom, a bond sign, a bracket or a ring number.' };
GUIDE.steps['gpt/A number for every piece'] = { notice: 'Pieces that behave alike have alike bars.', q: 'Why does a model give each piece a list of numbers and not just an ID?', a: 'An ID number says nothing about the piece. A list of numbers can show how the piece behaves, so alike pieces get alike lists.' };
GUIDE.steps['gpt/Why memory matters'] = { notice: 'The share of strings with every ring closed rises with memory.', q: 'Why does a short memory write so many strings that are not molecules?', a: 'It forgets that a ring or a bracket is still open, so it never closes it.' };
GUIDE.steps['pre/Fill in the blank'] = { notice: 'The answer is always in the molecule itself.', q: 'Why does this need no lab value?', a: 'The hidden piece is the answer, and the molecule already contains it.' };
GUIDE.steps['pre/Labelled or unlabelled'] = { notice: 'The red dots are rounded up. There are still very few of them.', q: 'Why are unlabelled molecules so much easier to find?', a: 'A structure is only a drawing. A label needs a lab test for every compound.' };
GUIDE.steps['pre/Three ways to use it'] = { notice: 'With 50 compounds, a small model on top scores below zero.', q: 'Which way scores best with 1,000 labelled compounds?', a: 'Adjusting everything, which scores about twice as high as starting from nothing.' };
