"use strict";
/* =====================================================================
   Model Lab core: utilities, datasets, metrics, tree engine, plotting,
   lesson framework. Everything runs in the browser.
   ===================================================================== */
var C = { red: '#E1242A', dred: '#891515', char: '#34373F', bg: '#A1B1C3', silver: '#D1D8E0', light: '#EDF2F3', tint: '#F4A6A8', mid: '#6B7280', white: '#FFFFFF' };
var $ = function(id){ return document.getElementById(id); };
var qs = function(s, r){ return (r || document).querySelector(s); };
var qsa = function(s, r){ return [].slice.call((r || document).querySelectorAll(s)); };
function h(tag, cls, html){ var e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t){ return a + (b - a) * t; }
function ease(t){ return t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
function mean(a){ var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }
function fix(v, n){ return (+v).toFixed(n); }
function rng(seed){ var s = seed >>> 0; return function(){ s = (s + 0x6D2B79F5) >>> 0; var t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function colorMix(t, a, b){ a = a || [209, 216, 224]; b = b || [225, 36, 42]; t = clamp(t, 0, 1); return 'rgb(' + a.map(function(x, i){ return Math.round(lerp(x, b[i], t)); }).join(',') + ')'; }
function esc(s){ return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

/* ---------- animation control ---------- */
var ANIM = true, AUTOPLAY = false, LAZY = !!(window.matchMedia && matchMedia('(max-width: 900px)').matches);
var tick = function(){ return new Promise(function(r){ setTimeout(r, 0); }); };
function progress(p){ var b = $('bar'); if (!b) return; b.style.width = (p * 100) + '%'; if (p >= 1) setTimeout(function(){ b.style.width = '0'; }, 400); }
function Cancel(){ this.cancel = true; }

/* ---------- datasets ---------- */
var DATA = {}, D = null, EP = 'HLM', EXTRAS = null, STRUCTS = null;
var FSEL = {};
function prepareDS(d){
  d.NF = d.cols.indexOf('y'); d.BASE = d.cols.slice(0, d.NF); d.FEATS = d.BASE.slice(); d.N = d.rows.length;
  d.COL = {}; d.BASE.forEach(function(f, j){ d.COL[f] = Float32Array.from(d.rows, function(r){ return r[j]; }); });
  d.Y = Float32Array.from(d.rows, function(r){ return r[d.NF]; });
  d.TRAIN = []; d.TEST = []; d.test.forEach(function(t, i){ (t ? d.TEST : d.TRAIN).push(i); });
  d.B = { edges: {}, codes: {} }; d.scores = {}; d.cache = {};
  d.unit = d.unit || '';
  d.BASE.forEach(function(f){ if (!(f in FSEL)) FSEL[f] = true; });
  ensureBins(d);
}
async function loadJSON(name){ var r = await fetch('data/' + name + '.json?v=e9'); if (!r.ok) throw new Error('Could not load data/' + name + '.json'); return r.json(); }
async function loadDataset(key){
  if (!DATA[key]) { var d = await loadJSON(key === 'HLM' ? 'hlm' : 'logd'); DATA[key] = d; prepareDS(d); }
  return DATA[key];
}
function ensureBins(d){
  d.FEATS.forEach(function(f){ if (d.B.edges[f]) return; var v = d.TRAIN.map(function(i){ return d.COL[f][i]; }).sort(function(a, b){ return a - b; }), e = [], nb = 32;
    for (var q = 1; q < nb; q++) { var x = v[Math.floor(q / nb * v.length)]; if (!e.length || x > e[e.length - 1]) e.push(x); }
    d.B.edges[f] = e; var c = new Uint8Array(d.N); for (var i = 0; i < d.N; i++) { var val = d.COL[f][i], k = 0; while (k < e.length && val > e[k]) k++; c[i] = k; } d.B.codes[f] = c; });
}
function addFeature(d, name, values){ d.COL[name] = Float32Array.from(values); if (d.FEATS.indexOf(name) < 0) d.FEATS.push(name); if (!(name in FSEL)) FSEL[name] = false; delete d.B.edges[name]; ensureBins(d); }
function activeFeats(){ return D.FEATS.filter(function(f){ return FSEL[f]; }); }
function baseFeats(){ return D.BASE.slice(); }
function codeOf(B, f, v){ var e = B.edges[f], k = 0; while (k < e.length && v > e[k]) k++; return k; }

/* ---------- metrics (current dataset) ---------- */
function r2(idx, pred, Y){ Y = Y || D.Y; var ym = 0; idx.forEach(function(i){ ym += Y[i]; }); ym /= idx.length; var a = 0, b = 0; idx.forEach(function(i){ a += (Y[i] - pred[i]) * (Y[i] - pred[i]); b += (Y[i] - ym) * (Y[i] - ym); }); return b ? 1 - a / b : 0; }
function rmse(idx, pred, Y){ Y = Y || D.Y; var a = 0; idx.forEach(function(i){ a += (Y[i] - pred[i]) * (Y[i] - pred[i]); }); return Math.sqrt(a / idx.length); }
function ranks(v){ var o = v.map(function(x, i){ return [x, i]; }).sort(function(p, q){ return p[0] - q[0]; }), r = new Array(v.length); for (var k = 0; k < o.length;) { var m = k; while (m + 1 < o.length && o[m + 1][0] === o[k][0]) m++; var avg = (k + m) / 2; for (var t = k; t <= m; t++) r[o[t][1]] = avg; k = m + 1; } return r; }
function spearman(idx, pred, Y){ Y = Y || D.Y; var a = ranks(idx.map(function(i){ return Y[i]; })), b = ranks(idx.map(function(i){ return pred[i]; })), ma = mean(a), mb = mean(b), n = 0, da = 0, db = 0; for (var i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) * (a[i] - ma); db += (b[i] - mb) * (b[i] - mb); } return db === 0 ? 0 : n / Math.sqrt(da * db); }

/* ---------- data filters (shared by every lesson, applied to training only) ---------- */
var FILT = { cens: 'keep', z: 0, mw: null, lp: null };
function filterOn(f){ f = f || FILT; return f.cens !== 'keep' || f.z > 0 || !!f.mw || !!f.lp; }
function passFilter(i, isTest, f){
  f = f || FILT; var d = D;
  if (d.cens && f.cens !== 'keep' && d.cens[i] && (!isTest || f.cens === 'both')) return false;
  if (f.z > 0 && !isTest) { if (!d.zstat) { var ys = d.TRAIN.map(function(k){ return d.Y[k]; }), m = mean(ys), sd = Math.sqrt(mean(ys.map(function(v){ return (v - m) * (v - m); }))) || 1; d.zstat = { m: m, sd: sd }; } if (Math.abs(d.Y[i] - d.zstat.m) / d.zstat.sd > f.z) return false; }
  if (f.mw && (d.COL.MW[i] < f.mw[0] || d.COL.MW[i] > f.mw[1])) return false;
  if (f.lp && (d.COL.logP[i] < f.lp[0] || d.COL.logP[i] > f.lp[1])) return false;
  return true;
}
function filteredIdx(isTest, f){ return (isTest ? D.TEST : D.TRAIN).filter(function(i){ return passFilter(i, isTest, f); }); }
function activeTrain(f){ f = f || FILT; return filterOn(f) ? filteredIdx(false, f) : D.TRAIN; }
function filtKey(f){ f = f || FILT; return filterOn(f) ? JSON.stringify([f.cens, f.z, f.mw, f.lp]) : '0'; }

/* ---------- tree engine ---------- */
function buildTree(B, feats, target, idx, depth, minLeaf, mtry, rand){
  var n = idx.length, sum = 0; for (var i = 0; i < n; i++) sum += target[idx[i]];
  var leaf = { v: sum / n, n: n };
  if (depth === 0 || n < 2 * minLeaf) return leaf;
  var fs = feats; if (mtry < feats.length) { fs = feats.slice(); for (var a = fs.length - 1; a > 0; a--) { var j = Math.floor(rand() * (a + 1)), t = fs[a]; fs[a] = fs[j]; fs[j] = t; } fs = fs.slice(0, mtry); }
  var best = 1e-9, bf = null, bb = -1, base = sum * sum / n;
  fs.forEach(function(f){ var c = B.codes[f], nbins = B.edges[f].length + 1, cnt = new Float64Array(nbins), sm = new Float64Array(nbins);
    for (var i = 0; i < n; i++) { var k = c[idx[i]]; cnt[k]++; sm[k] += target[idx[i]]; }
    var cl = 0, sl = 0; for (var b = 0; b < nbins - 1; b++) { cl += cnt[b]; sl += sm[b]; var cr = n - cl; if (cl < minLeaf || cr < minLeaf) continue; var sr = sum - sl, gain = sl * sl / cl + sr * sr / cr - base; if (gain > best) { best = gain; bf = f; bb = b; } } });
  if (bf === null) return leaf;
  var c2 = B.codes[bf], L = [], R = []; for (var q = 0; q < n; q++) (c2[idx[q]] <= bb ? L : R).push(idx[q]);
  return { f: bf, b: bb, v: leaf.v, n: n, g: best, l: buildTree(B, feats, target, L, depth - 1, minLeaf, mtry, rand), r: buildTree(B, feats, target, R, depth - 1, minLeaf, mtry, rand) };
}
function predictTree(node, B, i){ while (node.f !== undefined) node = B.codes[node.f][i] <= node.b ? node.l : node.r; return node.v; }
function predictVals(node, B, vals){ while (node.f !== undefined) node = codeOf(B, node.f, vals[node.f]) <= node.b ? node.l : node.r; return node.v; }
function treeAll(tree, N){ var p = new Float32Array(N); for (var i = 0; i < N; i++) p[i] = predictTree(tree, D.B, i); return p; }
function addGain(node, acc){ if (node.f === undefined) return; acc[node.f] = (acc[node.f] || 0) + node.g; addGain(node.l, acc); addGain(node.r, acc); }

function fitTree(feats, idx, depth, minLeaf, seed){ return buildTree(D.B, feats, D.Y, Int32Array.from(idx), depth, minLeaf || 5, feats.length, rng(seed || 1)); }
function fitForest(feats, idx, n, depth, seed){
  var t0 = performance.now(), rand = rng(seed || 7), tr = Int32Array.from(idx), mtry = Math.max(1, Math.round(Math.sqrt(feats.length))), trees = [], per = [];
  for (var q = 0; q < n; q++) { var boot = new Int32Array(tr.length); for (var z = 0; z < tr.length; z++) boot[z] = tr[Math.floor(rand() * tr.length)]; var t = buildTree(D.B, feats, D.Y, boot, depth, 3, mtry, rand); trees.push(t); per.push(treeAll(t, D.N)); }
  var mean_ = new Float64Array(D.N); per.forEach(function(p){ for (var i = 0; i < D.N; i++) mean_[i] += p[i]; }); for (var i = 0; i < D.N; i++) mean_[i] /= n;
  return { trees: trees, per: per, pred: mean_, secs: (performance.now() - t0) / 1000 };
}
function forestCurve(fr, step){ var N = D.N, acc = new Float64Array(N), out = []; for (var k = 0; k < fr.per.length; k++) { for (var i = 0; i < N; i++) acc[i] += fr.per[k][i]; if ((k + 1) % step === 0 || k === fr.per.length - 1) { var p = acc.map(function(v){ return v / (k + 1); }); out.push({ k: k + 1, te: r2(D.TEST, p), tr: r2(D.TRAIN, p) }); } } return out; }
function fitBoost(feats, idx, rounds, lr, depth, snapEvery, ds){
  ds = ds || D; var t0 = performance.now(), tr = Int32Array.from(idx), N = ds.N, ym = mean(idx.map(function(i){ return ds.Y[i]; })), F = new Float64Array(N).fill(ym), res = new Float32Array(N), rand = rng(3), snaps = [], curve = [], snapT = 0, gain = {}, trees = [];
  snapEvery = snapEvery || 0;
  for (var m = 0; m < rounds; m++) { for (var i = 0; i < N; i++) res[i] = ds.Y[i] - F[i]; var bt = buildTree(ds.B, feats, res, tr, depth || 3, 10, feats.length, rand); trees.push(bt); addGain(bt, gain); for (var j = 0; j < N; j++) F[j] += lr * predictTree(bt, ds.B, j);
    if (snapEvery && ((m + 1) % snapEvery === 0 || m === rounds - 1)) { var ts = performance.now(); snaps.push({ r: m + 1, p: Float64Array.from(F) }); curve.push({ k: m + 1, te: r2(ds.TEST, F, ds.Y), tr: r2(idx, F, ds.Y) }); snapT += performance.now() - ts; } }
  return { F: F, snaps: snaps, curve: curve, gain: gain, trees: trees, base: ym, lr: lr, secs: (performance.now() - t0 - snapT) / 1000 };
}
function boostPredictVals(model, B, vals){ var p = model.base; for (var k = 0; k < model.trees.length; k++) p += model.lr * predictVals(model.trees[k], B, vals); return p; }

/* ---------- scores per endpoint ---------- */
function record(name, feats, pred, secs){ var m = { feats: feats, secs: secs, r2: r2(D.TEST, pred), sp: spearman(D.TEST, pred), rmse: rmse(D.TEST, pred), tr: r2(D.TRAIN, pred), ep: EP }; D.scores[name] = m; return m; }
function scoreCard(name, m, i){ return '<div class="score" style="animation-delay:' + (i || 0) * 120 + 'ms"><b>' + esc(name) + '</b><div class="v">' + fix(m.r2, 2) + '</div><div class="s">R2 on unseen compounds<br>Spearman ' + fix(m.sp, 2) + ', studied R2 ' + fix(m.tr, 2) + (m.secs !== undefined && m.secs !== null ? '<br>trained in ' + fmtT(m.secs) : '') + '</div></div>'; }
function fmtT(t){ return t === null || t === undefined ? 'n/a' : (t < 10 ? t.toFixed(2) : t.toFixed(1)) + ' s'; }

/* ---------- canvas helpers ---------- */
function cv(el){ el = typeof el === 'string' ? $(el) : el; var dpr = Math.min(window.devicePixelRatio || 1, window.innerWidth < 900 ? 1.5 : 2), w = +el.getAttribute('width'), hh = +el.getAttribute('height'); if (!el._s) { el.width = w * dpr; el.height = hh * dpr; el.style.width = w + 'px'; el.style.height = hh + 'px'; el._s = 1; } var x = el.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, w, hh); x.font = '12px Aptos, "Segoe UI", system-ui, sans-serif'; return { x: x, w: w, h: hh }; }
function axes(g, box, xr, yr, xl, yl, ticks, o){
  o = o || {}; var x = g.x; x.strokeStyle = C.silver; x.fillStyle = C.mid; x.lineWidth = 1; x.textAlign = 'right'; ticks = ticks || 4;
  var yd = Math.abs(yr[1] - yr[0]) > 6 ? 0 : 1;
  for (var k = 0; k <= ticks; k++) { var v = yr[0] + (yr[1] - yr[0]) * k / ticks, y = box.t + (1 - k / ticks) * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText(v.toFixed(o.ydec !== undefined ? o.ydec : yd), box.l - 6, y + 4); }
  x.textAlign = 'center'; var xd = Math.abs(xr[1] - xr[0]) > 6 ? 0 : 1;
  for (var k2 = 0; k2 <= ticks; k2++) { var xv = xr[0] + (xr[1] - xr[0]) * k2 / ticks; x.fillText(xv.toFixed(o.xdec !== undefined ? o.xdec : xd), box.l + (box.r - box.l) * k2 / ticks, box.b + 16); }
  x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText(xl, (box.l + box.r) / 2, box.b + 33); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText(yl, 0, 0); x.restore(); x.font = '12px Aptos, "Segoe UI", sans-serif';
}
function lineChart(el, series, o){
  var g = cv(el), box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, xs = [], ys = []; series.forEach(function(s){ s.pts.forEach(function(p){ xs.push(p[0]); ys.push(p[1]); }); });
  if (!xs.length) return g; var x0 = o.x0 !== undefined ? o.x0 : Math.min.apply(null, xs), x1 = o.x1 !== undefined ? o.x1 : Math.max.apply(null, xs), y0 = o.y0 !== undefined ? o.y0 : Math.min.apply(null, ys), y1 = o.y1 !== undefined ? o.y1 : Math.max.apply(null, ys); if (x1 === x0) x1 = x0 + 1; if (y1 === y0) y1 = y0 + 1;
  axes(g, box, [x0, x1], [y0, y1], o.xl || '', o.yl || '', 4, o); var sx = function(v){ return box.l + (v - x0) / (x1 - x0) * (box.r - box.l); }, sy = function(v){ return box.b - (v - y0) / (y1 - y0) * (box.b - box.t); };
  series.forEach(function(s){ var x = g.x; x.strokeStyle = s.color; x.lineWidth = s.w || 2.6; x.lineJoin = 'round'; x.setLineDash(s.dash || []); x.beginPath(); s.pts.forEach(function(p, i){ i ? x.lineTo(sx(p[0]), sy(p[1])) : x.moveTo(sx(p[0]), sy(p[1])); }); x.stroke(); x.setLineDash([]);
    if (!s.nodot && s.pts.length) { var q = s.pts[s.pts.length - 1]; x.fillStyle = s.color; x.beginPath(); x.arc(sx(q[0]), sy(q[1]), 4.5, 0, 6.3); x.fill(); } if (s.label && s.pts.length) { var e = s.pts[s.pts.length - 1]; x.fillStyle = s.color; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText(s.label, Math.min(sx(e[0]) + 6, box.r - 60), sy(e[1]) - 6); x.font = '12px Aptos, "Segoe UI", sans-serif'; } });
  g.sx = sx; g.sy = sy; g.box = box; return g;
}
function scatterPV(el, pred, idx, label, o){
  o = o || {}; var g = cv(el), box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, lo = o.lo !== undefined ? o.lo : D.ylo, hi = o.hi !== undefined ? o.hi : D.yhi, x = g.x;
  axes(g, box, [lo, hi], [lo, hi], 'measured', 'predicted', 4); var sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - lo) / (hi - lo) * (box.b - box.t); };
  x.strokeStyle = C.char; x.setLineDash([5, 5]); x.beginPath(); x.moveTo(sx(lo), sy(lo)); x.lineTo(sx(hi), sy(hi)); x.stroke(); x.setLineDash([]);
  var step = Math.ceil(idx.length / (o.max || 380)); x.fillStyle = o.fill || 'rgba(225,36,42,.55)'; for (var k = 0; k < idx.length; k += step) { var i = idx[k]; x.beginPath(); x.arc(sx(clamp(D.Y[i], lo, hi)), sy(clamp(pred[i], lo, hi)), 3.4, 0, 6.3); x.fill(); }
  if (label) { x.fillStyle = C.char; x.font = '700 14px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText(label, box.l + 8, box.t + 18); }
  return g;
}
function setRange(d){ var lo = Infinity, hi = -Infinity; for (var i = 0; i < d.N; i++) { if (d.Y[i] < lo) lo = d.Y[i]; if (d.Y[i] > hi) hi = d.Y[i]; } var pad = (hi - lo) * 0.04; d.ylo = Math.floor((lo - pad) * 2) / 2; d.yhi = Math.ceil((hi + pad) * 2) / 2; }

/* ---------- RDKit in the browser (loaded on demand) ---------- */
var RDK = null, rdkP = null, RDK_BASE = 'vendor/rdkit/';
function loadRDKit(){
  if (rdkP) return rdkP;
  rdkP = new Promise(function(res, rej){ var s = document.createElement('script'); s.src = RDK_BASE + 'RDKit_minimal.js'; s.onload = function(){ initRDKitModule({ locateFile: function(f){ return RDK_BASE + f; } }).then(function(m){ RDK = m; res(m); }, rej); }; s.onerror = function(){ rdkP = null; rej(new Error('RDKit.js could not be loaded')); }; document.head.appendChild(s); });
  return rdkP;
}
var RDK_DRAW = { width: 300, height: 220, clearBackground: false, bondLineWidth: 2, padding: 0.07, atomColourPalette: { '6': [0.2, 0.22, 0.25], '7': [0.88, 0.14, 0.16], '8': [0.54, 0.08, 0.08], '16': [0.6, 0.45, 0], '9': [0.17, 0.48, 0.34], '17': [0.17, 0.48, 0.34], '35': [0.17, 0.48, 0.34] } };
function rdkSvg(smi, w, hh){ var m = RDK.get_mol(smi); if (!m || !m.is_valid()) { if (m) m.delete(); return null; } var o = Object.assign({}, RDK_DRAW, { width: w || 300, height: hh || 220 }), svg = m.get_svg_with_highlights(JSON.stringify(o)); m.delete(); return svg.replace(/<\?xml[^>]*\?>/, '').replace(/width='\d+px' height='\d+px'/, "width='100%' height='100%'"); }
function rdkValid(smi){ if (!smi) return false; var m = RDK.get_mol(smi); var ok = !!(m && m.is_valid()); if (m) m.delete(); return ok; }
function rdkDescriptors(smi){ var m = RDK.get_mol(smi); if (!m || !m.is_valid()) { if (m) m.delete(); return null; } var d = JSON.parse(m.get_descriptors()); m.delete(); return d; }

async function loadStructs(){ if (!STRUCTS) STRUCTS = await loadJSON('structs'); return STRUCTS; }
async function loadExtras(){ if (!EXTRAS) EXTRAS = await loadJSON('extras'); return EXTRAS; }

/* =====================================================================
   Lesson framework
   ===================================================================== */
var LESSONS = {}, LORDER = ['start', 'trees', 'nn', 'gpt', 'pre', 'compare'], ACTIVE = 'start', featListeners = [];
function Lesson(id, title){ this.id = id; this.title = title; this.steps = []; this.gloss = null; this.optsHTML = ''; this.onOpts = null; this.onData = null; LESSONS[id] = this; }
Lesson.prototype.add = function(def){ this.steps.push(def); return this; };
Lesson.prototype.glossary = function(items){ this.gloss = items; return this; };
Lesson.prototype.move = function(short, to){ var i = -1; this.steps.forEach(function(s, k){ if (s.short === short) i = k; }); if (i < 0) return; var s = this.steps.splice(i, 1)[0]; this.steps.splice(to, 0, s); };
Lesson.prototype.refresh = function(){ var L = this; L.steps.forEach(function(s){ s.dirty = true; if (s.visible && s.armed && s.handlers && s.handlers.render) { s.dirty = false; s.api.tok++; s.api.clearTexts(); s.safe(function(){ return s.handlers.render(); }); } }); };

function seg(items, cur, attr){ return '<div class="seg" ' + attr + '>' + items.map(function(it){ return '<button type="button" data-v="' + it[0] + '"' + (it[0] === cur ? ' class="on"' : '') + '>' + it[1] + '</button>'; }).join('') + '</div>'; }
function optGroup(label, html, hint){ return '<div class="og"><div class="ol">' + label + '</div>' + html + (hint ? '<div class="hint">' + hint + '</div>' : '') + '</div>'; }
function optRange(id, label, min, max, step, val, hint){ return '<div class="og"><label class="row" for="' + id + '">' + label + '<output id="o-' + id + '">' + val + '</output></label><input type="range" id="' + id + '" min="' + min + '" max="' + max + '" step="' + step + '" value="' + val + '">' + (hint ? '<div class="hint">' + hint + '</div>' : '') + '</div>'; }
function sharedOptions(o){ return optGroup('Endpoint', seg([['HLM', 'HLM clearance'], ['logD', 'logD']], EP, 'data-ep'), 'HLM is Biogen data. logD is AstraZeneca data.') + optGroup('Features', '<div class="chips featchips"></div>', 'Shared by every lesson.'); }
function renderChips(){
  qsa('.featchips').forEach(function(box){ box.innerHTML = D ? D.FEATS.map(function(f){ return '<label' + (f === 'logD_pred' ? ' title="logD predicted by a model trained on AstraZeneca data"' : '') + '><input type="checkbox" data-f="' + f + '"' + (FSEL[f] ? ' checked' : '') + '> ' + f + '</label>'; }).join('') : ''; });
}
document.addEventListener('change', function(e){ var t = e.target; if (t && t.dataset && t.dataset.f && t.closest && t.closest('.featchips')) { FSEL[t.dataset.f] = t.checked; renderChips(); featListeners.forEach(function(fn){ fn(); }); } });
document.addEventListener('input', function(e){ var t = e.target; if (t && t.type === 'range') { var o = $('o-' + t.id); if (o) o.textContent = t.value; } });
document.addEventListener('click', function(e){ var b = e.target.closest ? e.target.closest('.seg[data-ep] button') : null; if (b) setEndpoint(b.dataset.v); });

async function setEndpoint(key){
  if (key === EP && D) return; EP = key; qsa('.seg[data-ep] button').forEach(function(b){ b.classList.toggle('on', b.dataset.v === key); });
  progress(0.2); await loadDataset(key); D = DATA[key]; if (!D.ylo) setRange(D); renderChips(); if (window.syncFilterControls) syncFilterControls(true); progress(1);
  LORDER.forEach(function(id){ var L = LESSONS[id]; if (L && L.built) { if (L.onData) L.onData(); L.refresh(); } });
}

function buildLesson(L){
  var root = $('lesson-' + L.id), n = L.steps.length + (L.gloss ? 1 : 0);
  root.innerHTML = '<aside class="options"><div class="otitle">Options</div><div class="obody"></div><div class="aff"><div class="hint">Change an option to see which steps use it.</div></div></aside><div class="steps"><nav class="toc"></nav></div>';
  qs('.obody', root).innerHTML = typeof L.optsHTML === 'function' ? L.optsHTML() : L.optsHTML; var stepsEl = qs('.steps', root), toc = qs('.toc', root);
  var ig = window.GUIDE && GUIDE.intro[L.id]; if (ig) { var card = h('div', 'intro', '<div class="it"><b>Before you start</b><span>' + esc(ig.prior) + '</span></div>' + (ig.what ? '<p class="iwhat">' + esc(ig.what) + '</p>' : '') + '<div class="ig"><div><div class="ol">In this lesson you will</div><ul>' + ig.goals.map(function(g){ return '<li>' + esc(g) + '</li>'; }).join('') + '</ul></div><div><div class="ol">Words you will meet</div><div class="wd">' + ig.words.map(function(w){ return '<span>' + esc(w) + '</span>'; }).join('') + '</div></div></div>' + (ig.map ? '<div class="cmap"><div class="ol">The whole course, in order</div>' + ig.map.map(function(m){ return '<div><b>' + m[0] + '</b><span><i>' + esc(m[1]) + '</i><br>' + esc(m[3]) + '</span></div>'; }).join('') + '</div>' : '')); stepsEl.insertBefore(card, toc); }
  L.steps.forEach(function(s, i){ s.L = L; s.idx = i; buildStep(L, s, i, n, stepsEl); toc.appendChild(tocChip(L.id + '-s' + (i + 1), (i + 1) + '  ' + s.short)); });
  if (L.gloss) { var g = h('section', 'step gloss'); g.id = L.id + '-gl'; g.innerHTML = '<div class="shead"><i class="tick"></i><div><div class="eyebrow">GLOSSARY</div><h2>Terms in this lesson</h2></div></div><dl class="gl">' + L.gloss.map(function(t){ return '<div><dt>' + t[0] + '</dt><dd>' + t[1] + '</dd></div>'; }).join('') + '</dl>'; stepsEl.appendChild(g); toc.appendChild(tocChip(L.id + '-gl', 'Glossary')); }
  L.built = true; if (L.onOpts) L.onOpts(root); renderChips(); if (window.syncFilterControls) syncFilterControls();
}
function tocChip(target, label){ var b = h('button', 'tc', esc(label)); b.type = 'button'; b.addEventListener('click', function(){ var t = $(target); if (t) window.scrollTo({ top: t.getBoundingClientRect().top + window.scrollY - 70, behavior: ANIM ? 'smooth' : 'auto' }); }); return b; }

function buildStep(L, s, idx, total, parent){
  var sec = h('section', 'step'); sec.id = L.id + '-s' + (idx + 1); s.sec = sec;
  sec.innerHTML = '<div class="shead"><i class="tick"></i><div><div class="eyebrow">' + (s.bench ? 'WORKBENCH' : 'STEP ' + (idx + 1)) + '</div><h2>' + s.title + '</h2></div></div>' +
    '<p class="lead">' + s.watch + '</p><p class="try"><b>Try</b> ' + s.change + '</p>' +
    '<div class="stagewrap"><div class="screen"><div class="inner"></div></div>' + (s.bench ? '' : '<div class="sbar"><button type="button" class="btn sm ghost" data-play>Walk through</button><span class="sx"></span></div>') + '</div>';
  var gd = window.GUIDE && GUIDE.steps[L.id + '/' + s.short]; if (gd) sec.insertAdjacentHTML('beforeend', '<div class="guide"><p class="gn"><b>Notice</b>' + esc(gd.notice) + '</p><details class="gq"><summary>Pop quiz: ' + esc(gd.q) + '</summary><p>' + esc(gd.a) + '</p></details></div>');
  parent.appendChild(sec);
  var screen = qs('.screen', sec), inner = qs('.inner', sec), sbar = qs('.sbar', sec) || h('div', 'sbar'), sx = qs('.sx', sec) || h('span', 'sx');
  var cap = h('div', 'cap'), busy = h('div', 'busy', '<div class="bs"><i></i><span></span></div>'), stale = h('div', 'stalebadge'); inner.appendChild(cap); inner.appendChild(stale); inner.appendChild(busy);
  var api = { sec: sec, inner: inner, sbar: sbar, L: L, tok: 0, stepDef: s };
  api.el = function(cls, style, html){ var e = h('div', cls, html); if (style) e.style.cssText = style; inner.insertBefore(e, cap); return e; };
  api.cap = function(t){ if (t === null || t === undefined) { cap.classList.remove('on'); return; } cap.innerHTML = t; cap.classList.add('on'); };
  api.key = function(t){ api.cap(t || null); };
  api.clearTexts = function(){ cap.classList.remove('on'); };
  api.stat = function(t){ sx.innerHTML = t || ''; };
  api.stale = function(t){ if (t) { stale.textContent = t; stale.classList.add('on'); } else stale.classList.remove('on'); };
  api.busy = function(t){ qs('span', busy).textContent = t || 'Updating'; busy.classList.add('on'); };
  api.gate = function(t){ var sp = qs('span', busy); sp.innerHTML = esc(t) + ' <button type="button" class="btn sm" style="margin-left:10px">Go to workbench</button>'; qs('button', sp).addEventListener('click', function(){ var bi = 0; L.steps.forEach(function(st, k){ if (st.bench) bi = k; }); var w = $(L.id + '-s' + (bi + 1)); if (w) window.scrollTo({ top: w.getBoundingClientRect().top + window.scrollY - 70, behavior: ANIM ? 'smooth' : 'auto' }); }); busy.classList.add('on', 'gate'); };
  api.idle = function(){ busy.classList.remove('on', 'gate'); };
  api.sleep = function(ms){ var tok = api.tok; return new Promise(function(res, rej){ setTimeout(function(){ if (api.tok !== tok) rej(new Cancel()); else res(); }, ANIM ? ms : 0); }); };
  api.anim = function(ms, fn){ var tok = api.tok; return new Promise(function(res, rej){ if (!ANIM) { fn(1); return res(); } var t0 = performance.now(); (function f(now){ if (api.tok !== tok) return rej(new Cancel()); var t = clamp((now - t0) / ms, 0, 1); fn(ease(t)); if (t < 1) requestAnimationFrame(f); else res(); })(t0); }); };
  api.beat = async function(text, ms){ api.cap(text); await api.sleep(ms || 2200); };
  api.addControl = function(html){ var w = h('span', 'sc', html); sbar.insertBefore(w, sx); return w; };
  api.checkTok = function(tok){ if (api.tok !== tok) throw new Cancel(); };
  s.api = api;
  function fit(){ var w = screen.clientWidth; if (w) inner.style.transform = 'scale(' + (w / 960) + ')'; }
  if (window.ResizeObserver) new ResizeObserver(fit).observe(screen); fit();
  s.safe = function(fn){ try { var r = fn(); if (r && r.catch) r.catch(function(e){ if (!(e instanceof Cancel)) console.error(e); }); return r; } catch (e) { if (!(e instanceof Cancel)) console.error(e); } };
  s.handlers = s.build(api) || {};
  var pb = qs('[data-play]', sec); if (pb) { if (!s.handlers.play) pb.style.display = 'none'; else pb.addEventListener('click', function(){ s.play(); }); }
  s.play = function(){ if (!s.handlers.play) return; api.tok++; var tok = api.tok; api.clearTexts(); return s.safe(function(){ return s.handlers.play(tok); }); };
  s.visible = false; s.played = false; s.dirty = true; s.armed = !LAZY || idx === 0;
  function offerLoad(){ if (s.offered) return; s.offered = true; qs('span', busy).innerHTML = 'Tap to load this step <button type="button" class="btn sm" style="margin-left:10px">Load</button>'; busy.classList.add('on', 'gate'); qs('button', busy).addEventListener('click', function(){ s.armed = true; busy.classList.remove('on', 'gate'); if (s.handlers.render) { s.dirty = false; s.safe(function(){ return s.handlers.render(); }); } }); }
  var io = new IntersectionObserver(function(es){ es.forEach(function(en){
    var vis = en.isIntersecting; s.visible = vis; if (!vis || ACTIVE !== L.id) return; if (!s.armed) { offerLoad(); return; }
    if (s.dirty && s.handlers.render) { s.dirty = false; s.safe(function(){ return s.handlers.render(); }); }
  }); }, { threshold: [0, 0.25, 0.5, 0.75] });
  io.observe(sec);
}

/* a copy of a left-panel slider, placed under the stage so the step can be used without scrolling */
function mirrorSlider(api, id, label){
  var src = $(id), w = api.addControl(label + ' <input type="range" style="width:260px"><b class="mv"></b>'), r = qs('input', w), v = qs('.mv', w);
  function pull(){ r.min = src.min; r.max = src.max; r.step = src.step; r.value = src.value; v.textContent = src.value; }
  pull(); r.addEventListener('input', function(){ src.value = r.value; v.textContent = r.value; src.dispatchEvent(new Event('input', { bubbles: true })); }); src.addEventListener('input', pull); return w;
}
/* ---------- workbench: live preview, Train button, slowed replay ---------- */
var SPEED = 'slow';
function speedMul(){ return SPEED === 'slow' ? 1 : (SPEED === 'med' ? 0.3 : 0.04); }
function benchBar(api, o){
  o = o || {}; var bar = api.el('bbar', 'left:28px;top:20px;width:904px;height:48px', (o.leftHTML || '') + '<button type="button" class="btn" data-train>Train</button><div class="seg sp3" data-speed><span class="sl">Replay speed</span>' + [['slow', 'Slow'], ['med', 'Medium'], ['fast', 'Fast']].map(function(it){ return '<button type="button" data-v="' + it[0] + '"' + (it[0] === SPEED ? ' class="on"' : '') + '>' + it[1] + '</button>'; }).join('') + '</div><span class="bst"></span>');
  qsa('[data-speed] button', bar).forEach(function(b){ b.addEventListener('click', function(){ SPEED = b.dataset.v; qsa('[data-speed] button').forEach(function(x){ x.classList.toggle('on', x.dataset.v === SPEED); }); }); });
  return { bar: bar, btn: qs('[data-train]', bar), st: qs('.bst', bar), say: function(t){ qs('.bst', bar).textContent = t || ''; } };
}
function delay(ms){ return new Promise(function(r){ setTimeout(r, ANIM ? ms : 0); }); }


/* ---------- tabs ---------- */
function showLesson(id){
  ACTIVE = id; LORDER.forEach(function(k){ var el = $('lesson-' + k); if (el) el.hidden = k !== id; }); qsa('#tabs button').forEach(function(b){ b.setAttribute('aria-selected', b.dataset.tab === id ? 'true' : 'false'); });
  var L = LESSONS[id]; if (L && !L.built) buildLesson(L); window.scrollTo({ top: 0 });
  if (L && L.onShow) L.onShow();
  if (L) setTimeout(function(){ L.steps.forEach(function(s){ if (s.sec) { var r = s.sec.getBoundingClientRect(); s.visible = r.top < innerHeight && r.bottom > 0; } if (s.visible && s.armed && s.dirty && s.handlers.render) { s.dirty = false; s.safe(function(){ return s.handlers.render(); }); } }); }, 60);
}
function boot(){
  qsa('#tabs button').forEach(function(b){ b.addEventListener('click', function(){ showLesson(b.dataset.tab); }); });
  $('animtoggle').addEventListener('click', function(){ ANIM = !ANIM; document.body.classList.toggle('noanim', !ANIM); this.setAttribute('aria-pressed', ANIM ? 'false' : 'true'); this.textContent = ANIM ? 'Animations on' : 'Animations off'; });
  if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) $('animtoggle').click();
}

/* ---------- small shared helpers ---------- */
function animateFree(ms, fn){ return new Promise(function(res){ if (!ANIM) { fn(1); return res(); } var t0 = performance.now(); (function f(now){ var t = clamp((now - t0) / ms, 0, 1); fn(ease(t)); if (t < 1) requestAnimationFrame(f); else res(); })(t0); }); }
function countUp(el, to, dec, ms){ animateFree(ms || 900, function(t){ el.textContent = (to * t).toFixed(dec); }); }
function pct(n, d){ return d ? Math.round(100 * n / d) : 0; }
function sortedCol(f){ D.cache.sorted = D.cache.sorted || {}; if (!D.cache.sorted[f]) D.cache.sorted[f] = Float32Array.from(D.COL[f]).sort(); return D.cache.sorted[f]; }
function percentile(f, v){ var a = sortedCol(f), lo = 0, hi = a.length; while (lo < hi) { var m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; } return lo / a.length; }
function fineBins(feat, nb){ D.cache.fine = D.cache.fine || {}; var key = feat + nb; if (D.cache.fine[key]) return D.cache.fine[key]; var v = D.TRAIN.map(function(i){ return D.COL[feat][i]; }).sort(function(a, b){ return a - b; }), e = [];
  for (var q = 1; q < nb; q++) { var x = v[Math.floor(q / nb * v.length)]; if (!e.length || x > e[e.length - 1]) e.push(x); } var c = new Uint8Array(D.N); for (var i = 0; i < D.N; i++) { var val = D.COL[feat][i], k = 0; while (k < e.length && val > e[k]) k++; c[i] = k; } var ed = {}, cd = {}; ed[feat] = e; cd[feat] = c; return (D.cache.fine[key] = { edges: ed, codes: cd }); }
function demoTree(){ var f = activeFeats(); if (!f.length) return null; D.cache.demoT = D.cache.demoT || {}; var k = f.join(','); if (!D.cache.demoT[k]) D.cache.demoT[k] = fitTree(f, D.TRAIN, 5, 20, 1); return D.cache.demoT[k]; }
function fmtFeat(f, v){ return (f === 'logP' || f === 'Fsp3') ? v.toFixed(2) : v.toFixed(0); }
function structOf(i){ if (!STRUCTS) return null; var a = STRUCTS[EP] || []; for (var k = 0; k < a.length; k++) if (a[k].i === i) return a[k]; return null; }
async function typeInto(el, text, api, cps){ var tok = api.tok; el.textContent = ''; var step = Math.max(1, Math.round((cps || 60) / 30)); for (var k = 0; k <= text.length; k += step) { api.checkTok(tok); el.textContent = text.slice(0, k); await new Promise(function(r){ setTimeout(r, ANIM ? 33 : 0); }); } el.textContent = text; }
function miniTreeSVG(w, hh){ var pts = [[0.5, 0.1], [0.27, 0.45], [0.73, 0.45], [0.14, 0.85], [0.4, 0.85], [0.6, 0.85], [0.86, 0.85]], ed = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]], s = '';
  ed.forEach(function(e){ s += '<line x1="' + pts[e[0]][0] * w + '" y1="' + pts[e[0]][1] * hh + '" x2="' + pts[e[1]][0] * w + '" y2="' + pts[e[1]][1] * hh + '" stroke="#A1B1C3" stroke-width="2"/>'; });
  pts.forEach(function(p, k){ s += '<circle cx="' + p[0] * w + '" cy="' + p[1] * hh + '" r="' + (k ? 8 : 10) + '" fill="' + (k > 2 ? C.red : C.char) + '"><animate attributeName="r" values="' + (k ? 8 : 10) + ';' + (k ? 11 : 13) + ';' + (k ? 8 : 10) + '" dur="' + (1.6 + k * 0.15) + 's" repeatCount="indefinite"/></circle>'; });
  return '<svg viewBox="0 0 ' + w + ' ' + hh + '" width="' + w + '" height="' + hh + '">' + s + '</svg>'; }

function unitWord(){ return D.key === 'HLM' ? 'HLM clearance (log10)' : 'logD'; }


/* ---------- which steps does an option drive? ---------- */
function setUses(L, arr){ arr.forEach(function(u, i){ if (u && L.steps[i]) L.steps[i].uses = u; }); }
function optLabel(id){
  if (id === 'features') return 'Features'; if (id === 'endpoint') return 'Endpoint'; if (id === 'filters') return 'Filter training data'; if (id === 'nearly') return 'Keep the best epoch';
  var l = document.querySelector('label[for="' + id + '"]'); if (l) return l.childNodes[0].textContent.trim(); return id;
}
function usesHTML(s){ if (!s.uses || !s.uses.length) return ''; return '<div class="uses">Driven by: ' + s.uses.map(function(u){ return '<b>' + esc(optLabel(u)) + '</b>'; }).join(', ') + '</div>'; }
function optKey(t){ if (!t || !t.closest) return null; if (!t.closest('aside.options')) return null; if (t.closest('.featchips')) return 'features'; if (t.closest('.seg[data-ep]')) return 'endpoint'; if (t.closest('.filtgrp')) return 'filters'; return t.id || null; }
var affTimer = null;
function showAffects(key){
  var L = LESSONS[ACTIVE]; if (!L || !L.built || !key) return; var root = $('lesson-' + L.id), box = qs('.aff', root); if (!box) return; var hits = [];
  L.steps.forEach(function(st, i){ if (st.uses && st.uses.indexOf(key) >= 0) hits.push(i); });
  box.innerHTML = '<div class="ol">' + esc(optLabel(key)) + '</div>' + (hits.length ? '<div class="hint" style="margin-bottom:5px">Drives these steps:</div><div class="ah">' + hits.map(function(i){ return '<button type="button" data-g="' + L.id + '-s' + (i + 1) + '">' + (i + 1) + ' ' + esc(L.steps[i].short) + '</button>'; }).join('') + '</div>' : '<div class="hint">No step in this lesson uses this option.</div>');
  qsa('.ah button', box).forEach(function(b){ b.addEventListener('click', function(){ var t = $(b.dataset.g); if (t) window.scrollTo({ top: t.getBoundingClientRect().top + window.scrollY - 70, behavior: ANIM ? 'smooth' : 'auto' }); }); });
  qsa('.tc', root).forEach(function(c, i){ c.classList.toggle('hit', hits.indexOf(i) >= 0); }); clearTimeout(affTimer); affTimer = setTimeout(function(){ qsa('.tc.hit', root).forEach(function(c){ c.classList.remove('hit'); }); }, 4500);
}
['input', 'change'].forEach(function(ev){ document.addEventListener(ev, function(e){ var k = optKey(e.target); if (k) showAffects(k); }, true); });
document.addEventListener('click', function(e){ var b = e.target.closest ? e.target.closest('.seg button') : null; if (b && b.closest('aside.options')) { var k = optKey(b); if (k) showAffects(k); } }, true);

/* ---------- wait for a slider to be released before heavy work ---------- */
var HOLD = false;
document.addEventListener('pointerdown', function(e){ if (e.target && e.target.matches && e.target.matches('aside.options input[type=range]')) HOLD = true; }, true);
['pointerup', 'pointercancel'].forEach(function(ev){ window.addEventListener(ev, function(){ if (HOLD) { HOLD = false; var L = LESSONS[ACTIVE]; if (L && L.built) setTimeout(function(){ if (window.FQ && FQ.pending) flushFilter(); else L.refresh(); }, 60); } }, true); });
function waitRelease(){ return new Promise(function(res){ (function c(){ if (!HOLD) res(); else setTimeout(c, 150); })(); }); }
