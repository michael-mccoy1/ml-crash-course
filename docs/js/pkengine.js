"use strict";
/* PKE: pharmacokinetic simulation helpers for the PK and machine learning lessons.
   Contents: RK4 integration with dosing events, closed-form PK models, in vitro to in vivo extrapolation (IVIVE),
   a whole-body flow-limited PBPK model, non-compartmental analysis (NCA) and a seeded virtual population sampler.
   Public parameters only. Sources are cited next to each default. */
var PKE = (function(){
  var LN2 = Math.LN2;

  /* ---------- random numbers: use the site's rng(seed) when core.js is loaded ---------- */
  function mkRng(seed){
    if (typeof rng === 'function') return rng(seed);
    var s = seed >>> 0; /* mulberry32, the same generator as core.js */
    return function(){ s = (s + 0x6D2B79F5) >>> 0; var t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function mkNormal(seed){ var r = mkRng(seed); return function(){ var u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; }

  /* ---------- RK4 ---------- */
  function rk4Step(f, t, y, h){
    var n = y.length, k1 = f(t, y), y2 = new Array(n), i, k2, k3, k4, out = new Array(n);
    for (i = 0; i < n; i++) y2[i] = y[i] + 0.5 * h * k1[i]; k2 = f(t + 0.5 * h, y2);
    for (i = 0; i < n; i++) y2[i] = y[i] + 0.5 * h * k2[i]; k3 = f(t + 0.5 * h, y2);
    for (i = 0; i < n; i++) y2[i] = y[i] + h * k3[i]; k4 = f(t + h, y2);
    for (i = 0; i < n; i++) out[i] = y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return out;
  }
  /* Fixed-step RK4. f(t, y) returns dy/dt as an array. Returns { t: [...], y: [[...], ...] } on the grid t0, t0+dt, ..., t1. */
  function rk4(f, y0, t0, t1, dt){
    var n = Math.max(1, Math.ceil((t1 - t0) / dt - 1e-9)), h = (t1 - t0) / n, T = [t0], Y = [y0.slice()], y = y0.slice(), i;
    for (i = 0; i < n; i++) { y = rk4Step(f, t0 + i * h, y, h); T.push(t0 + (i + 1) * h); Y.push(y); }
    return { t: T, y: Y };
  }
  /* RK4 with bolus doses. doses: [{ t, index, amount }] adds amount to state[index] at time t.
     Dose times are inserted into the time grid, so a bolus is applied exactly. Output is stored every `every` steps and at dose times. */
  function rk4Events(f, y0, t0, t1, dt, doses, every){
    var grid = [], n = Math.max(1, Math.ceil((t1 - t0) / dt - 1e-9)), h = (t1 - t0) / n, i, ev = (doses || []).slice().sort(function(a, b){ return a.t - b.t; }), k;
    for (i = 0; i <= n; i++) grid.push(t0 + i * h);
    ev.forEach(function(d){ if (d.t > t0 && d.t < t1) grid.push(d.t); }); grid.sort(function(a, b){ return a - b; });
    var T = [], Y = [], y = y0.slice(), t = t0, ei = 0, stepNo = 0; every = every || 1;
    while (ei < ev.length && ev[ei].t <= t0 + 1e-12) { y[ev[ei].index] += ev[ei].amount; ei++; }
    T.push(t); Y.push(y.slice());
    for (k = 1; k < grid.length; k++) {
      if (grid[k] - t < 1e-14) continue;
      y = rk4Step(f, t, y, grid[k] - t); t = grid[k]; stepNo++;
      var dosed = false; while (ei < ev.length && ev[ei].t <= t + 1e-12) { y[ev[ei].index] += ev[ei].amount; ei++; dosed = true; }
      if (dosed || stepNo % every === 0 || k === grid.length - 1) { T.push(t); Y.push(y.slice()); }
    }
    return { t: T, y: Y };
  }
  /* Zero-order infusion schedule: [{ t0, t1, rate }] -> function rate(t) */
  function infusionRate(schedule){ return function(t){ var r = 0; for (var i = 0; i < schedule.length; i++) if (t >= schedule[i].t0 && t < schedule[i].t1) r += schedule[i].rate; return r; }; }

  /* ---------- closed-form models (ground truth for the numerical solvers) ---------- */
  /* One compartment. opts: { dose (mg), V (L), CL (L/h), ka (1/h), F, route: 'iv' | 'oral' }.
     Returns { conc(t), summary: { ke, thalf, auc, cmax, tmax } }. */
  function oneComp(o){
    var V = o.V, CL = o.CL, ke = CL / V, dose = o.dose, oral = o.route === 'oral', F = oral ? (o.F === undefined ? 1 : o.F) : 1, ka = o.ka, conc, cmax, tmax;
    if (!oral) { conc = function(t){ return t < 0 ? 0 : dose / V * Math.exp(-ke * t); }; cmax = dose / V; tmax = 0; }
    else if (Math.abs(ka - ke) < 1e-9 * Math.max(ka, ke)) { conc = function(t){ return t < 0 ? 0 : F * dose * ka * t * Math.exp(-ke * t) / V; }; tmax = 1 / ke; cmax = conc(tmax); }
    else { conc = function(t){ return t < 0 ? 0 : F * dose * ka / (V * (ka - ke)) * (Math.exp(-ke * t) - Math.exp(-ka * t)); }; tmax = Math.log(ka / ke) / (ka - ke); cmax = conc(tmax); }
    return { conc: conc, summary: { ke: ke, thalf: LN2 / ke, auc: F * dose / CL, cmax: cmax, tmax: tmax } };
  }
  /* Two compartments, IV bolus, macro-constants. opts: { dose, V1, CL, Q, V2 }. */
  function twoComp(o){
    var k10 = o.CL / o.V1, k12 = o.Q / o.V1, k21 = o.Q / o.V2, s = k10 + k12 + k21, d = Math.sqrt(s * s - 4 * k10 * k21), alpha = (s + d) / 2, beta = (s - d) / 2,
        A = (alpha - k21) / (o.V1 * (alpha - beta)), B = (k21 - beta) / (o.V1 * (alpha - beta));
    return { conc: function(t){ return t < 0 ? 0 : o.dose * (A * Math.exp(-alpha * t) + B * Math.exp(-beta * t)); },
      summary: { alpha: alpha, beta: beta, A: o.dose * A, B: o.dose * B, thalfAlpha: LN2 / alpha, thalfBeta: LN2 / beta, auc: o.dose / o.CL, cmax: o.dose / o.V1, tmax: 0, vss: o.V1 * (1 + k12 / k21) } };
  }
  /* Repeated doses by superposition of a single-dose profile: conc(t) = sum of single(t - i * tau) for i < n. */
  function multipleDose(single, tau, n){ return function(t){ var c = 0; for (var i = 0; i < n; i++) if (t >= i * tau) c += single(t - i * tau); return c; }; }
  /* Accumulation ratio at steady state for first-order elimination with interval tau. */
  function accumulation(ke, tau){ return 1 / (1 - Math.exp(-ke * tau)); }

  /* ---------- IVIVE ---------- */
  /* Well-stirred liver model, blood clearance. clint: unbound intrinsic clearance (same flow units as qh), fu: fraction unbound in plasma,
     bp: blood to plasma ratio, fub: fraction unbound in blood (default fu / bp).
     CLh = qh * fub * clint / (qh + fub * clint). Returns { clh (blood), clhPlasma, extraction, fh }. */
  function wellStirred(o){
    var bp = o.bp === undefined ? 1 : o.bp, fub = o.fub !== undefined ? o.fub : (o.fu === undefined ? 1 : o.fu) / bp, x = fub * o.clint, clh = o.qh * x / (o.qh + x), e = clh / o.qh;
    return { clh: clh, clhPlasma: clh * bp, extraction: e, fh: 1 - e };
  }
  /* Parallel-tube model: CLh = qh * (1 - exp(-fub * clint / qh)). */
  function parallelTube(o){
    var bp = o.bp === undefined ? 1 : o.bp, fub = o.fub !== undefined ? o.fub : (o.fu === undefined ? 1 : o.fu) / bp, clh = o.qh * (1 - Math.exp(-fub * o.clint / o.qh)), e = clh / o.qh;
    return { clh: clh, clhPlasma: clh * bp, extraction: e, fh: 1 - e };
  }
  /* The Biogen HLM value is log10 of CLint in mL/min/kg (the data column is 'LOG HLM_CLint (mL/min/kg)', see docs/data/hlm.json unit).
     The public sources do not state how the value was scaled or whether it was corrected for binding in the incubation.
     This function treats it as an intrinsic clearance already expressed per kg of body weight and applies only the corrections in opts.
     Defaults: body weight 70 kg; human hepatic blood flow 20.7 mL/min/kg (Davies and Morris 1993, Pharm Res 10:1093-1095, Table III,
     as quoted by open-access papers such as Front Pharmacol 2025, PMC12678371); fup 1, blood to plasma ratio 1 and fumic 1 (no binding),
     which are placeholders to be replaced by measured values. opts: { bw, qhPerKg, fup, bp, fumic, model: 'wellStirred' | 'parallelTube' }.
     Returns hepatic clearance in mL/min/kg and L/h. */
  function fromHLM(log10Clint, opts){
    var o = opts || {}, bw = o.bw || 70, qhk = o.qhPerKg || 20.7, fup = o.fup === undefined ? 1 : o.fup, bp = o.bp === undefined ? 1 : o.bp, fumic = o.fumic === undefined ? 1 : o.fumic,
        clintKg = Math.pow(10, log10Clint), clintU = clintKg / fumic, m = (o.model === 'parallelTube' ? parallelTube : wellStirred)({ clint: clintU, qh: qhk, fu: fup, bp: bp }), toLh = function(v){ return v * bw * 60 / 1000; };
    return { clintMLminKg: clintKg, clintUnboundMLminKg: clintU, qhMLminKg: qhk, clhBloodMLminKg: m.clh, clhPlasmaMLminKg: m.clhPlasma, clhBloodLh: toLh(m.clh), clhPlasmaLh: toLh(m.clhPlasma), extraction: m.extraction, fh: m.fh, bw: bw,
      assumptions: ['Value treated as intrinsic clearance per kg body weight.', 'No correction for binding in the incubation unless fumic is set.', 'Hepatic blood flow ' + qhk + ' mL/min/kg.'] };
  }

  /* ---------- NCA ---------- */
  /* AUC by the linear-up, log-down trapezoidal rule from t[0] to the last time point. */
  function auc(t, c){
    var s = 0;
    for (var i = 1; i < t.length; i++) { var dt = t[i] - t[i - 1], c0 = c[i - 1], c1 = c[i]; s += (c1 < c0 && c1 > 0 && c0 > 0) ? (c0 - c1) * dt / Math.log(c0 / c1) : (c0 + c1) * dt / 2; }
    return s;
  }
  function cmaxTmax(t, c){ var k = 0; for (var i = 1; i < c.length; i++) if (c[i] > c[k]) k = i; return { cmax: c[k], tmax: t[k], index: k }; }
  /* Terminal slope from the last nPoints positive concentrations after tmax (log-linear least squares). */
  function terminalHalfLife(t, c, nPoints){
    var n = nPoints || 3, ix = [], i, mt = cmaxTmax(t, c).index;
    for (i = c.length - 1; i > mt && ix.length < n; i--) if (c[i] > 0) ix.unshift(i);
    if (ix.length < 2) return { lambdaZ: NaN, thalf: NaN, r2: NaN, n: ix.length };
    var xs = ix.map(function(k){ return t[k]; }), ys = ix.map(function(k){ return Math.log(c[k]); }), mx = xs.reduce(function(a, b){ return a + b; }, 0) / xs.length, my = ys.reduce(function(a, b){ return a + b; }, 0) / ys.length, sxx = 0, sxy = 0, syy = 0;
    for (i = 0; i < xs.length; i++) { sxx += (xs[i] - mx) * (xs[i] - mx); sxy += (xs[i] - mx) * (ys[i] - my); syy += (ys[i] - my) * (ys[i] - my); }
    var slope = sxy / sxx, lz = -slope;
    return { lambdaZ: lz, thalf: LN2 / lz, r2: syy > 0 ? sxy * sxy / (sxx * syy) : 1, n: xs.length };
  }
  /* Non-compartmental summary. dose in mg. Returns cmax, tmax, aucLast, aucInf, lambdaZ, thalf, clF (L/h), vzF (L), r2. */
  function nca(t, c, dose, opts){
    var ct = cmaxTmax(t, c), al = auc(t, c), tz = terminalHalfLife(t, c, (opts && opts.nTerminal) || 3), clast = c[c.length - 1], extra = isFinite(tz.lambdaZ) && tz.lambdaZ > 0 ? clast / tz.lambdaZ : NaN, ai = al + extra, clf = dose / ai;
    return { cmax: ct.cmax, tmax: ct.tmax, aucLast: al, aucInf: ai, lambdaZ: tz.lambdaZ, thalf: tz.thalf, clF: clf, vzF: clf / tz.lambdaZ, r2: tz.r2 };
  }

  /* ---------- virtual population ---------- */
  /* n subjects with lognormal between-subject variability on each named parameter (diagonal covariance).
     spec: { name: { typical, cv } } or { name: { typical, omega } }, omega being the standard deviation of the log value.
     Returns an array of { name: value } objects. */
  function population(n, seed, spec){
    var nz = mkNormal(seed), names = Object.keys(spec), out = [];
    for (var i = 0; i < n; i++) { var p = {}; names.forEach(function(k){ var s = spec[k], om = s.omega !== undefined ? s.omega : Math.sqrt(Math.log(1 + (s.cv || 0) * (s.cv || 0))); p[k] = s.typical * Math.exp(om * nz()); }); out.push(p); }
    return out;
  }

  /* ---------- whole-body flow-limited PBPK ---------- */
  var pbpk = {};
  /* Reference physiology: adult ICRP European Caucasian male, 73 kg.
     Source: Table 1 of "Modeling principles for a physiology-based whole-body model of human metabolism", arXiv:2404.14224 (2024),
     which takes organ volumes and blood flows from ICRP Publication 89 (Valentin 2002, Ann ICRP 32) and PK-Sim. Retrieved 2026-10-28.
     Volumes in L (total organ volume). Blood flows in L/min in the source, stored here in L/h.
     Gut = small intestine + large intestine + pancreas + spleen + stomach (volumes 0.725 + 0.413 + 0.190 + 0.207 + 0.169 L;
     flows 0.650 + 0.260 + 0.065 + 0.166 + 0.065 L/min). The gut drains to the portal vein, which feeds the liver.
     Liver flow in the table is the hepatic artery (0.426 L/min). Rest of body is the gonads entry of the table (0.040 L, 0.003 L/min).
     Cardiac output is the lung flow of the table (6.088 L/min). */
  var REF = { bw: 73, volumes: { lung: 1.215, arterial: 0.419, venous: 0.964, portal: 1.037, gut: 1.704, liver: 2.377, kidney: 0.438, muscle: 32.65, adipose: 14.65, brain: 1.509, heart: 0.417, skin: 3.763, bone: 11.82, rest: 0.040 },
    flowsLmin: { gut: 1.206, liverArtery: 0.426, kidney: 1.327, muscle: 1.116, adipose: 0.320, brain: 0.780, heart: 0.260, skin: 0.325, bone: 0.325, rest: 0.003 },
    source: 'ICRP Publication 89 and PK-Sim values as tabulated in arXiv:2404.14224, Table 1 (adult male, 73 kg); retrieved 2026-10-28' };
  var TISSUES = ['lung', 'gut', 'liver', 'kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'];
  /* Reference physiology in L and L/h, optionally scaled to another body weight bw (volumes in proportion to weight, flows to weight^0.75).
     The scaling is a simple allometric assumption, not a fitted relation. */
  function reference(bw){
    var w = bw || REF.bw, sv = w / REF.bw, sf = Math.pow(w / REF.bw, 0.75), vol = {}, fl = {}, k;
    for (k in REF.volumes) vol[k] = REF.volumes[k] * sv;
    for (k in REF.flowsLmin) fl[k] = REF.flowsLmin[k] * 60 * sf;
    fl.liverTotal = fl.liverArtery + fl.gut;
    fl.cardiacOutput = fl.gut + fl.liverArtery + fl.kidney + fl.muscle + fl.adipose + fl.brain + fl.heart + fl.skin + fl.bone + fl.rest;
    return { bw: w, volumes: vol, flows: fl, source: REF.source };
  }
  /* Tissue composition as volume fractions (water, neutral lipids, phospholipids), human.
     Source: Table 1, "Unified tissue composition", Utsey et al., ACoP 2019 poster (Metrum Research Group), values from Ruark et al. 2014,
     J Pharm Sci 103:2189 and Poulin and Theil 2002, J Pharm Sci 91:129. Retrieved 2026-10-28.
     Rest of body uses the muscle entry (an assumption of this engine). */
  var COMP = { lung: { w: 0.782, nl: 0.003, pl: 0.009 }, gut: { w: 0.757, nl: 0.0487, pl: 0.0163 }, liver: { w: 0.734, nl: 0.037, pl: 0.0252 }, kidney: { w: 0.756, nl: 0.036, pl: 0.0162 },
    muscle: { w: 0.748, nl: 0.013, pl: 0.0072 }, adipose: { w: 0.152, nl: 0.798, pl: 0.002 }, brain: { w: 0.782, nl: 0.045, pl: 0.0565 }, heart: { w: 0.776, nl: 0.089, pl: 0.0166 },
    skin: { w: 0.673, nl: 0.036, pl: 0.0111 }, bone: { w: 0.446, nl: 0.074, pl: 0.0011 }, rest: { w: 0.748, nl: 0.013, pl: 0.0072 } };
  var PLASMA = { w: 0.928, nl: 0.003, pl: 0.00225 };
  /* Tissue to plasma partition coefficients by the Poulin and Theil mixture model, for neutral compounds.
     Tissue is treated as water, neutral lipid (octanol-like, partition P = 10^logP) and phospholipid (30% octanol, 70% water, the Poulin and Krishnan assumption):
       Kp = (P (fNL + 0.3 fPL) + (fW + 0.7 fPL)) / (P (fNLp + 0.3 fPLp) + (fWp + 0.7 fPLp)) times fup / fut.
     The unbound fraction in tissue defaults to fut = 1 / (1 + ((1 - fup) / fup) 0.5), the form used by Poulin and Theil (2002) as described in secondary sources;
     the primary paper was not accessible, so check this default against it before quantitative use. Pass opts.fut to override.
     Limits: neutral compounds only, no ionisation, no acidic phospholipid binding, adipose uses the octanol partition, errors of several fold are common. */
  function kpPT(logP, fup, opts){
    var P = Math.pow(10, logP), fut = opts && opts.fut !== undefined ? opts.fut : 1 / (1 + ((1 - fup) / fup) * 0.5), den = P * (PLASMA.nl + 0.3 * PLASMA.pl) + (PLASMA.w + 0.7 * PLASMA.pl), kp = {};
    TISSUES.forEach(function(t){ var c = COMP[t]; kp[t] = (P * (c.nl + 0.3 * c.pl) + (c.w + 0.7 * c.pl)) / den * fup / fut; });
    return kp;
  }
  /* Model object. opts: { logP, fup, bp (blood to plasma, default 1), kp (override object), clintU (unbound intrinsic hepatic clearance, L/h),
     clr (renal plasma clearance, L/h), ka (1/h), F (fraction of the oral dose reaching the gut tissue), bw (kg) } */
  function model(o){
    var phys = reference(o.bw), fup = o.fup === undefined ? 1 : o.fup, kp = o.kp || kpPT(o.logP === undefined ? 1 : o.logP, fup);
    return { phys: phys, kp: kp, fup: fup, bp: o.bp === undefined ? 1 : o.bp, clintU: o.clintU || 0, clr: o.clr || 0, ka: o.ka === undefined ? 1 : o.ka, F: o.F === undefined ? 1 : o.F };
  }
  /* Hepatic clearance (blood) that the model implements at steady state, for checks: the well-stirred expression with the model liver flow. */
  function modelHepatic(m){ return wellStirred({ clint: m.clintU, qh: m.phys.flows.liverTotal, fu: m.fup, bp: m.bp }); }
  /* State vector: lung, arterial, venous, portal, gut tissue, liver, kidney, muscle, adipose, brain, heart, skin, bone, rest (amounts, mg),
     gut lumen, metabolised, renal, unabsorbed, infused, and the running integral of arterial plasma concentration (AUC, mg h/L). */
  var IDX = { lung: 0, arterial: 1, venous: 2, portal: 3, gut: 4, liver: 5, kidney: 6, muscle: 7, adipose: 8, brain: 9, heart: 10, skin: 11, bone: 12, rest: 13, lumen: 14, met: 15, ren: 16, unabs: 17, infused: 18, auc: 19 };
  var SYS = ['kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'];
  function rhs(m, rateFn){
    var V = m.phys.volumes, Q = m.phys.flows, kp = m.kp, B = m.bp, co = Q.cardiacOutput, qL = Q.liverTotal;
    var qOf = { kidney: Q.kidney, muscle: Q.muscle, adipose: Q.adipose, brain: Q.brain, heart: Q.heart, skin: Q.skin, bone: Q.bone, rest: Q.rest, gut: Q.gut };
    return function(t, y){
      var d = new Array(20), Ca = y[1] / V.arterial, Cv = y[2] / V.venous, Cp = y[3] / V.portal, cb = {}, k, sumSys = 0, venIn = 0, art = 0, rate = rateFn ? rateFn(t) : 0;
      ['lung', 'gut', 'liver', 'kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'].forEach(function(n){ cb[n] = B * y[IDX[n]] / V[n] / kp[n]; });
      for (k = 0; k < SYS.length; k++) { var n = SYS[k]; d[IDX[n]] = qOf[n] * (Ca - cb[n]); venIn += qOf[n] * cb[n]; art += qOf[n] * Ca; }
      var absorb = m.ka * y[14];
      d[IDX.gut] = qOf.gut * (Ca - cb.gut) + m.F * absorb; art += qOf.gut * Ca;
      d[IDX.portal] = qOf.gut * (cb.gut - Cp);
      var elimMet = m.clintU * m.fup * y[IDX.liver] / V.liver / kp.liver;
      d[IDX.liver] = Q.liverArtery * Ca + qOf.gut * Cp - qL * cb.liver - elimMet; art += Q.liverArtery * Ca; venIn += qL * cb.liver;
      d[IDX.venous] = venIn - co * Cv + rate;
      d[IDX.lung] = co * (Cv - cb.lung);
      var elimRen = m.clr * Ca / B;
      d[IDX.arterial] = co * cb.lung - art - elimRen;
      d[14] = -absorb; d[15] = elimMet; d[16] = elimRen; d[17] = (1 - m.F) * absorb; d[18] = rate; d[19] = Ca / B;
      return d;
    };
  }
  /* Time step below which RK4 stays stable for the model: half of the shortest compartment time constant. */
  function stableDt(m){
    var V = m.phys.volumes, Q = m.phys.flows, B = m.bp, kp = m.kp, tau = [V.arterial / Q.cardiacOutput, V.venous / Q.cardiacOutput, V.portal / Q.gut], q = { lung: Q.cardiacOutput, gut: Q.gut, liver: Q.liverTotal, kidney: Q.kidney, muscle: Q.muscle, adipose: Q.adipose, brain: Q.brain, heart: Q.heart, skin: Q.skin, bone: Q.bone, rest: Q.rest };
    TISSUES.forEach(function(n){ tau.push(V[n] * kp[n] / (q[n] * B)); });
    if (m.clintU > 0) tau.push(V.liver * kp.liver / (m.clintU * m.fup + q.liver * B));
    return 0.5 * Math.min.apply(null, tau);
  }
  /* Simulate. dosing: { route: 'iv' | 'oral' | 'infusion', dose (mg), duration (h, infusion), times: [h] (default [0]) }.
     tEnd and dt in hours; dt defaults to stableDt(model); outEvery thins the stored output.
     Returns { t, plasma (arterial plasma conc, mg/L), aucPlasma (running AUC of the arterial plasma curve, mg h/L, integrated with the model and exact for the sampled times), venousPlasma, organs: { name: tissue conc, mg/L }, amounts: { name: mg }, eliminated: { met, ren, unabs }, doseGiven, massError, dt }.
     massError is the largest |amounts + eliminated + unabsorbed - dose given| divided by the total dose.
     After an IV bolus the arterial curve has a sharp first-pass peak lasting seconds, so NCA on a thinned output underestimates AUC: use aucPlasma. */
  function simulate(m, dosing, tEnd, dt, outEvery){
    var h = dt || stableDt(m), times = dosing.times || [0], rateFn = null, ev = [], y0 = new Array(20).fill(0), total = dosing.dose * times.length, i;
    if (dosing.route === 'infusion') { var rate = dosing.dose / dosing.duration; rateFn = infusionRate(times.map(function(t0){ return { t0: t0, t1: t0 + dosing.duration, rate: rate }; })); }
    else times.forEach(function(t0){ ev.push({ t: t0, index: dosing.route === 'oral' ? IDX.lumen : IDX.venous, amount: dosing.dose }); });
    var f = rhs(m, rateFn), res = rk4Events(f, y0, 0, tEnd, h, ev, outEvery || 1), V = m.phys.volumes, out = { t: res.t, plasma: [], venousPlasma: [], aucPlasma: [], organs: {}, amounts: {}, eliminated: { met: [], ren: [], unabs: [] }, doseGiven: [], massError: 0, dt: h, nsteps: res.t.length };
    Object.keys(IDX).forEach(function(n){ out.amounts[n] = []; });
    ['lung', 'portal', 'gut', 'liver', 'kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'].forEach(function(n){ out.organs[n] = []; });
    for (i = 0; i < res.t.length; i++) {
      var y = res.y[i], given = 0; times.forEach(function(t0){ if (dosing.route === 'infusion') given += dosing.dose * Math.min(1, Math.max(0, (res.t[i] - t0) / dosing.duration)); else if (res.t[i] >= t0 - 1e-12) given += dosing.dose; });
      out.plasma.push(y[1] / V.arterial / m.bp); out.aucPlasma.push(y[19]); out.venousPlasma.push(y[2] / V.venous / m.bp);
      Object.keys(IDX).forEach(function(n){ out.amounts[n].push(y[IDX[n]]); });
      Object.keys(out.organs).forEach(function(n){ out.organs[n].push(y[IDX[n]] / V[n]); });
      out.eliminated.met.push(y[15]); out.eliminated.ren.push(y[16]); out.eliminated.unabs.push(y[17]); out.doseGiven.push(given);
      var sum = 0, k; for (k = 0; k < 15; k++) sum += y[k]; sum += y[15] + y[16] + y[17]; var err = Math.abs(sum - (dosing.route === 'infusion' ? y[18] : given)) / total; if (err > out.massError) out.massError = err;
    }
    return out;
  }
  /* Oral bioavailability implied by the model: F (fraction reaching the gut tissue) times the fraction escaping hepatic first pass, 1 - E, with E from the well-stirred model at the model liver flow.
     Total plasma clearance of the model is hepatic plasma clearance plus renal clearance (modelHepatic(m).clhPlasma + m.clr); oral AUC = bioavailability * dose / that clearance. */
  function bioavailability(m){ return m.F * modelHepatic(m).fh; }
  /* Convert the unbound intrinsic clearance of the HLM value to a whole-liver unbound intrinsic clearance in L/h (see fromHLM for the assumptions). */
  function clintUFromHLM(log10Clint, bw, fumic){ return Math.pow(10, log10Clint) / (fumic === undefined ? 1 : fumic) * (bw || 70) * 60 / 1000; }

  pbpk.reference = reference(); pbpk.referenceScaled = reference; pbpk.composition = COMP; pbpk.plasmaComposition = PLASMA; pbpk.tissues = TISSUES; pbpk.index = IDX;
  pbpk.kpPT = kpPT; pbpk.bioavailability = bioavailability; pbpk.model = model; pbpk.simulate = simulate; pbpk.stableDt = stableDt; pbpk.modelHepatic = modelHepatic; pbpk.clintUFromHLM = clintUFromHLM;

  var api = { rk4: rk4, rk4Events: rk4Events, infusionRate: infusionRate, oneComp: oneComp, twoComp: twoComp, multipleDose: multipleDose, accumulation: accumulation,
    wellStirred: wellStirred, parallelTube: parallelTube, fromHLM: fromHLM, auc: auc, cmaxTmax: cmaxTmax, terminalHalfLife: terminalHalfLife, nca: nca, population: population, pbpk: pbpk };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  return api;
})();
