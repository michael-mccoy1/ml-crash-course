"""Build every data file the website loads (docs/data/*.json).

All chemistry is computed with RDKit: descriptors, scaffolds, fingerprints, 2D structure drawings,
atom-contribution maps and similarity statistics. Run notebooks 05 and 06 first if you want their scores in extras.json.
"""
import json, os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors as R
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import TruncatedSVD

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c

OUT = os.path.join(c.ROOT, "docs", "data")
os.makedirs(OUT, exist_ok=True)

FNS = {"logP": Crippen.MolLogP, "TPSA": R.CalcTPSA, "MW": Descriptors.MolWt, "HBD": R.CalcNumHBD, "HBA": R.CalcNumHBA,
       "RotB": R.CalcNumRotatableBonds, "Fsp3": R.CalcFractionCSP3, "AromRings": R.CalcNumAromaticRings}
PALETTE = {6: (0.20, 0.22, 0.25), 7: (0.88, 0.14, 0.16), 8: (0.54, 0.08, 0.08), 16: (0.60, 0.45, 0.0),
           9: (0.17, 0.48, 0.34), 17: (0.17, 0.48, 0.34), 35: (0.17, 0.48, 0.34), 15: (0.69, 0.35, 0.0)}


import re


def _compact(svg):
    """Shrink RDKit SVG: drop classes and comments, round numbers, turn repeated style strings into attributes."""
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    svg = re.sub(r" class='[^']*'", "", svg)
    svg = re.sub(r"<svg[^>]*?(viewBox='[^']*')[^>]*>", r"<svg xmlns='http://www.w3.org/2000/svg' width='100%' height='100%' \1>", svg, count=1, flags=re.S)
    svg = re.sub(r"<rect[^>]*style='opacity:1\.0;fill:#FFFFFF[^>]*/>", "", svg)

    def conv(m):
        kv = dict(x.split(":", 1) for x in m.group(1).split(";") if ":" in x)
        out = []
        fill = kv.get("fill", "none")
        for k in ("fill", "stroke", "stroke-width", "fill-opacity", "stroke-opacity", "fill-rule"):
            if k not in kv:
                continue
            v = kv[k].replace("px", "")
            if k in ("fill-opacity", "stroke-opacity") and v == "1":
                continue
            if k == "fill-rule" and fill == "none":
                continue
            if k == "stroke-width":
                v = "%g" % round(float(v), 1)
                if v == "0":
                    out.append("stroke='none'")
                    continue
            out.append("%s='%s'" % (k, v))
        return " ".join(out)
    svg = re.sub(r"style='([^']*)'", conv, svg)
    svg = re.sub(r"(\d+\.\d)\d+", r"\1", svg)
    return svg.replace("\n", "").replace("  ", " ")


def svg_of(mol, w=300, h=220, highlight=None):
    """RDKit 2D drawing as a compact SVG. highlight = {atom_index: (r, g, b)}."""
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 2
    o.padding = 0.07
    o.updateAtomPalette(PALETTE)
    if highlight:
        o.fillHighlights = True
        o.highlightRadius = 0.42
        d.DrawMolecule(mol, highlightAtoms=list(highlight), highlightAtomColors=highlight, highlightBonds=[])
    else:
        d.DrawMolecule(mol)
    d.FinishDrawing()
    s = d.GetDrawingText()
    return _compact(d.GetDrawingText())


def crippen_map(mol):
    contribs = [x[0] for x in rdMolDescriptors_crippen(mol)]
    m = max(1e-6, max(abs(v) for v in contribs))
    out = {}
    for i, v in enumerate(contribs):
        t = min(1.0, abs(v) / m)
        if abs(v) < 0.05 * m:
            continue
        out[i] = (0.88, 0.14 + 0.7 * (1 - t), 0.16 + 0.7 * (1 - t)) if v > 0 else (0.45 + 0.4 * (1 - t), 0.55 + 0.35 * (1 - t), 0.70 + 0.25 * (1 - t))
    return out


def rdMolDescriptors_crippen(mol):
    return R._CalcCrippenContribs(mol)


def tpsa_map(mol):
    contribs = R._CalcTPSAContribs(mol)
    out = {}
    m = max(1e-6, max(contribs))
    for i, v in enumerate(contribs):
        if v > 0.5:
            t = min(1.0, v / m)
            out[i] = (0.88, 0.14 + 0.7 * (1 - t), 0.16 + 0.7 * (1 - t))
    return out


def fp_bits(mols):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(m) for m in mols]


def similarity_hist(fps, tr, te, bins=20):
    trf = [fps[i] for i in tr]
    ms = []
    for i in te:
        ms.append(max(DataStructs.BulkTanimotoSimilarity(fps[i], trf)))
    ms = np.array(ms)
    h, _ = np.histogram(ms, bins=bins, range=(0, 1))
    return {"hist": h.tolist(), "mean": round(float(ms.mean()), 3)}


def scaffold_overlap(df, tr, te):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    sc = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in df.mol]
    trs = {sc[i] for i in tr}
    return round(float(np.mean([sc[i] in trs for i in te])), 3)


def build_dataset(key, df, extra_cols=None):
    tr, te = c.split(df, "scaffold")
    rtr, rte = c.split(df, "random")
    X = np.array([[f(m) for f in FNS.values()] for m in df.mol])
    rows = [[round(float(v), 2) for v in x] + [round(float(y), 3)] for x, y in zip(X, df.y)]
    test = np.zeros(len(df), int); test[te] = 1
    rtest = np.zeros(len(df), int); rtest[rte] = 1
    fps = fp_bits(df.mol)
    arr = np.zeros((len(df), 2048), dtype=np.float32)
    for i, f in enumerate(fps):
        DataStructs.ConvertToNumpyArray(f, arr[i])
    xy = TruncatedSVD(2, random_state=0).fit_transform(arr)
    xy = (xy - xy.mean(0)) / (np.abs(xy - xy.mean(0)).max(0) + 1e-9)
    d = {"key": key, "cols": list(FNS) + ["y"], "rows": rows, "test": test.tolist(), "rtest": rtest.tolist(),
         "mx": [round(float(v), 3) for v in xy[:, 0]], "my": [round(float(v), 3) for v in xy[:, 1]],
         "sim": {"scaffold": similarity_hist(fps, tr, te), "random": similarity_hist(fps, rtr, rte)},
         "seen": {"scaffold": scaffold_overlap(df, tr, te), "random": scaffold_overlap(df, rtr, rte)}}
    if extra_cols:
        d.update(extra_cols)
    return d, te


def structs_for(df, te, n, seed, with_maps=0):
    rs = np.random.RandomState(seed)
    out = []
    for i in rs.permutation(te):
        m = df.mol[i]
        if m.GetNumHeavyAtoms() > 34 or len(out) >= n:
            continue
        e = {"i": int(i), "smi": df.smiles[i], "svg": svg_of(m)}
        if len(out) < with_maps:
            e["logp"] = svg_of(m, highlight=crippen_map(m))
            e["tpsa"] = svg_of(m, highlight=tpsa_map(m))
        out.append(e)
    return out


def graph_of(m):
    mm = Chem.Mol(m)
    try:
        from rdkit.Chem import rdCoordGen
        rdCoordGen.AddCoords(mm)
    except Exception:
        AllChem.Compute2DCoords(mm)
    pos = mm.GetConformer().GetPositions()[:, :2]
    bl = np.mean([np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]) for b in mm.GetBonds()]) or 1.0
    pos = pos / bl
    return {"atoms": [[a.GetSymbol(), round(float(pos[a.GetIdx()][0]), 2), round(float(-pos[a.GetIdx()][1]), 2)] for a in mm.GetAtoms()],
            "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()) if b.GetBondTypeAsDouble() in (1.0, 2.0, 3.0) else 1] for b in mm.GetBonds()]}


def ref_scores(az):
    """Reference scores from LightGBM with full features on the same scaffold split (what the notebooks do)."""
    import time
    import lightgbm as lgb
    fp = c.ecfp(az.mol); desc, _ = c.rdkit_descriptors(az.mol)
    X = np.hstack([desc, fp]); y = az.y.values; tr, te = c.split(az, "scaffold")
    t0 = time.time()
    g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=0).fit(X[tr], y[tr])
    secs = time.time() - t0
    return round(float(c.metrics(y[te], g.predict(X[te]))["R2"]), 3), round(secs, 1)


def main():
    hlm = c.load("HLM")
    floor = float(hlm.y.min())
    cens = (hlm.y.values <= floor + 1e-6).astype(int).tolist()
    tr, te = c.split(hlm, "scaffold")
    smiles = [s for s, t in zip(hlm.smiles, np.isin(np.arange(len(hlm)), te)) if not t and len(s) <= 55][:1500]
    d_h, te_h = build_dataset("HLM", hlm, {"floor": round(floor, 4), "cens": cens, "smiles": smiles,
                                           "label": "HLM clearance", "full": "Human liver microsome clearance (Biogen)", "unit": "log10 mL/min/kg"})
    d_h["smi_all"] = [str(s) for s in hlm.smiles]
    json.dump(d_h, open(os.path.join(OUT, "hlm.json"), "w"), separators=(",", ":"))

    az = c.load_logd()
    d_a, te_a = build_dataset("logD", az, {"label": "logD", "full": "logD at pH 7.4 (AstraZeneca, ChEMBL)", "unit": "log D", "smiles": []})
    d_a["smi_all"] = [str(s) for s in az.smiles]
    json.dump(d_a, open(os.path.join(OUT, "logd.json"), "w"), separators=(",", ":"))

    structs = {"HLM": structs_for(hlm, te_h, 120, 4, with_maps=14), "logD": structs_for(az, te_a, 60, 5, with_maps=14)}
    json.dump(structs, open(os.path.join(OUT, "structs.json"), "w"), separators=(",", ":"))

    graphs = []
    for m in hlm.mol:
        if 13 <= m.GetNumHeavyAtoms() <= 20 and len(graphs) < 6:
            g = graph_of(m); g["smi"] = Chem.MolToSmiles(m); graphs.append(g)
    meta = {}
    if os.path.exists(c.RESULTS):
        for k, v in json.load(open(c.RESULTS)).items():
            meta[k.split("|")[2]] = v
    r2_az, secs_az = ref_scores(az)
    refs = {"HLM": [["LightGBM, fingerprints + descriptors", 0.51, None], ["Neural network, fingerprints + descriptors", 0.43, None], ["Random forest, fingerprints + descriptors", 0.40, None], ["Graph neural network (PyTorch Geometric)", 0.29, None], ["GPT embeddings + ridge", 0.16, None]],
            "logD": [["LightGBM, fingerprints + descriptors", r2_az, secs_az]]}
    if "Chemprop (D-MPNN)" in meta:
        refs["HLM"].append(["Chemprop", round(meta["Chemprop (D-MPNN)"]["R2"], 3), round(meta["Chemprop (D-MPNN)"]["train_secs"], 1)])
    extras = {"graphs": graphs, "meta": meta, "refs": refs}
    ft = os.path.join(c.ROOT, "results", "chemprop_finetune.json")
    if os.path.exists(ft):
        extras["chemprop_ft"] = json.load(open(ft))
    json.dump(extras, open(os.path.join(OUT, "extras.json"), "w"), separators=(",", ":"))
    for f in sorted(os.listdir(OUT)):
        print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1024), "KB")
    print("censored HLM:", round(100 * np.mean(cens), 1), "% | seen scaffolds in unseen set:", d_h["seen"], "| mean max similarity:", d_h["sim"]["random"]["mean"], d_h["sim"]["scaffold"]["mean"])


if __name__ == "__main__":
    main()
