"""Shared helpers for the ML crash course (data, features, splits, metrics)."""
import json
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "biogen_adme.csv")
RESULTS = os.path.join(ROOT, "results", "results.json")

ENDPOINTS = {
    "HLM": "LOG HLM_CLint (mL/min/kg)",
    "RLM": "LOG RLM_CLint (mL/min/kg)",
    "MDCK_ER": "LOG MDR1-MDCK ER (B-A/A-B)",
    "SOL": "LOG SOLUBILITY PH 6.8 (ug/mL)",
}


def load(endpoint="HLM"):
    """Return DataFrame[smiles, y] for one endpoint (log10 units)."""
    df = pd.read_csv(DATA)
    col = ENDPOINTS[endpoint]
    df = df[["SMILES", col]].dropna().rename(columns={"SMILES": "smiles", col: "y"})
    df["mol"] = df.smiles.apply(Chem.MolFromSmiles)
    return df[df.mol.notna()].reset_index(drop=True)


def ecfp(mols, radius=2, n_bits=2048):
    """Count-based Morgan fingerprints (ECFP4 by default)."""
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def rdkit_descriptors(mols):
    """~200 RDKit 2D physchem descriptors. Returns (X, names)."""
    names = [n for n, _ in Descriptors.descList]
    X = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)
    return X, names


def split(df, kind="scaffold", test_frac=0.2, seed=0):
    """Return (train_idx, test_idx). 'scaffold' puts whole Murcko scaffolds on one side."""
    n = len(df)
    rng = np.random.RandomState(seed)
    if kind == "random":
        idx = rng.permutation(n)
        k = int(n * test_frac)
        return idx[k:], idx[:k]
    scaf = df.mol.apply(lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s, []).append(i)
    sets = sorted(groups.values(), key=lambda g: (-len(g), rng.rand()))
    train, test = [], []
    for g in sets:  # big scaffolds -> train, small/singletons -> test
        (train if len(train) < (1 - test_frac) * n else test).extend(g)
    return np.array(train), np.array(test)


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred)[0]),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def log_result(name, m, split_kind="scaffold", endpoint="HLM"):
    """Append a result so the final notebook can compare every model."""
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    res[f"{endpoint}|{split_kind}|{name}"] = m
    json.dump(res, open(RESULTS, "w"), indent=1)
    print(f"{name:28s} R2={m['R2']:.3f}  Spearman={m['Spearman']:.3f}  RMSE={m['RMSE']:.3f}")


def get_features(df, cache="features_HLM.npz"):
    """ECFP + descriptors, cached on disk (descriptor calc takes ~30 s)."""
    path = os.path.join(ROOT, "results", cache)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["fp"], z["desc"], list(z["names"])
    fp = ecfp(df.mol)
    desc, names = rdkit_descriptors(df.mol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, fp=fp, desc=desc, names=np.array(names))
    return fp, desc, names
