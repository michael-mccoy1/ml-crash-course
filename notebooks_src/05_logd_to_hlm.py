# %% [markdown]
# # 05: A second endpoint: logD as an input to HLM
# HLM clearance depends on lipophilicity. We can train a **logD model on a different public dataset** (AstraZeneca, 4,200 compounds, ChEMBL deposit) and feed its prediction into the HLM model as one extra feature. This is *stacking* across endpoints.
#
# * logD model is trained on AstraZeneca data only. It never sees an HLM value, so there is no leakage.
# * Biogen compounds get a **predicted** logD (no measured logD needed).

# %%
import os, sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit", "lightgbm"], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import lightgbm as lgb
import common as c
from sklearn.model_selection import KFold, cross_val_predict

# %% [markdown]
# ## 1. Train the logD model (AstraZeneca)

# %%
az = c.load_logd()
fp_az = c.ecfp(az.mol); d_az, _ = c.rdkit_descriptors(az.mol)
X_az = np.hstack([d_az, fp_az]); y_az = az.y.values
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)
cv = cross_val_predict(mk(), X_az, y_az, cv=KFold(5, shuffle=True, random_state=0))
print("logD 5-fold CV:", {k: round(v, 3) for k, v in c.metrics(y_az, cv).items()})
logd_model = mk().fit(X_az, y_az)

# %% [markdown]
# ## 2. Predict logD for the Biogen compounds and save it as a feature

# %%
df = c.load("HLM"); fp, desc, names = c.get_features(df)
logd_pred = logd_model.predict(np.hstack([desc, fp]))
pd.DataFrame({"smiles": df.smiles, "logD_pred": logd_pred}).to_csv(os.path.join(c.ROOT, "results", "logd_pred.csv"), index=False)
print("predicted logD range:", logd_pred.min().round(2), logd_pred.max().round(2))
plt.scatter(logd_pred, df.y, s=6, alpha=.4); plt.xlabel("predicted logD"); plt.ylabel("HLM log CLint"); plt.show()
print("Pearson r with HLM:", np.corrcoef(logd_pred, df.y)[0, 1].round(3))

# %% [markdown]
# ## 3. Does the extra feature help HLM?
# Same scaffold split, same LightGBM settings. Repeat over a few seeds so a single split doesn't mislead.

# %%
def run(X, seed):
    tr, te = c.split(df, "scaffold", seed=seed); y = df.y.values
    g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=seed).fit(X[tr], y[tr])
    return c.metrics(y[te], g.predict(X[te]))["R2"]

sets = {"8 descriptors": desc[:, [names.index(n) for n in ["MolLogP", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "FractionCSP3", "NumAromaticRings"]]],
        "descriptors + ECFP": np.hstack([desc, fp])}
rows = []
for label, X in sets.items():
    base = [run(X, s) for s in range(5)]
    plus = [run(np.hstack([X, logd_pred[:, None]]), s) for s in range(5)]
    rows.append({"features": label, "without logD_pred": np.mean(base), "with logD_pred": np.mean(plus), "gain": np.mean(plus) - np.mean(base)})
res = pd.DataFrame(rows).round(3); res

# %%
tr, te = c.split(df, "scaffold"); X = np.hstack([desc, fp, logd_pred[:, None]])
g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=0).fit(X[tr], df.y.values[tr])
c.log_result("LightGBM + predicted logD", c.metrics(df.y.values[te], g.predict(X[te])))

# %% [markdown]
# **Discuss:** the logD model's own error propagates into the HLM model. When would you expect a stacked feature to help most, and when would the HLM model already have the information?
