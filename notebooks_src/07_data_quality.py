# %% [markdown]
# # 07 Data quality: censored HLM values
# 31% of the HLM values sit exactly at the assay floor. The assay cannot read clearance below that value, so every slower compound is reported as the floor. This notebook compares three ways to train on such data, all scored on the same unfiltered unseen scaffolds.

# %%
import os, sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit", "lightgbm"], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import lightgbm as lgb
import common as c

df = c.load("HLM"); fp, desc, names = c.get_features(df); X = np.hstack([desc, fp]); y = df.y.values
floor = y.min(); cens = (y <= floor + 1e-6); tr, te = c.split(df, "scaffold")
print(f"floor {floor:.3f}, censored {cens.mean():.1%}, censored in unseen set {cens[te].mean():.1%}")
plt.hist(y, bins=50); plt.axvline(floor, color="r"); plt.xlabel("log10 HLM CLint"); plt.ylabel("compounds"); plt.show()

# %% [markdown]
# ## Three strategies

# %%
mk = lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=0)
unc_te = te[~cens[te]]
rows = []

# 1. keep everything
m1 = mk().fit(X[tr], y[tr]); p1 = m1.predict(X)
rows.append(("Keep all", c.metrics(y[te], p1[te])["R2"], c.metrics(y[unc_te], p1[unc_te])["R2"]))

# 2. drop censored compounds from training
tr_u = tr[~cens[tr]]; m2 = mk().fit(X[tr_u], y[tr_u]); p2 = m2.predict(X)
rows.append(("Drop censored from training", c.metrics(y[te], p2[te])["R2"], c.metrics(y[unc_te], p2[unc_te])["R2"]))

# 3. two models: classifier for the floor, regressor for the rest
clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=0).fit(X[tr], cens[tr])
pc = clf.predict_proba(X)[:, 1]; p3 = pc * floor + (1 - pc) * p2
rows.append(("Two models", c.metrics(y[te], p3[te])["R2"], c.metrics(y[unc_te], p2[unc_te])["R2"]))
res = pd.DataFrame(rows, columns=["strategy", "R2 all unseen", "R2 measurable unseen"]).round(3)
print("floor classifier accuracy on unseen:", round(float(((pc[te] > 0.5) == cens[te]).mean()), 3))
res

# %% [markdown]
# Reading the table: keeping everything looks fine on all unseen compounds but ranks the measurable ones poorly. Dropping the censored values ranks measurable compounds better but cannot say which compounds are stable. Two models give both answers separately.
#
# **Discuss:** which number would you report to a project team that wants to prioritise stable compounds, and which to a team optimising potency?
