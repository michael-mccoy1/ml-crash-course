# %% [markdown]
# # 01 · Tree Models (25 min)
# **Idea:** repeatedly ask yes/no questions about features ("is logP > 3.1?") to split compounds into groups with similar `y`; predict the group mean.
#
# * **Decision tree** – one tree. Interpretable, but overfits.
# * **Random forest** – hundreds of trees on random data/feature subsets, averaged (*bagging*). Reduces variance.
# * **Gradient boosting** (LightGBM/XGBoost) – trees built *sequentially*, each fixing the errors of the last. Usually the strongest on tabular data.

# %%
import os, sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit", "lightgbm"], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import common as c
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

df = c.load("HLM"); fp, desc, names = c.get_features(df)
tr, te = c.split(df, "scaffold"); y = df.y.values
X_all = np.hstack([desc, fp]); all_names = names + [f"ecfp_{i}" for i in range(fp.shape[1])]

# %% [markdown]
# ## 1. A single decision tree, readable by a human
# Depth 3 on descriptors only. Read the rules aloud: which properties drive clearance?

# %%
t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(desc[tr], y[tr])
print(export_text(t, feature_names=names))
print(c.metrics(y[te], t.predict(desc[te])))

# %% [markdown]
# ## 2. Overfitting in one plot
# Let the tree grow deeper. Training score → perfect; test score stalls then falls.

# %%
depths = range(1, 25)
rows = []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(desc[tr], y[tr])
    rows.append((c.metrics(y[tr], m.predict(desc[tr]))["R2"], c.metrics(y[te], m.predict(desc[te]))["R2"]))
rows = np.array(rows)
plt.plot(depths, rows[:, 0], label="train R²"); plt.plot(depths, rows[:, 1], label="test R²")
plt.xlabel("max_depth"); plt.ylabel("R²"); plt.legend(); plt.show()

# %% [markdown]
# **Your turn #1:** which depth would you pick, and why can't you choose it using the test set?  (Hint: validation set / cross-validation.)

# %% [markdown]
# ## 3. Random forest

# %%
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=0)
rf.fit(X_all[tr], y[tr])
c.log_result("Random forest (desc+ECFP)", c.metrics(y[te], rf.predict(X_all[te])))

# %% [markdown]
# ## 4. Gradient boosting

# %%
gb = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.3, min_child_samples=10, random_state=0, verbose=-1)
gb.fit(X_all[tr], y[tr])
c.log_result("LightGBM (desc+ECFP)", c.metrics(y[te], gb.predict(X_all[te])))

# %% [markdown]
# **Your turn #2:** try `desc` only, or `fp` only, for LightGBM. Which feature set matters more? (Pairs with the *"82% of the performance from two descriptors"* style of finding.)

# %%
for label, X in [("descriptors only", desc), ("ECFP only", fp)]:
    g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=0).fit(X[tr], y[tr])
    print(label, {k: round(v, 3) for k, v in c.metrics(y[te], g.predict(X[te])).items()})

# %% [markdown]
# ## 5. What did the model learn? Feature importance

# %%
imp = pd.Series(gb.booster_.feature_importance("gain"), index=all_names).sort_values(ascending=False)[:15]
imp[::-1].plot.barh(); plt.xlabel("total gain"); plt.show()

# %% [markdown]
# ## 6. Random vs scaffold split – the honesty check

# %%
for kind in ["random", "scaffold"]:
    a, b = c.split(df, kind)
    g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=0).fit(X_all[a], y[a])
    print(f"{kind:9s}", {k: round(v, 3) for k, v in c.metrics(y[b], g.predict(X_all[b])).items()})

# %% [markdown]
# **Discuss:** the gap is small on this dataset (Biogen's compounds come from many vendor collections, so scaffold overlap is limited anyway). Why might the gap be much larger on an in-house project series? Which number would you quote to a project team?
