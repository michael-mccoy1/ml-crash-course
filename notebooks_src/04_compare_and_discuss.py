# %% [markdown]
# # 04: Comparison and discussion
# Every model from blocks 01-03 logged its score on the **same scaffold-split test set**. (Missing a row? Run that notebook first.)

# %%
import sys, json
sys.path.insert(0, "..")
import pandas as pd, matplotlib.pyplot as plt
import common as c

res = json.load(open(c.RESULTS))
rows = [{"model": k.split("|")[2], **v} for k, v in res.items() if k.startswith("HLM|scaffold")]
tab = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
tab.round(3)

# %%
ax = tab.set_index("model")["R2"][::-1].plot.barh(figsize=(7, 4))
ax.set_xlabel("test R² (scaffold split, HLM)"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Discussion questions (use the paper)
# 1. **Does the winner here match what Fang et al. found?** What would you need to change to claim one model family is "better"? (Hint: repeats over seeds/splits, confidence intervals, one split can mislead.)
# 2. **Why do trees often win on tabular chemistry data** while NNs dominate images/text? (Few samples, heterogeneous hand-crafted features, irregular target functions.)
# 3. **The GPT scored lowest, is it a bad model?** It is a *generative* model trained on 2.5k molecules; its strength is scale + generation, not small-data regression.
# 4. **Retrospective vs prospective:** the paper evaluates on compounds measured *after* model building. Which of our splits is closest, and what's still missing?
# 5. **Assay noise ceiling:** HLM log-CLint replicate error limits the achievable R². How would you estimate it?
#
# ## Optional: quantify the variance of a single split
# Re-run LightGBM over 5 random seeds of the *scaffold* split.

# %%
import numpy as np, lightgbm as lgb
df = c.load("HLM"); fp, desc, _ = c.get_features(df); X = np.hstack([desc, fp]); y = df.y.values
scores = []
for seed in range(5):
    a, b = c.split(df, "scaffold", seed=seed)
    g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, verbose=-1, random_state=seed).fit(X[a], y[a])
    scores.append(c.metrics(y[b], g.predict(X[b]))["R2"])
print("R² per split:", np.round(scores, 3), "| mean ± sd:", np.mean(scores).round(3), "±", np.std(scores).round(3))
