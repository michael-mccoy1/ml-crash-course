# ML Crash Course: Trees, Neural Nets and GPTs for ADME

A 90-minute course built on the public **Biogen ADME dataset** (Fang et al., *J. Chem. Inf. Model.* 2023; data from
[molecularinformatics/Computational-ADME](https://github.com/molecularinformatics/Computational-ADME)).
The task: predict log10 human liver microsome clearance (HLM CLint) from molecular structure.

## For participants: nothing to install

Open the course website (GitHub Pages): **`https://<your-org>.github.io/<repo-name>/`**

Every model trains inside your browser tab. Move the sliders, press Train, and watch what changes.

| Tab | Time | What you explore |
|---|---|---|
| Start | 10 min | The data, features, scaffold split |
| Trees | 25 min | Decision tree depth and overfitting, random forest, gradient boosting |
| Neural nets | 25 min | Layers, units, dropout, learning rate, early stopping; live loss curves |
| GPT | 20 min | Train a tiny transformer on SMILES, sample molecules, view attention |
| Compare | 10 min | Your scores side by side, plus discussion questions |

## Optional deep dive (Colab, free Google account)

The `notebooks/` folder repeats each block with full ECFP fingerprints and ~200 RDKit descriptors, adds a graph neural network and
an optional ChemBERTa comparison. Open a notebook via
`https://colab.research.google.com/github/<your-org>/<repo-name>/blob/main/notebooks/01_tree_models.ipynb`
(set `REPO_URL` in notebook 00 first).

| Notebook | Content |
|---|---|
| `00_setup_and_data` | Data, featurisation, random vs scaffold split, baseline |
| `01_tree_models` | Tree, forest, LightGBM, importances, split honesty check |
| `02_neural_networks` | MLP in PyTorch, regularisation, GNN |
| `03_gpt` | Tiny GPT from scratch, generation, embeddings for prediction |
| `04_compare_and_discuss` | Results table, variance over splits, discussion |

Reference scores (one run, scaffold split, full features): LightGBM R² 0.51, MLP 0.43, random forest 0.40, GNN 0.29, GPT embeddings + ridge 0.16.

## Publishing the website

1. Create a repo and push this folder.
2. Repo **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs`**.
3. The site appears at `https://<your-org>.github.io/<repo-name>/` after about a minute.

The site is a single static file, `docs/index.html`. It loads TensorFlow.js from a CDN; there is no server component.
If you edit `applet/applet_template.html`, regenerate with `python build_applet.py`.

## Repository layout

```
docs/index.html            the website (generated)
applet/applet_template.html  source for the website
build_applet.py            template + data -> docs/index.html
notebooks_src/*.py         notebook sources (percent format)
build_notebooks.py         notebooks_src -> notebooks/*.ipynb
notebooks/*.ipynb          Colab-ready notebooks
common.py                  shared data/feature/split/metric helpers
data/biogen_adme.csv       public dataset (3,521 compounds)
results/applet_data.json   8 descriptors + HLM values + SMILES used by the website
```

Run notebooks locally: `pip install rdkit scikit-learn lightgbm torch torch_geometric matplotlib pandas scipy`.
