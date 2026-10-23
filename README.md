# Model Lab: AI/ML for ADME

A course on tree models, neural networks and GPTs, built on two public ADME endpoints:

* **HLM clearance**: Biogen ADME set (Fang et al., J. Chem. Inf. Model. 2023), 3,087 compounds. 31% of the values sit at the assay floor.
* **logD**: AstraZeneca set from ChEMBL via MoleculeNet, 4,200 compounds. A logD model can feed its prediction into the HLM model.

Structures, descriptors, scaffolds and similarity statistics come from RDKit. RDKit also runs in the browser (loaded on demand) to draw and validate molecules.

## Website

**https://michael-mccoy1.github.io/ml-crash-course/**

Nothing to install. Each lesson opens with a card that says how long it takes, what you will be able to do, and the words you will meet. Every step has a line on what to notice and a Pop quiz question with a hidden answer. Each lesson has a fixed options panel on the left. The three model lessons open with a workbench: drag an option and the picture changes at once, press Train and the fit replays slowly with the error curve and predictions updating. Steps below the workbench respond to the same options. Filtering the training data is part of every lesson, and each lesson ends with its own glossary.

| Lesson | Content |
|---|---|
| 01 Data and ML basics | What learning means (fit a line), structure to prediction, descriptor maps, the unseen set, the target distribution, censoring, over and underfitting, scoring, logD feeding HLM, predict your own SMILES |
| 02 Tree models | Workbench (tree, forest, boosting), split search, one compound down the tree, forest votes, boosting, feature importance, filtering the data |
| 03 Neural networks | Workbench, one neuron, layers, learning rate, dropout, filtering the data, one compound through the network, one neuron opened, what it learned, Chemprop |
| 04 GPT | Workbench, tokens, embeddings, the transformer stack, attention, next-token odds, generation checked by RDKit, filtering the corpus |
| 05 Pre-trained models | Why pre-train, hiding atoms and groups (the pre-training tasks), GROVER and its scale, the GPT as a fixed feature extractor, fine-tuning (head only or everything) against a random start |
| 06 Compare | Leaderboard, accuracy against training time, discussion questions |

The endpoint (HLM or logD) and the feature toggles are shared by all lessons. The GPT always reads the Biogen SMILES.

## Notebooks (Colab or local)

| Notebook | Content |
|---|---|
| `00_setup_and_data` | Data, features, scaffold split, baseline |
| `01_tree_models` | Tree, forest, LightGBM, importances |
| `02_neural_networks` | MLP and a graph network |
| `03_gpt` | Tiny GPT from scratch, generation, embeddings |
| `04_compare_and_discuss` | Results table and variance over splits |
| `05_logd_to_hlm` | logD model trained on AstraZeneca data, predicted logD as an HLM feature |
| `06_chemprop` | Chemprop on the same scaffold split |
| `07_data_quality` | Three ways to treat censored HLM values |
| `08_chemprop_finetune` | Chemprop pre-trained on logD, then fine-tuned on few HLM compounds: from scratch, frozen encoder, full fine-tuning |

Reference results on the scaffold split with fingerprints and descriptors: LightGBM R2 0.51, Chemprop 0.46, MLP 0.43, random forest 0.40, graph network 0.29, GPT embeddings with ridge 0.16. Predicted logD adds about +0.06 R2 to eight descriptors and +0.005 to full fingerprints. Dropping censored values from training raises R2 on measurable compounds from 0.29 to 0.34 and lowers it on all unseen compounds.

Chemprop pre-trained on logD and fine-tuned on HLM (notebook 08, two random draws, 30 epochs): with 1,000 labelled HLM compounds the unseen R2 is 0.41 for full fine-tuning, 0.28 for a frozen encoder and 0.24 from scratch. With 50 compounds it is 0.15, -0.38 and 0.05.

Local run: `pip install rdkit scikit-learn lightgbm torch torch_geometric chemprop matplotlib pandas scipy`.

## Publishing

The site is the `docs` folder. A GitHub Actions workflow (`.github/workflows/pages.yml`) deploys it on every push to `main`. In the repository settings, set Pages source to GitHub Actions.

## Rebuilding

```
python build_data.py        # docs/data/*.json from RDKit
python build_notebooks.py   # notebooks_src/*.py to notebooks/*.ipynb
```

To preview the site, run `python -m http.server` inside `docs` and open http://localhost:8000.

## Layout

```
docs/index.html, css/site.css, js/*.js    the website
docs/data/                                hlm.json, logd.json, structs.json (RDKit drawings), extras.json
docs/assets/                              images
build_data.py, build_notebooks.py         builders
common.py                                 shared data, feature, split and metric helpers
data/                                     biogen_adme.csv, lipophilicity_az.csv
notebooks_src/, notebooks/                notebook sources and .ipynb files
```
