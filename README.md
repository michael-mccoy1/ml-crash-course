# Model Lab: AI/ML for ADME

A course on tree models, neural networks and GPTs. Two public endpoints:

* **HLM clearance**: Biogen set (Fang et al., J. Chem. Inf. Model. 2023), 3,087 compounds. 31% sit at the assay floor.
* **logD**: AstraZeneca set from ChEMBL via MoleculeNet, 4,200 compounds.

RDKit builds the data and also runs in the browser.

## Website

**https://michael-mccoy1.github.io/ml-crash-course/**

Nothing to install. Each lesson lists its goals, has an options panel, steps with a Notice line and a Pop quiz, and a glossary. The tree, network and GPT lessons have a workbench: change an option and the picture updates, press Train and the fit replays slowly. Every lesson has Filter training data in its options.

| Lesson | Content |
|---|---|
| 01 Data and ML basics | Fitting a line, features, the test set, censoring, overfitting, scoring, logD feeding HLM |
| 02 Tree models | Workbench, one cut, one compound, forest, boosting, features, filtering, all three side by side, seeds and ensembles |
| 03 Neural networks | Weights, one neuron, workbench, learning rate, dropout, filtering, Chemprop, a comparison test, seeds and ensembles |
| 04 GPT | Tokens, embeddings, the stack, workbench, attention, temperature, RDKit check, filtering, a design test |
| 05 Pre-trained models | Why pre-train, hiding atoms and groups, GROVER, fixed features, fine-tuning a GPT and Chemprop |
| 06 Compare | Leaderboard, accuracy against time, discussion |
| 07 Role of AI | Machine learning against generative AI, which fits which job, the training range, lab noise, how generative AI fails, a can and cannot sort, five checks |
| 08 Tree algorithms | Split search, depth and leaf size, bagging, boosting losses, tuning, how the libraries differ |
| 09 Network algorithms | Gradient descent, backpropagation, initialisation, weight decay, attention, next-token loss, parameter counts |

## Notebooks

`notebooks/` covers the same ideas in Python (Colab or local): setup, trees, networks, GPT, comparison, logD to HLM, Chemprop, data quality, and Chemprop fine-tuning.

Reference results (scaffold split, fingerprints and descriptors): LightGBM R2 0.51, Chemprop 0.46, MLP 0.43, random forest 0.40, graph network 0.29. Chemprop pre-trained on logD, then fine-tuned on 1,000 HLM compounds: R2 0.41 for full fine-tuning, 0.28 frozen encoder, 0.24 from scratch.

Install: `pip install rdkit scikit-learn lightgbm torch torch_geometric chemprop matplotlib pandas scipy`

## Publishing

The site is `docs/`. The workflow in `.github/workflows/pages.yml` deploys it on every push to `main`. Set Pages source to GitHub Actions.

## Rebuilding

```
python build_data.py        # docs/data/*.json
python build_notebooks.py   # notebooks_src/*.py to notebooks/*.ipynb
```

Preview: run `python -m http.server` in `docs` and open http://localhost:8000.

## Layout

```
docs/            the website (index.html, css, js, data, vendor, assets)
build_data.py    builds docs/data from RDKit
common.py        shared data, feature, split and metric helpers
data/            input CSVs
notebooks_src/   notebook sources
notebooks/       built notebooks
```
