# Instructor Guide (90 minutes)

**Anchor paper:** Fang et al., *Prospective Validation of Machine Learning Algorithms for ADME Prediction: An Industrial Perspective*, J. Chem. Inf. Model. 2023 (verify the exact citation before you circulate it). Biogen measured ADME endpoints for ~3,500 compounds and released them. The paper's lesson for us: judge models on data they could not have seen, and compare model families fairly.

**Setup:** share the website link. Participants need a browser and nothing else. Ask them to keep the Compare tab for the end.

## Run of show

The site has an "Animations on/off" button in the header. Turn animations off if a laptop struggles.

| Min | Tab | Say | Do |
|---|---|---|---|
| 0–5 | What is ML? | HLM clearance: why chemists care (half-life, dose). Target is log10, higher = faster metabolism. | Read the question card. |
| 5–10 | What is ML? | Scaffold split holds out whole chemical series, like predicting the next series. Features are 8 physchem descriptors. | Untick features, ask which they expect to matter. |
| 10–15 | Trees | A tree is yes/no questions; the prediction is a leaf average. | Train with defaults. Read the printed rules aloud. |
| 15–22 | Trees | Overfitting. | Point at the depth chart: train rises, test stalls. Ask which depth to pick and why not by the test curve. |
| 22–35 | Trees | Forest = average many noisy trees; boosting = trees fixing errors. | Move forest depth 3 vs 16, boosting rate 0.01 vs 0.3. Tick only logP and re-train. |
| 35–45 | Neural networks | Layers of weighted sums + ReLU, trained by gradient descent; inputs must be scaled. | Train defaults. Explain the two loss curves. |
| 45–60 | Neural networks | Regularisation. | Units 256, dropout 0, epochs 200: curves split. Add dropout 0.4. Toggle "keep best epoch". Compare with the forest. |
| 60–68 | GPT | Same machinery, different job: predict the next token. | Train 500 steps while you explain tokens, embeddings, attention. |
| 68–75 | GPT | Sampling and temperature. | Generate at 0.3 and 1.6. Count syntax-ok. |
| 75–80 | GPT | Attention. | Show the lower-triangle heatmap; why the upper half is blank (no peeking at the future). Ring-closing digits often attend to their opening digit. |
| 80–90 | Compare | Rank, then discuss. | Use the five questions on the page. |

## Optional extensions

- **logD to HLM (What is ML? tab):** tick logD_pred and run the comparison. Expect a small gain with eight descriptors (about +0.03 to +0.06) and almost none with full fingerprints. Discuss when a stacked feature helps.
- **Chemprop (Neural networks tab):** step through message passing, then show the notebook 06 score (R² 0.46, 75 s on CPU).

## Expected outcomes (8 descriptors, scaffold split)

Roughly: tree depth 4 R² ≈ 0.25, forest ≈ 0.29, boosting ≈ 0.28, MLP ≈ 0.3. Ranking among them is within noise; say so. The notebooks with ECFP + 200 descriptors reach LightGBM ≈ 0.51 and show clearer separation.

## Points to land

- **Trees:** one tree overfits; ensembles fix variance (forest) or bias (boosting); trees need no scaling and handle small tabular data well.
- **Neural nets:** more capacity does not mean better test scores; validation data picks the stopping point; scale your inputs.
- **GPT:** it is a generative model of text. Predicting clearance is not its job, which is why its regression score is lowest. Scale, not architecture, makes large models strong.
- **Evaluation:** one split is one sample. The Compare tab's reference table comes from single runs; see notebook 04 for variance over splits.

## If something goes wrong

- Page loads but Train does nothing: TensorFlow.js is blocked by a network proxy. Participants can use the notebooks instead, or you can host `tf.min.js` in `docs/` and change the script tag.
- GPT training is slow on old laptops: lower steps to 200–300 and use 1 layer.
- Keep the tab in the foreground while training.
