# Instructor Guide (90 minutes)

Open the website and let participants follow along. Each lesson has a fixed options panel on the left. Steps on the right play once as they scroll into view, and every step has a Play button to replay and a Reset button. The two toggles in the header turn animation and autoplay off.

## Run of show

| Minutes | Lesson | What to do |
|---|---|---|
| 0 to 12 | 01 Data and ML basics | Step 1 gives the idea. Step 3 shows how the unseen set is chosen: in HLM the scaffold split barely matters, in logD it matters a lot. Step 4 is the key picture for overfitting. Step 6 introduces logD feeding HLM. |
| 12 to 24 | 02 Data quality | Step 1 shows the 31% pile-up at the assay floor. Step 2 compares three treatments of censored data. Step 5 shows why the unseen set must not be filtered. |
| 24 to 44 | 03 Tree models | Step 1 shows how a tree picks a question. Ask the group to predict what happens at depth 3 versus 10 before pressing Play in step 2. Steps 4 and 5 show forests and boosting. |
| 44 to 64 | 04 Neural networks | Let participants drag the weights in step 1. Step 3 compares learning rates. Train the network once in step 4 and keep the group on step 6 and 7 while it trains if needed. Step 9 introduces Chemprop. |
| 64 to 82 | 05 GPT | Train once in step 4 (about 30 seconds at 400 steps) and discuss while it runs. Step 5 shows attention. Step 7 loads RDKit in the browser and checks generated molecules. |
| 82 to 90 | 06 Compare | Walk the leaderboard and the accuracy against time chart. Use the discussion questions. |

## Points to land

* The score depends on how the unseen set was chosen and how the data were treated. Report both.
* Censoring changes which model looks best. Keeping everything is best on all unseen compounds, dropping censored values is best on measurable ones, and two models give both answers.
* Trees need no scaling and are hard to beat on a few thousand compounds. Neural networks need care: scaling, validation, dropout, a stopping point.
* A GPT is a generative model. Predicting clearance is not its job. Check what it writes with RDKit.
* One endpoint can feed another (logD into HLM). The gain depends on how much the other features already say.

## If something goes wrong

* A blank page or no data: the site must be opened over http (GitHub Pages or `python -m http.server`), not from a file.
* Training does not start: the TensorFlow.js script may be blocked by a network proxy. Participants can use the notebooks instead.
* RDKit does not load (steps 7 and the predict-your-own-SMILES step): it comes from a public CDN and is about 7 MB. The rest of the site works without it.
* Slow laptop: turn animations off in the header, lower the GPT training steps to 150 and use one transformer layer.
