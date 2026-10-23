# Instructor Guide (90 minutes)

Open the website and let participants follow along. Each lesson has a fixed options panel on the left. The three model lessons start with a workbench: drag the options and the picture changes immediately, then press Train to watch a slowed-down fit. The steps below the workbench scroll and respond to the same options.

## How the workbench works

* The left picture follows the options as you drag them (tree depth, number of trees, layers, neurons, transformer layers).
* Train fits the model, then replays the fit at the chosen speed. Slow shows every epoch, round or step. Medium and Fast skip ahead. The real training time is reported separately and does not include the replay delay.
* If you change an option after training, the stage says so and the old result stays on screen until you train again.
* Filter training data is in the options of every lesson. Presets in each lesson's filtering step train the model with and without the filter on the same unseen compounds.

## Run of show

| Minutes | Lesson | What to do |
|---|---|---|
| 0 to 14 | 01 Data and ML basics | Step 1 gives the idea. Step 3 shows how the unseen set is chosen: in HLM the scaffold split barely matters, in logD it matters a lot. Steps 4 and 5 show the pile-up at the assay floor and three ways to treat it. Step 6 is the key picture for overfitting. |
| 14 to 36 | 02 Tree models | Use the workbench: ask the group what depth 3 versus 10 will do, then press Train. Switch to forest and boosting. Finish with the filtering step. |
| 36 to 62 | 03 Neural networks | Workbench first (layers, neurons, dropout, learning rate, epochs). Then the learning-rate landscape, dropout, the filtering step, and a trained network opened up. Chemprop is the last step. |
| 62 to 82 | 04 GPT | Train once in the workbench (about 25 seconds at the default). Discuss the confidence ribbon and the molecule it writes while it runs. Attention, next-token odds and RDKit validity follow. |
| 82 to 90 | 05 Compare | Walk the leaderboard and the accuracy against time chart. Use the discussion questions. |

## Points to land

* The score depends on how the unseen set was chosen and how the data were treated. Report both.
* Censoring changes which model looks best. Keeping everything is best on all unseen compounds, dropping censored values is best on measurable ones, and two models give both answers.
* Trees need no scaling and are hard to beat on a few thousand compounds. Neural networks need care: scaling, validation, dropout, a stopping point.
* A GPT is a generative model. Predicting clearance is not its job. Narrowing the molecular weight range changes what it writes. Check what it writes with RDKit.
* One endpoint can feed another (logD into HLM). The gain depends on how much the other features already say.

## If something goes wrong

* A blank page or no data: the site must be opened over http (GitHub Pages or `python -m http.server`), not from a file.
* The GPT does not train: the TensorFlow.js script may be blocked by a network proxy. Participants can use the notebooks instead.
* RDKit does not load (the generation step and the predict-your-own-SMILES step): it comes from a public CDN and is about 7 MB. The rest of the site works without it.
* Slow laptop: switch animations off in the header, lower the GPT training steps to 150, and use one transformer layer.
