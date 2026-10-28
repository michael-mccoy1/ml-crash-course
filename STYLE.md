# Writing style for the course site

## Audience

Scientists: medicinal chemists, DMPK scientists, computational chemists. They know chemistry, assays and basic statistics. They may know nothing about machine learning. Do not explain chemistry (logP, TPSA, molecular weight, SMILES notation, scaffolds, assays, clearance). Explain the machine learning, and how a molecule and an assay result are represented as model input.

Beginner means no machine learning experience. It does not mean no science.

## Tone

* Neutral, precise, concise technical prose. Impersonal. Active voice. Short sentences.
* Use the standard term from the first use: weights, bias, loss, gradient, residual, validation set, hyperparameter, overfitting, generalisation, applicability domain, RMSE, R2. Define each term once, where it first appears, in one clause.
* No everyday analogies: no students, exams, houses, phones, games, bags, crowds, seats, dials, votes.
* No anthropomorphism. A model is trained on data, receives features and returns predictions. It does not study, see, guess, learn a lesson, remember or know. An optimiser adjusts weights. "The computer" is not an agent.
* No rhetorical questions. No exclamations. No second-person storytelling ("you will see", "you might wonder").
* Forbidden words and phrases: why this matters, imagine, think of, notice how, let's, here is how, in plain words, in simple terms, simply, just, head start, sweet spot, magic, powerful, crucial, key insight, story, journey, dive, unlock, leverage, robust, seamless, "it turns out", "the idea is", "the trick is", "under the hood", "a closer look", "everything you need".
* Do not state that something is easy, simple or intuitive. Do not praise.
* State facts and numbers precisely. Do not hedge with filler. Where a result depends on the data set, say which one.
* Say "test set" and "test compounds". Never "unseen". Say "held-out" where a test set is not meant.
* Typography: no em dashes, no en dashes, no middle dots, no multiplication signs, no ellipsis characters. Write "times" or use a plain x only inside code. No time estimates.

## Titles

Step titles are set in one place (docs/js/titles.js). Sentence case, noun phrases, six words or fewer, no colons, no questions, no verbs in the imperative. Examples: "Molecular representations", "Split selection", "Gradient descent", "Applicability domain".

## Interface text

* TRY lines: one or two imperative sentences naming the control. "Drag the cut position. Select a feature."
* NOTICE lines: one factual statement about what the figure shows.
* Quiz questions: one precise question with a short precise answer.
* Panel headings and button labels: short, noun phrases or plain verbs. "Train", "Next compound", "Reset". Not "Tune the dials for me".
* Walk-through captions: one factual sentence each.
