"use strict";
/* Lesson 11: machine learning in PK, PBPK and MIDD. The steps are in mlpk_a.js (Beginner), mlpk_b.js (Intermediate) and mlpk_c.js (Advanced and Pro). */
var L11 = new Lesson('mlpk', 'ML in PK and MIDD');
L11.noOpts = true;
L11.glossary([
  ['Surrogate model', 'A machine learning model trained on the outputs of a simulator, used as a faster approximation of it.'],
  ['Hybrid model', 'A mechanistic model in which one unknown term is represented by a learned function.'],
  ['Neural ODE', 'A model whose derivative function is a neural network, integrated with an ODE solver.'],
  ['Covariate screening', 'Identifying which covariates explain between-subject variability.'],
  ['Prediction interval', 'A range expected to contain a new observation with a stated probability.'],
  ['Fold error', 'The ratio of a predicted value to the observed value, or its inverse, whichever is greater than 1.'],
  ['Error propagation', 'The change in a model output caused by uncertainty in its inputs.'],
  ['Transfer learning', 'Reuse of a model trained on one endpoint to improve prediction of another.'],
  ['Applicability domain', 'The region of feature space in which a model is reliable.'],
  ['Ensemble', 'Several models, or several runs of one model, combined into one prediction.']
]);
