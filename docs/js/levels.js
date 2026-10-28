"use strict";
/* Which section of each lesson a step belongs to. Steps that are not listed are Intermediate. */
var STEP_LEVELS = {
  start: { beginner: ['Molecular representations', 'The data is a table', 'What learning means', 'The test set', 'Practice and real exam', 'Scoring a model'],
    intermediate: ['Structure to prediction', 'Where features come from', 'The target', 'Underfitting and overfitting'],
    advanced: ['Handling censoring'], pro: ['logD feeds HLM', 'Predict a compound'] },
  trees: { intermediate: ['Choosing a question', 'Workbench', 'One compound', 'Results'], advanced: ['Forest', 'Boosting', 'Which features matter', 'Filtering the data', 'Is it useful'], pro: ['Seeds and ensembles'] },
  nn: { intermediate: ['What a weight is', 'One neuron', 'Workbench', 'One compound'], advanced: ['Layers and neurons', 'Learning rate', 'Dropout and early stopping', 'Filtering the data', 'What it learned', 'Is it useful'], pro: ['Inside a neuron', 'Chemprop', 'Seeds and ensembles'] },
  gpt: { intermediate: ['Tokens', 'Workbench', 'Next token', 'Generate and check'], advanced: ['Embeddings', 'The stack', 'Filtering the data', 'Is it useful'], pro: ['Attention'] },
  pre: { intermediate: ['Why pre-train', 'Learning without labels'], advanced: ['Which groups are present', 'The model and its size', 'Fixed features', 'Using it in practice'], pro: ['Fine-tune', 'Fine-tune Chemprop'] },
  compare: { intermediate: ['Leaderboard', 'Accuracy against time', 'Discussion'] },
  ai: { beginner: ['What AI means', 'Predict or generate'], intermediate: ['Outside the training range', 'Lab noise', 'How generative AI goes wrong', 'Can it or cannot it'], advanced: ['Using it in a project'] },
  treealgo: { advanced: ['Split search', 'Growing a tree', 'Bagging', 'Fitting a helix', 'Trees and data'], pro: ['Boosting math', 'Tuning', 'Library designs'] },
  netalgo: { advanced: ['Gradient descent', 'Backpropagation', 'Counting parameters', 'Scaling networks'], pro: ['Initialisation', 'Weight decay', 'Attention', 'Next-token loss'] }
};
/* the order of the Beginner steps in each lesson */
var STEP_ORDER = {
  start: ['Molecular representations', 'The data is a table', 'What learning means', 'The test set', 'Practice and real exam', 'Scoring a model'],
  trees: ['Twenty questions', 'Choosing the best question', 'A crowd of trees', 'Fixing mistakes round by round'],
  nn: ['A dial and a sum', 'Neurons side by side', 'Why a network can bend', 'A network that is too big'],
  gpt: ['Cutting into pieces', 'A number for every piece', 'Autocomplete for molecules', 'Why memory matters'],
  pre: ['Labelled or unlabelled', 'Fill in the blank', 'A head start', 'Three ways to use it'],
  compare: ['What the scores mean', 'Why no score is perfect', 'Fast or accurate'],
  ai: ['What AI means', 'Predict or generate', 'What a model has not seen', 'Checking an answer'],
  treealgo: ['Scoring a cut', 'Fewer places to cut', 'Random draws'],
  netalgo: ['Rolling downhill', 'Which way to turn the dial', 'Counting the dials']
};
function applyLevels(){
  Object.keys(STEP_LEVELS).forEach(function(id){ var L = LESSONS[id]; if (!L) return; var map = {}; Object.keys(STEP_LEVELS[id]).forEach(function(lv){ STEP_LEVELS[id][lv].forEach(function(n){ map[n] = lv; }); }); L.steps.forEach(function(s){ if (!s.level || (STEP_LEVELS[id] && map[s.short])) s.level = map[s.short] || s.level || 'intermediate'; var o = (STEP_ORDER[id] || []).indexOf(s.short); if (o >= 0) s.ord = o; }); });
}
