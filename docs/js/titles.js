"use strict";
/* Step titles, set in one place so that wording and capitalisation stay consistent.
   Sentence case, noun phrases, six words or fewer, no colons, no questions.
   The keys are the internal short names of the steps. */
var STEP_TITLES = {
  start: {
    'Molecular representations': 'Molecular representations',
    'The data is a table': 'Dataset structure',
    'What learning means': 'Fitting a model',
    'The test set': 'Training and test sets',
    'Practice and real exam': 'Training and test performance',
    'Scoring a model': 'Evaluation metrics',
    'Structure to prediction': 'From structure to prediction',
    'Where features come from': 'Descriptor contributions',
    'The target': 'Target distribution',
    'Underfitting and overfitting': 'Model complexity',
    'Handling censoring': 'Censored measurements',
    'logD feeds HLM': 'Stacking a logD model',
    'Predict a compound': 'Predicting a new compound'
  },
  trees: {
    'Twenty questions': 'Decision tree structure',
    'Choosing the best question': 'Split selection',
    'A crowd of trees': 'Random forests',
    'Fixing mistakes round by round': 'Gradient boosting',
    'Choosing a question': 'A single split',
    'Workbench': 'Workbench',
    'One compound': 'Prediction path',
    'Results': 'Scores and fit time',
    'Forest': 'Forest size and depth',
    'Boosting': 'Boosting rounds',
    'Which features matter': 'Feature importance',
    'Filtering the data': 'Training set filters',
    'Is it useful': 'Single-feature predictions',
    'Seeds and ensembles': 'Seeds and ensembles'
  },
  nn: {
    'A dial and a sum': 'Linear neuron',
    'Neurons side by side': 'Hidden layer',
    'Why a network can bend': 'Nonlinearity',
    'A network that is too big': 'Network capacity',
    'What a weight is': 'Weight and bias',
    'One neuron': 'Neuron activation',
    'Workbench': 'Workbench',
    'One compound': 'Forward pass',
    'Layers and neurons': 'Network size',
    'Learning rate': 'Learning rate',
    'Dropout and early stopping': 'Regularisation',
    'Filtering the data': 'Training set filters',
    'What it learned': 'Partial dependence',
    'Is it useful': 'Network and tree comparison',
    'Inside a neuron': 'Neuron weights',
    'Chemprop': 'Message passing',
    'Seeds and ensembles': 'Seeds and ensembles'
  },
  gpt: {
    'Cutting into pieces': 'Tokenisation',
    'A number for every piece': 'Token embeddings',
    'Autocomplete for molecules': 'Next-token prediction',
    'Why memory matters': 'Context length',
    'Tokens': 'Vocabulary and token IDs',
    'Workbench': 'Workbench',
    'Next token': 'Sampling and temperature',
    'Generate and check': 'Generation and validity',
    'Embeddings': 'Learned embeddings',
    'The stack': 'Transformer architecture',
    'Filtering the data': 'Training corpus filters',
    'Is it useful': 'Design test',
    'Attention': 'Attention weights'
  },
  pre: {
    'Labelled or unlabelled': 'Labelled and unlabelled data',
    'Fill in the blank': 'Masked prediction',
    'A head start': 'Effect of pre-training',
    'Three ways to use it': 'Transfer strategies',
    'Why pre-train': 'Learning curve from scratch',
    'Learning without labels': 'Self-supervised task',
    'Which groups are present': 'Functional group prediction',
    'The model and its size': 'GROVER',
    'Fixed features': 'Frozen embeddings',
    'Using it in practice': 'Practical use',
    'Fine-tune': 'Fine-tuning the GPT',
    'Fine-tune Chemprop': 'Fine-tuning Chemprop'
  },
  compare: {
    'What the scores mean': 'Metric behaviour',
    'Why no score is perfect': 'Assay repeatability ceiling',
    'Fast or accurate': 'Accuracy and training time',
    'Leaderboard': 'Leaderboard',
    'Accuracy against time': 'Accuracy against training time',
    'Discussion': 'Reading a result'
  },
  ai: {
    'What AI means': 'AI, machine learning and generative AI',
    'Predict or generate': 'Prediction and generation',
    'What a model has not seen': 'Applicability domain',
    'Checking an answer': 'Assessing a prediction',
    'Outside the training range': 'Extrapolation',
    'Lab noise': 'Assay noise',
    'How generative AI goes wrong': 'Generative model failures',
    'Can it or cannot it': 'Capabilities and limits',
    'Using it in a project': 'Checks before use'
  },
  treealgo: {
    'Scoring a cut': 'Split criterion',
    'Fewer places to cut': 'Histogram binning',
    'Random draws': 'Bootstrap sampling',
    'Split search': 'Exact and binned search',
    'Growing a tree': 'Depth and leaf size',
    'Bagging': 'Bagging and variance',
    'Fitting a helix': 'Fitting a 3D shape',
    'Trees and data': 'Data requirements',
    'Boosting math': 'Boosting losses',
    'Tuning': 'Hyperparameter tuning',
    'Library designs': 'Boosting libraries'
  },
  netalgo: {
    'Rolling downhill': 'Gradient descent in one dimension',
    'Which way to turn the dial': 'Finite-difference gradient',
    'Counting the dials': 'Parameter count',
    'Gradient descent': 'Optimisers',
    'Backpropagation': 'Backpropagation',
    'Counting parameters': 'Parameter count of a GPT',
    'Scaling networks': 'Network size and data',
    'Initialisation': 'Weight initialisation',
    'Weight decay': 'Weight decay',
    'Attention': 'Scaled dot-product attention',
    'Next-token loss': 'Cross-entropy and perplexity'
  }
};
function applyTitles(){
  Object.keys(STEP_TITLES).forEach(function(id){ var L = LESSONS[id]; if (!L) return; L.steps.forEach(function(s){ var t = STEP_TITLES[id][s.short]; if (t) s.title = t; }); });
}
