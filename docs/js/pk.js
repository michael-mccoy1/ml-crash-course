"use strict";
/* Lesson 10: PK, PBPK and MIDD interfaces. The steps are in pk_a.js (Beginner), pk_b.js (Intermediate) and pk_c.js (Advanced and Pro). */
var L10 = new Lesson('pk', 'PK and PBPK');
L10.noOpts = true;
L10.glossary([
  ['Clearance (CL)', 'The volume of plasma from which drug is removed per unit time, in L/h. For an intravenous dose it equals the dose divided by the AUC.'],
  ['Volume of distribution (V)', 'The proportionality factor between the amount of drug in the body and the plasma concentration.'],
  ['Half-life', 'The time for the plasma concentration to halve in the terminal phase. For a one-compartment model it is ln 2 times V divided by CL.'],
  ['AUC', 'The area under the plasma concentration-time curve.'],
  ['Cmax and tmax', 'The peak concentration and the time at which it occurs.'],
  ['Bioavailability (F)', 'The fraction of an oral dose that reaches the systemic circulation unchanged.'],
  ['IVIVE', 'In vitro to in vivo extrapolation: scaling an in vitro rate, such as microsomal intrinsic clearance, to an in vivo clearance.'],
  ['Well-stirred model', 'A liver model in which the unbound drug concentration in the liver equals that in the blood leaving it. Hepatic clearance is Q times fu times CLint divided by (Q + fu times CLint).'],
  ['Extraction ratio', 'The fraction of the drug entering an organ that is removed in one pass.'],
  ['PBPK', 'Physiologically based pharmacokinetic model: compartments represent organs, with physiological volumes and blood flows.'],
  ['Perfusion-limited distribution', 'Tissue uptake limited only by blood flow, so that tissue and venous blood are in instantaneous equilibrium.'],
  ['Partition coefficient (Kp)', 'The ratio of the tissue concentration to the plasma concentration at equilibrium.'],
  ['Non-compartmental analysis (NCA)', 'Estimation of PK parameters from the observed profile without fitting a compartmental model.'],
  ['Between-subject variability', 'Variation of a parameter across individuals, usually modelled as lognormal.'],
  ['Covariate', 'A subject characteristic, such as body weight, that explains part of the between-subject variability.'],
  ['MIDD', 'Model-informed drug development: the use of quantitative models to support drug development and regulatory decisions.'],
  ['Context of use', 'The specific question a model is used to answer, and the role of its output in the decision.'],
  ['Quantitative systems pharmacology (QSP)', 'Mechanistic models of biological pathways combined with PK to describe drug effect.']
]);
