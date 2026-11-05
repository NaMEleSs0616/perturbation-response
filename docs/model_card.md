# Model card

## Purpose

Research tool that predicts pseudobulk change in log-normalised mRNA expression after CRISPRa in K562 cells, and helps rank follow-up experiments.

**Not for** clinical, diagnostic or therapeutic decisions. It does not predict protein abundance, phenotype, efficacy or toxicity, and it does not simulate a cell.

## Served models (`bench-20261101-142417-9f7a7e`)

| Input | Model | Baseline compared | MSE difference, model − baseline (95% interval) | Verdict | Behaviour |
|---|---|---|---|---|---|
| Unmeasured single gene | `mlp_ensemble` | mean | +0.00036 (+0.00012, +0.00062) | Worse than baseline | **Abstains by default** |
| Unmeasured pair, both singles measured | `mlp_on_additive` | scaled additive | -0.00021 (-0.00043, +0.00004) | Not distinguishable | Predicts, with a warning |
| Unmeasured pair, ≥1 single unmeasured | `mlp_ensemble` | mean | -0.00082 (-0.00110, -0.00057) | Better than the mean, small skill (0.07) | Predicts, with a warning |
| Measured condition | As above | n/a | n/a | Not out of sample | Returned with the observed response and a warning |

Models are refitted on all measured conditions for serving; the benchmark results describe held-out versions of the same procedure, not this exact fit. Inputs are the sum of target gene representations (co-expression across control cells), target baseline expression, and a pair indicator. Output is low-rank (20 PCA components of training residuals), added to a mean or additive base. Five bootstrap members, Huber loss, AdamW, early stopping.

## Validated domain

Held-out perturbations in one K562 CRISPRa screen. Not validated: other cell lines or modalities, independent datasets, prospective experiments, biological replicates, or more than two targets. Other cell lines, other perturbation modalities and more than two targets cause abstention. A gene is only supported if it is in the processed gene set.

## Performance

See `benchmarking.md`. In brief: skill vs mean 0.63 for pairs with measured singles (the one-parameter scaled additive baseline reaches 0.60), -0.18 for unseen singles.

## Uncertainty

Each prediction reports ensemble disagreement and 90% per-gene intervals calibrated on out-of-fold residuals.

- Measured coverage: 0.86 over all genes for measured-singles pairs, but **0.74 on top responders**, and 0.41 for pairs with an unmeasured single.
- Disagreement vs error: Spearman 0.21 for pairs with measured singles, -0.00 for pairs with an unmeasured single. Low disagreement does not imply a correct prediction.

## Abstention and warnings

Abstains for: an unsupported cell context or perturbation type, missing context or type, more than two targets, a gene outside the processed set, and input types the benchmark shows to be worse than baseline. Warns for: a measured condition, a target detected in under 5% of control cells, a query far from measured perturbations, and benchmark caveats for that input type.

## Limitations

See `scientific_assumptions.md`. One screen, no biological replicates, non-random choice of targets and pairs, batch variation only partly represented, and all results on folds that informed some model and baseline additions.

## Experiment ranking

Ranking methods are heuristics. The only demonstrated benefit is about 2.6-fold enrichment for strong pairs, achieved equally well by the additive prediction. No acquisition strategy has a demonstrated practical advantage, and none is called optimal.

## License and citation

Data: cite Norman et al. 2019 and Peidli et al. 2024 (scPerturb); the harmonised file is CC-BY-4.0 on Zenodo. Gene2vec is MIT-licensed. Code license: not yet chosen.
