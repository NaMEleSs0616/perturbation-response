# Analysis plan for the multi-dataset study

Written 2026-11-06, before any dataset other than Norman 2019 was preprocessed or analysed. At the time of writing, the new datasets had only been downloaded and, for `tian2021_crispra`, inspected for column names and label counts. The SHA-256 of this file is recorded in every study output (`plan_sha256`) so that later edits are detectable. Any deviation from this plan is reported in the paper as a deviation.

## Design

**Discovery dataset:** Norman et al. 2019 (K562, CRISPRa). All model and baseline choices below were informed by results on it, so it cannot confirm them.

**Confirmation datasets:** every one of the following that passes the inclusion rules. None had been analysed when this plan was written.

| Id | Study | Modality | Cell context |
|---|---|---|---|
| `replogle2022_k562_essential` | Replogle 2022 | CRISPRi | K562 |
| `replogle2022_rpe1` | Replogle 2022 | CRISPRi | hTERT-RPE1 |
| `adamson2016_upr` | Adamson 2016 | CRISPRi | K562 |
| `dixit2016_k562_7d` | Dixit 2016 | CRISPR knockout | K562 |
| `tian2021_crispra` | Tian 2021 | CRISPRa | iPSC-derived neurons |
| `tian2021_crispri` | Tian 2021 | CRISPRi | iPSC-derived neurons |

## Inclusion rules (fixed before analysis)

A dataset is included if all of these hold:

1. It passes `datasets validate`, including the MD5 check.
2. At least 100 cells carry the label `control` in the `perturbation` column. The control group is never inferred from guide names.
3. At least 30 single-gene conditions have ≥ 50 cells. If fewer than 30 qualify at 50, the threshold drops to 25 cells; if still fewer than 30, the dataset is excluded.
4. At least 80% of single-gene target labels resolve to a gene row, directly or through HGNC and Ensembl.

Conditions with more than one target are excluded from the confirmation study. Exclusions are reported with reasons.

## Preprocessing (identical across datasets)

- As in `configs/benchmark_norman.yaml`: 2,000 HVGs selected from control cells, plus target genes; CP10k normalisation and log1p; disjoint control halves for training and evaluation targets.
- Batch column: the first present of `gemgroup`, `batch`, `library`; otherwise none.
- Cell filters: `good_coverage == True` and `number_of_cells == 1`, each applied only if the column exists.
- Co-expression embedding as before (32 dimensions, from at most 8,000 control cells).

## Splits

The `unseen_gene` task: 5 folds over target genes, seed 0, 15% of the remaining genes held out for calibration. Robustness: fold seeds 1, 2 and 3, reported but not used for decisions.

## Models (fixed)

Baseline: `mean`. Pre-specified learned models, all with co-expression features: `ridge`, `gene_loading_ridge`, `mlp_ensemble`. Hyperparameter grids as in `configs/benchmark_norman.yaml`. `zero`, `knn` and `bootstrap_ridge` are also run and reported but are not part of the confirmatory tests.

## Confirmatory tests

**C1. No learned model beats the mean baseline on unseen single genes.**
For each dataset × pre-specified model: paired per-perturbation MSE difference (model − mean), 95% percentile bootstrap interval (2,000 resamples over perturbations), and a one-sided Wilcoxon signed-rank test of "model lower". The Holm correction is applied across all dataset × model tests at α = 0.05. A model **beats the mean** on a dataset if its Holm-adjusted p < 0.05 and the upper interval bound is below 0. C1 is refuted for any dataset where some model beats the mean.

**C2. Co-expression features carry no usable information about an unseen gene.**
Paired per-perturbation skill difference (co-expression − shuffled co-expression) for `ridge` and `mlp_ensemble`. One shuffle per dataset, seed 0. C2 is refuted for a dataset if the 95% interval lies above 0 for either model.

**C3. Frozen Gene2vec does not improve skill over co-expression for unseen single genes.**
Paired per-perturbation skill difference (Gene2vec − co-expression) for `ridge` and `mlp_ensemble`, with the Holm correction across dataset × model at α = 0.05 (two-sided Wilcoxon). C3 is refuted if any test is significant with Gene2vec better.

## Descriptive analyses (no decision rules)

- **D1.** Reproducibility per dataset: batch-disjoint and split-half correlation of the perturbation-specific response, and the Spearman-Brown limit. If there is only one batch, the batch-disjoint value is not computed.
- **D2.** Ranking unseen singles by response magnitude: precision among the top 20% and Spearman correlation.
- **D3.** Interval coverage over all genes and the top 20 genes, and the Spearman correlation between ensemble disagreement and RMSE.
- **D4.** Fold-seed robustness of C1.

## Exploratory analyses (hypothesis-generating)

- **E1. Training-set size.** In the two Replogle datasets: subsample training conditions to 25, 50, 100, 200, 400, 800 and all, within each fold (3 random subsamples per size, seed 0). Record skill vs mean for `ridge` and `gene_loading_ridge`, with test folds unchanged. Question: does the gap to the mean close as training perturbations increase?
- **E2. Cross-context transfer.** For targets measured in both Replogle K562 and RPE1, on genes measured in both: predict RPE1's response as K562's measured response, and vice versa. Compare against the target dataset's mean of other perturbations, using skill and paired MSE. Question: does a measurement in another cell line carry information that unseen-gene models lack?
- **E3. Norman pairs.** The saturation and non-additivity analyses from the discovery benchmark are reported as exploratory; no confirmation dataset contains gene pairs.

## Reporting

All datasets, models and tests are reported, including those that fail inclusion. Results are not re-run with changed settings to obtain a different outcome; any re-run caused by a software error is disclosed with the reason.
