# Addendum 1 to the analysis plan

Written 2026-11-06 after the original plan (`docs/analysis_plan.md`, SHA-256 `602693ec…`) and **before any confirmation dataset was preprocessed or analysed**. The original confirmatory tests C1–C3 are unchanged.

## Reason

A literature check while the datasets were downloading found Miller et al. (Nature Biotechnology, 2026, doi:10.1038/s41587-026-03307-w). They argue that MSE and control-referenced correlation are poorly calibrated for perturbation prediction, and that conclusions about baselines depend on this. C1 uses MSE, so its calibration is now measured explicitly.

## E4. Metric calibration (exploratory)

For each included dataset and each single-gene condition:

- **Positive control (technical duplicate):** predict the response estimated from one random half of the condition's cells and half of the controls (`Y_half_b`), using the response from the other halves (`Y_half_a`).
- **Negative control:** predict `Y_half_b` with the mean `Y_fit` response of all other conditions.

For each metric (`mse`, `skill_vs_mean`, `pearson`, `pearson_specific`, `spearman_specific`, `pearson_specific_top`, `topk_recall`), record the fraction of conditions where the positive control scores better than the negative control. Correlation metrics that are undefined for the negative control count as 0. A metric is called *calibrated on a dataset* if the positive control wins for at least 80% of conditions.

The pre-specified learned models are then also summarised on every metric that is calibrated on that dataset, against the mean baseline, as an exploratory complement to C1. `spearman_specific` (the Spearman correlation of deviations from the training mean) is added to the metric set for this purpose.
